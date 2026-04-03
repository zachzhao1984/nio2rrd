#!/usr/bin/env python3
"""
NIO Data to Rerun RRD Converter

Converts NIO SIRIUS (NT2) autonomous driving data packets to Rerun's .rrd format.

Supported data types:
- Camera: H.265 video → Rerun Image
- Lidar PointCloud: PCD → Rerun Points3D
- Lidar Cluster Objects: Protobuf → Rerun Boxes
- Perception Objects: Protobuf → Rerun Boxes + Velocities

Usage:
    python nio_to_rerun.py <path_to_dat.zip> [output.rrd]

Dependencies:
    pip install rerun-sdk opencv-python protobuf numpy
"""

import argparse
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rerun as rr

# Add protos to path for imports
PROTO_DIR = Path(__file__).parent / "protos"
sys.path.insert(0, str(PROTO_DIR))

from common.perception.lidar_cluster_output_pb2 import LidarClusterObjects
from common.perception.perception_objects_pb2 import (
    DynamicObj,
    DynamicObjects,
    ObjectsDetection,
    StaticObjects,
)


@dataclass
class CameraFrame:
    """Single camera frame with metadata."""

    ptp_timestamp: int
    utc_timestamp: int
    frame_type: str


PCD_SCALAR_FIELDS = ("intensity", "reflectance", "i")
RERUN_TIMELINE = "time"
PCD_BINARY_DTYPES = {
    ("F", 4): np.dtype("<f4"),
    ("F", 8): np.dtype("<f8"),
    ("I", 1): np.dtype("<i1"),
    ("I", 2): np.dtype("<i2"),
    ("I", 4): np.dtype("<i4"),
    ("I", 8): np.dtype("<i8"),
    ("U", 1): np.dtype("<u1"),
    ("U", 2): np.dtype("<u2"),
    ("U", 4): np.dtype("<u4"),
    ("U", 8): np.dtype("<u8"),
}


def _extract_lidar_pcd_timestamp_ns(path: str) -> int:
    """Extract timestamp from a lidar PCD filename and normalize to ns."""
    try:
        timestamp_value = int(Path(path).stem)
    except ValueError:
        return 0

    if timestamp_value >= 10**18:
        return timestamp_value
    if timestamp_value >= 10**15:
        return timestamp_value * 1_000
    if timestamp_value >= 10**12:
        return timestamp_value * 1_000_000
    return timestamp_value * 1_000_000_000


def _parse_pcd_pointcloud(data: bytes) -> Optional[np.ndarray]:
    """Parse ASCII or binary PCD payload into Nx4 float32 [x, y, z, scalar]."""
    stream = io.BytesIO(data)
    header: dict[str, list[str]] = {}

    while True:
        line = stream.readline()
        if not line:
            return None

        text = line.decode("utf-8", "replace").strip()
        if not text or text.startswith("#"):
            continue

        key, *values = text.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            break

    fields = header.get("FIELDS", [])
    if not {"x", "y", "z"}.issubset(fields):
        return None

    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
    if len(counts) != len(fields):
        return None

    column_offsets: dict[str, int] = {}
    total_columns = 0
    for field_name, field_count in zip(fields, counts):
        column_offsets[field_name] = total_columns
        total_columns += field_count

    scalar_field = next(
        (field_name for field_name in PCD_SCALAR_FIELDS if field_name in column_offsets),
        None,
    )
    data_mode = header["DATA"][0].lower() if header.get("DATA") else ""
    point_count = int(header.get("POINTS", ["0"])[0])
    payload = stream.read()

    if data_mode == "ascii":
        if point_count == 0 or not payload.strip():
            points_data = np.empty((0, total_columns), dtype=np.float32)
        else:
            try:
                points_data = np.loadtxt(io.BytesIO(payload), dtype=np.float32)
            except ValueError:
                return None
            points_data = np.atleast_2d(points_data)

        if points_data.ndim != 2 or points_data.shape[1] < total_columns:
            return None

        xyz = np.column_stack(
            [
                points_data[:, column_offsets["x"]],
                points_data[:, column_offsets["y"]],
                points_data[:, column_offsets["z"]],
            ]
        ).astype(np.float32, copy=False)
        if scalar_field:
            scalar = points_data[:, column_offsets[scalar_field]].reshape(-1, 1)
        else:
            scalar = np.ones((xyz.shape[0], 1), dtype=np.float32)
        return np.concatenate([xyz, scalar.astype(np.float32, copy=False)], axis=1)

    if data_mode != "binary":
        return None

    sizes = [int(v) for v in header.get("SIZE", [])]
    types = [v.upper() for v in header.get("TYPE", [])]
    if len(sizes) != len(fields) or len(types) != len(fields):
        return None

    dtype_fields = []
    for field_name, field_size, field_type, field_count in zip(
        fields, sizes, types, counts
    ):
        dtype = PCD_BINARY_DTYPES.get((field_type, field_size))
        if dtype is None:
            return None
        if field_count == 1:
            dtype_fields.append((field_name, dtype))
        else:
            dtype_fields.append((field_name, dtype, (field_count,)))

    point_dtype = np.dtype(dtype_fields)
    if point_count <= 0:
        point_count = len(payload) // point_dtype.itemsize
    if len(payload) < point_count * point_dtype.itemsize:
        return None

    structured_points = np.frombuffer(
        payload[: point_count * point_dtype.itemsize],
        dtype=point_dtype,
        count=point_count,
    )
    xyz = np.column_stack(
        [
            np.asarray(structured_points["x"], dtype=np.float32).reshape(point_count, -1)[:, 0],
            np.asarray(structured_points["y"], dtype=np.float32).reshape(point_count, -1)[:, 0],
            np.asarray(structured_points["z"], dtype=np.float32).reshape(point_count, -1)[:, 0],
        ]
    )
    if scalar_field:
        scalar = np.asarray(structured_points[scalar_field], dtype=np.float32).reshape(
            point_count, -1
        )[:, :1]
    else:
        scalar = np.ones((point_count, 1), dtype=np.float32)
    return np.concatenate([xyz, scalar], axis=1)


def _extract_filename_timestamp_ns(path: str) -> int:
    """Extract the nanosecond timestamp suffix from a .pb.dat filename."""
    name = Path(path).name
    suffix = ".pb.dat"
    if not name.endswith(suffix):
        return 0

    stem = name[: -len(suffix)]
    if "_" not in stem:
        return 0

    _, timestamp_part = stem.rsplit("_", 1)
    try:
        return int(timestamp_part)
    except ValueError:
        return 0


def _looks_like_framed_pb_dat(data: bytes) -> bool:
    """Check whether a .pb.dat file uses the observed 40-byte record header."""
    offset = 0
    record_count = 0

    while offset + 40 <= len(data):
        _, _, _, _, payload_size = struct.unpack_from("<QQQQQ", data, offset)
        next_offset = offset + 40 + payload_size
        if payload_size <= 0 or next_offset > len(data):
            return False

        record_count += 1
        offset = next_offset
        if offset == len(data):
            return record_count > 0

    return False


def _iter_framed_pb_messages(data: bytes) -> list[tuple[int, bytes]]:
    """Iterate messages from a framed .pb.dat file."""
    if not _looks_like_framed_pb_dat(data):
        return []

    messages = []
    offset = 0
    while offset + 40 <= len(data):
        ptp_begin_ns, _, _, _, payload_size = struct.unpack_from("<QQQQQ", data, offset)
        payload_offset = offset + 40
        payload_end = payload_offset + payload_size
        messages.append((int(ptp_begin_ns), data[payload_offset:payload_end]))
        offset = payload_end

    return messages


def _message_timestamp_ns(message: object, fallback: int = 0) -> int:
    """Prefer protobuf-native timestamps over filename-derived values."""
    for field_name in ("publish_ptp_ts", "timestamp", "publish_ts"):
        if not hasattr(message, field_name):
            continue
        try:
            if hasattr(message, "HasField") and message.HasField(field_name):
                timestamp_ns = int(getattr(message, field_name))
                if timestamp_ns > 0:
                    return timestamp_ns
        except ValueError:
            continue

    return fallback


def _camera_frame_timestamp_ns(frame: CameraFrame, fallback: int = 0) -> int:
    """Prefer the camera timestamp column aligned with lidar/perception data."""
    if frame.utc_timestamp > 0:
        return frame.utc_timestamp
    if frame.ptp_timestamp > 0:
        return frame.ptp_timestamp
    return fallback


def _lidar_cluster_timestamp_ns(
    clusters: LidarClusterObjects, fallback: int = 0
) -> int:
    """Select the most reliable lidar cluster timestamp available."""
    message_timestamp = _message_timestamp_ns(clusters, fallback=0)
    if message_timestamp > 0:
        return message_timestamp

    cluster_timestamps = [
        int(obj.lidar_cluster_mean_timestamp)
        for obj in clusters.lidar_cluster_object_list
        if obj.HasField("lidar_cluster_mean_timestamp")
        and obj.lidar_cluster_mean_timestamp > 0
    ]
    if cluster_timestamps:
        return min(cluster_timestamps)

    return fallback


def _camera_intrinsics_from_calibration(
    calib_data: Optional[dict], width: int, height: int
) -> Optional[tuple[float, float, float, float]]:
    """Extract camera intrinsics from calibration data when available."""
    if not calib_data or "calibration_info" not in calib_data:
        return None

    calib = calib_data["calibration_info"]
    camera_matrix = calib.get("camera_matrix")
    if not isinstance(camera_matrix, dict):
        return None

    intrinsic = camera_matrix.get("intrinsic")
    if not isinstance(intrinsic, dict):
        return None

    try:
        fx = float(intrinsic.get("fx", width * 1.2))
        fy = float(intrinsic.get("fy", height * 1.2))
        cx = float(intrinsic.get("cx", width / 2))
        cy = float(intrinsic.get("cy", height / 2))
    except (TypeError, ValueError):
        return None

    return fx, fy, cx, cy


def _log_camera_frame(
    image_path: str,
    camera_path: str,
    timestamp_ns: int,
    frame: np.ndarray,
    calib_data: Optional[dict] = None,
) -> None:
    """Log one camera frame and optional pinhole calibration to Rerun."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    rr.set_time(RERUN_TIMELINE, timestamp=timestamp_ns / 1e9)
    rr.log(image_path, rr.Image(frame_rgb))

    intrinsics = _camera_intrinsics_from_calibration(calib_data, width, height)
    if intrinsics is None:
        return

    fx, fy, cx, cy = intrinsics
    rr.log(
        camera_path,
        rr.Pinhole(
            resolution=(width, height),
            focal_length=(fx, fy),
            principal_point=(cx, cy),
        ),
    )
    rr.log(camera_path, rr.Image(frame_rgb))


def _clear_stale_entity_paths(
    previous_paths: set[str], current_paths: set[str], timestamp_ns: int
) -> None:
    """Clear entity paths that disappeared in the current frame."""
    stale_paths = previous_paths - current_paths
    if not stale_paths:
        return

    rr.set_time(RERUN_TIMELINE, timestamp=timestamp_ns / 1e9)
    for stale_path in sorted(stale_paths):
        rr.log(stale_path, rr.Clear(recursive=False))


class NIODataExtractor:
    """Extract data from NIO SIRIUS data packets."""

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.zip_file = zipfile.ZipFile(zip_path, "r")
        self._load_metadata()
        self._init_dds_mapping()

    def _load_metadata(self):
        """Load metadata JSON files."""
        namelist = [n for n in self.zip_file.namelist() if not n.startswith("__MACOSX")]
        uuid_dirs = set()
        for name in namelist:
            if "/" in name:
                uuid_dirs.add(name.split("/")[0])

        if not uuid_dirs:
            raise ValueError("No data directories found in zip")

        self.uuid = list(uuid_dirs)[0]

        # Load meta.json
        meta_path = f"{self.uuid}/meta.json"
        self.meta = json.loads(self.zip_file.read(meta_path).decode("utf-8"))

        # Load da_data_meta.json
        da_meta_path = f"{self.uuid}/da_data_meta.json"
        self.da_meta = json.loads(self.zip_file.read(da_meta_path).decode("utf-8"))

        # Load calibration files
        self.calibration = {}
        cal_dir = f"{self.uuid}/calibration/"
        for name in self.zip_file.namelist():
            if name.startswith(cal_dir) and name.endswith(".json"):
                # Extract clean name (remove .json, ignore ._ files)
                if "/._" in name or name.endswith("._"):
                    continue
                cal_name = Path(name).stem
                if cal_name.startswith("._"):
                    continue
                try:
                    cal_data = json.loads(self.zip_file.read(name).decode("utf-8"))
                    self.calibration[cal_name] = cal_data
                except Exception as e:
                    pass

        # Map topic names to calibration file names
        self.camera_calib_map = {
            "/camera/front/main": "front_wide",
            "/camera/front/narrow": "front_wide",  # Use wide as fallback
            "/camera/rear": "rear_narrow",
            "/camera/surrouding/front": "svc_front",
            "/camera/surrouding/left": "svc_front",
            "/camera/surrouding/rear": "svc_front",
            "/camera/surrouding/right": "svc_front",
            "/camera/side/front/left": "front_left",
            "/camera/side/front/right": "front_right",
            "/camera/side/rear/left": "rear_left",
            "/camera/side/rear/right": "rear_right",
        }
        self.topic_metadata = {}
        for source in (self.meta.get("topics", {}), self.da_meta.get("topics", {})):
            for topic_name, topic_info in source.items():
                self.topic_metadata[topic_name] = topic_info

    def _init_dds_mapping(self):
        """Initialize DDS topic to proto message mapping."""
        self.dds_mapping = {
            "/common/perception/perception_objects": "PerceptionObjects",
            "/perception/objects": "PerceptionObjects",
            "/perception/perception_objects": "PerceptionObjects",
            "/perception/fusion_objects": "PerceptionObjects",
            "/perception/predicted_objects": "PerceptionObjects",
            "/perception/lidar_cluster": "LidarClusterObjects",
        }

        self.topic_aliases = {
            "/perception/objects": [
                "/common/perception/perception_objects",
                "/perception/perception_objects",
            ],
            "/perception/perception_objects": ["/common/perception/perception_objects"],
            "/perception/fusion_objects": ["/common/perception/perception_objects"],
            "/perception/predicted_objects": ["/common/perception/perception_objects"],
        }

    def _resolve_topic_names(self, topic_name: str) -> list[str]:
        """Return candidate topic names in lookup order."""
        candidates = [topic_name]
        for alias in self.topic_aliases.get(topic_name, []):
            if alias not in candidates:
                candidates.append(alias)
        return candidates

    def _resolve_topic_files(self, topic_name: str) -> list[str]:
        """Resolve topic metadata to exact zip members when available."""
        resolved_files = []
        for candidate_topic in self._resolve_topic_names(topic_name):
            topic_info = self.topic_metadata.get(candidate_topic)
            if not topic_info:
                continue

            file_name = topic_info.get("file_name")
            if not file_name:
                continue

            full_path = f"{self.uuid}/{file_name}"
            if full_path in self.zip_file.namelist() and full_path not in resolved_files:
                resolved_files.append(full_path)

        return resolved_files

    def read_camera_video(self, topic_name: str) -> tuple[bytes, list[CameraFrame]]:
        """Read H.264 video data and timestamps based on topic name."""
        # Topic name to filename pattern mapping
        topic_to_pattern = {
            "/camera/front/main": "Front120",
            "/camera/front/narrow": "Front30",
            "/camera/rear": "Rear",
            "/camera/surrouding/front": "Park_Front",
            "/camera/surrouding/left": "Park_Left",
            "/camera/surrouding/rear": "Park_Rear",
            "/camera/surrouding/right": "Park_Right",
            "/camera/side/front/left": "SideView_FL",
            "/camera/side/front/right": "SideView_FR",
            "/camera/side/rear/left": "SideView_RL",
            "/camera/side/rear/right": "SideView_RR",
        }

        pattern = topic_to_pattern.get(topic_name)
        if not pattern:
            return b"", []

        # Find the camera file matching the pattern
        camera_dir = f"{self.uuid}/data/camera/"
        video_file = None

        for name in self.zip_file.namelist():
            if name.startswith(camera_dir) and name.endswith(".h264"):
                # Check if this file matches our pattern
                if f"_{pattern}_" in name or name.endswith(f"_{pattern}.h264"):
                    video_file = name
                    break

        if not video_file:
            return b"", []

        try:
            video_data = self.zip_file.read(video_file)
        except KeyError:
            return b"", []

        # Read timestamp file
        ts_file = video_file.rsplit(".", 1)[0] + ".txt"
        timestamps = []
        try:
            ts_data = self.zip_file.read(ts_file).decode("utf-8")
            for line in ts_data.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 5:
                    timestamps.append(
                        CameraFrame(
                            ptp_timestamp=int(parts[0]),
                            utc_timestamp=int(parts[1]),
                            frame_type=parts[3],
                        )
                    )
        except KeyError:
            pass

        return video_data, timestamps

    def read_dds_message(self, topic_name: str) -> list[tuple[int, bytes]]:
        """Read DDS/Protobuf messages for a topic from .pb.dat files."""
        dds_files: list[str] = self._resolve_topic_files(topic_name)

        if not dds_files:
            topic_variants = {
                candidate.lstrip("/").replace("/", "-")
                for candidate in self._resolve_topic_names(topic_name)
            }
            dds_prefix = f"{self.uuid}/data/dds/{self.uuid}_"

            for name in self.zip_file.namelist():
                if not name.startswith(dds_prefix) or not name.endswith(".pb.dat"):
                    continue

                relative_name = name[len(dds_prefix) :]
                if "_" not in relative_name:
                    continue

                file_topic_part = relative_name[: relative_name.rfind("_")]
                normalized_topic = file_topic_part.lstrip("-")
                if normalized_topic not in topic_variants:
                    continue

                if name not in dds_files:
                    dds_files.append(name)

        messages: list[tuple[int, bytes]] = []
        for dds_file in dds_files:
            try:
                data = self.zip_file.read(dds_file)
            except KeyError:
                continue

            framed_messages = _iter_framed_pb_messages(data)
            if framed_messages:
                messages.extend(framed_messages)
                continue

            file_timestamp_ns = _extract_filename_timestamp_ns(dds_file)
            messages.append((file_timestamp_ns, data))

        messages.sort(key=lambda item: item[0])
        return messages

    def read_lidar_pointcloud(
        self, max_frames: int = 100
    ) -> list[tuple[int, np.ndarray]]:
        """Read LiDAR PCD files from zip and use filename timestamps."""
        lidar_dir = f"{self.uuid}/data/lidar/"
        lidar_files = [
            n
            for n in self.zip_file.namelist()
            if n.startswith(lidar_dir)
            and n.endswith(".pcd")
            and "/._" not in n
            and not Path(n).name.startswith(".")
        ]
        if not lidar_files:
            return []

        frames = []
        lidar_files.sort(key=_extract_lidar_pcd_timestamp_ns)
        for lidar_file in lidar_files:
            timestamp_ns = _extract_lidar_pcd_timestamp_ns(lidar_file)
            if timestamp_ns <= 0:
                print(f"  Skipping lidar PCD with invalid timestamp: {Path(lidar_file).name}")
                continue

            try:
                points = _parse_pcd_pointcloud(self.zip_file.read(lidar_file))
            except KeyError:
                continue

            if points is None:
                print(f"  Skipping unreadable lidar PCD: {Path(lidar_file).name}")
                continue

            frames.append((timestamp_ns, points))
            if len(frames) >= max_frames:
                break

        return frames

    def read_falcon_pointcloud(
        self, max_frames: int = 100
    ) -> list[tuple[int, np.ndarray]]:
        """Backward-compatible alias for the old LiDAR reader name."""
        return self.read_lidar_pointcloud(max_frames=max_frames)

    def close(self):
        self.zip_file.close()


def decode_video_to_frames(
    video_data: bytes,
    frame_metadata: Optional[list[CameraFrame]] = None,
    max_frames: int = 100,
) -> list[tuple[int, np.ndarray]]:
    """Decode H.264/H.265 video to frames using ffmpeg."""
    frames = []

    # Determine codec from first bytes
    if len(video_data) > 4:
        if video_data[:4] == b"\x00\x00\x00\x01" or video_data[:3] == b"\x00\x00\x01":
            # Check NAL unit type for H.264/H.265
            nal_type = video_data[4] & 0x1F if len(video_data) > 4 else 0
            if nal_type in [1, 5, 6, 7, 8, 9]:  # H.264 NAL types
                ext = ".h264"
            else:
                ext = ".h265"
        else:
            ext = ".h264"  # Default to h264 based on actual data
    else:
        ext = ".h264"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(video_data)
        video_path = f.name

    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = os.path.join(tmpdir, "frame_%04d.png")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            output_pattern,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            import glob

            png_files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.png")))

            for i, png_file in enumerate(png_files[:max_frames]):
                frame = cv2.imread(png_file)
                if frame is not None:
                    fallback_timestamp_ns = i * 100_000_000
                    if frame_metadata and i < len(frame_metadata):
                        timestamp_ns = _camera_frame_timestamp_ns(
                            frame_metadata[i], fallback=fallback_timestamp_ns
                        )
                    else:
                        timestamp_ns = fallback_timestamp_ns
                    frames.append((timestamp_ns, frame))

    try:
        os.unlink(video_path)
    except:
        pass

    return frames


def parse_lidar_cluster_objects(data: bytes) -> Optional[LidarClusterObjects]:
    """Parse LidarClusterObjects protobuf message."""
    try:
        msg = LidarClusterObjects()
        msg.ParseFromString(data)
        return msg
    except Exception:
        return None


def parse_perception_objects(data: bytes) -> Optional[ObjectsDetection]:
    """Parse PerceptionObjects protobuf message."""
    try:
        msg = ObjectsDetection()
        msg.ParseFromString(data)
        return msg
    except Exception:
        return None


def convert_nio_to_rerun(
    zip_path: str,
    output_path: str,
    max_frames_per_camera: int = 100,
    include_cameras: bool = True,
    include_lidar: bool = True,
    include_objects: bool = True,
) -> None:
    """Convert NIO data packet to Rerun RRD format."""

    print("\n" + "=" * 60)
    print("NIO to Rerun Converter")
    print("=" * 60)

    # Initialize Rerun recording
    rr.init("nio_data_converter", recording_id=output_path)
    rr.save(output_path)

    extractor = NIODataExtractor(zip_path)

    print(f"\n📦 Input: {zip_path}")
    print(f"💾 Output: {output_path}")
    print(f"🆔 UUID: {extractor.uuid}")

    # Camera topic to Rerun path mapping
    # For 2D view: camera_2d/<name>
    # For 3D projection: camera_3d/<name>/image + camera_3d/<name>/pinhole
    camera_mapping = {
        "/camera/front/main": ("camera_2d/front_wide", "camera_3d/front_wide"),
        "/camera/front/narrow": ("camera_2d/front_narrow", "camera_3d/front_narrow"),
        "/camera/rear": ("camera_2d/rear", "camera_3d/rear"),
        "/camera/surrouding/front": ("camera_2d/park_front", "camera_3d/park_front"),
        "/camera/surrouding/left": ("camera_2d/park_left", "camera_3d/park_left"),
        "/camera/surrouding/rear": ("camera_2d/park_rear", "camera_3d/park_rear"),
        "/camera/surrouding/right": ("camera_2d/park_right", "camera_3d/park_right"),
        "/camera/side/front/left": ("camera_2d/side_fl", "camera_3d/side_fl"),
        "/camera/side/front/right": ("camera_2d/side_fr", "camera_3d/side_fr"),
        "/camera/side/rear/left": ("camera_2d/side_rl", "camera_3d/side_rl"),
        "/camera/side/rear/right": ("camera_2d/side_rr", "camera_3d/side_rr"),
    }

    # Camera extrinsic (position in vehicle frame) - to be logged once
    camera_positions = {
        "/camera/front/main": ([3.5, 0, 1.6], [0, 0, 0]),  # front, center, 1.6m height
        "/camera/front/narrow": ([3.5, 0, 1.6], [0, 0, 0]),
        "/camera/rear": ([-3.0, 0, 1.6], [0, math.pi, 0]),  # rear, facing back
        "/camera/surrouding/front": ([3.0, 0, 1.8], [0, 0, 0]),
        "/camera/surrouding/left": ([0, 1.5, 1.8], [0, -math.pi / 2, 0]),
        "/camera/surrouding/rear": ([-3.0, 0, 1.8], [0, math.pi, 0]),
        "/camera/surrouding/right": ([0, -1.5, 1.8], [0, math.pi / 2, 0]),
        "/camera/side/front/left": ([3.0, 1.0, 1.6], [0, -math.pi / 6, 0]),
        "/camera/side/front/right": ([3.0, -1.0, 1.6], [0, math.pi / 6, 0]),
        "/camera/side/rear/left": ([-2.5, 1.0, 1.6], [0, -math.pi + math.pi / 6, 0]),
        "/camera/side/rear/right": ([-2.5, -1.0, 1.6], [0, math.pi - math.pi / 6, 0]),
    }

    # Camera intrinsic parameters (default for NIO cameras)
    # These should come from calibration files, using reasonable defaults
    camera_intrinsics = {
        "/camera/front/main": (
            1920,
            1200,
            1000,
            600,
            1000,
            600,
        ),  # fx, fy, cx, cy, w, h
        "/camera/front/narrow": (1920, 1200, 960, 600, 1920, 1200),
        "/camera/rear": (1920, 1200, 960, 600, 1920, 1200),
        "/camera/surrouding/front": (1280, 800, 640, 400, 1280, 800),
        "/camera/surrouding/left": (1280, 800, 640, 400, 1280, 800),
        "/camera/surrouding/rear": (1280, 800, 640, 400, 1280, 800),
        "/camera/surrouding/right": (1280, 800, 640, 400, 1280, 800),
        "/camera/side/front/left": (1280, 800, 640, 400, 1280, 800),
        "/camera/side/front/right": (1280, 800, 640, 400, 1280, 800),
        "/camera/side/rear/left": (1280, 800, 640, 400, 1280, 800),
        "/camera/side/rear/right": (1280, 800, 640, 400, 1280, 800),
    }

    # Convert cameras
    if include_cameras:
        print("\n[PROCESSING] Converting cameras...")
        for topic, rerun_paths in camera_mapping.items():
            print(f"  Processing {topic}...", end=" ")

            h265_data, timestamps = extractor.read_camera_video(topic)
            if not h265_data:
                print("No data found")
                continue

            frames = decode_video_to_frames(
                h265_data,
                frame_metadata=timestamps,
                max_frames=max_frames_per_camera,
            )
            print(f"Decoded {len(frames)} frames")

            # Get calibration data for this camera
            calib_name = extractor.camera_calib_map.get(topic)
            calib_data = extractor.calibration.get(calib_name) if calib_name else None

            # Image path (separate from camera entity)
            image_path = f"{rerun_paths[0]}/image"

            for timestamp_ns, frame in frames:
                _log_camera_frame(
                    image_path=image_path,
                    camera_path=rerun_paths[1],
                    timestamp_ns=timestamp_ns,
                    frame=frame,
                    calib_data=calib_data,
                )

    # Convert lidar cluster objects
    if include_lidar:
        print("\n[PROCESSING] Converting lidar...")

        # Try point cloud first
        print("  Reading PCD point cloud...")
        pointcloud_frames = extractor.read_lidar_pointcloud(
            max_frames=max_frames_per_camera
        )
        print(f"  Found {len(pointcloud_frames)} point cloud frames")

        if pointcloud_frames:
            for timestamp_ns, points in pointcloud_frames:
                time_sec = timestamp_ns / 1e9
                rr.set_time(RERUN_TIMELINE, timestamp=time_sec)
                # Separate positions (x,y,z) and colors (from intensity)
                positions = points[:, :3] if points.shape[1] >= 3 else points
                intensities = points[:, 3] if points.shape[1] >= 4 else None

                if intensities is not None:
                    intensities = intensities.astype(np.float32, copy=False)
                    finite_mask = np.isfinite(intensities)
                    if np.any(finite_mask):
                        valid_intensities = intensities[finite_mask]
                        min_intensity = float(valid_intensities.min())
                        max_intensity = float(valid_intensities.max())
                        if 0.0 <= min_intensity and max_intensity <= 1.0:
                            color_values = np.clip(intensities, 0.0, 1.0)
                        elif max_intensity > min_intensity:
                            color_values = (intensities - min_intensity) / (
                                max_intensity - min_intensity
                            )
                        else:
                            color_values = np.ones_like(intensities)
                    else:
                        color_values = np.zeros_like(intensities)

                    # Map intensity to grayscale color
                    colors = np.column_stack(
                        [
                            (color_values * 255).astype(np.uint8),
                            (color_values * 255).astype(np.uint8),
                            (color_values * 255).astype(np.uint8),
                        ]
                    )
                    rr.log(
                        "lidar/pointcloud",
                        rr.Points3D(
                            positions=positions,
                            colors=colors,
                        ),
                    )
                else:
                    rr.log(
                        "lidar/pointcloud",
                        rr.Points3D(
                            positions=positions,
                        ),
                    )
            print(f"  Logged {len(pointcloud_frames)} point cloud frames")

        # Also try DDS messages
        lidar_topic = "/perception/lidar_cluster"
        messages = extractor.read_dds_message(lidar_topic)
        print(f"  Found {len(messages)} lidar DDS messages")

        count = 0
        previous_cluster_box_paths: set[str] = set()
        for file_timestamp_ns, data in messages[:500]:
            clusters = parse_lidar_cluster_objects(data)
            if not clusters:
                continue

            timestamp_ns = _lidar_cluster_timestamp_ns(
                clusters, fallback=file_timestamp_ns
            )
            time_sec = timestamp_ns / 1e9
            current_cluster_logs: list[tuple[str, rr.Boxes3D]] = []
            current_cluster_box_paths: set[str] = set()

            for i, obj in enumerate(clusters.lidar_cluster_object_list):
                center = np.array(
                    [
                        obj.lidar_cluster_center_x,
                        obj.lidar_cluster_center_y,
                        obj.lidar_cluster_center_z,
                    ]
                )
                size = np.array(
                    [
                        obj.lidar_cluster_length,
                        obj.lidar_cluster_width,
                        obj.lidar_cluster_height,
                    ]
                )
                if np.any(size <= 0):
                    continue

                entity_name = f"lidar/cluster/{i}"
                box_kwargs = {
                    "centers": [center],
                    "sizes": [size],
                }
                if obj.HasField("lidar_cluster_mbr_yaw"):
                    box_kwargs["rotations"] = [
                        rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=obj.lidar_cluster_mbr_yaw)
                    ]

                box_path = f"{entity_name}/box"
                current_cluster_box_paths.add(box_path)
                current_cluster_logs.append((box_path, rr.Boxes3D(**box_kwargs)))

            _clear_stale_entity_paths(
                previous_cluster_box_paths, current_cluster_box_paths, timestamp_ns
            )
            rr.set_time(RERUN_TIMELINE, timestamp=time_sec)
            for box_path, box in current_cluster_logs:
                rr.log(box_path, box)
                count += 1
            previous_cluster_box_paths = current_cluster_box_paths

        print(f"  Logged {count} cluster objects")

    # Convert perception objects
    if include_objects:
        print("\n[PROCESSING] Converting perception objects...")
        obj_topic = "/perception/perception_objects"
        messages = extractor.read_dds_message(obj_topic)
        print(f"  Found {len(messages)} perception messages")

        count = 0
        previous_object_paths: set[str] = set()
        for file_timestamp_ns, data in messages[:200]:
            objects = parse_perception_objects(data)
            if not objects:
                continue

            timestamp_ns = _message_timestamp_ns(objects, fallback=file_timestamp_ns)
            time_sec = timestamp_ns / 1e9
            current_object_paths: set[str] = set()
            current_object_logs: list[tuple[str, object]] = []

            for obj in objects.dynamicobj.OBJ:
                pos = np.array(
                    [
                        obj.OBJ_Distance.Long_Position,
                        obj.OBJ_Distance.Lat_Position,
                        obj.OBJ_Distance.Vertical_Position,
                    ]
                )
                size = np.array(
                    [
                        obj.OBJ_Dimension.OBJ_Length,
                        obj.OBJ_Dimension.OBJ_Width,
                        obj.OBJ_Dimension.OBJ_Height,
                    ]
                )
                if np.any(size <= 0):
                    continue

                vel = np.array(
                    [
                        obj.OBJ_Abs_Velocity.Long_Velocity,
                        obj.OBJ_Abs_Velocity.Lat_Velocity,
                        obj.OBJ_Abs_Velocity.Up_Velocity,
                    ]
                )

                entity_name = f"perception/object/{obj.OBJ_Object_ID}"
                box_kwargs = {
                    "centers": [pos],
                    "sizes": [size],
                }
                if obj.HasField("OBJ_Heading"):
                    box_kwargs["rotations"] = [
                        rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=obj.OBJ_Heading)
                    ]

                box_path = f"{entity_name}/box"
                velocity_path = f"{entity_name}/velocity"
                current_object_paths.update((box_path, velocity_path))
                current_object_logs.append((box_path, rr.Boxes3D(**box_kwargs)))
                current_object_logs.append(
                    (
                        velocity_path,
                        rr.Arrows3D(
                            vectors=[vel * 0.1],
                            origins=[pos],
                        ),
                    )
                )

                count += 1

            _clear_stale_entity_paths(previous_object_paths, current_object_paths, timestamp_ns)
            rr.set_time(RERUN_TIMELINE, timestamp=time_sec)
            for entity_path, archetype in current_object_logs:
                rr.log(entity_path, archetype)
            previous_object_paths = current_object_paths

        print(f"  Logged {count} perception objects")

    extractor.close()
    rr.disconnect()

    print("\n" + "=" * 60)
    print(f"✅ Conversion complete!")
    print(f"💾 Output: {output_path}")
    print("=" * 60)
    print("\nTo view in Rerun:")
    print(f"  rerun {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NIO SIRIUS data to Rerun RRD format"
    )
    parser.add_argument("input", help="Path to input .zip file")
    parser.add_argument(
        "output", nargs="?", help="Path to output .rrd file (default: <input_name>.rrd)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames per camera (default: 100)",
    )
    parser.add_argument(
        "--no-cameras", action="store_true", help="Skip camera conversion"
    )
    parser.add_argument("--no-lidar", action="store_true", help="Skip lidar conversion")
    parser.add_argument(
        "--no-objects", action="store_true", help="Skip perception objects conversion"
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        input_name = Path(args.input).stem
        args.output = f"{input_name}.rrd"

    convert_nio_to_rerun(
        zip_path=args.input,
        output_path=args.output,
        max_frames_per_camera=args.max_frames,
        include_cameras=not args.no_cameras,
        include_lidar=not args.no_lidar,
        include_objects=not args.no_objects,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NIO Data to Rerun RRD Converter

Converts NIO SIRIUS (NT2) autonomous driving data packets to Rerun's .rrd format.

Supported data types:
- Camera: H.265 video → Rerun Image
- Lidar Cluster Objects: Protobuf → Rerun Boxes + EntityNames
- Perception Objects: Protobuf → Rerun Boxes + Velocities

Usage:
    python nio_to_rerun.py <path_to_dat.zip> [output.rrd]

Dependencies:
    pip install rerun-sdk opencv-python protobuf numpy
"""

import argparse
import importlib
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

from common.perception.lidar_cluster_output_pb2 import (
    LidarClusterObjects,
    LidarSemanticClass,
)
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


FALCON_MAGIC = 0x4451
FALCON_COMMON_HEADER_SIZE = 26
FALCON_PACKET_INFO_SIZE = 28
FALCON_PACKET_HEADER_SIZE = FALCON_COMMON_HEADER_SIZE + FALCON_PACKET_INFO_SIZE
FALCON_BLOCK_HEADER_SIZE = 17
_FALCON_SDK_CACHE: Optional[tuple[object | None, object | None, str | None]] = None


def _falcon_angle_to_radians(raw_value: int) -> float:
    """Convert Falcon uint16 angle to radians in [-pi, pi)."""
    return (raw_value / 32768.0 - 1.0) * math.pi


def _extract_falcon_type_and_item_count(packet_info: bytes) -> tuple[int, int]:
    """Decode packed Falcon packet type and block count."""
    packed_value = struct.unpack_from("<I", packet_info, 12)[0]
    data_type = packed_value & 0xFF
    item_count = packed_value >> 8
    return data_type, item_count


def _load_falcon_sdk() -> tuple[object | None, object | None, str | None]:
    """Load FalconReader from datafilter-sdk when available."""
    global _FALCON_SDK_CACHE
    if _FALCON_SDK_CACHE is not None:
        return _FALCON_SDK_CACHE

    extra_pythonpath = os.environ.get("NIO_FALCON_SDK_PYTHONPATH", "")
    for path in extra_pythonpath.split(os.pathsep):
        path = path.strip()
        if path and path not in sys.path:
            sys.path.insert(0, path)

    try:
        rawdatareader = importlib.import_module("datafilter.sdk.rawdatareader")
    except Exception as exc:  # pragma: no cover - exercised via caller behavior
        _FALCON_SDK_CACHE = (None, None, f"{type(exc).__name__}: {exc}")
        return _FALCON_SDK_CACHE

    falcon_reader = getattr(rawdatareader, "FalconReader", None)
    get_falcon_data_13n = getattr(rawdatareader, "get_falcon_data_13n", None)
    if falcon_reader is None:
        _FALCON_SDK_CACHE = (
            None,
            None,
            "AttributeError: datafilter.sdk.rawdatareader.FalconReader missing",
        )
        return _FALCON_SDK_CACHE

    _FALCON_SDK_CACHE = (falcon_reader, get_falcon_data_13n, None)
    return _FALCON_SDK_CACHE


def _downsample_nt3_falcon_points(
    scanner_direction: int, falcon_pcd_13n: np.ndarray, falcon_pcd: np.ndarray
) -> np.ndarray:
    """Match adviz_stream's Falcon downsampling for NT3 lidar."""
    if len(falcon_pcd_13n) != len(falcon_pcd):
        return falcon_pcd

    multireturn = falcon_pcd_13n[:, 8].astype(int)
    scan_id = falcon_pcd_13n[:, 11].astype(int)
    scan_idx = falcon_pcd_13n[:, 12].astype(int)
    keep_indices: list[int] = []

    for i in range(len(falcon_pcd)):
        if multireturn[i]:
            continue
        if scanner_direction == 0:
            in_dense_sector = 7 <= scan_id[i] <= 29 and (
                scan_idx[i] <= 375 or scan_idx[i] >= 775
            )
        elif scanner_direction == 1:
            in_dense_sector = 10 <= scan_id[i] <= 32 and (
                scan_idx[i] <= 375 or scan_idx[i] >= 775
            )
        else:
            in_dense_sector = False

        if in_dense_sector:
            if scan_id[i] % 2 != 0 and scan_idx[i] % 2 != 0:
                keep_indices.append(i)
            continue

        keep_indices.append(i)

    if not keep_indices:
        return falcon_pcd
    return falcon_pcd[keep_indices]


def _coerce_falcon_points(points: object) -> Optional[np.ndarray]:
    """Convert FalconReader output to Nx4 float32 point cloud."""
    points_array = np.asarray(points)
    if points_array.ndim != 2 or points_array.shape[1] < 3:
        return None

    points_array = points_array.astype(np.float32, copy=False)
    xyz = points_array[:, :3]
    if points_array.shape[1] >= 4:
        scalar = points_array[:, 3:4]
    else:
        scalar = np.ones((points_array.shape[0], 1), dtype=np.float32)
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

    def read_falcon_pointcloud(
        self, max_frames: int = 100
    ) -> list[tuple[int, np.ndarray]]:
        """Read Falcon LiDAR point cloud data from .dat file.

        Falcon format (from Seyond official specification):
        - InnoCommonHeader: 26 bytes
        - InnoDataPacket Information: 28 bytes
        - InnoBlock[]: 17 byte header + 4*N bytes per block
          - InnoChannelPoint: 4 bytes
            (radius=17bit, refl=8bit, elongation=4bit,
            is_2nd_return=1bit, type=2bit)
        """
        lidar_dir = f"{self.uuid}/data/lidar/"
        lidar_files = [
            n
            for n in self.zip_file.namelist()
            if n.startswith(lidar_dir) and n.endswith(".dat")
        ]

        if not lidar_files:
            return []

        FalconReader, _, sdk_error = _load_falcon_sdk()
        if FalconReader is None and sdk_error:
            print(
                "  FalconReader unavailable "
                f"({sdk_error}); falling back to built-in parser"
            )

        sdk_frames = self._read_falcon_pointcloud_with_sdk(lidar_files, max_frames)
        if sdk_frames:
            return sdk_frames

        frames_by_index: dict[int, list[list[float]]] = {}
        frame_timestamps_ns: dict[int, int] = {}
        spherical_block_size = FALCON_BLOCK_HEADER_SIZE + 4

        for lidar_file in sorted(lidar_files):
            raw_data = self.zip_file.read(lidar_file)
            offset = 0

            while offset + FALCON_PACKET_HEADER_SIZE <= len(raw_data):
                packet_start = offset
                magic = struct.unpack_from("<H", raw_data, packet_start)[0]
                if magic != FALCON_MAGIC:
                    offset += 1
                    continue

                packet_size = struct.unpack_from("<I", raw_data, packet_start + 10)[0]
                packet_end = packet_start + packet_size
                if (
                    packet_size < FALCON_PACKET_HEADER_SIZE
                    or packet_end > len(raw_data)
                ):
                    offset += 1
                    continue

                ts_start_us = struct.unpack_from("<Q", raw_data, packet_start + 18)[0]
                packet_info_offset = packet_start + FALCON_COMMON_HEADER_SIZE
                packet_info = raw_data[
                    packet_info_offset : packet_info_offset + FALCON_PACKET_INFO_SIZE
                ]

                frame_idx = struct.unpack_from("<Q", packet_info, 0)[0]
                data_type, block_count = _extract_falcon_type_and_item_count(packet_info)
                block_size = struct.unpack_from("<H", packet_info, 16)[0]

                if data_type != 1 or block_count == 0:
                    offset = packet_end
                    continue

                payload_offset = packet_start + FALCON_PACKET_HEADER_SIZE
                remaining_bytes = packet_end - payload_offset
                if block_size == 0 and remaining_bytes == block_count * spherical_block_size:
                    block_size = spherical_block_size

                if block_size != spherical_block_size:
                    offset = packet_end
                    continue
                if remaining_bytes < block_count * block_size:
                    offset = packet_end
                    continue

                frame_points = frames_by_index.setdefault(frame_idx, [])
                timestamp_ns = ts_start_us * 1000
                previous_timestamp_ns = frame_timestamps_ns.get(frame_idx)
                if previous_timestamp_ns is None:
                    frame_timestamps_ns[frame_idx] = timestamp_ns
                else:
                    frame_timestamps_ns[frame_idx] = min(
                        previous_timestamp_ns, timestamp_ns
                    )

                block_offset = payload_offset
                for _ in range(block_count):
                    if block_offset + block_size > packet_end:
                        break

                    block_header = raw_data[
                        block_offset : block_offset + FALCON_BLOCK_HEADER_SIZE
                    ]
                    h_angle_raw, v_angle_raw = struct.unpack_from("<HH", block_header, 0)
                    h_angle = _falcon_angle_to_radians(h_angle_raw)
                    v_angle = _falcon_angle_to_radians(v_angle_raw)
                    cos_v = math.cos(v_angle)
                    sin_v = math.sin(v_angle)
                    cos_h = math.cos(h_angle)
                    sin_h = math.sin(h_angle)

                    point_data = struct.unpack_from(
                        "<I", raw_data, block_offset + FALCON_BLOCK_HEADER_SIZE
                    )[0]
                    radius_raw = point_data & 0x1FFFF
                    reflectance = (point_data >> 17) & 0xFF
                    radius_m = radius_raw / 200.0

                    if not 0.1 < radius_m < 655.35:
                        block_offset += block_size
                        continue

                    x = radius_m * cos_v * cos_h
                    y = radius_m * cos_v * sin_h
                    z = radius_m * sin_v
                    frame_points.append([x, y, z, reflectance / 255.0])

                    block_offset += block_size

                offset = packet_end

        if not frames_by_index:
            return []

        frames = []
        for frame_idx in sorted(frames_by_index):
            frame_points = frames_by_index[frame_idx]
            if not frame_points:
                continue

            frames.append(
                (
                    frame_timestamps_ns[frame_idx],
                    np.array(frame_points, dtype=np.float32),
                )
            )

            if len(frames) >= max_frames:
                break

        return frames

    def _read_falcon_pointcloud_with_sdk(
        self, lidar_files: list[str], max_frames: int
    ) -> list[tuple[int, np.ndarray]]:
        """Use datafilter-sdk's FalconReader when available."""
        FalconReader, get_falcon_data_13n, _ = _load_falcon_sdk()
        if FalconReader is None:
            return []

        frames: list[tuple[int, np.ndarray]] = []
        for lidar_file in sorted(lidar_files):
            with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as temp_file:
                temp_file.write(self.zip_file.read(lidar_file))
                temp_path = temp_file.name

            reader = None
            try:
                reader = FalconReader(temp_path)
                if hasattr(reader, "set_fillrow_mode"):
                    try:
                        reader.set_fillrow_mode()
                    except Exception:
                        pass

                frame_with_info = getattr(reader, "get_frame_with_info", None)
                if callable(frame_with_info):
                    while len(frames) < max_frames:
                        frame = frame_with_info()
                        if frame is None:
                            break

                        falcon_pcd, frameinfo = frame
                        points = np.asarray(falcon_pcd)
                        if (
                            callable(get_falcon_data_13n)
                            and points.ndim == 2
                            and len(points) > 0
                            and isinstance(frameinfo, dict)
                            and "scanner_direction" in frameinfo
                        ):
                            try:
                                falcon_pcd_13n = np.asarray(get_falcon_data_13n(points))
                                points = _downsample_nt3_falcon_points(
                                    int(frameinfo.get("scanner_direction", 0)),
                                    falcon_pcd_13n,
                                    points,
                                )
                            except Exception:
                                points = np.asarray(falcon_pcd)

                        normalized = _coerce_falcon_points(points)
                        if normalized is None:
                            continue

                        timestamp_ns = int(frameinfo["frame_starttime"] * 1e9)
                        frames.append((timestamp_ns, normalized))
                else:
                    while len(frames) < max_frames:
                        frame = reader.get_next_frame()
                        if frame is None:
                            break

                        _, frame_starttime, falcon_pcd = frame
                        normalized = _coerce_falcon_points(falcon_pcd)
                        if normalized is None:
                            continue
                        frames.append((int(frame_starttime * 1e9), normalized))
            except Exception as exc:
                print(
                    f"  FalconReader failed for {Path(lidar_file).name}: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                if reader is not None and hasattr(reader, "close"):
                    try:
                        reader.close()
                    except Exception:
                        pass
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

            if len(frames) >= max_frames:
                break

        frames.sort(key=lambda item: item[0])
        return frames[:max_frames]

    def close(self):
        self.zip_file.close()


def decode_video_to_frames(
    video_data: bytes, max_frames: int = 100
) -> list[tuple[float, np.ndarray]]:
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
                    time_sec = i * 0.1  # ~10fps
                    frames.append((time_sec, frame))

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

            frames = decode_video_to_frames(h265_data, max_frames_per_camera)
            print(f"Decoded {len(frames)} frames")

            # Get calibration data for this camera
            calib_name = extractor.camera_calib_map.get(topic)
            calib_data = extractor.calibration.get(calib_name) if calib_name else None

            # Image path (separate from camera entity)
            image_path = f"{rerun_paths[0]}/image"

            for time_sec, frame in frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame_rgb.shape[:2]
                rr.set_time("camera_time", timestamp=time_sec)

                # Log image to separate path
                rr.log(image_path, rr.Image(frame_rgb))

                # Set up pinhole camera for 3D projection (if calibration available)
                if calib_data and "calibration_info" in calib_data:
                    calib = calib_data["calibration_info"]
                    if (
                        "camera_matrix" in calib
                        and "intrinsic" in calib["camera_matrix"]
                    ):
                        intrinsic = calib["camera_matrix"]["intrinsic"]
                        fx = intrinsic.get("fx", width * 1.2)
                        fy = intrinsic.get("fy", height * 1.2)
                        cx = intrinsic.get("cx", width / 2)
                        cy = intrinsic.get("cy", height / 2)

                        rr.log(
                            rerun_paths[1],
                            rr.Pinhole(
                                image_path=image_path,
                                resolution=(width, height),
                                focal_length_px=(fx, fy),
                            ),
                        )

    # Convert lidar cluster objects
    if include_lidar:
        print("\n[PROCESSING] Converting lidar...")

        # Try point cloud first
        print("  Reading Falcon point cloud...")
        pointcloud_frames = extractor.read_falcon_pointcloud()
        print(f"  Found {len(pointcloud_frames)} point cloud frames")

        if pointcloud_frames:
            for i, (timestamp_ns, points) in enumerate(pointcloud_frames[:20]):
                time_sec = timestamp_ns / 1e9
                rr.set_time("lidar_time", timestamp=time_sec)
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
            print(f"  Logged {len(pointcloud_frames[:20])} point cloud frames")

        # Also try DDS messages
        lidar_topic = "/perception/lidar_cluster"
        messages = extractor.read_dds_message(lidar_topic)
        print(f"  Found {len(messages)} lidar DDS messages")

        count = 0
        for file_timestamp_ns, data in messages[:500]:
            clusters = parse_lidar_cluster_objects(data)
            if not clusters:
                continue

            timestamp_ns = _lidar_cluster_timestamp_ns(
                clusters, fallback=file_timestamp_ns
            )
            time_sec = timestamp_ns / 1e9

            for i, obj in enumerate(clusters.lidar_cluster_object_list):
                class_name = LidarSemanticClass.Name(obj.lidar_cluster_class)

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
                    "labels": [class_name],
                }
                if obj.HasField("lidar_cluster_mbr_yaw"):
                    box_kwargs["rotations"] = [
                        rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=obj.lidar_cluster_mbr_yaw)
                    ]

                rr.set_time("lidar_time", timestamp=time_sec)
                rr.log(entity_name, rr.TextLog(class_name))
                rr.log(
                    f"{entity_name}/box",
                    rr.Boxes3D(**box_kwargs),
                )

                count += 1

        print(f"  Logged {count} cluster objects")

    # Convert perception objects
    if include_objects:
        print("\n[PROCESSING] Converting perception objects...")
        obj_topic = "/perception/perception_objects"
        messages = extractor.read_dds_message(obj_topic)
        print(f"  Found {len(messages)} perception messages")

        count = 0
        for file_timestamp_ns, data in messages[:200]:
            objects = parse_perception_objects(data)
            if not objects:
                continue

            timestamp_ns = _message_timestamp_ns(objects, fallback=file_timestamp_ns)
            time_sec = timestamp_ns / 1e9

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

                class_name = DynamicObj.OBJObjectClass.Name(obj.OBJ_Object_Class)
                entity_name = f"perception/object/{obj.OBJ_Object_ID}"
                box_kwargs = {
                    "centers": [pos],
                    "sizes": [size],
                    "labels": [class_name],
                }
                if obj.HasField("OBJ_Heading"):
                    box_kwargs["rotations"] = [
                        rr.RotationAxisAngle([0.0, 0.0, 1.0], radians=obj.OBJ_Heading)
                    ]

                rr.set_time("perception_time", timestamp=time_sec)
                rr.log(entity_name, rr.TextLog(class_name))
                rr.log(
                    f"{entity_name}/box",
                    rr.Boxes3D(**box_kwargs),
                )
                rr.log(
                    f"{entity_name}/velocity",
                    rr.Arrows3D(
                        vectors=[vel * 0.1],
                        origins=[pos],
                    ),
                )

                count += 1

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

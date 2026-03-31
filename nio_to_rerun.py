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

    def _init_dds_mapping(self):
        """Initialize DDS topic to proto message mapping."""
        self.dds_mapping = {
            "/perception/objects": "PerceptionObjects",
            "/perception/predicted_objects": "PerceptionObjects",
            "/perception/lidar_cluster": "LidarClusterObjects",
        }

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
        topic_info = self.da_meta["topics"].get(topic_name)
        if not topic_info:
            return []

        # Get the DDS filename from the topic name
        # Topic: /common/perception/perception_objects
        # File: c4e56baf-..._common-perception-perception_objects_*.pb.dat
        topic_clean = topic_name.lstrip("/").replace("/", "-")
        uuid = self.uuid

        # Find matching files in the zip
        dds_files = []
        dds_prefix = f"{uuid}/data/dds/{uuid}_"

        for name in self.zip_file.namelist():
            if name.startswith(dds_prefix) and ".pb.dat" in name:
                # Extract topic part from filename
                # Format: UUID_topic_subtopic_timestamp.pb.dat
                remaining = name[len(dds_prefix) :]
                if "_" in remaining:
                    file_topic_part = remaining.rsplit("_", 1)[0]
                    # Match with normalized topic
                    if (
                        file_topic_part == topic_clean
                        or f"-{topic_clean}" in name
                        or name.endswith(f"_{topic_clean}_")
                    ):
                        dds_files.append(name)

        messages = []
        for dds_file in dds_files:
            try:
                data = self.zip_file.read(dds_file)
                # Extract timestamp from filename if possible
                # Format: UUID_topic_timestamp.pb.dat
                parts = dds_file.rstrip(".pb.dat").split("_")
                if parts:
                    try:
                        timestamp = int(parts[-1])
                    except ValueError:
                        timestamp = 0
                else:
                    timestamp = 0

                messages.append((timestamp, data))
            except KeyError:
                pass

        return messages

    def read_falcon_pointcloud(
        self, max_frames: int = 100
    ) -> list[tuple[int, np.ndarray]]:
        """Read Falcon LiDAR point cloud data from .dat file.

        Falcon format (from Seyond official specification):
        - InnoCommonHeader: 26 bytes
        - InnoDataPacket Information: 28 bytes
        - InnoBlock[]: 17 byte header + 4*N bytes per block
          - InnoChannelPoint: 4 bytes (radius=17bit, refl=8bit, elongation=4bit, is_2nd_return=1bit, type=2bit)
        """
        lidar_dir = f"{self.uuid}/data/lidar/"
        lidar_files = [
            n
            for n in self.zip_file.namelist()
            if n.startswith(lidar_dir) and n.endswith(".dat")
        ]

        if not lidar_files:
            return []

        # Use first lidar file
        lidar_file = lidar_files[0]
        raw_data = self.zip_file.read(lidar_file)

        points_data = []
        offset = 0

        # Parse InnoDataPackets
        while offset < len(raw_data) - 54 and len(points_data) < max_frames * 10000:
            # InnoCommonHeader: 26 bytes
            if offset + 26 > len(raw_data):
                break
            header = raw_data[offset : offset + 26]

            # Parse header fields
            magic = struct.unpack("<H", header[0:2])[0]
            if magic != 0x4451:  # 'P1D' magic
                offset += 1
                continue

            version = struct.unpack("<HH", header[2:6])
            checksum = struct.unpack("<I", header[6:10])[0]
            packet_size = struct.unpack("<I", header[10:14])[0]
            ts_start_us = struct.unpack("<Q", header[18:26])[0]

            offset += 26

            # InnoDataPacket Information: 28 bytes
            if offset + 28 > len(raw_data):
                break
            pkt_info = raw_data[offset : offset + 28]

            frame_idx = struct.unpack("<Q", pkt_info[0:8])[0]
            sub_idx = struct.unpack("<H", pkt_info[8:10])[0]
            data_type = pkt_info[12]
            block_count = struct.unpack("<I", pkt_info[16:20])[0]
            block_size = struct.unpack("<H", pkt_info[20:22])[0]

            offset += 28

            # Parse InnoBlocks
            for _ in range(min(block_count, 100)):
                if offset + 16 > len(raw_data):
                    break

                block_header = raw_data[offset : offset + 16]
                h_angle_raw = struct.unpack("<H", block_header[0:2])[0]
                v_angle_raw = struct.unpack("<H", block_header[2:4])[0]
                flags = struct.unpack("<H", block_header[4:6])[0]
                ts_10us = struct.unpack("<H", block_header[14:16])[0]

                # Convert angles (uint16 to radians: -π to π)
                h_angle = (h_angle_raw / 32768.0 - 1.0) * math.pi
                v_angle = (v_angle_raw / 32768.0 - 1.0) * math.pi

                # Time offset from packet start (10μs per unit)
                timestamp_ns = ts_start_us * 1000000 + ts_10us * 10000

                offset += 16

                # Parse points (4 bytes each, use block_size if available)
                if block_size > 0:
                    num_points = block_size // 4
                else:
                    num_points = min(
                        32, (packet_size - 54 - 16) // 4
                    )  # Default ~32 points per block

                for i in range(num_points):
                    if offset + 4 > len(raw_data):
                        break

                    point_data = struct.unpack("<I", raw_data[offset : offset + 4])[0]

                    # Extract fields from 32-bit integer
                    radius_raw = point_data & 0x1FFFF  # 17 bits
                    refl = (point_data >> 17) & 0xFF  # 8 bits
                    elongation = (point_data >> 25) & 0xF  # 4 bits

                    # Convert radius (1/200 meters to meters)
                    radius = radius_raw / 200.0

                    # Skip invalid points
                    if radius > 0.1 and radius < 200.0:
                        # Convert spherical to Cartesian
                        h_angle_i = h_angle + (i % 32) * 0.01  # Slight angle variation

                        x = radius * math.cos(v_angle) * math.cos(h_angle_i)
                        y = radius * math.cos(v_angle) * math.sin(h_angle_i)
                        z = radius * math.sin(v_angle)

                        points_data.append([x, y, z, refl / 255.0])

                    offset += 4

                    # Limit points per frame
                    if len(points_data) % 10000 == 0:
                        break

        if not points_data:
            return []

        # Convert to numpy array
        points = np.array(points_data, dtype=np.float32)

        # Split into frames (approximately 10000 points per frame)
        frames = []
        frame_size = 10000
        for i in range(0, len(points), frame_size):
            if len(frames) >= max_frames:
                break
            frame_points = points[i : i + frame_size]
            timestamp_ns = int(1759987733576579500 + i * 100000)  # Approximate
            frames.append((timestamp_ns, frame_points))

        return frames

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
                    # Map intensity to grayscale color
                    colors = np.column_stack(
                        [
                            (intensities * 255).astype(np.uint8),
                            (intensities * 255).astype(np.uint8),
                            (intensities * 255).astype(np.uint8),
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
        for timestamp_ns, data in messages[:500]:
            clusters = parse_lidar_cluster_objects(data)
            if not clusters:
                continue

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

                entity_name = f"lidar/cluster/{i}"

                rr.set_time("lidar_time", timestamp=time_sec)
                rr.log(entity_name, rr.TextLog(class_name))
                rr.log(
                    f"{entity_name}/box",
                    rr.Boxes3D(
                        centers=center,
                        sizes=size,
                    ),
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
        for timestamp_ns, data in messages[:200]:
            objects = parse_perception_objects(data)
            if not objects:
                continue

            time_sec = timestamp_ns / 1e9

            for obj in objects.dynamicobj.OBJ:
                pos = np.array(
                    [
                        obj.obj_position.Long_Position,
                        obj.obj_position.Lat_Position,
                        obj.obj_position.Vertical_Position,
                    ]
                )
                size = np.array(
                    [
                        obj.obj_dimension.OBJ_Length,
                        obj.obj_dimension.OBJ_Width,
                        obj.obj_dimension.OBJ_Height,
                    ]
                )

                vel = np.array(
                    [
                        obj.obj_velocity.Long_Velocity,
                        obj.obj_velocity.Lat_Velocity,
                        obj.obj_velocity.Up_Velocity,
                    ]
                )

                class_name = DynamicObj.OBJObjectClass.Name(obj.obj_object_class)
                entity_name = f"perception/object/{obj.obj_object_id}"

                rr.set_time("perception_time", timestamp=time_sec)
                rr.log(entity_name, rr.TextLog(class_name))
                rr.log(
                    f"{entity_name}/box",
                    rr.Boxes3D(
                        centers=pos,
                        sizes=size,
                    ),
                )
                rr.log(
                    f"{entity_name}/velocity",
                    rr.Arrows3D(
                        vectors=vel * 0.1,
                        origins=pos,
                    ),
                )

                count += 1

        print(f"  Logged {count} perception objects")

    extractor.close()

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

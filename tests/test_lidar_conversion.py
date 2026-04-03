import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import nio_to_rerun as mod
from common.perception.lidar_cluster_output_pb2 import LidarClusterObjects
from common.perception.perception_objects_pb2 import ObjectsDetection


def _write_zip(entries: dict[str, bytes]) -> tuple[str, tempfile.TemporaryDirectory]:
    temp_dir = tempfile.TemporaryDirectory()
    zip_path = Path(temp_dir.name) / "fixture.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return str(zip_path), temp_dir


def _build_ascii_pcd(points: list[list[float]]) -> bytes:
    lines = [
        "# .PCD v0.7",
        "VERSION 0.7",
        "FIELDS x y z intensity",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {len(points)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(points)}",
        "DATA ascii",
    ]
    lines.extend(" ".join(str(value) for value in point) for point in points)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_binary_pcd(points: np.ndarray) -> bytes:
    header = "\n".join(
        [
            "# .PCD v0.7",
            "VERSION 0.7",
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
            "",
        ]
    ).encode("utf-8")
    return header + points.astype(np.float32).tobytes()


class LidarConversionTests(unittest.TestCase):
    def test_iter_framed_pb_messages_extracts_payloads(self) -> None:
        payload_a = b"\x08\x01"
        payload_b = b"\x08\x02"
        framed = b"".join(
            [
                struct.pack("<QQQQQ", 100, 100, 200, 200, len(payload_a)),
                payload_a,
                struct.pack("<QQQQQ", 300, 300, 400, 400, len(payload_b)),
                payload_b,
            ]
        )

        self.assertEqual(
            mod._iter_framed_pb_messages(framed),
            [
                (100, payload_a),
                (300, payload_b),
            ],
        )

    def test_read_dds_message_matches_exact_topic_and_orders_by_timestamp(self) -> None:
        uuid = "test-uuid"
        zip_path, temp_dir = _write_zip(
            {
                f"{uuid}/meta.json": b"{}",
                f"{uuid}/da_data_meta.json": json.dumps({"topics": {}}).encode(),
                f"{uuid}/data/dds/{uuid}_perception-lidar_cluster_200.pb.dat": b"b",
                f"{uuid}/data/dds/{uuid}_perception-lidar_cluster_100.pb.dat": b"a",
                f"{uuid}/data/dds/{uuid}_perception-lidar_cluster_extra_150.pb.dat": b"c",
            }
        )
        self.addCleanup(temp_dir.cleanup)

        extractor = mod.NIODataExtractor(zip_path)
        self.addCleanup(extractor.close)

        messages = extractor.read_dds_message("/perception/lidar_cluster")

        self.assertEqual(
            messages,
            [
                (100, b"a"),
                (200, b"b"),
            ],
        )

    def test_read_dds_message_uses_metadata_alias_and_pb_dat_framing(self) -> None:
        uuid = "test-uuid"
        msg = ObjectsDetection()
        msg.timestamp = 123
        payload = msg.SerializeToString()
        framed = struct.pack("<QQQQQ", 555, 555, 666, 666, len(payload)) + payload

        zip_path, temp_dir = _write_zip(
            {
                f"{uuid}/meta.json": json.dumps(
                    {
                        "topics": {
                            "/common/perception/perception_objects": {
                                "file_name": "data/dds/perception.pb.dat"
                            }
                        }
                    }
                ).encode(),
                f"{uuid}/da_data_meta.json": json.dumps({"topics": {}}).encode(),
                f"{uuid}/data/dds/perception.pb.dat": framed,
            }
        )
        self.addCleanup(temp_dir.cleanup)

        extractor = mod.NIODataExtractor(zip_path)
        self.addCleanup(extractor.close)

        messages = extractor.read_dds_message("/perception/perception_objects")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0][0], 555)
        self.assertEqual(messages[0][1], payload)

    def test_parse_pcd_pointcloud_reads_xyz_and_intensity(self) -> None:
        points = mod._parse_pcd_pointcloud(
            _build_ascii_pcd(
                [
                    [1.0, 2.0, 3.0, 10.0],
                    [4.0, 5.0, 6.0, 20.0],
                ]
            )
        )

        self.assertIsNotNone(points)
        np.testing.assert_allclose(
            points,
            np.array(
                [
                    [1.0, 2.0, 3.0, 10.0],
                    [4.0, 5.0, 6.0, 20.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_parse_pcd_pointcloud_reads_binary_payload(self) -> None:
        expected = np.array(
            [
                [7.0, 8.0, 9.0, 30.0],
                [10.0, 11.0, 12.0, 40.0],
            ],
            dtype=np.float32,
        )

        points = mod._parse_pcd_pointcloud(_build_binary_pcd(expected))

        self.assertIsNotNone(points)
        np.testing.assert_allclose(points, expected)

    def test_read_lidar_pointcloud_sorts_pcd_by_filename_timestamp(self) -> None:
        uuid = "test-uuid"

        zip_path, temp_dir = _write_zip(
            {
                f"{uuid}/meta.json": b"{}",
                f"{uuid}/da_data_meta.json": json.dumps({"topics": {}}).encode(),
                f"{uuid}/data/lidar/1759986615000000.pcd": _build_ascii_pcd(
                    [[4.0, 5.0, 6.0, 200.0]]
                ),
                f"{uuid}/data/lidar/1759986614000000.pcd": _build_ascii_pcd(
                    [[1.0, 2.0, 3.0, 100.0]]
                ),
            }
        )
        self.addCleanup(temp_dir.cleanup)

        extractor = mod.NIODataExtractor(zip_path)
        self.addCleanup(extractor.close)

        frames = extractor.read_lidar_pointcloud(max_frames=5)

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0], 1_759_986_614_000_000_000)
        self.assertEqual(frames[1][0], 1_759_986_615_000_000_000)
        np.testing.assert_allclose(
            frames[0][1],
            np.array([[1.0, 2.0, 3.0, 100.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            frames[1][1],
            np.array([[4.0, 5.0, 6.0, 200.0]], dtype=np.float32),
        )

    def test_camera_frame_timestamp_prefers_utc_column(self) -> None:
        frame = mod.CameraFrame(
            ptp_timestamp=1759987733455459840,
            utc_timestamp=1759986613789432000,
            frame_type="I",
        )

        self.assertEqual(mod._camera_frame_timestamp_ns(frame, fallback=123), 1759986613789432000)

        frame.utc_timestamp = 0
        self.assertEqual(mod._camera_frame_timestamp_ns(frame, fallback=123), 1759987733455459840)

    def test_log_camera_frame_uses_supported_pinhole_signature(self) -> None:
        frame = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
        calib_data = {
            "calibration_info": {
                "camera_matrix": {
                    "intrinsic": {
                        "fx": 11,
                        "fy": 22,
                        "cx": 33,
                        "cy": 44,
                    }
                }
            }
        }

        with (
            mock.patch.object(mod.rr, "set_time") as mock_set_time,
            mock.patch.object(mod.rr, "log") as mock_log,
            mock.patch.object(mod.rr, "Image", side_effect=lambda image: ("Image", image.shape)),
            mock.patch.object(mod.rr, "Pinhole", return_value="Pinhole") as mock_pinhole,
        ):
            mod._log_camera_frame(
                image_path="camera_2d/front/image",
                camera_path="camera_3d/front",
                timestamp_ns=2_000_000_000,
                frame=frame,
                calib_data=calib_data,
            )

        mock_set_time.assert_called_once_with(mod.RERUN_TIMELINE, timestamp=2.0)
        mock_pinhole.assert_called_once_with(
            resolution=(2, 1),
            focal_length=(11.0, 22.0),
            principal_point=(33.0, 44.0),
        )
        self.assertEqual(mock_log.call_args_list[0].args[0], "camera_2d/front/image")
        self.assertEqual(mock_log.call_args_list[1].args[0], "camera_3d/front")
        self.assertEqual(mock_log.call_args_list[2].args[0], "camera_3d/front")

    def test_clear_stale_entity_paths_logs_clear_for_removed_entities(self) -> None:
        with (
            mock.patch.object(mod.rr, "set_time") as mock_set_time,
            mock.patch.object(mod.rr, "log") as mock_log,
            mock.patch.object(
                mod.rr,
                "Clear",
                side_effect=lambda recursive=False: ("Clear", recursive),
            ) as mock_clear,
        ):
            mod._clear_stale_entity_paths(
                previous_paths={"lidar/cluster/0/box", "lidar/cluster/1/box"},
                current_paths={"lidar/cluster/1/box"},
                timestamp_ns=3_000_000_000,
            )

        mock_set_time.assert_called_once_with(mod.RERUN_TIMELINE, timestamp=3.0)
        mock_clear.assert_called_once_with(recursive=False)
        mock_log.assert_called_once_with("lidar/cluster/0/box", ("Clear", False))

    def test_lidar_cluster_timestamp_prefers_proto_fields(self) -> None:
        clusters = LidarClusterObjects()
        clusters.timestamp = 111
        clusters.publish_ptp_ts = 222
        obj = clusters.lidar_cluster_object_list.add()
        obj.lidar_cluster_mean_timestamp = 333

        self.assertEqual(mod._lidar_cluster_timestamp_ns(clusters, fallback=444), 222)

        clusters.ClearField("publish_ptp_ts")
        clusters.ClearField("timestamp")
        self.assertEqual(mod._lidar_cluster_timestamp_ns(clusters, fallback=444), 333)


if __name__ == "__main__":
    unittest.main()

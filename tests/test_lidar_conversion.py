import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

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


def _build_falcon_packet(
    *,
    frame_idx: int,
    timestamp_us: int,
    h_angle_raw: int,
    v_angle_raw: int,
    radius_raw: int,
    reflectance: int,
) -> bytes:
    block_size = mod.FALCON_BLOCK_HEADER_SIZE + 4
    packet_size = mod.FALCON_PACKET_HEADER_SIZE + block_size

    common_header = bytearray(mod.FALCON_COMMON_HEADER_SIZE)
    common_header[0:2] = mod.FALCON_MAGIC.to_bytes(2, "little")
    common_header[10:14] = packet_size.to_bytes(4, "little")
    common_header[18:26] = timestamp_us.to_bytes(8, "little")

    packet_info = bytearray(mod.FALCON_PACKET_INFO_SIZE)
    packet_info[0:8] = frame_idx.to_bytes(8, "little")
    packed_type_and_count = (1 << 8) | 1
    packet_info[12:16] = packed_type_and_count.to_bytes(4, "little")
    packet_info[16:18] = block_size.to_bytes(2, "little")

    block_header = bytearray(mod.FALCON_BLOCK_HEADER_SIZE)
    block_header[0:2] = h_angle_raw.to_bytes(2, "little")
    block_header[2:4] = v_angle_raw.to_bytes(2, "little")

    point = (radius_raw | (reflectance << 17)).to_bytes(4, "little")
    return bytes(common_header + packet_info + block_header + point)


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

    def test_read_falcon_pointcloud_groups_packets_by_frame_index(self) -> None:
        uuid = "test-uuid"
        packet_a = _build_falcon_packet(
            frame_idx=7,
            timestamp_us=1_000_000,
            h_angle_raw=32768,
            v_angle_raw=32768,
            radius_raw=400,
            reflectance=64,
        )
        packet_b = _build_falcon_packet(
            frame_idx=7,
            timestamp_us=1_000_100,
            h_angle_raw=49152,
            v_angle_raw=32768,
            radius_raw=600,
            reflectance=128,
        )

        zip_path, temp_dir = _write_zip(
            {
                f"{uuid}/meta.json": b"{}",
                f"{uuid}/da_data_meta.json": json.dumps({"topics": {}}).encode(),
                f"{uuid}/data/lidar/sample.dat": packet_a + packet_b,
            }
        )
        self.addCleanup(temp_dir.cleanup)

        extractor = mod.NIODataExtractor(zip_path)
        self.addCleanup(extractor.close)

        frames = extractor.read_falcon_pointcloud(max_frames=5)

        self.assertEqual(len(frames), 1)
        timestamp_ns, points = frames[0]
        self.assertEqual(timestamp_ns, 1_000_000_000)
        self.assertEqual(points.shape, (2, 4))

        self.assertAlmostEqual(points[0, 0], 2.0, places=5)
        self.assertAlmostEqual(points[0, 1], 0.0, places=5)
        self.assertAlmostEqual(points[0, 2], 0.0, places=5)
        self.assertAlmostEqual(points[0, 3], 64 / 255.0, places=5)

        self.assertAlmostEqual(points[1, 0], 0.0, places=5)
        self.assertAlmostEqual(points[1, 1], 3.0, places=5)
        self.assertAlmostEqual(points[1, 2], 0.0, places=5)
        self.assertAlmostEqual(points[1, 3], 128 / 255.0, places=5)

    def test_read_falcon_pointcloud_uses_sdk_reader_when_available(self) -> None:
        uuid = "test-uuid"

        class FakeFalconReader:
            def __init__(self, path: str) -> None:
                self.path = path
                self._frames = [
                    (
                        [
                            [1.0, 2.0, 3.0, 10.0],
                            [4.0, 5.0, 6.0, 20.0],
                        ],
                        {
                            "frame_starttime": 1.25,
                            "scanner_direction": 2,
                        },
                    )
                ]
                self.fillrow_mode = False

            def set_fillrow_mode(self) -> None:
                self.fillrow_mode = True

            def get_frame_with_info(self):
                if self._frames:
                    return self._frames.pop(0)
                return None

        def fake_get_falcon_data_13n(points):
            return [[0.0] * 13 for _ in points]

        zip_path, temp_dir = _write_zip(
            {
                f"{uuid}/meta.json": b"{}",
                f"{uuid}/da_data_meta.json": json.dumps({"topics": {}}).encode(),
                f"{uuid}/data/lidar/sample.dat": b"not-a-raw-falcon-packet",
            }
        )
        self.addCleanup(temp_dir.cleanup)

        extractor = mod.NIODataExtractor(zip_path)
        self.addCleanup(extractor.close)

        with mock.patch.object(
            mod,
            "_load_falcon_sdk",
            return_value=(FakeFalconReader, fake_get_falcon_data_13n, None),
        ):
            frames = extractor.read_falcon_pointcloud(max_frames=5)

        self.assertEqual(len(frames), 1)
        timestamp_ns, points = frames[0]
        self.assertEqual(timestamp_ns, 1_250_000_000)
        self.assertEqual(points.tolist(), [[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]])

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

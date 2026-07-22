from __future__ import annotations

import struct
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.stereo_runtime import (
    MESH_HEADER,
    StereoBrowserRuntime,
    build_mesh_packet,
    build_near_blue_texture,
)


class StereoWebTests(unittest.TestCase):
    def test_mesh_packet_contains_webgl_positions_uv_and_indices(self) -> None:
        vertices = np.asarray(
            [
                [[-1, -1, 10], [1, -1, 11]],
                [[-1, 1, 12], [1, 1, 13]],
            ],
            dtype=np.float32,
        )
        uv = np.asarray(
            [[[0, 1], [1, 1]], [[0, 0], [1, 0]]], dtype=np.float32
        )
        packet = build_mesh_packet(
            7, vertices, uv, np.ones((1, 1), bool), np.ones((2, 2), bool), (640, 480)
        )
        magic, version, sequence, vertex_count, index_count, width, height, _ = MESH_HEADER.unpack_from(packet)
        self.assertEqual(magic, b"ST3D")
        self.assertEqual((version, sequence), (1, 7))
        self.assertEqual((vertex_count, index_count), (4, 6))
        self.assertEqual((width, height), (640, 480))
        self.assertEqual(len(packet), MESH_HEADER.size + 4 * 3 * 4 + 4 * 2 * 4 + 6 * 4)

    def test_disabled_runtime_does_not_start_camera_thread(self) -> None:
        runtime = StereoBrowserRuntime({"stereo_enabled": False})
        runtime.start()
        self.assertFalse(runtime.status()["enabled"])
        self.assertIsNone(runtime._thread)

    def test_depth_palette_is_blue_near_and_red_far(self) -> None:
        distance = np.asarray([[100.0, 1000.0, 0.0]], dtype=np.float32)
        valid = np.asarray([[True, True, False]])
        texture = build_near_blue_texture(distance, valid, 100.0, 1000.0)
        near_b, _near_g, near_r = texture[0, 0]
        far_b, _far_g, far_r = texture[0, 1]
        self.assertGreater(int(near_b), int(near_r))
        self.assertGreater(int(far_r), int(far_b))
        np.testing.assert_array_equal(texture[0, 2], (0, 0, 0))

    def test_all_four_browser_modes_are_selectable(self) -> None:
        runtime = StereoBrowserRuntime({"stereo_enabled": False})
        for mode in ("3d_aruco", "3d_full", "rgbd", "superfast"):
            self.assertEqual(runtime.set_mode(mode)["mode"], mode)
            self.assertEqual(runtime.status()["mode"], mode)
        with self.assertRaises(ValueError):
            runtime.set_mode("unknown")

    def test_saved_camera_swap_reassigns_left_and_right_indices(self) -> None:
        with patch.object(Path, "read_text", return_value='{"swap_cameras": true}'):
            runtime = StereoBrowserRuntime({
                "stereo_enabled": False,
                "stereo_left_camera_index": 0,
                "stereo_right_camera_index": 1,
            })
        status = runtime.status()
        self.assertTrue(status["cameras_swapped"])
        self.assertEqual((status["left_camera"], status["right_camera"]), (1, 0))

    def test_original_controls_are_validated_and_persisted(self) -> None:
        settings_path = Path("test-stereo-settings.json")
        with (
            patch("backend.stereo_runtime.SETTINGS_PATH", settings_path),
            patch.object(Path, "write_text") as write_text,
            patch.object(Path, "replace") as replace,
        ):
            runtime = StereoBrowserRuntime({"stereo_enabled": False})
            settings = runtime.update_settings({
                "grid_size": 210,
                "median_kernel": 4,
                "z_scale": 3.2,
                "white_points": True,
                "swap_cameras": True,
            })
            self.assertEqual(settings["grid_size"], 210)
            self.assertEqual(settings["median_kernel"], 5)
            self.assertEqual(settings["z_scale"], 3.2)
            self.assertTrue(settings["white_points"])
            self.assertTrue(settings["swap_cameras"])
            write_text.assert_called_once()
            replace.assert_called_once_with(settings_path)


if __name__ == "__main__":
    unittest.main()

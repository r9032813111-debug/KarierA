"""Headless browser bridge for the original V10 RGBD implementation.

The calibrated stereo/depth algorithms remain in ``stereo_source``.  This
module exposes four browser modes without opening OpenCV or PyVista windows:
cropped 3D, full-frame 3D, a regular 2D RGBD view and a deliberately cheap
SUPERFAST 2D view.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "stereo_source"
CALIBRATION = ROOT / "stereo_calibration.yaml"
SETTINGS_PATH = ROOT / "stereo_settings.json"
MESH_HEADER = struct.Struct("<4s7I")
STEREO_MODES = {"3d_aruco", "3d_full", "rgbd", "superfast"}


def _load_original_modules() -> tuple[Any, Any, Any, Any]:
    """Import the copied V10 implementation without opening any GUI window."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    source_text = str(SOURCE)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    import config_v2  # type: ignore
    import rgbd_pyvista_v2 as v2  # type: ignore
    import rgbd_pyvista_v10_aruco_crop as v10  # type: ignore
    import stereo_depth_v2 as depth_v2  # type: ignore

    return config_v2.CONFIG, depth_v2, v2, v10


def build_mesh_packet(
    sequence: int,
    vertices: np.ndarray,
    texture_coordinates: np.ndarray,
    cell_mask: np.ndarray,
    vertex_valid: np.ndarray,
    texture_size: tuple[int, int],
) -> bytes:
    """Encode positions, UV coordinates and triangle indices for WebGL2."""
    points = np.asarray(vertices, dtype=np.float32).copy()
    valid = np.asarray(vertex_valid, dtype=bool)
    if np.any(valid) and not np.all(valid):
        points[:, :, 2][~valid] = float(np.median(points[:, :, 2][valid]))

    rows, columns = points.shape[:2]
    point_ids = np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns)
    cells = np.asarray(cell_mask, dtype=bool)
    top_left = point_ids[:-1, :-1][cells]
    top_right = point_ids[:-1, 1:][cells]
    bottom_right = point_ids[1:, 1:][cells]
    bottom_left = point_ids[1:, :-1][cells]
    indices = np.column_stack(
        (top_left, top_right, bottom_right, top_left, bottom_right, bottom_left)
    ).astype("<u4", copy=False).reshape(-1)
    positions = np.ascontiguousarray(points.reshape(-1, 3), dtype="<f4")
    uv = np.ascontiguousarray(texture_coordinates.reshape(-1, 2), dtype="<f4")
    width, height = texture_size
    header = MESH_HEADER.pack(
        b"ST3D", 1, int(sequence), len(positions), len(indices), int(width), int(height), 0
    )
    return header + positions.tobytes() + uv.tobytes() + indices.tobytes()


def build_near_blue_texture(
    distance_mm: np.ndarray,
    valid: np.ndarray,
    minimum_mm: float,
    maximum_mm: float,
    contour_step_mm: float = 0.0,
) -> np.ndarray:
    """Color depth monotonically: near is blue, far is red."""
    valid = np.asarray(valid, dtype=bool)
    normalized = np.zeros(distance_mm.shape, dtype=np.uint8)
    span = max(1.0, float(maximum_mm) - float(minimum_mm))
    normalized[valid] = np.clip(
        (distance_mm[valid] - float(minimum_mm)) * 255.0 / span,
        0,
        255,
    ).astype(np.uint8)
    topology = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    topology[~valid] = (0, 0, 0)
    if contour_step_mm <= 0.0 or not np.any(valid):
        return topology

    safe_depth = np.where(valid, distance_mm, minimum_mm).astype(np.float32)
    level = np.floor((safe_depth - minimum_mm) / contour_step_mm).astype(np.int32)
    contour = np.zeros(valid.shape, dtype=bool)
    contour[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (level[:, 1:] != level[:, :-1])
    contour[1:, :] |= valid[1:, :] & valid[:-1, :] & (level[1:, :] != level[:-1, :])
    if np.any(contour):
        contour = cv2.dilate(contour.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1)
        topology[contour != 0] = (0, 0, 0)
    return topology


SETTING_RULES: dict[str, tuple[str, float | None, float | None]] = {
    "grid_size": ("int", 10, 300),
    "z_scale": ("float", 0.1, 10.0),
    "plane_width_mm": ("float", 100.0, 10000.0),
    "min_distance_mm": ("float", 0.0, 20000.0),
    "max_distance_mm": ("float", 10.0, 20000.0),
    "z_smoothing": ("float", 0.0, 5.0),
    "filters_enabled": ("bool", None, None),
    "gradient_fill": ("bool", None, None),
    "inpaint_radius": ("float", 0.0, 10.0),
    "median_filter": ("bool", None, None),
    "median_kernel": ("odd", 3, 15),
    "bilateral_filter": ("bool", None, None),
    "bilateral_diameter": ("odd", 1, 25),
    "bilateral_sigma_color": ("float", 0.0, 100.0),
    "bilateral_sigma_space": ("float", 0.0, 100.0),
    "speckle_filter": ("bool", None, None),
    "speckle_area_px": ("int", 0, 2000),
    "morphological_fill": ("bool", None, None),
    "morphological_kernel": ("odd", 1, 15),
    "contour_fill": ("bool", None, None),
    "max_hole_area_px": ("int", 0, 200000),
    "median_area_px": ("int", 3, 400),
    "outlier_cm": ("float", 0.1, 200.0),
    "topology_contour_cm": ("float", 0.0, 200.0),
    "white_points": ("bool", None, None),
    "swap_cameras": ("bool", None, None),
    "processing_scale": ("float", 0.25, 1.0),
    "jpeg_quality": ("int", 40, 95),
    "superfast_width": ("int", 160, 960),
    "superfast_disparities": ("multiple16", 16, 160),
    "superfast_block_size": ("odd", 5, 31),
}


def _coerce_setting(name: str, value: Any) -> Any:
    kind, minimum, maximum = SETTING_RULES[name]
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    number = max(float(minimum), min(float(maximum), number))
    if kind in {"int", "odd", "multiple16"}:
        result = int(round(number))
        if kind == "odd" and result % 2 == 0:
            result = min(int(maximum), result + 1)
        if kind == "multiple16":
            result = max(16, (result // 16) * 16)
        return result
    return number


class StereoBrowserRuntime:
    """Capture a calibrated stereo pair and publish browser-ready frames."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("stereo_enabled", True))
        self.left_index = config.get("stereo_left_camera_index")
        self.right_index = config.get("stereo_right_camera_index")
        self.scan_limit = max(2, int(config.get("stereo_camera_scan_limit", 8)))
        self.backend = str(config.get("stereo_camera_backend", "dshow"))
        self.width = int(config.get("stereo_frame_width", 1920))
        self.height = int(config.get("stereo_frame_height", 1080))
        defaults = {
            "grid_size": config.get("stereo_grid_size", 150),
            "z_scale": config.get("stereo_z_scale", 2.1),
            "plane_width_mm": config.get("stereo_plane_width_mm", 2000.0),
            "min_distance_mm": config.get("stereo_min_distance_mm", 20.0),
            "max_distance_mm": config.get("stereo_max_distance_mm", 6670.0),
            "z_smoothing": config.get("stereo_z_smoothing", 1.0),
            "filters_enabled": False,
            "gradient_fill": True,
            "inpaint_radius": 5.0,
            "median_filter": True,
            "median_kernel": 3,
            "bilateral_filter": False,
            "bilateral_diameter": 7,
            "bilateral_sigma_color": 18.0,
            "bilateral_sigma_space": 9.0,
            "speckle_filter": True,
            "speckle_area_px": 120,
            "morphological_fill": True,
            "morphological_kernel": 3,
            "contour_fill": False,
            "max_hole_area_px": 40000,
            "median_area_px": 84,
            "outlier_cm": 59.0,
            "topology_contour_cm": 0.0,
            "white_points": False,
            "swap_cameras": False,
            "processing_scale": 1.0,
            "jpeg_quality": config.get("stereo_jpeg_quality", 78),
            "superfast_width": 480,
            "superfast_disparities": 64,
            "superfast_block_size": 9,
        }
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                defaults.update({key: value for key, value in saved.items() if key in SETTING_RULES})
        except (OSError, ValueError):
            pass
        self._settings = {key: _coerce_setting(key, value) for key, value in defaults.items()}
        self.matcher = str(config.get("stereo_matcher", "sgbm"))
        self.min_disparity = int(config.get("stereo_min_disparity", 2))
        self.num_disparities = max(16, (int(config.get("stereo_num_disparities", 192)) // 16) * 16)
        self.block_size = int(config.get("stereo_block_size", 15)) | 1

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._left_frame: np.ndarray | None = None
        self._left_sequence = 0
        self._mesh_packets: dict[str, bytes | None] = {"aruco": None, "full": None}
        self._rgb_jpeg: bytes | None = None
        self._topology_jpeg: bytes | None = None
        self._frame_jpeg: bytes | None = None
        self._sequence = 0
        self._mode = "3d_aruco"
        self._published_mode = ""
        self._settings_revision = 0
        self._active_swap = bool(self._settings["swap_cameras"])
        self._online = False
        self._roi_live = False
        self._fps = 0.0
        self._error = "Stereo runtime has not started"
        self._last_frame_at = 0.0

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="stereo-browser", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    def latest_left_frame(self) -> tuple[np.ndarray, int] | None:
        """Frame provider used by the existing ArUco/robot pipeline."""
        with self._lock:
            if self._left_frame is None:
                return None
            return self._left_frame.copy(), self._left_sequence

    def status(self) -> dict[str, Any]:
        with self._lock:
            ready = self._published_mode == self._mode
            return {
                "enabled": self.enabled,
                "online": self._online,
                "sequence": self._sequence,
                "fps": round(self._fps, 1),
                "roi_live": self._roi_live,
                "left_camera": self.right_index if self._active_swap else self.left_index,
                "right_camera": self.left_index if self._active_swap else self.right_index,
                "cameras_swapped": self._active_swap,
                "grid": self._settings["grid_size"],
                "mode": self._mode,
                "published_mode": self._published_mode,
                "error": self._error,
                "has_mesh": ready and self._mode.startswith("3d_") and self._mesh_packets["full"] is not None,
                "has_frame": ready and not self._mode.startswith("3d_") and self._frame_jpeg is not None,
            }

    def settings(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    def update_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(changes) - set(SETTING_RULES))
        if unknown:
            raise ValueError(f"Unknown stereo settings: {', '.join(unknown)}")
        normalized = {key: _coerce_setting(key, value) for key, value in changes.items()}
        with self._lock:
            candidate = {**self._settings, **normalized}
            if candidate["max_distance_mm"] <= candidate["min_distance_mm"]:
                raise ValueError("Maximum distance must be greater than minimum distance")
            self._settings = candidate
            self._settings_revision += 1
            self._published_mode = ""
            snapshot = dict(candidate)
        temporary = SETTINGS_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(SETTINGS_PATH)
        return snapshot

    def set_mode(self, mode: str) -> dict[str, Any]:
        if mode not in STEREO_MODES:
            raise ValueError(f"Unknown stereo mode: {mode}")
        with self._lock:
            if self._mode != mode:
                self._mode = mode
                self._published_mode = ""
                self._error = "Switching stereo mode"
            return {"ok": True, "mode": self._mode}

    def mesh_packet(self, crop: str) -> bytes | None:
        with self._lock:
            return self._mesh_packets["full" if crop == "full" else "aruco"]

    def texture_jpeg(self, mode: str) -> bytes | None:
        with self._lock:
            return self._topology_jpeg if mode == "topology" else self._rgb_jpeg

    def frame_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg

    def _publish_left(self, frame: np.ndarray) -> None:
        with self._lock:
            self._left_frame = frame.copy()
            self._left_sequence += 1

    def _job_state(self) -> tuple[str, int, dict[str, Any]]:
        with self._lock:
            return self._mode, self._settings_revision, dict(self._settings)

    def _publish_result(self, result: dict[str, Any]) -> None:
        now = time.monotonic()
        with self._lock:
            if result["mode"] != self._mode or result["revision"] != self._settings_revision:
                return
            self._sequence += 1
            sequence = self._sequence
            if result["mode"].startswith("3d_"):
                common = (
                    result["vertices"], result["uv"], result["vertex_valid"], result["texture_size"]
                )
                self._mesh_packets["aruco"] = build_mesh_packet(
                    sequence, common[0], common[1], result["aruco_mask"], common[2], common[3]
                )
                self._mesh_packets["full"] = build_mesh_packet(
                    sequence, common[0], common[1], result["full_mask"], common[2], common[3]
                )
                self._rgb_jpeg = result["rgb_jpeg"]
                self._topology_jpeg = result["topology_jpeg"]
            else:
                self._frame_jpeg = result["frame_jpeg"]
            self._published_mode = result["mode"]
            self._roi_live = bool(result.get("roi_live", False))
            self._fps = 1.0 / max(now - self._last_frame_at, 1e-6) if self._last_frame_at else 0.0
            self._last_frame_at = now
            self._online = True
            self._error = ""

    @staticmethod
    def _filters(depth_v2: Any, settings: dict[str, Any]) -> Any:
        return depth_v2.DepthFilterSettings(
            settings["filters_enabled"], settings["speckle_filter"], settings["speckle_area_px"],
            2.0, settings["morphological_fill"], settings["morphological_kernel"],
            settings["contour_fill"], settings["gradient_fill"], settings["max_hole_area_px"],
            settings["inpaint_radius"], settings["median_filter"], settings["median_kernel"],
            0.70, settings["bilateral_filter"], settings["bilateral_diameter"],
            settings["bilateral_sigma_color"], settings["bilateral_sigma_space"],
        )

    @staticmethod
    def _scaled_pair(left: np.ndarray, right: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
        if scale >= 0.999:
            return left, right
        size = (max(32, int(left.shape[1] * scale)), max(24, int(left.shape[0] * scale)))
        return cv2.resize(left, size, interpolation=cv2.INTER_AREA), cv2.resize(right, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _encode(image: np.ndarray, quality: int) -> bytes:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("Could not encode stereo image")
        return encoded.tobytes()

    @staticmethod
    def _fit_pair(rgb: np.ndarray, depth_image: np.ndarray, width: int = 1280) -> np.ndarray:
        half = max(160, width // 2)
        height = max(90, int(rgb.shape[0] * half / max(1, rgb.shape[1])))
        rgb_small = cv2.resize(rgb, (half, height), interpolation=cv2.INTER_AREA)
        depth_small = cv2.resize(depth_image, (half, height), interpolation=cv2.INTER_AREA)
        cv2.putText(rgb_small, "RGB", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_small, "DEPTH", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)
        return np.hstack((rgb_small, depth_small))

    def _build_3d(
        self, depth_v2: Any, depth: Any, temporal: Any, tracker: Any, v2: Any, v10: Any,
        left: np.ndarray, right: np.ndarray, mode: str, revision: int, settings: dict[str, Any],
    ) -> dict[str, Any]:
        left, right = self._scaled_pair(left, right, settings["processing_scale"])
        filters = self._filters(depth_v2, settings)
        rectified, gray, distance_mm, measured_valid = v10.compute_measured_frame(
            depth, left, right, filters, None, settings["min_distance_mm"], settings["max_distance_mm"]
        )
        completed, surface_valid, _confidence = v10.complete_low_texture_surfaces(
            rectified, gray, distance_mm, measured_valid,
            settings["min_distance_mm"], settings["max_distance_mm"], temporal,
        )
        vertices, uv, vertex_valid, _moved, _source = v10.relocate_grid_vertices(
            completed, surface_valid, settings["grid_size"], settings["z_scale"], settings["plane_width_mm"]
        )
        vertices = v10.suppress_sector_height_outliers(
            vertices, vertex_valid, settings["median_area_px"], settings["outlier_cm"] * 10.0
        )
        vertices = v2.smooth_mesh_z(vertices, settings["z_smoothing"])
        topology = build_near_blue_texture(
            completed, surface_valid, settings["min_distance_mm"], settings["max_distance_mm"],
            settings["topology_contour_cm"] * 10.0,
        )
        quad = tracker.update(rectified)
        full_mask = np.ones((vertices.shape[0] - 1, vertices.shape[1] - 1), dtype=bool)
        aruco_mask = full_mask if quad is None else v10.aruco_cell_mask(distance_mm.shape, vertex_valid.shape, quad)
        if not np.any(vertex_valid) or not np.any(full_mask):
            raise RuntimeError("No valid stereo surface")
        return {
            "mode": mode, "revision": revision, "vertices": vertices, "uv": uv,
            "vertex_valid": vertex_valid, "aruco_mask": aruco_mask, "full_mask": full_mask,
            "texture_size": (rectified.shape[1], rectified.shape[0]),
            "rgb_jpeg": self._encode(rectified, settings["jpeg_quality"]),
            "topology_jpeg": self._encode(topology, settings["jpeg_quality"]),
            "roi_live": tracker.roi_live,
        }

    def _build_rgbd(
        self, depth_v2: Any, depth: Any, temporal: Any, v10: Any,
        left: np.ndarray, right: np.ndarray, mode: str, revision: int, settings: dict[str, Any],
    ) -> dict[str, Any]:
        left, right = self._scaled_pair(left, right, settings["processing_scale"])
        filters = self._filters(depth_v2, settings)
        rectified, gray, distance_mm, measured_valid = v10.compute_measured_frame(
            depth, left, right, filters, None, settings["min_distance_mm"], settings["max_distance_mm"]
        )
        completed, surface_valid, confidence = v10.complete_low_texture_surfaces(
            rectified, gray, distance_mm, measured_valid,
            settings["min_distance_mm"], settings["max_distance_mm"], temporal,
        )
        depth_view = build_near_blue_texture(
            completed, surface_valid, settings["min_distance_mm"], settings["max_distance_mm"],
            settings["topology_contour_cm"] * 10.0,
        )
        v10.draw_completion_overlay(depth_view, confidence)
        if settings["white_points"]:
            _vertices, _uv, vertex_valid, moved, source_points = v10.relocate_grid_vertices(
                completed, surface_valid, settings["grid_size"], settings["z_scale"], settings["plane_width_mm"]
            )
            v10.draw_v5_vertex_overlay(depth_view, source_points, vertex_valid, moved, True)
        frame = self._fit_pair(rectified, depth_view)
        return {"mode": mode, "revision": revision, "frame_jpeg": self._encode(frame, settings["jpeg_quality"])}

    def _build_superfast(
        self, depth_v2: Any, depth: Any, v10: Any,
        left: np.ndarray, right: np.ndarray, mode: str, revision: int, settings: dict[str, Any],
    ) -> dict[str, Any]:
        target_width = min(settings["superfast_width"], left.shape[1])
        scale = target_width / max(1, left.shape[1])
        left, right = self._scaled_pair(left, right, scale)
        disabled = depth_v2.DepthFilterSettings(
            False, False, 0, 0.0, False, 1, False, False, 0, 0.0,
            False, 3, 0.0, False, 1, 0.0, 0.0,
        )
        rectified, _right, _disparity, distance_mm, valid = depth.compute(left, right, disabled)
        valid &= (distance_mm >= settings["min_distance_mm"]) & (distance_mm <= settings["max_distance_mm"])
        depth_view = build_near_blue_texture(
            distance_mm, valid, settings["min_distance_mm"], settings["max_distance_mm"], 0.0
        )
        frame = self._fit_pair(rectified, depth_view, width=960)
        return {"mode": mode, "revision": revision, "frame_jpeg": self._encode(frame, 55)}

    def _run(self) -> None:
        while not self._stop.is_set():
            left_camera = right_camera = None
            try:
                config, depth_v2, v2, v10 = _load_original_modules()
                cv2.setUseOptimized(True)
                cv2.setNumThreads(config.CPU_THREADS)
                calibration = depth_v2.load_stereo_calibration(CALIBRATION)
                left_hint = int(self.left_index) if self.left_index is not None else calibration.left_camera_index
                right_hint = int(self.right_index) if self.right_index is not None else calibration.right_camera_index
                left_index, right_index = depth_v2.find_camera_indices(left_hint, right_hint, self.backend, self.scan_limit)
                self.left_index, self.right_index = left_index, right_index
                left_camera = depth_v2.Camera(left_index, self.width, self.height, self.backend, "none")
                right_camera = depth_v2.Camera(right_index, self.width, self.height, self.backend, "none")
                left_camera.open()
                right_camera.open()
                if not depth_v2.warmup_camera_pair(left_camera, right_camera):
                    raise RuntimeError("Both stereo cameras opened but frames are unavailable")

                normal_depth = depth_v2.StereoDepth(
                    calibration, self.matcher, self.min_disparity, self.num_disparities, self.block_size
                )
                normal_signature = None
                fast_depth = None
                fast_signature = None
                temporal = v10.TemporalDepthCompleter(v10.SURFACE_FILL_TEMPORAL_FRAMES)
                tracker = v10.ArucoRoiTracker()
                _initial_mode, _initial_revision, initial_settings = self._job_state()
                active_swap = bool(initial_settings["swap_cameras"])
                with self._lock:
                    self._active_swap = active_swap
                future: Future[dict[str, Any]] | None = None
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stereo-browser-depth") as executor:
                    while not self._stop.is_set():
                        ok, left, right = depth_v2.read_camera_pair(left_camera, right_camera)
                        if not ok or left is None or right is None:
                            raise RuntimeError("Stereo camera frame was lost")
                        if future is not None and future.done():
                            self._publish_result(future.result())
                            future = None
                        if future is None:
                            mode, revision, settings = self._job_state()
                            requested_swap = bool(settings["swap_cameras"])
                            if requested_swap != active_swap:
                                # This mirrors the original C-key handler: apply
                                # only between depth jobs, then forget old ROI and
                                # temporal history because camera roles changed.
                                active_swap = requested_swap
                                tracker.reset()
                                temporal = v10.TemporalDepthCompleter(v10.SURFACE_FILL_TEMPORAL_FRAMES)
                                with self._lock:
                                    self._active_swap = active_swap
                        if active_swap:
                            left, right = right, left
                        self._publish_left(left)
                        if future is not None:
                            continue
                        if mode == "superfast":
                            signature = (settings["superfast_disparities"], settings["superfast_block_size"])
                            if fast_depth is None or fast_signature != signature:
                                fast_depth = depth_v2.StereoDepth(
                                    calibration, "bm", 0, signature[0], signature[1]
                                )
                                fast_signature = signature
                            future = executor.submit(
                                self._build_superfast, depth_v2, fast_depth, v10,
                                left.copy(), right.copy(), mode, revision, settings,
                            )
                        else:
                            signature = (self.matcher, self.min_disparity, self.num_disparities, self.block_size)
                            if normal_signature != signature:
                                normal_depth = depth_v2.StereoDepth(calibration, *signature)
                                normal_signature = signature
                            builder = self._build_rgbd if mode == "rgbd" else self._build_3d
                            arguments = (depth_v2, normal_depth, temporal)
                            if mode == "rgbd":
                                future = executor.submit(
                                    builder, *arguments, v10, left.copy(), right.copy(), mode, revision, settings
                                )
                            else:
                                future = executor.submit(
                                    builder, *arguments, tracker, v2, v10,
                                    left.copy(), right.copy(), mode, revision, settings,
                                )
            except Exception as error:
                with self._lock:
                    self._online = False
                    self._error = str(error)
                self._stop.wait(2.0)
            finally:
                if left_camera is not None:
                    left_camera.release()
                if right_camera is not None:
                    right_camera.release()

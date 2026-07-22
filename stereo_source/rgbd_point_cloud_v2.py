"""RGB-textured stereo point cloud built from the validated V2 depth pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from calibration_utils_v2 import can_start_3d
from config_v2 import CONFIG
import stereo_depth_v2 as depth_v2


WINDOW_NAME = "RGBD Point Cloud v2"
CONTROLS_NAME = "RGBD Point Cloud Controls v2"


@dataclass
class ViewState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    zoom: float = 1.2


class PointCloudControls:
    def __init__(self) -> None:
        cv2.namedWindow(CONTROLS_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROLS_NAME, 430, 350)
        self._add("Point grid", CONFIG.RGBD_GRID_SIZE, 160)
        self._add("Z scale x10", int(round(CONFIG.RGBD_Z_SCALE * 10.0)), 100)
        self._add("Plane width cm", int(round(CONFIG.RGBD_PLANE_WIDTH_MM / 10.0)), 1000)
        self._add("Min distance cm", int(round(CONFIG.RGBD_MIN_DISTANCE_MM / 10.0)), 2000)
        self._add("Max distance cm", int(round(CONFIG.RGBD_MAX_DISTANCE_MM / 10.0)), 2000)
        self._add("Point size", max(1, int(round(CONFIG.RGBD_POINT_SIZE))), 8)
        self._add("All filters", int(CONFIG.DEPTH_ENABLE_ALL_FILTERS), 1)

    def _add(self, name: str, value: int, maximum: int) -> None:
        cv2.createTrackbar(name, CONTROLS_NAME, min(max(0, value), maximum), maximum, lambda _value: None)

    def values(self) -> Tuple[int, float, float, float, float, int, bool]:
        grid = max(5, cv2.getTrackbarPos("Point grid", CONTROLS_NAME))
        z_scale = max(0.1, cv2.getTrackbarPos("Z scale x10", CONTROLS_NAME) / 10.0)
        plane_width = max(100.0, cv2.getTrackbarPos("Plane width cm", CONTROLS_NAME) * 10.0)
        min_distance = cv2.getTrackbarPos("Min distance cm", CONTROLS_NAME) * 10.0
        max_distance = cv2.getTrackbarPos("Max distance cm", CONTROLS_NAME) * 10.0
        if max_distance <= min_distance:
            max_distance = min_distance + 10.0
        point_size = max(1, cv2.getTrackbarPos("Point size", CONTROLS_NAME))
        filters_enabled = bool(cv2.getTrackbarPos("All filters", CONTROLS_NAME))
        return grid, z_scale, plane_width, min_distance, max_distance, point_size, filters_enabled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an RGB-textured stereo point cloud.")
    parser.add_argument("--stereo-calibration", default="stereo_calibration.yaml")
    parser.add_argument("--left-camera", type=int, default=None)
    parser.add_argument("--right-camera", type=int, default=None)
    parser.add_argument("--scan-cameras", type=int, default=8)
    parser.add_argument("--backend", choices=depth_v2.BACKENDS.keys(), default="dshow")
    parser.add_argument("--width", type=int, default=CONFIG.FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=CONFIG.FRAME_HEIGHT)
    parser.add_argument("--matcher", choices=("bm", "sgbm"), default=CONFIG.DEPTH_MATCHER_TYPE)
    parser.add_argument("--min-disparity", type=int, default=2)
    parser.add_argument("--num-disparities", type=int, default=CONFIG.DEPTH_NUM_DISPARITIES)
    parser.add_argument("--block-size", type=int, default=CONFIG.DEPTH_BLOCK_SIZE)
    return parser.parse_args()


def rotation_matrix(view: ViewState) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad((view.yaw_deg, view.pitch_deg, view.roll_deg))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float32)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)), dtype=np.float32)
    rz = np.array(((cr, -sr, 0.0), (sr, cr, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)
    return rz @ rx @ ry


def depth_mesh(
    rectified_bgr: np.ndarray,
    distance_mm: np.ndarray,
    valid: np.ndarray,
    grid_size: int,
    z_scale: float,
    plane_width_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an image-plane grid and displace only its Z values by depth."""
    height, width = distance_mm.shape
    grid_size = min(grid_size, height, width)
    ys = np.linspace(0, height - 1, grid_size, dtype=np.int32)
    xs = np.linspace(0, width - 1, grid_size, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)

    plane_height_mm = plane_width_mm * height / width
    x = (xx.astype(np.float32) / max(1, width - 1) - 0.5) * plane_width_mm
    y_normalized = yy.astype(np.float32) / max(1, height - 1)
    if CONFIG.RGBD_FLIP_VERTICAL:
        y_normalized = 1.0 - y_normalized
    y = (y_normalized - 0.5) * plane_height_mm
    z = distance_mm[yy, xx].astype(np.float32) * z_scale
    if CONFIG.RGBD_INVERT_DEPTH:
        z = -z
    vertices = np.dstack((x, y, z)).astype(np.float32)
    vertex_valid = valid[yy, xx] & np.isfinite(vertices).all(axis=2)
    return vertices, rectified_bgr[yy, xx], vertex_valid


def render_mesh(
    vertices: np.ndarray,
    colours: np.ndarray,
    vertex_valid: np.ndarray,
    view: ViewState,
    point_size: int,
) -> np.ndarray:
    canvas = np.full((720, 1280, 3), 238, dtype=np.uint8)
    if not np.any(vertex_valid):
        cv2.putText(canvas, "No valid depth points", (430, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        return canvas

    points = vertices[vertex_valid]
    center = np.median(points, axis=0)
    centred = vertices - center
    extent = float(np.percentile(np.linalg.norm(centred[vertex_valid], axis=1), 95))
    rotated = (centred / max(extent, 1.0)) @ rotation_matrix(view).T
    camera_z = rotated[:, :, 2] + 3.0 / view.zoom
    visible = vertex_valid & (camera_z > 0.05)
    focal = 760.0
    x = np.rint(rotated[:, :, 0] * focal / camera_z + canvas.shape[1] * 0.5).astype(np.int32)
    y = np.rint(rotated[:, :, 1] * focal / camera_z + canvas.shape[0] * 0.5).astype(np.int32)
    faces = []
    for row in range(vertices.shape[0] - 1):
        for column in range(vertices.shape[1] - 1):
            corners = np.array(((row, column), (row, column + 1), (row + 1, column + 1), (row + 1, column)))
            if np.all(visible[corners[:, 0], corners[:, 1]]):
                polygon = np.column_stack((x[corners[:, 0], corners[:, 1]], y[corners[:, 0], corners[:, 1]])).astype(np.int32)
                colour = tuple(int(value) for value in np.mean(colours[corners[:, 0], corners[:, 1]], axis=0))
                faces.append((float(np.mean(camera_z[corners[:, 0], corners[:, 1]])), polygon, colour))
    for _depth, polygon, colour in sorted(faces, key=lambda face: face[0], reverse=True):
        cv2.fillConvexPoly(canvas, polygon, colour, cv2.LINE_AA)
    visible_rows, visible_columns = np.where(visible)
    point_order = np.argsort(camera_z[visible_rows, visible_columns])[::-1]
    for index in point_order:
        row, column = visible_rows[index], visible_columns[index]
        cv2.circle(canvas, (int(x[row, column]), int(y[row, column])), point_size, tuple(int(value) for value in colours[row, column]), -1, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    allowed, reason = can_start_3d(True)
    if not allowed:
        print(f"Error: {reason}")
        return 2
    calibration = depth_v2.load_stereo_calibration(args.stereo_calibration)
    left_hint = args.left_camera if args.left_camera is not None else calibration.left_camera_index
    right_hint = args.right_camera if args.right_camera is not None else calibration.right_camera_index
    left_index, right_index = depth_v2.find_camera_indices(left_hint, right_hint, args.backend, args.scan_cameras)
    left_camera = depth_v2.Camera(left_index, args.width, args.height, args.backend, "none")
    right_camera = depth_v2.Camera(right_index, args.width, args.height, args.backend, "none")
    block_size = args.block_size if args.block_size % 2 else args.block_size + 1
    depth, cuda_matcher, parallel_bm_matcher = depth_v2.build_depth_pipeline(calibration, args, block_size)
    left_camera.open()
    right_camera.open()
    if not depth_v2.warmup_camera_pair(left_camera, right_camera):
        left_camera.release()
        right_camera.release()
        raise RuntimeError("Both cameras opened but cannot deliver frames together")

    print(f"RGBD cloud: cameras L{left_index} R{right_index}, baseline {calibration.baseline_mm:.2f} mm")
    print("Controls: W/S pitch, A/D yaw, Z/X roll, +/- zoom, R reset, Q/Esc quit.")
    controls = PointCloudControls()
    view = ViewState()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    try:
        while True:
            ok, left, right = depth_v2.read_camera_pair(left_camera, right_camera)
            if not ok or left is None or right is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue
            runtime_filters = depth_v2.DepthFilterSettings.from_config()
            grid, z_scale, plane_width, min_distance, max_distance, point_size, filters_enabled = controls.values()
            if not filters_enabled:
                runtime_filters = dataclass_replace(runtime_filters, enabled=False)
            rectified, _, disparity, distance_mm, valid = depth_v2.compute_depth_frame(
                depth, left, right, runtime_filters, cuda_matcher, parallel_bm_matcher
            )
            valid &= (distance_mm >= min_distance) & (distance_mm <= max_distance)
            distance_mm, valid = depth_v2.fill_distance_gaps(
                distance_mm, valid, runtime_filters, min_distance, max_distance
            )
            vertices, colours, vertex_valid = depth_mesh(rectified, distance_mm, valid, grid, z_scale, plane_width)
            frame = render_mesh(vertices, colours, vertex_valid, view, point_size)
            cv2.putText(frame, f"Grid: {grid} x {grid} | Points: {np.count_nonzero(vertex_valid)} | Z scale: {z_scale:.1f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Depth: {min_distance:.0f}-{max_distance:.0f} mm", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("a"):
                view.yaw_deg -= 5.0
            elif key == ord("d"):
                view.yaw_deg += 5.0
            elif key == ord("w"):
                view.pitch_deg -= 5.0
            elif key == ord("s"):
                view.pitch_deg += 5.0
            elif key == ord("z"):
                view.roll_deg -= 5.0
            elif key == ord("x"):
                view.roll_deg += 5.0
            elif key in (ord("+"), ord("=")):
                view.zoom = min(5.0, view.zoom * 1.15)
            elif key in (ord("-"), ord("_")):
                view.zoom = max(0.2, view.zoom / 1.15)
            elif key in (ord("r"), ord("R")):
                view = ViewState()
    finally:
        left_camera.release()
        right_camera.release()
        if parallel_bm_matcher is not None:
            parallel_bm_matcher.close()
        cv2.destroyAllWindows()
    return 0


def dataclass_replace(settings: depth_v2.DepthFilterSettings, *, enabled: bool) -> depth_v2.DepthFilterSettings:
    return depth_v2.DepthFilterSettings(
        enabled, settings.filter_speckles, settings.speckle_max_size_px, settings.speckle_disparity_range,
        settings.fill_morphological_holes, settings.morphological_kernel_size, settings.fill_contour_holes,
        settings.gradient_fill_all_gaps, settings.max_hole_area_px, settings.inpaint_radius,
        settings.median_filter, settings.median_kernel_size, settings.median_min_valid_ratio,
        settings.smooth_disparity, settings.bilateral_diameter, settings.bilateral_sigma_color,
        settings.bilateral_sigma_space,
    )


if __name__ == "__main__":
    raise SystemExit(main())

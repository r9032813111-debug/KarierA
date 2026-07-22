"""V10: V9 fast RGBD visualizer with final 3D meshes cropped by ArUco."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from itertools import combinations

import cv2
import numpy as np
import pyvista as pv

from calibration_utils_v2 import can_start_3d
from config_v2 import CONFIG
from rgbd_point_cloud_gpu_v2 import CONTROLS_NAME, GpuControls
import rgbd_pyvista_v2 as v2
import stereo_depth_v2 as depth_v2


WINDOW_TITLE = "RGBD PyVista v10 - ArUco cropped RGB"
TOPOLOGY_WINDOW_TITLE = "RGBD PyVista v10 - ArUco cropped topology"
DEPTH_WINDOW_TITLE = "RGBD Depth Map v10"
ARUCO_WINDOW_TITLE = "ArUco Detection v10"

SURFACE_FILL_ENABLED = True
SURFACE_FILL_TEMPORAL_FRAMES = 7
SURFACE_FILL_MOTION_THRESHOLD = 14
SURFACE_FILL_MIN_COMPONENT_AREA = 24
SURFACE_FILL_MAX_COMPONENT_RATIO = 0.35
SURFACE_FILL_EDGE_DILATION = 2
SURFACE_FILL_BORDER_RADIUS = 7
SURFACE_FILL_COLOR_THRESHOLD = 30.0
SURFACE_FILL_MIN_BORDER_POINTS = 30
SURFACE_FILL_MAX_MEDIAN_ERROR_MM = 35.0
SURFACE_FILL_MAX_P90_ERROR_MM = 90.0
SURFACE_FILL_MAX_FIT_POINTS = 2048
CONFIDENCE_INVALID = 0
CONFIDENCE_PLANE = 1
CONFIDENCE_TEMPORAL = 2
CONFIDENCE_MEASURED = 3
ARUCO_EDGE_MARGIN_RATIO = 0.40


class ArucoRoiTracker:
    """Cache the last valid rectangle from four standard 4x4 ArUco markers."""

    def __init__(self) -> None:
        dictionaries = (
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_4X4_100,
            cv2.aruco.DICT_4X4_250,
            cv2.aruco.DICT_4X4_1000,
        )
        self.detectors = tuple(
            cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(dictionary),
                cv2.aruco.DetectorParameters(),
            )
            for dictionary in dictionaries
        )
        self.last_quad: np.ndarray | None = None
        self.last_detected_corners: tuple[np.ndarray, ...] = ()
        self.last_detected_ids: np.ndarray | None = None
        self.roi_live = False

    @staticmethod
    def _order_corners(points: np.ndarray) -> np.ndarray:
        sums = points.sum(axis=1)
        differences = points[:, 0] - points[:, 1]
        return np.array(
            (
                points[np.argmin(sums)],
                points[np.argmax(differences)],
                points[np.argmax(sums)],
                points[np.argmin(differences)],
            ),
            dtype=np.float32,
        )

    @classmethod
    def _select_trapezoid(cls, points: np.ndarray) -> np.ndarray | None:
        """Choose the four edge markers most likely to form the field trapezoid."""
        if len(points) < 4:
            return None
        best_quad: np.ndarray | None = None
        best_score = -np.inf
        for selected in combinations(points, 4):
            quad = cls._order_corners(np.asarray(selected, dtype=np.float32))
            if len(np.unique(quad, axis=0)) != 4:
                continue
            area = abs(cv2.contourArea(quad))
            if area < 64.0:
                continue
            top, right, bottom, left = (
                quad[1] - quad[0],
                quad[2] - quad[1],
                quad[3] - quad[2],
                quad[0] - quad[3],
            )
            lengths = np.array(
                (np.linalg.norm(top), np.linalg.norm(right), np.linalg.norm(bottom), np.linalg.norm(left))
            )
            if np.any(lengths < 1.0):
                continue
            parallel = abs(float(np.dot(top, bottom) / (lengths[0] * lengths[2])))
            parallel += abs(float(np.dot(right, left) / (lengths[1] * lengths[3])))
            horizontal = abs(float(top[0]) / lengths[0]) + abs(float(bottom[0]) / lengths[2])
            vertical = abs(float(right[1]) / lengths[1]) + abs(float(left[1]) / lengths[3])
            score = parallel * 4.0 + horizontal + vertical + np.log1p(area) * 0.1
            if score > best_score:
                best_score = score
                best_quad = quad
        return best_quad

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        margin_x = max(1, int(width * ARUCO_EDGE_MARGIN_RATIO))
        margin_y = max(1, int(height * ARUCO_EDGE_MARGIN_RATIO))
        self.roi_live = False
        best_corners: tuple[np.ndarray, ...] = ()
        best_ids: np.ndarray | None = None
        for detector in self.detectors:
            corners, ids, _rejected = detector.detectMarkers(gray)
            if ids is None:
                continue
            if len(corners) > len(best_corners):
                best_corners = tuple(corners)
                best_ids = ids.copy()
            centers = np.asarray([corner.reshape(-1, 2).mean(axis=0) for corner in corners], dtype=np.float32)
            edge_centers = centers[
                (centers[:, 0] <= margin_x)
                | (centers[:, 0] >= width - 1 - margin_x)
                | (centers[:, 1] <= margin_y)
                | (centers[:, 1] >= height - 1 - margin_y)
            ]
            candidates = edge_centers if len(edge_centers) >= 4 else centers
            quad = self._select_trapezoid(candidates)
            if quad is not None:
                self.last_quad = quad
                self.roi_live = True
                break
        self.last_detected_corners = best_corners
        self.last_detected_ids = best_ids
        return self.last_quad

    def reset(self) -> None:
        self.last_quad = None
        self.roi_live = False

    def draw_debug(self, frame: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
        """Show live ArUco detections and the active crop boundary on video."""
        output = frame.copy()
        if self.last_detected_ids is not None:
            cv2.aruco.drawDetectedMarkers(
                output,
                list(self.last_detected_corners),
                self.last_detected_ids,
            )
        if quad is not None:
            cv2.polylines(
                output,
                [np.rint(quad).astype(np.int32)],
                True,
                (0, 255, 0) if self.roi_live else (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        state = "ROI live" if self.roi_live else "ROI cached or full frame"
        cv2.putText(output, state, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return output


def aruco_cell_mask(
    image_shape: tuple[int, int],
    grid_shape: tuple[int, int],
    quad: np.ndarray,
) -> np.ndarray:
    """Keep only mesh cells whose four original grid vertices are in the ROI."""
    height, width = image_shape
    rows, columns = grid_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    ys = np.linspace(0, height - 1, rows, dtype=np.int32)
    xs = np.linspace(0, width - 1, columns, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = mask[grid_y, grid_x].astype(bool)
    return inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, 1:] & inside[1:, :-1]


class TemporalDepthCompleter:
    """Fill static holes from recent frames without stacking full images."""

    _MOTION_KERNEL = np.ones((7, 7), np.uint8)

    def __init__(self, history_size: int) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=max(1, history_size))
        self.previous_gray: np.ndarray | None = None

    def commit_reconstructed(
        self,
        completed_depth_mm: np.ndarray,
        reconstructed_mask: np.ndarray,
    ) -> None:
        """Seed the newest history frame with accepted reconstructed surfaces."""
        if not self.history or not np.any(reconstructed_mask):
            return
        latest = self.history[-1]
        latest[reconstructed_mask] = completed_depth_mm[reconstructed_mask]

    def fill(
        self,
        gray: np.ndarray,
        depth_mm: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.full(depth_mm.shape, np.nan, dtype=np.float32)
        current[valid] = depth_mm[valid]
        self.history.append(current)

        result = depth_mm.copy()
        empty_mask = np.zeros(valid.shape, dtype=bool)
        if self.previous_gray is None or len(self.history) < 2:
            self.previous_gray = gray.copy()
            return result, empty_mask

        difference = cv2.absdiff(gray, self.previous_gray)
        motion = cv2.compare(
            difference,
            SURFACE_FILL_MOTION_THRESHOLD,
            cv2.CMP_GT,
        )
        motion = cv2.dilate(motion, self._MOTION_KERNEL, iterations=1)
        self.previous_gray = gray.copy()

        candidates = (~valid) & (motion == 0)
        candidate_count = int(np.count_nonzero(candidates))
        if candidate_count == 0:
            return result, empty_mask

        # Exact temporal median, but only for pixels that can actually be filled.
        # The old implementation stacked every 1920x1080 frame, even when only a
        # small fraction of pixels was missing.
        history_values = np.empty(
            (len(self.history), candidate_count),
            dtype=np.float32,
        )
        for index, frame in enumerate(self.history):
            history_values[index] = frame[candidates]

        # Persistent holes can be all-NaN in every history frame.  Excluding
        # them before nanmedian avoids both warnings and a large wasted sort.
        has_measurement = np.any(np.isfinite(history_values), axis=0)
        if not np.any(has_measurement):
            return result, empty_mask
        temporal_values = np.full(candidate_count, np.nan, dtype=np.float32)
        with np.errstate(all="ignore"):
            temporal_values[has_measurement] = np.nanmedian(
                history_values[:, has_measurement],
                axis=0,
            )
        finite = np.isfinite(temporal_values)

        candidate_y, candidate_x = np.nonzero(candidates)
        fill_y = candidate_y[finite]
        fill_x = candidate_x[finite]
        result[fill_y, fill_x] = temporal_values[finite]
        fill_mask = np.zeros(valid.shape, dtype=bool)
        fill_mask[fill_y, fill_x] = True
        return result, fill_mask


def _robust_inverse_depth_plane(
    xs: np.ndarray,
    ys: np.ndarray,
    depth_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    finite = np.isfinite(depth_mm) & (depth_mm > 0)
    if np.count_nonzero(finite) < SURFACE_FILL_MIN_BORDER_POINTS:
        return None
    xs = xs[finite].astype(np.float64, copy=False)
    ys = ys[finite].astype(np.float64, copy=False)
    depths = depth_mm[finite].astype(np.float64, copy=False)

    # Thousands of nearly adjacent boundary pixels do not improve a 3-parameter
    # plane fit, but make repeated least-squares solves much slower.
    if xs.size > SURFACE_FILL_MAX_FIT_POINTS:
        selection = np.linspace(
            0,
            xs.size - 1,
            SURFACE_FILL_MAX_FIT_POINTS,
            dtype=np.int32,
        )
        xs = xs[selection]
        ys = ys[selection]
        depths = depths[selection]

    inverse_depth = 1.0 / depths
    design = np.empty((xs.size, 3), dtype=np.float64)
    design[:, 0] = xs
    design[:, 1] = ys
    design[:, 2] = 1.0
    keep = np.ones(xs.shape[0], dtype=bool)
    coefficients: np.ndarray | None = None

    for _ in range(4):
        if np.count_nonzero(keep) < SURFACE_FILL_MIN_BORDER_POINTS:
            return None
        coefficients, *_ = np.linalg.lstsq(
            design[keep],
            inverse_depth[keep],
            rcond=None,
        )
        predicted_inverse = design @ coefficients
        predicted_depth = np.divide(
            1.0,
            predicted_inverse,
            out=np.full_like(predicted_inverse, np.nan),
            where=predicted_inverse > 1e-12,
        )
        residual = np.abs(predicted_depth - depths)
        kept_residual = residual[keep]
        center = np.nanmedian(kept_residual)
        mad = np.nanmedian(np.abs(kept_residual - center))
        robust_limit = max(
            SURFACE_FILL_MAX_MEDIAN_ERROR_MM * 2.0,
            center + 3.5 * max(mad, 1.0),
        )
        new_keep = np.isfinite(residual) & (residual <= robust_limit)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    if coefficients is None or np.count_nonzero(keep) < SURFACE_FILL_MIN_BORDER_POINTS:
        return None
    predicted_inverse = design @ coefficients
    predicted_depth = np.divide(
        1.0,
        predicted_inverse,
        out=np.full_like(predicted_inverse, np.nan),
        where=predicted_inverse > 1e-12,
    )
    residual = np.abs(predicted_depth - depths)
    kept_residual = residual[keep]
    if float(np.nanmedian(kept_residual)) > SURFACE_FILL_MAX_MEDIAN_ERROR_MM:
        return None
    if float(np.nanpercentile(kept_residual, 90)) > SURFACE_FILL_MAX_P90_ERROR_MM:
        return None
    return coefficients, keep


def complete_low_texture_surfaces(
    rectified_bgr: np.ndarray,
    rectified_gray: np.ndarray,
    depth_mm: np.ndarray,
    measured_valid: np.ndarray,
    minimum_mm: float,
    maximum_mm: float,
    temporal: TemporalDepthCompleter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measured_valid = (
        measured_valid
        & np.isfinite(depth_mm)
        & (depth_mm >= minimum_mm)
        & (depth_mm <= maximum_mm)
    )
    confidence = np.full(depth_mm.shape, CONFIDENCE_INVALID, dtype=np.uint8)
    confidence[measured_valid] = CONFIDENCE_MEASURED
    completed, temporal_mask = temporal.fill(
        rectified_gray,
        depth_mm,
        measured_valid,
    )
    confidence[temporal_mask] = CONFIDENCE_TEMPORAL
    valid = measured_valid | temporal_mask
    if not SURFACE_FILL_ENABLED:
        return completed, valid, confidence

    gray_f = rectified_gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    local_mean = cv2.boxFilter(
        gray_f,
        cv2.CV_32F,
        (9, 9),
        normalize=True,
    )
    local_sq_mean = cv2.sqrBoxFilter(
        gray_f,
        cv2.CV_32F,
        (9, 9),
        normalize=True,
    )
    local_variance = local_sq_mean - local_mean * local_mean
    textureless = (gradient < 10.0) & (local_variance < 55.0)

    edges = cv2.Canny(rectified_gray, 45, 120)
    if SURFACE_FILL_EDGE_DILATION > 0:
        edges = cv2.dilate(
            edges,
            np.ones((3, 3), np.uint8),
            iterations=SURFACE_FILL_EDGE_DILATION,
        )
    edge_mask = edges != 0
    candidate_u8 = ((~valid) & textureless & (~edge_mask)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_u8,
        connectivity=8,
    )
    if count <= 1:
        return completed, valid, confidence

    height, width = depth_mm.shape
    maximum_area = int(height * width * SURFACE_FILL_MAX_COMPONENT_RATIO)
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    lefts = stats[1:, cv2.CC_STAT_LEFT]
    tops = stats[1:, cv2.CC_STAT_TOP]
    eligible = (
        (areas >= SURFACE_FILL_MIN_COMPONENT_AREA)
        & (areas <= maximum_area)
        & (lefts > 0)
        & (tops > 0)
        & ((lefts + widths) < width)
        & ((tops + heights) < height)
    )
    eligible_ids = np.flatnonzero(eligible) + 1
    if eligible_ids.size == 0:
        return completed, valid, confidence

    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_kernel = np.ones(
        (SURFACE_FILL_BORDER_RADIUS, SURFACE_FILL_BORDER_RADIUS),
        np.uint8,
    )
    border_padding = SURFACE_FILL_BORDER_RADIUS // 2 + 1

    for label_id in eligible_ids:
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        component_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        x0 = max(0, x - border_padding)
        y0 = max(0, y - border_padding)
        x1 = min(width, x + component_width + border_padding)
        y1 = min(height, y + component_height + border_padding)

        labels_roi = labels[y0:y1, x0:x1]
        region = labels_roi == label_id
        if not np.any(region):
            continue
        region_u8 = region.astype(np.uint8)
        valid_roi = valid[y0:y1, x0:x1]
        edge_roi = edge_mask[y0:y1, x0:x1]
        border = cv2.dilate(
            region_u8,
            border_kernel,
            iterations=1,
        ).astype(bool)
        border &= ~region
        border &= valid_roi
        border &= ~edge_roi
        border_count = int(np.count_nonzero(border))
        if border_count < SURFACE_FILL_MIN_BORDER_POINTS:
            continue

        lab_roi = lab[y0:y1, x0:x1]
        region_colors = lab_roi[region]
        if region_colors.shape[0] > 4000:
            step = max(1, region_colors.shape[0] // 4000)
            region_colors = region_colors[::step]
        median_color = np.median(region_colors, axis=0)

        border_y, border_x = np.nonzero(border)
        border_colors = lab_roi[border_y, border_x]
        color_delta = border_colors - median_color
        similar = np.einsum(
            "ij,ij->i",
            color_delta,
            color_delta,
        ) <= SURFACE_FILL_COLOR_THRESHOLD ** 2
        if np.count_nonzero(similar) < SURFACE_FILL_MIN_BORDER_POINTS:
            continue
        border_y = border_y[similar]
        border_x = border_x[similar]
        global_border_y = border_y + y0
        global_border_x = border_x + x0

        fitted = _robust_inverse_depth_plane(
            global_border_x,
            global_border_y,
            completed[global_border_y, global_border_x],
        )
        if fitted is None:
            continue
        coefficients, inlier_mask = fitted

        # Rebuild the same deterministic fit subset used by the fitter so the
        # depth limits are based on corresponding inliers.
        fit_x = global_border_x
        fit_y = global_border_y
        fit_depth = completed[fit_y, fit_x]
        finite_fit = np.isfinite(fit_depth) & (fit_depth > 0)
        fit_x = fit_x[finite_fit]
        fit_y = fit_y[finite_fit]
        fit_depth = fit_depth[finite_fit]
        if fit_x.size > SURFACE_FILL_MAX_FIT_POINTS:
            selection = np.linspace(
                0,
                fit_x.size - 1,
                SURFACE_FILL_MAX_FIT_POINTS,
                dtype=np.int32,
            )
            fit_depth = fit_depth[selection]
        if inlier_mask.size != fit_depth.size:
            continue
        inlier_depth = fit_depth[inlier_mask]
        low = max(
            minimum_mm,
            float(np.nanpercentile(inlier_depth, 5)) - 150.0,
        )
        high = min(
            maximum_mm,
            float(np.nanpercentile(inlier_depth, 95)) + 150.0,
        )

        region_y, region_x = np.nonzero(region)
        global_region_y = region_y + y0
        global_region_x = region_x + x0
        inverse_prediction = (
            coefficients[0] * global_region_x
            + coefficients[1] * global_region_y
            + coefficients[2]
        )
        prediction = np.divide(
            1.0,
            inverse_prediction,
            out=np.full_like(inverse_prediction, np.nan, dtype=np.float64),
            where=inverse_prediction > 1e-12,
        )
        prediction_valid = (
            np.isfinite(prediction)
            & (prediction >= low)
            & (prediction <= high)
        )
        if np.count_nonzero(prediction_valid) < max(1, int(area * 0.65)):
            continue

        fill_y = global_region_y[prediction_valid]
        fill_x = global_region_x[prediction_valid]
        completed[fill_y, fill_x] = prediction[prediction_valid].astype(np.float32)
        valid[fill_y, fill_x] = True
        confidence[fill_y, fill_x] = CONFIDENCE_PLANE

    # On the next static frame these accepted planes can be restored by the
    # cheap temporal path.  Motion detection prevents stale surfaces from
    # leaking onto moving objects.
    temporal.commit_reconstructed(
        completed,
        confidence == CONFIDENCE_PLANE,
    )
    return completed, valid, confidence


def draw_completion_overlay(
    depth_view: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Blend confidence colors only at selected pixels; avoid full-frame copies."""
    for code, color in (
        (CONFIDENCE_TEMPORAL, np.array((0, 255, 255), dtype=np.float32)),
        (CONFIDENCE_PLANE, np.array((0, 255, 0), dtype=np.float32)),
    ):
        mask = confidence == code
        if not np.any(mask):
            continue
        pixels = depth_view[mask].astype(np.float32)
        depth_view[mask] = np.clip(
            pixels * 0.75 + color * 0.25,
            0,
            255,
        ).astype(np.uint8)



def build_topology_texture(
    distance_mm: np.ndarray,
    valid: np.ndarray,
    minimum_mm: float,
    maximum_mm: float,
    contour_step_mm: float,
) -> np.ndarray:
    """Build a pure distance-color texture with optional topographic contours."""
    topology = depth_v2.colorize_distance(
        distance_mm,
        valid,
        maximum_mm,
        minimum_mm,
        True,
    )
    if contour_step_mm <= 0.0 or not np.any(valid):
        return topology

    safe_depth = np.where(valid, distance_mm, minimum_mm).astype(np.float32)
    level = np.floor((safe_depth - minimum_mm) / contour_step_mm).astype(np.int32)
    contour = np.zeros(valid.shape, dtype=bool)
    contour[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (level[:, 1:] != level[:, :-1])
    contour[1:, :] |= valid[1:, :] & valid[:-1, :] & (level[1:, :] != level[:-1, :])
    if np.any(contour):
        contour_u8 = cv2.dilate(
            contour.astype(np.uint8),
            np.ones((2, 2), np.uint8),
            iterations=1,
        )
        # Black isolines preserve the distance colors and make relief readable.
        topology[contour_u8 != 0] = (0, 0, 0)
    return topology


def compute_measured_frame(
    depth: depth_v2.StereoDepth,
    left: np.ndarray,
    right: np.ndarray,
    filters: depth_v2.DepthFilterSettings,
    cuda_matcher: depth_v2.TorchBlockMatcher | None,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only depth values backed by matching correspondence measurements."""
    measured_filters = replace(
        filters,
        gradient_fill_all_gaps=False,
    )
    left_rectified, right_rectified = depth.rectify(left, right)
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    if cuda_matcher is not None:
        raw_disparity = cuda_matcher.compute(left_gray, right_gray)
    else:
        raw_disparity = depth.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    raw_valid = np.isfinite(raw_disparity) & (raw_disparity > depth.min_disparity + 0.5)
    _disparity, distance_mm, filtered_valid = depth.depth_from_disparity(raw_disparity, measured_filters)
    measured_valid = (
        raw_valid
        & filtered_valid
        & np.isfinite(distance_mm)
        & (distance_mm >= minimum)
        & (distance_mm <= maximum)
    )
    return left_rectified, left_gray, distance_mm, measured_valid


_GRID_CACHE_KEY: tuple[int, int, int, float, bool] | None = None
_GRID_CACHE_VALUE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


def _grid_geometry(
    height: int,
    width: int,
    grid: int,
    plane_width_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _GRID_CACHE_KEY, _GRID_CACHE_VALUE
    key = (height, width, grid, float(plane_width_mm), bool(CONFIG.RGBD_FLIP_VERTICAL))
    if key == _GRID_CACHE_KEY and _GRID_CACHE_VALUE is not None:
        return _GRID_CACHE_VALUE
    ys = np.linspace(0, height - 1, grid, dtype=np.int32)
    xs = np.linspace(0, width - 1, grid, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    plane_height_mm = plane_width_mm * height / width
    x = (grid_x.astype(np.float32) / max(1, width - 1) - 0.5) * plane_width_mm
    y_normalized = grid_y.astype(np.float32) / max(1, height - 1)
    if CONFIG.RGBD_FLIP_VERTICAL:
        y_normalized = 1.0 - y_normalized
    y = (y_normalized - 0.5) * plane_height_mm
    texture_coordinates = np.dstack((
        grid_x.astype(np.float32) / max(1, width - 1),
        1.0 - grid_y.astype(np.float32) / max(1, height - 1),
    )).astype(np.float32)
    _GRID_CACHE_KEY = key
    _GRID_CACHE_VALUE = grid_x, grid_y, x, y, texture_coordinates
    return _GRID_CACHE_VALUE


def _nearest_valid_sources(
    valid: np.ndarray,
    sample_y: np.ndarray,
    sample_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use OpenCV C++ distance labels instead of SciPy full-frame EDT indices."""
    invalid_u8 = (~valid).astype(np.uint8)
    _distance, labels = cv2.distanceTransformWithLabels(
        invalid_u8,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    valid_y, valid_x = np.nonzero(valid)
    valid_labels = labels[valid_y, valid_x]
    max_label = int(labels.max())
    lookup_y = np.zeros(max_label + 1, dtype=np.int32)
    lookup_x = np.zeros(max_label + 1, dtype=np.int32)
    lookup_y[valid_labels] = valid_y
    lookup_x[valid_labels] = valid_x
    sampled_labels = labels[sample_y, sample_x]
    return lookup_y[sampled_labels], lookup_x[sampled_labels]


def relocate_grid_vertices(
    distance_mm: np.ndarray,
    measured_valid: np.ndarray,
    grid: int,
    z_scale: float,
    plane_width_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep a regular mesh grid and take Z from the nearest measured pixel."""
    height, width = distance_mm.shape
    grid = min(grid, height, width)
    grid_x, grid_y, x, y, texture_coordinates = _grid_geometry(
        height,
        width,
        grid,
        plane_width_mm,
    )
    if np.any(measured_valid):
        source_y, source_x = _nearest_valid_sources(
            measured_valid,
            grid_y,
            grid_x,
        )
    else:
        source_y, source_x = grid_y, grid_x
    vertex_valid = measured_valid[source_y, source_x]
    moved = (source_y != grid_y) | (source_x != grid_x)
    z = distance_mm[source_y, source_x].astype(np.float32) * z_scale
    if CONFIG.RGBD_INVERT_DEPTH:
        z = -z
    vertices = np.empty((grid, grid, 3), dtype=np.float32)
    vertices[:, :, 0] = x
    vertices[:, :, 1] = y
    vertices[:, :, 2] = z
    source_points = np.empty((grid, grid, 2), dtype=np.int32)
    source_points[:, :, 0] = source_x
    source_points[:, :, 1] = source_y
    return vertices, texture_coordinates, vertex_valid, moved, source_points


def draw_v5_vertex_overlay(
    depth_view: np.ndarray,
    source_points: np.ndarray,
    vertex_valid: np.ndarray,
    moved: np.ndarray,
    show_measured: bool,
) -> None:
    """Show only grid points that have a direct stereo correspondence."""
    height, width = depth_view.shape[:2]
    source_x = source_points[:, :, 0]
    source_y = source_points[:, :, 1]
    measured_mask = np.zeros((height, width), dtype=np.uint8)
    direct = vertex_valid & ~moved
    measured_mask[source_y[direct], source_x[direct]] = 255
    marker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    measured_mask = cv2.dilate(measured_mask, marker)
    if show_measured:
        depth_view[measured_mask > 0] = (255, 255, 255)


def create_relocated_mesh(vertices: np.ndarray, texture_coordinates: np.ndarray) -> pv.StructuredGrid:
    mesh = v2.create_mesh(vertices)
    mesh.active_texture_coordinates = np.ascontiguousarray(
        texture_coordinates.reshape(-1, 2),
        dtype=np.float32,
    )
    return mesh


def update_relocated_mesh(
    mesh: pv.StructuredGrid,
    vertices: np.ndarray,
    vertex_valid: np.ndarray,
    texture_coordinates: np.ndarray,
) -> None:
    v2.update_mesh(mesh, vertices, vertex_valid)
    mesh.active_texture_coordinates = np.ascontiguousarray(
        texture_coordinates.reshape(-1, 2),
        dtype=np.float32,
    )
    mesh.Modified()


def create_aruco_clipped_mesh(
    vertices: np.ndarray,
    texture_coordinates: np.ndarray,
    cell_mask: np.ndarray,
) -> pv.PolyData:
    """Use the V9 geometry but omit faces outside the marker rectangle."""
    rows, columns = vertices.shape[:2]
    point_ids = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    quads = np.stack(
        (
            point_ids[:-1, :-1],
            point_ids[:-1, 1:],
            point_ids[1:, 1:],
            point_ids[1:, :-1],
        ),
        axis=2,
    )[cell_mask]
    faces = np.column_stack((np.full(len(quads), 4, dtype=np.int64), quads)).ravel()
    mesh = pv.PolyData(np.ascontiguousarray(vertices.reshape(-1, 3), dtype=np.float32), faces)
    mesh.active_texture_coordinates = np.ascontiguousarray(
        texture_coordinates.reshape(-1, 2),
        dtype=np.float32,
    )
    return mesh


def update_aruco_clipped_mesh(
    mesh: pv.PolyData,
    vertices: np.ndarray,
    vertex_valid: np.ndarray,
    texture_coordinates: np.ndarray,
) -> None:
    """Update the clipped mesh height while keeping its ArUco-selected faces."""
    v2.update_mesh(mesh, vertices, vertex_valid)
    mesh.active_texture_coordinates = np.ascontiguousarray(
        texture_coordinates.reshape(-1, 2),
        dtype=np.float32,
    )
    mesh.Modified()


def suppress_sector_height_outliers(
    vertices: np.ndarray,
    vertex_valid: np.ndarray,
    sector_size_px: int,
    maximum_error_mm: float,
) -> np.ndarray:
    """Apply a one-point-step sliding median window to the mesh height map."""
    smallest_side = min(vertices.shape[0], vertices.shape[1])
    if smallest_side < 3:
        return vertices
    output = vertices.copy()
    depth = np.abs(output[:, :, 2]).astype(np.float32)
    reliable = vertex_valid & np.isfinite(depth)
    if not np.any(reliable):
        return output

    # Fill unavailable vertices only for the median calculation.  A zero-height
    # gap must not pull a local sector median down and create a boundary spike.
    median_input = depth.copy()
    if not np.all(reliable):
        missing_y, missing_x = np.nonzero(~reliable)
        source_y, source_x = _nearest_valid_sources(
            reliable,
            missing_y,
            missing_x,
        )
        median_input[missing_y, missing_x] = depth[source_y, source_x]
    requested_size = max(3, int(sector_size_px))
    kernel_size = min(requested_size, smallest_side)
    if kernel_size % 2 == 0:
        kernel_size = max(3, kernel_size - 1)

    if kernel_size <= 5:
        # OpenCV preserves float32 exactly for 3x3 and 5x5 kernels.
        local_median = cv2.medianBlur(median_input, kernel_size)
    else:
        # OpenCV's histogram median is nearly constant-time for large kernels,
        # but supports only uint8 there.  Quantization error is tiny compared
        # with the centimetre-scale outlier threshold used by this tool.
        reliable_values = median_input[reliable]
        depth_low = float(np.min(reliable_values))
        depth_high = float(np.max(reliable_values))
        if depth_high <= depth_low + 1e-6:
            local_median = np.full_like(median_input, depth_low)
        else:
            quantize_scale = 255.0 / (depth_high - depth_low)
            quantized = np.clip(
                (median_input - depth_low) * quantize_scale,
                0.0,
                255.0,
            ).astype(np.uint8)
            median_u8 = cv2.medianBlur(quantized, kernel_size)
            local_median = (
                median_u8.astype(np.float32) / quantize_scale + depth_low
            )
    outliers = reliable & (
        np.abs(depth - local_median) > maximum_error_mm
    )
    if np.any(outliers):
        output[:, :, 2][outliers] = np.copysign(
            local_median[outliers],
            output[:, :, 2][outliers],
        )
    return output


def main() -> int:
    args = v2.parse_args()
    cv2.setUseOptimized(True)
    cv2.setNumThreads(CONFIG.CPU_THREADS)
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
    if args.depth_backend == "torch-cuda":
        depth, cuda_matcher, parallel_bm_matcher = depth_v2.build_depth_pipeline(calibration, args, block_size)
        print("Depth backend: experimental PyTorch CUDA matcher")
    else:
        depth = depth_v2.StereoDepth(calibration, args.matcher, args.min_disparity, args.num_disparities, block_size)
        cuda_matcher = None
        parallel_bm_matcher = None
        print(f"Depth backend: native OpenCV {args.matcher.upper()} at {CONFIG.CPU_THREADS} threads")

    controls = GpuControls()
    cv2.setTrackbarMax("Grid", CONTROLS_NAME, 300)
    cv2.createTrackbar("White pts", CONTROLS_NAME, int(CONFIG.V9_WHITE_POINTS), 1, lambda _value: None)
    cv2.createTrackbar("Median area px", CONTROLS_NAME, CONFIG.V9_MEDIAN_AREA_PX, 400, lambda _value: None)
    cv2.createTrackbar("Outlier cm", CONTROLS_NAME, CONFIG.V9_OUTLIER_CM, 200, lambda _value: None)
    cv2.createTrackbar("Topo contour cm", CONTROLS_NAME, CONFIG.V9_TOPO_CONTOUR_CM, 200, lambda _value: None)
    cv2.namedWindow(DEPTH_WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEPTH_WINDOW_TITLE, 900, 600)
    cv2.namedWindow(ARUCO_WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ARUCO_WINDOW_TITLE, 900, 600)
    swap_requested = False
    temporal_completer = TemporalDepthCompleter(SURFACE_FILL_TEMPORAL_FRAMES)
    aruco_tracker = ArucoRoiTracker()

    def request_camera_swap() -> None:
        nonlocal swap_requested
        swap_requested = True

    rgb_plotter: pv.Plotter | None = None
    topology_plotter: pv.Plotter | None = None
    rgb_mesh_actor = None
    topology_mesh_actor = None
    try:
        left_camera.open()
        right_camera.open()
        if not depth_v2.warmup_camera_pair(left_camera, right_camera):
            raise RuntimeError("Both cameras opened but cannot deliver frames together")
        print("V10 crops both final 3D meshes to the rectangle bounded by four ArUco markers.")
        mesh: pv.PolyData | None = None
        current_cell_mask: np.ndarray | None = None
        rgb_texture_pixels: np.ndarray | None = None
        rgb_texture = None
        topology_texture_pixels: np.ndarray | None = None
        topology_texture = None
        depth_future: Future[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgbd-v10-depth") as executor:
            while (
                (rgb_plotter is None and topology_plotter is None)
                or (rgb_plotter is not None and not rgb_plotter._closed)
                or (topology_plotter is not None and not topology_plotter._closed)
            ):
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("c"), ord("C")):
                    request_camera_swap()
                frame_updated = False
                if depth_future is not None and depth_future.done():
                    try:
                        rectified, rectified_gray, distance_mm, measured_valid = depth_future.result()
                    except Exception as exc:
                        print(f"Warning: depth worker failed: {exc}")
                    else:
                        grid, z_scale, plane_width, minimum, maximum, current_filters, z_smoothing = controls.values()
                        sector_size_px = max(3, cv2.getTrackbarPos("Median area px", CONTROLS_NAME))
                        maximum_error_mm = max(1.0, cv2.getTrackbarPos("Outlier cm", CONTROLS_NAME) * 10.0)
                        completed_distance_mm, surface_valid, confidence = complete_low_texture_surfaces(
                            rectified,
                            rectified_gray,
                            distance_mm,
                            measured_valid,
                            minimum,
                            maximum,
                            temporal_completer,
                        )
                        vertices, texture_coordinates, vertex_valid, moved, source_points = relocate_grid_vertices(
                            completed_distance_mm,
                            surface_valid,
                            grid,
                            z_scale,
                            plane_width,
                        )
                        vertices = suppress_sector_height_outliers(
                            vertices,
                            vertex_valid,
                            sector_size_px,
                            maximum_error_mm,
                        )
                        vertices = v2.smooth_mesh_z(vertices, z_smoothing)
                        contour_step_mm = max(
                            0.0,
                            float(cv2.getTrackbarPos("Topo contour cm", CONTROLS_NAME)) * 10.0,
                        )
                        topology_view = build_topology_texture(
                            completed_distance_mm,
                            surface_valid,
                            minimum,
                            maximum,
                            contour_step_mm,
                        )
                        depth_view = topology_view.copy()
                        draw_completion_overlay(depth_view, confidence)
                        draw_v5_vertex_overlay(
                            depth_view,
                            source_points,
                            vertex_valid,
                            moved,
                            bool(cv2.getTrackbarPos("White pts", CONTROLS_NAME)),
                        )
                        cv2.imshow(DEPTH_WINDOW_TITLE, depth_v2.render_for_window(depth_view, DEPTH_WINDOW_TITLE, CONFIG.DISPLAY_SCALE))
                        roi_quad = aruco_tracker.update(rectified)
                        aruco_view = aruco_tracker.draw_debug(rectified, roi_quad)
                        cv2.imshow(
                            ARUCO_WINDOW_TITLE,
                            depth_v2.render_for_window(aruco_view, ARUCO_WINDOW_TITLE, CONFIG.DISPLAY_SCALE),
                        )
                        if not np.any(vertex_valid):
                            print("Warning: no valid stereo correspondence available for the V10 mesh")
                            depth_future = None
                            continue
                        if roi_quad is None:
                            cell_mask = np.ones(
                                (vertex_valid.shape[0] - 1, vertex_valid.shape[1] - 1),
                                dtype=bool,
                            )
                        else:
                            cell_mask = aruco_cell_mask(distance_mm.shape, vertex_valid.shape, roi_quad)
                        if not np.any(cell_mask):
                            depth_future = None
                            continue
                        topology_changed = (
                            current_cell_mask is None
                            or current_cell_mask.shape != cell_mask.shape
                            or not np.array_equal(current_cell_mask, cell_mask)
                        )
                        if mesh is None:
                            mesh = create_aruco_clipped_mesh(vertices, texture_coordinates, cell_mask)
                            current_cell_mask = cell_mask.copy()

                            rgb_texture = pv.numpy_to_texture(
                                np.flipud(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
                            )
                            rgb_texture_pixels = rgb_texture.to_image().point_data["Image"]
                            rgb_plotter = pv.Plotter(
                                window_size=(1280, 720),
                                title=WINDOW_TITLE,
                            )
                            rgb_mesh_actor = rgb_plotter.add_mesh(
                                mesh,
                                texture=rgb_texture,
                                smooth_shading=False,
                                show_edges=False,
                            )
                            rgb_plotter.add_key_event("r", rgb_plotter.reset_camera)
                            rgb_plotter.add_key_event("s", lambda: v2.save_textured_model(rgb_plotter))
                            rgb_plotter.add_key_event("c", request_camera_swap)
                            rgb_plotter.show(auto_close=False, interactive_update=True)
                            rgb_plotter.view_isometric()

                            topology_texture = pv.numpy_to_texture(
                                np.flipud(cv2.cvtColor(topology_view, cv2.COLOR_BGR2RGB))
                            )
                            topology_texture_pixels = topology_texture.to_image().point_data["Image"]
                            topology_plotter = pv.Plotter(
                                window_size=(1280, 720),
                                title=TOPOLOGY_WINDOW_TITLE,
                            )
                            topology_mesh_actor = topology_plotter.add_mesh(
                                mesh,
                                texture=topology_texture,
                                smooth_shading=False,
                                show_edges=False,
                            )
                            topology_plotter.add_key_event("r", topology_plotter.reset_camera)
                            topology_plotter.add_key_event(
                                "s",
                                lambda: v2.save_textured_model(topology_plotter),
                            )
                            topology_plotter.add_key_event("c", request_camera_swap)
                            topology_plotter.show(auto_close=False, interactive_update=True)
                            topology_plotter.view_isometric()

                        elif topology_changed:
                            assert rgb_texture is not None and topology_texture is not None
                            mesh = create_aruco_clipped_mesh(vertices, texture_coordinates, cell_mask)
                            current_cell_mask = cell_mask.copy()
                            if rgb_plotter is not None and not rgb_plotter._closed:
                                assert rgb_mesh_actor is not None
                                rgb_plotter.remove_actor(
                                    rgb_mesh_actor,
                                    reset_camera=False,
                                    render=False,
                                )
                                rgb_mesh_actor = rgb_plotter.add_mesh(
                                    mesh,
                                    texture=rgb_texture,
                                    smooth_shading=False,
                                    show_edges=False,
                                )
                            if topology_plotter is not None and not topology_plotter._closed:
                                assert topology_mesh_actor is not None
                                topology_plotter.remove_actor(
                                    topology_mesh_actor,
                                    reset_camera=False,
                                    render=False,
                                )
                                topology_mesh_actor = topology_plotter.add_mesh(
                                    mesh,
                                    texture=topology_texture,
                                    smooth_shading=False,
                                    show_edges=False,
                                )
                            if rgb_texture_pixels is not None:
                                v2.update_texture(rgb_texture_pixels, rectified)
                            if topology_texture_pixels is not None:
                                v2.update_texture(topology_texture_pixels, topology_view)
                        else:
                            update_aruco_clipped_mesh(
                                mesh,
                                vertices,
                                vertex_valid,
                                texture_coordinates,
                            )
                            if rgb_texture_pixels is not None:
                                v2.update_texture(rgb_texture_pixels, rectified)
                            if topology_texture_pixels is not None:
                                v2.update_texture(topology_texture_pixels, topology_view)
                        frame_updated = True
                    depth_future = None
                if depth_future is None and swap_requested:
                    left_camera, right_camera = right_camera, left_camera
                    left_index, right_index = right_index, left_index
                    aruco_tracker.reset()
                    swap_requested = False
                    print(f"Camera input indices swapped: left {left_index}, right {right_index}")
                if depth_future is None:
                    ok, left, right = depth_v2.read_camera_pair(left_camera, right_camera)
                    if ok and left is not None and right is not None:
                        _grid, _z_scale, _plane_width, minimum, maximum, filters, _z_smoothing = controls.values()
                        depth_future = executor.submit(
                            compute_measured_frame,
                            depth,
                            left.copy(),
                            right.copy(),
                            filters,
                            cuda_matcher,
                            minimum,
                            maximum,
                        )
                if rgb_plotter is not None and not rgb_plotter._closed:
                    rgb_plotter.update(stime=1, force_redraw=frame_updated)
                if topology_plotter is not None and not topology_plotter._closed:
                    topology_plotter.update(stime=1, force_redraw=frame_updated)
    finally:
        left_camera.release()
        right_camera.release()
        if parallel_bm_matcher is not None:
            parallel_bm_matcher.close()
        if rgb_plotter is not None and not rgb_plotter._closed:
            rgb_plotter.close()
        if topology_plotter is not None and not topology_plotter._closed:
            topology_plotter.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

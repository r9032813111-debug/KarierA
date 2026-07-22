"""Shared validation and geometry helpers for the existing calibration tools."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from config_v2 import CONFIG


PATTERN_SIZE = (CONFIG.CHESSBOARD_COLS, CONFIG.CHESSBOARD_ROWS)
EXPECTED_CORNER_COUNT = CONFIG.CHESSBOARD_COLS * CONFIG.CHESSBOARD_ROWS


def startup_validation() -> None:
    if CONFIG.SQUARE_SIZE_MM <= 0:
        raise ValueError("SQUARE_SIZE_MM must be positive")
    print(f"Chessboard pattern: {CONFIG.CHESSBOARD_COLS} x {CONFIG.CHESSBOARD_ROWS} inner corners")
    print(f"Printed board required: {CONFIG.CHESSBOARD_COLS + 1} x {CONFIG.CHESSBOARD_ROWS + 1} squares")
    print(f"Square size: {CONFIG.SQUARE_SIZE_MM:.1f} mm")


def make_object_points() -> np.ndarray:
    objp = np.zeros((EXPECTED_CORNER_COUNT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CONFIG.CHESSBOARD_COLS, 0:CONFIG.CHESSBOARD_ROWS].T.reshape(-1, 2)
    objp *= CONFIG.SQUARE_SIZE_MM
    return objp


def ensure_corner_count(corners: np.ndarray) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(points) != EXPECTED_CORNER_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_CORNER_COUNT} corners, got {len(points)}")
    return points.reshape(-1, 1, 2)


def validate_exact_resolution(frame: np.ndarray) -> None:
    if frame is None or frame.ndim < 2:
        raise ValueError("Camera returned an invalid frame")
    actual = (int(frame.shape[1]), int(frame.shape[0]))
    expected = (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT)
    if actual != expected:
        raise ValueError(f"Calibration requires {expected[0]}x{expected[1]}; received {actual[0]}x{actual[1]}")


def baseline_mm_from_t(translation: np.ndarray) -> float:
    value = float(np.linalg.norm(np.asarray(translation, dtype=np.float64)))
    if not np.isfinite(value):
        raise ValueError("Translation contains non-finite values")
    return value


def validate_baseline(baseline_mm: float) -> bool:
    return bool(np.isfinite(baseline_mm) and CONFIG.BASELINE_MIN_MM <= baseline_mm <= CONFIG.BASELINE_MAX_MM)


def can_start_3d(calibration_valid: bool) -> Tuple[bool, str]:
    if not (CONFIG.ENABLE_3D or CONFIG.ENABLE_DISPARITY):
        return False, "3D/disparity is disabled in config.py"
    if not calibration_valid:
        return False, "3D/disparity requires a valid saved stereo calibration"
    return True, ""


def _normalized(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centered = pts - pts.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale <= 1e-9:
        raise ValueError("Degenerate chessboard corners")
    return centered / scale


def orient_right_corners(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Select the physical right-corner order; reject indistinguishable choices."""
    left_points = ensure_corner_count(left).reshape(-1, 2)
    grid = ensure_corner_count(right).reshape(CONFIG.CHESSBOARD_ROWS, CONFIG.CHESSBOARD_COLS, 2)
    variants = (
        ("original", grid),
        ("rows_reversed", grid[::-1, :]),
        ("cols_reversed", grid[:, ::-1]),
        ("both_reversed", grid[::-1, ::-1]),
    )
    normalized_left = _normalized(left_points)
    scored = []
    for name, candidate in variants:
        points = candidate.reshape(-1, 2)
        score = float(np.sqrt(np.mean(np.sum((_normalized(points) - normalized_left) ** 2, axis=1))))
        scored.append((score, name, points))
    scored.sort(key=lambda item: item[0])
    best, _, points = scored[0]
    second = scored[1][0]
    if not np.isfinite(best):
        raise ValueError("Left/right chessboard geometry does not correspond")
    if CONFIG.ENFORCE_CAPTURE_QUALITY and best > 0.45:
        raise ValueError("Left/right chessboard geometry does not correspond")
    if CONFIG.ENFORCE_CAPTURE_QUALITY and abs(second - best) < 0.01:
        raise ValueError("Ambiguous right chessboard corner ordering")
    return points.reshape(-1, 1, 2).astype(np.float32)


def board_is_inside_frame(corners: np.ndarray, frame: np.ndarray, margin_px: float = 2.0) -> bool:
    pts = ensure_corner_count(corners).reshape(-1, 2)
    height, width = frame.shape[:2]
    return bool(
        np.all(pts[:, 0] >= margin_px)
        and np.all(pts[:, 1] >= margin_px)
        and np.all(pts[:, 0] <= width - 1 - margin_px)
        and np.all(pts[:, 1] <= height - 1 - margin_px)
    )


def outlier_indices(errors: Sequence[float]) -> List[int]:
    values = np.asarray(errors, dtype=np.float64)
    invalid = np.flatnonzero(~np.isfinite(values)).tolist()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return list(range(len(values)))
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    threshold = median + 2.5 * max(mad, 1e-6)
    flagged = np.flatnonzero((~np.isfinite(values)) | (values > threshold) | (values > CONFIG.MONO_RMS_FAILURE_PX))
    return sorted(set(invalid + flagged.tolist()))


class StabilityGate:
    def __init__(self) -> None:
        self._left: Deque[np.ndarray] = deque(maxlen=CONFIG.STABILITY_FRAME_COUNT)
        self._right: Deque[np.ndarray] = deque(maxlen=CONFIG.STABILITY_FRAME_COUNT)
        self.sync_ms = float("inf")
        self.max_motion_px = float("inf")
        self.mean_motion_px = float("inf")

    def observe(self, left: np.ndarray, right: np.ndarray, sync_ms: float) -> None:
        self.sync_ms = float(sync_ms)
        self._left.append(ensure_corner_count(left).reshape(-1, 2).copy())
        self._right.append(ensure_corner_count(right).reshape(-1, 2).copy())
        if len(self._left) < CONFIG.STABILITY_FRAME_COUNT:
            self.max_motion_px = float("inf")
            self.mean_motion_px = float("inf")
            return
        motions = []
        for sequence in (self._left, self._right):
            frames = list(sequence)
            for previous, current in zip(frames, frames[1:]):
                motions.extend(np.linalg.norm(current - previous, axis=1).tolist())
        self.max_motion_px = float(np.max(motions))
        self.mean_motion_px = float(np.mean(motions))

    def ready(self) -> Tuple[bool, str]:
        if self.sync_ms > CONFIG.MAX_SYNC_DIFFERENCE_MS:
            return False, f"SYNC ERROR {self.sync_ms:.1f} ms"
        if len(self._left) < CONFIG.STABILITY_FRAME_COUNT:
            return False, "HOLD BOARD STILL"
        if self.max_motion_px > CONFIG.MAX_CORNER_MOTION_PX:
            return False, "HOLD BOARD STILL"
        return True, ""


def draw_debug_corner_indices(frame: np.ndarray, corners: np.ndarray) -> None:
    if not CONFIG.DEBUG_CORNER_INDICES:
        return
    points = ensure_corner_count(corners).reshape(-1, 2)
    for index in (0, 5, 24, 29):
        point = tuple(np.round(points[index]).astype(int))
        cv2.putText(frame, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

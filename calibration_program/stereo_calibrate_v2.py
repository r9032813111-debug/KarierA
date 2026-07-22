import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from calibration_utils_v2 import (
    EXPECTED_CORNER_COUNT,
    StabilityGate,
    baseline_mm_from_t,
    board_is_inside_frame,
    draw_debug_corner_indices,
    ensure_corner_count,
    make_object_points,
    orient_right_corners,
    startup_validation,
    validate_baseline,
    validate_exact_resolution,
)
from config_v2 import CONFIG


BACKENDS = {
    "default": 0,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "vfw": cv2.CAP_VFW,
}

ROTATIONS = {
    "none": None,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


# =========================
# USER SETTINGS
# =========================
# 1) First run: python dual_camera_calibrate.py
# 2) Then run:  python stereo_calibrate.py
LEFT_CAMERA_INDEX = None if CONFIG.AUTO_DETECT_CAMERAS else CONFIG.LEFT_CAMERA_INDEX
RIGHT_CAMERA_INDEX = None if CONFIG.AUTO_DETECT_CAMERAS else CONFIG.RIGHT_CAMERA_INDEX
SCAN_CAMERA_INDEXES = 8

LEFT_CALIBRATION_YAML = "left_camera_calibration.yaml"
RIGHT_CALIBRATION_YAML = "right_camera_calibration.yaml"
STEREO_OUTPUT_YAML = "stereo_calibration.yaml"
STEREO_OUTPUT_JSON = "stereo_calibration.json"

CHESSBOARD_COLS = CONFIG.CHESSBOARD_COLS
CHESSBOARD_ROWS = CONFIG.CHESSBOARD_ROWS
SQUARE_SIZE_MM = CONFIG.SQUARE_SIZE_MM
MIN_STEREO_PAIRS = CONFIG.MIN_CALIBRATION_SAMPLES

EXPECTED_BASELINE_MIN_MM = CONFIG.BASELINE_MIN_MM
EXPECTED_BASELINE_MAX_MM = CONFIG.BASELINE_MAX_MM
MAX_ACCEPTABLE_STEREO_RMS = CONFIG.STEREO_RMS_FAILURE_PX
MAX_ACCEPTABLE_MONO_RMS_FOR_STEREO = CONFIG.MONO_RMS_FAILURE_PX
ALLOW_SAVE_BAD_CALIBRATION = not CONFIG.ENFORCE_FINAL_CALIBRATION_QUALITY

CAMERA_WIDTH = CONFIG.FRAME_WIDTH
CAMERA_HEIGHT = CONFIG.FRAME_HEIGHT
CAMERA_FPS = CONFIG.FPS
DISPLAY_SCALE = CONFIG.DISPLAY_SCALE
BACKEND = "dshow"
LEFT_ROTATE = "none"
RIGHT_ROTATE = "none"

USE_SB_DETECTOR = True
# Capture remains full-HD. Detection uses a smaller copy so the preview stays fluid.
DETECTION_SCALE = 0.55
DETECT_EVERY_N_FRAMES = 1
ROBUST_DETECT_EVERY_N_FRAMES = 1
ROBUST_DETECTION_SCALE = 0.72
DETECTION_CONTRAST_ENHANCE = True
DETECTION_CLAHE_CLIP_LIMIT = 3.0
DETECTION_CLAHE_TILE_GRID = 8
AUTO_DETECT_CHESSBOARD_SIZE = False
# These are INNER corner counts, not the number of printed squares.
# The first value is the expected board from the current setup.
CHESSBOARD_CANDIDATES = [
    (CONFIG.CHESSBOARD_COLS, CONFIG.CHESSBOARD_ROWS),
]
AUTO_CAPTURE = False
AUTO_CAPTURE_COOLDOWN_SECONDS = 1.2
AUTO_SAVE_WHEN_READY = True
BLACK_FRAME_MEAN_LIMIT = 8.0
BLACK_FRAME_STD_LIMIT = 4.0
OPENCV_WORKER_THREADS = max(1, min(CONFIG.CPU_THREADS, os.cpu_count() or CONFIG.CPU_THREADS))


class WorkspaceFiles:
    def __init__(self) -> None:
        self.workspace = Path.cwd().resolve()

    def path(self, value: str, must_exist: bool = False) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace / path
        resolved = path.resolve()
        if resolved.parent != self.workspace:
            raise ValueError(f"File must be directly inside this folder: {value}")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved.name}")
        return resolved


class MonoCalibration:
    def __init__(self, path: Path, payload: dict) -> None:
        self.path = path
        self.payload = payload
        self.camera_matrix = np.array(payload["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(payload["distortion_coefficients"], dtype=np.float64)
        self.rms = float(payload.get("rms_reprojection_error", 999.0))
        self.board_cols = None
        self.board_rows = None
        board = payload.get("chessboard_inner_corners")
        if isinstance(board, dict):
            self.board_cols = board.get("cols")
            self.board_rows = board.get("rows")
        resolution = payload.get("resolution") or payload.get("image_size")
        self.image_size = None
        if isinstance(resolution, dict):
            width = resolution.get("width")
            height = resolution.get("height")
            if isinstance(width, int) and isinstance(height, int):
                self.image_size = (width, height)


def load_mono_calibration(filename: str) -> MonoCalibration:
    path = WorkspaceFiles().path(filename, must_exist=True)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) if path.suffix.lower() in (".yaml", ".yml") else json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid calibration file: {path.name}")
    return MonoCalibration(path, payload)


def warn_bad_mono_calibration(calibration: MonoCalibration, expected_cols: int, expected_rows: int) -> None:
    if calibration.rms > 2.0:
        print(f"WARNING: {calibration.path.name} RMS is {calibration.rms:.3f}; this mono calibration is bad.")
    if calibration.board_cols is not None and calibration.board_rows is not None:
        if (calibration.board_cols, calibration.board_rows) != (expected_cols, expected_rows):
            print(
                f"WARNING: {calibration.path.name} was calibrated with "
                f"{calibration.board_cols}x{calibration.board_rows}, "
                f"but current board is {expected_cols}x{expected_rows}."
            )
            print("Re-run dual_camera_calibrate.py before stereo calibration.")


def mono_calibration_errors(calibration: MonoCalibration, expected_cols: int, expected_rows: int) -> List[str]:
    errors = []
    if calibration.rms > MAX_ACCEPTABLE_MONO_RMS_FOR_STEREO:
        errors.append(f"{calibration.path.name} RMS {calibration.rms:.3f} is too high")
    if calibration.board_cols is not None and calibration.board_rows is not None:
        if (calibration.board_cols, calibration.board_rows) != (expected_cols, expected_rows):
            errors.append(
                f"{calibration.path.name} board is {calibration.board_cols}x{calibration.board_rows}, "
                f"expected {expected_cols}x{expected_rows}"
            )
    return errors


class Camera:
    def __init__(self, index: int, width: int, height: int, backend_name: str, rotate: str) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.backend = BACKENDS[backend_name]
        self.rotate = rotate
        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = CAMERA_FPS
        self.autofocus_supported: Optional[bool] = None
        self.autofocus_enabled: Optional[bool] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.index, self.backend) if self.backend else cv2.VideoCapture(self.index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.index}")
        for prop, value, name in (
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"), "MJPG"),
            (cv2.CAP_PROP_FRAME_WIDTH, self.width, "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.height, "height"),
            (cv2.CAP_PROP_FPS, CAMERA_FPS, "fps"),
            (cv2.CAP_PROP_BUFFERSIZE, 1, "buffer size"),
            (cv2.CAP_PROP_AUTOFOCUS, int(CONFIG.CAMERA_AUTOFOCUS), "autofocus"),
            (cv2.CAP_PROP_FOCUS, CONFIG.CAMERA_FOCUS, "focus"),
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, "auto exposure"),
            (cv2.CAP_PROP_AUTO_WB, 1, "auto white balance"),
        ):
            if not self.cap.set(prop, value):
                print(f"Warning: camera {self.index} property '{name}' may not be supported")
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if (self.actual_width, self.actual_height) != (self.width, self.height):
            print(f"Warning: camera {self.index} requested {self.width}x{self.height}, reports {self.actual_width}x{self.actual_height}")

    def lock_autofocus(self) -> None:
        if not self.cap:
            return
        if not self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
            self.autofocus_supported = False
            self.autofocus_enabled = None
            print(f"Warning: camera {self.index} does not support autofocus control")
            return
        self.autofocus_supported = True
        self.autofocus_enabled = bool(self.cap.get(cv2.CAP_PROP_AUTOFOCUS) > 0.5)
        if self.autofocus_enabled:
            self.autofocus_supported = False
            self.autofocus_enabled = None
            print(f"Warning: camera {self.index} ignores autofocus lock; treating autofocus control as unsupported")
        else:
            print(f"Camera {self.index} autofocus: LOCKED")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap:
            return False, None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        code = ROTATIONS[self.rotate]
        if code is not None:
            frame = cv2.rotate(frame, code)
        return True, frame

    def grab(self) -> bool:
        return bool(self.cap and self.cap.grab())

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap:
            return False, None
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return False, None
        code = ROTATIONS[self.rotate]
        if code is not None:
            frame = cv2.rotate(frame, code)
        return True, frame

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


def frame_has_signal(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) > BLACK_FRAME_MEAN_LIMIT and float(gray.std()) > BLACK_FRAME_STD_LIMIT


def camera_can_read(index: int, backend_name: str) -> bool:
    backend = BACKENDS[backend_name]
    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    try:
        if not cap or not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None and frame_has_signal(frame):
                return True
        print(f"Camera index {index} opened but frames look black/empty; skipping")
        return False
    finally:
        if cap:
            cap.release()


def find_camera_indices(left: Optional[int], right: Optional[int], backend_name: str, scan_limit: int) -> Tuple[int, int]:
    if left is not None and right is not None:
        return left, right
    print(f"Scanning camera indexes 0..{scan_limit - 1} with backend {backend_name}...")
    found = []
    for index in range(scan_limit):
        if camera_can_read(index, backend_name):
            found.append(index)
            print(f"Found camera index {index}")
            if len(found) >= 2 and left is None and right is None:
                break
    if left is None and right is None:
        if len(found) < 2:
            raise RuntimeError(f"Need two cameras, found {len(found)}")
        return found[0], found[1]
    if left is None:
        for index in found:
            if index != right:
                return index, right
        raise RuntimeError("Could not auto-detect left camera")
    for index in found:
        if index != left:
            return left, index
    raise RuntimeError("Could not auto-detect right camera")


def read_camera_pair(left_cam: Camera, right_cam: Camera) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], float]:
    left_grabbed = left_cam.grab()
    left_timestamp = time.monotonic()
    right_grabbed = right_cam.grab()
    right_timestamp = time.monotonic()
    if left_grabbed and right_grabbed:
        ok_l, left = left_cam.retrieve()
        ok_r, right = right_cam.retrieve()
        if ok_l and ok_r and left is not None and right is not None:
            return True, left, right, abs(left_timestamp - right_timestamp) * 1000.0
    ok_l, left = left_cam.read()
    left_timestamp = time.monotonic()
    ok_r, right = right_cam.read()
    right_timestamp = time.monotonic()
    return bool(ok_l and ok_r and left is not None and right is not None), left, right, abs(left_timestamp - right_timestamp) * 1000.0


def resize_to_size(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) != size:
        raise ValueError(
            f"Calibration frames must be {size[0]}x{size[1]}; got {frame.shape[1]}x{frame.shape[0]}."
        )
    validate_exact_resolution(frame)
    return frame


def detect_pair(
    board: "Chessboard",
    left: np.ndarray,
    right: np.ndarray,
    detection_scale: float,
    robust: bool,
) -> Tuple[bool, Optional[np.ndarray], float, bool, Optional[np.ndarray], float]:
    found_l, corners_l, blur_l = board.detect(left, detection_scale, robust=robust)
    found_r, corners_r, blur_r = board.detect(right, detection_scale, robust=robust)
    return found_l, corners_l, blur_l, found_r, corners_r, blur_r


class Chessboard:
    def __init__(self, cols: int, rows: int, square_size: float, use_sb: bool) -> None:
        self.cols = cols
        self.rows = rows
        self.pattern = (cols, rows)
        self.square_size = square_size
        self.use_sb = use_sb
        self.object_points = self._make_object_points(cols, rows)
        self.candidates = self._make_candidates(cols, rows)
        self.clahe = cv2.createCLAHE(
            clipLimit=DETECTION_CLAHE_CLIP_LIMIT,
            tileGridSize=(DETECTION_CLAHE_TILE_GRID, DETECTION_CLAHE_TILE_GRID),
        ) if DETECTION_CONTRAST_ENHANCE else None

    def _make_object_points(self, cols: int, rows: int) -> np.ndarray:
        if (cols, rows) != (CONFIG.CHESSBOARD_COLS, CONFIG.CHESSBOARD_ROWS):
            raise ValueError(f"Only the configured {CONFIG.CHESSBOARD_COLS}x{CONFIG.CHESSBOARD_ROWS} inner-corner board is valid")
        if self.square_size != CONFIG.SQUARE_SIZE_MM:
            raise ValueError(f"Square size must remain {CONFIG.SQUARE_SIZE_MM} mm for this hardware setup")
        return make_object_points()

    def _make_candidates(self, cols: int, rows: int) -> List[Tuple[int, int]]:
        candidates = [(cols, rows)]
        if AUTO_DETECT_CHESSBOARD_SIZE:
            candidates.extend(CHESSBOARD_CANDIDATES)
        return list(dict.fromkeys(candidates))

    def _set_pattern(self, pattern: Tuple[int, int]) -> None:
        if pattern == self.pattern:
            return
        self.pattern = pattern
        self.cols, self.rows = pattern
        self.object_points = self._make_object_points(self.cols, self.rows)
        self.candidates = [pattern] + [candidate for candidate in self.candidates if candidate != pattern]
        print(f"Detected chessboard inner corners: {self.cols}x{self.rows}")

    def _prepare_gray(self, frame: np.ndarray, scale: float, enhance: bool) -> Tuple[np.ndarray, float]:
        work = frame
        if scale < 1:
            work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        if enhance:
            assert self.clahe is not None
            gray = self.clahe.apply(gray)
        return gray, scale

    def _try_pattern(
        self,
        gray: np.ndarray,
        pattern: Tuple[int, int],
        scale: float,
        robust: bool,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        if self.use_sb and hasattr(cv2, "findChessboardCornersSB"):
            sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            found, corners = cv2.findChessboardCornersSB(gray, pattern, sb_flags)
            if found and corners is not None:
                if scale < 1:
                    corners = corners / scale
                return True, corners.astype(np.float32)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if found and corners is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if scale < 1:
                corners = corners / scale
            return True, corners.astype(np.float32)
        return False, None

    def _refine_full_resolution(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        corners = corners.astype(np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.0005)
        return cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria).astype(np.float32)

    def detect(self, frame: np.ndarray, detection_scale: float, robust: bool = False) -> Tuple[bool, Optional[np.ndarray], float]:
        if detection_scale <= 0 or detection_scale > 1:
            detection_scale = 1.0
        gray_base = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray_base, cv2.CV_64F).var())
        # Match the reference project's reliable path first: full-size grayscale,
        # then sub-pixel refinement. Enhancement is only a fallback.
        passes = [(detection_scale, False, False), (1.0, False, True)]
        if robust:
            robust_scale = min(1.0, max(detection_scale, ROBUST_DETECTION_SCALE))
            passes.extend(
                [
                    (robust_scale, True, True),
                    (robust_scale, False, True),
                    (1.0, True, True),
                ]
            )

        for scale, enhance, is_robust in passes:
            gray, actual_scale = self._prepare_gray(frame, scale, enhance)
            if enhance:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
            for pattern in self.candidates:
                found, corners = self._try_pattern(gray, pattern, actual_scale, is_robust)
                if found and corners is not None:
                    self._set_pattern(pattern)
                    corners = self._refine_full_resolution(gray_base, corners)
                    return True, corners, blur
        return False, None, blur

    def draw(self, frame: np.ndarray, found: bool, corners: Optional[np.ndarray]) -> None:
        if found and corners is not None:
            cv2.drawChessboardCorners(frame, self.pattern, corners, found)


class StereoSession:
    def __init__(self, board: Chessboard, min_pairs: int) -> None:
        self.board = board
        self.min_pairs = min_pairs
        self.object_points: List[np.ndarray] = []
        self.left_points: List[np.ndarray] = []
        self.right_points: List[np.ndarray] = []
        self.signatures: List[np.ndarray] = []
        self.status = "No stereo samples yet"
        self.valid_payload: Optional[dict] = None

    def reset(self) -> None:
        self.object_points.clear()
        self.left_points.clear()
        self.right_points.clear()
        self.signatures.clear()
        self.status = "Session reset"

    def add(self, left: np.ndarray, right: np.ndarray, image_size: Tuple[int, int]) -> bool:
        if len(left.reshape(-1, 2)) != EXPECTED_CORNER_COUNT or len(right.reshape(-1, 2)) != EXPECTED_CORNER_COUNT:
            self.status = "Detected board sizes differ; pair not added"
            print("Warning: detected board sizes differ between cameras; pair not added")
            return False
        try:
            left = ensure_corner_count(left)
            right = orient_right_corners(left, right)
        except ValueError as exc:
            self.status = str(exc)
            print(f"Warning: {self.status}; pair not added")
            return False
        signature = self._signature(left, right, image_size)
        if CONFIG.ENFORCE_CAPTURE_QUALITY and self.signatures:
            nearest = min(float(np.linalg.norm(signature - old)) for old in self.signatures)
            if nearest < 0.08:
                self.status = "Too similar; move/tilt chessboard"
                print("Warning: stereo pair is too similar; not added")
                return False
        self.object_points.append(self.board.object_points.copy())
        self.left_points.append(left.reshape(-1, 1, 2).astype(np.float32))
        self.right_points.append(right.reshape(-1, 1, 2).astype(np.float32))
        self.signatures.append(signature)
        self.board.candidates = [self.board.pattern]
        self.status = f"Captured stereo pair {len(self.left_points)}"
        return True

    def _signature(self, left: np.ndarray, right: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        return np.concatenate([self._single_signature(left, image_size), self._single_signature(right, image_size)])

    def _single_signature(self, corners: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        pts = corners.reshape(-1, 2)
        scale = np.array([image_size[0], image_size[1]], dtype=np.float32)
        return np.concatenate([pts.mean(axis=0) / scale, (pts.max(axis=0) - pts.min(axis=0)) / scale]).astype(np.float32)

    def calibrate(self, left_cal: MonoCalibration, right_cal: MonoCalibration, image_size: Tuple[int, int], metadata: dict) -> dict:
        if len(self.left_points) < self.min_pairs:
            raise RuntimeError(f"Not enough stereo pairs: {len(self.left_points)}/{self.min_pairs}")
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        rms, k1, d1, k2, d2, r, t, e, f = cv2.stereoCalibrate(
            self.object_points,
            self.left_points,
            self.right_points,
            left_cal.camera_matrix,
            left_cal.dist_coeffs,
            right_cal.camera_matrix,
            right_cal.dist_coeffs,
            image_size,
            criteria=criteria,
            flags=flags,
        )
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(k1, d1, k2, d2, image_size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        epipolar_errors = self._symmetric_epipolar_errors(f)
        vertical_errors = self._rectification_vertical_errors(k1, d1, k2, d2, r1, r2, p1, p2)
        return {
            **metadata,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "calibration_mode": "stereo_fixed_intrinsics",
            "rms_reprojection_error": float(rms),
            "number_of_pairs": len(self.left_points),
            "flags": int(flags),
            "left_camera_matrix": k1.tolist(),
            "left_distortion_coefficients": d1.tolist(),
            "right_camera_matrix": k2.tolist(),
            "right_distortion_coefficients": d2.tolist(),
            "rotation": r.tolist(),
            "translation_mm": t.tolist(),
            "essential_matrix": e.tolist(),
            "fundamental_matrix": f.tolist(),
            "rectification_left": r1.tolist(),
            "rectification_right": r2.tolist(),
            "projection_left": p1.tolist(),
            "projection_right": p2.tolist(),
            "q_matrix": q.tolist(),
            "baseline_mm": baseline_mm_from_t(t),
            "expected_baseline_mm": CONFIG.EXPECTED_BASELINE_MM,
            "baseline_error_mm": baseline_mm_from_t(t) - CONFIG.EXPECTED_BASELINE_MM,
            "symmetric_epipolar_error_per_pair_px": epipolar_errors,
            "symmetric_epipolar_error_median_px": float(np.median(epipolar_errors)),
            "symmetric_epipolar_error_max_px": float(np.max(epipolar_errors)),
            "rectified_vertical_error_median_px": vertical_errors["median"],
            "rectified_vertical_error_p95_px": vertical_errors["p95"],
            "rectified_vertical_error_max_px": vertical_errors["max"],
            "roi_left": list(map(int, roi1)),
            "roi_right": list(map(int, roi2)),
        }

    def _symmetric_epipolar_errors(self, fundamental: np.ndarray) -> List[float]:
        # Vectorize all pairs into one native NumPy operation instead of a
        # Python loop. BLAS/OpenCV can use the complete logical-CPU budget.
        f = np.asarray(fundamental, dtype=np.float64)
        left = np.stack([points.reshape(-1, 2) for points in self.left_points]).astype(np.float64)
        right = np.stack([points.reshape(-1, 2) for points in self.right_points]).astype(np.float64)
        ones = np.ones((*left.shape[:2], 1), dtype=np.float64)
        x1 = np.concatenate((left, ones), axis=2)
        x2 = np.concatenate((right, ones), axis=2)
        l2 = x1 @ f.T
        l1 = x2 @ f
        numer = np.sum(x2 * l2, axis=2)
        d1 = np.abs(numer) / np.maximum(np.hypot(l1[:, :, 0], l1[:, :, 1]), 1e-12)
        d2 = np.abs(numer) / np.maximum(np.hypot(l2[:, :, 0], l2[:, :, 1]), 1e-12)
        return np.sqrt(np.mean((d1 * d1 + d2 * d2) * 0.5, axis=1)).astype(float).tolist()

    def _rectification_vertical_errors(
        self,
        k1: np.ndarray,
        d1: np.ndarray,
        k2: np.ndarray,
        d2: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> dict:
        shape = np.asarray(self.left_points).shape
        left = np.concatenate(self.left_points, axis=0)
        right = np.concatenate(self.right_points, axis=0)
        rect_left = cv2.undistortPoints(left, k1, d1, R=r1, P=p1).reshape(shape[0], -1, 2)
        rect_right = cv2.undistortPoints(right, k2, d2, R=r2, P=p2).reshape(shape[0], -1, 2)
        errors = np.abs(rect_left[:, :, 1] - rect_right[:, :, 1]).reshape(-1).astype(np.float64)
        return {"median": float(np.median(errors)), "p95": float(np.percentile(errors, 95)), "max": float(np.max(errors))}


def create_rectification_maps(payload: dict, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1 = np.asarray(payload["left_camera_matrix"], dtype=np.float64)
    d1 = np.asarray(payload["left_distortion_coefficients"], dtype=np.float64)
    k2 = np.asarray(payload["right_camera_matrix"], dtype=np.float64)
    d2 = np.asarray(payload["right_distortion_coefficients"], dtype=np.float64)
    r1 = np.asarray(payload["rectification_left"], dtype=np.float64)
    r2 = np.asarray(payload["rectification_right"], dtype=np.float64)
    p1 = np.asarray(payload["projection_left"], dtype=np.float64)
    p2 = np.asarray(payload["projection_right"], dtype=np.float64)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(k1, d1, r1, p1, image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(k2, d2, r2, p2, image_size, cv2.CV_16SC2)
    return left_map1, left_map2, right_map1, right_map2


def draw_overlay(frame: np.ndarray, lines: Sequence[str]) -> None:
    x, y, dy = 16, 28, 24
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] for line in lines)
    cv2.rectangle(frame, (8, 8), (width + 32, 24 + dy * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def save_yaml_json(payload: dict, yaml_path: Path, json_path: Path) -> None:
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_quality(payload: dict) -> None:
    print("")
    print("=== Stereo calibration quality ===")
    print(f"Pairs used: {payload['number_of_pairs']}")
    print(f"Stereo RMS: {payload['rms_reprojection_error']:.6f}")
    print(f"Baseline: {payload['baseline_mm']:.2f} mm")
    print(f"Baseline error from {CONFIG.EXPECTED_BASELINE_MM:.1f} mm: {payload['baseline_error_mm']:+.2f} mm")
    print(f"Expected baseline: {EXPECTED_BASELINE_MIN_MM:.1f}-{EXPECTED_BASELINE_MAX_MM:.1f} mm")
    print(f"Epipolar error median/max: {payload['symmetric_epipolar_error_median_px']:.3f}/{payload['symmetric_epipolar_error_max_px']:.3f} px")
    print(f"Rectified vertical error median/p95/max: {payload['rectified_vertical_error_median_px']:.3f}/{payload['rectified_vertical_error_p95_px']:.3f}/{payload['rectified_vertical_error_max_px']:.3f} px")
    if payload["rectified_vertical_error_p95_px"] > 1.0:
        print("WARNING: rectified vertical error p95 is above 1.0 px")
    print("==================================")
    print("")


def calibration_quality_errors(payload: dict) -> List[str]:
    errors = []
    if payload["rms_reprojection_error"] > MAX_ACCEPTABLE_STEREO_RMS:
        errors.append(f"Stereo RMS {payload['rms_reprojection_error']:.3f} is too high")
    if not validate_baseline(float(payload["baseline_mm"])):
        errors.append(f"Baseline {payload['baseline_mm']:.2f} mm is outside expected range")
    if payload["rectified_vertical_error_max_px"] > 2.0:
        errors.append(f"Rectified vertical error {payload['rectified_vertical_error_max_px']:.3f} px exceeds 2.0 px")
    for key in ("left_camera_matrix", "left_distortion_coefficients", "right_camera_matrix", "right_distortion_coefficients", "rotation", "translation_mm", "projection_left", "projection_right", "q_matrix"):
        if not np.isfinite(np.asarray(payload[key], dtype=np.float64)).all():
            errors.append(f"{key} contains non-finite values")
    return errors


def try_save_stereo_calibration(
    session: StereoSession,
    left_cal: MonoCalibration,
    right_cal: MonoCalibration,
    image_size: Tuple[int, int],
    metadata: dict,
    out_yaml: Path,
    out_json: Path,
) -> bool:
    payload = session.calibrate(left_cal, right_cal, image_size, metadata)
    print_quality(payload)
    errors = calibration_quality_errors(payload)
    if errors and not ALLOW_SAVE_BAD_CALIBRATION:
        print("Stereo quality check failed; files were NOT saved.")
        for error in errors:
            print(f"  {error}")
        if not validate_baseline(float(payload["baseline_mm"])):
            print("Likely causes: incorrect square size, reversed corner order, moving board, unsynchronized frames, wrong resolution, or outlier samples.")
        print("Keep capturing more varied stereo pairs, then press Space again.")
        session.status = "Stereo quality check failed"
        return False
    if errors:
        print("WARNING: saving despite calibration quality errors because ENFORCE_FINAL_CALIBRATION_QUALITY=False")
        for error in errors:
            print(f"  {error}")
    save_yaml_json(payload, out_yaml, out_json)
    session.valid_payload = payload
    print(f"Saved: {out_yaml.name}, {out_json.name}")
    session.status = "Stereo calibration saved"
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo-only calibration using fixed left/right intrinsics")
    parser.add_argument("--left-camera", type=int, default=LEFT_CAMERA_INDEX)
    parser.add_argument("--right-camera", type=int, default=RIGHT_CAMERA_INDEX)
    parser.add_argument("--scan-cameras", type=int, default=SCAN_CAMERA_INDEXES)
    parser.add_argument("--left-calibration", default=LEFT_CALIBRATION_YAML)
    parser.add_argument("--right-calibration", default=RIGHT_CALIBRATION_YAML)
    parser.add_argument("--output-yaml", default=STEREO_OUTPUT_YAML)
    parser.add_argument("--output-json", default=STEREO_OUTPUT_JSON)
    parser.add_argument("--cols", type=int, default=CHESSBOARD_COLS)
    parser.add_argument("--rows", type=int, default=CHESSBOARD_ROWS)
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE_MM)
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--display-scale", type=float, default=DISPLAY_SCALE)
    parser.add_argument("--min-pairs", type=int, default=MIN_STEREO_PAIRS)
    parser.add_argument("--backend", choices=BACKENDS.keys(), default=BACKEND)
    parser.add_argument("--left-rotate", choices=ROTATIONS.keys(), default=LEFT_ROTATE)
    parser.add_argument("--right-rotate", choices=ROTATIONS.keys(), default=RIGHT_ROTATE)
    parser.add_argument("--use-sb-detector", action=argparse.BooleanOptionalAction, default=USE_SB_DETECTOR)
    parser.add_argument("--auto-capture", action=argparse.BooleanOptionalAction, default=AUTO_CAPTURE)
    parser.add_argument("--auto-save", action=argparse.BooleanOptionalAction, default=AUTO_SAVE_WHEN_READY)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.square_size <= 0:
        return "--square-size must be greater than 0"
    if (args.cols, args.rows) != (CONFIG.CHESSBOARD_COLS, CONFIG.CHESSBOARD_ROWS):
        return f"stereo calibration requires --cols {CONFIG.CHESSBOARD_COLS} --rows {CONFIG.CHESSBOARD_ROWS} (inner corners)"
    if args.square_size != CONFIG.SQUARE_SIZE_MM:
        return "stereo calibration requires --square-size 24.2 (millimetres)"
    if (args.width, args.height) != (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT):
        return "stereo calibration requires --width 1920 --height 1080"
    if args.min_pairs < CONFIG.MIN_CALIBRATION_SAMPLES:
        return f"--min-pairs must be at least {CONFIG.MIN_CALIBRATION_SAMPLES}"
    return None


def main() -> int:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(OPENCV_WORKER_THREADS)
    args = parse_args()
    error = validate_args(args)
    if error:
        print(f"Error: {error}")
        return 2
    try:
        startup_validation()
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2
    files = WorkspaceFiles()
    try:
        left_cal = load_mono_calibration(args.left_calibration)
        right_cal = load_mono_calibration(args.right_calibration)
        out_yaml = files.path(args.output_yaml)
        out_json = files.path(args.output_json)
    except Exception as exc:
        print(f"Error: {exc}")
        print("Run dual_camera_calibrate.py first.")
        return 2

    try:
        left_index, right_index = find_camera_indices(args.left_camera, args.right_camera, args.backend, args.scan_cameras)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Using left camera index {left_index}, right camera index {right_index}")
    print(f"Loaded fixed intrinsics: {left_cal.path.name}, {right_cal.path.name}")
    warn_bad_mono_calibration(left_cal, args.cols, args.rows)
    warn_bad_mono_calibration(right_cal, args.cols, args.rows)
    mono_errors = []
    mono_errors.extend(mono_calibration_errors(left_cal, args.cols, args.rows))
    mono_errors.extend(mono_calibration_errors(right_cal, args.cols, args.rows))
    if mono_errors:
        print("Error: stereo calibration cannot use these mono calibration files.")
        for message in mono_errors:
            print(f"  {message}")
        print("Re-run dual_camera_calibrate.py and save only when Left/Right RMS is below 2.0.")
        print("For a more reliable board, use charuco_stereo_calibrate.py instead.")
        return 2

    image_size = (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT)
    if left_cal.image_size != image_size or right_cal.image_size != image_size:
        print("Error: both mono calibrations must have verified 1920x1080 resolution.")
        return 2

    left_cam = Camera(left_index, args.width, args.height, args.backend, args.left_rotate)
    right_cam = Camera(right_index, args.width, args.height, args.backend, args.right_rotate)
    board = Chessboard(args.cols, args.rows, args.square_size, args.use_sb_detector)
    session = StereoSession(board, args.min_pairs)

    try:
        left_cam.open()
        right_cam.open()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        left_cam.release()
        right_cam.release()
        return 1

    actual_sizes = ((left_cam.actual_width, left_cam.actual_height), (right_cam.actual_width, right_cam.actual_height))
    if actual_sizes[0] != image_size or actual_sizes[1] != image_size:
        print(f"Error: calibration requires both cameras at 1920x1080; got left {actual_sizes[0]}, right {actual_sizes[1]}")
        left_cam.release()
        right_cam.release()
        return 1

    left_cam.lock_autofocus()
    right_cam.lock_autofocus()
    if left_cam.autofocus_enabled or right_cam.autofocus_enabled:
        print("Error: autofocus is still enabled; stereo samples are rejected.")
        left_cam.release()
        right_cam.release()
        return 1

    metadata = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "backend": args.backend,
        "left_camera_index": left_index,
        "right_camera_index": right_index,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "resolution": {"width": image_size[0], "height": image_size[1]},
        "requested_fps": CAMERA_FPS,
        "left_actual_fps": left_cam.actual_fps,
        "right_actual_fps": right_cam.actual_fps,
        "left_calibration_file": left_cal.path.name,
        "right_calibration_file": right_cal.path.name,
        "chessboard_inner_corners": {"cols": board.cols, "rows": board.rows},
        "square_size_mm": args.square_size,
    }

    print("Stereo-only calibration started.")
    print(f"Expected baseline: {EXPECTED_BASELINE_MIN_MM:.1f}-{EXPECTED_BASELINE_MAX_MM:.1f} mm")
    print("C capture | Space calibrate/save | R reset | V raw/rectified | L epipolar lines | Q/Esc quit")
    print(f"Auto-save after {args.min_pairs} pairs: {'ON' if args.auto_save else 'OFF'}")
    print(f"OpenCV worker threads: {OPENCV_WORKER_THREADS}")

    found_l = found_r = False
    corners_l = corners_r = None
    blur_l = blur_r = 0.0
    sync_ms = float("inf")
    stability_gate = StabilityGate()
    frame_index = 0
    last_auto_tick = 0
    last_auto_save_pair_count = 0
    failed_reads = 0
    detection_future: Optional[Future] = None
    detector_executor = ThreadPoolExecutor(max_workers=2)
    show_rectified = False
    show_epipolar_lines = False
    rectification_maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    try:
        while True:
            ok, left, right, sync_ms = read_camera_pair(left_cam, right_cam)
            if not ok or left is None or right is None:
                failed_reads += 1
                if failed_reads == 1 or failed_reads % 30 == 0:
                    print(f"Warning: failed to read one of the cameras ({failed_reads} times)")
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue
            failed_reads = 0
            try:
                left = resize_to_size(left, image_size)
                right = resize_to_size(right, image_size)
            except ValueError as exc:
                print(f"Error: {exc}")
                session.status = "Invalid camera resolution"
                break

            if detection_future is not None and detection_future.done():
                try:
                    found_l, corners_l, blur_l, found_r, corners_r, blur_r = detection_future.result()
                    if found_l and found_r and corners_l is not None and corners_r is not None:
                        stability_gate.observe(corners_l, corners_r, sync_ms)
                except Exception as exc:
                    print(f"Warning: background detector failed: {exc}")
                    found_l = found_r = False
                    corners_l = corners_r = None
                detection_future = None

            # Never run the heavy detector on the UI thread.
            if frame_index % DETECT_EVERY_N_FRAMES == 0 and detection_future is None:
                robust_detect = frame_index % ROBUST_DETECT_EVERY_N_FRAMES == 0
                detection_future = detector_executor.submit(
                    detect_pair,
                    board,
                    left.copy(),
                    right.copy(),
                    DETECTION_SCALE,
                    robust_detect,
                )
            frame_index += 1

            if show_rectified and rectification_maps is not None:
                left_map1, left_map2, right_map1, right_map2 = rectification_maps
                view_l = cv2.remap(left, left_map1, left_map2, cv2.INTER_LINEAR)
                view_r = cv2.remap(right, right_map1, right_map2, cv2.INTER_LINEAR)
            else:
                view_l = left.copy()
                view_r = right.copy()
            if not show_rectified:
                board.draw(view_l, found_l, corners_l)
                board.draw(view_r, found_r, corners_r)
                if found_l and corners_l is not None:
                    draw_debug_corner_indices(view_l, corners_l)
                if found_r and corners_r is not None:
                    draw_debug_corner_indices(view_r, corners_r)
            if show_rectified and show_epipolar_lines:
                for y in range(0, image_size[1], 80):
                    cv2.line(view_l, (0, y), (image_size[0] - 1, y), (0, 255, 0), 1)
                    cv2.line(view_r, (0, y), (image_size[0] - 1, y), (0, 255, 0), 1)
            draw_overlay(view_l, [f"LEFT cam {left_index}: {'FOUND' if found_l else 'not found'}", f"Pairs: {len(session.left_points)}/{args.min_pairs}", f"Sync: {sync_ms:.1f} ms", f"AF: {'ON' if left_cam.autofocus_enabled else 'LOCKED/unsupported'}", session.status])
            draw_overlay(view_r, [f"RIGHT cam {right_index}: {'FOUND' if found_r else 'not found'}", f"Board: {board.cols}x{board.rows} inner corners", f"Sharpness: {blur_l:.0f}/{blur_r:.0f}", f"AF: {'ON' if right_cam.autofocus_enabled else 'LOCKED/unsupported'}", "C capture | Space save | V rectified | L lines | Q quit"])
            if args.display_scale < 1:
                view_l = cv2.resize(view_l, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
                view_r = cv2.resize(view_r, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
            combined = np.hstack([view_l, view_r])
            cv2.imshow("Stereo Calibration Fixed Intrinsics", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("v"), ord("V")):
                if session.valid_payload is None:
                    session.status = "Rectified view requires valid calibration"
                else:
                    if rectification_maps is None:
                        rectification_maps = create_rectification_maps(session.valid_payload, image_size)
                    show_rectified = not show_rectified
                    session.status = f"Rectified view {'ON' if show_rectified else 'OFF'}"
            elif key in (ord("l"), ord("L")):
                if session.valid_payload is None:
                    session.status = "Epipolar lines require valid calibration"
                else:
                    show_epipolar_lines = not show_epipolar_lines
                    session.status = f"Epipolar lines {'ON' if show_epipolar_lines else 'OFF'}"
            if key in (ord("a"), ord("A")):
                args.auto_capture = not args.auto_capture
                session.status = f"Auto capture {'ON' if args.auto_capture else 'OFF'}"
            elif key in (ord("r"), ord("R")):
                session.reset()
                last_auto_save_pair_count = 0
            elif key in (ord("c"), ord("C")):
                if detection_future is not None:
                    try:
                        found_l, corners_l, blur_l, found_r, corners_r, blur_r = detection_future.result()
                    except Exception as exc:
                        print(f"Warning: background detector failed: {exc}")
                    detection_future = None
                capture_gate = StabilityGate()
                capture_failed = ""
                required_checks = CONFIG.STABILITY_FRAME_COUNT if CONFIG.ENFORCE_CAPTURE_QUALITY else 1
                for check_index in range(required_checks):
                    if check_index:
                        ok, left, right, sync_ms = read_camera_pair(left_cam, right_cam)
                        if not ok or left is None or right is None:
                            capture_failed = "Could not read a stereo frame pair"
                            break
                    found_l, corners_l, blur_l = board.detect(left, DETECTION_SCALE, robust=True)
                    found_r, corners_r, blur_r = board.detect(right, DETECTION_SCALE, robust=True)
                    if not found_l or not found_r or corners_l is None or corners_r is None:
                        capture_failed = "Chessboard must be found in both cameras"
                        break
                    capture_gate.observe(corners_l, corners_r, sync_ms)
                if capture_failed:
                    session.status = capture_failed
                else:
                    stable, reason = capture_gate.ready()
                    quality_ok = (
                        not CONFIG.ENFORCE_CAPTURE_QUALITY
                        or (
                            (not CONFIG.REQUIRE_BLUR_CHECK or (blur_l >= CONFIG.MIN_BLUR_VARIANCE and blur_r >= CONFIG.MIN_BLUR_VARIANCE))
                            and board_is_inside_frame(corners_l, left)
                            and board_is_inside_frame(corners_r, right)
                            and stable
                        )
                    )
                    if not quality_ok:
                        session.status = reason or "Capture quality check failed"
                    else:
                        stability_gate = capture_gate
                        session.add(corners_l, corners_r, image_size)
                print(session.status)
            elif key == 32:
                try:
                    print(f"Starting stereo calibration with {len(session.left_points)}/{args.min_pairs} RAM pairs...", flush=True)
                    if len(session.left_points) < args.min_pairs:
                        session.status = f"Need {args.min_pairs - len(session.left_points)} more pair(s) before calibration"
                        print(session.status, flush=True)
                        continue
                    metadata["chessboard_inner_corners"] = {"cols": board.cols, "rows": board.rows}
                    saved = try_save_stereo_calibration(session, left_cal, right_cal, image_size, metadata, out_yaml, out_json)
                    print(f"Stereo calibration {'saved' if saved else 'not saved'}.", flush=True)
                except Exception as exc:
                    print(f"Error: stereo calibration failed: {type(exc).__name__}: {exc}", flush=True)
                    session.status = "Stereo calibration failed"

            if args.auto_capture and found_l and found_r and corners_l is not None and corners_r is not None:
                now = cv2.getTickCount()
                cooldown = AUTO_CAPTURE_COOLDOWN_SECONDS * cv2.getTickFrequency()
                if last_auto_tick == 0 or now - last_auto_tick >= cooldown:
                    stable, _ = stability_gate.ready()
                    if (
                        stable
                        and (not CONFIG.REQUIRE_BLUR_CHECK or (blur_l >= CONFIG.MIN_BLUR_VARIANCE and blur_r >= CONFIG.MIN_BLUR_VARIANCE))
                        and board_is_inside_frame(corners_l, left)
                        and board_is_inside_frame(corners_r, right)
                        and session.add(corners_l, corners_r, image_size)
                    ):
                        last_auto_tick = now
                        print(f"Auto captured stereo pair {len(session.left_points)}")
            if args.auto_save and len(session.left_points) >= args.min_pairs and len(session.left_points) != last_auto_save_pair_count:
                try:
                    metadata["chessboard_inner_corners"] = {"cols": board.cols, "rows": board.rows}
                    print(f"Auto-save check with {len(session.left_points)} stereo pairs...")
                    saved = try_save_stereo_calibration(session, left_cal, right_cal, image_size, metadata, out_yaml, out_json)
                    last_auto_save_pair_count = len(session.left_points)
                    if saved:
                        args.auto_save = False
                        print("Stereo calibration file is ready. You can quit with Q and run stereo_depth.py.")
                except Exception as exc:
                    print(f"Error: stereo auto-save failed: {exc}")
                    session.status = "Stereo auto-save failed"
                    last_auto_save_pair_count = len(session.left_points)
    finally:
        detector_executor.shutdown(wait=False, cancel_futures=True)
        left_cam.release()
        right_cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

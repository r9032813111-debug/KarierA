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
    board_is_inside_frame,
    draw_debug_corner_indices,
    ensure_corner_count,
    make_object_points,
    orient_right_corners,
    outlier_indices,
    startup_validation,
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


# User-editable camera and calibration values are in config.py.
LEFT_CAMERA_INDEX = None if CONFIG.AUTO_DETECT_CAMERAS else CONFIG.LEFT_CAMERA_INDEX
RIGHT_CAMERA_INDEX = None if CONFIG.AUTO_DETECT_CAMERAS else CONFIG.RIGHT_CAMERA_INDEX
SCAN_CAMERA_INDEXES = 8

CHESSBOARD_COLS = CONFIG.CHESSBOARD_COLS
CHESSBOARD_ROWS = CONFIG.CHESSBOARD_ROWS
SQUARE_SIZE_MM = CONFIG.SQUARE_SIZE_MM
MIN_STEREO_PAIRS = CONFIG.TARGET_CALIBRATION_SAMPLES
MIN_GOOD_MONO_PAIRS = CONFIG.MIN_CALIBRATION_SAMPLES
MAX_ACCEPTABLE_STEREO_RMS = CONFIG.STEREO_RMS_FAILURE_PX
MAX_MONO_FILTER_PASSES = 4
MAX_ASPECT_MISMATCH = 0.35
MAX_ROW_COL_ANGLE_ERROR_DEG = 35.0
ALLOW_SAVE_BAD_CALIBRATION = not CONFIG.ENFORCE_FINAL_CALIBRATION_QUALITY

CAMERA_WIDTH = CONFIG.FRAME_WIDTH
CAMERA_HEIGHT = CONFIG.FRAME_HEIGHT
CAMERA_FPS = CONFIG.FPS
AUTO_RETRY_720P = False
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
CHESSBOARD_CANDIDATES = [
    (CONFIG.CHESSBOARD_COLS, CONFIG.CHESSBOARD_ROWS),
]
AUTO_CAPTURE = False
AUTO_CAPTURE_COOLDOWN_SECONDS = 1.2
BLACK_FRAME_MEAN_LIMIT = 8.0
BLACK_FRAME_STD_LIMIT = 4.0
PAIR_READ_WARMUP_ATTEMPTS = 30
# Use the complete logical-CPU budget requested by this project. During the
# two independent mono solves it is split evenly, so left and right execute at
# the same time without oversubscribing the machine (2 jobs x 8 = 16).
OPENCV_WORKER_THREADS = max(1, min(CONFIG.CPU_THREADS, os.cpu_count() or CONFIG.CPU_THREADS))
MONO_PARALLEL_JOBS = 2
MONO_THREADS_PER_JOB = max(1, OPENCV_WORKER_THREADS // MONO_PARALLEL_JOBS)

LEFT_OUTPUT_YAML = "left_camera_calibration.yaml"
LEFT_OUTPUT_JSON = "left_camera_calibration.json"
RIGHT_OUTPUT_YAML = "right_camera_calibration.yaml"
RIGHT_OUTPUT_JSON = "right_camera_calibration.json"


class WorkspaceFiles:
    def __init__(self) -> None:
        self.workspace = Path.cwd().resolve()

    def path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace / path
        resolved = path.resolve()
        if resolved.parent != self.workspace:
            raise ValueError(f"Output file must be directly inside this folder: {value}")
        return resolved


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
        self._set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"), "MJPG")
        self._set(cv2.CAP_PROP_FRAME_WIDTH, self.width, "width")
        self._set(cv2.CAP_PROP_FRAME_HEIGHT, self.height, "height")
        self._set(cv2.CAP_PROP_FPS, CAMERA_FPS, "fps")
        self._set(cv2.CAP_PROP_BUFFERSIZE, 1, "buffer size")
        self._set(cv2.CAP_PROP_AUTOFOCUS, int(CONFIG.CAMERA_AUTOFOCUS), "autofocus")
        self._set(cv2.CAP_PROP_FOCUS, CONFIG.CAMERA_FOCUS, "focus")
        self._set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75, "auto exposure")
        self._set(cv2.CAP_PROP_AUTO_WB, 1, "auto white balance")
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_width = actual_w
        self.actual_height = actual_h
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if actual_w != self.width or actual_h != self.height:
            print(f"Warning: camera {self.index} requested {self.width}x{self.height}, reports {actual_w}x{actual_h}")

    def _set(self, prop: int, value: float, name: str) -> None:
        assert self.cap is not None
        if not self.cap.set(prop, value):
            print(f"Warning: camera {self.index} property '{name}' may not be supported")

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
        if not self.cap:
            return False
        return bool(self.cap.grab())

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
    mean = float(gray.mean())
    std = float(gray.std())
    return mean > BLACK_FRAME_MEAN_LIMIT and std > BLACK_FRAME_STD_LIMIT


def camera_can_read(index: int, backend_name: str) -> bool:
    backend = BACKENDS[backend_name]
    cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    try:
        if not cap or not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        good_frames = 0
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None and frame_has_signal(frame):
                good_frames += 1
        if good_frames == 0:
            print(f"Camera index {index} opened but frames look black/empty; skipping")
            return False
        return True
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
            if left is None and right is None and len(found) >= 2:
                break
            if left is not None and index != left:
                break
            if right is not None and index != right:
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


def resize_to_size(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    if (frame.shape[1], frame.shape[0]) != size:
        raise ValueError(
            f"Calibration frames must be {size[0]}x{size[1]}; got {frame.shape[1]}x{frame.shape[0]}."
        )
    validate_exact_resolution(frame)
    return frame


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
    if ok_l and ok_r and left is not None and right is not None:
        return True, left, right, abs(left_timestamp - right_timestamp) * 1000.0
    return False, left, right, abs(left_timestamp - right_timestamp) * 1000.0


def warmup_camera_pair(left_cam: Camera, right_cam: Camera, attempts: int) -> bool:
    for _ in range(attempts):
        ok, left, right, _ = read_camera_pair(left_cam, right_cam)
        if ok and left is not None and right is not None:
            try:
                validate_exact_resolution(left)
                validate_exact_resolution(right)
                return True
            except ValueError as exc:
                print(f"Error: {exc}")
                return False
        cv2.waitKey(30)
    return False


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
        unique = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

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
        # Locate the board on a cheap reduced frame, then refine the accepted
        # corners on the original 1920x1080 image. Full-size search is only a
        # fallback, avoiding the previous full-HD SB solve on every preview.
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


class UnifiedStereoSession:
    def __init__(self, board: Chessboard, min_pairs: int) -> None:
        self.board = board
        self.min_pairs = min_pairs
        self.object_points: List[np.ndarray] = []
        self.left_points: List[np.ndarray] = []
        self.right_points: List[np.ndarray] = []
        self.signatures: List[np.ndarray] = []
        self.status = "No samples yet"
        self.last_intermediate_pair_count = 0

    def reset(self) -> None:
        self.object_points.clear()
        self.left_points.clear()
        self.right_points.clear()
        self.signatures.clear()
        self.last_intermediate_pair_count = 0
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
        left_ok, left_reason = self._corners_look_valid(left)
        right_ok, right_reason = self._corners_look_valid(right)
        if CONFIG.ENFORCE_CAPTURE_QUALITY and (not left_ok or not right_ok):
            self.status = f"Bad chessboard geometry: L {left_reason}, R {right_reason}"
            print(f"Warning: {self.status}; pair not added")
            return False
        if CONFIG.ENFORCE_CAPTURE_QUALITY and self.signatures:
            nearest = min(float(np.linalg.norm(signature - old)) for old in self.signatures)
            if nearest < 0.05:
                self.status = "Too similar; move/tilt chessboard"
                print("Warning: stereo pair is too similar to previous pairs; not added")
                return False
        self.object_points.append(self.board.object_points.copy())
        self.left_points.append(left.reshape(-1, 1, 2).astype(np.float32))
        self.right_points.append(right.reshape(-1, 1, 2).astype(np.float32))
        self.signatures.append(signature)
        self.board.candidates = [self.board.pattern]
        self.status = f"Captured pair {len(self.left_points)}"
        return True

    def _corners_look_valid(self, corners: np.ndarray) -> Tuple[bool, str]:
        pts = corners.reshape(self.board.rows, self.board.cols, 2)
        row_vectors = pts[:, -1, :] - pts[:, 0, :]
        col_vectors = pts[-1, :, :] - pts[0, :, :]
        row_lengths = np.linalg.norm(row_vectors, axis=1)
        col_lengths = np.linalg.norm(col_vectors, axis=1)
        if float(np.min(row_lengths)) < 20.0 or float(np.min(col_lengths)) < 20.0:
            return False, "too small"
        row_ratio = float(np.max(row_lengths) / max(1.0, np.min(row_lengths)))
        col_ratio = float(np.max(col_lengths) / max(1.0, np.min(col_lengths)))
        if row_ratio > 3.0 or col_ratio > 3.0:
            return False, "warped lines"
        main_row = row_vectors[len(row_vectors) // 2]
        main_col = col_vectors[len(col_vectors) // 2]
        dot = float(np.dot(main_row, main_col))
        denom = max(1e-6, float(np.linalg.norm(main_row) * np.linalg.norm(main_col)))
        angle = abs(np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0))) - 90.0)
        if angle > MAX_ROW_COL_ANGLE_ERROR_DEG:
            return False, "row/col angle"
        expected_aspect = (self.board.cols - 1) / max(1, self.board.rows - 1)
        actual_aspect = float(np.median(row_lengths) / max(1.0, np.median(col_lengths)))
        if abs(actual_aspect - expected_aspect) / expected_aspect > MAX_ASPECT_MISMATCH:
            return False, "aspect"
        return True, "ok"

    def calibrate_all(self, image_size: Tuple[int, int], metadata: dict) -> Tuple[dict, dict]:
        if len(self.left_points) < self.min_pairs:
            raise RuntimeError(f"Not enough pairs: {len(self.left_points)}/{self.min_pairs}")

        mono_flags = 0
        object_points, left_points, right_points, rejected_indices, filtered_results = self._filtered_points(
            image_size, mono_flags
        )
        if len(left_points) < MIN_GOOD_MONO_PAIRS:
            raise RuntimeError(f"Not enough good pairs after filtering: {len(left_points)}/{MIN_GOOD_MONO_PAIRS}")

        # The last outlier pass already solved both cameras. Reuse that exact
        # result instead of performing the most expensive mono step twice.
        if filtered_results is None:
            left_result, right_result = self._calibrate_left_right_parallel(
                object_points, left_points, right_points, image_size, mono_flags
            )
        else:
            left_result, right_result = filtered_results
        left_rms, left_k, left_d, left_rvecs, left_tvecs, _ = left_result
        right_rms, right_k, right_d, right_rvecs, right_tvecs, _ = right_result

        timestamp = datetime.now().isoformat(timespec="seconds")
        left_payload = self._mono_payload(metadata, timestamp, "left", left_rms, left_k, left_d, left_rvecs, left_tvecs, mono_flags, object_points, left_points, rejected_indices)
        right_payload = self._mono_payload(metadata, timestamp, "right", right_rms, right_k, right_d, right_rvecs, right_tvecs, mono_flags, object_points, right_points, rejected_indices)
        return left_payload, right_payload

    def _initial_camera_matrix(self, image_size: Tuple[int, int]) -> np.ndarray:
        w, h = image_size
        focal = float(max(w, h))
        return np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _calibrate_points(
        self,
        object_points: List[np.ndarray],
        image_points: List[np.ndarray],
        image_size: Tuple[int, int],
        flags: int,
    ) -> Tuple[float, np.ndarray, np.ndarray, Sequence[np.ndarray], Sequence[np.ndarray], List[float]]:
        if hasattr(cv2, "calibrateCameraExtended"):
            result = cv2.calibrateCameraExtended(object_points, image_points, image_size, None, None, flags=flags)
            rms, k, d, rvecs, tvecs, _, _, per_view = result
            errors = [float(value) for value in np.asarray(per_view).reshape(-1)]
        else:
            rms, k, d, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None, flags=flags)
            errors = self._sample_errors(object_points, image_points, rvecs, tvecs, k, d)
        return rms, k, d, rvecs, tvecs, errors

    def _calibrate_left_right_parallel(
        self,
        object_points: List[np.ndarray],
        left_points: List[np.ndarray],
        right_points: List[np.ndarray],
        image_size: Tuple[int, int],
        flags: int,
    ) -> Tuple[tuple, tuple]:
        """Solve both independent mono cameras concurrently within 16 CPUs."""
        previous_threads = max(1, cv2.getNumThreads())
        cv2.setNumThreads(MONO_THREADS_PER_JOB)
        try:
            with ThreadPoolExecutor(max_workers=MONO_PARALLEL_JOBS, thread_name_prefix="mono-calibration") as executor:
                left_future = executor.submit(
                    self._calibrate_points, object_points, left_points, image_size, flags
                )
                right_future = executor.submit(
                    self._calibrate_points, object_points, right_points, image_size, flags
                )
                return left_future.result(), right_future.result()
        finally:
            # The dependent stereo solve receives the full 16-thread budget.
            cv2.setNumThreads(previous_threads)

    def _filtered_points(
        self,
        image_size: Tuple[int, int],
        flags: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], Optional[Tuple[tuple, tuple]]]:
        object_points = list(self.object_points)
        left_points = list(self.left_points)
        right_points = list(self.right_points)
        original_indices = list(range(len(object_points)))
        rejected_indices: List[int] = []
        final_results: Optional[Tuple[tuple, tuple]] = None
        for _ in range(MAX_MONO_FILTER_PASSES):
            if len(left_points) <= MIN_GOOD_MONO_PAIRS:
                break
            left_result, right_result = self._calibrate_left_right_parallel(
                object_points, left_points, right_points, image_size, flags
            )
            left_errors = left_result[5]
            right_errors = right_result[5]
            combined_errors = [max(l, r) for l, r in zip(left_errors, right_errors)]
            candidates = outlier_indices(combined_errors)
            if not candidates:
                final_results = (left_result, right_result)
                break
            removable = max(0, len(left_points) - MIN_GOOD_MONO_PAIRS)
            ranked = sorted(
                candidates,
                key=lambda index: combined_errors[index] if np.isfinite(combined_errors[index]) else float("inf"),
                reverse=True,
            )[:removable]
            if not ranked:
                final_results = (left_result, right_result)
                break
            # Remove every statistically bad view found by this solve in one
            # batch. The old one-at-a-time loop could repeat calibration up to
            # 40 times while most CPU cores waited.
            for bad_index in ranked:
                print(
                    f"Rejecting mono pair {original_indices[bad_index]}: "
                    f"left={left_errors[bad_index]:.3f}, right={right_errors[bad_index]:.3f}"
                )
                rejected_indices.append(original_indices[bad_index])
            for bad_index in sorted(ranked, reverse=True):
                del object_points[bad_index]
                del left_points[bad_index]
                del right_points[bad_index]
                del original_indices[bad_index]
        if rejected_indices:
            print(f"Removed {len(rejected_indices)} bad calibration pair(s) before saving mono files")
        return object_points, left_points, right_points, rejected_indices, final_results

    def _mono_payload(
        self,
        metadata: dict,
        timestamp: str,
        side: str,
        rms: float,
        k: np.ndarray,
        d: np.ndarray,
        rvecs: Sequence[np.ndarray],
        tvecs: Sequence[np.ndarray],
        flags: int,
        object_points: List[np.ndarray],
        points: List[np.ndarray],
        rejected_sample_indices: List[int],
    ) -> dict:
        errors = self._sample_errors(object_points, points, rvecs, tvecs, k, d)
        return {
            **metadata,
            "timestamp": timestamp,
            "calibration_mode": "dual_camera_mono",
            "camera_side": side,
            "calibration_model": "pinhole",
            "camera_matrix": k.tolist(),
            "distortion_coefficients": d.tolist(),
            "rms_reprojection_error": float(rms),
            "per_sample_reprojection_errors": errors,
            "number_of_samples": len(points),
            "removed_bad_samples": len(rejected_sample_indices),
            "rejected_sample_indices": rejected_sample_indices,
            "opencv_calibration_flags": int(flags),
        }

    def _sample_errors(
        self,
        object_points: List[np.ndarray],
        image_points: List[np.ndarray],
        rvecs: Sequence[np.ndarray],
        tvecs: Sequence[np.ndarray],
        k: np.ndarray,
        d: np.ndarray,
    ) -> List[float]:
        errors = []
        for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
            projected, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
            errors.append(float(cv2.norm(img, projected, cv2.NORM_L2) / np.sqrt(len(projected))))
        return errors

    def _signature(self, left: np.ndarray, right: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        return np.concatenate([self._single_signature(left, image_size), self._single_signature(right, image_size)])

    def _single_signature(self, corners: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        pts = corners.reshape(-1, 2)
        w, h = image_size
        scale = np.array([w, h], dtype=np.float32)
        center = pts.mean(axis=0) / scale
        span = (pts.max(axis=0) - pts.min(axis=0)) / scale
        return np.concatenate([center, span]).astype(np.float32)

    def maybe_print_intermediate(self, image_size: Tuple[int, int], metadata: dict) -> None:
        pair_count = len(self.left_points)
        if pair_count < self.min_pairs:
            return
        if pair_count == self.last_intermediate_pair_count:
            return
        self.last_intermediate_pair_count = pair_count
        try:
            metadata = dict(metadata)
            metadata["chessboard_inner_corners"] = {"cols": self.board.cols, "rows": self.board.rows}
            left_payload, right_payload = self.calibrate_all(image_size, metadata)
            print_mono_quality_report(left_payload, right_payload)
        except Exception as exc:
            print(f"Intermediate calibration failed: {exc}")


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


def print_mono_quality_report(left_payload: dict, right_payload: dict) -> None:
    left_rms = float(left_payload["rms_reprojection_error"])
    right_rms = float(right_payload["rms_reprojection_error"])
    pairs = int(left_payload["number_of_samples"])
    print("")
    print("=== Dual mono calibration quality ===")
    print(f"Samples used per camera: {pairs}")
    print(f"Left RMS: {left_rms:.6f}")
    print(f"Right RMS: {right_rms:.6f}")
    print(f"Removed bad samples: {left_payload.get('removed_bad_samples', 0)}")
    for label, payload in (("Left", left_payload), ("Right", right_payload)):
        errors = np.asarray(payload["per_sample_reprojection_errors"], dtype=float)
        print(f"{label} per-view error median/max: {np.median(errors):.3f}/{np.max(errors):.3f} px")
        matrix = np.asarray(payload["camera_matrix"], dtype=float)
        print(f"{label} fx/fy/cx/cy: {matrix[0, 0]:.2f}/{matrix[1, 1]:.2f}/{matrix[0, 2]:.2f}/{matrix[1, 2]:.2f}")
        print(f"{label} distortion: {np.asarray(payload['distortion_coefficients']).reshape(-1).tolist()}")
    if left_rms > CONFIG.MONO_RMS_FAILURE_PX or right_rms > CONFIG.MONO_RMS_FAILURE_PX:
        print("BAD CALIBRATION: keep capturing more varied, fully visible board positions.")
    print(f"Square size: {left_payload['square_size_mm']} mm")
    print("==================================")
    print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simultaneous mono calibration for two cameras")
    parser.add_argument("--left-camera", type=int, default=LEFT_CAMERA_INDEX, help="Optional. If omitted, auto-detect first working camera.")
    parser.add_argument("--right-camera", type=int, default=RIGHT_CAMERA_INDEX, help="Optional. If omitted, auto-detect second working camera.")
    parser.add_argument("--scan-cameras", type=int, default=SCAN_CAMERA_INDEXES, help="How many camera indexes to scan when auto-detecting.")
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
    parser.add_argument("--use-sb-detector", action="store_true", default=USE_SB_DETECTOR)
    parser.add_argument("--detection-scale", type=float, default=DETECTION_SCALE)
    parser.add_argument("--detect-every", type=int, default=DETECT_EVERY_N_FRAMES)
    parser.add_argument("--auto-capture", action=argparse.BooleanOptionalAction, default=AUTO_CAPTURE)
    parser.add_argument("--auto-capture-cooldown", type=float, default=AUTO_CAPTURE_COOLDOWN_SECONDS)
    parser.add_argument("--left-output-yaml", default=LEFT_OUTPUT_YAML)
    parser.add_argument("--left-output-json", default=LEFT_OUTPUT_JSON)
    parser.add_argument("--right-output-yaml", default=RIGHT_OUTPUT_YAML)
    parser.add_argument("--right-output-json", default=RIGHT_OUTPUT_JSON)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Optional[str]:
    if args.left_camera is not None and args.right_camera is not None and args.left_camera == args.right_camera:
        return "left and right camera indexes must be different"
    if args.square_size <= 0:
        return "--square-size must be greater than 0"
    if (args.cols, args.rows) != (CONFIG.CHESSBOARD_COLS, CONFIG.CHESSBOARD_ROWS):
        return f"calibration requires --cols {CONFIG.CHESSBOARD_COLS} --rows {CONFIG.CHESSBOARD_ROWS} (inner corners)"
    if args.square_size != CONFIG.SQUARE_SIZE_MM:
        return "calibration requires --square-size 24.2 (millimetres)"
    if (args.width, args.height) != (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT):
        return "calibration requires --width 1920 --height 1080"
    if args.display_scale <= 0 or args.display_scale > 1:
        return "--display-scale must be greater than 0 and no more than 1"
    if args.min_pairs < 8:
        return "--min-pairs must be at least 8"
    if args.scan_cameras < 2:
        return "--scan-cameras must be at least 2"
    if args.detection_scale <= 0 or args.detection_scale > 1:
        return "--detection-scale must be greater than 0 and no more than 1"
    if args.detect_every < 1:
        return "--detect-every must be at least 1"
    if args.auto_capture_cooldown < 0:
        return "--auto-capture-cooldown must be at least 0"
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
        left_yaml = files.path(args.left_output_yaml)
        left_json = files.path(args.left_output_json)
        right_yaml = files.path(args.right_output_yaml)
        right_json = files.path(args.right_output_json)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    try:
        left_index, right_index = find_camera_indices(args.left_camera, args.right_camera, args.backend, args.scan_cameras)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Using left camera index {left_index}, right camera index {right_index}")

    left_cam = Camera(left_index, args.width, args.height, args.backend, args.left_rotate)
    right_cam = Camera(right_index, args.width, args.height, args.backend, args.right_rotate)
    board = Chessboard(args.cols, args.rows, args.square_size, args.use_sb_detector)
    session = UnifiedStereoSession(board, args.min_pairs)

    try:
        left_cam.open()
        right_cam.open()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        left_cam.release()
        right_cam.release()
        return 1

    requested_size = (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT)
    actual_sizes = ((left_cam.actual_width, left_cam.actual_height), (right_cam.actual_width, right_cam.actual_height))
    if actual_sizes[0] != requested_size or actual_sizes[1] != requested_size:
        print(f"Error: calibration requires both cameras at 1920x1080; got left {actual_sizes[0]}, right {actual_sizes[1]}")
        left_cam.release()
        right_cam.release()
        return 1
    if not warmup_camera_pair(left_cam, right_cam, PAIR_READ_WARMUP_ATTEMPTS):
        print("Error: both cameras must deliver exact 1920x1080 frames together.")
        left_cam.release()
        right_cam.release()
        cv2.destroyAllWindows()
        return 1

    left_cam.lock_autofocus()
    right_cam.lock_autofocus()
    if left_cam.autofocus_enabled or right_cam.autofocus_enabled:
        print("Error: autofocus is still enabled; calibration samples are rejected.")
        left_cam.release()
        right_cam.release()
        return 1

    common_size = requested_size
    print("Using verified calibration size 1920x1080 for both cameras")

    metadata = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "backend": args.backend,
        "left_camera_index": left_index,
        "right_camera_index": right_index,
        "image_size": {"width": common_size[0], "height": common_size[1]},
        "resolution": {"width": common_size[0], "height": common_size[1]},
        "requested_resolution": {"width": args.width, "height": args.height},
        "requested_fps": CAMERA_FPS,
        "left_actual_fps": left_cam.actual_fps,
        "right_actual_fps": right_cam.actual_fps,
        "left_actual_resolution": {"width": left_cam.actual_width, "height": left_cam.actual_height},
        "right_actual_resolution": {"width": right_cam.actual_width, "height": right_cam.actual_height},
        "left_rotation": args.left_rotate,
        "right_rotation": args.right_rotate,
        "detection_scale": args.detection_scale,
        "detect_every_n_frames": args.detect_every,
        "detection_contrast_enhance": DETECTION_CONTRAST_ENHANCE,
        "detection_clahe_clip_limit": DETECTION_CLAHE_CLIP_LIMIT,
        "detection_clahe_tile_grid": DETECTION_CLAHE_TILE_GRID,
        "auto_capture": bool(args.auto_capture),
        "auto_capture_cooldown_seconds": args.auto_capture_cooldown,
        "chessboard_inner_corners": {"cols": args.cols, "rows": args.rows},
        "square_size_mm": args.square_size,
    }

    print("Dual camera mono calibration started.")
    print(f"Square size: {args.square_size} mm")
    print(f"Chessboard inner corners expected: cols={args.cols}, rows={args.rows}")
    print("Important: this is INNER corners, not printed squares.")
    print(f"Detection scale: {args.detection_scale}")
    print(f"Detect every N frames: {args.detect_every}")
    print(f"OpenCV worker threads: {OPENCV_WORKER_THREADS}")
    print(f"SB detector: {'ON' if args.use_sb_detector else 'OFF'}")
    print(f"Capture target: {args.min_pairs} pairs")
    print(f"Minimum good pairs after filtering: {MIN_GOOD_MONO_PAIRS}")
    print("Left/Right RMS will appear after enough pairs are captured and calibration can run.")
    print("C capture pair | Space calibrate and save left/right camera files | R reset | Q/Esc quit")
    print("Move the chessboard through center, corners, near/far distances, and tilted angles.")

    frame_index = 0
    failed_reads = 0
    last_auto_capture_tick = 0
    found_l = False
    found_r = False
    corners_l = None
    corners_r = None
    blur_l = 0.0
    blur_r = 0.0
    sync_ms = float("inf")
    stability_gate = StabilityGate()
    detection_future: Optional[Future] = None
    detector_executor = ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            ok_pair, left, right, sync_ms = read_camera_pair(left_cam, right_cam)
            if not ok_pair or left is None or right is None:
                failed_reads += 1
                if failed_reads == 1 or failed_reads % 30 == 0:
                    print(f"Warning: failed to read one of the cameras ({failed_reads} times)")
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue
            failed_reads = 0

            try:
                left = resize_to_size(left, common_size)
                right = resize_to_size(right, common_size)
            except ValueError as exc:
                print(f"Error: {exc}")
                session.status = "Invalid camera resolution"
                break
            image_size = common_size

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
            should_detect = frame_index % args.detect_every == 0
            if should_detect and detection_future is None:
                robust_detect = frame_index % ROBUST_DETECT_EVERY_N_FRAMES == 0
                detection_future = detector_executor.submit(
                    detect_pair,
                    board,
                    left.copy(),
                    right.copy(),
                    args.detection_scale,
                    robust_detect,
                )
            frame_index += 1

            view_l = left.copy()
            view_r = right.copy()
            board.draw(view_l, found_l, corners_l)
            board.draw(view_r, found_r, corners_r)
            if found_l and corners_l is not None:
                draw_debug_corner_indices(view_l, corners_l)
            if found_r and corners_r is not None:
                draw_debug_corner_indices(view_r, corners_r)

            draw_overlay(
                view_l,
                [
                    f"LEFT cam {left_index}: {'FOUND' if found_l else 'not found'}",
                    f"Board: {board.cols}x{board.rows} inner corners",
                    f"Pairs: {len(session.left_points)}/{args.min_pairs}",
                    f"Pair sync: {sync_ms:.1f} ms",
                    f"AF: {'ON' if left_cam.autofocus_enabled else 'LOCKED/unsupported'}",
                    f"Auto: {'ON' if args.auto_capture else 'OFF'}",
                    f"Sharpness: {blur_l:.0f}",
                    session.status,
                ],
            )
            draw_overlay(
                view_r,
                [
                    f"RIGHT cam {right_index}: {'FOUND' if found_r else 'not found'}",
                    f"Board: {board.cols}x{board.rows} inner corners",
                    f"Pair sync: {sync_ms:.1f} ms",
                    f"AF: {'ON' if right_cam.autofocus_enabled else 'LOCKED/unsupported'}",
                    f"Sharpness: {blur_r:.0f}",
                    "A auto | C capture | Space save | R reset | Q quit",
                    "Need varied angles/distances/corners",
                ],
            )

            if args.display_scale < 1:
                view_l = cv2.resize(view_l, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
                view_r = cv2.resize(view_r, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
            combined = np.hstack([view_l, view_r])
            cv2.imshow("Unified Stereo Calibration", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                session.reset()
            elif key in (ord("a"), ord("A")):
                args.auto_capture = not args.auto_capture
                session.status = f"Auto capture {'ON' if args.auto_capture else 'OFF'}"
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
                        ok_pair, left, right, sync_ms = read_camera_pair(left_cam, right_cam)
                        if not ok_pair or left is None or right is None:
                            capture_failed = "Could not read a stereo frame pair"
                            break
                    found_l, corners_l, blur_l = board.detect(left, args.detection_scale, robust=True)
                    found_r, corners_r, blur_r = board.detect(right, args.detection_scale, robust=True)
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
                    elif session.add(corners_l, corners_r, image_size):
                        stability_gate = capture_gate
                        session.maybe_print_intermediate(image_size, metadata)
                print(session.status)
            elif key == 32:
                try:
                    final_metadata = dict(metadata)
                    final_metadata["chessboard_inner_corners"] = {"cols": board.cols, "rows": board.rows}
                    left_payload, right_payload = session.calibrate_all(image_size, final_metadata)
                    print_mono_quality_report(left_payload, right_payload)
                    quality_errors = []
                    if float(left_payload["rms_reprojection_error"]) > CONFIG.MONO_RMS_FAILURE_PX:
                        quality_errors.append(f"Left RMS {left_payload['rms_reprojection_error']:.3f} is high")
                    if float(right_payload["rms_reprojection_error"]) > CONFIG.MONO_RMS_FAILURE_PX:
                        quality_errors.append(f"Right RMS {right_payload['rms_reprojection_error']:.3f} is high")
                    if quality_errors:
                        print("Calibration quality check failed.")
                        for message in quality_errors:
                            print(f"  {message}")
                        if not ALLOW_SAVE_BAD_CALIBRATION:
                            print("Files were not saved because mono RMS exceeds 2.0 px.")
                            print("Move the board through more varied positions and press C only when both sides are FOUND.")
                            session.status = "Calibration failed quality check"
                            continue
                    save_yaml_json(left_payload, left_yaml, left_json)
                    save_yaml_json(right_payload, right_yaml, right_json)
                    print(f"Left RMS: {left_payload['rms_reprojection_error']:.6f}")
                    print(f"Right RMS: {right_payload['rms_reprojection_error']:.6f}")
                    print(f"Saved: {left_yaml.name}, {right_yaml.name}")
                    session.status = "Left/right camera files saved"
                except Exception as exc:
                    session.status = f"Calibration failed: {exc}"
                    print(f"Error: {session.status}")

            if args.auto_capture and found_l and found_r and corners_l is not None and corners_r is not None:
                now = cv2.getTickCount()
                cooldown_ticks = args.auto_capture_cooldown * cv2.getTickFrequency()
                if last_auto_capture_tick == 0 or now - last_auto_capture_tick >= cooldown_ticks:
                    stable, _ = stability_gate.ready()
                    if (
                        stable
                        and (not CONFIG.REQUIRE_BLUR_CHECK or (blur_l >= CONFIG.MIN_BLUR_VARIANCE and blur_r >= CONFIG.MIN_BLUR_VARIANCE))
                        and board_is_inside_frame(corners_l, left)
                        and board_is_inside_frame(corners_r, right)
                        and session.add(corners_l, corners_r, image_size)
                    ):
                        last_auto_capture_tick = now
                        print(f"Auto captured pair {len(session.left_points)}")
                        session.maybe_print_intermediate(image_size, metadata)
    finally:
        detector_executor.shutdown(wait=False, cancel_futures=True)
        left_cam.release()
        right_cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

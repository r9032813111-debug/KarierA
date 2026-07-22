import argparse
import json
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from calibration_utils_v2 import can_start_3d
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


BLACK_FRAME_MEAN_LIMIT = 8.0
BLACK_FRAME_STD_LIMIT = 4.0


# =========================
# USER SETTINGS
# =========================
# Edit these values here, then run:
# python stereo_depth.py
STEREO_CALIBRATION_FILE = "stereo_calibration.yaml"
LEFT_CAMERA_INDEX = None if CONFIG.AUTO_DETECT_CAMERAS else CONFIG.LEFT_CAMERA_INDEX
RIGHT_CAMERA_INDEX = None if CONFIG.AUTO_DETECT_CAMERAS else CONFIG.RIGHT_CAMERA_INDEX
SCAN_CAMERA_INDEXES = 8

# Must have the same 16:9 aspect ratio as the calibration images (1920x1080).
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
AUTO_RETRY_720P = False
DISPLAY_SCALE = CONFIG.DISPLAY_SCALE
BACKEND = "dshow"
LEFT_ROTATE = "none"
RIGHT_ROTATE = "none"
# Like the reference project: capture at 720p, calculate disparity at half size.
# This keeps the calibration geometry correct and is much faster than 1080p matching.
DEPTH_PROCESSING_SCALE = CONFIG.DEPTH_PROCESSING_SCALE

MIN_DISPARITY = 2
NUM_DISPARITIES = CONFIG.DEPTH_NUM_DISPARITIES
BLOCK_SIZE = CONFIG.DEPTH_BLOCK_SIZE
MATCHER_TYPE = CONFIG.DEPTH_MATCHER_TYPE  # "bm" or "sgbm"
BM_PREFILTER_SIZE = 23
BM_PREFILTER_CAP = 31
BM_PREFILTER_TYPE = 0
BM_TEXTURE_THRESHOLD = 100
BM_UNIQUENESS_RATIO = 3
BM_SPECKLE_WINDOW_SIZE = 200
BM_SPECKLE_RANGE = 10
BM_DISP12_MAX_DIFF = 1
MIN_DISTANCE_MM = CONFIG.DEPTH_MIN_DISTANCE_MM
MAX_DISTANCE_MM = CONFIG.DEPTH_MAX_DISTANCE_MM
SHOW_DISPARITY_DEBUG = CONFIG.DEPTH_SHOW_DISPARITY_DEBUG
EXPECTED_BASELINE_MIN_MM = CONFIG.BASELINE_MIN_MM
EXPECTED_BASELINE_MAX_MM = CONFIG.BASELINE_MAX_MM
OPENCV_WORKER_THREADS = max(1, min(CONFIG.CPU_THREADS, os.cpu_count() or 1))


@dataclass(frozen=True)
class DepthFilterSettings:
    enabled: bool
    filter_speckles: bool
    speckle_max_size_px: int
    speckle_disparity_range: float
    fill_morphological_holes: bool
    morphological_kernel_size: int
    fill_contour_holes: bool
    gradient_fill_all_gaps: bool
    max_hole_area_px: int
    inpaint_radius: float
    median_filter: bool
    median_kernel_size: int
    median_min_valid_ratio: float
    smooth_disparity: bool
    bilateral_diameter: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float

    @classmethod
    def from_config(cls) -> "DepthFilterSettings":
        return cls(
            CONFIG.DEPTH_ENABLE_ALL_FILTERS,
            CONFIG.DEPTH_FILTER_SPECKLES,
            CONFIG.DEPTH_SPECKLE_MAX_SIZE_PX,
            CONFIG.DEPTH_SPECKLE_DISPARITY_RANGE,
            CONFIG.DEPTH_FILL_MORPHOLOGICAL_HOLES,
            CONFIG.DEPTH_MORPHOLOGICAL_KERNEL_SIZE,
            CONFIG.DEPTH_FILL_CONTOUR_HOLES,
            CONFIG.DEPTH_GRADIENT_FILL_ALL_GAPS,
            CONFIG.DEPTH_MAX_HOLE_AREA_PX,
            CONFIG.DEPTH_INPAINT_RADIUS,
            CONFIG.DEPTH_MEDIAN_FILTER,
            CONFIG.DEPTH_MEDIAN_KERNEL_SIZE,
            CONFIG.DEPTH_MEDIAN_MIN_VALID_RATIO,
            CONFIG.DEPTH_SMOOTH_DISPARITY,
            CONFIG.DEPTH_BILATERAL_DIAMETER,
            CONFIG.DEPTH_BILATERAL_SIGMA_COLOR,
            CONFIG.DEPTH_BILATERAL_SIGMA_SPACE,
        )


@dataclass(frozen=True)
class DepthRuntimeSettings:
    min_distance_mm: float
    max_distance_mm: float
    colorized_view: bool
    match_window_size: int
    filters: DepthFilterSettings


class DepthControls:
    """OpenCV controls for live depth display and post-processing settings."""

    window_name = "Stereo Depth Controls"

    def __init__(self, min_distance_mm: float, max_distance_mm: float, match_window_size: int) -> None:
        self.initial = DepthFilterSettings.from_config()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 440, 620)
        self._add("All filters", int(self.initial.enabled), 1)
        self._add("Color depth", int(CONFIG.DEPTH_COLORIZED_VIEW), 1)
        self._add("Min distance cm", int(round(min_distance_mm / 10.0)), 2000)
        self._add("Max distance cm", int(round(max_distance_mm / 10.0)), 2000)
        self._add("Match window", max(5, (match_window_size // 2) * 2 + 1), 31)
        self._add("Speckles", int(self.initial.filter_speckles), 1)
        self._add("Speckle area px", self.initial.speckle_max_size_px, 2000)
        self._add("Morph hole fill", int(self.initial.fill_morphological_holes), 1)
        self._add("Contour hole fill", int(self.initial.fill_contour_holes), 1)
        self._add("Gradient fill all", int(self.initial.gradient_fill_all_gaps), 1)
        self._add("Max hole area x100", int(round(self.initial.max_hole_area_px / 100.0)), 2000)
        self._add("Inpaint radius x10", int(round(self.initial.inpaint_radius * 10.0)), 100)
        self._add("Median", int(self.initial.median_filter), 1)
        self._add("Bilateral", int(self.initial.smooth_disparity), 1)
        self._add("Bilateral diameter", self.initial.bilateral_diameter, 25)

    def _add(self, name: str, value: int, maximum: int) -> None:
        cv2.createTrackbar(name, self.window_name, max(0, min(value, maximum)), maximum, lambda _: None)

    def _get(self, name: str) -> int:
        return cv2.getTrackbarPos(name, self.window_name)

    def read(self) -> DepthRuntimeSettings:
        minimum = float(self._get("Min distance cm") * 10)
        maximum = float(max(self._get("Max distance cm") * 10, minimum + 10.0))
        match_window = max(5, self._get("Match window"))
        if match_window % 2 == 0:
            match_window += 1
        diameter = self._get("Bilateral diameter")
        if diameter > 0 and diameter % 2 == 0:
            diameter += 1
        filters = DepthFilterSettings(
            bool(self._get("All filters")),
            bool(self._get("Speckles")),
            max(0, self._get("Speckle area px")),
            self.initial.speckle_disparity_range,
            bool(self._get("Morph hole fill")),
            self.initial.morphological_kernel_size,
            bool(self._get("Contour hole fill")),
            bool(self._get("Gradient fill all")),
            max(0, self._get("Max hole area x100")) * 100,
            self._get("Inpaint radius x10") / 10.0,
            bool(self._get("Median")),
            self.initial.median_kernel_size,
            self.initial.median_min_valid_ratio,
            bool(self._get("Bilateral")),
            diameter,
            self.initial.bilateral_sigma_color,
            self.initial.bilateral_sigma_space,
        )
        return DepthRuntimeSettings(minimum, maximum, bool(self._get("Color depth")), match_window, filters)


class TorchBlockMatcher:
    """Full-resolution CUDA block matcher backed by the installed PyTorch runtime."""

    def __init__(self, min_disparity: int, num_disparities: int, block_size: int, batch_size: int) -> None:
        import torch
        import torch.nn.functional as functional

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        self.torch = torch
        self.functional = functional
        self.device = torch.device("cuda:0")
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.radius = self.block_size // 2
        self.batch_size = max(1, batch_size)
        torch.backends.cudnn.benchmark = True

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        torch = self.torch
        functional = self.functional
        height, width = left_gray.shape
        left = torch.from_numpy(np.ascontiguousarray(left_gray)).to(self.device, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
        right = torch.from_numpy(np.ascontiguousarray(right_gray)).to(self.device, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
        best_cost = torch.full((height, width), float("inf"), device=self.device, dtype=torch.float16)
        best_disparity = torch.zeros((height, width), device=self.device, dtype=torch.float16)
        with torch.inference_mode():
            for start in range(self.min_disparity, self.min_disparity + self.num_disparities, self.batch_size):
                stop = min(start + self.batch_size, self.min_disparity + self.num_disparities)
                costs = []
                for disparity in range(start, stop):
                    shifted = right if disparity == 0 else functional.pad(right[..., : width - disparity], (disparity, 0))
                    costs.append(torch.abs(left - shifted))
                costs = functional.avg_pool2d(torch.cat(costs, dim=0), self.block_size, stride=1, padding=self.radius)
                for batch_index, disparity in enumerate(range(start, stop)):
                    costs[batch_index, 0, :, : disparity + self.radius] = float("inf")
                    costs[batch_index, 0, : self.radius, :] = float("inf")
                    costs[batch_index, 0, height - self.radius :, :] = float("inf")
                local_cost, local_index = torch.min(costs[:, 0], dim=0)
                local_disparity = torch.arange(start, stop, device=self.device, dtype=torch.float16)[local_index]
                replace = local_cost < best_cost
                best_cost[replace] = local_cost[replace]
                best_disparity[replace] = local_disparity[replace]
        torch.cuda.synchronize(self.device)
        return best_disparity.float().cpu().numpy()


def create_bm_matcher(min_disparity: int, num_disparities: int, block_size: int):
    matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    matcher.setMinDisparity(min_disparity)
    matcher.setPreFilterSize(BM_PREFILTER_SIZE if BM_PREFILTER_SIZE % 2 == 1 else BM_PREFILTER_SIZE + 1)
    matcher.setPreFilterCap(BM_PREFILTER_CAP)
    matcher.setPreFilterType(BM_PREFILTER_TYPE)
    matcher.setTextureThreshold(BM_TEXTURE_THRESHOLD)
    matcher.setUniquenessRatio(BM_UNIQUENESS_RATIO)
    matcher.setSpeckleWindowSize(BM_SPECKLE_WINDOW_SIZE)
    matcher.setSpeckleRange(BM_SPECKLE_RANGE)
    matcher.setDisp12MaxDiff(BM_DISP12_MAX_DIFF)
    return matcher


class ParallelStereoBM:
    """CPU fallback: independent horizontal bands with enough overlap for BM windows."""

    def __init__(self, min_disparity: int, num_disparities: int, block_size: int, workers: int) -> None:
        self.workers = max(1, workers)
        self.overlap = max(block_size // 2, BM_PREFILTER_SIZE // 2)
        self.matchers = [create_bm_matcher(min_disparity, num_disparities, block_size) for _ in range(self.workers)]
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="stereo-bm")

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        height = left_gray.shape[0]
        band_count = min(self.workers, height)
        boundaries = np.linspace(0, height, band_count + 1, dtype=np.int32)

        def compute_band(index: int) -> Tuple[int, int, np.ndarray]:
            start, stop = int(boundaries[index]), int(boundaries[index + 1])
            source_start = max(0, start - self.overlap)
            source_stop = min(height, stop + self.overlap)
            disparity = self.matchers[index].compute(
                left_gray[source_start:source_stop],
                right_gray[source_start:source_stop],
            )
            return start, stop, disparity[start - source_start:stop - source_start]

        output = np.empty(left_gray.shape, dtype=np.int16)
        for future in [self.executor.submit(compute_band, index) for index in range(band_count)]:
            start, stop, disparity = future.result()
            output[start:stop] = disparity
        return output

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class StereoCalibration:
    left_k: np.ndarray
    left_d: np.ndarray
    right_k: np.ndarray
    right_d: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q: np.ndarray
    rotation: np.ndarray
    translation_mm: np.ndarray
    image_size: Tuple[int, int]
    baseline_mm: float
    source: Path
    left_camera_index: Optional[int]
    right_camera_index: Optional[int]


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


def load_stereo_calibration(value: str) -> StereoCalibration:
    path = WorkspaceFiles().path(value, must_exist=True)
    if path.suffix.lower() in (".yaml", ".yml"):
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        raise ValueError("Stereo calibration file must be .yaml, .yml, or .json")
    size = payload["image_size"]
    return StereoCalibration(
        np.array(payload["left_camera_matrix"], dtype=np.float64),
        np.array(payload["left_distortion_coefficients"], dtype=np.float64),
        np.array(payload["right_camera_matrix"], dtype=np.float64),
        np.array(payload["right_distortion_coefficients"], dtype=np.float64),
        np.array(payload["rectification_left"], dtype=np.float64),
        np.array(payload["rectification_right"], dtype=np.float64),
        np.array(payload["projection_left"], dtype=np.float64),
        np.array(payload["projection_right"], dtype=np.float64),
        np.array(payload["q_matrix"], dtype=np.float64),
        np.array(payload["rotation"], dtype=np.float64),
        np.array(payload["translation_mm"], dtype=np.float64),
        (int(size["width"]), int(size["height"])),
        float(payload.get("baseline_mm", 0.0)),
        path,
        payload.get("left_camera_index") if isinstance(payload.get("left_camera_index"), int) else None,
        payload.get("right_camera_index") if isinstance(payload.get("right_camera_index"), int) else None,
    )


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
        if self.actual_width != self.width or self.actual_height != self.height:
            print(f"Warning: camera {self.index} requested {self.width}x{self.height}, reports {self.actual_width}x{self.actual_height}")

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
    if (frame.shape[1], frame.shape[0]) == size:
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def read_camera_pair(left_cam: Camera, right_cam: Camera) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    left_grabbed = left_cam.grab()
    right_grabbed = right_cam.grab()
    if left_grabbed and right_grabbed:
        ok_l, left = left_cam.retrieve()
        ok_r, right = right_cam.retrieve()
        if ok_l and ok_r and left is not None and right is not None:
            return True, left, right

    ok_l, left = left_cam.read()
    ok_r, right = right_cam.read()
    if ok_l and ok_r and left is not None and right is not None:
        return True, left, right
    return False, left, right


def warmup_camera_pair(left_cam: Camera, right_cam: Camera, attempts: int = 30) -> bool:
    for _ in range(attempts):
        ok, _, _ = read_camera_pair(left_cam, right_cam)
        if ok:
            return True
        cv2.waitKey(30)
    return False


class StereoDepth:
    def __init__(self, calibration: StereoCalibration, matcher_type: str, min_disparity: int, num_disparities: int, block_size: int) -> None:
        self.calibration = calibration
        self.matcher_type = matcher_type
        self.min_disparity = min_disparity
        self.num_disparities = max(16, (num_disparities // 16) * 16)
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.map_size: Optional[Tuple[int, int]] = None
        self.left_map1: Optional[np.ndarray] = None
        self.left_map2: Optional[np.ndarray] = None
        self.right_map1: Optional[np.ndarray] = None
        self.right_map2: Optional[np.ndarray] = None
        self.q: Optional[np.ndarray] = None
        self.matcher = self._create_matcher()
        self.compute_backend = self._select_compute_backend()

    def _select_compute_backend(self) -> str:
        cpu_name = "CPU SGBM" if self.matcher_type == "sgbm" else "CPU BM"
        if CONFIG.DEPTH_GPU_BACKEND == "cpu":
            return cpu_name
        if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() <= 0:
            return f"{cpu_name} (CUDA OpenCV unavailable)"
        # CUDA StereoSGM is not exposed by many Windows OpenCV builds. Keep the
        # higher-quality CPU SGBM path unless a dedicated CUDA implementation is added.
        return f"{cpu_name} (CUDA stereo matcher unavailable in this OpenCV build)"

    def _create_matcher(self):
        if self.matcher_type == "bm":
            return create_bm_matcher(self.min_disparity, self.num_disparities, self.block_size)

        channels = 1
        return cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * channels * self.block_size * self.block_size,
            P2=32 * channels * self.block_size * self.block_size,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=80,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def _scaled_intrinsics(self, matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Scale a full-resolution camera matrix to the current matching resolution."""
        calibration_width, calibration_height = self.calibration.image_size
        sx = size[0] / calibration_width
        sy = size[1] / calibration_height
        scaled = matrix.copy()
        scaled[0, :] *= sx
        scaled[1, :] *= sy
        scaled[2, 2] = 1.0
        return scaled

    def _build_maps(self, size: Tuple[int, int]) -> None:
        c = self.calibration
        left_k = self._scaled_intrinsics(c.left_k, size)
        right_k = self._scaled_intrinsics(c.right_k, size)
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            left_k, c.left_d, right_k, c.right_d, size, c.rotation, c.translation_mm,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(left_k, c.left_d, r1, p1, size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(right_k, c.right_d, r2, p2, size, cv2.CV_16SC2)
        self.q = q
        self.map_size = size

    def rectify(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        size = (left.shape[1], left.shape[0])
        if self.map_size != size:
            self._build_maps(size)
        assert self.left_map1 is not None and self.left_map2 is not None
        assert self.right_map1 is not None and self.right_map2 is not None
        left_rect = cv2.remap(left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def depth_from_disparity(
        self,
        disparity: np.ndarray,
        filter_settings: Optional[DepthFilterSettings] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_valid = np.isfinite(disparity) & (disparity > self.min_disparity + 0.5)
        disparity, valid = improve_disparity(disparity, raw_valid, filter_settings)
        assert self.q is not None
        points_3d = cv2.reprojectImageTo3D(disparity, self.q)
        signed_z_mm = points_3d[:, :, 2]
        distance_mm = np.abs(signed_z_mm)
        valid &= np.isfinite(distance_mm) & (distance_mm > 0)
        return disparity, distance_mm, valid

    def compute(
        self,
        left: np.ndarray,
        right: np.ndarray,
        filter_settings: Optional[DepthFilterSettings] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_rect, right_rect = self.rectify(left, right)
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity, distance_mm, valid = self.depth_from_disparity(disparity, filter_settings)
        return left_rect, right_rect, disparity, distance_mm, valid


def contour_hole_mask(valid: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    """Find only small invalid islands enclosed by valid-depth contours."""
    valid_u8 = np.asarray(valid, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(valid_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    enclosed = np.zeros_like(valid_u8)
    cv2.drawContours(enclosed, contours, -1, 255, thickness=cv2.FILLED)
    candidates = cv2.bitwise_and(enclosed, cv2.bitwise_not(valid_u8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    holes = np.zeros_like(valid_u8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= max_hole_area_px:
            holes[labels == label] = 255
    return holes


def morphological_hole_mask(valid: np.ndarray, kernel_size: int, max_hole_area_px: int) -> np.ndarray:
    """Select only tiny gaps that a local closing operation can safely bridge."""
    if kernel_size < 2:
        return np.zeros(valid.shape, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    valid_u8 = np.asarray(valid, dtype=np.uint8) * 255
    closed = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, kernel)
    candidates = cv2.bitwise_and(closed, cv2.bitwise_not(valid_u8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    holes = np.zeros_like(valid_u8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= max_hole_area_px:
            holes[labels == label] = 255
    return holes


def filter_disparity_speckles(
    disparity: np.ndarray,
    valid: np.ndarray,
    settings: Optional[DepthFilterSettings] = None,
) -> np.ndarray:
    """Reject small isolated disparity islands without smoothing depth boundaries."""
    settings = settings or DepthFilterSettings.from_config()
    if not settings.filter_speckles:
        return valid.copy()
    fixed_point = np.where(valid, np.rint(disparity * 16.0), 0.0).astype(np.int16)
    cv2.filterSpeckles(
        fixed_point,
        0,
        max(0, int(settings.speckle_max_size_px)),
        max(1, int(round(settings.speckle_disparity_range * 16.0))),
    )
    return valid & (fixed_point > 0)


def masked_median_filter(disparity: np.ndarray, valid: np.ndarray, settings: DepthFilterSettings) -> np.ndarray:
    """Remove one-pixel noise while leaving areas near missing depth unchanged."""
    kernel_size = int(settings.median_kernel_size)
    if not settings.median_filter or kernel_size < 3:
        return disparity
    if kernel_size % 2 == 0:
        kernel_size += 1
    median = cv2.medianBlur(disparity.astype(np.float32), kernel_size)
    support = cv2.boxFilter(valid.astype(np.float32), -1, (kernel_size, kernel_size), normalize=True)
    output = disparity.copy()
    replace = valid & (support >= settings.median_min_valid_ratio)
    output[replace] = median[replace]
    return output


def improve_disparity(
    disparity: np.ndarray,
    valid: np.ndarray,
    settings: Optional[DepthFilterSettings] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    settings = settings or DepthFilterSettings.from_config()
    if not settings.enabled:
        return disparity.astype(np.float32, copy=True), valid.copy()
    filtered_valid = filter_disparity_speckles(disparity, valid, settings)
    holes = np.zeros(filtered_valid.shape, dtype=np.uint8)
    if not settings.gradient_fill_all_gaps:
        holes = (
            contour_hole_mask(filtered_valid, settings.max_hole_area_px)
            if settings.fill_contour_holes
            else holes
        )
        if settings.fill_morphological_holes:
            holes = cv2.bitwise_or(
                holes,
                morphological_hole_mask(
                    filtered_valid,
                    settings.morphological_kernel_size,
                    settings.max_hole_area_px,
                ),
            )
    accepted = filtered_valid | (holes > 0)
    filled = (
        cv2.inpaint(disparity.astype(np.float32), holes, settings.inpaint_radius, cv2.INPAINT_TELEA)
        if np.any(holes)
        else disparity.astype(np.float32)
    )
    output = disparity.copy()
    filled = masked_median_filter(filled, accepted, settings)
    if settings.smooth_disparity and settings.bilateral_diameter > 0:
        filled = cv2.bilateralFilter(
            filled,
            settings.bilateral_diameter,
            settings.bilateral_sigma_color,
            settings.bilateral_sigma_space,
        )
    output[accepted] = filled[accepted]
    return output, accepted


def fill_distance_gaps(
    distance_mm: np.ndarray,
    valid: np.ndarray,
    settings: DepthFilterSettings,
    min_distance_mm: float,
    max_distance_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill display-only depth gaps from neighbouring valid distances."""
    if not (settings.enabled and settings.gradient_fill_all_gaps):
        return distance_mm, valid

    gaps = np.asarray(~valid, dtype=np.uint8) * 255
    if not np.any(gaps):
        return distance_mm, valid

    if CONFIG.DEPTH_FAST_GAP_FILL and np.any(valid):
        try:
            from scipy.ndimage import distance_transform_edt

            nearest = distance_transform_edt(
                gaps > 0,
                return_distances=False,
                return_indices=True,
            )
            filled = distance_mm[nearest[0], nearest[1]].astype(np.float32, copy=False)
            if settings.smooth_disparity:
                sigma = max(0.5, min(2.0, settings.bilateral_sigma_space / 10.0))
                smoothed = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma, sigmaY=sigma)
                filled[gaps > 0] = smoothed[gaps > 0]
        except Exception as exc:
            print(f"Warning: fast gap fill unavailable; using inpaint: {exc}")
            filled = cv2.inpaint(
                distance_mm.astype(np.float32),
                gaps,
                max(1.0, settings.inpaint_radius),
                cv2.INPAINT_TELEA,
            )
    else:
        filled = cv2.inpaint(
            distance_mm.astype(np.float32),
            gaps,
            max(1.0, settings.inpaint_radius),
            cv2.INPAINT_TELEA,
        )
    fillable = (
        (gaps > 0)
        & np.isfinite(filled)
        & (filled >= min_distance_mm)
        & (filled <= max_distance_mm)
    )
    output = distance_mm.copy()
    output[fillable] = filled[fillable]
    return output, valid | fillable


def colorize_distance(
    distance_mm: np.ndarray,
    valid: np.ndarray,
    max_distance_mm: float,
    min_distance_mm: float = 0.0,
    colorized: bool = True,
) -> np.ndarray:
    if not colorized:
        gray = np.zeros(distance_mm.shape, dtype=np.uint8)
        span = max(1.0, max_distance_mm - min_distance_mm)
        gray[valid] = np.clip(
            (max_distance_mm - distance_mm[valid]) * 255.0 / span,
            0,
            255,
        ).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if CONFIG.DEPTH_VISUALIZATION == "equalized_jet":
        # This matches the reference demo's equalizeHist + JET appearance while
        # retaining distance_mm from the calibrated Q matrix.
        gray = np.zeros(distance_mm.shape, dtype=np.uint8)
        if np.any(valid):
            values = distance_mm[valid]
            low, high = np.percentile(values, (2, 98))
            if high <= low:
                high = low + 1.0
            gray[valid] = np.clip((distance_mm[valid] - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        equalized = cv2.equalizeHist(gray)
        colored = cv2.applyColorMap(equalized, cv2.COLORMAP_JET)
        colored[~valid] = (0, 0, 0)
        return colored
    clipped = np.zeros(distance_mm.shape, dtype=np.float32)
    clipped[valid] = np.clip(distance_mm[valid], 0, max_distance_mm)
    normalized = np.zeros(distance_mm.shape, dtype=np.uint8)
    normalized[valid] = (255 - (clipped[valid] / max_distance_mm * 255)).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    finite = np.isfinite(disparity)
    positive = finite & (disparity > 0)
    view = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(positive):
        hi = float(np.percentile(disparity[positive], 98))
        if hi <= 0:
            hi = 1.0
        view[positive] = np.clip(disparity[positive] / hi * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(view, cv2.COLORMAP_TURBO)


def compute_depth_frame(
    depth: StereoDepth,
    left: np.ndarray,
    right: np.ndarray,
    filter_settings: Optional[DepthFilterSettings] = None,
    cuda_matcher: Optional[TorchBlockMatcher] = None,
    parallel_bm_matcher: Optional[ParallelStereoBM] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cuda_matcher is not None:
        left_rect, right_rect = depth.rectify(left, right)
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = cuda_matcher.compute(left_gray, right_gray)
        disparity, distance_mm, valid = depth.depth_from_disparity(disparity, filter_settings)
        return left_rect, right_rect, disparity, distance_mm, valid
    if parallel_bm_matcher is not None:
        left_rect, right_rect = depth.rectify(left, right)
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = parallel_bm_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity, distance_mm, valid = depth.depth_from_disparity(disparity, filter_settings)
        return left_rect, right_rect, disparity, distance_mm, valid
    return depth.compute(left, right, filter_settings)


def build_depth_pipeline(
    calibration: StereoCalibration,
    args: argparse.Namespace,
    block_size: int,
) -> Tuple[StereoDepth, Optional[TorchBlockMatcher], Optional[ParallelStereoBM]]:
    depth = StereoDepth(calibration, args.matcher, args.min_disparity, args.num_disparities, block_size)
    cuda_matcher: Optional[TorchBlockMatcher] = None
    if CONFIG.DEPTH_USE_TORCH_CUDA and CONFIG.DEPTH_GPU_BACKEND != "cpu":
        try:
            cuda_matcher = TorchBlockMatcher(
                args.min_disparity,
                args.num_disparities,
                block_size,
                CONFIG.DEPTH_CUDA_DISPARITY_BATCH,
            )
        except Exception as exc:
            print(f"Warning: PyTorch CUDA matcher unavailable; using CPU fallback: {exc}")
    parallel_bm_matcher: Optional[ParallelStereoBM] = None
    if cuda_matcher is None and args.matcher == "bm" and CONFIG.DEPTH_CPU_BM_PARALLEL:
        cpu_workers = max(1, min(CONFIG.DEPTH_CPU_BM_WORKERS, os.cpu_count() or 1))
        cv2.setNumThreads(1)
        parallel_bm_matcher = ParallelStereoBM(
            args.min_disparity,
            args.num_disparities,
            block_size,
            cpu_workers,
        )
    return depth, cuda_matcher, parallel_bm_matcher


def render_for_window(frame: np.ndarray, window_name: str, fallback_scale: float) -> np.ndarray:
    if CONFIG.DEPTH_FIT_TO_WINDOW:
        try:
            _, _, width, height = cv2.getWindowImageRect(window_name)
            if width > 0 and height > 0:
                return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            pass
    if fallback_scale < 1:
        return cv2.resize(frame, None, fx=fallback_scale, fy=fallback_scale, interpolation=cv2.INTER_LINEAR)
    return frame


def draw_overlay(frame: np.ndarray, lines: Sequence[str]) -> None:
    x, y, dy = 16, 28, 24
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] for line in lines)
    cv2.rectangle(frame, (8, 8), (width + 32, 24 + dy * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live stereo depth map from calibrated two-camera rig")
    parser.add_argument("--stereo-calibration", default=STEREO_CALIBRATION_FILE)
    parser.add_argument("--left-camera", type=int, default=LEFT_CAMERA_INDEX, help="Optional. Defaults to saved calibration index or auto-detect.")
    parser.add_argument("--right-camera", type=int, default=RIGHT_CAMERA_INDEX, help="Optional. Defaults to saved calibration index or auto-detect.")
    parser.add_argument("--scan-cameras", type=int, default=SCAN_CAMERA_INDEXES, help="How many camera indexes to scan when auto-detecting.")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--display-scale", type=float, default=DISPLAY_SCALE)
    parser.add_argument("--backend", choices=BACKENDS.keys(), default=BACKEND)
    parser.add_argument("--left-rotate", choices=ROTATIONS.keys(), default=LEFT_ROTATE)
    parser.add_argument("--right-rotate", choices=ROTATIONS.keys(), default=RIGHT_ROTATE)
    parser.add_argument("--matcher", choices=("bm", "sgbm"), default=MATCHER_TYPE)
    parser.add_argument("--min-disparity", type=int, default=MIN_DISPARITY)
    parser.add_argument("--num-disparities", type=int, default=NUM_DISPARITIES)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--min-distance-mm", type=float, default=MIN_DISTANCE_MM)
    parser.add_argument("--max-distance-mm", type=float, default=MAX_DISTANCE_MM)
    return parser.parse_args()


def main() -> int:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(OPENCV_WORKER_THREADS)
    args = parse_args()
    if not (CONFIG.ENABLE_3D or CONFIG.ENABLE_DISPARITY):
        print("Error: 3D/disparity is disabled in config.py. Staying out of disparity/depth mode.")
        return 2
    if args.display_scale <= 0 or args.display_scale > 1:
        print("Error: --display-scale must be greater than 0 and no more than 1")
        return 2
    if args.width <= 0 or args.height <= 0:
        print("Error: invalid camera resolution")
        return 2
    if args.scan_cameras < 2:
        print("Error: --scan-cameras must be at least 2")
        return 2
    if args.num_disparities < 16:
        print("Error: --num-disparities must be at least 16")
        return 2
    if args.block_size < 3:
        print("Error: --block-size must be at least 3")
        return 2
    if args.min_distance_mm < 0:
        print("Error: --min-distance-mm must be non-negative")
        return 2
    if args.max_distance_mm <= args.min_distance_mm:
        print("Error: --max-distance-mm must be greater than --min-distance-mm")
        return 2
    try:
        calibration = load_stereo_calibration(args.stereo_calibration)
    except Exception as exc:
        print(f"Error: {exc}")
        print("Missing final stereo extrinsics file.")
        print("Run stereo_calibrate.py, wait until it saves stereo_calibration.yaml, then run stereo_depth.py again.")
        print("If stereo_calibrate.py says quality check failed, keep capturing more varied board positions.")
        return 2
    allowed, reason = can_start_3d(True)
    if not allowed:
        print(f"Error: {reason}. Staying out of disparity/depth mode.")
        return 2
    print(f"Loaded stereo baseline: {calibration.baseline_mm:.2f} mm")
    if not (EXPECTED_BASELINE_MIN_MM <= calibration.baseline_mm <= EXPECTED_BASELINE_MAX_MM):
        print(
            f"WARNING: baseline is outside expected range "
            f"{EXPECTED_BASELINE_MIN_MM:.1f}-{EXPECTED_BASELINE_MAX_MM:.1f} mm. "
            "Depth will probably be wrong; recalibrate."
        )
    left_arg = args.left_camera if args.left_camera is not None else calibration.left_camera_index
    right_arg = args.right_camera if args.right_camera is not None else calibration.right_camera_index
    if left_arg is not None and right_arg is not None and left_arg == right_arg:
        print("Error: left and right camera indexes must be different")
        return 2
    try:
        left_index, right_index = find_camera_indices(left_arg, right_arg, args.backend, args.scan_cameras)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Using left camera index {left_index}, right camera index {right_index}")

    left_cam = Camera(left_index, args.width, args.height, args.backend, args.left_rotate)
    right_cam = Camera(right_index, args.width, args.height, args.backend, args.right_rotate)
    active_block_size = args.block_size if args.block_size % 2 == 1 else args.block_size + 1
    depth, cuda_matcher, parallel_bm_matcher = build_depth_pipeline(calibration, args, active_block_size)
    print(f"Requested stereo depth camera resolution: {args.width}x{args.height} at {CAMERA_FPS} FPS")
    print(f"Stereo matcher: {args.matcher.upper()}")
    if cuda_matcher is not None:
        print("Depth backend: PyTorch CUDA full-HD matcher")
    elif parallel_bm_matcher is not None:
        print(f"Depth backend: CPU BM, {parallel_bm_matcher.workers} parallel horizontal bands")
    else:
        print(f"Depth backend: {depth.compute_backend}")
    print(f"OpenCV worker threads: {1 if parallel_bm_matcher is not None else OPENCV_WORKER_THREADS}")
    try:
        left_cam.open()
        right_cam.open()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        left_cam.release()
        right_cam.release()
        return 1
    if not warmup_camera_pair(left_cam, right_cam):
        if AUTO_RETRY_720P and (args.width, args.height) != (1280, 720):
            print(f"Warning: both cameras opened, but {args.width}x{args.height} pair streaming failed. Retrying at 1280x720.")
            left_cam.release()
            right_cam.release()
            left_cam = Camera(left_index, 1280, 720, args.backend, args.left_rotate)
            right_cam = Camera(right_index, 1280, 720, args.backend, args.right_rotate)
            try:
                left_cam.open()
                right_cam.open()
            except RuntimeError as exc:
                print(f"Error: {exc}")
                left_cam.release()
                right_cam.release()
                return 1
        if not warmup_camera_pair(left_cam, right_cam):
            print("Error: both cameras opened, but they cannot deliver frames together.")
            left_cam.release()
            right_cam.release()
            cv2.destroyAllWindows()
            return 1
    print(
        f"Camera actual sizes: left {left_cam.actual_width}x{left_cam.actual_height}, "
        f"right {right_cam.actual_width}x{right_cam.actual_height}"
    )
    live_size = (
        min(left_cam.actual_width, right_cam.actual_width),
        min(left_cam.actual_height, right_cam.actual_height),
    )
    calibration_ratio = calibration.image_size[0] / calibration.image_size[1]
    live_ratio = live_size[0] / live_size[1]
    if abs(live_ratio - calibration_ratio) > 0.01:
        print(
            f"Error: live cameras are {live_size[0]}x{live_size[1]} ({live_ratio:.3f}), but calibration is "
            f"{calibration.image_size[0]}x{calibration.image_size[1]} ({calibration_ratio:.3f})."
        )
        print("Use a 16:9 camera mode such as 1280x720 or recalibrate at the camera's real resolution.")
        left_cam.release()
        right_cam.release()
        return 2
    processing_size = (
        max(16, int(live_size[0] * DEPTH_PROCESSING_SCALE) // 2 * 2),
        max(2, int(live_size[1] * DEPTH_PROCESSING_SCALE) // 2 * 2),
    )
    print(
        f"Depth processing size: {processing_size[0]}x{processing_size[1]} "
        f"(calibration {calibration.image_size[0]}x{calibration.image_size[1]})"
    )
    window_name = "Stereo Depth"
    fullscreen = CONFIG.DEPTH_FULLSCREEN
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    controls = DepthControls(args.min_distance_mm, args.max_distance_mm, active_block_size)
    print("Stereo depth started. Use the Stereo Depth Controls window for filters and range. F fullscreen | Q/Esc quit.")
    fps = 0.0
    last_tick = cv2.getTickCount()
    failed_reads = 0
    depth_future: Optional[Future] = None
    depth_result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
    depth_executor = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            ok_pair, left, right = read_camera_pair(left_cam, right_cam)
            if not ok_pair or left is None or right is None:
                failed_reads += 1
                if failed_reads == 1 or failed_reads % 30 == 0:
                    print(f"Warning: failed to read one of the cameras ({failed_reads} times)")
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue
            failed_reads = 0
            # Do not stretch a low-resolution frame to 1080p. Rectification maps are
            # rebuilt with correctly scaled intrinsics for this matching resolution.
            left = resize_to_size(left, processing_size)
            right = resize_to_size(right, processing_size)
            runtime = controls.read()
            if depth_future is not None and depth_future.done():
                try:
                    depth_result = depth_future.result()
                except Exception as exc:
                    print(f"Warning: background depth failed: {exc}")
                    depth_result = None
                depth_future = None
            if depth_future is None and runtime.match_window_size != active_block_size:
                if parallel_bm_matcher is not None:
                    parallel_bm_matcher.close()
                active_block_size = runtime.match_window_size
                depth, cuda_matcher, parallel_bm_matcher = build_depth_pipeline(calibration, args, active_block_size)
                depth_result = None
                print(f"Stereo match window changed to {active_block_size} px")
            if depth_future is None:
                depth_future = depth_executor.submit(
                    compute_depth_frame,
                    depth,
                    left.copy(),
                    right.copy(),
                    runtime.filters,
                    cuda_matcher,
                    parallel_bm_matcher,
                )
            if depth_result is None:
                preview = np.hstack([left, right])
                draw_overlay(preview, ["Depth worker warming up", "F fullscreen | Q/Esc quit"])
                cv2.imshow(window_name, render_for_window(preview, window_name, args.display_scale))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("f"), ord("F")):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                continue
            rect, right_rect, disparity, distance_mm, valid = depth_result
            valid &= (distance_mm >= runtime.min_distance_mm) & (distance_mm <= runtime.max_distance_mm)
            distance_mm, valid = fill_distance_gaps(
                distance_mm,
                valid,
                runtime.filters,
                runtime.min_distance_mm,
                runtime.max_distance_mm,
            )
            now = cv2.getTickCount()
            dt = (now - last_tick) / cv2.getTickFrequency()
            last_tick = now
            if dt > 0:
                instant = 1.0 / dt
                fps = instant if fps <= 0 else fps * 0.9 + instant * 0.1
            depth_view = colorize_distance(
                distance_mm,
                valid,
                runtime.max_distance_mm,
                runtime.min_distance_mm,
                runtime.colorized_view,
            )
            disparity_view = colorize_disparity(disparity) if SHOW_DISPARITY_DEBUG else None
            valid_dist = distance_mm[valid]
            center = distance_mm[distance_mm.shape[0] // 2, distance_mm.shape[1] // 2]
            median = float(np.median(valid_dist)) if valid_dist.size else 0.0
            valid_percent = float(np.count_nonzero(valid) / valid.size * 100.0)
            positive_disp = disparity[np.isfinite(disparity) & (disparity > 0)]
            disp_text = "n/a"
            if positive_disp.size:
                disp_text = f"{float(np.percentile(positive_disp, 50)):.1f}/{float(np.percentile(positive_disp, 95)):.1f}"
            center_text = f"{center:.0f} mm" if np.isfinite(center) and center > 0 else "n/a"
            draw_overlay(
                depth_view,
                [
                    f"FPS: {fps:.1f}",
                    f"Cameras: L{left_index} R{right_index}",
                    f"Matcher: {args.matcher.upper()}",
                    f"Valid depth: {valid_percent:.1f}%",
                    f"Disp p50/p95: {disp_text}",
                    f"Center Z: {center_text}",
                    f"Median Z: {median:.0f} mm" if median > 0 else "Median Z: n/a",
                    f"Baseline: {calibration.baseline_mm:.1f} mm",
                    f"Depth view: {'color' if runtime.colorized_view else 'grayscale'}",
                    "F fullscreen | Q/Esc quit",
                ],
            )
            panels = [depth_view]
            if CONFIG.DEPTH_SHOW_RECTIFIED_PREVIEW:
                panels.insert(0, rect)
            if SHOW_DISPARITY_DEBUG:
                assert disparity_view is not None
                draw_overlay(disparity_view, ["Disparity debug", "If black: matching failed"])
                panels.append(disparity_view)
            combined = np.hstack(panels)
            cv2.imshow(window_name, render_for_window(combined, window_name, args.display_scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        depth_executor.shutdown(wait=False, cancel_futures=True)
        if parallel_bm_matcher is not None:
            parallel_bm_matcher.close()
        left_cam.release()
        right_cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

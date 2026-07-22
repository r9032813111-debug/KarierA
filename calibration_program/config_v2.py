"""All user-editable stereo calibration settings live here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StereoConfig:
    LEFT_CAMERA_INDEX: int = 0
    RIGHT_CAMERA_INDEX: int = 1
    AUTO_DETECT_CAMERAS: bool = True
    FRAME_WIDTH: int = 1920
    FRAME_HEIGHT: int = 1080
    FPS: int = 30
    DISPLAY_SCALE: float = 0.4
    CPU_THREADS: int = 16
    CAMERA_AUTOFOCUS: bool = False
    CAMERA_FOCUS: int = 0

    # An 8x8 printed chessboard has 7x7 INNER corners.
    CHESSBOARD_COLS: int = 7
    CHESSBOARD_ROWS: int = 6
    SQUARE_SIZE_MM: float = 22.8125 #15.888

    EXPECTED_BASELINE_MM: float = 120.0
    BASELINE_MIN_MM: float = 90.0
    BASELINE_MAX_MM: float = 150.0

    MIN_CALIBRATION_SAMPLES: int = 15
    TARGET_CALIBRATION_SAMPLES: int = 15

    MAX_SYNC_DIFFERENCE_MS: float = 15.0
    STABILITY_FRAME_COUNT: int = 4
    MAX_CORNER_MOTION_PX: float = 0.75
    MIN_BLUR_VARIANCE: float = 80.0
    REQUIRE_BLUR_CHECK: bool = False
    ENFORCE_CAPTURE_QUALITY: bool = False
    ENFORCE_FINAL_CALIBRATION_QUALITY: bool = False

    MONO_RMS_WARNING_PX: float = 1.0
    MONO_RMS_FAILURE_PX: float = 2.0
    STEREO_RMS_FAILURE_PX: float = 2.0

    ENABLE_3D: bool = True
    ENABLE_DISPARITY: bool = True
    DEBUG_CORNER_INDICES: bool = True

    # Keep depth matching at the calibrated camera resolution. Do not lower this
    # value when geometric detail is required.
    DEPTH_PROCESSING_SCALE: float = 1.0
    DEPTH_MIN_DISTANCE_MM: float = 20.0
    DEPTH_MAX_DISTANCE_MM: float = 6670.0
    DEPTH_GPU_BACKEND: str = "cpu"  # "auto", "cpu", or "cuda"
    DEPTH_USE_TORCH_CUDA: bool = True
    DEPTH_CUDA_DISPARITY_BATCH: int = 8
    DEPTH_CPU_BM_PARALLEL: bool = True
    DEPTH_CPU_BM_WORKERS: int = 8
    DEPTH_MATCHER_TYPE: str = "sgbm"
    DEPTH_NUM_DISPARITIES: int = 192
    DEPTH_BLOCK_SIZE: int = 15
    DEPTH_ENABLE_ALL_FILTERS: bool = False
    DEPTH_FILTER_SPECKLES: bool = True
    DEPTH_SPECKLE_MAX_SIZE_PX: int = 120
    DEPTH_SPECKLE_DISPARITY_RANGE: float = 2.0
    DEPTH_FILL_MORPHOLOGICAL_HOLES: bool = True
    DEPTH_MORPHOLOGICAL_KERNEL_SIZE: int = 3
    DEPTH_FILL_CONTOUR_HOLES: bool = False
    DEPTH_GRADIENT_FILL_ALL_GAPS: bool = True
    DEPTH_FAST_GAP_FILL: bool = True
    DEPTH_MAX_HOLE_AREA_PX: int = 40000
    DEPTH_INPAINT_RADIUS: float = 5.0
    DEPTH_MEDIAN_FILTER: bool = True
    DEPTH_MEDIAN_KERNEL_SIZE: int = 3
    DEPTH_MEDIAN_MIN_VALID_RATIO: float = 0.70
    DEPTH_BILATERAL_DIAMETER: int = 7
    DEPTH_BILATERAL_SIGMA_COLOR: float = 18.0
    DEPTH_BILATERAL_SIGMA_SPACE: float = 9.0
    DEPTH_SMOOTH_DISPARITY: bool = False
    DEPTH_SHOW_RECTIFIED_PREVIEW: bool = False
    DEPTH_SHOW_DISPARITY_DEBUG: bool = False
    DEPTH_COLORIZED_VIEW: bool = False
    DEPTH_VISUALIZATION: str = "equalized_jet"  # "equalized_jet" or "turbo"
    DEPTH_FULLSCREEN: bool = False
    DEPTH_FIT_TO_WINDOW: bool = True

    RGBD_PROCESSING_SCALE: float = 1.0
    RGBD_MIN_DISTANCE_MM: float = 20.0
    RGBD_MAX_DISTANCE_MM: float = 6670.0
    RGBD_Z_SCALE: float = 2.1
    RGBD_GRID_SIZE: int = 150
    RGBD_PLANE_WIDTH_MM: float = 2000.0
    RGBD_SMOOTHING_STRENGTH: float = 1.0
    RGBD_POINT_STEP: int = 6
    RGBD_POINT_SIZE: float = 1.0
    RGBD_UPDATE_INTERVAL_MS: int = 0
    RGBD_MAX_RENDER_POINTS: int = 40000
    RGBD_USE_CUDA: bool = True
    RGBD_CUDA_NUM_DISPARITIES: int = 128
    RGBD_CUDA_BLOCK_SIZE: int = 5
    RGBD_CUDA_DISPARITY_BATCH: int = 8
    RGBD_FLIP_VERTICAL: bool = False
    RGBD_INVERT_DEPTH: bool = False

    V9_WHITE_POINTS: bool = False
    V9_MEDIAN_AREA_PX: int = 84
    V9_OUTLIER_CM: int = 59
    V9_TOPO_CONTOUR_CM: int = 0

    DEPTH_WARP_Z_SCALE: float = 1.0
    DEPTH_WARP_PLANE_WIDTH: float = 2.0
    DEPTH_WARP_MIN_DISTANCE_MM: float = 200.0
    DEPTH_WARP_MAX_DISTANCE_MM: float = 3000.0
    DEPTH_WARP_POINT_STEP: int = 6
    DEPTH_WARP_MAX_RENDER_POINTS: int = 8000
    DEPTH_WARP_POINT_SIZE: float = 2.0
    DEPTH_WARP_MESH_STEP: int = 8
    DEPTH_WARP_MAX_FACE_DEPTH_JUMP_MM: float = 250.0


CONFIG = StereoConfig()

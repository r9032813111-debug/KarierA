"""One-pass stereo calibration: capture RAM pairs, mono-calibrate, then stereo-calibrate."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import dual_camera_calibrate_v2 as mono
import stereo_calibrate_v2 as stereo
from calibration_utils_v2 import startup_validation
from config_v2 import CONFIG


DEFAULT_TARGET_SAMPLES = 20
WINDOW_NAME = "Full Stereo Calibration v2"
PREVIEW_OPENCV_THREADS = max(1, min(8, CONFIG.CPU_THREADS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one set of stereo pairs and calibrate mono plus stereo in one run")
    parser.add_argument("--left-camera", type=int, default=None)
    parser.add_argument("--right-camera", type=int, default=None)
    parser.add_argument("--scan-cameras", type=int, default=8)
    parser.add_argument("--backend", choices=mono.BACKENDS.keys(), default=mono.BACKEND)
    parser.add_argument("--left-rotate", choices=mono.ROTATIONS.keys(), default=mono.LEFT_ROTATE)
    parser.add_argument("--right-rotate", choices=mono.ROTATIONS.keys(), default=mono.RIGHT_ROTATE)
    parser.add_argument("--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES)
    parser.add_argument("--display-scale", type=float, default=CONFIG.DISPLAY_SCALE)
    parser.add_argument("--left-output-yaml", default=mono.LEFT_OUTPUT_YAML)
    parser.add_argument("--left-output-json", default=mono.LEFT_OUTPUT_JSON)
    parser.add_argument("--right-output-yaml", default=mono.RIGHT_OUTPUT_YAML)
    parser.add_argument("--right-output-json", default=mono.RIGHT_OUTPUT_JSON)
    parser.add_argument("--stereo-output-yaml", default=stereo.STEREO_OUTPUT_YAML)
    parser.add_argument("--stereo-output-json", default=stereo.STEREO_OUTPUT_JSON)
    return parser.parse_args()


def workspace_file(value: str) -> Path:
    root = Path.cwd().resolve()
    path = (root / value).resolve()
    if path.parent != root:
        raise ValueError("Output files must be directly inside this project folder")
    return path


def metadata(
    left_index: int,
    right_index: int,
    left_camera: mono.Camera,
    right_camera: mono.Camera,
    backend: str,
) -> dict:
    width, height = CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "backend": backend,
        "left_camera_index": left_index,
        "right_camera_index": right_index,
        "image_size": {"width": width, "height": height},
        "resolution": {"width": width, "height": height},
        "requested_resolution": {"width": width, "height": height},
        "requested_fps": CONFIG.FPS,
        "left_actual_fps": left_camera.actual_fps,
        "right_actual_fps": right_camera.actual_fps,
        "left_actual_resolution": {"width": left_camera.actual_width, "height": left_camera.actual_height},
        "right_actual_resolution": {"width": right_camera.actual_width, "height": right_camera.actual_height},
        "chessboard_inner_corners": {"cols": CONFIG.CHESSBOARD_COLS, "rows": CONFIG.CHESSBOARD_ROWS},
        "square_size_mm": CONFIG.SQUARE_SIZE_MM,
        "workflow": "full_calibrate_v2_single_capture_set",
    }


def run_calibration(
    session: mono.UnifiedStereoSession,
    board: mono.Chessboard,
    image_size: Tuple[int, int],
    base_metadata: dict,
    outputs: Tuple[Path, Path, Path, Path, Path, Path],
) -> bool:
    started_at = time.perf_counter()
    cv2.setNumThreads(mono.OPENCV_WORKER_THREADS)
    base_metadata = dict(base_metadata)
    base_metadata["calibration_logical_cpu_budget"] = mono.OPENCV_WORKER_THREADS
    base_metadata["mono_parallel_jobs"] = mono.MONO_PARALLEL_JOBS
    base_metadata["mono_threads_per_job"] = mono.MONO_THREADS_PER_JOB
    print(
        f"Calibration CPU plan: {mono.MONO_PARALLEL_JOBS} mono jobs x "
        f"{mono.MONO_THREADS_PER_JOB} threads; stereo solve x {mono.OPENCV_WORKER_THREADS} threads"
    )
    left_yaml, left_json, right_yaml, right_json, stereo_yaml, stereo_json = outputs
    left_payload, right_payload = session.calibrate_all(image_size, dict(base_metadata))
    mono.print_mono_quality_report(left_payload, right_payload)

    mono.save_yaml_json(left_payload, left_yaml, left_json)
    mono.save_yaml_json(right_payload, right_yaml, right_json)
    print(f"Saved mono calibration: {left_yaml.name}, {right_yaml.name}")

    rejected = set(left_payload.get("rejected_sample_indices", []))
    stereo_session = stereo.StereoSession(board, CONFIG.MIN_CALIBRATION_SAMPLES)
    for index, (left, right) in enumerate(zip(session.left_points, session.right_points)):
        if index not in rejected:
            stereo_session.add(left, right, image_size)
    if len(stereo_session.left_points) < CONFIG.MIN_CALIBRATION_SAMPLES:
        raise RuntimeError("Not enough pairs remain after mono outlier removal for stereo calibration")

    stereo_metadata = dict(base_metadata)
    stereo_metadata["number_of_captured_pairs"] = len(session.left_points)
    stereo_metadata["mono_rejected_sample_indices"] = sorted(rejected)
    stereo_metadata["number_of_pairs_after_mono_filtering"] = len(stereo_session.left_points)
    left_calibration = stereo.MonoCalibration(left_yaml, left_payload)
    right_calibration = stereo.MonoCalibration(right_yaml, right_payload)
    print("Applying the newly calculated K/D intrinsics to the same RAM pairs for fixed-intrinsics stereo calibration.")
    saved = stereo.try_save_stereo_calibration(
        stereo_session,
        left_calibration,
        right_calibration,
        image_size,
        stereo_metadata,
        stereo_yaml,
        stereo_json,
    )
    if saved:
        print(f"Saved stereo calibration: {stereo_yaml.name}")
    print(f"Full calibration calculation time: {time.perf_counter() - started_at:.2f} s")
    return saved


def main() -> int:
    args = parse_args()
    if args.target_samples < CONFIG.MIN_CALIBRATION_SAMPLES:
        print(f"Error: --target-samples must be at least {CONFIG.MIN_CALIBRATION_SAMPLES}")
        return 2
    if args.display_scale <= 0 or args.display_scale > 1:
        print("Error: --display-scale must be greater than 0 and no more than 1")
        return 2
    try:
        startup_validation()
        outputs = (
            workspace_file(args.left_output_yaml),
            workspace_file(args.left_output_json),
            workspace_file(args.right_output_yaml),
            workspace_file(args.right_output_json),
            workspace_file(args.stereo_output_yaml),
            workspace_file(args.stereo_output_json),
        )
        left_index, right_index = mono.find_camera_indices(args.left_camera, args.right_camera, args.backend, args.scan_cameras)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2

    cv2.setUseOptimized(True)
    cv2.setNumThreads(PREVIEW_OPENCV_THREADS)
    print(f"Logical CPU calibration budget: {mono.OPENCV_WORKER_THREADS}")
    board = mono.Chessboard(CONFIG.CHESSBOARD_COLS, CONFIG.CHESSBOARD_ROWS, CONFIG.SQUARE_SIZE_MM, mono.USE_SB_DETECTOR)
    session = mono.UnifiedStereoSession(board, args.target_samples)
    left_camera = mono.Camera(left_index, CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT, args.backend, args.left_rotate)
    right_camera = mono.Camera(right_index, CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT, args.backend, args.right_rotate)

    try:
        left_camera.open()
        right_camera.open()
        expected_size = (CONFIG.FRAME_WIDTH, CONFIG.FRAME_HEIGHT)
        actual_sizes = ((left_camera.actual_width, left_camera.actual_height), (right_camera.actual_width, right_camera.actual_height))
        if actual_sizes != (expected_size, expected_size):
            raise RuntimeError(f"Calibration requires both cameras at {expected_size[0]}x{expected_size[1]}, got {actual_sizes}")
        if not mono.warmup_camera_pair(left_camera, right_camera, mono.PAIR_READ_WARMUP_ATTEMPTS):
            raise RuntimeError("Both cameras cannot deliver an exact 1920x1080 pair")
        left_camera.lock_autofocus()
        right_camera.lock_autofocus()
        print("Full calibration v2 started.")
        print(f"Capture exactly {args.target_samples} varied RAM stereo pairs. C capture | Space calibrate/save | R reset | Q/Esc quit")

        base_metadata = metadata(left_index, right_index, left_camera, right_camera, args.backend)
        found_left = found_right = False
        corners_left: Optional[np.ndarray] = None
        corners_right: Optional[np.ndarray] = None
        status = "Move the board through centre, edges, distances, and tilt angles"
        while True:
            ok, left, right, sync_ms = mono.read_camera_pair(left_camera, right_camera)
            if not ok or left is None or right is None:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue
            try:
                mono.validate_exact_resolution(left)
                mono.validate_exact_resolution(right)
            except ValueError as exc:
                print(f"Error: {exc}")
                break
            found_left, corners_left, blur_left, found_right, corners_right, blur_right = mono.detect_pair(
                board, left, right, mono.DETECTION_SCALE, robust=False
            )
            view_left, view_right = left.copy(), right.copy()
            board.draw(view_left, found_left, corners_left)
            board.draw(view_right, found_right, corners_right)
            mono.draw_overlay(view_left, [
                f"LEFT: {'FOUND' if found_left else 'not found'}",
                f"Pairs: {len(session.left_points)}/{args.target_samples}",
                f"Sync: {sync_ms:.1f} ms | blur: {blur_left:.0f}",
                status,
            ])
            mono.draw_overlay(view_right, [
                f"RIGHT: {'FOUND' if found_right else 'not found'}",
                f"Board: {board.cols}x{board.rows} inner corners",
                f"Sync: {sync_ms:.1f} ms | blur: {blur_right:.0f}",
                "C capture | Space full calibrate | R reset | Q quit",
            ])
            if args.display_scale < 1:
                view_left = cv2.resize(view_left, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
                view_right = cv2.resize(view_right, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, np.hstack((view_left, view_right)))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                session.reset()
                status = "RAM samples cleared"
            elif key in (ord("c"), ord("C")):
                if found_left and found_right and corners_left is not None and corners_right is not None:
                    if session.add(corners_left, corners_right, expected_size):
                        status = session.status
                    else:
                        status = session.status
                else:
                    status = "Board must be found in both cameras"
                print(status)
            elif key == 32:
                try:
                    if len(session.left_points) < args.target_samples:
                        raise RuntimeError(f"Capture {args.target_samples} pairs first; have {len(session.left_points)}")
                    try:
                        if run_calibration(session, board, expected_size, base_metadata, outputs):
                            status = "Calibration saved. Space recalculates and overwrites files; C adds another pair"
                        else:
                            status = "Stereo quality failed; capture more varied pairs or reset"
                    finally:
                        cv2.setNumThreads(PREVIEW_OPENCV_THREADS)
                except Exception as exc:
                    status = f"Full calibration failed: {exc}"
                    print(f"Error: {status}")
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        left_camera.release()
        right_camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

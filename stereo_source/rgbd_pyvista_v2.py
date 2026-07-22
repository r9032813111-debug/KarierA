"""PyVista/VTK RGBD height-field visualizer for Python 3.13."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv

from calibration_utils_v2 import can_start_3d
from config_v2 import CONFIG
from rgbd_point_cloud_gpu_v2 import GpuControls
from rgbd_point_cloud_v2 import depth_mesh
import stereo_depth_v2 as depth_v2


WINDOW_TITLE = "RGBD PyVista v2"
DEPTH_WINDOW_TITLE = "RGBD Depth Map v2"
SAVED_DIR = Path(__file__).resolve().parent / "saved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyVista RGBD textured height-field.")
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
    parser.add_argument(
        "--depth-backend",
        choices=("native", "torch-cuda"),
        default="native",
        help="native uses OpenCV C++ SGBM; torch-cuda keeps the experimental matcher.",
    )
    return parser.parse_args()


def create_mesh(vertices: np.ndarray) -> pv.StructuredGrid:
    rows, columns = vertices.shape[:2]
    mesh = pv.StructuredGrid()
    mesh.dimensions = (columns, rows, 1)
    mesh.points = np.ascontiguousarray(vertices.reshape(-1, 3), dtype=np.float32)
    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, columns, dtype=np.float32),
        np.linspace(1.0, 0.0, rows, dtype=np.float32),
    )
    mesh.active_texture_coordinates = np.ascontiguousarray(
        np.column_stack((u.reshape(-1), v.reshape(-1))), dtype=np.float32
    )
    return mesh


def update_mesh(mesh: pv.StructuredGrid, vertices: np.ndarray, vertex_valid: np.ndarray) -> None:
    points = vertices.copy()
    if not np.all(vertex_valid):
        replacement_z = float(np.median(points[:, :, 2][vertex_valid])) if np.any(vertex_valid) else 0.0
        points[~vertex_valid, 2] = replacement_z
    mesh.points[:] = np.ascontiguousarray(points.reshape(-1, 3), dtype=np.float32)
    mesh.points.Modified()
    mesh.Modified()


def update_texture(texture_pixels: np.ndarray, rectified_bgr: np.ndarray) -> None:
    rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
    texture_pixels[:] = np.flipud(rgb).reshape(-1, 3)
    texture_pixels.Modified()


def save_textured_model(plotter: pv.Plotter) -> None:
    """Save the current VTK scene as one self-contained textured glTF model."""
    SAVED_DIR.mkdir(exist_ok=True)
    filename = SAVED_DIR / f"stereo_depth_{datetime.now():%Y%m%d_%H%M%S}.gltf"
    try:
        plotter.export_gltf(str(filename), inline_data=True, rotate_scene=False)
    except Exception as exc:
        print(f"Error: could not save textured model: {exc}")
        return
    print(f"Saved textured 3D model: {filename}")


def smooth_mesh_z(vertices: np.ndarray, z_smoothing: float) -> np.ndarray:
    """Smooth only height/depth Z values while retaining the image-plane mesh."""
    if z_smoothing <= 0.0:
        return vertices
    output = vertices.copy()
    sigma = min(5.0, z_smoothing)
    output[:, :, 2] = cv2.GaussianBlur(
        output[:, :, 2],
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return output


def draw_3d_vertex_overlay(depth_view: np.ndarray, vertex_valid: np.ndarray) -> None:
    """Mark the exact sampled depth vertices used by the 3D mesh."""
    height, width = depth_view.shape[:2]
    rows, columns = vertex_valid.shape
    ys = np.linspace(0, height - 1, rows, dtype=np.int32)
    xs = np.linspace(0, width - 1, columns, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    point_mask = np.zeros((height, width), dtype=np.uint8)
    point_mask[yy[vertex_valid], xx[vertex_valid]] = 255
    point_mask = cv2.dilate(
        point_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
    )
    depth_view[point_mask > 0] = (255, 255, 255)


def compute_visual_frame(
    depth: depth_v2.StereoDepth,
    left: np.ndarray,
    right: np.ndarray,
    filters: depth_v2.DepthFilterSettings,
    cuda_matcher: depth_v2.TorchBlockMatcher | None,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Fill depth only once after the distance range has been applied. Filling
    # disparity and distance separately doubles the full-HD inpaint cost.
    matching_filters = replace(filters, gradient_fill_all_gaps=False)
    rectified, _, _disparity, distance_mm, valid = depth_v2.compute_depth_frame(
        depth, left, right, matching_filters, cuda_matcher, None
    )
    valid &= (distance_mm >= minimum) & (distance_mm <= maximum)
    distance_mm, valid = depth_v2.fill_distance_gaps(
        distance_mm, valid, filters, minimum, maximum
    )
    return rectified, distance_mm, valid


def main() -> int:
    args = parse_args()
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
        depth = depth_v2.StereoDepth(
            calibration, args.matcher, args.min_disparity, args.num_disparities, block_size
        )
        cuda_matcher = None
        parallel_bm_matcher = None
        print(f"Depth backend: native OpenCV {args.matcher.upper()} at {CONFIG.CPU_THREADS} threads")
    controls = GpuControls()
    cv2.namedWindow(DEPTH_WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEPTH_WINDOW_TITLE, 900, 600)
    plotter: pv.Plotter | None = None
    mesh_actor = None
    try:
        left_camera.open()
        right_camera.open()
        if not depth_v2.warmup_camera_pair(left_camera, right_camera):
            raise RuntimeError("Both cameras opened but cannot deliver frames together")
        print("PyVista GPU renderer started. Drag: rotate, wheel: zoom, right drag: pan, R: reset camera, Q: close.")

        mesh: pv.StructuredGrid | None = None
        texture_pixels: np.ndarray | None = None
        depth_future: Future[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgbd-depth") as executor:
            while plotter is None or not plotter._closed:
                cv2.waitKey(1)
                frame_updated = False
                if depth_future is not None and depth_future.done():
                    try:
                        rectified, distance_mm, valid = depth_future.result()
                    except Exception as exc:
                        print(f"Warning: depth worker failed: {exc}")
                    else:
                        grid, z_scale, plane_width, _minimum, _maximum, _filters, z_smoothing = controls.values()
                        vertices, _colours, vertex_valid = depth_mesh(
                            rectified, distance_mm, valid, grid, z_scale, plane_width
                        )
                        vertices = smooth_mesh_z(vertices, z_smoothing)
                        depth_view = depth_v2.colorize_distance(
                            distance_mm,
                            valid,
                            _maximum,
                            _minimum,
                            True,
                        )
                        draw_3d_vertex_overlay(depth_view, vertex_valid)
                        cv2.imshow(
                            DEPTH_WINDOW_TITLE,
                            depth_v2.render_for_window(
                                depth_view,
                                DEPTH_WINDOW_TITLE,
                                CONFIG.DISPLAY_SCALE,
                            ),
                        )
                        if mesh is None:
                            mesh = create_mesh(vertices)
                            texture = pv.numpy_to_texture(np.flipud(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)))
                            texture_pixels = texture.to_image().point_data["Image"]
                            plotter = pv.Plotter(window_size=(1280, 720), title=WINDOW_TITLE)
                            mesh_actor = plotter.add_mesh(mesh, texture=texture, smooth_shading=False, show_edges=False)
                            plotter.add_key_event("r", plotter.reset_camera)
                            plotter.add_key_event("s", lambda: save_textured_model(plotter))
                            plotter.show(auto_close=False, interactive_update=True)
                            plotter.view_isometric()
                        elif mesh.dimensions[:2] != (grid, grid):
                            assert plotter is not None and mesh_actor is not None
                            mesh = create_mesh(vertices)
                            plotter.remove_actor(mesh_actor, reset_camera=False, render=False)
                            mesh_actor = plotter.add_mesh(mesh, texture=texture, smooth_shading=False, show_edges=False)
                            update_texture(texture_pixels, rectified)
                        else:
                            assert texture_pixels is not None
                            update_mesh(mesh, vertices, vertex_valid)
                            update_texture(texture_pixels, rectified)
                        frame_updated = True
                    depth_future = None
                if depth_future is None:
                    ok, left, right = depth_v2.read_camera_pair(left_camera, right_camera)
                    if ok and left is not None and right is not None:
                        _grid, _z_scale, _plane_width, minimum, maximum, filters, _z_smoothing = controls.values()
                        depth_future = executor.submit(
                            compute_visual_frame,
                            depth,
                            left.copy(),
                            right.copy(),
                            filters,
                            cuda_matcher,
                            minimum,
                            maximum,
                        )
                if plotter is not None:
                    plotter.update(stime=1, force_redraw=frame_updated)
    finally:
        left_camera.release()
        right_camera.release()
        if parallel_bm_matcher is not None:
            parallel_bm_matcher.close()
        if plotter is not None and not plotter._closed:
            plotter.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

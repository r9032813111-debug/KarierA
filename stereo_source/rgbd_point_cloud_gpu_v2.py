"""GPU OpenGL RGBD height-field renderer using the V2 stereo depth pipeline."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass

import cv2
import numpy as np
import pygame

from calibration_utils_v2 import can_start_3d
from config_v2 import CONFIG
from rgbd_point_cloud_v2 import ViewState, depth_mesh
import stereo_depth_v2 as depth_v2


WINDOW_NAME = "RGBD Point Cloud GPU v2"
CONTROLS_NAME = "RGBD Controls v2"

GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_DEPTH_TEST = 0x0B71
GL_LEQUAL = 0x0203
GL_PROJECTION = 0x1701
GL_MODELVIEW = 0x1700
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_LINEAR = 0x2601
GL_RGB = 0x1907
GL_UNSIGNED_BYTE = 0x1401
GL_T2F_V3F = 0x2A27
GL_TRIANGLES = 0x0004


class GpuControls:
    def __init__(self) -> None:
        cv2.namedWindow(CONTROLS_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROLS_NAME, 460, 720)
        self._add("Grid", CONFIG.RGBD_GRID_SIZE, 160)
        self._add("Z x10", int(CONFIG.RGBD_Z_SCALE * 10), 100)
        self._add("Width cm", int(CONFIG.RGBD_PLANE_WIDTH_MM / 10), 1000)
        self._add("Min cm", int(CONFIG.RGBD_MIN_DISTANCE_MM / 10), 2000)
        self._add("Max cm", int(CONFIG.RGBD_MAX_DISTANCE_MM / 10), 2000)
        self._add("Filters", int(CONFIG.DEPTH_ENABLE_ALL_FILTERS), 1)
        self._add("Z smooth x10", int(CONFIG.RGBD_SMOOTHING_STRENGTH * 10), 50)
        self._add("Gap fill", int(CONFIG.DEPTH_GRADIENT_FILL_ALL_GAPS), 1)
        self._add("Inpaint x10", int(CONFIG.DEPTH_INPAINT_RADIUS * 10), 100)
        self._add("Median", int(CONFIG.DEPTH_MEDIAN_FILTER), 1)
        self._add("Med kernel", CONFIG.DEPTH_MEDIAN_KERNEL_SIZE, 15)
        self._add("Bilateral", int(CONFIG.DEPTH_SMOOTH_DISPARITY), 1)
        self._add("Bil diam", CONFIG.DEPTH_BILATERAL_DIAMETER, 25)
        self._add("Bil color", int(CONFIG.DEPTH_BILATERAL_SIGMA_COLOR), 100)
        self._add("Bil space", int(CONFIG.DEPTH_BILATERAL_SIGMA_SPACE), 100)
        self._add("Speckles", int(CONFIG.DEPTH_FILTER_SPECKLES), 1)
        self._add("Morph fill", int(CONFIG.DEPTH_FILL_MORPHOLOGICAL_HOLES), 1)
        self._add("Contour fill", int(CONFIG.DEPTH_FILL_CONTOUR_HOLES), 1)

    def _add(self, name: str, value: int, maximum: int) -> None:
        cv2.createTrackbar(name, CONTROLS_NAME, min(max(0, value), maximum), maximum, lambda _value: None)

    def values(self) -> tuple[int, float, float, float, float, depth_v2.DepthFilterSettings, float]:
        get = lambda name: cv2.getTrackbarPos(name, CONTROLS_NAME)
        z_smoothing = get("Z smooth x10") / 10.0
        median_kernel = max(3, get("Med kernel") | 1)
        bilateral_diameter = max(1, get("Bil diam") | 1)
        settings = depth_v2.DepthFilterSettings(
            bool(get("Filters")), bool(get("Speckles")), CONFIG.DEPTH_SPECKLE_MAX_SIZE_PX,
            CONFIG.DEPTH_SPECKLE_DISPARITY_RANGE, bool(get("Morph fill")),
            CONFIG.DEPTH_MORPHOLOGICAL_KERNEL_SIZE, bool(get("Contour fill")),
            bool(get("Gap fill")), CONFIG.DEPTH_MAX_HOLE_AREA_PX,
            get("Inpaint x10") / 10.0, bool(get("Median")), median_kernel,
            CONFIG.DEPTH_MEDIAN_MIN_VALID_RATIO, bool(get("Bilateral")), bilateral_diameter,
            float(get("Bil color")), float(get("Bil space")),
        )
        minimum = get("Min cm") * 10.0
        maximum = max(minimum + 10.0, get("Max cm") * 10.0)
        return max(5, get("Grid")), max(0.1, get("Z x10") / 10.0), max(100.0, get("Width cm") * 10.0), minimum, maximum, settings, z_smoothing


def triangle_buffer(vertices: np.ndarray, vertex_valid: np.ndarray) -> np.ndarray:
    """Create one interleaved UV/XYZ OpenGL buffer for all valid mesh faces."""
    rows, columns = vertices.shape[:2]
    u, v = np.meshgrid(np.linspace(0.0, 1.0, columns, dtype=np.float32), np.linspace(1.0, 0.0, rows, dtype=np.float32))
    uv = np.dstack((u, v))
    cells = vertex_valid[:-1, :-1] & vertex_valid[:-1, 1:] & vertex_valid[1:, 1:] & vertex_valid[1:, :-1]
    if not np.any(cells):
        return np.empty((0, 5), dtype=np.float32)
    positions = np.stack((vertices[:-1, :-1], vertices[:-1, 1:], vertices[1:, 1:], vertices[:-1, :-1], vertices[1:, 1:], vertices[1:, :-1]), axis=2)[cells]
    texcoords = np.stack((uv[:-1, :-1], uv[:-1, 1:], uv[1:, 1:], uv[:-1, :-1], uv[1:, 1:], uv[1:, :-1]), axis=2)[cells]
    return np.ascontiguousarray(np.concatenate((texcoords, positions), axis=2).reshape(-1, 5), dtype=np.float32)


class OpenGLRenderer:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption(WINDOW_NAME)
        gl = ctypes.WinDLL("opengl32")
        self.gl_clear_color = self._bind(gl, "glClearColor", None, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
        self.gl_enable = self._bind(gl, "glEnable", None, ctypes.c_uint)
        self.gl_depth_func = self._bind(gl, "glDepthFunc", None, ctypes.c_uint)
        self.gl_clear = self._bind(gl, "glClear", None, ctypes.c_uint)
        self.gl_viewport = self._bind(gl, "glViewport", None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        self.gl_matrix_mode = self._bind(gl, "glMatrixMode", None, ctypes.c_uint)
        self.gl_load_identity = self._bind(gl, "glLoadIdentity", None)
        self.gl_frustum = self._bind(gl, "glFrustum", None, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
        self.gl_translate = self._bind(gl, "glTranslatef", None, ctypes.c_float, ctypes.c_float, ctypes.c_float)
        self.gl_rotate = self._bind(gl, "glRotatef", None, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float)
        self.gl_scale = self._bind(gl, "glScalef", None, ctypes.c_float, ctypes.c_float, ctypes.c_float)
        self.gl_gen_textures = self._bind(gl, "glGenTextures", None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.gl_bind_texture = self._bind(gl, "glBindTexture", None, ctypes.c_uint, ctypes.c_uint)
        self.gl_tex_parameter = self._bind(gl, "glTexParameteri", None, ctypes.c_uint, ctypes.c_uint, ctypes.c_int)
        self.gl_tex_image = self._bind(gl, "glTexImage2D", None, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p)
        self.gl_interleaved_arrays = self._bind(gl, "glInterleavedArrays", None, ctypes.c_uint, ctypes.c_int, ctypes.c_void_p)
        self.gl_draw_arrays = self._bind(gl, "glDrawArrays", None, ctypes.c_uint, ctypes.c_int, ctypes.c_int)
        self.texture = ctypes.c_uint()
        self.gl_gen_textures(1, ctypes.byref(self.texture))
        self.gl_bind_texture(GL_TEXTURE_2D, self.texture.value)
        self.gl_tex_parameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        self.gl_tex_parameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.gl_enable(GL_DEPTH_TEST)
        self.gl_enable(GL_TEXTURE_2D)
        self.gl_depth_func(GL_LEQUAL)
        self.gl_clear_color(0.12, 0.12, 0.14, 1.0)

    @staticmethod
    def _bind(library: ctypes.WinDLL, name: str, restype: object, *argtypes: object) -> object:
        function = getattr(library, name)
        function.restype = restype
        function.argtypes = argtypes
        return function

    def events(self, view: ViewState) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEWHEEL:
                view.zoom = min(5.0, max(0.2, view.zoom * (1.12 if event.y > 0 else 1.0 / 1.12)))
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key == pygame.K_a:
                    view.yaw_deg -= 5.0
                elif event.key == pygame.K_d:
                    view.yaw_deg += 5.0
                elif event.key == pygame.K_w:
                    view.pitch_deg -= 5.0
                elif event.key == pygame.K_s:
                    view.pitch_deg += 5.0
                elif event.key == pygame.K_z:
                    view.roll_deg -= 5.0
                elif event.key == pygame.K_x:
                    view.roll_deg += 5.0
                elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                    view.zoom = min(5.0, view.zoom * 1.15)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    view.zoom = max(0.2, view.zoom / 1.15)
                elif event.key == pygame.K_r:
                    view.yaw_deg = view.pitch_deg = view.roll_deg = 0.0
                    view.zoom = 1.2
        return True

    def draw(self, texture_bgr: np.ndarray, vertices: np.ndarray, vertex_valid: np.ndarray, view: ViewState) -> None:
        width, height = pygame.display.get_window_size()
        self.gl_viewport(0, 0, width, height)
        self.gl_clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not np.any(vertex_valid):
            pygame.display.flip()
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB))
        self.gl_bind_texture(GL_TEXTURE_2D, self.texture.value)
        self.gl_tex_image(GL_TEXTURE_2D, 0, GL_RGB, rgb.shape[1], rgb.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.ctypes.data_as(ctypes.c_void_p))
        valid_points = vertices[vertex_valid]
        center = np.median(valid_points, axis=0)
        extent = max(1.0, float(np.percentile(np.linalg.norm(valid_points - center, axis=1), 95)))
        aspect = width / max(1, height)
        self.gl_matrix_mode(GL_PROJECTION)
        self.gl_load_identity()
        self.gl_frustum(-0.75 * aspect, 0.75 * aspect, -0.75, 0.75, 1.0, 50.0)
        self.gl_matrix_mode(GL_MODELVIEW)
        self.gl_load_identity()
        self.gl_translate(0.0, 0.0, -4.0 / view.zoom)
        self.gl_rotate(view.pitch_deg, 1.0, 0.0, 0.0)
        self.gl_rotate(view.yaw_deg, 0.0, 1.0, 0.0)
        self.gl_rotate(view.roll_deg, 0.0, 0.0, 1.0)
        self.gl_scale(1.0 / extent, 1.0 / extent, 1.0 / extent)
        self.gl_translate(-float(center[0]), -float(center[1]), -float(center[2]))
        buffer = triangle_buffer(vertices, vertex_valid)
        if buffer.size:
            self.gl_interleaved_arrays(GL_T2F_V3F, 0, buffer.ctypes.data_as(ctypes.c_void_p))
            self.gl_draw_arrays(GL_TRIANGLES, 0, len(buffer))
        pygame.display.set_caption(f"{WINDOW_NAME} | {vertices.shape[0]}x{vertices.shape[1]} vertices | {len(buffer) // 3} triangles")
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU OpenGL RGBD textured height-field.")
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
    renderer = OpenGLRenderer()
    controls = GpuControls()
    view = ViewState()
    left_camera.open()
    right_camera.open()
    if not depth_v2.warmup_camera_pair(left_camera, right_camera):
        raise RuntimeError("Both cameras opened but cannot deliver frames together")
    print("GPU renderer started. W/S pitch, A/D yaw, Z/X roll, mouse wheel or +/- zoom, R reset, Q/Esc quit.")
    try:
        running = True
        while running:
            running = renderer.events(view)
            cv2.waitKey(1)
            ok, left, right = depth_v2.read_camera_pair(left_camera, right_camera)
            if not ok or left is None or right is None:
                continue
            grid, z_scale, plane_width, minimum, maximum, filters, _z_smoothing = controls.values()
            rectified, _, _disparity, distance_mm, valid = depth_v2.compute_depth_frame(depth, left, right, filters, cuda_matcher, parallel_bm_matcher)
            valid &= (distance_mm >= minimum) & (distance_mm <= maximum)
            distance_mm, valid = depth_v2.fill_distance_gaps(distance_mm, valid, filters, minimum, maximum)
            vertices, _colours, vertex_valid = depth_mesh(rectified, distance_mm, valid, grid, z_scale, plane_width)
            renderer.draw(rectified, vertices, vertex_valid, view)
    finally:
        left_camera.release()
        right_camera.release()
        if parallel_bm_matcher is not None:
            parallel_bm_matcher.close()
        renderer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

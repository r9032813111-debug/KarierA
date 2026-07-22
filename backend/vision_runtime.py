"""Headless ArUco camera pipeline with rectified MJPEG output."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
import yaml

from .robot_controller import RobotPose, WebTargetController


CONFIG_PATH = Path(__file__).with_name("hardware.yaml")
CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


def load_hardware_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise RuntimeError("hardware.yaml must contain a mapping")
    return config


def _aruco_dictionary(name: str) -> Any:
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise RuntimeError(f"Unsupported ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _detect(frame: np.ndarray, dictionary: Any) -> tuple[list[np.ndarray], np.ndarray | None]:
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids


def _records(corners: list[np.ndarray], ids: np.ndarray | None) -> dict[int, np.ndarray]:
    if ids is None:
        return {}
    return {
        int(marker_id): marker_corners.reshape(4, 2).astype(np.float32)
        for marker_corners, marker_id in zip(corners, ids.flatten())
    }


def _order_corner_records(records: dict[int, np.ndarray]) -> dict[str, tuple[int, np.ndarray]]:
    items = [(marker_id, points, np.mean(points, axis=0)) for marker_id, points in records.items()]
    by_y = sorted(items, key=lambda item: float(item[2][1]))
    top = sorted(by_y[:2], key=lambda item: float(item[2][0]))
    bottom = sorted(by_y[-2:], key=lambda item: float(item[2][0]))
    return {
        "top_left": (top[0][0], top[0][1]),
        "top_right": (top[1][0], top[1][1]),
        "bottom_right": (bottom[1][0], bottom[1][1]),
        "bottom_left": (bottom[0][0], bottom[0][1]),
    }


def _inward_polygon(ordered: dict[str, tuple[int, np.ndarray]]) -> list[tuple[int, int]]:
    centers = [np.mean(ordered[name][1], axis=0) for name in CORNER_NAMES]
    field_center = np.mean(centers, axis=0)
    polygon: list[tuple[int, int]] = []
    for name in CORNER_NAMES:
        points = ordered[name][1]
        inward = points[int(np.argmin(np.linalg.norm(points - field_center, axis=1)))]
        polygon.append((int(round(float(inward[0]))), int(round(float(inward[1])))))
    return polygon


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    shaped = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(shaped, transform)[0]


def _marker_top_heading(points: np.ndarray, offset_deg: float = 0.0) -> float:
    """Heading from ArUco's upper edge: the printed marker top is robot front."""
    corners = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = np.mean(corners, axis=0)
    top_edge_center = np.mean(corners[:2], axis=0)
    forward = top_edge_center - center
    return float((np.degrees(np.arctan2(float(forward[1]), float(forward[0]))) + offset_deg) % 360.0)


def _estimate_polygon(
    reference_centers: dict[int, tuple[float, float]],
    visible_records: dict[int, np.ndarray],
    reference_polygon: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    common = sorted(set(reference_centers) & set(visible_records))
    if not common:
        return None
    source = np.asarray([reference_centers[item] for item in common], dtype=np.float32)
    target = np.asarray([np.mean(visible_records[item], axis=0) for item in common], dtype=np.float32)
    polygon = np.asarray(reference_polygon, dtype=np.float32).reshape(1, -1, 2)
    transformed: np.ndarray | None = None
    if len(common) >= 4:
        homography, _ = cv2.findHomography(source, target, method=cv2.RANSAC)
        if homography is not None:
            transformed = cv2.perspectiveTransform(polygon, homography)[0]
    elif len(common) >= 2:
        affine, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
        if affine is not None:
            transformed = cv2.transform(polygon, affine)[0]
    else:
        transformed = polygon[0] + (target[0] - source[0])
    if transformed is None:
        return None
    return [(int(round(float(x))), int(round(float(y)))) for x, y in transformed]


class VisionRuntime:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_hardware_config()
        self.enabled = bool(self.config.get("hardware_enabled", True))
        # Try the configured camera first, then look for any working USB camera.
        # `camera_index: -1` means "auto" and starts scanning from zero.
        self.camera_index = int(self.config.get("camera_index", -1))
        self.camera_scan_limit = max(1, int(self.config.get("camera_scan_limit", 6)))
        self.frame_width = int(self.config.get("frame_width", 1280))
        self.frame_height = int(self.config.get("frame_height", 720))
        self.stream_width = int(self.config.get("stream_width", 960))
        self.stream_height = int(self.config.get("stream_height", 640))
        self.jpeg_quality = int(self.config.get("stream_jpeg_quality", 78))
        self.target_fps = float(self.config.get("stream_fps", 20))
        self.marker_size_cm = float(self.config.get("marker_size_cm", 6.0))
        # A missed camera frame must not immediately drop the crop or pose.
        self.marker_hold_seconds = max(0.0, float(self.config.get("marker_hold_seconds", 1.0)))
        self.heading_offset = float(self.config.get("robot_heading_offset_deg", 0.0))
        configured_agent_id = self.config.get("controlled_agent_id")
        self.controlled_agent_id: int | None = int(configured_agent_id) if configured_agent_id is not None else None
        self.dictionary = _aruco_dictionary(str(self.config.get("aruco_dictionary", "DICT_4X4_100")))
        backend_name = str(self.config.get("camera_backend", "CAP_DSHOW"))
        self.camera_backend = int(getattr(cv2, backend_name, cv2.CAP_ANY))
        self.controller = WebTargetController(self.config)

        self._marker_roles: dict[int, str] = {}
        self._thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition(threading.RLock())
        self._jpeg: bytes | None = self._encode_placeholder("CAMERA OFFLINE", "Waiting for USB camera")
        self._frame_sequence = 0
        self._snapshot: dict[str, Any] = self._offline_snapshot("Camera is not connected")
        self._reference_centers: dict[int, tuple[float, float]] = {}
        self._reference_polygon: list[tuple[int, int]] | None = None
        self._last_polygon: list[tuple[int, int]] | None = None
        self._last_scale: float | None = None
        self._marker_cache: dict[int, tuple[np.ndarray, float]] = {}
        self._control_observation: tuple[RobotPose | None, float | None, float | None, float] = (
            None, None, None, time.monotonic()
        )
        self._frame_provider: Callable[[], tuple[np.ndarray, int] | None] | None = None
        self._external_frame_sequence = -1

    def set_frame_provider(
        self, provider: Callable[[], tuple[np.ndarray, int] | None] | None
    ) -> None:
        """Use frames captured by the stereo runtime instead of reopening camera 0."""
        self._frame_provider = provider

    def _offline_snapshot(self, error: str = "") -> dict[str, Any]:
        return {
            "camera": {
                "online": False,
                "calibrated": False,
                "index": self.camera_index,
                "fps": 0.0,
                "tracking": "offline",
                "field_width_cm": None,
                "field_height_cm": None,
                "error": error,
            },
            "detected_markers": [],
            "agents": [],
            "robot": self.controller.status() if hasattr(self, "controller") else {},
        }

    def _encode_placeholder(self, title: str, copy: str) -> bytes | None:
        frame = np.full((self.stream_height, self.stream_width, 3), (231, 233, 226), dtype=np.uint8)
        center_x, center_y = self.stream_width // 2, self.stream_height // 2
        cv2.circle(frame, (center_x, center_y - 42), 28, (190, 196, 188), 2, cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y - 42), 7, (61, 69, 63), -1, cv2.LINE_AA)
        for text, y, scale, color, thickness in (
            (title, center_y + 20, .85, (45, 53, 47), 2),
            (copy, center_y + 55, .55, (105, 114, 107), 1),
        ):
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
            cv2.putText(frame, text, (center_x - size[0] // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return encoded.tobytes() if ok else None

    def _publish_offline(self, error: str) -> None:
        jpeg = self._encode_placeholder("CAMERA OFFLINE", error[:48] or "Waiting for USB camera")
        with self._condition:
            if jpeg is not None:
                self._jpeg = jpeg
            self._snapshot = self._offline_snapshot(error)
            self._frame_sequence += 1
            self._condition.notify_all()

    def set_marker_config(self, markers: list[dict[str, Any]]) -> None:
        with self._condition:
            roles = {int(marker["id"]): str(marker.get("role", "unassigned")) for marker in markers}
            agent_ips = {
                int(marker["id"]): str(marker.get("ip_address", "")).strip()
                for marker in markers
                if marker.get("role") == "agent" and str(marker.get("ip_address", "")).strip()
            }
            self.controller.set_agent_ips(agent_ips)
            if self._marker_roles and roles != self._marker_roles:
                self._reference_centers = {}
                self._reference_polygon = None
                self._last_polygon = None
                self._last_scale = None
                self.controller.reset_calibration()
            self._marker_roles = roles

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="aruco-vision", daemon=True)
        self._control_thread = threading.Thread(target=self._run_controller, name="robot-control", daemon=True)
        self._thread.start()
        self._control_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._control_thread is not None:
            self._control_thread.join(timeout=3.0)
        self.controller.close()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "camera": dict(self._snapshot["camera"]),
                "detected_markers": [dict(item) for item in self._snapshot["detected_markers"]],
                "agents": [dict(item) for item in self._snapshot["agents"]],
                "robot": dict(self._snapshot["robot"]),
            }

    def set_target(self, agent_id: int, x: float, y: float, **options: Any) -> tuple[bool, str]:
        self.select_agent(agent_id)
        return self.controller.set_target(agent_id, x, y, **options)

    def select_agent(self, agent_id: int, manual: bool = False) -> bool:
        selected_id = int(agent_id)
        with self._condition:
            self.controlled_agent_id = selected_id
            self._control_observation = (None, None, None, time.monotonic())
        changed = self.controller.select_agent(selected_id)
        self.controller.set_manual_mode(manual)
        return changed

    def set_debug_enabled(self, enabled: bool) -> None:
        self.controller.set_debug_enabled(enabled)

    def emergency_stop(self) -> None:
        self.controller.emergency_stop()

    def mjpeg(self) -> Iterator[bytes]:
        sequence = -1
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._frame_sequence != sequence or self._stop_event.is_set(), timeout=1.0)
                if self._stop_event.is_set():
                    return
                sequence = self._frame_sequence
                jpeg = self._jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpeg + b"\r\n"

    def latest_jpeg(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def aruco_marker_png(self, marker_id: int, size: int = 160) -> bytes:
        """Generate the exact marker image from the active OpenCV dictionary."""
        marker = cv2.aruco.generateImageMarker(self.dictionary, marker_id, size, borderBits=1)
        ok, encoded = cv2.imencode(".png", marker)
        if not ok:
            raise RuntimeError("OpenCV failed to encode ArUco marker")
        return encoded.tobytes()

    def _camera_indices(self) -> list[int]:
        """Configured camera first, then every common local camera index."""
        indices = list(range(self.camera_scan_limit))
        if self.camera_index >= 0:
            indices.insert(0, self.camera_index)
        return list(dict.fromkeys(indices))

    def _open_camera(self) -> tuple[cv2.VideoCapture | None, int | None]:
        """Find the first camera that can actually deliver a frame."""
        for index in self._camera_indices():
            capture = cv2.VideoCapture(index, self.camera_backend)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                capture.release()
                continue
            ok, _ = capture.read()
            if ok:
                self.camera_index = index
                return capture, index
            capture.release()
        return None, None

    def _publish_frame(self, frame: np.ndarray, snapshot: dict[str, Any]) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._snapshot = snapshot
            self._frame_sequence += 1
            self._condition.notify_all()

    def _set_control_observation(
        self,
        pose: RobotPose | None,
        width_cm: float | None,
        height_cm: float | None,
    ) -> None:
        with self._condition:
            self._control_observation = (pose, width_cm, height_cm, time.monotonic())

    def _run_controller(self) -> None:
        """Keep slow network reconnects away from the camera capture loop.

        The latest camera pose is reused between frames. A 100 Hz service
        cycle and 50 Hz motor refresh keep the ESP32 watchdog fed without
        creating a TCP backlog in front of an emergency STOP.
        """
        while not self._stop_event.is_set():
            with self._condition:
                pose, width_cm, height_cm, observed_at = self._control_observation
            if time.monotonic() - observed_at > 0.25:
                pose, width_cm, height_cm = None, None, None
            self.controller.update(pose, width_cm, height_cm)
            time.sleep(0.01)

    def _selected_vertex_ids(self, records: dict[int, np.ndarray]) -> list[int]:
        configured = [marker_id for marker_id, role in self._marker_roles.items() if role == "vertex"]
        visible = [marker_id for marker_id in configured if marker_id in records]
        if len(visible) == 4:
            return visible
        # Never crop by a guessed polygon. Until the operator explicitly assigns
        # four visible vertices, the stream remains the complete camera frame.
        return []

    def _records_with_hold(
        self, records: dict[int, np.ndarray], now: float
    ) -> tuple[dict[int, np.ndarray], set[int]]:
        """Keep the last marker corners during a short, bounded detection loss."""
        for marker_id, corners in records.items():
            self._marker_cache[marker_id] = (corners.copy(), now)

        combined = dict(records)
        held_ids: set[int] = set()
        for marker_id, (corners, last_seen_at) in list(self._marker_cache.items()):
            if marker_id in combined:
                continue
            if now - last_seen_at <= self.marker_hold_seconds:
                combined[marker_id] = corners.copy()
                held_ids.add(marker_id)
            else:
                del self._marker_cache[marker_id]
        return combined, held_ids

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        retry_at = 0.0
        frame_times: list[float] = []
        while not self._stop_event.is_set():
            started = time.monotonic()
            if self._frame_provider is not None:
                if capture is not None:
                    capture.release()
                    capture = None
                supplied = self._frame_provider()
                if supplied is None:
                    self._publish_offline("Waiting for stereo camera pair")
                    self._set_control_observation(None, None, None)
                    time.sleep(0.05)
                    continue
                frame, sequence = supplied
                if sequence == self._external_frame_sequence:
                    time.sleep(0.002)
                    continue
                self._external_frame_sequence = sequence
            else:
                if capture is None or not capture.isOpened():
                    now = time.monotonic()
                    if now < retry_at:
                        time.sleep(0.1)
                        continue
                    retry_at = now + 2.0
                    if capture is not None:
                        capture.release()
                    capture, index = self._open_camera()
                    if capture is None:
                        self._publish_offline(f"No camera found (checked indices 0-{self.camera_scan_limit - 1})")
                        self._set_control_observation(None, None, None)
                        continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.release()
                    capture = None
                    self._publish_offline("Camera frame was lost")
                    self._set_control_observation(None, None, None)
                    continue

            corners, ids = _detect(frame, self.dictionary)
            live_records = _records(corners, ids)
            records, held_ids = self._records_with_hold(live_records, started)
            vertex_ids = self._selected_vertex_ids(records)
            polygon: list[tuple[int, int]] | None = None
            tracking = "waiting"

            if len(vertex_ids) == 4:
                ordered = _order_corner_records({marker_id: records[marker_id] for marker_id in vertex_ids})
                polygon = _inward_polygon(ordered)
                tracked_ids = {
                    marker_id for marker_id, role in self._marker_roles.items()
                    if role in {"vertex", "edge"} and marker_id in records
                } | set(vertex_ids)
                self._reference_centers = {
                    marker_id: tuple(float(value) for value in np.mean(records[marker_id], axis=0))
                    for marker_id in tracked_ids
                }
                self._reference_polygon = polygon
                self._last_polygon = polygon
                tracking = "held" if held_ids.intersection(vertex_ids) else "live"
            elif self._reference_polygon is not None:
                estimated = _estimate_polygon(self._reference_centers, records, self._reference_polygon)
                if estimated is not None:
                    polygon = estimated
                    self._last_polygon = estimated
                    tracking = "estimated"
                elif self._last_polygon is not None:
                    polygon = self._last_polygon
                    tracking = "cached"

            output = cv2.resize(frame, (self.stream_width, self.stream_height))
            transform: np.ndarray | None = None
            field_width_cm: float | None = None
            field_height_cm: float | None = None
            # Cached corners are usable for marker_hold_seconds, preventing a
            # short recognition gap from dropping the rectified crop.
            vertices_ready = polygon is not None and len(vertex_ids) == 4
            if vertices_ready:
                source = np.asarray(polygon, dtype=np.float32)
                destination = np.asarray([
                    (0, 0), (self.stream_width - 1, 0),
                    (self.stream_width - 1, self.stream_height - 1), (0, self.stream_height - 1),
                ], dtype=np.float32)
                transform = cv2.getPerspectiveTransform(source, destination)
                output = cv2.warpPerspective(frame, transform, (self.stream_width, self.stream_height))
                side_lengths: list[float] = []
                for marker_id in vertex_ids:
                    rectified = _transform_points(records[marker_id], transform)
                    edges = np.roll(rectified, -1, axis=0) - rectified
                    side_lengths.extend(float(length) for length in np.linalg.norm(edges, axis=1))
                if side_lengths and float(np.mean(side_lengths)) > 0:
                    self._last_scale = self.marker_size_cm / float(np.mean(side_lengths))
                if self._last_scale is not None:
                    field_width_cm = self.stream_width * self._last_scale
                    field_height_cm = self.stream_height * self._last_scale

            detected: list[dict[str, Any]] = []
            agents: list[dict[str, Any]] = []
            controlled_pose: RobotPose | None = None
            for marker_id, marker_corners in records.items():
                if transform is not None:
                    projected = _transform_points(marker_corners, transform)
                    center = np.mean(projected, axis=0)
                    x_norm = float(np.clip(center[0] / self.stream_width * 100.0, 0, 100))
                    y_norm = float(np.clip(100.0 - center[1] / self.stream_height * 100.0, 0, 100))
                else:
                    projected = marker_corners
                    center = np.mean(projected, axis=0)
                    x_norm = float(np.clip(center[0] / frame.shape[1] * 100.0, 0, 100))
                    y_norm = float(np.clip(100.0 - center[1] / frame.shape[0] * 100.0, 0, 100))
                role = self._marker_roles.get(marker_id, "unassigned")
                reading = {
                    "id": marker_id, "x": x_norm, "y": y_norm, "role": role,
                    "seen": True, "held": marker_id in held_ids,
                }
                detected.append(reading)
                if role == "agent" or marker_id == self.controlled_agent_id:
                    offset = self.controller.heading_offset_for(marker_id, self.heading_offset)
                    heading = _marker_top_heading(projected, offset)
                    agent = {**reading, "heading": float(heading)}
                    agents.append(agent)
                    if marker_id == self.controlled_agent_id and self._last_scale is not None and transform is not None:
                        controlled_pose = RobotPose(
                            x_cm=float(center[0]) * self._last_scale,
                            y_cm=float(center[1]) * self._last_scale,
                            heading_deg=float(heading),
                        )
                    cv2.circle(output, tuple(np.rint(center).astype(int)), 13, (55, 55, 220), 2, cv2.LINE_AA)
                    cv2.putText(output, f"ID {marker_id}", tuple(np.rint(center + (16, -12)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 40, 220), 2, cv2.LINE_AA)

            self._set_control_observation(controlled_pose, field_width_cm, field_height_cm)
            robot_status = self.controller.status()
            frame_times.append(started)
            frame_times = [value for value in frame_times if started - value <= 1.0]
            fps = float(len(frame_times))
            calibrated = vertices_ready and transform is not None and self._last_scale is not None
            status_text = f"CAM {self.camera_index}  {tracking.upper()}  {fps:.0f} FPS"
            cv2.rectangle(output, (0, 0), (self.stream_width, 34), (18, 20, 18), -1)
            cv2.putText(output, status_text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(output, robot_status["phase"], (self.stream_width - 190, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 90, 230), 2, cv2.LINE_AA)
            snapshot = {
                "camera": {
                    "online": True,
                    "calibrated": calibrated,
                    "index": self.camera_index,
                    "fps": fps,
                    "tracking": tracking,
                    "field_width_cm": field_width_cm,
                    "field_height_cm": field_height_cm,
                    "error": "" if calibrated else "Waiting for four vertex markers",
                },
                "detected_markers": detected,
                "agents": agents,
                "robot": robot_status,
            }
            self._publish_frame(output, snapshot)
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, 1.0 / max(self.target_fps, 1.0) - elapsed))

        if capture is not None:
            capture.release()

"""Calibrated, vision-guided controller for web targets on the ESP32 robot."""

from __future__ import annotations

import math
import select
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

from .path_planner import point_has_clearance, point_in_work

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # Camera-only mode remains available.
    serial = None
    list_ports = None


@dataclass(frozen=True)
class RobotPose:
    """Robot center and forward heading in rectified field coordinates (cm)."""

    x_cm: float
    y_cm: float
    heading_deg: float  # 0=right, 90=down, matching the ESP32 firmware.


class _TcpConnection:
    """Small pyserial-compatible TCP adapter."""

    def __init__(self, host: str, port: int) -> None:
        self.socket = socket.create_connection((host, port), timeout=0.7)
        self.socket.setblocking(False)

    @property
    def in_waiting(self) -> int:
        readable, _, _ = select.select([self.socket], [], [], 0)
        return 4096 if readable else 0

    def write(self, data: bytes) -> int:
        self.socket.sendall(data)
        return len(data)

    def read(self, size: int) -> bytes:
        try:
            return self.socket.recv(size)
        except BlockingIOError:
            return b""

    def close(self) -> None:
        self.socket.close()


def _angle_error(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


class WebTargetController:
    """Run the existing calibration once, then drive to targets supplied by the web UI."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("hardware_enabled", True))
        configured_agent_id = config.get("controlled_agent_id")
        self.controlled_agent_id: int | None = int(configured_agent_id) if configured_agent_id is not None else None
        self.inset_cm = float(config.get("navigation_inset_cm", 15.0))
        self.hard_stop_margin_cm = float(config.get("hard_stop_margin_cm", 7.0))
        self.target_tolerance_cm = float(config.get("target_tolerance_cm", 3.0))
        self.heading_tolerance_deg = float(config.get("heading_tolerance_deg", 12.0))
        self.heading_kp = float(config.get("heading_kp", 1.8))
        self.heading_kd = float(config.get("heading_kd", 0.18))
        self.min_drive_pwm = max(0, min(250, int(config.get("min_drive_pwm", 80))))
        self.in_place_turn_threshold_deg = max(1.0, min(180.0, float(config.get("in_place_turn_threshold_deg", 90.0))))
        self.heading_offset_deg = max(-180.0, min(180.0, float(config.get("robot_heading_offset_deg", 0.0))))
        self.left_motor_inverted = bool(config.get("left_motor_inverted", False))
        self.right_motor_inverted = bool(config.get("right_motor_inverted", False))
        self.invert_turn_direction = bool(config.get("invert_turn_direction", True))
        self.arc_turn_single_motor = bool(config.get("arc_turn_single_motor", True))
        self.stuck_timeout_s = max(0.5, float(config.get("stuck_timeout_s", 2.5)))
        self.stuck_min_progress_cm = max(0.1, float(config.get("stuck_min_progress_cm", 1.5)))
        self.stuck_action = str(config.get("stuck_action", "stop"))
        self.stuck_boost_pwm = max(1, min(150, int(config.get("stuck_boost_pwm", 35))))
        self.pose_smoothing_alpha = min(1.0, max(0.01, float(config.get("pose_smoothing_alpha", 0.25))))
        self.speed = int(config.get("robot_speed", 255))
        self.auto_calibrate = bool(config.get("auto_calibrate", True))
        self.calibration_forward_seconds = float(config.get("calibration_forward_seconds", 2.0))
        self.calibration_turn_90_seconds = float(config.get("calibration_turn_90_seconds", 1.0))
        self.calibration_turn_180_seconds = float(config.get("calibration_turn_180_seconds", 2.0))
        # TCP preserves order, so flooding it can place an emergency STOP
        # behind stale DRIVE commands. 50 Hz is the safe refresh ceiling.
        self.command_interval = max(0.02, float(config.get("serial_command_interval_s", 0.02)))
        self.transport = str(config.get("robot_transport", "wifi")).lower()
        self.wifi_host = str(config.get("robot_wifi_host", "")).strip()
        self.wifi_port = int(config.get("robot_wifi_port", 3333))
        self.port_name = config.get("serial_port")
        self.baud = int(config.get("serial_baud", 115200))
        self._default_controller_settings: dict[str, Any] = {
            "speed": self.speed,
            "target_tolerance_cm": self.target_tolerance_cm,
            "heading_tolerance_deg": self.heading_tolerance_deg,
            "heading_kp": self.heading_kp,
            "heading_kd": self.heading_kd,
            "min_drive_pwm": self.min_drive_pwm,
            "in_place_turn_threshold_deg": self.in_place_turn_threshold_deg,
            "heading_offset_deg": self.heading_offset_deg,
            "left_motor_inverted": self.left_motor_inverted,
            "right_motor_inverted": self.right_motor_inverted,
            "stuck_timeout_s": self.stuck_timeout_s,
            "stuck_min_progress_cm": self.stuck_min_progress_cm,
            "stuck_action": self.stuck_action,
            "stuck_boost_pwm": self.stuck_boost_pwm,
        }

        self.connection: Any | None = None
        # A manually supplied connection (serial or tests) is already usable;
        # a newly opened Wi-Fi socket is explicitly marked unconfirmed below.
        self._link_confirmed = True
        self._connection_opened_at = 0.0
        self.phase = "WAIT_FIELD"
        self.calibrated = False
        self.last_command = "STOP"
        self.last_error = ""
        self.last_response = ""
        self._response_buffer = ""
        # Values confirmed by ESP32, not values merely requested by the server.
        self._esp_motor_pwm: tuple[int, int] | None = None
        self._esp_motor_command = "STOP"
        self._requested_servo_angles: dict[int, int] = {}
        self._esp_servo_state: dict[int, dict[str, int | bool | None]] = {
            1: {"angle": None, "active": False},
            2: {"angle": None, "active": False},
        }
        self.last_send = 0.0
        self.last_motion_send = 0.0
        self._last_drive_pwm: tuple[int, int] = (0, 0)
        self._motor_inversion_signature: tuple[bool, bool] | None = None
        self.next_reconnect_time = 0.0
        self.phase_started = 0.0
        self.phase_command_sent = False
        self.calibration_start_pose: RobotPose | None = None
        self.calibration_max_distance_cm = 0.0
        self.calibration_max_turn_deg = 0.0
        self.linear_speed_cm_s: float | None = None
        self.angular_speed_deg_s: float | None = None
        self._bounds_signature: tuple[float, float] | None = None
        self._vision_lost_since: float | None = None
        self._target_normalized: tuple[float, float] | None = None
        self._target_cm: tuple[float, float] | None = None
        self._debug_mode = False
        self._command_speed = self.speed
        self._speed_limit = self.speed
        self._speed_signature: int | None = None
        self._previous_heading_error: float | None = None
        self._previous_heading_at: float | None = None
        self._smoothed_position: tuple[float, float] | None = None
        self._stuck_reference_position: tuple[float, float] | None = None
        self._stuck_reference_at: float | None = None
        self._active_stuck_boost_pwm = 0
        self._route_normalized: list[tuple[float, float]] = []
        self._route_index = 0
        self._work_regions: list[dict[str, Any]] = []
        self._work_clearance = 1.4
        self._agent_settings: dict[int, dict[str, Any]] = {}
        self._agent_ips: dict[int, str] = {}
        self._active_wifi_host = ""
        self._lock = threading.RLock()
        self._emergency_stop_requested = threading.Event()
        self._emergency_stop_sent = False
        self._resume_confirmation_pending = False
        self._manual_enabled = False
        self._manual_command: tuple[int, int] | None = None
        self._manual_expires_at = 0.0
        self._manual_deadman_timeout_s = 0.45

    def _settings_for(self, agent_id: int) -> dict[str, Any]:
        return {**self._default_controller_settings, **self._agent_settings.get(int(agent_id), {})}

    def _apply_agent_settings(self, agent_id: int) -> None:
        settings = self._settings_for(agent_id)
        self.speed = int(settings["speed"])
        self.target_tolerance_cm = float(settings["target_tolerance_cm"])
        self.heading_tolerance_deg = float(settings["heading_tolerance_deg"])
        self.heading_kp = float(settings["heading_kp"])
        self.heading_kd = float(settings["heading_kd"])
        self.min_drive_pwm = int(settings["min_drive_pwm"])
        self.in_place_turn_threshold_deg = float(settings["in_place_turn_threshold_deg"])
        self.heading_offset_deg = float(settings["heading_offset_deg"])
        self.left_motor_inverted = bool(settings["left_motor_inverted"])
        self.right_motor_inverted = bool(settings["right_motor_inverted"])
        self.stuck_timeout_s = float(settings["stuck_timeout_s"])
        self.stuck_min_progress_cm = float(settings["stuck_min_progress_cm"])
        self.stuck_action = str(settings["stuck_action"])
        self.stuck_boost_pwm = int(settings["stuck_boost_pwm"])

    def set_agent_settings(self, agent_id: int, *, speed: int, target_tolerance_cm: float, heading_tolerance_deg: float, heading_kp: float = 1.8, heading_kd: float = 0.18, min_drive_pwm: int = 80, in_place_turn_threshold_deg: float = 90.0, heading_offset_deg: float = 0.0, left_motor_inverted: bool = False, right_motor_inverted: bool = False, stuck_timeout_s: float = 2.5, stuck_min_progress_cm: float = 1.5, stuck_action: str = "stop", stuck_boost_pwm: int = 35) -> dict[str, Any]:
        """Save safe trajectory-controller parameters for one ArUco agent."""
        with self._lock:
            settings: dict[str, Any] = {
                "speed": max(0, min(250, int(speed))),
                "target_tolerance_cm": max(1.0, min(20.0, float(target_tolerance_cm))),
                "heading_tolerance_deg": max(1.0, min(90.0, float(heading_tolerance_deg))),
                "heading_kp": max(-10.0, min(10.0, float(heading_kp))),
                "heading_kd": max(0.0, min(5.0, float(heading_kd))),
                "min_drive_pwm": max(0, min(250, int(min_drive_pwm))),
                "in_place_turn_threshold_deg": max(1.0, min(180.0, float(in_place_turn_threshold_deg))),
                "heading_offset_deg": max(-180.0, min(180.0, float(heading_offset_deg))),
                "left_motor_inverted": bool(left_motor_inverted),
                "right_motor_inverted": bool(right_motor_inverted),
                "stuck_timeout_s": max(0.5, min(30.0, float(stuck_timeout_s))),
                "stuck_min_progress_cm": max(0.1, min(20.0, float(stuck_min_progress_cm))),
                "stuck_action": str(stuck_action) if str(stuck_action) in {"stop", "boost", "continue"} else "stop",
                "stuck_boost_pwm": max(1, min(150, int(stuck_boost_pwm))),
            }
            self._agent_settings[int(agent_id)] = settings
            if self.controlled_agent_id == int(agent_id):
                self._apply_agent_settings(int(agent_id))
                self._sync_motor_inversion()
                if not self._debug_mode:
                    self._speed_limit = self.speed
                self._speed_signature = None
            return dict(settings)

    def heading_offset_for(self, agent_id: int, fallback: float = 0.0) -> float:
        with self._lock:
            settings = self._settings_for(agent_id)
            return max(-180.0, min(180.0, float(settings.get("heading_offset_deg", fallback))))

    def _close_connection(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        self.connection = None
        self._link_confirmed = False
        self._esp_motor_pwm = None
        self._requested_servo_angles.clear()
        self._esp_servo_state = {
            1: {"angle": None, "active": False},
            2: {"angle": None, "active": False},
        }
        self._emergency_stop_sent = False
        self._response_buffer = ""
        self.next_reconnect_time = 0.0

    def set_agent_ips(self, agent_ips: dict[int, str]) -> None:
        """Update LAN addresses from the saved ArUco-agent configuration."""
        with self._lock:
            cleaned = {int(agent_id): str(address).strip() for agent_id, address in agent_ips.items() if str(address).strip()}
            previous = self._agent_ips
            self._agent_ips = cleaned
            selected = self.controlled_agent_id
            if selected is not None and previous.get(selected) != cleaned.get(selected):
                self._stop("IP выбранного агента изменён — переподключение")
                self._close_connection()
                self._active_wifi_host = ""

    def _selected_wifi_host(self) -> str:
        if self.controlled_agent_id is not None:
            return self._agent_ips.get(self.controlled_agent_id, "")
        return self.wifi_host

    def _open_connection(self) -> None:
        if not self.enabled or self.connection is not None:
            return
        if self.transport == "wifi":
            try:
                host = self._selected_wifi_host()
                if not host:
                    self.last_error = "Для выбранного агента не задан IP-адрес"
                    return
                self.connection = _TcpConnection(host, self.wifi_port)
                self._active_wifi_host = host
                # A TCP port can accept a client while the ESP32 command loop
                # is still occupied by an older client. Wait for a controller
                # response before treating this as a live robot connection.
                self._link_confirmed = False
                self._esp_motor_pwm = None
                self._response_buffer = ""
                self._connection_opened_at = time.monotonic()
                self.connection.write(b"STATUS\n")
                self.last_error = "Waiting for ESP32 response"
            except OSError as error:
                self.last_error = f"ESP32 Wi-Fi ({self._selected_wifi_host()}): {error}"
            return
        if serial is None or list_ports is None:
            self.last_error = "pyserial is not installed"
            return
        candidates = [self.port_name] if self.port_name else [port.device for port in list_ports.comports()]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.connection = serial.Serial(candidate, self.baud, timeout=0, write_timeout=0.2)
                time.sleep(1.5)  # CP210x resets ESP32 when the serial port opens.
                self.connection.reset_input_buffer()
                self._link_confirmed = True
                self._esp_motor_pwm = None
                self._response_buffer = ""
                self.last_error = ""
                return
            except (OSError, serial.SerialException) as error:
                self.last_error = f"ESP32 serial: {error}"

    def _send(self, line: str, force: bool = False) -> bool:
        if self.connection is None:
            return False
        now = time.monotonic()
        if not force and now - self.last_send < self.command_interval:
            return False
        try:
            self.connection.write((line + "\n").encode("ascii"))
            self.last_send = now
            if line.split(" ", 1)[0] in {"HALT", "STOP", "FORWARD", "BACKWARD", "LEFT", "RIGHT", "DRIVE", "MANUAL_DRIVE", "TEST_LEFT", "TEST_RIGHT"}:
                # Do not leave a previous PWM displayed while waiting for the
                # ESP32 to confirm its newly applied output.
                self._esp_motor_pwm = None
                self._esp_motor_command = "WAIT_ESP"
            return True
        except Exception as error:
            self.last_error = f"Connection lost: {error}"
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
            self._link_confirmed = False
            self._esp_motor_pwm = None
            self._emergency_stop_sent = False
            return False

    def _accept_esp_response(self, line: str) -> None:
        """Consume one complete ESP32 response line, including PWM feedback."""
        line = line.strip()
        if not line:
            return
        self.last_response = line
        if not line.startswith("STATE "):
            return
        fields: dict[str, str] = {}
        for token in line.split()[1:]:
            key, separator, value = token.partition("=")
            if separator:
                fields[key] = value
        for channel, angle_key, active_key in (
            (1, "arm", "arm_active"),
            (2, "bucket", "bucket_active"),
        ):
            if angle_key not in fields:
                continue
            try:
                angle = max(0, min(180, int(fields[angle_key])))
            except ValueError:
                continue
            self._esp_servo_state[channel] = {
                "angle": angle,
                "active": fields.get(active_key) == "1",
            }
        try:
            left = max(-255, min(255, int(fields["left_pwm"])))
            right = max(-255, min(255, int(fields["right_pwm"])))
        except (KeyError, ValueError):
            # Compatibility with an ESP32 that has not yet received the new
            # firmware: leave the output unconfirmed instead of inventing it.
            return
        # ESP channel names are electrically reversed on this chassis. Expose
        # physical left/right values in the browser, not connector labels.
        if self.invert_turn_direction:
            left, right = right, left
        self._esp_motor_pwm = (left, right)
        reported_motion = fields.get("motion", self.last_command)
        if self.invert_turn_direction:
            reported_motion = {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(reported_motion, reported_motion)
        self._esp_motor_command = "DRIVE" if reported_motion == "STOP" and (left or right) else reported_motion
        if fields.get("emergency") == "1" and not self._resume_confirmation_pending:
            self._emergency_stop_requested.set()
            self._emergency_stop_sent = True
            self._target_normalized = None
            self._target_cm = None
            self._route_normalized = []
            self._route_index = 0
            self.last_command = "STOP"
            self.phase = "EMERGENCY_STOP"
        elif fields.get("emergency") == "0" and self._resume_confirmation_pending:
            self._resume_confirmation_pending = False
            self._emergency_stop_requested.clear()

    def _drain_responses(self) -> None:
        if self.connection is None:
            return
        try:
            waiting = self.connection.in_waiting
            if waiting:
                data = self.connection.read(waiting)
                if not data:
                    raise ConnectionError("ESP32 closed the TCP connection")
                self._response_buffer += data.decode("utf-8", errors="replace")
                lines = self._response_buffer.split("\n")
                self._response_buffer = lines.pop()
                for line in lines:
                    self._accept_esp_response(line)
                self._link_confirmed = True
                self.last_error = ""
        except Exception as error:
            self.last_error = str(error)
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
            self._link_confirmed = False
            self._esp_motor_pwm = None

    def _set_motor_speed(self, speed: int) -> None:
        """Update ESP32 PWM speed only when the PD output changes it."""
        requested = max(0, min(250, int(speed)))
        self._command_speed = requested
        if self._speed_signature != requested and self._send(f"SET_SPEED {requested}", force=True):
            self._speed_signature = requested

    def set_servo(self, channel: int, angle: int) -> tuple[bool, str]:
        """Set one attachment servo and wait for ESP feedback in STATE."""
        with self._lock:
            servo_channel = int(channel)
            target_angle = int(angle)
            if servo_channel not in {1, 2}:
                return False, "Неизвестный канал сервопривода"
            if not 0 <= target_angle <= 180:
                return False, "Угол сервопривода должен быть от 0 до 180°"
            if self._emergency_stop_requested.is_set() or self.phase == "EMERGENCY_STOP":
                return False, "Аварийная остановка включена — сервопривод заблокирован"
            if self.connection is None or not self._link_confirmed:
                return False, self.last_error or "Нет связи с ESP32"
            command = "SET_ARM" if servo_channel == 1 else "SET_BUCKET"
            if not self._send(f"{command} {target_angle}", force=True):
                return False, self.last_error or "Команда сервопривода не отправлена"
            self._requested_servo_angles[servo_channel] = target_angle
            self.last_error = ""
            return True, "Положение сервопривода отправлено"

    def _servo_output(self) -> dict[str, Any]:
        channels: list[dict[str, int | bool | None]] = []
        for channel in (1, 2):
            requested = self._requested_servo_angles.get(channel)
            feedback = self._esp_servo_state[channel]
            esp_angle = feedback.get("angle")
            active = bool(feedback.get("active", False))
            channels.append({
                "channel": channel,
                "requested_angle": requested,
                "esp_angle": esp_angle,
                "active": active,
                "confirmed": requested is not None and active and esp_angle == requested,
            })
        return {"channels": channels}

    def _sync_motor_inversion(self) -> None:
        """Keep the ESP32 channel polarity aligned with this agent's settings."""
        signature = (self.left_motor_inverted, self.right_motor_inverted)
        if signature == self._motor_inversion_signature:
            return
        left, right = (int(value) for value in signature)
        if self._send(f"SET_MOTOR_INVERSION {left} {right}", force=True):
            self._motor_inversion_signature = signature

    def _reset_heading_pd(self) -> None:
        self._previous_heading_error = None
        self._previous_heading_at = None

    def _turn_command(self, turn_right: bool) -> str:
        """Map logical direction through the chassis' swapped ESP channels."""
        if self.invert_turn_direction:
            turn_right = not turn_right
        return "RIGHT" if turn_right else "LEFT"

    def _arc_turn_command(self, turn_right: bool) -> str:
        # With normal wiring, a right arc drives the left wheel only. The
        # installed channels are swapped, so use the opposite ESP connector.
        drive_left_wheel = turn_right
        if self.invert_turn_direction:
            drive_left_wheel = not drive_left_wheel
        return "TEST_LEFT" if drive_left_wheel else "TEST_RIGHT"

    def _arc_turn(self, turn_right: bool, speed: int) -> None:
        """One-wheel pivot reserved for the calibration sequence."""
        if not self.arc_turn_single_motor:
            self._motion(self._turn_command(turn_right))
            return
        self._sync_motor_inversion()
        command = self._arc_turn_command(turn_right)
        requested = max(0, min(250, int(speed)))
        self._command_speed = requested
        now = time.monotonic()
        if command != self.last_command or now - self.last_motion_send >= self.command_interval:
            if self._send(f"{command} {requested}", force=True):
                self.last_command = command
                self.last_motion_send = now
        self._reset_stuck_watch()

    def _drive_wheels(self, left: int, right: int) -> None:
        """Send independent forward PWM values to the smooth-drive firmware."""
        self._sync_motor_inversion()
        requested = (max(0, min(250, int(left))), max(0, min(250, int(right))))
        now = time.monotonic()
        if (
            requested != self._last_drive_pwm
            or self.last_command != "DRIVE"
            or now - self.last_motion_send >= self.command_interval
        ):
            if self._send(f"DRIVE {requested[0]} {requested[1]}", force=True):
                self._last_drive_pwm = requested
                self.last_command = "DRIVE"
                self.last_motion_send = now

    def _send_manual_wheels(self, left: int, right: int) -> bool:
        """Drive signed wheel PWM without requiring camera coordinates."""
        self._sync_motor_inversion()
        requested = (max(-250, min(250, int(left))), max(-250, min(250, int(right))))
        now = time.monotonic()
        if (
            requested != self._last_drive_pwm
            or self.last_command != "MANUAL_DRIVE"
            or now - self.last_motion_send >= self.command_interval
        ):
            if not self._send(f"MANUAL_DRIVE {requested[0]} {requested[1]}", force=True):
                return False
            self._last_drive_pwm = requested
            self.last_command = "MANUAL_DRIVE"
            self.last_motion_send = now
        return True

    def _drive_curve(self, effort: float, speed: int) -> None:
        """Differential-drive PD arc: both wheels move forward, at different PWM."""
        outer = max(45, min(250, int(speed)))
        # Keep the inside wheel positive, so no on-the-spot tank rotation occurs.
        steering = min(0.75, max(0.08, abs(effort) / 120.0))
        inner = max(20, int(round(outer * (1.0 - steering))))
        turn_right = effort >= 0.0
        left, right = (outer, inner) if turn_right else (inner, outer)
        if self.invert_turn_direction:
            left, right = right, left
        self._drive_wheels(left, right)

    def _reset_stuck_watch(self) -> None:
        self._stuck_reference_position = None
        self._stuck_reference_at = None

    def _smoothed_pose_position(self, pose: RobotPose) -> tuple[float, float]:
        current = (pose.x_cm, pose.y_cm)
        if self._smoothed_position is None:
            self._smoothed_position = current
            return current
        alpha = self.pose_smoothing_alpha
        self._smoothed_position = (
            alpha * current[0] + (1.0 - alpha) * self._smoothed_position[0],
            alpha * current[1] + (1.0 - alpha) * self._smoothed_position[1],
        )
        return self._smoothed_position

    def _forward_progress_is_stuck(self, pose: RobotPose, now: float) -> bool:
        position = self._smoothed_pose_position(pose)
        if self._stuck_reference_position is None or self._stuck_reference_at is None:
            self._stuck_reference_position = position
            self._stuck_reference_at = now
            return False
        progress = math.dist(position, self._stuck_reference_position)
        if progress >= self.stuck_min_progress_cm:
            self._stuck_reference_position = position
            self._stuck_reference_at = now
            return False
        return now - self._stuck_reference_at >= self.stuck_timeout_s

    def _cruise_speed_limit(self) -> int:
        """Leave PWM headroom for boost recovery when that action is selected."""
        ceiling = max(0, min(250, self._speed_limit))
        if self.stuck_action != "boost":
            return ceiling
        reserve = min(self.stuck_boost_pwm, max(0, ceiling - 45))
        return ceiling - reserve

    def _effective_speed_limit(self) -> int:
        return min(250, self._cruise_speed_limit() + self._active_stuck_boost_pwm)

    def _tracking_pwm(self) -> int:
        """PWM used after coarse alignment; stuck boost is added on top of it."""
        configured = self.min_drive_pwm
        if configured == 0:
            configured = self._effective_speed_limit()
        return max(0, min(self._effective_speed_limit(), configured + self._active_stuck_boost_pwm))

    def _handle_stuck(self) -> bool:
        """Apply the selected recovery action. True means motion stays latched off."""
        if self.stuck_action == "continue":
            self._reset_stuck_watch()
            self.last_error = "Robot stuck: continuing at current PWM"
            return False
        if self.stuck_action == "boost":
            available = max(0, min(250, self._speed_limit) - self._cruise_speed_limit())
            self._active_stuck_boost_pwm = min(available, self.stuck_boost_pwm)
            self._reset_stuck_watch()
            self.last_error = f"Robot stuck: PWM boost {self._active_stuck_boost_pwm}, continuing"
            # Boost mode deliberately continues even when PWM is already at its
            # configured ceiling; it must not silently turn into a motor stop.
            return False
        self._stop("Robot is stuck: motors switched off")
        self.phase = "STUCK"
        return True

    def _drive_forward(self, pose: RobotPose, now: float, speed: int | None = None) -> bool:
        if self._forward_progress_is_stuck(pose, now):
            if self._handle_stuck():
                return False
        requested = self._effective_speed_limit() if speed is None else max(0, min(250, int(speed)))
        self._drive_wheels(requested, requested)
        return True

    def _requested_motor_pwm(self) -> tuple[int, int]:
        """Most recent PWM calculated by the server, before ESP32 feedback."""
        if self.last_command in {"DRIVE", "MANUAL_DRIVE"}:
            left, right = self._last_drive_pwm
        else:
            throttle = max(0, min(250, self._command_speed))
            left, right = {
                "FORWARD": (throttle, throttle),
                "BACKWARD": (-throttle, -throttle),
                "LEFT": (-throttle, throttle),
                "RIGHT": (throttle, -throttle),
                "TEST_LEFT": (throttle, 0),
                "TEST_RIGHT": (0, throttle),
            }.get(self.last_command, (0, 0))
        # Values above are ESP connector order; the UI always shows physical
        # left motor first and physical right motor second.
        return (right, left) if self.invert_turn_direction else (left, right)

    def _motor_output(self) -> dict[str, int | str | bool]:
        """Show both server-calculated and ESP32-confirmed PWM, never conflating them."""
        requested_left, requested_right = self._requested_motor_pwm()
        result: dict[str, int | str | bool] = {
            "server_left_pwm": requested_left,
            "server_right_pwm": requested_right,
            "left_pwm": 0,
            "right_pwm": 0,
            "left_percent": 0,
            "right_percent": 0,
            "command": "STOP",
            "confirmed": False,
        }
        if self.connection is None or not self._link_confirmed:
            return result
        if self._esp_motor_pwm is None:
            result["command"] = "WAIT_ESP"
            return result
        left, right = self._esp_motor_pwm
        result.update({
            "left_pwm": left,
            "right_pwm": right,
            "left_percent": int(round(left / 250.0 * 100.0)),
            "right_percent": int(round(right / 250.0 * 100.0)),
            "command": self._esp_motor_command,
            "confirmed": True,
        })
        return result

    def _motion(self, command: str) -> None:
        if command != "FORWARD":
            self._reset_stuck_watch()
        self._sync_motor_inversion()
        now = time.monotonic()
        if command != self.last_command or now - self.last_motion_send >= self.command_interval:
            if self._send(command, force=True):
                self.last_command = command
                self.last_motion_send = now

    def _stop(self, reason: str = "") -> None:
        if self.connection is not None and self.last_command != "STOP":
            # HALT is a normal motor stop. Firmware STOP is deliberately
            # reserved for the red emergency button because it latches until
            # RESUME. Using STOP here used to turn target arrival, a dropped
            # marker, or a stuck-action stop into a false emergency.
            self._send("HALT", force=True)
        self.last_command = "STOP"
        self._reset_heading_pd()
        self._reset_stuck_watch()
        if reason:
            self.last_error = reason

    def _enter_phase(self, phase: str, pose: RobotPose | None = None) -> None:
        self.phase = phase
        self.phase_started = time.monotonic()
        self.calibration_start_pose = pose
        self.phase_command_sent = False
        if phase == "CAL_FORWARD":
            self.calibration_max_distance_cm = 0.0
        if phase in {"CAL_TURN_90", "CAL_TURN_180"}:
            self.calibration_max_turn_deg = 0.0

    def set_target(
        self,
        agent_id: int,
        x_normalized: float,
        y_normalized: float,
        *,
        debug_mode: bool = False,
        speed: int | None = None,
        route: list[tuple[float, float]] | None = None,
        work_regions: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """Queue a target. Web Y grows upward; vision Y grows downward."""
        with self._lock:
            if self._emergency_stop_requested.is_set() or self.phase == "EMERGENCY_STOP":
                return False, "Аварийная остановка включена — сначала снимите её"
            if agent_id != self.controlled_agent_id:
                self.select_agent(agent_id)
            if self.connection is None:
                return False, "Нет связи с ESP32"
            if not self.calibrated:
                return False, "Дождитесь завершения калибровки"
            if debug_mode and (not route or not work_regions):
                return False, "Не удалось построить маршрут по рабочей поверхности"
            self._target_normalized = (
                max(0.0, min(100.0, float(x_normalized))),
                max(0.0, min(100.0, float(y_normalized))),
            )
            self._debug_mode = bool(debug_mode)
            if self._debug_mode:
                self._speed_limit = max(0, min(250, int(speed if speed is not None else 120)))
                self._route_normalized = [(float(x), float(y)) for x, y in route]
                self._route_index = 0
                self._work_regions = [dict(region) for region in work_regions]
            else:
                self._speed_limit = self.speed
                self._route_normalized = []
                self._route_index = 0
                self._work_regions = []
            self._speed_signature = None
            self._reset_heading_pd()
            self._active_stuck_boost_pwm = 0
            self.phase = "NAVIGATING"
            return True, "Команда передана роботу"

    def select_agent(self, agent_id: int) -> bool:
        """Bind the single robot connection to the marker selected in the UI."""
        with self._lock:
            selected_id = int(agent_id)
            if selected_id == self.controlled_agent_id:
                return False
            had_agent = self.controlled_agent_id is not None
            if had_agent:
                self._stop("Выбран другой ArUco-агент")
                self._close_connection()
            self.controlled_agent_id = selected_id
            self._apply_agent_settings(selected_id)
            self._speed_limit = self.speed
            self._reset_heading_pd()
            self._active_stuck_boost_pwm = 0
            self._target_normalized = None
            self._target_cm = None
            self._route_normalized = []
            self._route_index = 0
            self._work_regions = []
            if had_agent and not self._emergency_stop_requested.is_set():
                self.calibrated = True
                self.phase = "READY"
                self._bounds_signature = None
                self.linear_speed_cm_s = None
                self.angular_speed_deg_s = None
            return True

    def set_manual_mode(self, enabled: bool) -> None:
        """Select whether the current agent is controlled without vision."""
        with self._lock:
            requested = bool(enabled)
            if requested != self._manual_enabled:
                self._manual_command = None
                self._manual_expires_at = 0.0
                self._target_normalized = None
                self._target_cm = None
                self._route_normalized = []
                self._route_index = 0
                self._stop()
            self._manual_enabled = requested
            if not self._emergency_stop_requested.is_set():
                if requested:
                    self.phase = "MANUAL_READY"
                    self.calibrated = True
                elif self.phase in {"MANUAL_READY", "MANUAL_CONTROL"}:
                    self.phase = "READY" if self.calibrated else "WAIT_FIELD"

    def manual_drive(self, direction: str, speed: int | None = None) -> tuple[bool, str]:
        """Refresh one dead-man WASD command for a camera-less agent."""
        with self._lock:
            if not self._manual_enabled:
                return False, "Выбранный агент не включён в ручной режим"
            if self._emergency_stop_requested.is_set() or self.phase == "EMERGENCY_STOP":
                return False, "Аварийная остановка включена — сначала снимите её"
            if self.connection is None or not self._link_confirmed:
                return False, self.last_error or "Нет связи с ESP32"
            throttle = max(0, min(250, int(self.speed if speed is None else speed)))
            wheels = {
                "forward": (throttle, throttle),
                "backward": (-throttle, -throttle),
                "left": (-throttle, throttle),
                "right": (throttle, -throttle),
                # Combined WASD commands make a wide one-wheel turn instead
                # of rotating both tracks in opposite directions.
                "forward_left": (0, throttle),
                "forward_right": (throttle, 0),
                "backward_left": (-throttle, 0),
                "backward_right": (0, -throttle),
            }.get(str(direction).lower())
            if wheels is None:
                return False, "Неизвестное направление ручного управления"
            # Manual directions are expressed in physical chassis sides. Swap
            # them once into the ESP connector order before transmission.
            if self.invert_turn_direction:
                wheels = (wheels[1], wheels[0])
            self._manual_command = wheels
            self._manual_expires_at = time.monotonic() + self._manual_deadman_timeout_s
            self._command_speed = throttle
            if not self._send_manual_wheels(*wheels):
                return False, self.last_error or "Команда не отправлена на ESP32"
            self.phase = "MANUAL_CONTROL"
            self.last_error = ""
            return True, "Ручная команда принята"

    def manual_halt(self) -> None:
        """Release WASD motion without engaging the emergency latch."""
        with self._lock:
            self._manual_command = None
            self._manual_expires_at = 0.0
            self._stop()
            if self._manual_enabled and not self._emergency_stop_requested.is_set():
                self.phase = "MANUAL_READY"

    def set_debug_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._debug_mode = bool(enabled)
            if not enabled:
                self._speed_limit = self.speed
                self._command_speed = self.speed
                self._speed_signature = None
                self._route_normalized = []
                self._route_index = 0
                self._work_regions = []

    def emergency_stop(self, reason: str = "Emergency stop from web") -> None:
        # Latch before waiting for the lock: the control thread sees the stop
        # request even if an HTTP handler briefly loses the lock race.
        self._emergency_stop_requested.set()
        self._resume_confirmation_pending = False
        with self._lock:
            self._target_normalized = None
            self._target_cm = None
            self._route_normalized = []
            self._route_index = 0
            self._manual_command = None
            self._manual_expires_at = 0.0
            if self.connection is not None:
                self._emergency_stop_sent = self._send("STOP", force=True)
            self.last_command = "STOP"
            self._reset_heading_pd()
            self._reset_stuck_watch()
            self.last_error = reason
            self.phase = "EMERGENCY_STOP"

    def resume_after_emergency_stop(self) -> tuple[bool, str]:
        """Safely clear a latched web emergency stop after checking the link."""
        with self._lock:
            if self.connection is not None and self._link_confirmed:
                if not self._send("RESUME", force=True):
                    return False, "Не удалось передать ESP32 команду снятия аварийной остановки"
                self._resume_confirmation_pending = True
                self._emergency_stop_requested.clear()
                self._emergency_stop_sent = False
            if self.connection is None or not self._link_confirmed:
                return False, "Нет связи с ESP32 — аварийную остановку нельзя снять"
            self._target_normalized = None
            self._target_cm = None
            self._route_normalized = []
            self._route_index = 0
            self._stop()
            self.phase = "MANUAL_READY" if self._manual_enabled else ("READY" if self.calibrated else "WAIT_FIELD")
            self.last_error = ""
            return True, "Аварийная остановка снята"

    def clear_stuck(self) -> tuple[bool, str]:
        """Release a stopped stuck state while keeping the selected target."""
        with self._lock:
            if self.phase != "STUCK":
                return False, "Robot is not in the stuck state"
            if self.connection is None or not self._link_confirmed:
                return False, "No ESP32 connection — stuck state cannot be cleared"
            self._active_stuck_boost_pwm = 0
            self._reset_stuck_watch()
            self._reset_heading_pd()
            self.last_error = ""
            self.phase = "NAVIGATING" if self._target_normalized is not None else "READY"
            return True, "Stuck state cleared"

    def reset_calibration(self, reason: str = "Field configuration changed") -> None:
        with self._lock:
            self._target_normalized = None
            self._target_cm = None
            self._route_normalized = []
            self._route_index = 0
            self._work_regions = []
            self._stop(reason)
            self.calibrated = True
            self.phase = "EMERGENCY_STOP" if self._emergency_stop_requested.is_set() else "READY"
            self._bounds_signature = None
            self.linear_speed_cm_s = None
            self.angular_speed_deg_s = None

    def _reconnect_if_needed(self) -> None:
        if self.connection is not None:
            # Firmware sends READY/STATE right after accepting a client. A
            # silent TCP socket is discarded so reconnecting Wi-Fi or resetting
            # the ESP32 always gets a fresh controller connection.
            if self.transport == "wifi" and not self._link_confirmed and time.monotonic() - self._connection_opened_at > 1.5:
                try:
                    self.connection.close()
                except Exception:
                    pass
                self.connection = None
                self._link_confirmed = False
                self.last_error = "ESP32 did not confirm the connection"
            else:
                return
        now = time.monotonic()
        if now < self.next_reconnect_time:
            return
        self.next_reconnect_time = now + 2.0
        self._open_connection()
        if self.connection is not None:
            self._bounds_signature = None
            self._speed_signature = None

    def update(self, pose: RobotPose | None, width_cm: float | None, height_cm: float | None) -> None:
        """Consume one vision observation without blocking the camera loop."""
        with self._lock:
            self._reconnect_if_needed()
            self._drain_responses()
            if self._emergency_stop_requested.is_set():
                if self.connection is not None and not self._emergency_stop_sent:
                    self._emergency_stop_sent = self._send("STOP", force=True)
                self.last_command = "STOP"
                self.phase = "EMERGENCY_STOP"
                return
            if self._manual_enabled:
                now = time.monotonic()
                if self._manual_command is not None and now <= self._manual_expires_at:
                    if self.connection is not None and self._link_confirmed:
                        self._send_manual_wheels(*self._manual_command)
                        self.phase = "MANUAL_CONTROL"
                    return
                if self._manual_command is not None:
                    self._manual_command = None
                    self._manual_expires_at = 0.0
                    self._stop("Ручное управление остановлено: клавиша отпущена или потерян сигнал браузера")
                self.phase = "MANUAL_READY"
                return
            if pose is None or width_cm is None or height_cm is None:
                if self.connection is not None:
                    self._stop("Vision unavailable — robot stopped")
                if self._vision_lost_since is None:
                    self._vision_lost_since = time.monotonic()
                if not self.calibrated and self.phase == "WAIT_FIELD":
                    return
                return

            if self._vision_lost_since is not None:
                self.phase_started += time.monotonic() - self._vision_lost_since
                self._vision_lost_since = None
                if self.last_error.startswith("Vision unavailable"):
                    self.last_error = ""

            if self.connection is None or not self._link_confirmed:
                return

            signature = (round(width_cm, 1), round(height_cm, 1))
            if signature != self._bounds_signature:
                self._send(
                    f"SET_BOUNDS 0 0 {width_cm:.2f} {height_cm:.2f} {self.inset_cm:.2f}",
                    force=True,
                )
                self._bounds_signature = signature
                self._speed_signature = None
            if self.phase != "NAVIGATING":
                self._set_motor_speed(self._speed_limit)

            # This heartbeat feeds the 500 ms watchdog in the existing firmware.
            self._send(f"POSITION {pose.x_cm:.2f} {pose.y_cm:.2f} {pose.heading_deg:.1f}", force=True)
            now = time.monotonic()

            if self.phase == "EMERGENCY_STOP":
                self._stop()
                return

            if self.phase == "STUCK":
                self._stop()
                return

            if self.phase == "WAIT_FIELD" and not self.calibrated:
                self._stop()
                if self.auto_calibrate:
                    self._enter_phase("CAL_FORWARD", pose)
                else:
                    self.calibrated = True
                    self.phase = "READY"
                return

            # The following timing and pulses are preserved from PatrolPerimeter.
            if self.phase == "CAL_FORWARD":
                self._motion("FORWARD")
                assert self.calibration_start_pose is not None
                distance = math.hypot(
                    pose.x_cm - self.calibration_start_pose.x_cm,
                    pose.y_cm - self.calibration_start_pose.y_cm,
                )
                self.calibration_max_distance_cm = max(self.calibration_max_distance_cm, distance)
                if now - self.phase_started >= self.calibration_forward_seconds:
                    self._stop()
                    self.linear_speed_cm_s = self.calibration_max_distance_cm / max(now - self.phase_started, 0.01)
                    self._enter_phase("CAL_PAUSE_90", pose)
                return

            if self.phase == "CAL_PAUSE_90":
                self._stop()
                if now - self.phase_started >= 0.5:
                    self._enter_phase("CAL_TURN_90", pose)
                return

            if self.phase == "CAL_TURN_90":
                self._arc_turn(True, self._speed_limit)
                self.phase_command_sent = True
                assert self.calibration_start_pose is not None
                turned = abs(_angle_error(pose.heading_deg, self.calibration_start_pose.heading_deg))
                self.calibration_max_turn_deg = max(self.calibration_max_turn_deg, turned)
                elapsed = now - self.phase_started
                if elapsed >= self.calibration_turn_90_seconds:
                    self._stop()
                    self.angular_speed_deg_s = max(self.calibration_max_turn_deg, 1.0) / elapsed
                    self._enter_phase("CAL_PAUSE_180", pose)
                return

            if self.phase == "CAL_PAUSE_180":
                self._stop()
                if now - self.phase_started >= 0.5:
                    self._enter_phase("CAL_TURN_180", pose)
                return

            if self.phase == "CAL_TURN_180":
                self._arc_turn(True, self._speed_limit)
                self.phase_command_sent = True
                assert self.calibration_start_pose is not None
                turned = abs(_angle_error(pose.heading_deg, self.calibration_start_pose.heading_deg))
                self.calibration_max_turn_deg = max(self.calibration_max_turn_deg, turned)
                elapsed = now - self.phase_started
                if elapsed >= self.calibration_turn_180_seconds:
                    self._stop()
                    measured = max(self.calibration_max_turn_deg, 1.0) / elapsed
                    self.angular_speed_deg_s = measured if self.angular_speed_deg_s is None else (self.angular_speed_deg_s + measured) / 2.0
                    self.calibrated = True
                    self.phase = "NAVIGATING" if self._target_normalized else "READY"
                return

            if self._target_normalized is None:
                self._target_cm = None
                self._stop()
                self.phase = "READY"
                return

            active_target = (
                self._route_normalized[self._route_index]
                if self._debug_mode and self._route_normalized and self._route_index < len(self._route_normalized)
                else self._target_normalized
            )
            target_x = active_target[0] / 100.0 * width_cm
            target_y = (100.0 - active_target[1]) / 100.0 * height_cm
            target_x = max(self.inset_cm, min(width_cm - self.inset_cm, target_x))
            target_y = max(self.inset_cm, min(height_cm - self.inset_cm, target_y))
            self._target_cm = (target_x, target_y)
            self.phase = "NAVIGATING"

            pose_normalized = (pose.x_cm / width_cm * 100.0, 100.0 - pose.y_cm / height_cm * 100.0)
            if self._debug_mode and self._work_regions and not point_in_work(pose_normalized, self._work_regions):
                self._stop("Агент вышел за рабочую поверхность")
                self.phase = "WORK_BOUNDARY_STOP"
                return

            # If the center gets too close to an edge, first drive toward a safe point.
            hard = self.hard_stop_margin_cm
            if pose.x_cm <= hard or pose.x_cm >= width_cm - hard or pose.y_cm <= hard or pose.y_cm >= height_cm - hard:
                target_x = max(self.inset_cm, min(width_cm - self.inset_cm, pose.x_cm))
                target_y = max(self.inset_cm, min(height_cm - self.inset_cm, pose.y_cm))

            dx, dy = target_x - pose.x_cm, target_y - pose.y_cm
            distance = math.hypot(dx, dy)
            braking_distance = max(self.target_tolerance_cm, (self.linear_speed_cm_s or 0.0) * 0.12)
            if distance <= braking_distance:
                if self._debug_mode and self._route_index < len(self._route_normalized) - 1:
                    self._stop()
                    self._route_index += 1
                    self.phase = "NAVIGATING"
                else:
                    self._stop()
                    self._target_normalized = None
                    self._target_cm = None
                    self._route_normalized = []
                    self._route_index = 0
                    self.phase = "TARGET_REACHED"
                return

            desired_heading = math.degrees(math.atan2(dy, dx))
            error = _angle_error(desired_heading, pose.heading_deg)
            error_now = time.monotonic()
            dt = max(error_now - (self._previous_heading_at or error_now), 0.01)
            derivative = 0.0 if self._previous_heading_error is None else (error - self._previous_heading_error) / dt
            self._previous_heading_error = error
            self._previous_heading_at = error_now
            # PD steering: P turns toward the target; D damps a fast turn so
            # the robot does not repeatedly overshoot the route direction.
            effort = self.heading_kp * error + self.heading_kd * derivative
            # For very large errors, turn on the spot using both motors. The
            # threshold is adjustable per robot; default is the requested ±90°.
            if abs(error) > self.in_place_turn_threshold_deg:
                self._set_motor_speed(250)
                self._motion(self._turn_command(error >= 0.0))
                return

            # Medium errors are resolved by a one-wheel arc at PWM 250 until
            # its bearing is inside the ±45 degree PD capture sector.
            if abs(error) > 45.0:
                self._set_motor_speed(250)
                self._arc_turn(error >= 0.0, 250)
                return

            # Once coarsely aligned, PD controls a smooth forward arc at the
            # per-agent minimum tracking PWM. This value is intentionally not
            # used during the coarse one-wheel turn above.
            tracking_pwm = self._tracking_pwm()
            if abs(error) > self.heading_tolerance_deg:
                self._set_motor_speed(tracking_pwm)
                if self._forward_progress_is_stuck(pose, now):
                    if self._handle_stuck():
                        return
                self._drive_curve(effort, tracking_pwm)
            else:
                self._set_motor_speed(tracking_pwm)
                if self._debug_mode and self._work_regions:
                    lookahead = 2.0
                    radians = math.radians(pose.heading_deg)
                    predicted = (
                        pose_normalized[0] + math.cos(radians) * lookahead,
                        pose_normalized[1] - math.sin(radians) * lookahead,
                    )
                    if not point_has_clearance(predicted, self._work_regions, .35):
                        if abs(error) > 1.0:
                            self._drive_curve(error, self._speed_limit)
                        else:
                            self._stop("Граница рабочей поверхности — движение заблокировано")
                            self.phase = "WORK_BOUNDARY_STOP"
                        return
                self._drive_forward(pose, now, tracking_pwm)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self.connection is not None and self._link_confirmed,
                "connecting": self.connection is not None and not self._link_confirmed,
                "controlled_agent_id": self.controlled_agent_id,
                "manual_mode": self._manual_enabled,
                "phase": self.phase,
                "calibrated": self.calibrated,
                "last_command": self.last_command,
                "last_error": self.last_error,
                "last_response": self.last_response,
                "target": None if self._target_normalized is None else {
                    "x": self._target_normalized[0], "y": self._target_normalized[1]
                },
                "debug_mode": self._debug_mode,
                "speed": self._command_speed,
                "motor_output": self._motor_output(),
                "servo_output": self._servo_output(),
                "controller_settings": {
                    str(agent_id): self._settings_for(agent_id)
                    for agent_id in set(self._agent_settings) | ({self.controlled_agent_id} if self.controlled_agent_id is not None else set())
                },
                "route": [
                    {"x": point[0], "y": point[1]}
                    for point in self._route_normalized
                ],
                "route_index": self._route_index,
                "linear_speed_cm_s": self.linear_speed_cm_s,
                "angular_speed_deg_s": self.angular_speed_deg_s,
            }

    def close(self) -> None:
        with self._lock:
            self._stop("Application closing")
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            self._link_confirmed = False

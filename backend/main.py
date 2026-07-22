from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import sys
import time
from uuid import uuid4
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend.path_planner import plan_work_route
    from backend.stereo_runtime import StereoBrowserRuntime
    from backend.vision_runtime import VisionRuntime, load_hardware_config
else:
    from .path_planner import plan_work_route
    from .stereo_runtime import StereoBrowserRuntime
    from .vision_runtime import VisionRuntime, load_hardware_config


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
CONFIGURATION_PATH = ROOT / "configuration.json"
REGIONS_PATH = ROOT / "regions.json"
CONTROLLER_SETTINGS_PATH = ROOT / "controller_settings.json"
SERVO_SETTINGS_PATH = ROOT / "servo_settings.json"

SERVO_PROFILES: dict[str, tuple[str, ...]] = {
    "loader": ("Подъём", "Ковш"),
    "bulldozer": ("Отвал",),
    "dumper": ("Кузов",),
}


INITIAL_MARKERS = [
    {"id": 12, "role": "vertex", "agent_type": None, "x": 8, "y": 10, "seen": False},
    {"id": 18, "role": "vertex", "agent_type": None, "x": 92, "y": 8, "seen": False},
    {"id": 23, "role": "vertex", "agent_type": None, "x": 96, "y": 84, "seen": False},
    {"id": 31, "role": "vertex", "agent_type": None, "x": 12, "y": 91, "seen": False},
    {"id": 42, "role": "agent", "agent_type": "loader", "x": 30, "y": 39, "seen": False},
    {"id": 57, "role": "agent", "agent_type": "bulldozer", "x": 62, "y": 58, "seen": False},
    {"id": 66, "role": "agent", "agent_type": "dumper", "x": 73, "y": 29, "seen": False},
    {"id": 74, "role": "unassigned", "agent_type": None, "x": 45, "y": 77, "seen": False},
    {"id": 81, "role": "unassigned", "agent_type": None, "x": 19, "y": 62, "seen": False},
]


class MarkerConfig(BaseModel):
    id: int
    role: str = Field(pattern="^(unassigned|vertex|edge|agent)$")
    agent_type: str | None = Field(default=None, pattern="^(loader|bulldozer|dumper)$")
    ip_address: str | None = Field(default=None, max_length=45)
    name: str | None = Field(default=None, max_length=48)
    manual: bool = False

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, value: str | None) -> str | None:
        normalized = (value or "").strip()
        if not normalized:
            return None
        try:
            return str(ipaddress.ip_address(normalized))
        except ValueError as error:
            raise ValueError("Укажите корректный IPv4 или IPv6 адрес") from error

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


class Configuration(BaseModel):
    markers: list[MarkerConfig]


class MoveCommand(BaseModel):
    agent_id: int
    target_x: float = Field(ge=0, le=100)
    target_y: float = Field(ge=0, le=100)
    task_type: str = "move"
    debug_mode: bool = False
    speed: int = Field(default=120, ge=0, le=250)


class DebugModeRequest(BaseModel):
    enabled: bool


class ControlAgentRequest(BaseModel):
    agent_id: int


class ManualAgentCreate(BaseModel):
    agent_type: str = Field(default="loader", pattern="^(loader|bulldozer|dumper)$")
    ip_address: str = Field(max_length=45)
    name: str | None = Field(default=None, max_length=48)

    @field_validator("ip_address")
    @classmethod
    def validate_ip_address(cls, value: str) -> str:
        try:
            return str(ipaddress.ip_address(value.strip()))
        except ValueError as error:
            raise ValueError("Укажите корректный IPv4 или IPv6 адрес") from error

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


class ManualDriveRequest(BaseModel):
    agent_id: int
    direction: str = Field(pattern="^(forward|backward|left|right|forward_left|forward_right|backward_left|backward_right|halt)$")
    speed: int | None = Field(default=None, ge=0, le=250)


class StereoModeRequest(BaseModel):
    mode: str = Field(pattern="^(3d_aruco|3d_full|rgbd|superfast)$")


class ControllerSettingsRequest(BaseModel):
    speed: int = Field(ge=0, le=250)
    target_tolerance_cm: float = Field(ge=1.0, le=20.0)
    heading_tolerance_deg: float = Field(ge=1.0, le=90.0)
    # A negative P is useful when a particular robot's steering/motor mapping
    # is physically mirrored. Keep the bound symmetric with the positive one.
    heading_kp: float = Field(ge=-10.0, le=10.0)
    heading_kd: float = Field(ge=0.0, le=5.0)
    min_drive_pwm: int = Field(default=80, ge=0, le=250)
    in_place_turn_threshold_deg: float = Field(default=90.0, ge=1.0, le=180.0)
    heading_offset_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    left_motor_inverted: bool = False
    right_motor_inverted: bool = False
    stuck_timeout_s: float = Field(ge=0.5, le=30.0)
    stuck_min_progress_cm: float = Field(ge=0.1, le=20.0)
    stuck_action: str = Field(default="stop", pattern="^(stop|boost|continue)$")
    stuck_boost_pwm: int = Field(default=35, ge=1, le=150)


class ServoChannelSettingsRequest(BaseModel):
    channel: int = Field(ge=1, le=2)
    min_angle: int = Field(ge=0, le=180)
    max_angle: int = Field(ge=0, le=180)
    position: int = Field(ge=0, le=180)


class ServoSettingsRequest(BaseModel):
    channels: list[ServoChannelSettingsRequest] = Field(min_length=1, max_length=2)


class ServoPositionRequest(BaseModel):
    angle: int = Field(ge=0, le=180)


class RegionCreate(BaseModel):
    type: str = Field(pattern="^(quarry|storage|work)$")
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    width: float = Field(gt=0.2, le=100)
    height: float = Field(gt=0.2, le=100)


class TelemetryMarker(BaseModel):
    id: int
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)
    heading: float | None = None
    battery: int | None = Field(default=None, ge=0, le=100)


class TelemetryFrame(BaseModel):
    markers: list[TelemetryMarker]
    source: str = "external-vision"


def _load_saved_markers() -> list[dict[str, Any]]:
    if not CONFIGURATION_PATH.exists():
        return [dict(marker) for marker in INITIAL_MARKERS]
    try:
        data = json.loads(CONFIGURATION_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return [dict(marker) for marker in INITIAL_MARKERS]


def _load_saved_regions() -> list[dict[str, Any]]:
    if not REGIONS_PATH.exists():
        return []
    try:
        data = json.loads(REGIONS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _load_controller_settings() -> dict[int, dict[str, Any]]:
    if not CONTROLLER_SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(CONTROLLER_SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {int(agent_id): dict(settings) for agent_id, settings in data.items() if isinstance(settings, dict)}
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {}


def _load_servo_settings() -> dict[int, dict[str, Any]]:
    if not SERVO_SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SERVO_SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {int(agent_id): dict(settings) for agent_id, settings in data.items() if isinstance(settings, dict)}
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {}


def _servo_settings_for(agent_type: str, saved: dict[str, Any] | None = None) -> dict[str, Any]:
    def safe_angle(value: Any, fallback: int) -> int:
        try:
            return max(0, min(180, int(value)))
        except (TypeError, ValueError):
            return fallback

    normalized_type = agent_type if agent_type in SERVO_PROFILES else "loader"
    labels = SERVO_PROFILES[normalized_type]
    saved_channels = {
        int(item.get("channel", 0)): item
        for item in (saved or {}).get("channels", [])
        if isinstance(item, dict)
    }
    channels: list[dict[str, Any]] = []
    for channel, label in enumerate(labels, start=1):
        raw = saved_channels.get(channel, {})
        minimum = safe_angle(raw.get("min_angle"), 0)
        maximum = safe_angle(raw.get("max_angle"), 180)
        if minimum >= maximum:
            minimum, maximum = 0, 180
        position = max(minimum, min(maximum, safe_angle(raw.get("position"), 90)))
        channels.append({
            "channel": channel,
            "name": label,
            "min_angle": minimum,
            "max_angle": maximum,
            "position": position,
        })
    return {"agent_type": normalized_type, "channels": channels}


class FieldState:
    def __init__(self) -> None:
        self.markers: list[dict[str, Any]] = _load_saved_markers()
        self.regions: list[dict[str, Any]] = _load_saved_regions()
        self.targets: dict[int, dict[str, float]] = {}
        self.servo_settings: dict[int, dict[str, Any]] = _load_servo_settings()
        self.camera: dict[str, Any] = {
            "online": False, "calibrated": False, "tracking": "offline", "error": "Camera is not connected"
        }
        self.robot: dict[str, Any] = {
            "connected": False, "calibrated": False, "phase": "WAIT_FIELD", "last_command": "STOP"
        }
        self.emergency_latched = False
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.sync_task: asyncio.Task[None] | None = None

    def snapshot(self) -> dict[str, Any]:
        agents = []
        for marker in self.markers:
            manual = bool(marker.get("manual", False))
            available = manual or bool(marker.get("seen", False))
            if marker.get("role") != "agent" or not available:
                continue
            target = self.targets.get(marker["id"])
            is_controlled = marker["id"] == self.robot.get("controlled_agent_id")
            disconnected = is_controlled and not self.robot.get("connected", False)
            moving = is_controlled and self.robot.get("phase") in {"NAVIGATING", "MANUAL_CONTROL"} and not disconnected
            stuck = is_controlled and self.robot.get("phase") == "STUCK"
            agents.append({
                "id": marker["id"],
                "agent_type": marker.get("agent_type") or "loader",
                "ip_address": marker.get("ip_address"),
                "name": marker.get("name"),
                "x": round(float(marker.get("x", 0)), 3),
                "y": round(float(marker.get("y", 0)), 3),
                "status": "disconnected" if disconnected else ("moving" if moving else ("stuck" if stuck else ("offline" if not available else "ready"))),
                "target": target,
                "heading": marker.get("heading", 0),
                "battery": marker.get("battery", 87 - marker["id"] % 12),
                "seen": available,
                "manual": manual,
                "is_controlled": is_controlled,
                "debug_route": self.robot.get("route", []) if is_controlled else [],
                "debug_route_index": self.robot.get("route_index", 0) if is_controlled else 0,
                "debug_speed": self.robot.get("speed") if is_controlled else None,
                "controller_settings": self.robot.get("controller_settings", {}).get(str(marker["id"]), {}),
                "motor_output": self.robot.get("motor_output", {}) if is_controlled else {},
                "servo_settings": self.servo_settings_for(marker),
                "servo_output": self.robot.get("servo_output", {}) if is_controlled else {},
            })
        return {
            "type": "state",
            "timestamp": int(time.time() * 1000),
            "mode": "hardware",
            # Browsers only receive markers visible in the current camera frame.
            # Roles for temporarily hidden IDs remain stored internally/configuration.json.
            "markers": [marker for marker in self.markers if marker.get("seen", False) or marker.get("manual", False)],
            "agents": agents,
            "regions": self.regions,
            "camera": self.camera,
            "robot": self.robot,
        }

    async def broadcast(self, payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for client in list(self.clients):
            try:
                await client.send_json(payload)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def sync_hardware(self, runtime: VisionRuntime) -> None:
        """Merge camera observations into application state and push them to browsers."""
        while True:
            await asyncio.sleep(0.12)
            hardware = runtime.snapshot()
            async with self.lock:
                self.camera = hardware["camera"]
                incoming_robot = hardware["robot"]
                if self.emergency_latched:
                    incoming_robot["phase"] = "EMERGENCY_STOP"
                    incoming_robot["last_command"] = "STOP"
                    incoming_robot["target"] = None
                self.robot = incoming_robot
                known = {int(marker["id"]): marker for marker in self.markers}
                for marker in self.markers:
                    # Camera-less agents are persistent UI/controller records,
                    # not visual observations. Keep them available even when
                    # both cameras are disconnected.
                    marker["seen"] = bool(marker.get("manual", False))
                for reading in hardware["detected_markers"]:
                    marker_id = int(reading["id"])
                    marker = known.get(marker_id)
                    if marker is None:
                        marker = {
                            "id": marker_id,
                            "role": "unassigned",
                            "agent_type": None,
                            "x": reading["x"],
                            "y": reading["y"],
                            "seen": True,
                        }
                        self.markers.append(marker)
                        known[marker_id] = marker
                    marker["x"] = reading["x"]
                    marker["y"] = reading["y"]
                    marker["seen"] = True
                for reading in hardware["agents"]:
                    marker = known.get(int(reading["id"]))
                    if marker is not None:
                        marker["heading"] = reading.get("heading", marker.get("heading", 0))
                robot_target = self.robot.get("target")
                controlled_id = self.robot.get("controlled_agent_id")
                if controlled_id is not None:
                    if robot_target:
                        self.targets[int(controlled_id)] = dict(robot_target)
                    else:
                        self.targets.pop(int(controlled_id), None)
            runtime.set_marker_config(self.markers)
            await self.broadcast(self.snapshot())

    def save_configuration(self) -> None:
        serializable = [
            {
                "id": marker["id"], "role": marker.get("role", "unassigned"),
                "agent_type": marker.get("agent_type"), "x": marker.get("x", 0),
                "y": marker.get("y", 0), "seen": marker.get("seen", False),
                "ip_address": marker.get("ip_address"),
                "name": marker.get("name"),
                "manual": bool(marker.get("manual", False)),
            }
            for marker in self.markers
        ]
        CONFIGURATION_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_regions(self) -> None:
        REGIONS_PATH.write_text(json.dumps(self.regions, ensure_ascii=False, indent=2), encoding="utf-8")

    def servo_settings_for(self, marker: dict[str, Any]) -> dict[str, Any]:
        agent_id = int(marker["id"])
        return _servo_settings_for(
            str(marker.get("agent_type") or "loader"),
            self.servo_settings.get(agent_id),
        )

    def save_servo_settings(self) -> None:
        SERVO_SETTINGS_PATH.write_text(
            json.dumps(self.servo_settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


hardware_config = load_hardware_config()
stereo_runtime = StereoBrowserRuntime(hardware_config)
runtime = VisionRuntime(hardware_config)
if stereo_runtime.enabled:
    runtime.set_frame_provider(stereo_runtime.latest_left_frame)
for _agent_id, _settings in _load_controller_settings().items():
    try:
        runtime.controller.set_agent_settings(_agent_id, **_settings)
    except (TypeError, ValueError):
        continue
state = FieldState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.set_marker_config(state.markers)
    stereo_runtime.start()
    runtime.start()
    state.sync_task = asyncio.create_task(state.sync_hardware(runtime))
    yield
    if state.sync_task:
        state.sync_task.cancel()
    runtime.stop()
    stereo_runtime.stop()


app = FastAPI(title="КарьерА WEB3 Stereo", version="3.0.0", lifespan=lifespan)


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return state.snapshot()


@app.put("/api/configuration")
async def update_configuration(config: Configuration) -> dict[str, Any]:
    field_markers = [marker for marker in config.markers if marker.role in {"vertex", "edge"}]
    vertices = [marker for marker in config.markers if marker.role == "vertex"]
    if len(field_markers) > 8:
        raise HTTPException(status_code=422, detail="Для поля можно назначить не более 8 меток")
    # With cameras offline the browser only receives persistent manual agents.
    # Let their name/IP be edited without demanding currently visible field
    # markers; existing ArUco assignments remain untouched by this partial PUT.
    has_visual_records = any(not marker.manual for marker in config.markers)
    if has_visual_records and len(vertices) != 4:
        raise HTTPException(status_code=422, detail="Для калибровки камеры назначьте ровно 4 вершины")
    if len({marker.id for marker in config.markers}) != len(config.markers):
        raise HTTPException(status_code=422, detail="ID меток должны быть уникальными")
    known = {marker["id"]: marker for marker in state.markers}
    async with state.lock:
        for item in config.markers:
            if item.id not in known:
                continue
            if known[item.id].get("manual") and item.role != "agent":
                raise HTTPException(status_code=422, detail="Робот без ArUco всегда должен иметь роль «Агент»")
            known[item.id]["role"] = item.role
            known[item.id]["agent_type"] = item.agent_type if item.role == "agent" else None
            known[item.id]["ip_address"] = ((item.ip_address or "").strip() or None) if item.role == "agent" else None
            known[item.id]["name"] = item.name if item.role == "agent" else None
            known[item.id]["manual"] = bool(known[item.id].get("manual", False) or item.manual)
            if item.role != "agent":
                state.targets.pop(item.id, None)
        state.save_configuration()
    runtime.set_marker_config(state.markers)
    snapshot = state.snapshot()
    await state.broadcast(snapshot)
    return snapshot


@app.post("/api/commands/move")
async def move_agent(command: MoveCommand) -> dict[str, Any]:
    marker = next((item for item in state.markers if item["id"] == command.agent_id), None)
    if not marker or marker.get("role") != "agent":
        raise HTTPException(status_code=404, detail="Агент с таким ID не найден")
    if marker.get("manual"):
        raise HTTPException(status_code=409, detail="Робот без ArUco управляется клавишами W/A/S/D")
    options: dict[str, Any] = {"debug_mode": command.debug_mode}
    route: list[tuple[float, float]] = []
    if command.debug_mode:
        if not marker.get("seen", False):
            raise HTTPException(status_code=409, detail="Метка агента сейчас не видна")
        work_regions = [region for region in state.regions if region.get("type") == "work"]
        try:
            route = plan_work_route(
                (float(marker["x"]), float(marker["y"])),
                (command.target_x, command.target_y),
                work_regions,
                clearance=float(hardware_config.get("debug_work_clearance_percent", 1.4)),
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        options.update({"speed": command.speed, "route": route, "work_regions": work_regions})
    accepted, message = runtime.set_target(command.agent_id, command.target_x, command.target_y, **options)
    if not accepted:
        raise HTTPException(status_code=409, detail=message)
    async with state.lock:
        state.targets[command.agent_id] = {"x": command.target_x, "y": command.target_y}
    payload = {
        "type": "command_accepted", "agent_id": command.agent_id,
        "target_x": command.target_x, "target_y": command.target_y,
        "task_type": command.task_type, "message": message,
        "debug_mode": command.debug_mode, "speed": command.speed if command.debug_mode else hardware_config.get("robot_speed", 255),
        "route": [{"x": x, "y": y} for x, y in route],
    }
    await state.broadcast(payload)
    return {"ok": True, **payload}


@app.post("/api/debug-mode")
async def set_debug_mode(request: DebugModeRequest) -> dict[str, Any]:
    runtime.set_debug_enabled(request.enabled)
    return {"ok": True, "enabled": request.enabled}


@app.post("/api/control-agent")
async def select_control_agent(request: ControlAgentRequest) -> dict[str, Any]:
    marker = next((item for item in state.markers if item["id"] == request.agent_id), None)
    if not marker or marker.get("role") != "agent":
        raise HTTPException(status_code=404, detail="Сначала назначьте метке роль «Агент»")
    changed = runtime.select_agent(request.agent_id, manual=bool(marker.get("manual", False)))
    async with state.lock:
        if changed:
            state.targets.clear()
        state.robot["controlled_agent_id"] = request.agent_id
        state.robot["phase"] = runtime.controller.status().get("phase", "WAIT_FIELD")
    snapshot = state.snapshot()
    await state.broadcast(snapshot)
    return {"ok": True, "agent_id": request.agent_id, "changed": changed}


@app.post("/api/manual-agents")
async def create_manual_agent(request: ManualAgentCreate) -> dict[str, Any]:
    async with state.lock:
        marker = next((
            item for item in state.markers
            if item.get("manual") and item.get("ip_address") == request.ip_address
        ), None)
        if marker is None:
            used_ids = {int(item["id"]) for item in state.markers}
            agent_id = -1
            while agent_id in used_ids:
                agent_id -= 1
            marker = {
                "id": agent_id,
                "role": "agent",
                "manual": True,
                "x": 0.0,
                "y": 0.0,
                "heading": 0.0,
                "battery": None,
            }
            state.markers.append(marker)
        else:
            agent_id = int(marker["id"])
        # Repeated clicks for the same ESP IP update the existing row instead
        # of silently creating duplicate robots.
        marker.update({
            "role": "agent",
            "agent_type": request.agent_type,
            "ip_address": request.ip_address,
            "name": request.name,
            "manual": True,
            "seen": True,
        })
        state.save_configuration()
    runtime.set_marker_config(state.markers)
    snapshot = state.snapshot()
    await state.broadcast(snapshot)
    return {"ok": True, "agent": next(agent for agent in snapshot["agents"] if agent["id"] == agent_id)}


@app.delete("/api/manual-agents/{agent_id}")
async def delete_manual_agent(agent_id: int) -> dict[str, Any]:
    marker = next((item for item in state.markers if int(item["id"]) == agent_id), None)
    if not marker or not marker.get("manual"):
        raise HTTPException(status_code=404, detail="Ручной агент не найден")
    if runtime.controller.controlled_agent_id == agent_id:
        runtime.controller.manual_halt()
        runtime.controller.set_manual_mode(False)
    async with state.lock:
        state.markers = [item for item in state.markers if int(item["id"]) != agent_id]
        state.targets.pop(agent_id, None)
        if state.servo_settings.pop(agent_id, None) is not None:
            state.save_servo_settings()
        state.save_configuration()
    runtime.set_marker_config(state.markers)
    await state.broadcast(state.snapshot())
    return {"ok": True, "agent_id": agent_id}


@app.post("/api/commands/manual")
async def manual_drive(request: ManualDriveRequest) -> dict[str, Any]:
    marker = next((item for item in state.markers if int(item["id"]) == request.agent_id), None)
    if not marker or marker.get("role") != "agent" or not marker.get("manual"):
        raise HTTPException(status_code=404, detail="Робот без ArUco не найден")
    if runtime.controller.controlled_agent_id != request.agent_id:
        runtime.select_agent(request.agent_id, manual=True)
    if request.direction == "halt":
        runtime.controller.manual_halt()
        message = "Моторы остановлены"
    else:
        accepted, message = runtime.controller.manual_drive(request.direction, request.speed)
        if not accepted:
            raise HTTPException(status_code=409, detail=message)
    async with state.lock:
        state.robot = runtime.controller.status()
    await state.broadcast(state.snapshot())
    return {"ok": True, "direction": request.direction, "message": message}


@app.put("/api/agents/{agent_id}/controller-settings")
async def save_controller_settings(agent_id: int, settings: ControllerSettingsRequest) -> dict[str, Any]:
    marker = next((item for item in state.markers if item["id"] == agent_id), None)
    if not marker or marker.get("role") != "agent":
        raise HTTPException(status_code=404, detail="Сначала назначьте метке роль «Агент»")
    saved = runtime.controller.set_agent_settings(agent_id, **settings.model_dump())
    all_settings = runtime.controller.status().get("controller_settings", {})
    CONTROLLER_SETTINGS_PATH.write_text(json.dumps(all_settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "agent_id": agent_id, "settings": saved}


@app.put("/api/agents/{agent_id}/servo-settings")
async def save_servo_settings(agent_id: int, settings: ServoSettingsRequest) -> dict[str, Any]:
    marker = next((item for item in state.markers if int(item["id"]) == agent_id), None)
    if not marker or marker.get("role") != "agent":
        raise HTTPException(status_code=404, detail="Сначала назначьте метке роль «Агент»")
    agent_type = str(marker.get("agent_type") or "loader")
    expected_channels = len(SERVO_PROFILES.get(agent_type, SERVO_PROFILES["loader"]))
    submitted = {item.channel: item for item in settings.channels}
    if set(submitted) != set(range(1, expected_channels + 1)) or len(submitted) != len(settings.channels):
        raise HTTPException(
            status_code=422,
            detail=f"Для этого типа робота требуется каналов сервоприводов: {expected_channels}",
        )
    channels: list[dict[str, int]] = []
    for channel in range(1, expected_channels + 1):
        item = submitted[channel]
        if item.min_angle >= item.max_angle:
            raise HTTPException(status_code=422, detail=f"Серво {channel}: минимум должен быть меньше максимума")
        if not item.min_angle <= item.position <= item.max_angle:
            raise HTTPException(status_code=422, detail=f"Серво {channel}: положение находится вне заданных пределов")
        channels.append({
            "channel": channel,
            "min_angle": item.min_angle,
            "max_angle": item.max_angle,
            "position": item.position,
        })
    async with state.lock:
        state.servo_settings[agent_id] = {"channels": channels}
        state.save_servo_settings()
        saved = state.servo_settings_for(marker)
    await state.broadcast(state.snapshot())
    return {"ok": True, "agent_id": agent_id, "settings": saved}


@app.post("/api/agents/{agent_id}/servos/{channel}")
async def set_servo_position(agent_id: int, channel: int, request: ServoPositionRequest) -> dict[str, Any]:
    marker = next((item for item in state.markers if int(item["id"]) == agent_id), None)
    if not marker or marker.get("role") != "agent":
        raise HTTPException(status_code=404, detail="Агент не найден")
    servo_settings = state.servo_settings_for(marker)
    channel_settings = next((item for item in servo_settings["channels"] if item["channel"] == channel), None)
    if channel_settings is None:
        raise HTTPException(status_code=404, detail="У этого типа робота нет такого сервопривода")
    if not channel_settings["min_angle"] <= request.angle <= channel_settings["max_angle"]:
        raise HTTPException(
            status_code=422,
            detail=f"Допустимый угол: {channel_settings['min_angle']}–{channel_settings['max_angle']}°",
        )
    if runtime.controller.controlled_agent_id != agent_id:
        runtime.select_agent(agent_id, manual=bool(marker.get("manual", False)))
    accepted, message = runtime.controller.set_servo(channel, request.angle)
    if not accepted:
        raise HTTPException(status_code=409, detail=message)
    async with state.lock:
        stored = state.servo_settings_for(marker)
        for item in stored["channels"]:
            if item["channel"] == channel:
                item["position"] = request.angle
        state.servo_settings[agent_id] = {
            "channels": [
                {key: item[key] for key in ("channel", "min_angle", "max_angle", "position")}
                for item in stored["channels"]
            ]
        }
        state.save_servo_settings()
        state.robot = runtime.controller.status()
    await state.broadcast(state.snapshot())
    return {
        "ok": True,
        "agent_id": agent_id,
        "channel": channel,
        "angle": request.angle,
        "message": message,
        "settings": state.servo_settings_for(marker),
        "servo_output": state.robot.get("servo_output", {}),
    }


@app.post("/api/commands/stop")
async def stop_robot() -> dict[str, Any]:
    runtime.emergency_stop()
    async with state.lock:
        state.emergency_latched = True
        state.targets.clear()
        state.robot = runtime.controller.status()
    payload = {"type": "emergency_stop", "ok": True}
    await state.broadcast(state.snapshot())
    return payload


@app.post("/api/commands/resume")
async def resume_robot() -> dict[str, Any]:
    resumed, message = runtime.controller.resume_after_emergency_stop()
    if not resumed:
        raise HTTPException(status_code=409, detail=message)
    async with state.lock:
        state.emergency_latched = False
        state.targets.clear()
        state.robot = runtime.controller.status()
    payload = {"type": "emergency_cleared", "ok": True, "message": message}
    await state.broadcast(state.snapshot())
    return payload


@app.post("/api/commands/clear-stuck")
async def clear_stuck() -> dict[str, Any]:
    cleared, message = runtime.controller.clear_stuck()
    if not cleared:
        raise HTTPException(status_code=409, detail=message)
    async with state.lock:
        state.robot = runtime.controller.status()
    payload = {"type": "stuck_cleared", "ok": True, "message": message}
    await state.broadcast(state.snapshot())
    return payload


@app.post("/api/regions")
async def create_region(region: RegionCreate) -> dict[str, Any]:
    if region.x + region.width > 100.001 or region.y + region.height > 100.001:
        raise HTTPException(status_code=422, detail="Область должна полностью находиться внутри поля")
    record = {
        "id": uuid4().hex[:10],
        "type": region.type,
        "x": round(region.x, 3),
        "y": round(region.y, 3),
        "width": round(region.width, 3),
        "height": round(region.height, 3),
        "created_at": int(time.time() * 1000),
    }
    async with state.lock:
        state.regions.append(record)
        state.save_regions()
    await state.broadcast(state.snapshot())
    return {"ok": True, "region": record}


@app.delete("/api/regions/{region_id}")
async def delete_region(region_id: str) -> dict[str, Any]:
    async with state.lock:
        before = len(state.regions)
        state.regions = [region for region in state.regions if region.get("id") != region_id]
        if len(state.regions) == before:
            raise HTTPException(status_code=404, detail="Область не найдена")
        state.save_regions()
    await state.broadcast(state.snapshot())
    return {"ok": True, "region_id": region_id}


@app.post("/api/telemetry")
async def ingest_telemetry(frame: TelemetryFrame) -> dict[str, Any]:
    """Optional input for a separate vision computer; the local camera uses the same state."""
    async with state.lock:
        known = {marker["id"]: marker for marker in state.markers}
        for reading in frame.markers:
            marker = known.get(reading.id)
            if marker is None:
                marker = {"id": reading.id, "role": "unassigned", "agent_type": None}
                state.markers.append(marker)
                known[reading.id] = marker
            marker.update({"x": reading.x, "y": reading.y, "seen": True})
            if reading.heading is not None:
                marker["heading"] = reading.heading
            if reading.battery is not None:
                marker["battery"] = reading.battery
    await state.broadcast(state.snapshot())
    return {"ok": True, "source": frame.source, "received": len(frame.markers)}


@app.get("/api/camera/status")
async def camera_status() -> dict[str, Any]:
    return {"camera": state.camera, "robot": state.robot}


@app.get("/api/stereo/status")
async def stereo_status() -> dict[str, Any]:
    return stereo_runtime.status()


@app.post("/api/stereo/mode")
async def stereo_mode(request: StereoModeRequest) -> dict[str, Any]:
    try:
        return stereo_runtime.set_mode(request.mode)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/stereo/settings")
async def stereo_settings() -> dict[str, Any]:
    return stereo_runtime.settings()


@app.post("/api/stereo/settings")
async def update_stereo_settings(changes: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "settings": stereo_runtime.update_settings(changes)}
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/stereo/mesh.bin")
async def stereo_mesh(crop: str = "aruco") -> Response:
    packet = stereo_runtime.mesh_packet("full" if crop == "full" else "aruco")
    if packet is None:
        raise HTTPException(status_code=503, detail=stereo_runtime.status().get("error") or "3D mesh is not ready")
    return Response(content=packet, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/stereo/texture.jpg")
async def stereo_texture(mode: str = "rgb") -> Response:
    texture = stereo_runtime.texture_jpeg("topology" if mode == "topology" else "rgb")
    if texture is None:
        raise HTTPException(status_code=503, detail=stereo_runtime.status().get("error") or "3D texture is not ready")
    return Response(content=texture, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/stereo/frame.jpg")
async def stereo_frame() -> Response:
    frame = stereo_runtime.frame_jpeg()
    if frame is None:
        raise HTTPException(status_code=503, detail=stereo_runtime.status().get("error") or "2D RGBD frame is not ready")
    return Response(content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/camera/stream")
async def camera_stream() -> StreamingResponse:
    return StreamingResponse(
        runtime.mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/video_feed")
async def local_network_video_feed() -> StreamingResponse:
    """Simple MJPEG URL for browsers and other devices on the local network."""
    return StreamingResponse(
        runtime.mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/camera/frame.jpg")
async def camera_frame() -> Response:
    jpeg = runtime.latest_jpeg()
    if jpeg is None:
        raise HTTPException(status_code=503, detail="Камера пока не передала кадр")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/aruco/{marker_id}.png")
async def aruco_marker_image(marker_id: int) -> Response:
    dictionary_size = int(hardware_config.get("aruco_dictionary_size", 100))
    if marker_id < 0 or marker_id >= dictionary_size:
        raise HTTPException(status_code=404, detail="ID отсутствует в словаре DICT_4X4_100")
    try:
        png = runtime.aruco_marker_png(marker_id)
    except (cv2.error, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state.clients.add(websocket)
    await websocket.send_json(state.snapshot())
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": int(time.time() * 1000)})
    except WebSocketDisconnect:
        state.clients.discard(websocket)


app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


@app.get("/{path:path}")
async def frontend(path: str) -> FileResponse:
    return FileResponse(FRONTEND / "index.html", headers={"Cache-Control": "no-store"})


def _local_ip() -> str:
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("8.8.8.8", 80))
        return str(connection.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        connection.close()


if __name__ == "__main__":
    import uvicorn

    lan_ip = _local_ip()
    print("КарьерА: локальный сервер камеры", flush=True)
    print("Веб-интерфейс: http://127.0.0.1:5000", flush=True)
    print(f"MJPEG в локальной сети: http://{lan_ip}:5000/video_feed", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=5000)

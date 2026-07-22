const AGENT_META = {
  loader: { name: "Погрузчик", color: "#1c422a" },
  bulldozer: { name: "Бульдозер", color: "#e57d3e" },
  dumper: { name: "Самосвал", color: "#5988a3" },
};

const ROLE_LABELS = {
  unassigned: "Без назначения",
  vertex: "Вершина полигона",
  edge: "Ребро полигона",
  agent: "Агент",
};

const REGION_META = {
  quarry: { name: "Карьер", copy: "Зона добычи" },
  storage: { name: "Хранилище", copy: "Склад материала" },
  work: { name: "Рабочая область", copy: "Зона операций" },
};

const appState = {
  markers: [],
  agents: [],
  regions: [],
  draftMarkers: [],
  selectedAgentId: null,
  choosingTarget: false,
  zoom: 1,
  cameraVisible: true,
  debugMode: false,
  debugSpeed: 120,
  camera: { online: false, calibrated: false },
  robot: { connected: false, calibrated: false, phase: "WAIT_FIELD" },
  activeRegionType: null,
  selectedRegionId: null,
  regionDraft: null,
  deletingRegionIds: new Set(),
  ws: null,
  latencyTimer: null,
  manualKeys: new Set(),
  manualPointers: new Map(),
  manualDirection: null,
  manualTimer: null,
  manualRequestPending: false,
  manualQueuedDirection: null,
};

const INTERFACE_SETTINGS_KEY = "kariera.interface.settings.v1";
const DEFAULT_INTERFACE_SETTINGS = Object.freeze({
  notificationsEnabled: true,
  soundEnabled: true,
  scale: 100,
  notificationDuration: 3.5,
});

let interfaceSettings = { ...DEFAULT_INTERFACE_SETTINGS };
let interfaceSettingsBeforeOpen = null;
let notificationAudioContext = null;

const MANUAL_KEY_DIRECTIONS = {
  KeyW: "forward",
  KeyA: "left",
  KeyS: "backward",
  KeyD: "right",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const svgNS = "http://www.w3.org/2000/svg";

function agentIcon(type) {
  if (type === "bulldozer") {
    return `<svg viewBox="0 0 40 40" aria-hidden="true"><path d="M8 26h23l4-7H18l-3-8H8v15Z"/><path d="M5 29h28M10 29a4 4 0 1 0 8 0M24 29a4 4 0 1 0 8 0M18 19v-5h9l3 5"/></svg>`;
  }
  if (type === "dumper") {
    return `<svg viewBox="0 0 40 40" aria-hidden="true"><path d="M7 11h21l-3 12H10L7 11Zm20 8h6l3 6v5H8v-7M12 30a4 4 0 1 0 8 0M27 30a4 4 0 1 0 8 0"/></svg>`;
  }
  return `<svg viewBox="0 0 40 40" aria-hidden="true"><path d="M9 28V12h13v16M22 19h7l3 9M6 28h29M10 29a4 4 0 1 0 8 0M26 29a4 4 0 1 0 8 0M28 12v7M28 12h5"/></svg>`;
}

function formatAgentName(agent) {
  return agent.name?.trim() || `${AGENT_META[agent.agent_type]?.name || "Агент"} ${String(agent.id).padStart(2, "0")}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function apiErrorMessage(payload, fallback = "Сервер отклонил запрос") {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (typeof item === "string") return item;
      const field = Array.isArray(item?.loc) ? item.loc.filter((part) => part !== "body").join(" → ") : "";
      const message = item?.msg || item?.message;
      return message ? `${field ? `${field}: ` : ""}${message}` : "";
    }).filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  if (detail && typeof detail === "object") return detail.message || detail.msg || fallback;
  return fallback;
}

function roleOptions(selected) {
  return Object.entries(ROLE_LABELS).map(([value, label]) => `<option value="${value}" ${value === selected ? "selected" : ""}>${label}</option>`).join("");
}

function typeOptions(selected) {
  return Object.entries(AGENT_META).map(([value, meta]) => `<option value="${value}" ${value === selected ? "selected" : ""}>${meta.name}</option>`).join("");
}

function updateClock() {
  const now = new Date();
  $("#clock-time").textContent = now.toLocaleTimeString("ru-RU", { hour12: false });
  $("#clock-date").textContent = now.toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric" });
}

function switchPage(page) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.page === page));
  $$(".page").forEach((item) => item.classList.toggle("active", item.id === `${page}-page`));
  window.stereoViewer?.setActive(page === "stereo");
  if (page === "markers") {
    appState.draftMarkers = appState.markers.map((marker) => ({ ...marker }));
    renderMarkerTable();
  }
}

function mapPoint(x, y) {
  return { x: 80 + x * 8.8, y: 620 - y * 5.6 };
}

function realPoint(x, y) {
  return { x: Math.max(0, Math.min(100, (x - 80) / 8.8)), y: Math.max(0, Math.min(100, (620 - y) / 5.6)) };
}

function orderedBoundaryMarkers() {
  const items = appState.markers.filter((m) => m.role === "vertex");
  if (items.length < 3) return [];
  const cx = items.reduce((sum, item) => sum + item.x, 0) / items.length;
  const cy = items.reduce((sum, item) => sum + item.y, 0) / items.length;
  return [...items].sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
}

function currentPolygonPoints() {
  const ordered = orderedBoundaryMarkers();
  if (ordered.length < 3) return [{ x: 80, y: 80 }, { x: 920, y: 68 }, { x: 960, y: 570 }, { x: 120, y: 610 }];
  return ordered.map((marker) => mapPoint(marker.x, marker.y));
}

function polygonString(points) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const crosses = ((yi > point.y) !== (yj > point.y)) && (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
    if (crosses) inside = !inside;
  }
  return inside;
}

function pointInWorkRegion(point) {
  return appState.regions.some((region) => region.type === "work"
    && point.x >= region.x && point.x <= region.x + region.width
    && point.y >= region.y && point.y <= region.y + region.height);
}

function renderAgentList() {
  const visibleAgents = appState.agents.filter((agent) => agent.seen);
  const list = $("#agent-list");
  // Camera snapshots arrive several times per second and only change an
  // agent's coordinates. Replacing the cards on every snapshot removes the
  // hovered button from under the cursor, which makes it impossible to click.
  // Rebuild the list only if something visible on a card has actually changed.
  const renderKey = JSON.stringify(visibleAgents.map((agent) => [
    agent.id,
    agent.agent_type,
    agent.name,
    agent.status,
    Boolean(agent.is_controlled),
    agent.id === appState.selectedAgentId,
  ]));
  $("#agent-count").textContent = visibleAgents.length;
  if (list.dataset.renderKey === renderKey) return;
  list.dataset.renderKey = renderKey;
  list.innerHTML = visibleAgents.length ? visibleAgents.map((agent) => `
    <button class="agent-card ${agent.id === appState.selectedAgentId ? "selected" : ""}" data-agent-id="${agent.id}" data-type="${agent.agent_type}">
      <span class="agent-avatar">${agentIcon(agent.agent_type)}</span>
      <span class="agent-info"><b>${escapeHtml(formatAgentName(agent))}</b><span>${agent.manual ? "БЕЗ ARUCO · WASD" : `ID ${agent.id}`} · ${agent.status === "moving" ? (agent.manual ? "Ручное движение" : "Выполняет маршрут") : agent.status === "stuck" ? "Застрял" : agent.status === "disconnected" ? "Нет связи с ESP32" : agent.status === "offline" ? "Метка не видна" : agent.is_controlled ? "Выбран для управления" : "Готов к выбору"}</span></span>
      <i class="agent-state ${agent.status}"></i>
    </button>`).join("") : `<div class="agent-list-empty"><span>⌁</span><b>Агенты не видны</b><small>Назначенные метки появятся здесь, когда камера их распознает</small></div>`;
  list.querySelectorAll(".agent-card").forEach((card) => card.addEventListener("click", () => selectAgent(Number(card.dataset.agentId))));
}

function createSvg(tag, attrs = {}) {
  const node = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function regionToSvg(region) {
  const topLeft = mapPoint(region.x, region.y + region.height);
  return { x: topLeft.x, y: topLeft.y, width: region.width * 8.8, height: region.height * 5.6 };
}

function selectRegion(id) {
  appState.selectedRegionId = id;
  renderRegions();
  renderRegionList();
}

function renderRegions() {
  const layer = $("#region-layer");
  const renderKey = JSON.stringify({
    selected: appState.selectedRegionId,
    regions: appState.regions.map(({ id, type, x, y, width, height }) => [id, type, x, y, width, height]),
  });
  // Camera state arrives many times per second. Preserve region DOM while it
  // is unchanged, otherwise recreating a hovered region causes hover flicker.
  if (layer.dataset.renderKey === renderKey) return;
  layer.dataset.renderKey = renderKey;
  layer.replaceChildren();
  appState.regions.forEach((region, index) => {
    const box = regionToSvg(region);
    const meta = REGION_META[region.type] || REGION_META.work;
    const group = createSvg("g", {
      class: `region-shape ${region.type} ${region.id === appState.selectedRegionId ? "selected" : ""}`,
      "data-region-id": region.id,
    });
    group.append(createSvg("rect", { class: "region-area", x: box.x, y: box.y, width: box.width, height: box.height, rx: 3 }));
    [[box.x, box.y], [box.x + box.width, box.y], [box.x + box.width, box.y + box.height], [box.x, box.y + box.height]].forEach(([x, y]) => {
      group.append(createSvg("rect", { class: "region-corner", x: x - 4, y: y - 4, width: 8, height: 8, rx: 1 }));
    });
    const labelWidth = Math.max(76, meta.name.length * 7 + 22);
    group.append(createSvg("rect", { class: "region-label-bg", x: box.x + 8, y: box.y + 8, width: labelWidth, height: 23, rx: 3 }));
    const label = createSvg("text", { class: "region-label", x: box.x + 17, y: box.y + 24 });
    label.textContent = `${String(index + 1).padStart(2, "0")} · ${meta.name.toUpperCase()}`;
    group.append(label);
    group.addEventListener("click", (event) => {
      if (appState.choosingTarget || appState.activeRegionType) return;
      event.stopPropagation();
      selectRegion(region.id);
    });
    layer.append(group);
  });
}

function renderRegionList() {
  $("#region-count").textContent = appState.regions.length;
  const list = $("#region-list");
  const renderKey = JSON.stringify({
    selected: appState.selectedRegionId,
    regions: appState.regions.map(({ id, type, width, height }) => [id, type, width, height]),
  });
  if (list.dataset.renderKey === renderKey) return;
  list.dataset.renderKey = renderKey;
  if (!appState.regions.length) {
    list.innerHTML = "<p>Областей пока нет</p>";
    return;
  }
  list.innerHTML = appState.regions.map((region, index) => {
    const meta = REGION_META[region.type] || REGION_META.work;
    return `<div class="region-list-item ${region.type} ${region.id === appState.selectedRegionId ? "selected" : ""}" data-region-id="${region.id}">
      <i></i><span><b>${index + 1}. ${meta.name}</b><small>${region.width.toFixed(1)} × ${region.height.toFixed(1)}% поля</small></span>
      <button class="region-delete" data-delete-region="${region.id}" aria-label="Удалить область">×</button>
    </div>`;
  }).join("");
  $$(".region-list-item").forEach((item) => item.addEventListener("click", (event) => {
    if (event.target.closest(".region-delete")) return;
    selectRegion(item.dataset.regionId);
  }));
  $$(".region-delete").forEach((button) => button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    deleteRegion(button.dataset.deleteRegion, button);
  }));
}

function setRegionMode(type = null) {
  appState.activeRegionType = type;
  appState.regionDraft = null;
  $("#region-preview-layer").replaceChildren();
  $$("[data-region-type]").forEach((button) => button.classList.toggle("active", button.dataset.regionType === type));
  $("#map-card").classList.toggle("region-drawing", Boolean(type));
  $("#region-draw-hint").classList.toggle("visible", Boolean(type));
  if (type) {
    setTargetMode(false);
    const meta = REGION_META[type];
    $("#region-hint-title").textContent = `Нарисуйте: ${meta.name}`;
  }
}

function drawRegionPreview() {
  const layer = $("#region-preview-layer");
  layer.replaceChildren();
  if (!appState.regionDraft) return;
  const { start, current } = appState.regionDraft;
  const x = Math.min(start.x, current.x);
  const y = Math.min(start.y, current.y);
  const width = Math.abs(current.x - start.x);
  const height = Math.abs(current.y - start.y);
  const corners = [{ x, y }, { x: x + width, y }, { x: x + width, y: y + height }, { x, y: y + height }];
  const valid = corners.every((point) => pointInPolygon(point, currentPolygonPoints()));
  const rect = createSvg("rect", { class: `region-preview ${valid ? "" : "invalid"}`, x, y, width, height, rx: 3 });
  layer.append(rect);
  corners.forEach((point) => layer.append(createSvg("rect", { class: "region-preview-handle", x: point.x - 4, y: point.y - 4, width: 8, height: 8, rx: 1 })));
  if (width > 12 && height > 12) {
    const realWidth = width / 8.8;
    const realHeight = height / 5.6;
    const labelWidth = 105;
    layer.append(createSvg("rect", { class: "region-preview-label-bg", x: x + width / 2 - labelWidth / 2, y: y + height / 2 - 12, width: labelWidth, height: 24, rx: 4 }));
    const label = createSvg("text", { class: "region-preview-label", x: x + width / 2, y: y + height / 2 + 4, "text-anchor": "middle" });
    label.textContent = `${realWidth.toFixed(1)} × ${realHeight.toFixed(1)}%`;
    layer.append(label);
  }
  appState.regionDraft.valid = valid;
}

function beginRegionDraw(event) {
  if (!appState.activeRegionType || event.button !== 0 || event.target.closest(".agent-node")) return;
  const start = eventToSvgPoint(event);
  if (!pointInPolygon(start, currentPolygonPoints())) return notify("Начните внутри поля", "Область должна находиться в границе ArUco", false);
  event.preventDefault();
  $("#field-map").setPointerCapture(event.pointerId);
  appState.regionDraft = { start, current: start, pointerId: event.pointerId, valid: true };
  drawRegionPreview();
}

function moveRegionDraw(event) {
  if (!appState.regionDraft || appState.regionDraft.pointerId !== event.pointerId) return;
  appState.regionDraft.current = eventToSvgPoint(event);
  drawRegionPreview();
}

async function finishRegionDraw(event) {
  const draft = appState.regionDraft;
  if (!draft || draft.pointerId !== event.pointerId) return;
  appState.regionDraft = null;
  $("#region-preview-layer").replaceChildren();
  const left = Math.min(draft.start.x, draft.current.x);
  const right = Math.max(draft.start.x, draft.current.x);
  const top = Math.min(draft.start.y, draft.current.y);
  const bottom = Math.max(draft.start.y, draft.current.y);
  const width = (right - left) / 8.8;
  const height = (bottom - top) / 5.6;
  if (width < 1 || height < 1) return notify("Слишком маленькая область", "Протяните рамку немного дальше", false);
  if (!draft.valid) return notify("Область выходит за поле", "Все четыре угла должны быть внутри границы ArUco", false);
  const realTopLeft = realPoint(left, top);
  const payload = {
    type: appState.activeRegionType,
    x: realTopLeft.x,
    y: realTopLeft.y - height,
    width,
    height,
  };
  try {
    const response = await fetch("/api/regions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!response.ok) throw new Error(apiErrorMessage(await response.json(), "Не удалось создать область"));
    const result = await response.json();
    commandAccepted = response.ok;
    appState.selectedRegionId = result.region.id;
    notify("Область создана", REGION_META[payload.type].name);
  } catch (error) {
    notify("Ошибка области", error.message, false);
  }
}

async function deleteRegion(id, button = null) {
  if (appState.deletingRegionIds.has(id)) return;
  appState.deletingRegionIds.add(id);
  if (button) {
    button.disabled = true;
    button.classList.add("deleting");
    button.textContent = "…";
  }
  try {
    const response = await fetch(`/api/regions/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(apiErrorMessage(await response.json(), "Не удалось удалить область"));
    if (appState.selectedRegionId === id) appState.selectedRegionId = null;
    appState.regions = appState.regions.filter((region) => region.id !== id);
    renderRegions();
    renderRegionList();
    notify("Область удалена", "Разметка поля обновлена");
  } catch (error) {
    if (button && button.isConnected) {
      button.disabled = false;
      button.classList.remove("deleting");
      button.textContent = "×";
    }
    notify("Ошибка удаления", error.message, false);
  } finally {
    appState.deletingRegionIds.delete(id);
  }
}

function renderMap() {
  const polygon = currentPolygonPoints();
  const points = polygonString(polygon);
  const boundaryReady = Boolean(appState.camera?.calibrated && orderedBoundaryMarkers().length === 4);
  $("#field-polygon").setAttribute("points", points);
  $("#field-polygon").style.visibility = boundaryReady ? "visible" : "hidden";
  $("#field-clip-polygon").setAttribute("points", points);
  $("#camera-feed").style.clipPath = boundaryReady ? `polygon(${polygon.map((point) => `${point.x / 10}% ${point.y / 6.8}%`).join(",")})` : "none";
  renderRegions();
  renderRegionList();

  const markerLayer = $("#marker-layer");
  markerLayer.replaceChildren();
  appState.markers.filter((m) => m.role === "vertex" || m.role === "edge").forEach((marker) => {
    const point = mapPoint(marker.x, marker.y);
    const group = createSvg("g", { class: marker.role === "vertex" ? "boundary-marker" : "edge-marker", transform: `translate(${point.x} ${point.y})` });
    if (marker.role === "vertex") {
      group.append(createSvg("rect", { class: "outer", x: -18, y: -18, width: 36, height: 36, rx: 2 }));
      group.append(createSvg("rect", { class: "inner", x: -11, y: -11, width: 22, height: 22 }));
      const label = createSvg("text", { y: 4 });
      label.textContent = marker.id;
      label.setAttribute("fill", "white");
      group.append(label);
    } else {
      group.append(createSvg("circle", { r: 16 }));
      const label = createSvg("text"); label.textContent = marker.id; group.append(label);
    }
    markerLayer.append(group);
  });

  const routeLayer = $("#route-layer");
  routeLayer.replaceChildren();
  appState.agents.filter((agent) => agent.target).forEach((agent) => {
    const start = mapPoint(agent.x, agent.y);
    const target = mapPoint(agent.target.x, agent.target.y);
    const routeIndex = Math.max(0, Number(agent.debug_route_index || 0));
    const debugRoute = Array.isArray(agent.debug_route) ? agent.debug_route.slice(routeIndex) : [];
    if (appState.debugMode && debugRoute.length) {
      const routePoints = [start, ...debugRoute.map((point) => mapPoint(point.x, point.y))];
      const routePath = routePoints.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" ");
      routeLayer.append(createSvg("path", { class: "debug-route-line", d: routePath }));
      const next = routePoints[1];
      routeLayer.append(createSvg("path", {
        class: "debug-motion-vector",
        d: `M${start.x},${start.y} L${next.x},${next.y}`,
        "marker-end": "url(#debug-arrow)",
      }));
      routePoints.slice(1, -1).forEach((point, index) => {
        routeLayer.append(createSvg("circle", { class: "debug-waypoint", cx: point.x, cy: point.y, r: index === 0 ? 5 : 3 }));
      });
    } else {
      routeLayer.append(createSvg("path", { class: "route-line", d: `M${start.x},${start.y} L${target.x},${target.y}` }));
    }
    const targetGroup = createSvg("g", { class: "target-node", transform: `translate(${target.x} ${target.y})` });
    targetGroup.append(createSvg("circle", { r: 10 }));
    targetGroup.append(createSvg("circle", { r: 4 }));
    const targetLabel = createSvg("text", { class: "target-label", x: 15, y: -12 });
    targetLabel.textContent = "ЦЕЛЬ";
    targetGroup.append(targetLabel);
    routeLayer.append(targetGroup);
  });

  const agentLayer = $("#agent-layer");
  agentLayer.replaceChildren();
  appState.agents.filter((agent) => agent.seen && !agent.manual).forEach((agent) => {
    const point = mapPoint(agent.x, agent.y);
    const group = createSvg("g", {
      class: `agent-node ${agent.agent_type} ${agent.id === appState.selectedAgentId ? "selected" : ""}`,
      transform: `translate(${point.x} ${point.y})`,
      "data-agent-id": agent.id,
      role: "button",
      tabindex: "0",
    });
    group.append(createSvg("circle", { class: "selection-ring", r: 33 }));
    group.append(createSvg("circle", { class: "agent-body", r: 24 }));
    const symbol = createSvg("path", { class: "agent-symbol", d: agent.agent_type === "dumper" ? "M-11-6h17l-3 11H-7zM7 0h7l4 8M-8 11h23" : agent.agent_type === "bulldozer" ? "M-12 8h20l7-8H2l-4-9h-10zM-12 12h26" : "M-10 10V-8H2v18M2-2h8l4 12M-13 11h29" });
    group.append(symbol);
    // Camera heading uses a downward Y axis; SVG uses an upward map Y axis.
    const heading = Number(agent.heading || 0);
    group.append(createSvg("path", {
      class: "agent-heading",
      d: "M30 0 L17 -7 L20 0 L17 7 Z",
      transform: `rotate(${heading})`,
    }));
    const idBg = createSvg("circle", { class: "agent-id-bg", cx: 18, cy: -18, r: 11 });
    group.append(idBg);
    const text = createSvg("text", { x: 18, y: -18 }); text.textContent = agent.id; group.append(text);
    group.addEventListener("click", (event) => { event.stopPropagation(); selectAgent(agent.id); });
    group.addEventListener("keydown", (event) => { if (event.key === "Enter") selectAgent(agent.id); });
    agentLayer.append(group);
  });
  updateMissionPanel();
}

async function selectAgent(id) {
  const previous = appState.agents.find((item) => item.id === appState.selectedAgentId);
  if (previous?.manual) stopManualDrive();
  const selected = appState.agents.find((item) => item.id === id);
  appState.selectedAgentId = id;
  appState.choosingTarget = false;
  appState.agents.forEach((agent) => { agent.is_controlled = agent.id === id; });
  renderAgentList();
  renderMap();
  if (selected?.manual) setTargetMode(false);
  try {
    const response = await fetch("/api/control-agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: id }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Агент не выбран"));
    if (!selected?.manual) setTargetMode(true);
  } catch (error) {
    notify("Не удалось выбрать агента", error.message, false);
  }
}

function updateControllerSettings(agent, force = false) {
  const editor = $("#controller-settings");
  if (!force && editor?.contains(document.activeElement)) return;
  const settings = agent.controller_settings || {};
  const speed = Number(settings.speed ?? 255);
  $("#controller-speed").value = String(speed);
  $("#controller-speed-value").textContent = String(speed);
  $("#controller-target-tolerance").value = String(settings.target_tolerance_cm ?? 3);
  $("#controller-heading-tolerance").value = String(settings.heading_tolerance_deg ?? 12);
  $("#controller-heading-kp").value = String(settings.heading_kp ?? 1.8);
  $("#controller-heading-kd").value = String(settings.heading_kd ?? 0.18);
  $("#controller-min-drive-pwm").value = String(settings.min_drive_pwm ?? 80);
  $("#controller-in-place-turn-threshold").value = String(settings.in_place_turn_threshold_deg ?? 90);
  $("#controller-heading-offset").value = String(settings.heading_offset_deg ?? 0);
  $("#controller-left-motor-inverted").checked = Boolean(settings.left_motor_inverted);
  $("#controller-right-motor-inverted").checked = Boolean(settings.right_motor_inverted);
  $("#controller-stuck-timeout").value = String(settings.stuck_timeout_s ?? 2.5);
  $("#controller-stuck-progress").value = String(settings.stuck_min_progress_cm ?? 1.5);
  $("#controller-stuck-action").value = settings.stuck_action ?? "stop";
  $("#controller-stuck-boost").value = String(settings.stuck_boost_pwm ?? 35);
  updateStuckActionUi();
}

function updateStuckActionUi() {
  const boostEnabled = $("#controller-stuck-action").value === "boost";
  $("#stuck-boost-toggle").textContent = `Повышение ШИМ при застревании: ${boostEnabled ? "вкл" : "выкл"}`;
  $("#controller-stuck-boost").disabled = !boostEnabled;
}

async function saveControllerSettings() {
  const agent = appState.agents.find((item) => item.id === appState.selectedAgentId);
  if (!agent) return;
  const payload = {
    speed: Number($("#controller-speed").value),
    target_tolerance_cm: Number($("#controller-target-tolerance").value),
    heading_tolerance_deg: Number($("#controller-heading-tolerance").value),
    heading_kp: Number($("#controller-heading-kp").value),
    heading_kd: Number($("#controller-heading-kd").value),
    min_drive_pwm: Number($("#controller-min-drive-pwm").value),
    in_place_turn_threshold_deg: Number($("#controller-in-place-turn-threshold").value),
    heading_offset_deg: Number($("#controller-heading-offset").value),
    left_motor_inverted: $("#controller-left-motor-inverted").checked,
    right_motor_inverted: $("#controller-right-motor-inverted").checked,
    stuck_timeout_s: Number($("#controller-stuck-timeout").value),
    stuck_min_progress_cm: Number($("#controller-stuck-progress").value),
    stuck_action: $("#controller-stuck-action").value,
    stuck_boost_pwm: Number($("#controller-stuck-boost").value),
  };
  try {
    const response = await fetch(`/api/agents/${agent.id}/controller-settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Настройки регулятора отклонены"));
    agent.controller_settings = result.settings;
    updateControllerSettings(agent, true);
    notify("Настройки сохранены", `${formatAgentName(agent)} · регулятор движения обновлён`);
  } catch (error) {
    notify("Ошибка настроек", error.message, false);
  }
}

function updateMotorOutput(output = {}) {
  const command = output.command || "STOP";
  // Keep PD calculation and ESP feedback side by side for diagnosing wiring,
  // driver power and stale commands. The bar is confirmed ESP output only.
  const apply = (side) => {
    const value = Math.max(-100, Math.min(100, Number(output[`${side}_percent`] || 0)));
    const serverPwm = Math.round(Number(output[`server_${side}_pwm`] || 0));
    const espPwm = output.confirmed ? Math.round(Number(output[`${side}_pwm`] || 0)) : null;
    const fill = $(`#motor-${side}-fill`);
    fill.style.width = `${Math.abs(value)}%`;
    fill.classList.toggle("reverse", value < 0);
    $(`#motor-${side}-value`).textContent = `С:${serverPwm} · ESP:${espPwm ?? "—"}`;
  };
  $("#motor-command").textContent = command;
  apply("left");
  apply("right");
}

function updateServoFeedback(agent) {
  const outputByChannel = new Map((agent?.servo_output?.channels || []).map((item) => [Number(item.channel), item]));
  const settingsByChannel = new Map((agent?.servo_settings?.channels || []).map((item) => [Number(item.channel), item]));
  $$(".servo-channel").forEach((row) => {
    const channel = Number(row.dataset.servoChannel);
    const output = outputByChannel.get(channel) || {};
    const setting = settingsByChannel.get(channel) || {};
    const requested = output.requested_angle ?? setting.position;
    const esp = output.active && output.esp_angle != null ? `${output.esp_angle}°` : "—";
    const feedback = row.querySelector(".servo-feedback");
    feedback.textContent = `СЕРВЕР: ${requested ?? "—"}° · ESP: ${esp}`;
    feedback.classList.toggle("confirmed", Boolean(output.confirmed));
  });
}

function bindServoChannelEvents(row) {
  const range = row.querySelector(".servo-position");
  const number = row.querySelector(".servo-position-number");
  const minimum = row.querySelector(".servo-min");
  const maximum = row.querySelector(".servo-max");
  const syncPosition = (value) => {
    const min = Number(minimum.value);
    const max = Number(maximum.value);
    const angle = Math.round(Math.max(min, Math.min(max, Number(value))));
    range.min = String(min);
    range.max = String(max);
    range.value = String(angle);
    number.min = String(min);
    number.max = String(max);
    number.value = String(angle);
  };
  range.addEventListener("input", () => syncPosition(range.value));
  number.addEventListener("input", () => syncPosition(number.value));
  [minimum, maximum].forEach((input) => input.addEventListener("input", () => {
    if (Number(minimum.value) < Number(maximum.value)) syncPosition(number.value);
  }));
  row.querySelector(".servo-apply").addEventListener("click", () => setServoPosition(Number(row.dataset.servoChannel)));
}

function updateServoControls(agent, force = false) {
  const editor = $("#servo-settings");
  const channels = agent?.servo_settings?.channels || [];
  editor.hidden = !agent || !channels.length;
  if (!agent || !channels.length) return;
  const agentChanged = editor.dataset.agentId !== String(agent.id);
  if (!force && !agentChanged && editor.contains(document.activeElement)) {
    updateServoFeedback(agent);
    return;
  }
  const renderKey = JSON.stringify([
    agent.id,
    agent.agent_type,
    channels.map((item) => [item.channel, item.name, item.min_angle, item.max_angle]),
  ]);
  if (force || editor.dataset.renderKey !== renderKey) {
    editor.dataset.agentId = String(agent.id);
    editor.dataset.renderKey = renderKey;
    $("#servo-channels").innerHTML = channels.map((channel) => `
      <div class="servo-channel" data-servo-channel="${channel.channel}">
        <div class="servo-channel-head"><b>${escapeHtml(channel.name)}</b><span class="servo-feedback">СЕРВЕР: ${channel.position}° · ESP: —</span></div>
        <div class="servo-position-row">
          <input class="servo-position" type="range" min="${channel.min_angle}" max="${channel.max_angle}" step="1" value="${channel.position}" aria-label="${escapeHtml(channel.name)} — положение">
          <input class="servo-position-number" type="number" min="${channel.min_angle}" max="${channel.max_angle}" step="1" value="${channel.position}" aria-label="${escapeHtml(channel.name)} — угол">
        </div>
        <div class="servo-limits">
          <label>Мин., ° <input class="servo-min" type="number" min="0" max="179" step="1" value="${channel.min_angle}"></label>
          <label>Макс., ° <input class="servo-max" type="number" min="1" max="180" step="1" value="${channel.max_angle}"></label>
        </div>
        <button class="servo-apply" type="button">Установить положение</button>
      </div>
    `).join("");
    $$(".servo-channel").forEach(bindServoChannelEvents);
  }
  const blocked = !agent.is_controlled || !appState.robot.connected || appState.robot.phase === "EMERGENCY_STOP";
  $$(".servo-apply").forEach((button) => {
    button.disabled = blocked;
    button.title = blocked ? "Дождитесь связи с ESP32 и снимите аварийную остановку" : "Отправить угол на ESP32";
  });
  updateServoFeedback(agent);
}

async function saveServoSettings() {
  const agent = appState.agents.find((item) => item.id === appState.selectedAgentId);
  if (!agent) return;
  const channels = $$(".servo-channel").map((row) => ({
    channel: Number(row.dataset.servoChannel),
    min_angle: Number(row.querySelector(".servo-min").value),
    max_angle: Number(row.querySelector(".servo-max").value),
    position: Number(row.querySelector(".servo-position-number").value),
  }));
  const invalid = channels.find((item) => item.min_angle >= item.max_angle || item.position < item.min_angle || item.position > item.max_angle);
  if (invalid) return notify("Неверные пределы серво", "Минимум должен быть меньше максимума, а положение — находиться внутри диапазона", false);
  try {
    const response = await fetch(`/api/agents/${agent.id}/servo-settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channels }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Настройки сервоприводов отклонены"));
    agent.servo_settings = result.settings;
    updateServoControls(agent, true);
    notify("Пределы сервоприводов сохранены", formatAgentName(agent));
  } catch (error) {
    notify("Ошибка настроек серво", error.message, false);
  }
}

async function setServoPosition(channel) {
  const agent = appState.agents.find((item) => item.id === appState.selectedAgentId);
  const row = $(`.servo-channel[data-servo-channel="${channel}"]`);
  if (!agent || !row) return;
  const button = row.querySelector(".servo-apply");
  const angle = Number(row.querySelector(".servo-position-number").value);
  button.disabled = true;
  button.textContent = "Отправка…";
  try {
    const response = await fetch(`/api/agents/${agent.id}/servos/${channel}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ angle }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "ESP32 не приняла угол сервопривода"));
    agent.servo_settings = result.settings;
    agent.servo_output = result.servo_output;
    updateServoControls(agent, true);
    notify("Положение отправлено", `${formatAgentName(agent)} · ${angle}°`);
  } catch (error) {
    button.disabled = false;
    button.textContent = "Установить положение";
    notify("Сервопривод недоступен", error.message, false);
  }
}

function updateMissionPanel() {
  const agent = appState.agents.find((item) => item.id === appState.selectedAgentId);
  $("#empty-mission").hidden = Boolean(agent);
  $("#mission-details").hidden = !agent;
  $("#manual-controls").hidden = !agent?.manual;
  $("#target-mission-step").hidden = Boolean(agent?.manual);
  $("#select-target-button").hidden = Boolean(agent?.manual);
  updateDebugUi();
  updateServoControls(agent);
  if (!agent) return;
  const manual = Boolean(agent.manual);
  updateControllerSettings(agent);
  updateMotorOutput(agent.motor_output);
  $("#mission-icon").className = `agent-avatar large ${agent.agent_type}`;
  $("#mission-icon").innerHTML = agentIcon(agent.agent_type);
  $("#mission-name").textContent = formatAgentName(agent);
  $("#mission-id").textContent = manual ? `Без ArUco · внутренний ID ${agent.id}` : `ArUco ID ${agent.id}`;
  $("#mission-status").textContent = agent.status === "moving" ? (manual ? "Ручное движение" : "В движении") : agent.status === "stuck" ? "Застрял" : agent.status === "disconnected" ? "Нет связи с ESP32" : agent.status === "offline" ? "Не виден" : manual ? "WASD готов" : "Готов";
  $("#mission-battery").textContent = manual || agent.battery == null ? "—" : `${agent.battery}%`;
  $("#mission-coords").textContent = manual ? "Без камеры" : `${agent.x.toFixed(1)} / ${agent.y.toFixed(1)}`;
  $("#mission-heading").textContent = manual ? "—" : `${Math.round(agent.heading || 0)}°`;
  $("#target-copy").textContent = manual ? "Удерживайте W, A, S или D" : agent.target ? `X ${agent.target.x.toFixed(1)} · Y ${agent.target.y.toFixed(1)}` : "Укажите точку на карте";
  $("#command-log p").textContent = agent.status === "moving" ? (manual ? "Клавиша удерживается · команда обновляется" : "Команда принята · агент следует по маршруту") : agent.status === "stuck" ? "Нет движения по данным камеры · робот остановлен" : agent.status === "disconnected" ? `Нет связи с ESP32 по адресу ${agent.ip_address || "—"}` : manual ? "Ручной режим готов · удерживайте W/A/S/D" : "Агент готов к приёму команд";
  $("#command-log .log-dot").style.background = agent.status === "moving" ? "#f78b45" : (agent.status === "stuck" || agent.status === "disconnected") ? "#d84a3d" : "#8da938";
}

function updateDebugUi() {
  const toggle = $("#debug-mode-toggle");
  const controls = $("#debug-controls");
  if (toggle) toggle.checked = appState.debugMode;
  $("#debug-toggle")?.classList.toggle("active", appState.debugMode);
  if (controls) controls.hidden = !(appState.debugMode && appState.selectedAgentId !== null);
  const speed = $("#debug-speed");
  if (speed) speed.value = String(appState.debugSpeed);
  const value = $("#debug-speed-value");
  if (value) value.textContent = String(appState.debugSpeed);
  $("#map-card")?.classList.toggle("debug-mode", appState.debugMode);
}

async function setDebugMode(enabled) {
  const previous = appState.debugMode;
  appState.debugMode = Boolean(enabled);
  if (!appState.debugMode) setTargetMode(false);
  updateDebugUi();
  renderMap();
  try {
    const response = await fetch("/api/debug-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: appState.debugMode }),
    });
    if (!response.ok) throw new Error(apiErrorMessage(await response.json(), "Режим не переключён"));
    notify(appState.debugMode ? "Режим отладки включён" : "Обычный режим включён",
      appState.debugMode ? "Доступны скорость и безопасный маршрут по рабочей области" : "Управление возвращено в обычный режим");
  } catch (error) {
    appState.debugMode = previous;
    updateDebugUi();
    renderMap();
    notify("Ошибка режима отладки", error.message, false);
  }
}

const PHASE_LABELS = {
  WAIT_FIELD: "ОЖИДАНИЕ ПОЛЯ",
  CAL_FORWARD: "КАЛИБРОВКА · ВПЕРЁД",
  CAL_PAUSE_90: "КАЛИБРОВКА · ПАУЗА",
  CAL_TURN_90: "КАЛИБРОВКА · ПОВОРОТ 90°",
  CAL_PAUSE_180: "КАЛИБРОВКА · ПАУЗА",
  CAL_TURN_180: "КАЛИБРОВКА · ПОВОРОТ 180°",
  READY: "ГОТОВ К ЦЕЛИ",
  NAVIGATING: "ДВИЖЕНИЕ К ЦЕЛИ",
  TARGET_REACHED: "ЦЕЛЬ ДОСТИГНУТА",
  WORK_BOUNDARY_STOP: "ГРАНИЦА РАБОЧЕЙ ЗОНЫ",
  STUCK: "РОБОТ ЗАСТРЯЛ",
  MANUAL_READY: "РУЧНОЙ РЕЖИМ · WASD",
  MANUAL_CONTROL: "РУЧНОЕ ДВИЖЕНИЕ",
  EMERGENCY_STOP: "АВАРИЙНАЯ ОСТАНОВКА",
};

function updateHardwareStatus() {
  const camera = appState.camera || {};
  const robot = appState.robot || {};
  $("#resume-stop").hidden = robot.phase !== "EMERGENCY_STOP";
  $("#clear-stuck").hidden = robot.phase !== "STUCK";
  const feed = $("#camera-feed");
  const showVideo = camera.online && appState.cameraVisible;
  if (camera.online && !feed.src) feed.src = `/api/camera/stream?t=${Date.now()}`;
  feed.classList.toggle("live", showVideo);
  $("#field-map").classList.toggle("camera-live", showVideo);
  $("#camera-toggle").classList.toggle("active", showVideo);
  $("#camera-toggle").disabled = !camera.online;
  $$("[data-region-type]").forEach((button) => {
    button.disabled = !camera.calibrated;
    button.title = camera.calibrated ? "Нарисовать область" : "Сначала назначьте 4 вершины поля";
  });
  if (!camera.calibrated && appState.activeRegionType) setRegionMode(null);
  $("#camera-offline").classList.toggle("hidden", camera.online);
  $("#camera-pill").classList.toggle("offline", !camera.online);
  $("#camera-name").textContent = "Камера 01";
  $("#camera-state").textContent = camera.online ? (camera.calibrated ? "LIVE" : "CAL") : "OFFLINE";
  $("#hardware-camera").textContent = camera.online ? (camera.calibrated ? "ПОЛЕ ВИДНО" : "НЕТ 4 ВЕРШИН") : "OFFLINE";
  $("#hardware-robot").textContent = robot.connected ? "НА СВЯЗИ" : "НЕТ СВЯЗИ";
  $("#calibration-phase").textContent = PHASE_LABELS[robot.phase] || robot.phase || "—";
  const selectedAgent = appState.agents.find((agent) => agent.id === appState.selectedAgentId);
  const manual = Boolean(selectedAgent?.manual);
  $("#system-indicator").classList.toggle("offline", !robot.connected || (!manual && !camera.online));
  $("#system-copy").textContent = manual ? (robot.connected ? "Ручное управление WASD готово" : "Ручной агент · ожидание ESP32") : !camera.online ? "Ожидание камеры" : !robot.connected ? "Камера активна · робот не подключён" : robot.calibrated ? "Система готова к управлению" : "Выполняется калибровка";
  const scanTitle = $(".scan-status b");
  if (scanTitle) scanTitle.textContent = camera.online ? "Камера активна" : "Камера не подключена";
}

function setTargetMode(enabled) {
  if (enabled && !appState.selectedAgentId) return;
  const selectedAgent = appState.agents.find((agent) => agent.id === appState.selectedAgentId);
  if (enabled && selectedAgent?.manual) return;
  if (enabled && !appState.robot?.connected) {
    notify("Робот не подключён", appState.robot?.last_error || "Подключитесь к Wi-Fi robot_pogruzchik1 и дождитесь связи с ESP32", false);
    return;
  }
  if (enabled && appState.robot?.phase === "EMERGENCY_STOP") {
    notify("Аварийная остановка активна", "Сначала нажмите «Снять аварийную остановку»", false);
    return;
  }
  if (enabled && appState.debugMode && !appState.regions.some((region) => region.type === "work")) {
    notify("Нет рабочей поверхности", "Сначала нарисуйте хотя бы одну рабочую область", false);
    return;
  }
  if (enabled && appState.activeRegionType) setRegionMode(null);
  appState.choosingTarget = enabled;
  $("#instruction-toast").classList.toggle("visible", enabled);
  $("#map-card").style.outline = enabled ? "2px solid #d84a3d" : "none";
  $("#field-map").classList.toggle("targeting", enabled);
}

function eventToSvgPoint(event) {
  const svg = $("#field-map");
  const point = new DOMPoint(event.clientX, event.clientY);
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

async function sendMoveCommand(point) {
  const real = realPoint(point.x, point.y);
  if (appState.debugMode && !pointInWorkRegion(real)) {
    setTargetMode(false);
    notify("Точка вне рабочей поверхности", "В режиме отладки цель можно поставить только внутри рабочей области", false);
    return;
  }
  const agent = appState.agents.find((item) => item.id === appState.selectedAgentId);
  const previous = agent ? {
    target: agent.target ? { ...agent.target } : null,
    status: agent.status,
    debug_route: agent.debug_route,
    debug_route_index: agent.debug_route_index,
  } : null;
  if (agent) {
    // Give immediate visual feedback; undo it if the controller rejects this target.
    agent.target = { x: real.x, y: real.y };
    agent.status = "moving";
    renderAgentList();
    renderMap();
  }
  let commandAccepted = false;
  try {
    const response = await fetch("/api/commands/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_id: appState.selectedAgentId,
        target_x: real.x,
        target_y: real.y,
        task_type: "move",
        debug_mode: appState.debugMode,
        speed: appState.debugSpeed,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Команда отклонена"));
    if (agent) {
      agent.target = { x: real.x, y: real.y };
      agent.debug_route = result.route || [];
      agent.debug_route_index = 0;
      agent.debug_speed = appState.debugSpeed;
      agent.status = "moving";
      renderMap();
      renderAgentList();
    }
    notify("Команда отправлена", `${formatAgentName(agent)} · X ${real.x.toFixed(1)}, Y ${real.y.toFixed(1)}`);
  } catch (error) {
    notify("Ошибка команды", error.message, false);
  } finally {
    if (!commandAccepted && agent && previous) {
      agent.target = previous.target;
      agent.status = previous.status;
      agent.debug_route = previous.debug_route;
      agent.debug_route_index = previous.debug_route_index;
      renderAgentList();
      renderMap();
    }
    setTargetMode(false);
  }
}

function renderMarkerTable(force = false) {
  const markers = appState.draftMarkers;
  const visibleMarkers = markers.map((marker, index) => ({ marker, index })).filter(({ marker }) => marker.seen);
  const tableBody = $("#marker-table-body");
  const renderKey = JSON.stringify(visibleMarkers.map(({ marker }) => [marker.id, marker.role, marker.agent_type, marker.name, marker.ip_address, Boolean(marker.manual)]));

  // Coordinates change on every camera frame. Update them in place so native
  // select dropdowns aren't destroyed while the user is choosing a value.
  visibleMarkers.forEach(({ marker }) => {
    const row = tableBody.querySelector(`[data-marker-id="${marker.id}"]`);
    if (row) row.querySelector(".coord-value").textContent = marker.manual ? "Без камеры" : `X ${marker.x.toFixed(1)} · Y ${marker.y.toFixed(1)}`;
  });
  if (!force && tableBody.dataset.renderKey === renderKey) {
    updateSettingsStats();
    return;
  }
  const activeControl = document.activeElement;
  if (!force && activeControl && tableBody.contains(activeControl) && activeControl.matches(".role-select, .type-select, .name-input, .ip-input")) {
    updateSettingsStats();
    return;
  }

  tableBody.dataset.renderKey = renderKey;
  tableBody.innerHTML = visibleMarkers.length ? visibleMarkers.map(({ marker, index }) => `
    <tr data-marker-index="${index}" data-marker-id="${marker.id}">
      <td><span class="marker-id">${marker.manual ? `<span class="manual-mini">WASD</span>БЕЗ ARUCO` : `<span class="aruco-mini"><img src="/api/aruco/${marker.id}.png" alt="ArUco ID ${marker.id}"></span>ID ${marker.id}`}</span></td>
      <td><span class="coord-value">${marker.manual ? "Без камеры" : `X ${marker.x.toFixed(1)} · Y ${marker.y.toFixed(1)}`}</span></td>
      <td><select class="role-select" data-index="${index}" ${marker.manual ? "disabled" : ""}>${roleOptions(marker.role)}</select></td>
      <td><select class="type-select" data-index="${index}" ${marker.role !== "agent" ? "disabled" : ""}>${typeOptions(marker.agent_type || "loader")}</select></td>
      <td><input class="name-input" data-index="${index}" type="text" maxlength="48" placeholder="Погрузчик 1" value="${escapeHtml(marker.name || "")}" ${marker.role !== "agent" ? "disabled" : ""}></td>
      <td><input class="ip-input" data-index="${index}" type="text" inputmode="decimal" maxlength="45" placeholder="192.168.1.50" value="${marker.ip_address || ""}" ${marker.role !== "agent" ? "disabled" : ""}></td>
      <td>${marker.manual ? `<button class="delete-manual-agent" type="button" data-agent-id="${marker.id}">УДАЛИТЬ</button>` : `<span class="signal-value ${marker.seen ? "" : "lost"}"><i style="--signal:${marker.seen ? 92 - index * 3 : 0}%"></i><b>${marker.seen ? `${92 - index * 3}%` : "НЕ ВИДНА"}</b></span>`}</td>
    </tr>`).join("") : `<tr class="empty-marker-row"><td colspan="7"><span>⌁</span><b>В кадре нет ArUco-меток</b><small>Покажите метку камере — строка появится автоматически</small></td></tr>`;
  $$(".role-select").forEach((select) => select.addEventListener("change", (event) => {
    const index = Number(event.target.dataset.index);
    appState.draftMarkers[index].role = event.target.value;
    if (event.target.value === "agent" && !appState.draftMarkers[index].agent_type) appState.draftMarkers[index].agent_type = "loader";
    if (event.target.value !== "agent") {
      appState.draftMarkers[index].agent_type = null;
      appState.draftMarkers[index].ip_address = null;
      appState.draftMarkers[index].name = null;
    }
    renderMarkerTable(true);
  }));
  $$(".type-select").forEach((select) => select.addEventListener("change", (event) => {
    appState.draftMarkers[Number(event.target.dataset.index)].agent_type = event.target.value;
    updateSettingsStats();
  }));
  $$(".name-input").forEach((input) => input.addEventListener("input", (event) => {
    appState.draftMarkers[Number(event.target.dataset.index)].name = event.target.value.trim();
  }));
  $$(".ip-input").forEach((input) => input.addEventListener("input", (event) => {
    appState.draftMarkers[Number(event.target.dataset.index)].ip_address = event.target.value.trim();
  }));
  $$(".role-select, .type-select, .name-input, .ip-input").forEach((select) => select.addEventListener("blur", () => setTimeout(() => renderMarkerTable(), 0)));
  $$(".delete-manual-agent").forEach((button) => button.addEventListener("click", () => deleteManualAgent(Number(button.dataset.agentId))));
  updateSettingsStats();
}

function updateSettingsStats() {
  const markers = appState.draftMarkers;
  const visibleMarkers = markers.filter((m) => m.seen);
  const seenCount = visibleMarkers.filter((m) => !m.manual).length;
  const fieldCount = visibleMarkers.filter((m) => !m.manual && (m.role === "vertex" || m.role === "edge")).length;
  const agentCount = visibleMarkers.filter((m) => m.role === "agent").length;
  const unassigned = visibleMarkers.filter((m) => !m.manual && m.role === "unassigned").length;
  $("#stat-total").textContent = seenCount;
  $("#stat-field").textContent = `${fieldCount} / 8`;
  $("#stat-agents").textContent = agentCount;
  $("#stat-unassigned").textContent = unassigned;
  $("#scan-copy").textContent = appState.camera.online ? `Распознано ${seenCount} меток` : `Сохранено ${markers.length} меток`;
}

async function saveConfiguration() {
  const visualMarkers = appState.draftMarkers.filter((m) => !m.manual);
  const fieldCount = visualMarkers.filter((m) => m.role === "vertex" || m.role === "edge").length;
  const vertexCount = visualMarkers.filter((m) => m.role === "vertex").length;
  if (fieldCount > 8) return notify("Лимит превышен", "Для поля доступно не более 8 меток", false);
  if (visualMarkers.length && vertexCount !== 4) return notify("Нужны четыре вершины", "Для калибровки камеры назначьте ровно 4 вершины", false);
  const payload = appState.draftMarkers.map(({ id, role, agent_type, name, ip_address, manual }) => ({ id, role, agent_type, manual: Boolean(manual), name: role === "agent" ? (name || null) : null, ip_address: role === "agent" ? (ip_address || null) : null }));
  try {
    const response = await fetch("/api/configuration", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ markers: payload }) });
    if (!response.ok) throw new Error(apiErrorMessage(await response.json(), "Не удалось сохранить"));
    notify("Конфигурация сохранена", "Полигон и состав агентов обновлены");
    switchPage("map");
  } catch (error) {
    notify("Ошибка сохранения", error.message, false);
  }
}

function clampNumber(value, minimum, maximum, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(minimum, Math.min(maximum, numeric)) : fallback;
}

function normalizeInterfaceSettings(value = {}) {
  return {
    notificationsEnabled: value.notificationsEnabled !== false,
    soundEnabled: value.soundEnabled !== false,
    scale: Math.round(clampNumber(value.scale, 80, 120, DEFAULT_INTERFACE_SETTINGS.scale) / 5) * 5,
    notificationDuration: clampNumber(value.notificationDuration, 0.5, 30, DEFAULT_INTERFACE_SETTINGS.notificationDuration),
  };
}

function loadInterfaceSettings() {
  try {
    interfaceSettings = normalizeInterfaceSettings(JSON.parse(localStorage.getItem(INTERFACE_SETTINGS_KEY) || "{}"));
  } catch (_) {
    interfaceSettings = { ...DEFAULT_INTERFACE_SETTINGS };
  }
}

function persistInterfaceSettings() {
  try {
    localStorage.setItem(INTERFACE_SETTINGS_KEY, JSON.stringify(interfaceSettings));
  } catch (_) {
    // The interface remains usable when browser storage is unavailable.
  }
}

function updateInterfaceSettingsFields(settings = interfaceSettings) {
  $("#interface-notifications-enabled").checked = settings.notificationsEnabled;
  $("#interface-sound-enabled").checked = settings.soundEnabled;
  $("#interface-scale").value = String(settings.scale);
  $("#interface-scale-value").textContent = `${settings.scale}%`;
  $("#interface-notification-duration").value = String(settings.notificationDuration);
}

function applyInterfaceSettings(settings = interfaceSettings) {
  const normalized = normalizeInterfaceSettings(settings);
  document.documentElement.style.zoom = String(normalized.scale / 100);
  const soundButton = $("#sound-button");
  soundButton.classList.toggle("muted", !normalized.soundEnabled);
  soundButton.setAttribute("aria-pressed", String(!normalized.soundEnabled));
  soundButton.title = normalized.soundEnabled ? "Звук уведомлений включён" : "Звук уведомлений выключен";
  updateInterfaceSettingsFields(normalized);
}

function readInterfaceSettingsFields() {
  return normalizeInterfaceSettings({
    notificationsEnabled: $("#interface-notifications-enabled").checked,
    soundEnabled: $("#interface-sound-enabled").checked,
    scale: $("#interface-scale").value,
    notificationDuration: $("#interface-notification-duration").value,
  });
}

function setInterfaceSettingsOpen(open) {
  const overlay = $("#interface-settings-overlay");
  if (open) {
    interfaceSettingsBeforeOpen = { ...interfaceSettings };
    updateInterfaceSettingsFields(interfaceSettings);
    overlay.hidden = false;
    $("#interface-notifications-enabled").focus();
    return;
  }
  if (interfaceSettingsBeforeOpen) applyInterfaceSettings(interfaceSettingsBeforeOpen);
  interfaceSettingsBeforeOpen = null;
  overlay.hidden = true;
  $("#interface-settings-button").focus();
}

function playNotificationSound(success = true) {
  if (!interfaceSettings.soundEnabled) return;
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    notificationAudioContext ||= new AudioContext();
    if (notificationAudioContext.state === "suspended") notificationAudioContext.resume().catch(() => {});
    const oscillator = notificationAudioContext.createOscillator();
    const gain = notificationAudioContext.createGain();
    const now = notificationAudioContext.currentTime;
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(success ? 740 : 260, now);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.055, now + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.13);
    oscillator.connect(gain).connect(notificationAudioContext.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.14);
  } catch (_) {
    // Audio may be blocked until the first user gesture; notifications still work.
  }
}

function notify(title, copy, success = true) {
  if (!interfaceSettings.notificationsEnabled) return;
  const node = document.createElement("div");
  node.className = "notification";
  const icon = document.createElement("i");
  const content = document.createElement("div");
  const heading = document.createElement("b");
  const message = document.createElement("span");
  icon.textContent = success ? "✓" : "!";
  heading.textContent = String(title);
  message.textContent = String(copy);
  content.append(heading, message);
  node.append(icon, content);
  $("#notifications").append(node);
  playNotificationSound(success);
  setTimeout(() => node.remove(), interfaceSettings.notificationDuration * 1000);
}

function applySnapshot(data) {
  const selectedStillExists = data.agents.some((agent) => agent.id === appState.selectedAgentId && agent.seen);
  appState.markers = data.markers;
  appState.agents = data.agents;
  appState.regions = data.regions || [];
  if (appState.selectedRegionId && !appState.regions.some((region) => region.id === appState.selectedRegionId)) appState.selectedRegionId = null;
  appState.camera = data.camera || appState.camera;
  appState.robot = data.robot || appState.robot;
  if (!selectedStillExists) appState.selectedAgentId = null;
  $("#marker-count").textContent = data.markers.filter((marker) => marker.seen && !marker.manual).length;
  if ($("#markers-page").classList.contains("active")) {
    if (!appState.draftMarkers.length) {
      // The camera may be completely offline, leaving the table empty. A
      // newly created camera-less agent must still become the first row.
      appState.draftMarkers = data.markers.map((marker) => ({ ...marker }));
    } else {
      const incoming = new Map(data.markers.map((marker) => [marker.id, marker]));
      appState.draftMarkers = appState.draftMarkers.map((draft) => {
        const fresh = incoming.get(draft.id);
        return fresh ? { ...draft, x: fresh.x, y: fresh.y, seen: fresh.seen } : { ...draft, seen: false };
      });
      const draftedIds = new Set(appState.draftMarkers.map((marker) => marker.id));
      data.markers.filter((marker) => !draftedIds.has(marker.id)).forEach((marker) => appState.draftMarkers.push({ ...marker }));
    }
    renderMarkerTable();
  }
  renderAgentList();
  renderMap();
  updateHardwareStatus();
}

function selectedManualAgent() {
  return appState.agents.find((agent) => agent.id === appState.selectedAgentId && agent.manual) || null;
}

async function sendManualCommand(direction, quiet = false) {
  const agent = selectedManualAgent();
  if (!agent || !direction) return;
  if (appState.manualRequestPending) {
    appState.manualQueuedDirection = direction;
    return;
  }
  appState.manualRequestPending = true;
  try {
    const response = await fetch("/api/commands/manual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: agent.id, direction }),
      keepalive: direction === "halt",
    });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "ESP32 не приняла ручную команду"));
  } catch (error) {
    if (!quiet) notify("Ручное управление недоступно", error.message, false);
    if (direction !== "halt") stopManualDrive(false);
  } finally {
    appState.manualRequestPending = false;
    const queued = appState.manualQueuedDirection;
    appState.manualQueuedDirection = null;
    if (queued) sendManualCommand(queued, queued === "halt");
  }
}

function setManualButtonState() {
  const pressed = new Set([
    ...[...appState.manualKeys].map((code) => MANUAL_KEY_DIRECTIONS[code]).filter(Boolean),
    ...appState.manualPointers.values(),
  ]);
  $$('[data-manual-direction]').forEach((button) => button.classList.toggle("active", pressed.has(button.dataset.manualDirection)));
}

function resolveManualDirection() {
  const pressed = new Set([
    ...[...appState.manualKeys].map((code) => MANUAL_KEY_DIRECTIONS[code]).filter(Boolean),
    ...appState.manualPointers.values(),
  ]);
  const forward = pressed.has("forward") && !pressed.has("backward");
  const backward = pressed.has("backward") && !pressed.has("forward");
  const left = pressed.has("left") && !pressed.has("right");
  const right = pressed.has("right") && !pressed.has("left");
  if (forward && left) return "forward_left";
  if (forward && right) return "forward_right";
  if (backward && left) return "backward_left";
  if (backward && right) return "backward_right";
  if (forward) return "forward";
  if (backward) return "backward";
  if (left) return "left";
  if (right) return "right";
  return null;
}

function refreshManualDriveFromInputs() {
  const direction = resolveManualDirection();
  if (direction) startManualDrive(direction);
  else stopManualDrive(false);
}

function startManualDrive(direction) {
  if (!selectedManualAgent()) return;
  if (appState.robot?.phase === "EMERGENCY_STOP") {
    notify("Аварийная остановка активна", "Сначала снимите аварийную остановку", false);
    return;
  }
  if (appState.manualDirection !== direction) {
    appState.manualDirection = direction;
    setManualButtonState();
    sendManualCommand(direction);
  }
  clearInterval(appState.manualTimer);
  appState.manualTimer = setInterval(() => sendManualCommand(appState.manualDirection, true), 180);
}

function stopManualDrive(clearKeys = true) {
  clearInterval(appState.manualTimer);
  appState.manualTimer = null;
  if (clearKeys) {
    appState.manualKeys.clear();
    appState.manualPointers.clear();
  }
  const wasDriving = Boolean(appState.manualDirection);
  appState.manualDirection = null;
  setManualButtonState();
  if (wasDriving && selectedManualAgent()) sendManualCommand("halt", true);
}

function manualKeyDown(event) {
  const direction = MANUAL_KEY_DIRECTIONS[event.code];
  if (!direction || !selectedManualAgent()) return;
  if (event.target.closest("input, select, textarea, [contenteditable=true]")) return;
  event.preventDefault();
  if (event.repeat && appState.manualKeys.has(event.code)) return;
  appState.manualKeys.add(event.code);
  refreshManualDriveFromInputs();
}

function manualKeyUp(event) {
  if (!MANUAL_KEY_DIRECTIONS[event.code]) return;
  appState.manualKeys.delete(event.code);
  if (!selectedManualAgent()) return;
  event.preventDefault();
  refreshManualDriveFromInputs();
}

async function emergencyStop() {
  stopManualDrive();
  try {
    const response = await fetch("/api/commands/stop", { method: "POST" });
    if (!response.ok) throw new Error("Контроллер не подтвердил остановку");
    appState.robot = { ...(appState.robot || {}), phase: "EMERGENCY_STOP", last_command: "STOP", target: null };
    setTargetMode(false);
    updateHardwareStatus();
    renderAgentList();
    renderMap();
    notify("Робот остановлен", "Цель сброшена, отправлена команда STOP");
  } catch (error) {
    notify("Ошибка остановки", error.message, false);
  }
}

async function createManualAgent() {
  const name = $("#manual-agent-name").value.trim();
  const ipAddress = $("#manual-agent-ip").value.trim();
  const agentType = $("#manual-agent-type").value;
  if (!ipAddress) return notify("Нужен IP-адрес", "Укажите адрес ESP32 в локальной Wi-Fi сети", false);
  try {
    const response = await fetch("/api/manual-agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name || null, ip_address: ipAddress, agent_type: agentType }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Не удалось добавить робота"));
    $("#manual-agent-name").value = "";
    $("#manual-agent-ip").value = "";
    // Do not wait for the next WebSocket frame: update the empty marker table
    // immediately so the click always has visible feedback.
    await loadInitialState();
    notify("Робот без ArUco добавлен", "Он появился в списке агентов · выберите его для управления WASD");
  } catch (error) {
    notify("Ошибка добавления", error.message, false);
  }
}

async function deleteManualAgent(agentId) {
  stopManualDrive();
  try {
    const response = await fetch(`/api/manual-agents/${encodeURIComponent(agentId)}`, { method: "DELETE" });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Не удалось удалить ручного агента"));
    appState.draftMarkers = appState.draftMarkers.filter((marker) => marker.id !== agentId);
    if (appState.selectedAgentId === agentId) appState.selectedAgentId = null;
    renderMarkerTable(true);
    notify("Ручной агент удалён", "Настройки списка сохранены");
  } catch (error) {
    notify("Ошибка удаления", error.message, false);
  }
}

async function resumeAfterEmergencyStop() {
  try {
    const response = await fetch("/api/commands/resume", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Не удалось снять аварийную остановку"));
    appState.robot.phase = "READY";
    updateHardwareStatus();
    notify("Аварийная остановка снята", "Теперь выберите точку на карте");
  } catch (error) {
    notify("Не удалось продолжить", error.message, false);
  }
}

async function clearStuck() {
  try {
    const response = await fetch("/api/commands/clear-stuck", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(apiErrorMessage(result, "Не удалось снять состояние застревания"));
    appState.robot.phase = appState.robot.target ? "NAVIGATING" : "READY";
    updateHardwareStatus();
    notify("Состояние снято", "Робот снова может продолжить движение к цели");
  } catch (error) {
    notify("Не удалось снять «Застрял»", error.message, false);
  }
}

async function loadInitialState() {
  const response = await fetch("/api/state");
  if (!response.ok) throw new Error("Backend недоступен");
  applySnapshot(await response.json());
}

function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${location.host}/ws`);
  appState.ws = ws;
  ws.addEventListener("open", () => {
    $("#ws-status").textContent = "активен";
    $("#ws-status").style.color = "#d8f45b";
    clearInterval(appState.latencyTimer);
    appState.latencyTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.lastPing = performance.now();
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 3000);
  });
  ws.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "state") applySnapshot(data);
    if (data.type === "pong" && ws.lastPing) $("#latency").textContent = `${Math.round(performance.now() - ws.lastPing)} мс`;
  });
  ws.addEventListener("close", () => {
    $("#ws-status").textContent = "переподключение…";
    $("#ws-status").style.color = "#f3a36e";
    clearInterval(appState.latencyTimer);
    setTimeout(connectWebSocket, 1800);
  });
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => switchPage(button.dataset.page)));
  $("#select-target-button").addEventListener("click", () => setTargetMode(true));
  $("#close-selection").addEventListener("click", () => { stopManualDrive(); appState.selectedAgentId = null; setTargetMode(false); renderAgentList(); renderMap(); });
  $("#field-map").addEventListener("click", (event) => {
    if (!appState.choosingTarget) return;
    const point = eventToSvgPoint(event);
    if (!pointInPolygon(point, currentPolygonPoints())) return notify("Точка вне поля", "Выберите позицию внутри границы ArUco", false);
    sendMoveCommand(point);
  });
  $("#field-map").addEventListener("mousemove", (event) => {
    const point = realPoint(eventToSvgPoint(event).x, eventToSvgPoint(event).y);
    $("#coordinates").innerHTML = `X ${point.x.toFixed(1)}&nbsp;&nbsp; Y ${point.y.toFixed(1)}`;
  });
  $("#field-map").addEventListener("pointerdown", beginRegionDraw);
  $("#field-map").addEventListener("pointermove", moveRegionDraw);
  $("#field-map").addEventListener("pointerup", finishRegionDraw);
  $("#field-map").addEventListener("pointercancel", () => { appState.regionDraft = null; $("#region-preview-layer").replaceChildren(); });
  $("#zoom-in").addEventListener("click", () => setZoom(Math.min(1.3, appState.zoom + .1)));
  $("#zoom-out").addEventListener("click", () => setZoom(Math.max(.8, appState.zoom - .1)));
  $("#fit-map").addEventListener("click", () => setZoom(1));
  $("#camera-toggle").addEventListener("click", () => { appState.cameraVisible = !appState.cameraVisible; updateHardwareStatus(); });
  $("#debug-mode-toggle").addEventListener("change", (event) => setDebugMode(event.target.checked));
  $("#debug-speed").addEventListener("input", (event) => {
    appState.debugSpeed = Math.max(0, Math.min(250, Number(event.target.value)));
    $("#debug-speed-value").textContent = String(appState.debugSpeed);
  });
  $("#controller-speed").addEventListener("input", (event) => {
    $("#controller-speed-value").textContent = event.target.value;
  });
  $("#save-controller-settings").addEventListener("click", saveControllerSettings);
  $("#save-servo-settings").addEventListener("click", saveServoSettings);
  $("#controller-stuck-action").addEventListener("change", updateStuckActionUi);
  $("#stuck-boost-toggle").addEventListener("click", async () => {
    $("#controller-stuck-action").value = $("#controller-stuck-action").value === "boost" ? "stop" : "boost";
    updateStuckActionUi();
    await saveControllerSettings();
  });
  $("#emergency-stop").addEventListener("click", emergencyStop);
  $("#resume-stop").addEventListener("click", resumeAfterEmergencyStop);
  $("#clear-stuck").addEventListener("click", clearStuck);
  $$("[data-region-type]").forEach((button) => button.addEventListener("click", () => setRegionMode(appState.activeRegionType === button.dataset.regionType ? null : button.dataset.regionType)));
  $("#region-mode-close").addEventListener("click", () => setRegionMode(null));
  $("#save-config").addEventListener("click", saveConfiguration);
  $("#add-manual-agent").addEventListener("click", createManualAgent);
  $("#reset-config").addEventListener("click", () => { appState.draftMarkers = appState.markers.map((marker) => ({ ...marker })); renderMarkerTable(); notify("Изменения сброшены", "Возвращена сохранённая конфигурация"); });
  $("#sound-button").addEventListener("click", () => {
    interfaceSettings.soundEnabled = !interfaceSettings.soundEnabled;
    persistInterfaceSettings();
    applyInterfaceSettings();
    notify("Уведомления", interfaceSettings.soundEnabled ? "Звук включён" : "Звук выключен");
  });
  $("#interface-settings-button").addEventListener("click", () => setInterfaceSettingsOpen(true));
  $("#interface-settings-close").addEventListener("click", () => setInterfaceSettingsOpen(false));
  $("#interface-settings-overlay").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) setInterfaceSettingsOpen(false);
  });
  $("#interface-scale").addEventListener("input", (event) => {
    const scale = clampNumber(event.target.value, 80, 120, DEFAULT_INTERFACE_SETTINGS.scale);
    $("#interface-scale-value").textContent = `${scale}%`;
    applyInterfaceSettings({ ...readInterfaceSettingsFields(), scale });
  });
  $("#interface-settings-reset").addEventListener("click", () => {
    updateInterfaceSettingsFields(DEFAULT_INTERFACE_SETTINGS);
    applyInterfaceSettings(DEFAULT_INTERFACE_SETTINGS);
  });
  $("#interface-settings-save").addEventListener("click", () => {
    interfaceSettings = readInterfaceSettingsFields();
    persistInterfaceSettings();
    interfaceSettingsBeforeOpen = null;
    $("#interface-settings-overlay").hidden = true;
    applyInterfaceSettings();
    notify("Настройки сохранены", "Параметры интерфейса применены");
  });
  $$('[data-manual-direction]').forEach((button) => {
    button.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      appState.manualPointers.set(event.pointerId, button.dataset.manualDirection);
      refreshManualDriveFromInputs();
    });
    const releasePointer = (event) => {
      if (!appState.manualPointers.delete(event.pointerId)) return;
      refreshManualDriveFromInputs();
    };
    button.addEventListener("pointerup", releasePointer);
    button.addEventListener("pointercancel", releasePointer);
  });
  window.addEventListener("keydown", (event) => { manualKeyDown(event); if (event.key === "Escape") { stopManualDrive(); setTargetMode(false); setRegionMode(null); if (!$("#interface-settings-overlay").hidden) setInterfaceSettingsOpen(false); } });
  window.addEventListener("keyup", manualKeyUp);
  window.addEventListener("blur", () => stopManualDrive());
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopManualDrive(); });
}

function setZoom(value) {
  appState.zoom = value;
  $("#field-map").style.transform = `scale(${value})`;
  $("#zoom-value").textContent = `${Math.round(value * 100)}%`;
}

async function init() {
  loadInterfaceSettings();
  applyInterfaceSettings();
  updateClock();
  setInterval(updateClock, 1000);
  bindEvents();
  try {
    await loadInitialState();
    const requestedPage = new URLSearchParams(location.search).get("page");
    if (["markers", "stereo"].includes(requestedPage)) switchPage(requestedPage);
    connectWebSocket();
  } catch (error) {
    $("#ws-status").textContent = "недоступен";
    notify("Нет связи с сервером", "Запустите backend по инструкции в README", false);
  }
}

init();

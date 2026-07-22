(() => {
  "use strict";

  const canvas = document.getElementById("stereo-canvas");
  if (!canvas) return;
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  const wait = document.getElementById("stereo-wait");
  const waitKind = wait.querySelector("span");
  const waitTitle = wait.querySelector("b");
  const errorCopy = document.getElementById("stereo-error");
  const statusNode = document.getElementById("stereo-status");
  const fpsNode = document.getElementById("stereo-fps");
  const roiNode = document.getElementById("stereo-roi");
  const helpNode = document.querySelector(".stereo-help");
  const navDot = document.getElementById("stereo-nav-dot");
  const rgbButton = document.getElementById("stereo-rgb");
  const topologyButton = document.getElementById("stereo-topology");
  const textureSwitch = document.getElementById("stereo-texture-switch");
  const frame2d = document.getElementById("stereo-2d-frame");
  const resetButton = document.getElementById("stereo-reset");
  const modeButtons = [...document.querySelectorAll("[data-stereo-mode]")];
  const settingsPanel = document.getElementById("stereo-settings-panel");
  const settingInputs = [...document.querySelectorAll("[data-stereo-setting]")];
  const viewKey = "kariera.stereo.view.v2";
  let savedView = {};
  try { savedView = JSON.parse(localStorage.getItem(viewKey) || "{}"); } catch (_) { savedView = {}; }

  const state = {
    active: false,
    online: false,
    loading: false,
    sequence: -1,
    indexCount: 0,
    yaw: Number.isFinite(savedView.yaw) ? savedView.yaw : 0.62,
    pitch: Number.isFinite(savedView.pitch) ? savedView.pitch : -0.82,
    zoom: Number.isFinite(savedView.zoom) ? savedView.zoom : 1,
    sceneTransform: savedView.sceneTransform || null,
    textureMode: savedView.textureMode === "topology" ? "topology" : "rgb",
    displayMode: ["3d_aruco", "3d_full", "rgbd", "superfast"].includes(savedView.displayMode) ? savedView.displayMode : "3d_aruco",
    dragging: false,
    pointerX: 0,
    pointerY: 0,
    frameUrl: null,
  };

  function saveView() {
    localStorage.setItem(viewKey, JSON.stringify({
      yaw: state.yaw,
      pitch: state.pitch,
      zoom: state.zoom,
      sceneTransform: state.sceneTransform,
      textureMode: state.textureMode,
      displayMode: state.displayMode,
    }));
  }

  if (!gl) {
    errorCopy.textContent = "WebGL2 недоступен в этом браузере";
    window.stereoViewer = { setActive() {} };
    return;
  }

  const vertexSource = `#version 300 es
    in vec3 aPosition;
    in vec2 aUv;
    uniform float uYaw;
    uniform float uPitch;
    uniform float uZoom;
    uniform float uAspect;
    out vec2 vUv;
    void main() {
      float cy = cos(uYaw), sy = sin(uYaw);
      float cx = cos(uPitch), sx = sin(uPitch);
      vec3 yRot = vec3(cy * aPosition.x + sy * aPosition.z, aPosition.y, -sy * aPosition.x + cy * aPosition.z);
      vec3 p = vec3(yRot.x, cx * yRot.y - sx * yRot.z, sx * yRot.y + cx * yRot.z);
      float nearPlane = 0.1;
      float farPlane = 100.0;
      float cameraDistance = 3.1 / uZoom;
      vec3 view = vec3(p.xy, p.z - cameraDistance);
      float f = 1.85;
      float zClip = ((farPlane + nearPlane) / (nearPlane - farPlane)) * view.z
                  + (2.0 * farPlane * nearPlane / (nearPlane - farPlane));
      gl_Position = vec4(f * view.x / uAspect, f * view.y, zClip, -view.z);
      vUv = aUv;
    }`;
  const fragmentSource = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D uTexture;
    out vec4 outColor;
    void main() {
      outColor = vec4(texture(uTexture, vUv).rgb, 1.0);
    }`;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const vao = gl.createVertexArray();
  const positionBuffer = gl.createBuffer();
  const uvBuffer = gl.createBuffer();
  const indexBuffer = gl.createBuffer();
  const texture = gl.createTexture();
  const locations = {
    position: gl.getAttribLocation(program, "aPosition"),
    uv: gl.getAttribLocation(program, "aUv"),
    yaw: gl.getUniformLocation(program, "uYaw"),
    pitch: gl.getUniformLocation(program, "uPitch"),
    zoom: gl.getUniformLocation(program, "uZoom"),
    aspect: gl.getUniformLocation(program, "uAspect"),
  };

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(locations.position);
  gl.vertexAttribPointer(locations.position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
  gl.enableVertexAttribArray(locations.uv);
  gl.vertexAttribPointer(locations.uv, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([65, 76, 67]));
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  gl.clearColor(0.06, 0.08, 0.065, 1);

  function resetView() {
    state.yaw = 0.62;
    state.pitch = -0.82;
    state.zoom = 1;
    saveView();
  }

  function resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
  }

  function percentile(values, ratio) {
    if (!values.length) return 0;
    values.sort((a, b) => a - b);
    return values[Math.min(values.length - 1, Math.floor(values.length * ratio))];
  }

  function createStableTransform(source) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const zValues = [];
    for (let index = 0; index < source.length; index += 3) {
      const x = source[index], y = source[index + 1], z = source[index + 2];
      if (!Number.isFinite(x + y + z)) continue;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      if (index % 24 === 0) zValues.push(z);
    }
    return {
      center: [(minX + maxX) * 0.5, (minY + maxY) * 0.5, percentile(zValues, 0.5)],
      scale: Math.max(1, maxX - minX, maxY - minY),
    };
  }

  function normalizePositions(source) {
    if (!state.sceneTransform || !Number.isFinite(state.sceneTransform.scale)) {
      state.sceneTransform = createStableTransform(source);
      saveView();
    }
    const { center, scale } = state.sceneTransform;
    const output = new Float32Array(source.length);
    for (let index = 0; index < source.length; index += 3) {
      output[index] = (source[index] - center[0]) / scale * 1.8;
      output[index + 1] = -(source[index + 1] - center[1]) / scale * 1.8;
      // Stereo distance grows away from the cameras, while WebGL +Z points
      // toward the viewer. Invert it so blue/near terrain really protrudes.
      output[index + 2] = -(source[index + 2] - center[2]) / scale * 1.8;
    }
    return output;
  }

  async function loadTexture(sequence) {
    const image = new Image();
    image.decoding = "async";
    image.src = `/api/stereo/texture.jpg?mode=${state.textureMode}&seq=${sequence}`;
    await image.decode();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
  }

  async function loadMesh(sequence) {
    if (state.loading) return;
    state.loading = true;
    try {
      const crop = state.displayMode === "3d_full" ? "full" : "aruco";
      const response = await fetch(`/api/stereo/mesh.bin?crop=${crop}&seq=${sequence}`, { cache: "no-store" });
      if (!response.ok) throw new Error((await response.json()).detail || "Сетка ещё не готова");
      const data = await response.arrayBuffer();
      const header = new DataView(data, 0, 32);
      const magic = String.fromCharCode(...new Uint8Array(data, 0, 4));
      if (magic !== "ST3D" || header.getUint32(4, true) !== 1) throw new Error("Неизвестный формат 3D-сетки");
      const packetSequence = header.getUint32(8, true);
      const vertexCount = header.getUint32(12, true);
      const indexCount = header.getUint32(16, true);
      let offset = 32;
      const rawPositions = new Float32Array(data, offset, vertexCount * 3);
      offset += vertexCount * 3 * 4;
      const uv = new Float32Array(data, offset, vertexCount * 2);
      offset += vertexCount * 2 * 4;
      const indices = new Uint32Array(data, offset, indexCount);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, normalizePositions(rawPositions), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, uv, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
      state.indexCount = indexCount;
      state.sequence = packetSequence;
      await loadTexture(packetSequence);
      wait.classList.add("hidden");
    } catch (error) {
      errorCopy.textContent = error.message;
      wait.classList.remove("hidden");
    } finally {
      state.loading = false;
    }
  }

  async function load2dFrame(sequence) {
    if (state.loading) return;
    state.loading = true;
    try {
      const response = await fetch(`/api/stereo/frame.jpg?seq=${sequence}`, { cache: "no-store" });
      if (!response.ok) throw new Error((await response.json()).detail || "RGBD-кадр ещё не готов");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      await new Promise((resolve, reject) => {
        frame2d.onload = resolve;
        frame2d.onerror = () => reject(new Error("Не удалось показать RGBD-кадр"));
        frame2d.src = url;
      });
      if (state.frameUrl) URL.revokeObjectURL(state.frameUrl);
      state.frameUrl = url;
      state.sequence = sequence;
      wait.classList.add("hidden");
    } catch (error) {
      errorCopy.textContent = error.message;
      wait.classList.remove("hidden");
    } finally {
      state.loading = false;
    }
  }

  async function refreshStatus() {
    try {
      const response = await fetch("/api/stereo/status", { cache: "no-store" });
      const status = await response.json();
      state.online = Boolean(status.online);
      statusNode.classList.toggle("online", state.online);
      navDot.classList.toggle("online", state.online);
      statusNode.querySelector("b").textContent = state.online ? "СТЕРЕО АКТИВНО" : "ОЖИДАНИЕ КАМЕР";
      fpsNode.textContent = state.online ? `${Number(status.fps || 0).toFixed(1)} FPS` : "—";
      roiNode.textContent = state.displayMode === "3d_full"
        ? "ARUCO ROI: ОТКЛЮЧЁН"
        : `ARUCO ROI: ${status.roi_live ? "АКТИВЕН" : "ОЖИДАНИЕ"}`;
      roiNode.classList.toggle("live", Boolean(status.roi_live) || state.displayMode === "3d_full");
      if (!state.online) {
        errorCopy.textContent = status.error || "Подключите две стереокамеры";
        if (!state.indexCount && !frame2d.src) wait.classList.remove("hidden");
      }
      if (!state.active || status.mode !== state.displayMode) return;
      const sequence = Number(status.sequence);
      if (state.displayMode.startsWith("3d_") && status.has_mesh && sequence !== state.sequence) {
        await loadMesh(sequence);
      } else if (!state.displayMode.startsWith("3d_") && status.has_frame && sequence !== state.sequence) {
        await load2dFrame(sequence);
      }
    } catch (error) {
      state.online = false;
      statusNode.classList.remove("online");
      navDot.classList.remove("online");
      errorCopy.textContent = error.message;
    }
  }

  function render() {
    if (state.active && state.displayMode.startsWith("3d_")) {
      resize();
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (state.indexCount) {
        gl.useProgram(program);
        gl.bindVertexArray(vao);
        gl.uniform1f(locations.yaw, state.yaw);
        gl.uniform1f(locations.pitch, state.pitch);
        gl.uniform1f(locations.zoom, state.zoom);
        gl.uniform1f(locations.aspect, canvas.width / Math.max(1, canvas.height));
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.drawElements(gl.TRIANGLES, state.indexCount, gl.UNSIGNED_INT, 0);
      }
    }
    requestAnimationFrame(render);
  }

  function applyDisplayMode() {
    const is3d = state.displayMode.startsWith("3d_");
    canvas.classList.toggle("hidden", !is3d);
    frame2d.classList.toggle("hidden", is3d);
    textureSwitch.classList.toggle("hidden", !is3d);
    helpNode.classList.toggle("hidden", !is3d);
    roiNode.classList.toggle("hidden", !is3d);
    modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.stereoMode === state.displayMode));
    waitKind.textContent = is3d ? "3D" : (state.displayMode === "superfast" ? "FAST" : "2D");
    waitTitle.textContent = is3d ? "ПОСТРОЕНИЕ СТЕРЕОКАРТЫ" : "ПОСТРОЕНИЕ RGBD-КАДРА";
  }

  async function setDisplayMode(mode, force = false) {
    if (!force && state.displayMode === mode) return;
    state.displayMode = mode;
    state.sequence = -1;
    state.loading = false;
    applyDisplayMode();
    saveView();
    wait.classList.remove("hidden");
    errorCopy.textContent = "Переключение режима…";
    try {
      const response = await fetch("/api/stereo/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "Не удалось переключить режим");
      await refreshStatus();
    } catch (error) {
      errorCopy.textContent = error.message;
    }
  }

  function setTextureMode(mode) {
    state.textureMode = mode;
    rgbButton.classList.toggle("active", mode === "rgb");
    topologyButton.classList.toggle("active", mode === "topology");
    saveView();
    if (state.sequence >= 0) loadTexture(state.sequence).catch((error) => { errorCopy.textContent = error.message; });
  }

  async function loadSettings() {
    const response = await fetch("/api/stereo/settings", { cache: "no-store" });
    if (!response.ok) throw new Error("Не удалось загрузить настройки RGBD");
    const settings = await response.json();
    settingInputs.forEach((input) => {
      const value = settings[input.dataset.stereoSetting];
      if (input.type === "checkbox") input.checked = Boolean(value);
      else input.value = value;
    });
  }

  function setSettingsOpen(open) {
    settingsPanel.classList.toggle("open", open);
    settingsPanel.setAttribute("aria-hidden", String(!open));
    if (open) loadSettings().catch((error) => { errorCopy.textContent = error.message; });
  }

  async function saveSettings() {
    const changes = {};
    settingInputs.forEach((input) => {
      changes[input.dataset.stereoSetting] = input.type === "checkbox" ? input.checked : Number(input.value);
    });
    const button = document.getElementById("stereo-settings-save");
    button.disabled = true;
    try {
      const response = await fetch("/api/stereo/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "Не удалось сохранить настройки");
      state.sequence = -1;
      saveView();
      setSettingsOpen(false);
      wait.classList.remove("hidden");
      errorCopy.textContent = "Настройки сохранены";
    } catch (error) {
      errorCopy.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  canvas.addEventListener("pointerdown", (event) => {
    state.dragging = true;
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging) return;
    state.yaw += (event.clientX - state.pointerX) * 0.008;
    state.pitch = Math.max(-1.55, Math.min(1.55, state.pitch + (event.clientY - state.pointerY) * 0.008));
    state.pointerX = event.clientX;
    state.pointerY = event.clientY;
  });
  const endDrag = () => {
    if (state.dragging) saveView();
    state.dragging = false;
    canvas.classList.remove("dragging");
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.zoom = Math.max(0.25, Math.min(4.0, state.zoom * Math.exp(-event.deltaY * 0.001)));
    saveView();
  }, { passive: false });
  canvas.addEventListener("dblclick", resetView);
  resetButton.addEventListener("click", resetView);
  rgbButton.addEventListener("click", () => setTextureMode("rgb"));
  topologyButton.addEventListener("click", () => setTextureMode("topology"));
  modeButtons.forEach((button) => button.addEventListener("click", () => setDisplayMode(button.dataset.stereoMode)));
  document.getElementById("stereo-settings-toggle").addEventListener("click", () => setSettingsOpen(true));
  document.getElementById("stereo-settings-close").addEventListener("click", () => setSettingsOpen(false));
  document.getElementById("stereo-settings-reload").addEventListener("click", () => setSettingsOpen(false));
  document.getElementById("stereo-settings-save").addEventListener("click", saveSettings);

  window.stereoViewer = {
    setActive(active) {
      state.active = Boolean(active);
      if (state.active) refreshStatus();
    },
  };
  setTextureMode(state.textureMode);
  applyDisplayMode();
  setDisplayMode(state.displayMode, true);
  setInterval(refreshStatus, 350);
  requestAnimationFrame(render);
})();

/* VMS — Vanilla JS SPA
 * Talks to the FastAPI backend (served from same origin, behind nginx SSO).
 * Views: Live (MJPEG grid), Cameras (CRUD), Events (filter + playback), People (face DB).
 */
"use strict";

/* ============================================================
 * Small helpers
 * ============================================================ */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(node.dataset, v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

let _toastTimer = null;
function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (kind ? " toast-" + kind : "");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.add("hidden"), 3500);
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString([], {
    year: "2-digit", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

// datetime-local value -> ISO string (or "" if empty)
function localToIso(v) {
  if (!v) return "";
  const d = new Date(v);
  return isNaN(d) ? "" : d.toISOString();
}

/* ============================================================
 * API layer
 * ============================================================ */
const api = {
  async req(method, path, { body, isForm } = {}) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      if (isForm) {
        opts.body = body; // FormData; browser sets content-type
      } else {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(body);
      }
    }
    const res = await fetch(path, opts);
    if (res.status === 204) return null;
    const ct = res.headers.get("content-type") || "";
    const data = ct.includes("application/json") ? await res.json().catch(() => null) : await res.text();
    if (!res.ok) {
      const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : (typeof data === "string" ? data : res.statusText);
      throw new Error(detail || `HTTP ${res.status}`);
    }
    return data;
  },
  get: (p) => api.req("GET", p),
  post: (p, body) => api.req("POST", p, { body }),
  postForm: (p, fd) => api.req("POST", p, { body: fd, isForm: true }),
  put: (p, body) => api.req("PUT", p, { body }),
  del: (p) => api.req("DELETE", p),

  // endpoints
  health: () => api.get("/health"),
  cameras: () => api.get("/api/cameras"),
  cameraStatus: (id) => api.get(`/api/cameras/${id}/status`),
  detectClasses: () => api.get("/api/cameras/detect/classes"),
  createCamera: (b) => api.post("/api/cameras", b),
  updateCamera: (id, b) => api.put(`/api/cameras/${id}`, b),
  deleteCamera: (id) => api.del(`/api/cameras/${id}`),

  events: (params) => api.get("/api/events?" + new URLSearchParams(params).toString()),
  event: (id) => api.get(`/api/events/${id}`),
  deleteEvent: (id) => api.del(`/api/events/${id}`),
  clearEvents: (body) => api.post("/api/events/clear-all", body),
  clearPeople: () => api.post("/api/people/clear-all", { confirm: true }),

  people: () => api.get("/api/people"),
  createPerson: (b) => api.post("/api/people", b),
  updatePerson: (id, b) => api.put(`/api/people/${id}`, b),
  deletePerson: (id) => api.del(`/api/people/${id}`),
  personFaces: (id) => api.get(`/api/people/${id}/faces`),
  enrollFace: (id, fd) => api.postForm(`/api/people/${id}/faces`, fd),
  deleteFace: (id, fid) => api.del(`/api/people/${id}/faces/${fid}`),
};

/* ============================================================
 * Modal
 * ============================================================ */
const modal = {
  open(title, bodyNode, { wide = false } = {}) {
    $("#modal-title").textContent = title;
    const body = $("#modal-body");
    body.innerHTML = "";
    body.append(bodyNode);
    $(".modal").classList.toggle("modal-wide", wide);
    $("#modal-backdrop").classList.remove("hidden");
  },
  close() {
    if (typeof this._onClose === "function") {
      try { this._onClose(); } catch (_) {}
      this._onClose = null;
    }
    $("#modal-backdrop").classList.add("hidden");
    $("#modal-body").innerHTML = "";
  },
};

/* ============================================================
 * Navigation / view switching
 * ============================================================ */
const views = {
  live: { render: renderLive, teardown: teardownLive },
  cameras: { render: renderCameras },
  events: { render: renderEvents },
  // Identities view render/teardown live in identities.js (loaded after this
  // file). Indirected through window.Identities so app.js stays self-contained.
  identities: {
    render: () => (window.Identities ? window.Identities.render() : Promise.resolve()),
    teardown: () => { if (window.Identities && window.Identities.teardown) window.Identities.teardown(); },
  },
  faces: {
    render: () => (window.Faces ? window.Faces.render() : Promise.resolve()),
    teardown: () => { if (window.Faces && window.Faces.teardown) window.Faces.teardown(); },
  },
  people: {
    render: () => (window.People ? window.People.render() : Promise.resolve()),
    teardown: () => { if (window.People && window.People.teardown) window.People.teardown(); },
  },
};
let currentView = null;

function switchView(name) {
  if (name === currentView) return;
  if (currentView && views[currentView] && views[currentView].teardown) {
    views[currentView].teardown();
  }
  currentView = name;
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  views[name].render().catch((e) => toast(e.message, "error"));
}

/* ============================================================
 * LIVE
 * ============================================================ */
let liveStreams = []; // <img> nodes with active MJPEG src

function teardownLive() {
  // Stop MJPEG connections so the browser drops the multipart streams.
  liveStreams.forEach((img) => { img.src = ""; });
  liveStreams = [];
}

async function renderLive() {
  const grid = $("#live-grid");
  teardownLive();
  let cams;
  try {
    cams = await api.cameras();
  } catch (e) {
    grid.innerHTML = "";
    grid.append(el("div", { class: "empty" }, "Failed to load cameras: " + e.message));
    return;
  }
  grid.innerHTML = "";
  const enabled = cams.filter((c) => c.enabled);
  if (!enabled.length) {
    grid.append(el("div", { class: "empty" }, "No enabled cameras. Add one in the Cameras tab."));
    return;
  }
  for (const cam of enabled) {
    const online = cam.status === "online";
    const frame = el("div", { class: "frame" });
    if (online) {
      const ts = Date.now();
      // Grid tiles stream at a LOW fps (many at once) to stay smooth; the
      // focused monitor opens at full rate. This is the main anti-lag win.
      const img = el("img", {
        src: `/api/live/${cam.id}/stream?fps=4&t=${ts}`,
        alt: cam.name,
        onerror: function () {
          // fall back to snapshot polling if stream errors
          this.onerror = null;
          this.src = `/api/live/${cam.id}/snapshot?t=${Date.now()}`;
        },
      });
      liveStreams.push(img);
      frame.append(img);
    } else {
      frame.append(el("div", { class: "offline-overlay" }, cam.status === "error" ? "Camera error" : "Offline"));
    }
    const cell = el("div", { class: "live-cell" + (online ? " expandable" : "") },
      frame,
      el("div", { class: "meta" },
        el("span", { class: "name" }, cam.name),
        statusBadge(cam.status)
      )
    );
    if (online) {
      frame.title = "Click to monitor (fullscreen · zoom · record)";
      cell.addEventListener("click", () => openLiveMonitor(cam));
    }
    grid.append(cell);
  }
}

/* Click-to-expand LIVE WITH SOUND via on-demand HLS (MJPEG carries no audio).
 * The tile click is a user gesture, so unmuted autoplay is allowed. Falls back
 * to the silent MJPEG stream if HLS is unavailable/at-capacity. */
function openLivePlayer(cam) {
  const id = cam.id;
  const stage = el("div", { class: "video-stage" });
  const video = el("video", { class: "event-video", controls: "", autoplay: "", playsinline: "" });
  video.muted = false; video.volume = 1.0;
  stage.append(video);
  const note = el("div", { class: "vv-hint" }, "Live with sound (HLS, ~6–12s behind). Click ⛶ in controls for fullscreen.");
  const wrap = el("div", { class: "video-viewer" }, stage, note);

  let hls = null;
  const m3u8 = `/api/live/${id}/hls/index.m3u8`;

  const toMjpeg = (msg) => {
    if (hls) { try { hls.destroy(); } catch (_) {} hls = null; }
    stage.innerHTML = "";
    const img = el("img", { src: `/api/live/${id}/stream?t=${Date.now()}`, class: "event-video", alt: cam.name });
    stage.append(img);
    note.textContent = "Live (no sound) — " + (msg || "HLS unavailable") + ". MJPEG fallback.";
  };

  // Pre-warm the playlist (the endpoint starts ffmpeg and waits ~4s), retrying
  // a couple of times through the startup window, then attach hls.js.
  const startHls = async () => {
    let ready = false;
    for (let i = 0; i < 4 && !ready; i++) {
      try {
        const r = await fetch(m3u8, { cache: "no-store" });
        if (r.ok) { ready = true; break; }
      } catch (_) {}
      await new Promise((res) => setTimeout(res, 1200));
    }
    if (!ready) { toMjpeg("starting timed out"); return; }
    if (window.Hls && window.Hls.isSupported()) {
      hls = new window.Hls({ lowLatencyMode: false, liveSyncDurationCount: 3 });
      hls.loadSource(m3u8);
      hls.attachMedia(video);
      hls.on(window.Hls.Events.MANIFEST_PARSED, () => { const p = video.play(); if (p && p.catch) p.catch(() => {}); });
      hls.on(window.Hls.Events.ERROR, (_e, data) => { if (data && data.fatal) toMjpeg("stream error"); });
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = m3u8;  // Safari native HLS
      const p = video.play(); if (p && p.catch) p.catch(() => {});
    } else {
      toMjpeg("no HLS support");
    }
  };

  // Teardown when the modal closes (X / backdrop / Escape): stop the player and
  // tell the server to stop the ffmpeg session promptly.
  modal._onClose = () => {
    if (hls) { try { hls.destroy(); } catch (_) {} hls = null; }
    try { video.pause(); video.removeAttribute("src"); video.load(); } catch (_) {}
    try {
      if (navigator.sendBeacon) navigator.sendBeacon(`/api/live/${id}/hls/close`);
      else fetch(`/api/live/${id}/hls/close`, { method: "POST", keepalive: true });
    } catch (_) {}
  };

  modal.open(cam.name + " — live", wrap, { wide: true });
  startHls();
}

/* Focused LIVE MONITOR — low-latency MJPEG (not HLS) with zoom/pan, fullscreen,
 * and a manual record button. This is the default tile action; sound (HLS) is a
 * secondary toggle since it is ~6–12 s behind and heavier. */
function openLiveMonitor(cam) {
  const id = cam.id;
  const img = el("img", {
    class: "event-video live-monitor-img",
    src: `/api/live/${id}/stream?t=${Date.now()}`,   // full fps for the focused view
    alt: cam.name,
    onerror: function () { this.onerror = null; this.src = `/api/live/${id}/snapshot?t=${Date.now()}`; },
  });
  const stage = el("div", { class: "video-stage" }, img);

  // Zoom + pan (same model as the clip viewer), applied to the <img>.
  let scale = 1, tx = 0, ty = 0, dragging = false, sx = 0, sy = 0;
  const MIN = 1, MAX = 6;
  const apply = () => {
    img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    stage.classList.toggle("zoomed", scale > 1);
  };
  const reset = () => { scale = 1; tx = 0; ty = 0; apply(); };
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const prev = scale;
    scale = Math.min(MAX, Math.max(MIN, scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    const k = scale / prev;
    tx = cx - (cx - tx) * k; ty = cy - (cy - ty) * k;
    if (scale === 1) { tx = 0; ty = 0; }
    apply();
  }, { passive: false });
  stage.addEventListener("mousedown", (e) => {
    if (scale <= 1) return;
    dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
    stage.classList.add("panning"); e.preventDefault();
  });
  const onMove = (e) => { if (dragging) { tx = e.clientX - sx; ty = e.clientY - sy; apply(); } };
  const onUp = () => { dragging = false; stage.classList.remove("panning"); };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  stage.addEventListener("dblclick", (e) => { e.preventDefault(); if (scale > 1) reset(); else { scale = 2; apply(); } });

  // Manual recording: server is stateless (holds nothing); we keep started_at.
  let startedAt = null, recTimer = null, recBtn;
  const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  async function startRec() {
    recBtn.disabled = true;
    try {
      const r = await fetch(`/api/live/${id}/record/start`, { method: "POST" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      startedAt = (await r.json()).started_at;
      const t0 = Date.now();
      recBtn.classList.add("recording");
      recBtn.innerHTML = "■ Stop · 0:00";
      recTimer = setInterval(() => {
        recBtn.innerHTML = "■ Stop · " + fmt(Math.floor((Date.now() - t0) / 1000));
      }, 1000);
    } catch (e) { toast("Could not start recording: " + e.message, "error"); }
    finally { recBtn.disabled = false; }
  }
  async function stopRec(silent) {
    if (recTimer) { clearInterval(recTimer); recTimer = null; }
    const sa = startedAt; startedAt = null;
    recBtn.classList.remove("recording");
    recBtn.innerHTML = "● Record";
    if (!sa) return;
    try {
      const r = await fetch(`/api/live/${id}/record/stop`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ started_at: sa }),
      });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const j = await r.json();
      if (!silent) toast(j.clip ? `Saved manual clip (${j.duration}s) → Events` : "Saved (clip pending)", "ok");
    } catch (e) { if (!silent) toast("Could not save recording: " + e.message, "error"); }
  }
  recBtn = el("button", { class: "btn btn-sm btn-rec", title: "Manual record", onclick: () => (startedAt ? stopRec(false) : startRec()) }, "● Record");

  const toolbar = el("div", { class: "video-toolbar" },
    recBtn,
    el("button", { class: "btn btn-sm", title: "Reset zoom", onclick: () => reset() }, "Reset zoom"),
    el("button", { class: "btn btn-sm", title: "Fullscreen", onclick: () => { (stage.requestFullscreen ? stage : img).requestFullscreen?.(); } }, "⛶ Fullscreen"),
    el("button", { class: "btn btn-sm", title: "Live with sound (HLS)", onclick: () => { modal.close(); openLivePlayer(cam); } }, "🔊 Sound"),
  );

  const wrap = el("div", { class: "video-viewer" }, stage, toolbar,
    el("div", { class: "vv-hint" }, "Low-latency monitor · scroll to zoom · drag to pan · double-click to toggle zoom"));

  modal._onClose = () => {
    if (startedAt) stopRec(true);            // auto-save an in-progress recording
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    try { img.src = ""; } catch (_) {}        // stop the MJPEG connection
  };
  modal.open(cam.name + " — monitor", wrap, { wide: true });
}

function statusBadge(status) {
  const cls = status === "online" ? "badge-online" : status === "error" ? "badge-error" : "badge-offline";
  const dotCls = status === "online" ? "dot-online" : status === "error" ? "dot-error" : "dot-offline";
  return el("span", { class: "badge " + cls }, el("span", { class: "dot " + dotCls }), status || "offline");
}

/* ============================================================
 * CAMERAS
 * ============================================================ */
async function renderCameras() {
  const body = $("#cameras-body");
  body.innerHTML = "";
  let cams;
  try {
    cams = await api.cameras();
  } catch (e) {
    body.append(el("tr", {}, el("td", { colspan: "7", class: "empty" }, "Error: " + e.message)));
    return;
  }
  if (!cams.length) {
    body.append(el("tr", {}, el("td", { colspan: "7", class: "empty" }, "No cameras yet.")));
    return;
  }
  for (const c of cams) {
    const row = el("tr", {},
      el("td", {}, String(c.id)),
      el("td", {}, c.name),
      el("td", { class: "url", title: c.rtsp_url }, c.rtsp_url),
      el("td", {}, c.enabled ? "Yes" : "No"),
      el("td", {}, statusBadge(c.status)),
      el("td", {}, fmtTime(c.last_seen)),
      el("td", {}, el("div", { class: "row-actions" },
        el("button", { class: "btn btn-sm", onclick: () => openCameraForm(c) }, "Edit"),
        el("button", { class: "btn btn-sm btn-danger", onclick: () => confirmDeleteCamera(c) }, "Delete")
      ))
    );
    body.append(row);
  }
}

// COCO preset groups for the object-trigger picker.
const CLASS_PRESETS = {
  vehicles: ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
  animals: ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
};
let _detectClassesCache = null;
async function loadDetectClasses() {
  if (_detectClassesCache) return _detectClassesCache;
  try {
    _detectClassesCache = await api.detectClasses();
  } catch (e) {
    _detectClassesCache = { classes: ["person"], default: ["person"] };
  }
  return _detectClassesCache;
}

// Build the object-trigger multi-select (search + presets + checkbox grid).
function buildClassPicker(selectedSet, allClasses) {
  const grid = el("div", { class: "class-grid" });
  const checks = new Map();
  for (const name of allClasses) {
    const cb = el("input", { type: "checkbox", value: name, ...(selectedSet.has(name) ? { checked: "" } : {}) });
    checks.set(name, cb);
    grid.append(el("label", { class: "class-chip", dataset: { name } }, cb, el("span", {}, name)));
  }
  const count = el("span", { class: "class-count" });
  const refreshCount = () => {
    const n = Array.from(checks.values()).filter((c) => c.checked).length;
    count.textContent = `${n} selected`;
  };
  grid.addEventListener("change", refreshCount);
  refreshCount();

  const search = el("input", { type: "text", class: "class-search", placeholder: "Filter objects… (e.g. car, dog)" });
  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    $$(".class-chip", grid).forEach((chip) => {
      chip.style.display = !q || chip.dataset.name.includes(q) ? "" : "none";
    });
  });

  const setAll = (names, on) => {
    const target = names || Array.from(checks.keys());
    target.forEach((n) => { const c = checks.get(n); if (c) c.checked = on; });
    refreshCount();
  };
  const presetBtn = (label, fn) => el("button", { type: "button", class: "btn btn-sm", onclick: fn }, label);
  const presets = el("div", { class: "class-presets" },
    presetBtn("People only", () => { setAll(null, false); setAll(["person"], true); }),
    presetBtn("+ Vehicles", () => setAll(CLASS_PRESETS.vehicles, true)),
    presetBtn("+ Animals", () => setAll(CLASS_PRESETS.animals, true)),
    presetBtn("All", () => setAll(null, true)),
    presetBtn("Clear", () => setAll(null, false)),
  );

  const wrap = el("div", { class: "class-picker" },
    el("div", { class: "class-toolbar" }, search, count),
    presets,
    grid);
  return {
    node: wrap,
    value: () => Array.from(checks.entries()).filter(([, c]) => c.checked).map(([n]) => n),
  };
}

async function openCameraForm(cam) {
  const isEdit = !!cam;
  const c = cam || {};
  const meta = await loadDetectClasses();
  const allClasses = meta.classes || ["person"];
  const selected = new Set(
    (c.trigger_classes ? String(c.trigger_classes).split(",") : (meta.default || ["person"]))
      .map((s) => s.trim()).filter(Boolean)
  );
  const picker = buildClassPicker(selected, allClasses);

  const form = el("form", { class: "form-grid" });
  const opt = (v) => (v ?? "");
  form.append(
    el("label", {}, "Name",
      el("input", { type: "text", name: "name", required: "", value: c.name || "" })),
    el("label", {}, "RTSP URL",
      el("input", { type: "text", name: "rtsp_url", required: "", placeholder: "rtsp://user:pass@host:554/stream", value: c.rtsp_url || "" })),
    el("label", { class: "checkbox-row" },
      el("input", { type: "checkbox", name: "enabled", ...(c.enabled === false ? {} : { checked: "" }) }),
      "Enabled"),
    el("div", { class: "field-block" },
      el("div", { class: "field-title" }, "Objects that trigger recording"),
      el("div", { class: "hint" }, "Pick which detected objects start a recording. Default: people only."),
      picker.node),
  );

  // --- Advanced detection / recording tunables (collapsible) ---
  const adv = el("details", { class: "advanced" });
  adv.append(el("summary", {}, "Advanced detection & recording"));
  const advGrid = el("div", { class: "form-grid" });
  const numField = (label, name, ph, val) =>
    el("label", {}, label, el("input", { type: "text", name, placeholder: ph, value: opt(val) }));
  advGrid.append(
    numField("Detection confidence (0–1)", "detect_conf", "global default", c.detect_conf),
    numField("NMS IoU (0–1)", "detect_iou", "global default", c.detect_iou),
    numField("Input size px (dynamic models)", "detect_imgsz", "global default", c.detect_imgsz),
    numField("Detect interval (s, 0=every frame)", "detect_interval", "global default", c.detect_interval),
    numField("Trigger cooldown (s)", "trigger_cooldown", "global default", c.trigger_cooldown),
    numField("Min consecutive frames", "min_trigger_frames", "global default", c.min_trigger_frames),
    numField("Pre-roll seconds", "pre_seconds", "global default", c.pre_seconds),
    numField("Post-roll seconds", "post_seconds", "global default", c.post_seconds),
    el("label", {}, "RTSP transport",
      el("select", { name: "rtsp_transport" },
        el("option", { value: "", ...(c.rtsp_transport ? {} : { selected: "" }) }, "global default"),
        el("option", { value: "tcp", ...(c.rtsp_transport === "tcp" ? { selected: "" } : {}) }, "TCP"),
        el("option", { value: "udp", ...(c.rtsp_transport === "udp" ? { selected: "" } : {}) }, "UDP"))),
    el("label", { class: "checkbox-row" },
      el("input", { type: "checkbox", name: "faces_enabled", ...(c.faces_enabled === false ? {} : { checked: "" }) }),
      "Face recognition"),
    el("label", { class: "checkbox-row" },
      el("input", { type: "checkbox", name: "reid_enabled", ...(c.reid_enabled === false ? {} : { checked: "" }) }),
      "Cross-camera Re-ID"),
  );
  adv.append(advGrid);
  form.append(adv);

  form.append(
    el("div", { class: "form-actions" },
      el("button", { type: "button", class: "btn", onclick: () => modal.close() }, "Cancel"),
      el("button", { type: "submit", class: "btn btn-primary" }, isEdit ? "Save" : "Create"))
  );

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const numOrNull = (v) => (v === "" || v == null ? null : Number(v));
    const strOrNull = (v) => (v === "" || v == null ? null : String(v));
    const classes = picker.value();
    const payload = {
      name: fd.get("name").trim(),
      rtsp_url: fd.get("rtsp_url").trim(),
      enabled: fd.get("enabled") === "on",
      trigger_classes: classes.length ? classes.join(",") : null,
      detect_conf: numOrNull(fd.get("detect_conf")),
      detect_iou: numOrNull(fd.get("detect_iou")),
      detect_imgsz: numOrNull(fd.get("detect_imgsz")),
      detect_interval: numOrNull(fd.get("detect_interval")),
      trigger_cooldown: numOrNull(fd.get("trigger_cooldown")),
      min_trigger_frames: numOrNull(fd.get("min_trigger_frames")),
      pre_seconds: numOrNull(fd.get("pre_seconds")),
      post_seconds: numOrNull(fd.get("post_seconds")),
      rtsp_transport: strOrNull(fd.get("rtsp_transport")),
      faces_enabled: fd.get("faces_enabled") === "on",
      reid_enabled: fd.get("reid_enabled") === "on",
    };
    try {
      if (isEdit) {
        await api.updateCamera(c.id, payload);
        toast("Camera updated", "ok");
      } else {
        await api.createCamera(payload);
        toast("Camera created", "ok");
      }
      modal.close();
      renderCameras();
    } catch (e) {
      toast(e.message, "error");
    }
  });

  modal.open(isEdit ? `Edit camera #${c.id}` : "Add camera", form, { wide: true });
}

async function confirmDeleteCamera(c) {
  if (!confirm(`Delete camera "${c.name}"? The worker stops; existing events are kept.`)) return;
  try {
    await api.deleteCamera(c.id);
    toast("Camera deleted", "ok");
    renderCameras();
  } catch (e) {
    toast(e.message, "error");
  }
}

/* ============================================================
 * EVENTS
 * ============================================================ */
const eventsState = { limit: 24, offset: 0, total: 0, params: {} };

async function populateFilters() {
  const camSel = $("#filter-camera");
  const perSel = $("#filter-person");
  // only repopulate once (keep selection across refreshes is fine to re-do)
  try {
    const [cams, people] = await Promise.all([api.cameras(), api.people()]);
    const camVal = camSel.value, perVal = perSel.value;
    camSel.innerHTML = '<option value="">All</option>';
    cams.forEach((c) => camSel.append(el("option", { value: c.id }, `${c.name} (#${c.id})`)));
    camSel.value = camVal;
    perSel.innerHTML = '<option value="">All</option>';
    people.forEach((p) => perSel.append(el("option", { value: p.id }, p.name)));
    perSel.value = perVal;
  } catch (e) {
    /* filters are best-effort */
  }
}

function readEventFilters() {
  const params = {};
  const cam = $("#filter-camera").value;
  const per = $("#filter-person").value;
  const from = localToIso($("#filter-from").value);
  const to = localToIso($("#filter-to").value);
  const label = $("#filter-label").value.trim();
  if (cam) params.camera_id = cam;
  if (per) params.person_id = per;
  if (from) params.from = from;
  if (to) params.to = to;
  if (label) params.label = label;
  return params;
}

async function renderEvents() {
  await populateFilters();
  await loadEvents();
}

async function loadEvents() {
  const list = $("#events-list");
  list.innerHTML = "";
  list.append(el("div", { class: "empty" }, "Loading…"));
  const params = { ...eventsState.params, limit: eventsState.limit, offset: eventsState.offset };
  let resp;
  try {
    resp = await api.events(params);
  } catch (e) {
    list.innerHTML = "";
    list.append(el("div", { class: "empty" }, "Error: " + e.message));
    return;
  }
  eventsState.total = resp.total || 0;
  const items = resp.items || [];
  list.innerHTML = "";
  if (!items.length) {
    list.append(el("div", { class: "empty" }, "No events match the filter."));
  } else {
    for (const ev of items) list.append(eventCard(ev));
  }
  // meta + pager
  $("#events-total").textContent = `${eventsState.total} event${eventsState.total === 1 ? "" : "s"}`;
  const start = eventsState.total ? eventsState.offset + 1 : 0;
  const end = Math.min(eventsState.offset + eventsState.limit, eventsState.total);
  $("#events-page").textContent = eventsState.total ? `${start}–${end} of ${eventsState.total}` : "";
  $("#events-prev").disabled = eventsState.offset <= 0;
  $("#events-next").disabled = end >= eventsState.total;
}

function eventCard(ev) {
  const thumb = ev.thumb_url
    ? el("img", { src: ev.thumb_url, alt: "thumbnail", loading: "lazy" })
    : el("div", { class: "no-thumb" }, "no thumbnail");
  const personLine = ev.person_name
    ? el("span", {}, "Match: ", el("span", { class: "name" }, ev.person_name),
        ev.match_score != null ? el("span", { class: "unknown" }, ` (${(ev.match_score * 100).toFixed(0)}%)`) : null)
    : el("span", { class: "unknown" }, "No face match");

  const thumbBox = el("div", { class: "thumb" }, thumb);

  // Hover-to-preview: while the mouse is over the card, swap the still
  // thumbnail for a muted, looping <video> of the recorded motion clip so the
  // user sees what happened without opening the event. The preview is muted
  // (browsers block autoplay-with-sound without a click); sound plays in the
  // detail modal, which opens on click — a user gesture that permits audio.
  if (ev.clip_url) {
    let preview = null;
    const startPreview = () => {
      if (preview) return;
      preview = el("video", {
        class: "preview",
        src: ev.clip_url,
        muted: "", loop: "", autoplay: "", playsinline: "", preload: "metadata",
      });
      preview.muted = true; // attribute alone is unreliable; force the property
      thumbBox.append(preview);
      const p = preview.play();
      if (p && p.catch) p.catch(() => {});
    };
    const stopPreview = () => {
      if (!preview) return;
      try { preview.pause(); } catch (_) {}
      preview.removeAttribute("src");
      preview.load();
      preview.remove();
      preview = null;
    };
    thumbBox.addEventListener("mouseenter", startPreview);
    thumbBox.addEventListener("mouseleave", stopPreview);
  }

  return el("div", { class: "event-card", onclick: () => openEventDetail(ev.id) },
    thumbBox,
    el("div", { class: "info" },
      el("div", { class: "line1" },
        el("span", { class: "cam" }, ev.camera_name || `Camera #${ev.camera_id}`),
        el("span", { class: "when" }, fmtTime(ev.ts))),
      el("div", { class: "person" },
        ev.label ? el("span", { class: "label-badge" }, ev.label) : null,
        " ", personLine))
  );
}

/* Rich clip viewer: HTML5 controls + sound, scroll-to-zoom, drag-to-pan,
 * playback-speed buttons, frame-step, and fullscreen. Returns the viewer node.
 * Autoplays with sound (opening the modal is a user gesture, so audio is OK). */
function buildVideoViewer(ev) {
  const video = el("video", {
    class: "event-video",
    controls: "", autoplay: "", preload: "auto", playsinline: "",
  }, el("source", { src: ev.clip_url, type: "video/mp4" }));
  video.muted = false;
  video.volume = 1.0;

  const stage = el("div", { class: "video-stage" }, video);

  // Zoom + pan state, applied via CSS transform on the <video>.
  let scale = 1, tx = 0, ty = 0, dragging = false, sx = 0, sy = 0;
  const MIN = 1, MAX = 6;
  const apply = () => {
    video.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    stage.classList.toggle("zoomed", scale > 1);
  };
  const reset = () => { scale = 1; tx = 0; ty = 0; apply(); };

  // Scroll to zoom toward the cursor.
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const prev = scale;
    scale = Math.min(MAX, Math.max(MIN, scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    // keep the point under the cursor stable
    const k = scale / prev;
    tx = cx - (cx - tx) * k;
    ty = cy - (cy - ty) * k;
    if (scale === 1) { tx = 0; ty = 0; }
    apply();
  }, { passive: false });

  // Drag to pan while zoomed (doesn't interfere with the controls bar).
  stage.addEventListener("mousedown", (e) => {
    if (scale <= 1) return;
    dragging = true; sx = e.clientX - tx; sy = e.clientY - ty;
    stage.classList.add("panning");
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    tx = e.clientX - sx; ty = e.clientY - sy; apply();
  });
  window.addEventListener("mouseup", () => { dragging = false; stage.classList.remove("panning"); });
  // Double-click toggles a 2x zoom / reset.
  stage.addEventListener("dblclick", (e) => {
    e.preventDefault();
    if (scale > 1) { reset(); }
    else { scale = 2; apply(); }
  });

  const speeds = [0.25, 0.5, 1, 1.5, 2];
  const speedBtns = speeds.map((r) =>
    el("button", { class: "btn btn-sm" + (r === 1 ? " active" : ""), dataset: { rate: r } },
      r === 1 ? "1×" : r + "×"));
  const speedWrap = el("div", { class: "vv-speed" }, ...speedBtns);
  speedWrap.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-rate]");
    if (!b) return;
    video.playbackRate = Number(b.dataset.rate);
    speedBtns.forEach((x) => x.classList.toggle("active", x === b));
  });

  const toolbar = el("div", { class: "video-toolbar" },
    el("span", { class: "vv-label" }, "Speed:"),
    speedWrap,
    el("button", { class: "btn btn-sm", title: "Step back 1/25s", onclick: () => { video.pause(); video.currentTime = Math.max(0, video.currentTime - 0.04); } }, "⏴ frame"),
    el("button", { class: "btn btn-sm", title: "Step forward 1/25s", onclick: () => { video.pause(); video.currentTime += 0.04; } }, "frame ⏵"),
    el("button", { class: "btn btn-sm", title: "Reset zoom", onclick: () => reset() }, "Reset zoom"),
    el("button", { class: "btn btn-sm", title: "Fullscreen", onclick: () => { (stage.requestFullscreen ? stage : video).requestFullscreen?.(); } }, "⛶ Fullscreen"),
  );

  const p = video.play && video.play();
  if (p && p.catch) p.catch(() => {});

  return el("div", { class: "video-viewer" }, stage, toolbar,
    el("div", { class: "vv-hint" }, "Scroll to zoom · drag to pan · double-click to toggle zoom"));
}

async function openEventDetail(id) {
  let ev;
  try {
    ev = await api.event(id);
  } catch (e) {
    toast(e.message, "error");
    return;
  }
  const wrap = el("div", {});
  if (ev.clip_url) {
    wrap.append(buildVideoViewer(ev));
  } else if (ev.thumb_url) {
    wrap.append(el("img", { src: ev.thumb_url, style: "width:100%;border-radius:8px;background:#000" }));
  } else {
    wrap.append(el("div", { class: "empty" }, "No clip or thumbnail available."));
  }
  const grid = el("div", { class: "detail-grid" });
  const add = (k, v) => { grid.append(el("div", { class: "k" }, k), el("div", {}, v == null || v === "" ? "—" : String(v))); };
  add("Event ID", ev.id);
  add("Camera", ev.camera_name || `#${ev.camera_id}`);
  add("Detected", ev.label || "person");
  add("Started", fmtTime(ev.ts));
  add("Ended", fmtTime(ev.end_ts));
  add("Recognized", ev.person_name || "— (no match)");
  add("Match score", ev.match_score != null ? `${(ev.match_score * 100).toFixed(1)}%` : "—");
  add("Duration", ev.duration_seconds != null ? `${ev.duration_seconds.toFixed(1)}s` : null);
  add("Objects", ev.num_objects);
  add("Classes", ev.object_classes);
  add("Frames detected", ev.num_frames);
  add("Peak confidence", ev.peak_confidence != null ? `${(ev.peak_confidence * 100).toFixed(0)}%` : null);
  // Cross-link to the auto-discovered identity (object), if any.
  if (ev.identity_id != null) {
    const whoLink = el("a", { href: "#", class: "link",
      onclick: (e) => { e.preventDefault(); if (window.Identities) window.Identities.openById(ev.identity_id); } },
      ev.identity_name || `Identity #${ev.identity_id}`);
    grid.append(el("div", { class: "k" }, "Who"), el("div", {}, whoLink));
  }
  wrap.append(grid);
  wrap.append(el("div", { class: "form-actions" },
    ev.clip_url ? el("a", { class: "btn", href: ev.clip_url, download: `event_${ev.id}.mp4` }, "Download clip") : null,
    el("button", { class: "btn btn-danger", onclick: () => confirmDeleteEvent(ev.id) }, "Delete event")
  ));
  modal.open(`Event #${ev.id}`, wrap, { wide: true });
}
// Expose for cross-script navigation (identities.js / faces.js).
window.openEventDetail = openEventDetail;

async function confirmDeleteEvent(id) {
  if (!confirm("Delete this event? The clip and thumbnail files are removed.")) return;
  try {
    await api.deleteEvent(id);
    toast("Event deleted", "ok");
    modal.close();
    loadEvents();
  } catch (e) {
    toast(e.message, "error");
  }
}

/* ============================================================
 * PEOPLE
 * ============================================================ */
async function renderPeople() {
  const list = $("#people-list");
  list.innerHTML = "";
  let people;
  try {
    people = await api.people();
  } catch (e) {
    list.append(el("div", { class: "empty" }, "Error: " + e.message));
    return;
  }
  if (!people.length) {
    list.append(el("div", { class: "empty" }, "No people enrolled. Add one to start matching faces."));
    return;
  }
  for (const p of people) list.append(personCard(p));
}

function personCard(p) {
  const card = el("div", { class: "person-card" },
    el("div", { class: "ph-head" },
      el("div", {},
        el("div", { class: "ph-name" }, p.name),
        p.notes ? el("div", { class: "ph-notes" }, p.notes) : null,
        el("div", { class: "ph-count" }, `${p.num_faces ?? 0} enrolled face${(p.num_faces ?? 0) === 1 ? "" : "s"}`)
      )
    ),
    el("div", { class: "ph-actions" },
      el("button", { class: "btn btn-sm", onclick: () => openEnrollForm(p) }, "Enroll photo"),
      el("button", { class: "btn btn-sm", onclick: () => openPersonFaces(p) }, "View faces"),
      el("button", { class: "btn btn-sm", onclick: () => openPersonForm(p) }, "Edit"),
      el("button", { class: "btn btn-sm btn-danger", onclick: () => confirmDeletePerson(p) }, "Delete")
    )
  );
  return card;
}

function openPersonForm(person) {
  const isEdit = !!person;
  const p = person || {};
  const form = el("form", { class: "form-grid" });
  form.append(
    el("label", {}, "Name",
      el("input", { type: "text", name: "name", required: "", value: p.name || "" })),
    el("label", {}, "Notes (optional)",
      el("textarea", { name: "notes", rows: "3" }, p.notes || "")),
    el("div", { class: "form-actions" },
      el("button", { type: "button", class: "btn", onclick: () => modal.close() }, "Cancel"),
      el("button", { type: "submit", class: "btn btn-primary" }, isEdit ? "Save" : "Create"))
  );
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const payload = { name: fd.get("name").trim(), notes: (fd.get("notes") || "").trim() || null };
    try {
      if (isEdit) { await api.updatePerson(p.id, payload); toast("Person updated", "ok"); }
      else { await api.createPerson(payload); toast("Person created", "ok"); }
      modal.close();
      renderPeople();
    } catch (e) { toast(e.message, "error"); }
  });
  modal.open(isEdit ? `Edit ${p.name}` : "Add person", form);
}

function openEnrollForm(p) {
  const form = el("form", { class: "form-grid" });
  const fileInput = el("input", { type: "file", name: "file", accept: "image/*", required: "" });
  const result = el("div", { class: "hint" }, "Upload a clear, front-facing photo. One face is detected and embedded.");
  form.append(
    el("label", {}, "Photo of " + p.name, fileInput),
    result,
    el("div", { class: "form-actions" },
      el("button", { type: "button", class: "btn", onclick: () => modal.close() }, "Close"),
      el("button", { type: "submit", class: "btn btn-primary" }, "Enroll"))
  );
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!fileInput.files.length) { toast("Choose an image first", "error"); return; }
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    result.textContent = "Detecting and embedding face…";
    try {
      const r = await api.enrollFace(p.id, fd);
      const n = r && r.faces_detected != null ? r.faces_detected : "?";
      result.textContent = `Enrolled. Faces detected in photo: ${n}.`;
      toast("Face enrolled", "ok");
      renderPeople();
    } catch (e) {
      result.textContent = "Failed: " + e.message;
      toast(e.message, "error");
    }
  });
  modal.open("Enroll face — " + p.name, form);
}

async function openPersonFaces(p) {
  const wrap = el("div", {});
  wrap.append(el("div", { class: "hint" }, "Loading faces…"));
  modal.open("Faces — " + p.name, wrap);
  let faces;
  try {
    faces = await api.personFaces(p.id);
  } catch (e) {
    wrap.innerHTML = "";
    wrap.append(el("div", { class: "empty" }, "Error: " + e.message));
    return;
  }
  wrap.innerHTML = "";
  if (!faces.length) {
    wrap.append(el("div", { class: "empty" }, "No faces enrolled yet."));
  } else {
    const grid = el("div", { class: "face-grid" });
    for (const f of faces) {
      const cell = el("div", { class: "face-thumb" });
      if (f.image_url) cell.append(el("img", { src: f.image_url, alt: "face", loading: "lazy" }));
      cell.append(el("button", {
        class: "del", title: "Remove this face",
        onclick: async () => {
          if (!confirm("Remove this enrolled face?")) return;
          try {
            await api.deleteFace(p.id, f.id);
            cell.remove();
            toast("Face removed", "ok");
            renderPeople();
          } catch (e) { toast(e.message, "error"); }
        },
      }, "×"));
      grid.append(cell);
    }
    wrap.append(grid);
  }
  wrap.append(el("div", { class: "form-actions" },
    el("button", { class: "btn btn-primary", onclick: () => openEnrollForm(p) }, "+ Enroll photo")));
}

async function confirmDeletePerson(p) {
  if (!confirm(`Delete "${p.name}" and all enrolled faces? Past events keep the recognized name but lose the link.`)) return;
  try {
    await api.deletePerson(p.id);
    toast("Person deleted", "ok");
    renderPeople();
  } catch (e) { toast(e.message, "error"); }
}

/* ============================================================
 * Health poller
 * ============================================================ */
async function pollHealth() {
  const dot = $("#health-dot"), text = $("#health-text"), gpu = $("#gpu-text"), user = $("#user-text");
  try {
    const h = await api.health();
    dot.className = "dot dot-ok";
    text.textContent = h.version ? "v" + h.version : "online";
    if (h.gpu && h.gpu.total_mb) {
      gpu.textContent = `GPU ${Math.round(h.gpu.used_mb || 0)}/${Math.round(h.gpu.total_mb)} MB`;
    } else {
      gpu.textContent = "";
    }
  } catch (e) {
    dot.className = "dot dot-error";
    text.textContent = "offline";
    gpu.textContent = "";
  }
  // surface the SSO user if the header exposed it via /api/system (best-effort, optional)
  if (!user.textContent) {
    api.get("/api/system").then((s) => {
      if (s && s.user) {
        const u = s.user;
        user.textContent = typeof u === "string" ? u : (u.email || u.name || "");
      }
    }).catch(() => {});
  }
}

/* ============================================================
 * Wire up
 * ============================================================ */
function init() {
  // tab nav
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));

  // modal close
  $("#modal-close").addEventListener("click", () => modal.close());
  $("#modal-backdrop").addEventListener("click", (e) => { if (e.target === $("#modal-backdrop")) modal.close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.close(); });

  // live
  $("#live-refresh").addEventListener("click", () => renderLive());

  // cameras
  $("#camera-add").addEventListener("click", () => openCameraForm(null));
  $("#cameras-refresh").addEventListener("click", () => renderCameras());

  // events
  $("#events-refresh").addEventListener("click", () => loadEvents());
  $("#events-clear").addEventListener("click", async () => {
    const f = readEventFilters();
    const scoped = f.camera_id || f.label;
    if (!confirm(scoped ? "Delete ALL events matching the current filter (camera/label), including their clip + thumbnail files? This cannot be undone."
                        : "Delete ALL events and their clip + thumbnail files? This cannot be undone.")) return;
    try {
      const body = { confirm: true };
      if (f.camera_id) body.camera_id = Number(f.camera_id);
      if (f.label) body.label = f.label;
      const r = await api.clearEvents(body);
      toast(`Deleted ${r.deleted} events`, "ok");
      eventsState.offset = 0; loadEvents();
    } catch (e) { toast(e.message, "error"); }
  });
  $("#events-filter").addEventListener("submit", (e) => {
    e.preventDefault();
    eventsState.params = readEventFilters();
    eventsState.offset = 0;
    loadEvents();
  });
  $("#filter-clear").addEventListener("click", () => {
    $("#filter-camera").value = ""; $("#filter-person").value = "";
    $("#filter-from").value = ""; $("#filter-to").value = ""; $("#filter-label").value = "";
    eventsState.params = {}; eventsState.offset = 0;
    loadEvents();
  });
  $("#events-prev").addEventListener("click", () => {
    eventsState.offset = Math.max(0, eventsState.offset - eventsState.limit);
    loadEvents();
  });
  $("#events-next").addEventListener("click", () => {
    eventsState.offset += eventsState.limit;
    loadEvents();
  });

  // identities (handlers owned by identities.js; wired there on init if present)
  if (window.Identities && window.Identities.init) window.Identities.init();
  if (window.Faces && window.Faces.init) window.Faces.init();

  // people
  $("#person-add").addEventListener("click", () => openPersonForm(null));
  $("#people-refresh").addEventListener("click", () => renderPeople());
  $("#people-clear").addEventListener("click", async () => {
    if (!confirm("Delete ALL enrolled people and their face images? This cannot be undone.")) return;
    try { const r = await api.clearPeople(); toast(`Deleted ${r.deleted} people`, "ok"); renderPeople(); }
    catch (e) { toast(e.message, "error"); }
  });

  // health
  pollHealth();
  setInterval(pollHealth, 10000);

  // initial view from hash
  const start = (location.hash || "#live").slice(1);
  switchView(views[start] ? start : "live");
}

window.addEventListener("hashchange", () => {
  const name = (location.hash || "#live").slice(1);
  if (views[name]) switchView(name);
});

document.addEventListener("DOMContentLoaded", init);

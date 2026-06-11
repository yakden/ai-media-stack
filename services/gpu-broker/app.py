"""GPU job-queue broker + live dashboard.

One worker processes jobs sequentially. Each job needs a model; the broker keeps
exactly one heavy model resident on the T4 and swaps on demand — only when the next
job needs a different model than the one currently loaded (so a batch of renders
loads the model once). When the queue drains, it restores the default duty model
(whisper STT/TTS). The dashboard polls /api/state and shows the loaded model, the
job in progress with elapsed/ETA, the pending queue with wait estimates, and a grid
of finished results. Fill the queue from the form; it processes gradually.

Single GPU + ~4 GB host RAM, no swap: this central broker is the platform's
"GPU eviction" scheduler. It owns docker start/stop for the heavy services.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import List

import httpx
import subprocess
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

RESULTS = "/opt/gpu-broker/results"
SCENES = "/opt/gpu-broker/scenes"      # saved 3D scene geometry per floor (persistent)
PROJECTS = "/opt/gpu-broker/projects"  # saved project metadata (the library, persistent)
INVENTORY = "/opt/gpu-broker/inventory"  # detailed per-photo object inventory (for acts/valuation)
for _d in (RESULTS, SCENES, PROJECTS, INVENTORY):
    os.makedirs(_d, exist_ok=True)

RENDER_URL = "http://127.0.0.1:8210"
RENDER_COMPOSE = "/home/deploy/interior-render/docker-compose.yml"
RENDER_CONTAINER = "interior-render"
FLOORPLAN_URL = "http://127.0.0.1:8204"
CUBI_URL = "http://127.0.0.1:8205"        # CubiCasa5k neural parser (CPU)
OLLAMA = "http://127.0.0.1:11434"
VLM_MODEL = "qwen2.5vl:7b"
DUTY = ["whisper-xtts-server", "avatar-muse"]   # default resident models (STT/TTS, avatar)
# heavy translation LLMs that need the WHOLE GPU — run via /api/llm, which swaps everything
# else off (whisper/avatar/render/vlm) so the model loads fully on the T4 (no CPU offload).
HEAVY_LLM_MODELS = ["translategemma:12b"]

JOB_TYPES = {"render", "furnish", "interior", "reference"}
FURNISH_SUFFIX = (", fully furnished, sofa, bed, dining table, wardrobe, rugs, plants, "
                  "lamps, decorated rooms, interior design")
ANALYZE_PROMPT = (
    "You are an expert architect analyzing this floor plan image. Return STRICT JSON only:\n"
    '{"building_type":"apartment|detached house|townhouse|office building|office space|retail|commercial|other","levels":1,"floor_name":"",'
    '"total_area_m2":null,"north":"","doors":0,"windows":0,"stairs":false,'
    '"rooms":[{"name":"<exactly as written on the plan>",'
    '"type":"kitchen|bedroom|bathroom|living room|dining room|hallway|storage|utility|guest room|walk-in closet|home office|terrace|other",'
    '"area_m2":null,"dimensions":"","windows":0,"fixtures":[],'
    '"description":"one vivid sentence describing THIS specific room: its function, how many windows and on which wall, which fixtures and where they sit, the flooring"}],'
    '"summary":"a 2-3 sentence overview of the whole project",'
    '"reasoning":"a concise analysis in 2-3 sentences: building type, the rooms and their purpose, notable fixtures and observations"}\n'
    "Rules: classify each room's English `type` correctly (POM.GOSPOD=utility, GOSCINNY=guest room, "
    "SALON/SALONIK=living room, JADALNIA=dining room, LAZIENKA=bathroom, KUCHNIA=kitchen, "
    "SYPIALNIA=bedroom, GARDEROBA=walk-in closet, HALL=hallway, TARAS=terrace). "
    "ALWAYS fill `fixtures` for each room with the symbols drawn inside (sink, stove, toilet, "
    "bathtub, bed, sofa, table, fireplace, stairs) — never leave it empty if anything is visible. "
    "Count `windows` per room from the walls. Decide building_type from the LAYOUT: many similar "
    "small rooms / cubicles / meeting rooms / open-plan desks = office building or office space; a "
    "standalone plan with a terrace and 'parter/pietro' titles = detached house; a single "
    "self-contained unit = apartment; shops or large open halls = retail/commercial. Decide levels "
    "from the titles. Be precise and exhaustive.")

PLAN_CATEGORY_PROMPT = (
    "You are an architect. Look at this plan/drawing and decide WHAT KIND of object it is. "
    "This is a high-level classification used to drive further processing. Return STRICT JSON only:\n"
    '{"category":"apartment_building_floor|single_apartment|house|multi_storey_building|office|'
    'retail_commercial|warehouse_industrial|public_civic|site_masterplan|other",'
    '"units_estimate":0,"floors_estimate":1,"confidence":0.0,'
    '"label_ru":"<short Russian name of the object type>",'
    '"reasoning":"2-3 sentences: what you see (corridor with many separate dwellings? one self-contained '
    'unit? many same-size offices/cubicles? large open halls/racking? a building section with several '
    'stacked floors? a site/master plan?) and why you chose this category"}\n'
    "Guidance: a floor with a central corridor/stair-core and MANY separate dwellings each with its own "
    "kitchen+bath = apartment_building_floor (units_estimate = how many). One self-contained dwelling = "
    "single_apartment. A standalone home (often with terrace, 'parter/pietro') = house. A drawing showing "
    "several stacked floors / a section / a facade = multi_storey_building. Many similar offices, meeting "
    "rooms or open-plan desks = office. Shops, showrooms, large retail halls = retail_commercial. Big open "
    "spans with racking/loading = warehouse_industrial. Be decisive; set confidence honestly.")

CATEGORY_RU = {
    "apartment_building_floor": "🏢 Этаж многоквартирного дома",
    "single_apartment": "🏠 Квартира",
    "house": "🏡 Частный дом",
    "multi_storey_building": "🏬 Многоэтажный дом (общий план)",
    "office": "🏢 Офисное помещение",
    "retail_commercial": "🏬 Коммерческая недвижимость",
    "warehouse_industrial": "🏭 Склад / производство",
    "public_civic": "🏛 Общественное здание",
    "site_masterplan": "🗺 Генплан участка",
    "other": "🏠 Объект",
}

READ_ALL_PROMPT = (
    "You are reading an architectural floor plan. Transcribe EVERYTHING on it. Return STRICT JSON only:\n"
    '{"overall_width_mm":null,"overall_height_mm":null,"scale_note":"","north":"",'
    '"dimensions_mm":[],'
    '"labels":[{"text":"<verbatim as printed>","role":"room_name|area|dimension|apartment_id|legend|title|annotation|north|scale|other","value_m2":null}],'
    '"legend":[],"title":""}\n'
    "Read the OVERALL outer width and height in millimetres from the dimension lines on each side — "
    "if a side is a chain of segments (e.g. 2800 + 4000), SUM them. List every dimension number you can "
    "read in `dimensions_mm`. Read EVERY text label into `labels` with its role; for areas put the m² in "
    "value_m2 (for '28.0(40.5)' use the larger, 40.5). Mark legend keys, the title block, the north arrow "
    "and any scale note ('1:100'). Be exact with digits — this is used for measurement.")

app = FastAPI(title="gpu-broker")

_lock = threading.Lock()
_swap_lock = threading.Lock()          # serialize GPU model swaps (worker restore vs /api/llm)
_jobs: dict[str, dict] = {}
_order: list[str] = []                 # submission order
_state = {"model": "whisper", "swapping": None, "idle_since": time.time()}
_avg = {"render": 70.0, "furnish": 70.0, "interior": 240.0, "reference": 45.0, "project": 420.0}
RESTORE_IDLE = 25                      # restore duty model after this many idle seconds


def _sh(cmd, t=240):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=t)


def _alive(url, path="/health", t=3):
    try:
        return httpx.get(url + path, timeout=t).status_code < 500
    except Exception:
        return False


def _gpu():
    try:
        out = _sh(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                   "--format=csv,noheader,nounits"], 10).stdout.strip().split(",")
        return {"vram_used": int(out[0]), "vram_total": int(out[1]), "util": int(out[2])}
    except Exception:
        return {"vram_used": 0, "vram_total": 0, "util": 0}


# --------------------------------------------------------------------------- #
# Model swapping (single resident model)                                      #
# --------------------------------------------------------------------------- #
TTS_URL = "http://127.0.0.1:8000"        # whisper-xtts (duty)


def _swap_to(target):
    """Keep exactly one resident model on the T4. target in {whisper, vlm, render, llm}.
    Serialized by _swap_lock so the idle-restore (worker thread) and /api/llm (request
    thread) can never interleave docker stop/start on the GPU."""
    with _swap_lock:
        # only short-circuit if the target is ALSO actually alive — guards against a state
        # desync (e.g. broker restarted while duty containers were stopped) that would leave
        # whisper "resident" on paper but the container down (STT/TTS 000).
        alive = True
        if target == "render":
            alive = _alive(RENDER_URL)
        elif target == "whisper":
            alive = _alive(TTS_URL)
        if _state["model"] == target and alive:
            return
        _state["swapping"] = {"whisper": "Возврат STT/TTS…", "vlm": "Загрузка qwen2.5vl (анализ плана)…",
                              "render": "Загрузка рендера (SD)…",
                              "llm": "Освобождаю GPU под перевод (LLM)…"}.get(target, "Своп…")
        if target != "render":
            _sh(["sudo", "docker", "stop", RENDER_CONTAINER])
        if target != "whisper":
            _sh(["sudo", "docker", "stop", *DUTY])
        if target != "vlm":
            _sh(["sudo", "docker", "exec", "ollama", "ollama", "stop", VLM_MODEL], 30)
        if target != "llm":                       # leaving LLM mode -> unload the heavy model
            for _m in HEAVY_LLM_MODELS:
                _sh(["sudo", "docker", "exec", "ollama", "ollama", "stop", _m], 30)
        if target == "render":
            _sh(["sudo", "docker", "compose", "-f", RENDER_COMPOSE, "up", "-d"])
            for _ in range(120):
                if _alive(RENDER_URL):
                    break
                time.sleep(2)
        elif target == "whisper":
            _sh(["sudo", "docker", "start", *DUTY])
        # vlm: ollama container stays up; qwen loads on first call with free VRAM
        _state["model"] = target
        _state["swapping"] = None


def _restore_duty():
    _swap_to("whisper")


CLASSIFY_PROMPT = (
    'Classify this image. Return STRICT JSON only: '
    '{"kind":"plan|photo","floor":"ground|upper|unknown","floor_name":"","room_type":""}. '
    'Decide carefully: kind="plan" ONLY if it is a 2D architectural floor plan / blueprint '
    '(a flat top-down black-and-white line drawing of room outlines with labels and dimensions). '
    'kind="photo" if it is a REAL photograph of an interior (perspective, furniture, real '
    'textures, colors, lighting) — most camera pictures of rooms are "photo". '
    'If plan: title "parter/rzut na parter"=ground, "pietro/piÄtro"=upper; floor_name like "Партер"/"Этаж". '
    'If photo: room_type = the room shown (kitchen/bedroom/living room/bathroom/dining room/hallway/...).')


def _img_b64(src_path, max_long=1600):
    """Base64 for the VLM — downscale oversized images so they fit the model context.
    A huge plan can produce >4000 image tokens and overflow ollama's context window."""
    import base64
    import io
    try:
        from PIL import Image
        im = Image.open(src_path).convert("RGB")
        w, h = im.size
        if max(w, h) > max_long:
            s = max_long / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return base64.b64encode(open(src_path, "rb").read()).decode()


def _parse_json(raw):
    """Robustly parse a JSON object from a possibly-noisy VLM response."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.split("```", 1)[0]
    try:
        return json.loads(s)
    except Exception:
        i, jx = s.find("{"), s.rfind("}")
        if 0 <= i < jx:
            try:
                return json.loads(s[i:jx + 1])
            except Exception:
                return None
    return None


INVENTORY_PROMPT = (
    "You are compiling a DETAILED property inventory for a handover/acceptance act and "
    "valuation. Look at this room photo and list EVERYTHING visible as STRICT JSON:\n"
    '{"room_type":"","summary":"a 1-2 sentence overview of the room and its condition",'
    '"materials":{"floor":"","walls":"","ceiling":""},"windows":0,"doors":0,'
    '"objects":[{"name":"","category":"furniture|appliance|electronics|lighting|plumbing|fixture|decor|textile|other",'
    '"material":"","color":"","condition":"new|good|worn|damaged","qty":1,"note":"","bbox":[0.0,0.0,0.0,0.0]}]}\n'
    "For EACH object also give bbox = [x1,y1,x2,y2], the bounding box of that object as "
    "FRACTIONS of the image width and height (each value between 0.0 and 1.0; x1<x2, y1<y2). "
    "Be EXHAUSTIVE: every piece of furniture, every appliance and electronic device "
    "(refrigerator, oven, stove, range hood, dishwasher, washing machine, TV, AC, "
    "switches, sockets), every light fixture, every plumbing fixture (sink, faucet, "
    "toilet, bathtub, shower), flooring/wall/ceiling materials and colors, curtains, "
    "rugs, decor. For each item give its material, color and condition. This data will "
    "be reused for handover documents and cost estimation, so be precise and complete.")


def _inventory(src_path):
    """Detailed object/material/condition inventory of a real room photo (for acts)."""
    img = _img_b64(src_path)
    def _call(temp):
        r = httpx.post(OLLAMA + "/api/generate",
                       json={"model": VLM_MODEL, "prompt": INVENTORY_PROMPT, "images": [img],
                             "stream": False, "format": "json",
                             "options": {"temperature": temp, "num_predict": 3072, "num_ctx": 8192}}, timeout=400)
        return _parse_json(r.json().get("response", "")) or {"objects": [], "summary": ""}
    inv = _call(0.1)
    if not inv.get("objects"):          # the VLM is occasionally lazy — retry once
        inv2 = _call(0.3)
        if inv2.get("objects"):
            inv = inv2
    return inv


def _segment_objects(psrc, inv, jid, pj):
    """Crop each detected object out of the photo by its VLM bbox -> segmented record."""
    try:
        from PIL import Image
        im = Image.open(psrc).convert("RGB")
        W, H = im.size
    except Exception:
        return
    for k, o in enumerate(inv.get("objects", []) or []):
        b = o.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        try:
            v = [float(x) for x in b]
        except Exception:
            continue
        mx = max(v) if v else 0
        if mx <= 1.5:                              # fractions 0..1
            x1, y1, x2, y2 = v[0] * W, v[1] * H, v[2] * W, v[3] * H
        elif mx <= 1000:                           # Qwen2.5-VL grounding: 0..1000 normalized
            x1, y1, x2, y2 = v[0] / 1000 * W, v[1] / 1000 * H, v[2] / 1000 * W, v[3] / 1000 * H
        else:                                       # absolute pixels
            x1, y1, x2, y2 = v
        X1, X2 = sorted([x1, x2])
        Y1, Y2 = sorted([y1, y2])
        X1, Y1, X2, Y2 = max(0, int(X1)), max(0, int(Y1)), min(W, int(X2)), min(H, int(Y2))
        if X2 - X1 > 8 and Y2 - Y1 > 8:
            try:
                im.crop((X1, Y1, X2, Y2)).save(os.path.join(INVENTORY, f"{jid}_{pj}_obj{k}.png"))
                o["crop"] = k
                o["bbox_px"] = [X1, Y1, X2, Y2]
            except Exception:
                pass


def _scene(src_path, vlm_read=None):
    """Build the precise 3D geometry of a plan via the Mask R-CNN backend (CPU):
    walls/doors/windows boxes + OCR room labels. Saved so the 3D is reproducible.
    vlm_read: optional output of _read_all() (VLM full-text reading) for the understanding stage."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.open(src_path).convert("RGB").save(buf, format="PNG")
    files = {"image": ("plan.png", buf.getvalue(), "image/png")}
    out = {}
    try:
        det = httpx.post(FLOORPLAN_URL + "/", files=files, timeout=180).json()
        out.update({"Width": det.get("Width"), "Height": det.get("Height"),
                    "points": det.get("points"), "classes": det.get("classes")})
    except Exception as exc:
        out["error"] = str(exc)
    try:
        ocr = httpx.post(FLOORPLAN_URL + "/ocr", files=files, timeout=180).json()
        out["rooms"] = ocr.get("rooms", [])
        out["stairs_pts"] = ocr.get("stairs", [])
        out["area_labels"] = ocr.get("area_labels", [])   # numeric room areas (m²) w/ positions
        out["dim_mm"] = ocr.get("dim_mm", [])              # overall dimensions (mm) -> true scale
        out["ocr_width"] = ocr.get("Width")
        out["ocr_height"] = ocr.get("Height")
    except Exception:
        pass
    # Angle-preserving wall centre-lines (handles non-orthogonal walls — the box
    # detector above can only emit axis-aligned rectangles). Stored alongside.
    try:
        vec = httpx.post(FLOORPLAN_URL + "/vectorize", files=files, timeout=180).json()
        out["wall_segments"] = vec.get("wall_segments", [])
        out["vec_width"] = vec.get("Width")
        out["vec_height"] = vec.get("Height")
        out["wall_thickness_px"] = vec.get("wall_thickness_px")
        out["diagonal_fraction"] = vec.get("diagonal_fraction")
    except Exception:
        pass
    # True room regions that respect walls, seeded by the area-label anchors
    # (fixes zones bleeding across walls). Seeds use OCR-space pixel coords.
    try:
        seeds = [{"id": i, "x": a["x"], "y": a["y"]} for i, a in enumerate(out.get("area_labels", []))]
        if seeds:
            seg = httpx.post(FLOORPLAN_URL + "/segment",
                             files={"image": ("plan.png", buf.getvalue(), "image/png")},
                             data={"seeds": json.dumps(seeds)}, timeout=180).json()
            out["room_geom"] = seg.get("rooms", [])
            out["seg_width"] = seg.get("Width")
            out["seg_height"] = seg.get("Height")
    except Exception:
        pass
    # CubiCasa5k neural parser — PRIMARY geometry: vector walls + room polygons +
    # doors/windows + fixtures. Hybrid: types/inventory still come from VLM+OCR.
    try:
        cubi = httpx.post(CUBI_URL + "/parse",
                          files={"image": ("plan.png", buf.getvalue(), "image/png")},
                          timeout=300).json()
        out["cubi"] = {k: cubi.get(k) for k in
                       ("Width", "Height", "walls", "wall_paths", "rooms", "openings",
                        "fixtures", "apartments", "colored_fraction")}
    except Exception as exc:
        out["cubi_error"] = str(exc)
    out["apartments_info"] = _apartment_summary(out)   # server-side: persisted + library
    if vlm_read:
        out["vlm_read"] = vlm_read
    out["understand"] = _understand(out)               # scale solver + roles + quality + VLM fusion
    return out


def _point_in_poly(px, py, pl):
    c, j = False, len(pl) - 1
    for i in range(len(pl)):
        xi, yi = pl[i]
        xj, yj = pl[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            c = not c
        j = i
    return c


def _apartment_summary(sc):
    """Server-side structured apartment data (persisted to scene + project + library):
    each colour-coded apartment, its area in m² (from the plan's PRINTED numbers, which
    also calibrate scale), and its rooms (the area numbers printed inside it)."""
    cubi = sc.get("cubi") or {}
    apts = cubi.get("apartments") or []
    if not apts:
        return None
    W = sc.get("Width") or cubi.get("Width") or 1000
    H = sc.get("Height") or cubi.get("Height") or 1000
    ax, ay = W / (cubi.get("Width") or W), H / (cubi.get("Height") or H)
    ox, oy = W / (sc.get("ocr_width") or W), H / (sc.get("ocr_height") or H)
    labs = [(a["value"], a["x"] * ox, a["y"] * oy) for a in (sc.get("area_labels") or [])]
    polys = [[[p[0] * ax, p[1] * ay] for p in ap.get("polygon", [])] for ap in apts]
    # UNIQUE assignment: each printed area number belongs to exactly ONE apartment (the
    # smallest-area polygon that contains it) -> clean per-apartment room composition.
    inside = {k: [] for k in range(len(apts))}
    for v, x, y in labs:
        cand = [k for k, pg in enumerate(polys) if len(pg) >= 3 and _point_in_poly(x, y, pg)]
        if not cand:
            continue
        k = min(cand, key=lambda k: apts[k]["area_px"])
        inside[k].append(v)
    for k in inside:
        inside[k].sort(reverse=True)
    sm2 = spx = 0.0
    for k, ap in enumerate(apts):
        vals = inside.get(k, [])
        if vals and vals[0] >= 8:                       # largest printed number = total flat area
            sm2 += vals[0]
            spx += ap["area_px"] * ax * ay
    mpp = (sm2 / spx) ** 0.5 if (sum(1 for v in inside.values() if v and v[0] >= 8) >= 2 and spx > 0) else None
    out = []
    for k, ap in enumerate(apts):
        vals = inside.get(k, [])
        area = vals[0] if (vals and vals[0] >= 8) else (round(ap["area_px"] * ax * ay * mpp * mpp, 1) if mpp else None)
        rooms = [v for v in vals[1:] if v >= 2] if len(vals) > 1 else []
        out.append({"id": k + 1, "area_m2": round(area, 1) if area else None,
                    "rooms": rooms, "rooms_count": len(rooms) or (1 if area else 0)})
    total = round(sum(a["area_m2"] for a in out if a["area_m2"]), 1)
    return {"count": len(apts), "total_area_m2": total or None,
            "mpp": mpp, "scale_long_m": round(max(W, H) * mpp, 1) if mpp else None,
            "apartments": out}


def _understand(out):
    """Preliminary PLAN UNDERSTANDING (CPU, no GPU): solve a reliable scale from the
    plan's own numbers and tag the text by role + a quality score.
    Scale solver (arithmetic, outlier-rejecting):
      E1 dimension chains — collinear dimension numbers sit at segment midpoints, so the
         pixel gap between adjacent numbers = (mm_a+mm_b)/2; mpp = (mm_a+mm_b)/(2000·Δpx).
         A single misread (5100→9100) becomes an outlier and is dropped by the median.
      E2 printed area ÷ polygon pixel-area (from apartments_info).
    Consensus prefers E1; emits confidence + needs_review."""
    W = out.get("Width") or 1000
    H = out.get("Height") or 1000
    osx, osy = W / (out.get("ocr_width") or W), H / (out.get("ocr_height") or H)
    dims = [{"mm": d["mm"], "x": d["x"] * osx, "y": d["y"] * osy}
            for d in (out.get("dim_mm") or []) if d.get("mm")]
    tol = 0.04 * max(W, H)
    ratios = []
    for axis in ("h", "v"):
        key = (lambda t: t["y"]) if axis == "h" else (lambda t: t["x"])
        pos = (lambda t: t["x"]) if axis == "h" else (lambda t: t["y"])
        used = [False] * len(dims)
        order = sorted(range(len(dims)), key=lambda i: key(dims[i]))
        for i in order:
            if used[i]:
                continue
            grp = [i]
            used[i] = True
            for j in order:
                if not used[j] and abs(key(dims[j]) - key(dims[i])) < tol:
                    grp.append(j)
                    used[j] = True
            if len(grp) < 2:
                continue
            grp.sort(key=lambda k: pos(dims[k]))
            for a, b in zip(grp, grp[1:]):
                dpx = abs(pos(dims[b]) - pos(dims[a]))
                ssum = dims[a]["mm"] + dims[b]["mm"]
                if dpx > 3 and ssum > 0:
                    ratios.append(ssum / (2000.0 * dpx))         # metres per pixel
    est = []
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        good = [r for r in ratios if abs(r - med) <= 0.25 * med] if len(ratios) >= 3 else ratios
        if good:
            est.append({"method": "dim_chain", "mpp": round(sum(good) / len(good), 6),
                        "support": len(good), "rejected": len(ratios) - len(good)})
    ai = out.get("apartments_info") or {}
    if ai.get("mpp"):
        est.append({"method": "area_vs_polygon", "mpp": round(ai["mpp"], 6),
                    "support": ai.get("count", 1)})
    else:
        # single-apartment / single plan: calibrate from ROOM area labels vs CubiCasa room polygons
        cubi = out.get("cubi") or {}
        rms = cubi.get("rooms") or []
        if rms and (out.get("area_labels")):
            ax = W / (cubi.get("Width") or W); ay = H / (cubi.get("Height") or H)
            labs = [(a["value"], a["x"] * osx, a["y"] * osy) for a in out["area_labels"]]
            sm2 = spx = 0.0
            for rm in rms:
                pg = [[p[0] * ax, p[1] * ay] for p in rm.get("polygon", [])]
                if len(pg) < 3:
                    continue
                # polygon area via shoelace
                a2 = 0.0
                for i in range(len(pg)):
                    x1, y1 = pg[i]; x2, y2 = pg[(i + 1) % len(pg)]
                    a2 += x1 * y2 - x2 * y1
                area_px = abs(a2) / 2.0
                best = 0.0
                for v, x, y in labs:
                    if v >= 2 and _point_in_poly(x, y, pg):
                        best = max(best, v)
                if best >= 2 and area_px > 50:
                    sm2 += best; spx += area_px
            if sm2 > 0 and spx > 0:
                est.append({"method": "area_vs_polygon", "mpp": round((sm2 / spx) ** 0.5, 6),
                            "support": "rooms"})
    # E3: VLM-read OVERALL dimension vs the building's pixel bbox (catches what OCR missed).
    vr = out.get("vlm_read") or {}
    cubi = out.get("cubi") or {}
    cw, ch = cubi.get("Width") or W, cubi.get("Height") or H
    bx = by = 1e9
    bX = bY = -1e9
    for pa in (cubi.get("wall_paths") or []):
        for p in (pa.get("p") or pa):
            X, Y = p[0] * W / cw, p[1] * H / ch
            bx, bX, by, bY = min(bx, X), max(bX, X), min(by, Y), max(bY, Y)
    if bX > bx:
        blong = max(bX - bx, bY - by)
        omax = max([v for v in (vr.get("overall_width_mm"), vr.get("overall_height_mm"))
                    if isinstance(v, (int, float)) and v > 500] or [0])
        if omax and blong > 10:
            cal = (omax / 1000.0) / blong
            if 0.002 <= cal <= 0.05:
                est.append({"method": "vlm_overall_dim", "mpp": round(cal, 6), "support": 1})
    # also feed VLM-read dimension VALUES that OCR missed into the chain pool (cross-check only)
    vlm_dims = [d for d in (vr.get("dimensions_mm") or []) if isinstance(d, (int, float)) and 500 < d < 30000]

    chain = [e for e in est if e["method"] == "dim_chain"]
    if chain:
        mpp, method = chain[0]["mpp"], "dim_chain"
    elif est:
        mpp, method = est[0]["mpp"], est[0]["method"]
    else:
        mpp, method = None, None
    agree = len([e for e in est if mpp and abs(e["mpp"] - mpp) <= 0.25 * mpp])
    conf = round(min(1.0, 0.4 * agree + (0.2 if chain else 0.0) + (0.1 if len(est) >= 2 else 0.0)), 2)

    # --- per-token catalogue (roles) — OCR tokens (with positions) + VLM-only labels ---
    tokens = []
    for a in (out.get("area_labels") or []):
        tokens.append({"text": a.get("text", str(a.get("value"))), "role": "area",
                       "value_m2": a.get("value"), "x": a.get("x"), "y": a.get("y"), "src": "ocr"})
    for d in (out.get("dim_mm") or []):
        tokens.append({"text": str(d.get("mm")), "role": "dimension", "mm": d.get("mm"),
                       "x": d.get("x"), "y": d.get("y"), "src": "ocr"})
    for r in (out.get("rooms") or []):
        tokens.append({"text": r.get("name"), "role": "room_name", "x": r.get("x"), "y": r.get("y"), "src": "ocr"})
    vlm_labels = [l for l in (vr.get("labels") or [])
                  if l.get("text") and "<" not in str(l.get("text")) and str(l.get("text")).strip()]
    for l in vlm_labels:
        tokens.append({"text": l.get("text"), "role": l.get("role") or "other",
                       "value_m2": l.get("value_m2"), "src": "vlm"})
    roles = {}
    for tk in tokens:
        roles[tk["role"]] = roles.get(tk["role"], 0) + 1
    # OCR↔VLM agreement on label count (rough text-coverage signal)
    ocr_n = len(out.get("area_labels") or []) + len(out.get("dim_mm") or []) + len(out.get("rooms") or [])
    cover = round(min(1.0, ocr_n / max(1, len(vlm_labels))), 2) if vlm_labels else None

    return {"scale": {"mpp": mpp, "method": method, "estimators": est, "confidence": conf,
                      "scale_long_m": round(max(W, H) * mpp, 1) if mpp else None,
                      "vlm_overall_mm": [vr.get("overall_width_mm"), vr.get("overall_height_mm")]},
            "north": vr.get("north") or "", "scale_note": vr.get("scale_note") or "",
            "legend": vr.get("legend") or [], "title": vr.get("title") or "",
            "roles": roles, "tokens": tokens, "vlm_dims": vlm_dims, "text_coverage": cover,
            "needs_review": (mpp is None or agree < 2 or conf < 0.5)}


def _classify(src_path):
    """Decide if an uploaded image is a floor plan or a real room photo."""
    img = _img_b64(src_path)
    r = httpx.post(OLLAMA + "/api/generate",
                   json={"model": VLM_MODEL, "prompt": CLASSIFY_PROMPT, "images": [img],
                         "stream": False, "format": "json",
                         "options": {"temperature": 0, "num_predict": 256, "num_ctx": 8192}},
                   timeout=200)
    return _parse_json(r.json().get("response", "")) or {"kind": "plan", "floor": "unknown",
                                                          "floor_name": "", "room_type": ""}


def _classify_plan(src_path):
    """UNIVERSAL plan-type classification (works on any plan via the VLM): is this an
    apartment-building floor / single apartment / house / multi-storey building / office /
    retail / warehouse / ...  Drives downstream processing + display."""
    img = _img_b64(src_path)
    r = httpx.post(OLLAMA + "/api/generate",
                   json={"model": VLM_MODEL, "prompt": PLAN_CATEGORY_PROMPT, "images": [img],
                         "stream": False, "format": "json",
                         "options": {"temperature": 0, "num_predict": 400, "num_ctx": 8192}},
                   timeout=240)
    c = _parse_json(r.json().get("response", "")) or {}
    cat = c.get("category") if c.get("category") in CATEGORY_RU else "other"
    c["category"] = cat
    c["label_ru"] = c.get("label_ru") or CATEGORY_RU[cat]
    return c


def _read_all(src_path):
    """ONE consolidated VLM pass that reads ALL text on the plan (catches dimensions and
    labels Tesseract misses/misreads) + overall size, scale note, north, legend, title.
    Runs while the vlm model is already resident → no extra GPU swap."""
    img = _img_b64(src_path)
    r = httpx.post(OLLAMA + "/api/generate",
                   json={"model": VLM_MODEL, "prompt": READ_ALL_PROMPT, "images": [img],
                         "stream": False, "format": "json",
                         "options": {"temperature": 0, "num_predict": 1200, "num_ctx": 8192}},
                   timeout=300)
    return _parse_json(r.json().get("response", "")) or {}


def _resolve_category(vlm_cat, apt_count, units_est=0, n_kitchens=0, n_baths=0):
    """Combine the universal VLM category with hard geometric/semantic evidence.
    An apartment-building FLOOR must actually contain SEVERAL dwellings — each with its
    own kitchen+bath. Guard against the VLM calling a single dwelling a 'floor' just
    because it sees many rooms (the 1-kitchen, 1-bath single-apartment case)."""
    if (apt_count or 0) >= 3:
        return "apartment_building_floor"            # strong geometric evidence (colour units)
    if vlm_cat == "apartment_building_floor":
        multi = (units_est or 0) >= 3 and (n_kitchens >= 2 or n_baths >= 2)
        return "apartment_building_floor" if multi else "single_apartment"
    if vlm_cat in CATEGORY_RU and vlm_cat != "other":
        return vlm_cat
    return "single_apartment"


def _analyze(src_path, extra=""):
    """qwen2.5vl reads the whole plan -> structured architectural context."""
    img = _img_b64(src_path)
    prompt = ANALYZE_PROMPT
    if extra:
        prompt += "\nThe owner also provided this description — use it as additional context: " + extra
    r = httpx.post(OLLAMA + "/api/generate",
                   json={"model": VLM_MODEL, "prompt": prompt, "images": [img],
                         "stream": False, "format": "json",
                         "options": {"temperature": 0.1, "num_predict": 2048, "num_ctx": 8192}},
                   timeout=400)
    raw = r.json().get("response", "") if r.status_code == 200 else r.text
    ctx = _parse_json(raw) or {"rooms": [], "summary": str(raw)[:400]}
    # Fallback: the rich schema sometimes returns empty rooms on busy plans — re-ask
    # with a simpler rooms-only prompt (proven reliable) and merge.
    if not ctx.get("rooms"):
        simple = ('Read this floor plan image. Return STRICT JSON only: '
                  '{"rooms":[{"name":"","type":"kitchen|bedroom|bathroom|living room|dining room|'
                  'hallway|utility|guest room|walk-in closet|terrace|other","area_m2":null,'
                  '"dimensions":"","windows":0,"fixtures":[]}],"stairs":false,"summary":""}. '
                  "List EVERY room with its name (as written), English type, area in m2, dimensions, "
                  "window count, and the fixtures drawn inside it.")
        r2 = httpx.post(OLLAMA + "/api/generate",
                        json={"model": VLM_MODEL, "prompt": simple, "images": [img],
                              "stream": False, "format": "json",
                              "options": {"temperature": 0.1, "num_predict": 1536, "num_ctx": 8192}}, timeout=400)
        c2 = _parse_json(r2.json().get("response", "")) or {}
        if c2.get("rooms"):
            for k, v in c2.items():
                if not ctx.get(k):
                    ctx[k] = v
            ctx["rooms"] = c2["rooms"]
            raw = (raw or "") + "\n\n[fallback rooms]\n" + r2.json().get("response", "")
    # total area: trust the VLM, else sum the room areas
    if not ctx.get("total_area_m2"):
        areas = [x.get("area_m2") for x in ctx.get("rooms", []) if isinstance(x.get("area_m2"), (int, float))]
        if areas:
            ctx["total_area_m2"] = round(sum(areas), 1)
    return ctx, raw


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #
def _next_queued():
    with _lock:
        for jid in _order:
            if _jobs[jid]["status"] == "queued":
                return _jobs[jid]
    return None


def _worker():
    while True:
        job = _next_queued()
        if job is None:
            # idle: restore the duty model after a grace period
            if _state["model"] != "whisper" and time.time() - _state["idle_since"] > RESTORE_IDLE:
                _restore_duty()
            time.sleep(1)
            continue
        job["status"] = "running"
        job["started"] = time.time()
        try:
            t = job["type"]
            if t == "project":
                desc = job.get("description", "")
                _swap_to("vlm")
                plans, photos = [], []
                for k, src in enumerate(job["srcs"]):
                    job["step"] = f"Классификация изображений {k + 1}/{len(job['srcs'])}…"
                    cls = _classify(src)
                    (plans if cls.get("kind") == "plan" else photos).append((src, cls))
                floors = []
                _seen = {}
                for pi, (src, cls) in enumerate(plans):
                    fname = cls.get("floor_name") or {"ground": "Партер", "upper": "Этаж"}.get(cls.get("floor"), "")
                    if not fname or fname in _seen:           # disambiguate duplicate/empty names
                        fname = f"Этаж {pi + 1}"
                    _seen[fname] = 1
                    job["step"] = f"Тип объекта: {fname}…"
                    pcat = _classify_plan(src)                    # UNIVERSAL plan-type classification
                    job["step"] = f"Чтение надписей и размеров: {fname}…"
                    vlm_read = _read_all(src)                     # VLM reads ALL text (vlm resident)
                    job["step"] = f"Анализ плана: {fname}…"
                    ctx, raw = _analyze(src, desc)
                    job["step"] = f"3D-схема: {fname}…"          # precise geometry via Mask R-CNN
                    sc = _scene(src, vlm_read=vlm_read)
                    sc["vlm_rooms"] = ctx.get("rooms", [])
                    sc["floor_name"] = fname
                    sc["building_type"] = ctx.get("building_type")
                    sc["floors_total"] = len(plans)
                    _apt_n = (sc.get("apartments_info") or {}).get("count", 0)
                    _rt = [str(r.get("type") or "").lower() for r in ctx.get("rooms", [])]
                    _nk = sum("kitchen" in t for t in _rt)
                    _nb = sum("bath" in t for t in _rt)
                    sc["category"] = _resolve_category(pcat.get("category"), _apt_n,
                                                       pcat.get("units_estimate"), _nk, _nb)
                    sc["category_label"] = CATEGORY_RU.get(sc["category"], pcat.get("label_ru"))
                    sc["category_reasoning"] = pcat.get("reasoning")
                    sc["category_raw"] = pcat
                    with open(os.path.join(SCENES, f"{job['id']}_{pi}.json"), "w") as sf:
                        json.dump(sc, sf)
                    floors.append({"name": fname, "level": cls.get("floor"), "ctx": ctx,
                                   "raw": raw, "scene": pi, "apt": sc.get("apartments_info"),
                                   "category": sc["category"], "category_label": sc["category_label"]})
                all_rooms = []
                for fi, fl in enumerate(floors):
                    for r in fl["ctx"].get("rooms", []):
                        if r.get("name"):
                            all_rooms.append({**r, "floor": fl["name"], "floor_idx": fi})
                # aggregate apartments across floors (multi-unit understanding)
                apts_all = []
                for fl in floors:
                    ai = fl.get("apt") or {}
                    for a in ai.get("apartments", []):
                        apts_all.append({**a, "floor": fl["name"]})
                apts_total = len(apts_all)
                living = round(sum((a.get("area_m2") or 0) for a in apts_all), 1)
                is_multi = apts_total >= 3
                # universal category: prefer the VLM-resolved one from floor 0
                cat_key = floors[0].get("category") if floors else None
                if is_multi:
                    cat_key = "apartment_building_floor"
                cat_label = CATEGORY_RU.get(cat_key, "🏠 Объект")
                job["project"] = {
                    "building_type": cat_label,
                    "category": cat_key or "other",
                    "category_reasoning": floors[0].get("ctx", {}).get("reasoning") if floors else None,
                    "levels": len(floors) or 1,
                    "floors": [{"name": fl["name"], "summary": fl["ctx"].get("summary"),
                                "scene": fl.get("scene"),
                                "apartments": (fl.get("apt") or {}).get("count"),
                                "living_area_m2": (fl.get("apt") or {}).get("total_area_m2")} for fl in floors],
                    "apartments_total": apts_total or None,
                    "living_area_m2": living or None,
                    "apartments": apts_all,
                    "total_area_m2": round(sum((r.get("area_m2") or 0) for r in all_rooms), 1) if all_rooms else None,
                    "rooms_total": len(all_rooms),
                    "stairs": any(fl["ctx"].get("stairs") for fl in floors),
                    "photos": len(photos), "description": desc,
                    # full per-room records (persisted for documents/acts)
                    "rooms": [{"name": r.get("name"), "floor": r.get("floor"), "type": r.get("type"),
                               "area_m2": r.get("area_m2"), "dimensions": r.get("dimensions"),
                               "windows": r.get("windows"), "fixtures": r.get("fixtures"),
                               "description": r.get("description")} for r in all_rooms]}
                job["reasoning"] = "\n\n".join(f"[{fl['name']}] " + str(fl["ctx"].get("reasoning", "")) for fl in floors)
                job["raw"] = "\n\n".join(fl["raw"] for fl in floors)
                # detailed inventory of each reference photo (still on the VLM) — saved for acts
                photo_inv = {}
                for pj, (psrc, pcls) in enumerate(photos):
                    job["step"] = f"Опись референса {pj + 1}/{len(photos)}…"
                    inv = _inventory(psrc)
                    inv["room_type"] = inv.get("room_type") or pcls.get("room_type", "")
                    _segment_objects(psrc, inv, job["id"], pj)   # crop each object out
                    with open(os.path.join(INVENTORY, f"{job['id']}_{pj}.json"), "w") as invf:
                        json.dump(inv, invf, ensure_ascii=False)
                    photo_inv[pj] = inv
                do_render = bool(job.get("render"))       # renders OFF by default (save GPU)
                if do_render:
                    _swap_to("render")
                photo_by_type = {}
                for src, cls in photos:
                    photo_by_type.setdefault((cls.get("room_type") or "").lower(), src)
                imgs = []
                for i, r in enumerate(all_rooms[:16] if do_render else []):
                    job["step"] = f"Рендер: {r.get('name')} · {r.get('floor')} ({i + 1}/{len(all_rooms)})"
                    ref = photo_by_type.get((r.get("type") or "").lower())
                    src_kind = "generated"
                    if ref:
                        with open(ref, "rb") as fh:
                            rr = httpx.post(RENDER_URL + "/photoreal",
                                            data={"prompt": f"photorealistic {r.get('type','room')} interior, {job['style']}, clean, tidy, improved lighting and materials, high detail"},
                                            files={"image": ("ref.png", fh, "image/png")}, timeout=400)
                        src_kind = "reference"
                    else:
                        fx = r.get("fixtures", [])
                        fx = ", ".join(str(x) for x in fx) if isinstance(fx, list) else str(fx)
                        rr = httpx.post(RENDER_URL + "/room",
                                        data={"name": r.get("name", ""), "room_type": r.get("type", ""),
                                              "fixtures": fx, "description": r.get("description", ""),
                                              "windows": r.get("windows", 0), "style": job["style"]}, timeout=400)
                    if rr.status_code == 200:
                        open(os.path.join(RESULTS, f"{job['id']}_{i}.png"), "wb").write(rr.content)
                        imgs.append({"room": r.get("name"), "floor": r.get("floor"), "type": r.get("type"),
                                     "area": r.get("area_m2"), "dimensions": r.get("dimensions"),
                                     "windows": r.get("windows"), "fixtures": r.get("fixtures"),
                                     "description": r.get("description"), "idx": i, "source": src_kind})
                # reference photos: improve via img2img — only when rendering is enabled
                for pj, (psrc, pcls) in enumerate(photos if do_render else []):
                    rt = pcls.get("room_type") or "комната"
                    job["step"] = f"Референс → улучшение: {rt} ({pj + 1}/{len(photos)})"
                    try:
                        with open(psrc, "rb") as fh:
                            # neutral prompt + low strength -> keep the SAME room/layout, just improve it
                            rr = httpx.post(RENDER_URL + "/photoreal",
                                            data={"prompt": "photorealistic interior, the same room and layout, "
                                                  "decluttered, clean and tidy, improved lighting, realistic "
                                                  "materials, sharp focus, high detail, 8k",
                                                  "strength": 0.35},
                                            files={"image": ("ref.png", fh, "image/png")}, timeout=400)
                        if rr.status_code == 200:
                            idx = 1000 + pj
                            open(os.path.join(RESULTS, f"{job['id']}_{idx}.png"), "wb").write(rr.content)
                            imgs.append({"room": rt, "floor": "📷 Референсы (улучшенные)", "type": rt,
                                         "idx": idx, "source": "reference",
                                         "inventory": photo_inv.get(pj, {})})
                    except Exception:
                        pass
                job["images"] = imgs
                job["status"] = "done"
            elif t == "interior":
                # 1) understand the plan with the VLM, 2) render each room photoreal
                job["step"] = "Анализ плана (qwen2.5vl)…"
                _swap_to("vlm")
                ctx, raw = _analyze(job["src"])
                job["context"] = ctx
                job["reasoning"] = ctx.get("reasoning", "")
                job["raw"] = raw
                rooms = [r for r in ctx.get("rooms", []) if r.get("name")][:8]
                _swap_to("render")
                imgs = []
                for i, r in enumerate(rooms):
                    job["step"] = f"Рендер комнаты: {r.get('name', '?')} ({i + 1}/{len(rooms)})"
                    fx = r.get("fixtures", [])
                    fx = ", ".join(str(x) for x in fx) if isinstance(fx, list) else str(fx)
                    rr = httpx.post(RENDER_URL + "/room",
                                    data={"name": r.get("name", ""), "room_type": r.get("type", ""),
                                          "fixtures": fx, "description": r.get("description", ""),
                                          "windows": r.get("windows", 0),
                                          "style": job.get("style", "")}, timeout=400)
                    if rr.status_code == 200:
                        open(os.path.join(RESULTS, f"{job['id']}_{i}.png"), "wb").write(rr.content)
                        imgs.append({"room": r.get("name", "?"), "area": r.get("area_m2"), "idx": i})
                job["images"] = imgs
                job["status"] = "done"
            elif t == "reference":
                job["step"] = "Фотореализация (img2img)…"
                _swap_to("render")
                with open(job["src"], "rb") as fh:
                    rr = httpx.post(RENDER_URL + "/photoreal",
                                    data={"prompt": job["prompt"] or "photorealistic interior, clean, tidy"},
                                    files={"image": ("ref.png", fh, "image/png")}, timeout=400)
                if rr.status_code != 200:
                    raise RuntimeError(f"photoreal {rr.status_code}: {rr.text[:120]}")
                open(os.path.join(RESULTS, f"{job['id']}.png"), "wb").write(rr.content)
                job["result"] = os.path.join(RESULTS, f"{job['id']}.png")
                job["status"] = "done"
            else:  # render / furnish (top-down ControlNet, kept)
                job["step"] = "Рендер…"
                _swap_to("render")
                prompt = job["prompt"] or "interior, realistic materials, soft daylight"
                if t == "furnish":
                    prompt = (job["prompt"] or "cozy furnished apartment") + FURNISH_SUFFIX
                with open(job["src"], "rb") as fh:
                    r = httpx.post(RENDER_URL + "/render",
                                   data={"prompt": prompt, "steps": job["steps"], "scale": job["scale"]},
                                   files={"image": ("plan.png", fh, "image/png")}, timeout=900)
                if r.status_code != 200:
                    raise RuntimeError(f"render {r.status_code}: {r.text[:120]}")
                open(os.path.join(RESULTS, f"{job['id']}.png"), "wb").write(r.content)
                job["result"] = os.path.join(RESULTS, f"{job['id']}.png")
                job["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            job["status"] = "error"
            job["error"] = str(exc)[:300]
        job["finished"] = time.time()
        dt = job["finished"] - job["started"]
        t = job["type"]
        _avg[t] = 0.6 * _avg.get(t, dt) + 0.4 * dt   # rolling estimate
        _state["idle_since"] = time.time()
        # persist to the library (survives broker restarts)
        if job["status"] == "done" and (job.get("images") or job.get("result") or job.get("project")):
            try:
                meta = {"id": job["id"], "type": t, "finished": job["finished"],
                        "took": round(dt), "style": job.get("style"),
                        "project": job.get("project"), "context": job.get("context"),
                        "reasoning": job.get("reasoning"), "images": job.get("images"),
                        "result": bool(job.get("result"))}
                with open(os.path.join(PROJECTS, f"{job['id']}.json"), "w") as pf:
                    json.dump(meta, pf)
            except Exception:
                pass
        try:
            os.remove(job["src"])
        except Exception:
            pass


def _reconcile_startup():
    """On broker start the in-memory model defaults to 'whisper', but the duty containers
    may be stopped (e.g. a swap was in effect when the broker restarted). Make reality match
    so STT/TTS/avatar are actually up. Never touches 1C."""
    try:
        if _state["model"] == "whisper" and not _alive(TTS_URL):
            _sh(["sudo", "docker", "start", *DUTY])
    except Exception:
        pass


_reconcile_startup()
threading.Thread(target=_worker, daemon=True).start()


# --------------------------------------------------------------------------- #
# API                                                                          #
# --------------------------------------------------------------------------- #
@app.post("/api/jobs")
async def enqueue(image: UploadFile = File(...), type: str = Form("interior"),
                  prompt: str = Form(""), style: str = Form("scandinavian, warm wood, natural daylight"),
                  steps: int = Form(22), scale: float = Form(1.0)):
    if type not in JOB_TYPES:
        raise HTTPException(400, f"type must be one of {sorted(JOB_TYPES)}")
    jid = uuid.uuid4().hex[:10]
    src = os.path.join(RESULTS, f"{jid}.src")
    with open(src, "wb") as f:
        f.write(await image.read())
    with _lock:
        _jobs[jid] = {"id": jid, "type": type, "prompt": prompt, "style": style, "steps": steps,
                      "scale": scale, "status": "queued", "src": src, "step": None,
                      "submitted": time.time(), "started": None, "finished": None,
                      "result": None, "error": None, "context": None, "images": None}
        _order.append(jid)
    return {"id": jid, "queued": True}


@app.post("/api/project")
async def enqueue_project(files: List[UploadFile] = File(...),
                          description: str = Form(""),
                          style: str = Form("scandinavian, warm wood, natural daylight"),
                          render: bool = Form(False)):
    """Multi-image project: drop floor plans + reference photos (+ a text description).
    Classifies each image, analyzes every floor, builds the 3D plan. Photoreal RENDERS
    are OFF by default (render=true to also generate per-room SD renders) — saves GPU."""
    jid = uuid.uuid4().hex[:10]
    srcs = []
    for k, f in enumerate(files):
        p = os.path.join(RESULTS, f"{jid}_in{k}")
        with open(p, "wb") as out:
            out.write(await f.read())
        srcs.append(p)
    with _lock:
        _jobs[jid] = {"id": jid, "type": "project", "prompt": "", "style": style,
                      "description": description, "render": bool(render),
                      "steps": 22, "scale": 1.0, "status": "queued",
                      "src": srcs[0] if srcs else None, "srcs": srcs, "step": None,
                      "submitted": time.time(), "started": None, "finished": None,
                      "result": None, "error": None, "context": None, "images": None,
                      "project": None, "reasoning": None, "raw": None}
        _order.append(jid)
    return {"id": jid, "queued": True, "files": len(srcs), "render": bool(render)}


_llm_lock = threading.Lock()


@app.post("/api/llm")
def api_llm(payload: dict):
    """Run a heavy LLM with the WHOLE GPU. Swaps every other model off the T4
    (whisper/avatar/render/vlm) so the model loads fully on-GPU (no CPU offload),
    runs the generate, and lets the idle-restore bring the duty model back afterwards.
    Serialized: only one heavy LLM call runs at a time; waits out any running GPU job."""
    model = (payload or {}).get("model")
    if not model:
        raise HTTPException(400, "model required")
    # FAIL FAST under contention: only one heavy LLM call holds the GPU at a time. If we
    # can't get the slot quickly, return 503 (busy) instead of blocking forever — this is
    # what prevents a burst of concurrent requests from piling up and exhausting the
    # threadpool (the 2026-06-11 hang). Callers should retry with backoff.
    if not _llm_lock.acquire(timeout=25):
        raise HTTPException(503, "GPU busy with another translation; retry shortly")
    try:
        # don't yank the GPU out from under a running broker (3D) job — wait briefly, then give up
        waited = 0.0
        while waited < 60:
            with _lock:
                busy = any(_jobs[j]["status"] == "running" for j in _order)
            if not busy:
                break
            time.sleep(0.5)
            waited += 0.5
        else:
            raise HTTPException(503, "GPU busy with a render/3D job; retry shortly")
        _swap_to("llm")                       # stop everything else -> full GPU
        _state["idle_since"] = time.time()
        try:
            body = {**payload, "stream": False}
            body.setdefault("keep_alive", "5m")
            r = httpx.post(OLLAMA + "/api/generate", json=body, timeout=180)
            r.raise_for_status()
            return JSONResponse(r.json())
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"llm error: {exc}")
        finally:
            _state["idle_since"] = time.time()  # start the idle clock -> duty restored in ~25s
    finally:
        _llm_lock.release()


@app.get("/api/state")
def state():
    now = time.time()
    g = _gpu()
    with _lock:
        jobs = [dict(_jobs[j]) for j in _order]
    # ETA: time left on the running job, then each queued job adds its avg
    cum = 0.0
    running = next((j for j in jobs if j["status"] == "running"), None)
    if running:
        est = _avg.get(running["type"], 70.0)
        cum = max(est - (now - running["started"]), 2)
    out = []
    qpos = 0
    for j in jobs:
        item = {k: j[k] for k in ("id", "type", "prompt", "status", "error")}
        if j["status"] == "running":
            item["elapsed"] = round(now - j["started"])
            item["est"] = round(_avg.get(j["type"], 70.0))
            item["step"] = j.get("step")
        elif j["status"] == "queued":
            qpos += 1
            item["position"] = qpos
            item["eta"] = round(cum)
            cum += _avg.get(j["type"], 70.0)
        elif j["status"] in ("done", "error") and j["finished"]:
            item["took"] = round(j["finished"] - j["started"])
            if j.get("context"):
                item["context"] = j["context"]
            if j.get("project"):
                item["project"] = j["project"]
            if j.get("images"):
                item["images"] = j["images"]
            if j.get("reasoning"):
                item["reasoning"] = j["reasoning"]
            if j.get("raw"):
                item["raw"] = j["raw"][:8000]
            item["has_result"] = bool(j.get("result"))
        out.append(item)
    return JSONResponse({
        "model": _state["model"], "swapping": _state["swapping"],
        "gpu": g, "avg_render": round(_avg.get("render", 70.0)),
        "counts": {"queued": sum(1 for j in jobs if j["status"] == "queued"),
                   "running": 1 if running else 0,
                   "done": sum(1 for j in jobs if j["status"] == "done"),
                   "error": sum(1 for j in jobs if j["status"] == "error")},
        "jobs": list(reversed(out)),   # newest first
    })


@app.get("/api/jobs/{jid}/result")
def result(jid: str):
    p = os.path.join(RESULTS, f"{jid}.png")
    j = _jobs.get(jid)
    if j and j.get("result") and os.path.exists(j["result"]):
        p = j["result"]
    if not os.path.exists(p):
        raise HTTPException(404, "not ready")
    return FileResponse(p, media_type="image/png")


@app.get("/api/jobs/{jid}/room/{i}")
def room_result(jid: str, i: int):
    p = os.path.join(RESULTS, f"{jid}_{i}.png")
    if not os.path.exists(p):
        raise HTTPException(404, "not ready")
    return FileResponse(p, media_type="image/png")


@app.get("/api/jobs/{jid}/inventory/{pj}")
def inventory_api(jid: str, pj: int):
    """Detailed per-photo inventory (objects/materials/condition) for acts & valuation."""
    p = os.path.join(INVENTORY, f"{jid}_{pj}.json")
    if not os.path.exists(p):
        raise HTTPException(404, "no inventory")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{jid}/inventory/{pj}/object/{k}")
def inventory_object(jid: str, pj: int, k: int):
    """A single segmented object cropped out of the reference photo."""
    p = os.path.join(INVENTORY, f"{jid}_{pj}_obj{k}.png")
    if not os.path.exists(p):
        raise HTTPException(404, "no crop")
    return FileResponse(p, media_type="image/png")


@app.get("/api/scene/{jid}/{fi}")
def scene_data(jid: str, fi: int):
    p = os.path.join(SCENES, f"{jid}_{fi}.json")
    if not os.path.exists(p):
        raise HTTPException(404, "no scene")
    return FileResponse(p, media_type="application/json")


@app.get("/scene/{jid}/{fi}", response_class=HTMLResponse)
def scene_page(jid: str, fi: int):
    return SCENE_PAGE.replace("{{JID}}", jid).replace("{{FI}}", str(fi))


@app.get("/api/library")
def library_api():
    items = []
    for fn in os.listdir(PROJECTS):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(PROJECTS, fn)) as f:
                    items.append(json.load(f))
            except Exception:
                pass
    # join the per-photo object inventories so the library shows EVERYTHING recorded
    for it in items:
        jid = it.get("id")
        invs = []
        for inv_fn in sorted(os.listdir(INVENTORY)):
            if inv_fn.endswith(".json") and inv_fn.startswith(jid + "_"):
                try:
                    inv = json.load(open(os.path.join(INVENTORY, inv_fn)))
                    inv["_pj"] = inv_fn[len(jid) + 1:-5]
                    invs.append(inv)
                except Exception:
                    pass
        it["inventories"] = invs
    items.sort(key=lambda x: x.get("finished", 0), reverse=True)
    return JSONResponse({"items": items})


@app.get("/api/objects")
def objects_api():
    """Unified OBJECT LIBRARY: every identified+segmented object across ALL photo
    inventories of every project, with its crop, attributes and source."""
    objs = []
    for fn in sorted(os.listdir(INVENTORY)):
        if not fn.endswith(".json"):
            continue
        base = fn[:-5]                              # {jid}_{pj}
        if "_" not in base:
            continue
        jid, _, pj = base.rpartition("_")
        try:
            inv = json.load(open(os.path.join(INVENTORY, fn)))
        except Exception:
            continue
        room = inv.get("room_type") or inv.get("room") or ""
        for o in inv.get("objects", []):
            crop = o.get("crop")
            objs.append({
                "name": o.get("name"), "category": o.get("category"),
                "material": o.get("material"), "color": o.get("color"),
                "condition": o.get("condition"), "qty": o.get("qty"),
                "note": o.get("note"), "room": room, "jid": jid, "pj": pj,
                "crop_url": (f"/api/jobs/{jid}/inventory/{pj}/object/{crop}" if crop is not None else None)})
    return JSONResponse({"count": len(objs), "objects": objs})


@app.get("/api/project/{jid}/export")
def project_export(jid: str):
    """Full transferable data package: project + per-room data + 3D scenes + inventories.
    Other projects/systems can ingest this single JSON."""
    pf = os.path.join(PROJECTS, f"{jid}.json")
    if not os.path.exists(pf):
        raise HTTPException(404, "no project")
    with open(pf) as f:
        bundle = json.load(f)
    scenes, invs = {}, {}
    for fn in os.listdir(SCENES):
        if fn.startswith(jid + "_") and fn.endswith(".json"):
            try:
                scenes[fn] = json.load(open(os.path.join(SCENES, fn)))
            except Exception:
                pass
    for fn in os.listdir(INVENTORY):
        if fn.startswith(jid + "_") and fn.endswith(".json"):
            try:
                invs[fn] = json.load(open(os.path.join(INVENTORY, fn)))
            except Exception:
                pass
    bundle["scenes"], bundle["inventories"] = scenes, invs
    bundle["schema"] = "interior-project/v1"
    return JSONResponse(bundle)


@app.get("/library", response_class=HTMLResponse)
def library_page():
    return LIBRARY_PAGE


SCENE_PAGE = r"""<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>3D-схема</title>
<style>body{margin:0;background:#0e1116;color:#e6edf3;font:14px system-ui;overflow:hidden}
#hud{position:fixed;top:10px;left:10px;background:rgba(13,17,23,.85);border:1px solid #222a35;border-radius:10px;padding:10px 14px;z-index:5}
.leg{font-size:12px;color:#8b97a7;margin-top:4px}#view{width:100vw;height:100vh}</style></head><body>
<div id=hud><b id=title>3D-схема</b>
<div id=scenetype style="font-size:15px;font-weight:700;color:#ffd479;margin-top:3px"></div>
<div class=leg>стены · <span style=color:#4f8cff>окна</span> · <span style=color:#e0a14f>двери</span> · <span style=color:#e05a4a>лестница</span> · ЛКМ/колесо · <b>F</b> вписать</div>
<div class=leg style=margin-top:6px>
 <label><input type=checkbox id=lstruct checked> Каркас</label> &nbsp;
 <label><input type=checkbox id=lzones checked> Зоны</label> &nbsp;
 <label><input type=checkbox id=lfurn checked> Мебель</label> &nbsp;
 <label><input type=checkbox id=llabels checked> Подписи</label> &nbsp;
 <label><input type=checkbox id=lapt checked> Квартиры</label></div>
<div id=info class=leg></div></div>
<div id=view></div>
<script type=importmap>{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three'; import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const JID="{{JID}}",FI="{{FI}}",BASE=location.pathname.slice(0,location.pathname.indexOf('/scene/'));
let renderer,scene,camera,controls;
function init(){const el=document.getElementById('view');
  renderer=new THREE.WebGLRenderer({antialias:true});renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.toneMapping=THREE.ACESFilmicToneMapping;el.appendChild(renderer.domElement);
  scene=new THREE.Scene();scene.background=new THREE.Color(0x141922);
  camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,1,40000);
  controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.maxPolarAngle=Math.PI*0.49;
  scene.add(new THREE.HemisphereLight(0xeaf0ff,0x2a3340,0.9));
  const sun=new THREE.DirectionalLight(0xfff4e6,1.5);sun.position.set(900,1700,700);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);
  const sc=3000;Object.assign(sun.shadow.camera,{left:-sc,right:sc,top:sc,bottom:-sc,far:9000});sun.shadow.bias=-0.0004;scene.add(sun);
  animate();}
function animate(){requestAnimationFrame(animate);controls&&controls.update();renderer&&renderer.render(scene,camera);}
addEventListener('resize',()=>{renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();if(_fit)fitView(_fit[0],_fit[1],_fit[2]);});
addEventListener('keydown',e=>{if((e.key==='f'||e.key==='F')&&_fit)fitView(_fit[0],_fit[1],_fit[2]);});
function label(t,x,z,color,size,y){const c=document.createElement('canvas'),g=c.getContext('2d');g.font='600 44px system-ui';c.width=Math.ceil(g.measureText(t).width)+36;c.height=64;g.font='600 44px system-ui';g.fillStyle='rgba(13,17,23,.82)';g.fillRect(0,0,c.width,c.height);g.fillStyle=color;g.textBaseline='middle';g.fillText(t,18,c.height/2+2);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),depthTest:false}));s.scale.set(size*c.width/c.height,size,1);s.position.set(x,y,z);s.renderOrder=999;(arguments[6]||scene).add(s);}
function median(a){if(!a.length)return 0;const b=a.slice().sort((x,y)=>x-y);return b[Math.floor(b.length/2)];}
const TYPE_COLORS={kitchen:0xe0a14f,bathroom:0x4fc3e0,bedroom:0x9a7fe0,'living':0x6fd08a,'salon':0x6fd08a,'dining':0xe07f9a,'hall':0x9aa7b8,'utility':0xb08968,'guest':0xc0a0e0,'closet':0xd0b070,'garderob':0xd0b070,'office':0x7f9ad0,'terrace':0x4fd0c0,'taras':0x4fd0c0,'other':0x8b97a7};
function typeColor(t){t=(t||'').toString().toLowerCase();for(const k in TYPE_COLORS)if(t.includes(k))return TYPE_COLORS[k];return TYPE_COLORS.other;}
function fixtureMesh(name,U){
  const n=(name||'').toLowerCase(),g=new THREE.Group();
  const wood=new THREE.MeshStandardMaterial({color:0xb98a5a,roughness:.7}),soft=new THREE.MeshStandardMaterial({color:0x8a93a3,roughness:.9}),metal=new THREE.MeshStandardMaterial({color:0xc8ccd2,roughness:.4,metalness:.6}),white=new THREE.MeshStandardMaterial({color:0xeef2f6,roughness:.5});
  const box=(w,h,dp,m,x,y,z)=>{const me=new THREE.Mesh(new THREE.BoxGeometry(w,h,dp),m);me.position.set(x||0,y||0,z||0);me.castShadow=true;me.receiveShadow=true;g.add(me);};
  if(n.includes('bed')){box(1.6*U,.5*U,2*U,soft,0,.25*U,0);box(1.6*U,.7*U,.15*U,wood,0,.45*U,-1*U);}
  else if(n.includes('sofa')||n.includes('диван')){box(2*U,.45*U,.9*U,soft,0,.22*U,0);box(2*U,.5*U,.2*U,soft,0,.5*U,-.35*U);}
  else if(n.includes('table')||n.includes('стол')){box(1.4*U,.08*U,.8*U,wood,0,.72*U,0);[[-.6,-.35],[.6,-.35],[-.6,.35],[.6,.35]].forEach(p=>box(.06*U,.72*U,.06*U,wood,p[0]*U,.36*U,p[1]*U));}
  else if(n.includes('stove')||n.includes('oven')||n.includes('плит')){box(.6*U,.9*U,.6*U,metal,0,.45*U,0);}
  else if(n.includes('fridge')||n.includes('refrig')||n.includes('холод')){box(.7*U,1.8*U,.7*U,metal,0,.9*U,0);}
  else if(n.includes('sink')||n.includes('мойк')){box(.6*U,.9*U,.5*U,white,0,.45*U,0);}
  else if(n.includes('bath')||n.includes('ванн')){box(1.7*U,.6*U,.75*U,white,0,.3*U,0);}
  else if(n.includes('toilet')||n.includes('унитаз')){box(.4*U,.4*U,.6*U,white,0,.2*U,0);box(.4*U,.5*U,.2*U,white,0,.45*U,-.2*U);}
  else if(n.includes('fireplace')||n.includes('камин')||n.includes('kominek')){box(1*U,1*U,.4*U,soft,0,.5*U,0);box(.6*U,.4*U,.12*U,new THREE.MeshStandardMaterial({color:0xff7733,emissive:0xff5500,emissiveIntensity:.7}),0,.3*U,.2*U);}
  else if(n.includes('chair')||n.includes('стул')||n.includes('armchair')||n.includes('кресл')){box(.6*U,.45*U,.6*U,soft,0,.22*U,0);box(.6*U,.5*U,.12*U,soft,0,.5*U,-.24*U);}
  else if(n.includes('wardrobe')||n.includes('cabinet')||n.includes('шкаф')||n.includes('garderob')||n.includes('closet')){box(1*U,1.8*U,.55*U,wood,0,.9*U,0);}
  else if(n.includes('counter')){box(2*U,.9*U,.6*U,wood,0,.45*U,0);}
  else {box(.5*U,.5*U,.5*U,soft,0,.25*U,0);}
  return g;
}
let G=null;
function applyLayers(){if(!G)return;const v=id=>{const e=document.getElementById(id);return e?e.checked:true;};
  G.struct.visible=v('lstruct');G.zones.visible=v('lzones');G.furn.visible=v('lfurn');G.labels.visible=v('llabels');if(G.apt)G.apt.visible=v('lapt');}
function build(d){
  G={struct:new THREE.Group(),zones:new THREE.Group(),furn:new THREE.Group(),labels:new THREE.Group(),apt:new THREE.Group()};
  scene.add(G.struct,G.zones,G.furn,G.labels,G.apt);
  const W=d.Width||1000,H=d.Height||1000,cx=W/2,cz=H/2,M=Math.max(W,H);
  const walls=[],doors=[],wins=[],TH=[];
  (d.points||[]).forEach((p,i)=>{const c=(d.classes||[])[i],k=(c&&c.name)?c.name:(typeof c==='string'?c:'wall');
    const x1=Math.min(p.x1,p.x2),x2=Math.max(p.x1,p.x2),y1=Math.min(p.y1,p.y2),y2=Math.max(p.y1,p.y2);
    const o={x1,y1,x2,y2,w:x2-x1,h:y2-y1};
    if(k==='wall'){walls.push(o);TH.push(Math.min(o.w,o.h));}else if(k==='window')wins.push(o);else doors.push(o);});
  let T=median(TH)||M*0.018; T=Math.min(Math.max(T,M*0.01),M*0.03);
  // wall bounding box in the viewer (W,H) frame — drives scale + room zones
  const _vs=d.wall_segments||[],_useV=_vs.length>=4,_vsx=W/(d.vec_width||W),_vsy=H/(d.vec_height||H);
  const _cb=d.cubi||{},_cbx=W/(_cb.Width||W),_cby=H/(_cb.Height||H);
  let BB=null;
  if(_cb.wall_paths&&_cb.wall_paths.length){let a=1e9,b=1e9,c=-1e9,e=-1e9;_cb.wall_paths.forEach(pa=>{(pa.p||pa).forEach(p=>{const X=p[0]*_cbx,Y=p[1]*_cby;a=Math.min(a,X);c=Math.max(c,X);b=Math.min(b,Y);e=Math.max(e,Y);});});BB={x0:a,y0:b,x1:c,y1:e};}
  else if(_cb.walls&&_cb.walls.length>=3){let a=1e9,b=1e9,c=-1e9,e=-1e9;_cb.walls.forEach(q=>q.forEach(p=>{const X=p[0]*_cbx,Y=p[1]*_cby;a=Math.min(a,X);c=Math.max(c,X);b=Math.min(b,Y);e=Math.max(e,Y);}));BB={x0:a,y0:b,x1:c,y1:e};}
  else if(_useV){let a=1e9,b=1e9,c=-1e9,e=-1e9;_vs.forEach(v=>{const X1=v[0]*_vsx,Y1=v[1]*_vsy,X2=v[2]*_vsx,Y2=v[3]*_vsy;a=Math.min(a,X1,X2);c=Math.max(c,X1,X2);b=Math.min(b,Y1,Y2);e=Math.max(e,Y1,Y2);});BB={x0:a,y0:b,x1:c,y1:e};}
  else if(walls.length){const xs=walls.flatMap(o=>[o.x1,o.x2]),ys=walls.flatMap(o=>[o.y1,o.y2]);BB={x0:Math.min.apply(0,xs),x1:Math.max.apply(0,xs),y0:Math.min.apply(0,ys),y1:Math.max.apply(0,ys)};}
  // scale: prefer REAL mm dimension numbers, else a door (~0.85 m), else fallback
  const dims=(d.dim_mm||[]).map(o=>o.mm).filter(n=>n>=1000&&n<=30000);
  const dspan=median(doors.map(o=>Math.max(o.w,o.h)));
  let mpp=0;
  if(dims.length&&BB){const longpx=Math.max(BB.x1-BB.x0,BB.y1-BB.y0),longm=Math.max.apply(0,dims)/1000;
    if(longpx>10&&longm>1){mpp=longm/longpx;mpp=Math.min(Math.max(mpp,0.002),0.05);}}
  if(!mpp)mpp=(dspan&&dspan>4)?0.85/dspan:(2.7/(M*0.11));
  let scaleSrc=(dims.length&&BB)?'размеры (мм)':(dspan&&dspan>4?'дверь':'оценка');
  // BEST calibration on multi-unit plans: the plan PRINTS each apartment's m². Match the
  // printed area to the apartment polygon it sits in -> mpp from real areas (also fixes labels).
  const _ap0=(d.cubi&&d.cubi.apartments)||[];
  if(_ap0.length&&(d.area_labels||[]).length){
    const _ax=W/(d.cubi.Width||W),_ay=H/(d.cubi.Height||H),_ox=W/(d.ocr_width||W),_oy=H/(d.ocr_height||H);
    const labs=(d.area_labels||[]).map(a=>({v:a.value,x:a.x*_ox,y:a.y*_oy}));
    let sm2=0,spx=0,nm=0;
    _ap0.forEach(ap=>{const pg=(ap.polygon||[]).map(p=>[p[0]*_ax,p[1]*_ay]);if(pg.length<3)return;
      let best=0;labs.forEach(l=>{if(pointInPoly(l.x,l.y,pg))best=Math.max(best,l.v);});
      if(best>=8){sm2+=best;spx+=ap.area_px*_ax*_ay;nm++;ap._m2=best;}});
    if(nm>=2&&spx>0){const cal=Math.sqrt(sm2/spx);if(cal>=0.002&&cal<=0.05){mpp=cal;scaleSrc='площади квартир';}}
  }
  // AUTHORITATIVE: the server-side understanding stage (dimension chains + areas, outlier-
  // rejected) gives the most reliable scale — prefer it when confident.
  const _u=d.understand&&d.understand.scale;
  const _SM={dim_chain:'размерные цепочки',area_vs_polygon:'площади (анализ)',vlm_overall_dim:'габариты (VLM)'};
  if(_u&&_u.mpp&&_u.mpp>=0.002&&_u.mpp<=0.05){mpp=_u.mpp;scaleSrc=_SM[_u.method]||'анализ';}
  // GEOMETRIC clamp: a wall is short relative to the floor footprint. Cap WALL_H to a
  // fraction of the plan extent so a wrong scale can NEVER produce a tower/pillar forest.
  let WALL_H=2.7/mpp;
  if(BB){const ext=Math.max(BB.x1-BB.x0,BB.y1-BB.y0);WALL_H=Math.min(WALL_H,ext*0.09);WALL_H=Math.max(WALL_H,ext*0.025);}
  const SILL=WALL_H*0.33,LINTEL=WALL_H*0.78;
  const slab=new THREE.Mesh(new THREE.BoxGeometry(W*1.08,T*0.6,H*1.08),new THREE.MeshStandardMaterial({color:0x3a414e,roughness:.95}));
  slab.position.y=-T*0.3;slab.receiveShadow=true;G.struct.add(slab);
  G.struct.add(new THREE.GridHelper(M*1.08,30,0x4a525f,0x2c323c));
  // merge wall boxes -> clean joined segments (quantize centerline + union intervals)
  const q=T*0.6,hG={},vG={};
  walls.forEach(o=>{if(o.w>=o.h){const y=Math.round(((o.y1+o.y2)/2)/q)*q;(hG[y]=hG[y]||[]).push([o.x1,o.x2]);}
    else{const x=Math.round(((o.x1+o.x2)/2)/q)*q;(vG[x]=vG[x]||[]).push([o.y1,o.y2]);}});
  const mergeIv=(arr,gap)=>{arr.sort((a,b)=>a[0]-b[0]);const out=[];arr.forEach(iv=>{const L=out[out.length-1];if(L&&iv[0]<=L[1]+gap)L[1]=Math.max(L[1],iv[1]);else out.push([iv[0],iv[1]]);});return out;};
  const segs=[];for(const y in hG)mergeIv(hG[y],T*1.3).forEach(iv=>segs.push({dir:'h',c:+y,a:iv[0],b:iv[1]}));
  for(const x in vG)mergeIv(vG[x],T*1.3).forEach(iv=>segs.push({dir:'v',c:+x,a:iv[0],b:iv[1]}));
  const matWall=new THREE.MeshStandardMaterial({color:0xe9ecf2,roughness:.9}),
        matWin=new THREE.MeshPhysicalMaterial({color:0xbfe0ff,roughness:.05,transmission:.85,transparent:true,opacity:.5,ior:1.5,thickness:T}),
        matSill=new THREE.MeshStandardMaterial({color:0xd7dbe2,roughness:.7}),
        matStair=new THREE.MeshStandardMaterial({color:0xe05a4a,roughness:.5});
  function openingsOn(s){const list=[];
    doors.forEach(o=>{const ox=(o.x1+o.x2)/2,oy=(o.y1+o.y2)/2;
      if(s.dir==='h'&&Math.abs(oy-s.c)<=T*1.8&&ox>=s.a-T&&ox<=s.b+T)list.push({k:'door',t:Math.max(s.a,ox-o.w/2),u:Math.min(s.b,ox+o.w/2)});
      if(s.dir==='v'&&Math.abs(ox-s.c)<=T*1.8&&oy>=s.a-T&&oy<=s.b+T)list.push({k:'door',t:Math.max(s.a,oy-o.h/2),u:Math.min(s.b,oy+o.h/2)});});
    wins.forEach(o=>{const ox=(o.x1+o.x2)/2,oy=(o.y1+o.y2)/2;
      if(s.dir==='h'&&Math.abs(oy-s.c)<=T*1.8&&ox>=s.a-T&&ox<=s.b+T)list.push({k:'win',t:Math.max(s.a,ox-o.w/2),u:Math.min(s.b,ox+o.w/2)});
      if(s.dir==='v'&&Math.abs(ox-s.c)<=T*1.8&&oy>=s.a-T&&oy<=s.b+T)list.push({k:'win',t:Math.max(s.a,oy-o.h/2),u:Math.min(s.b,oy+o.h/2)});});
    return list.filter(o=>o.u>o.t+2).sort((a,b)=>a.t-b.t);}
  function piece(s,t,u,y0,y1,mat){const L=u-t;if(L<=1||y1-y0<=1)return;
    const m=new THREE.Mesh(new THREE.BoxGeometry(s.dir==='h'?L:T,y1-y0,s.dir==='h'?T:L),mat);const mid=(t+u)/2;
    m.position.set((s.dir==='h'?mid:s.c)-cx,(y0+y1)/2,(s.dir==='h'?s.c:mid)-cz);m.castShadow=true;m.receiveShadow=true;G.struct.add(m);}
  // helpers for polygon geometry (CubiCasa coords)
  function polyCentroid(pl){let x=0,y=0;pl.forEach(p=>{x+=p[0];y+=p[1];});return [x/pl.length,y/pl.length];}
  function pointInPoly(px,pz,pl){let c=false;for(let i=0,j=pl.length-1;i<pl.length;j=i++){const xi=pl[i][0],yi=pl[i][1],xj=pl[j][0],yj=pl[j][1];
    if(((yi>pz)!=(yj>pz))&&(px<(xj-xi)*(pz-yi)/(yj-yi)+xi))c=!c;}return c;}
  // ===== PRIMARY geometry: CubiCasa5k neural parser (vector walls, doors, windows) =====
  const cubi=d.cubi||{};const cubiW=cubi.Width||W,cubiH=cubi.Height||H,csx=W/cubiW,csy=H/cubiH;
  const useCubi=!!(cubi.walls&&cubi.walls.length>=3);
  // ===== fallback: angle-preserving wall centre-lines (Hough vectorizer) =====
  const vseg=d.wall_segments||[];const useVec=!useCubi&&vseg.length>=4;
  if(useCubi){
    const matDoor=new THREE.MeshStandardMaterial({color:0xc69a6b,roughness:.7});
    const wpaths=cubi.wall_paths||[],cavg=(csx+csy)/2;
    const matBear=matWall,matPart=new THREE.MeshStandardMaterial({color:0xc7ccd6,roughness:.92});
    if(wpaths.length){
      // ANY-SHAPE walls: medial-axis centre-lines -> overlapping oriented boxes of LOCAL
      // thickness. Load-bearing (thick) vs partition (thin) shown distinctly; partitions
      // a touch lower so the load-bearing structure reads at a glance.
      wpaths.forEach(pa=>{const pts=pa.p||pa,bearing=(pa.c!=='partition');
        const mat=bearing?matBear:matPart,wh=bearing?WALL_H:WALL_H*0.82;
        // ONE uniform thickness per wall (no bulging at junctions)
        let th=((pa.t!=null?pa.t:6)*cavg);th=Math.min(Math.max(th,M*0.005),M*0.05);
        for(let i=0;i<pts.length-1;i++){
          const x1=pts[i][0]*csx-cx,z1=pts[i][1]*csy-cz,x2=pts[i+1][0]*csx-cx,z2=pts[i+1][1]*csy-cz;
          const dx=x2-x1,dz=z2-z1,L=Math.hypot(dx,dz);if(L<0.5)continue;
          const m=new THREE.Mesh(new THREE.BoxGeometry(L+th,wh,th),mat);
          m.position.set((x1+x2)/2,wh/2,(z1+z2)/2);m.rotation.y=-Math.atan2(dz,dx);
          m.castShadow=true;m.receiveShadow=true;G.struct.add(m);}
        pts.forEach(p=>{const j=new THREE.Mesh(new THREE.CylinderGeometry(th/2,th/2,wh,8),mat);
          j.position.set(p[0]*csx-cx,wh/2,p[1]*csy-cz);j.castShadow=true;G.struct.add(j);});});
    } else {
      (cubi.walls||[]).forEach(qd=>{if(!qd||qd.length<3)return;
        const sh=new THREE.Shape();qd.forEach((p,i)=>{const X=p[0]*csx-cx,Z=-(p[1]*csy-cz);i?sh.lineTo(X,Z):sh.moveTo(X,Z);});
        const g=new THREE.ExtrudeGeometry(sh,{depth:WALL_H,bevelEnabled:false});g.rotateX(-Math.PI/2);
        const m=new THREE.Mesh(g,matWall);m.castShadow=true;m.receiveShadow=true;G.struct.add(m);});
    }
    (cubi.openings||[]).forEach(o=>{if(!o.polygon||o.polygon.length<3)return;const c=polyCentroid(o.polygon),X=c[0]*csx-cx,Z=c[1]*csy-cz;
      const sz=Math.max(M*0.022,Math.hypot((o.polygon[0][0]-o.polygon[2][0])*csx,(o.polygon[0][1]-o.polygon[2][1])*csy)*0.6);
      if(o.class==='Window'){const m=new THREE.Mesh(new THREE.BoxGeometry(sz,WALL_H*0.5,Math.max(M*0.01,sz*0.35)),matWin);m.position.set(X,SILL+WALL_H*0.25,Z);G.struct.add(m);}
      else{const m=new THREE.Mesh(new THREE.BoxGeometry(sz,WALL_H*0.92,Math.max(M*0.01,sz*0.3)),matDoor);m.position.set(X,WALL_H*0.46,Z);G.struct.add(m);}});
    (cubi.fixtures||[]).forEach(f=>{if(!f.polygon||f.polygon.length<3)return;const c=polyCentroid(f.polygon);
      const fm=fixtureMesh(f.class,WALL_H*0.6);fm.position.set(c[0]*csx-cx,0,c[1]*csy-cz);G.furn.add(fm);});
  } else if(useVec){
    const sx=W/(d.vec_width||W),sy=H/(d.vec_height||H),s2=(sx+sy)/2;
    let WT=(d.wall_thickness_px||0)*s2*1.3;WT=Math.min(Math.max(WT||T,M*0.008),M*0.04);
    vseg.forEach(v=>{const x1=v[0]*sx,y1=v[1]*sy,x2=v[2]*sx,y2=v[3]*sy;
      const dx=x2-x1,dz=y2-y1,L=Math.hypot(dx,dz);if(L<M*0.012)return;
      const wall=new THREE.Mesh(new THREE.BoxGeometry(L+WT,WALL_H,WT),matWall);
      wall.position.set((x1+x2)/2-cx,WALL_H/2,(y1+y2)/2-cz);
      wall.rotation.y=-Math.atan2(dz,dx);wall.castShadow=true;wall.receiveShadow=true;G.struct.add(wall);});
    // doors/windows as overlay markers (their boxes are axis-aligned, can't be cut into angled walls)
    doors.forEach(o=>{const m=new THREE.Mesh(new THREE.BoxGeometry(Math.max(o.w,T*1.2),WALL_H*0.04,Math.max(o.h,T*1.2)),matStair);
      m.position.set((o.x1+o.x2)/2-cx,WALL_H*0.02,(o.y1+o.y2)/2-cz);G.struct.add(m);});
    wins.forEach(o=>{const m=new THREE.Mesh(new THREE.BoxGeometry(Math.max(o.w,T),WALL_H*0.5,Math.max(o.h,T)),matWin);
      m.position.set((o.x1+o.x2)/2-cx,SILL+WALL_H*0.2,(o.y1+o.y2)/2-cz);G.struct.add(m);});
  } else {
    segs.forEach(s=>{const ops=openingsOn(s);let cur=s.a;
      ops.forEach(op=>{piece(s,cur,op.t,0,WALL_H,matWall);piece(s,op.t,op.u,LINTEL,WALL_H,matWall);
        if(op.k==='win'){piece(s,op.t,op.u,0,SILL,matSill);piece(s,op.t,op.u,SILL,LINTEL,matWin);}cur=op.u;});
      piece(s,cur,s.b,0,WALL_H,matWall);});
  }
  // ---- REAL rooms: name-based (filtered) OR numeric area-label anchors -------
  const vById={},vlist=(d.vlm_rooms||[]).slice();
  vlist.forEach(r=>{if(r.name)vById[r.name.toString().toLowerCase().trim()]=r;});
  const ROOMKW=['кімнат','кухн','ванн','спальн','вітальн','лоджі','салон','гостин','коридор','прихож','санвузол','санузел','туалет','балкон','гардероб','каб','kitchen','bedroom','bathroom','living','dining','hall','office','room','salon','salonik','kuchnia','sypialnia','lazienka','garderoba','goscinny','pom','taras','jadalnia'];
  const NOISE=/розмір|^н[\s.]|ст\.|вікн|підвік|відкр|рами|отвор|отв|вент|каналіз|електрощ|умовн|познач|примітк|загальн|експлік|soprano|славут|про[єе]кт|план|d=|мм|канал|вивід|вивод|hove|ekc|kohoa|васот|висот|рамих/i;
  function isRoom(nm){const s=(nm||'').toString().toLowerCase().trim();if(s.length<3)return false;if(NOISE.test(s))return false;if(vById[s])return true;return ROOMKW.some(k=>s.includes(k));}
  // each realRoom carries its own VLM data in ._v, a centre (._cx,._cz) and optional
  // floor polygon ._poly — ALL in the viewer W,H frame.
  const CUBI_KEY={Kitchen:'kitchen',Bath:'bathroom','Living Room':'living','Bed Room':'bedroom',
                  Entry:'hall',Storage:'utility',Garage:'utility',Outdoor:'terrace'};
  let realRooms=[];
  const osx=W/(d.ocr_width||W),osy=H/(d.ocr_height||H);
  // PRIORITY 1 — CubiCasa room polygons; assign area+type by the area anchor inside each.
  if(useCubi&&(cubi.rooms||[]).length){
    const anchorsW=(d.area_labels||[]).map(a=>({v:a.value,x:a.x*osx,y:a.y*osy,used:false}));
    const pool=vlist.slice();
    realRooms=(cubi.rooms||[]).map(rm=>{
      const poly=(rm.polygon||[]).map(p=>[p[0]*csx,p[1]*csy]);
      if(poly.length<3)return null;
      const c=polyCentroid(poly);
      let area=null;for(const a of anchorsW){if(!a.used&&pointInPoly(a.x,a.y,poly)){a.used=true;area=a.v;break;}}
      let v={};
      if(area!=null){let bi=-1,bd=1e9;pool.forEach((vv,i)=>{const av=parseFloat(vv.area_m2);if(isFinite(av)){const dd=Math.abs(av-area);if(dd<bd){bd=dd;bi=i;}}});
        if(bi>=0&&bd<=Math.max(2.5,area*0.25)){v=Object.assign({},pool[bi]);pool.splice(bi,1);}}
      if(area!=null&&v.area_m2==null)v.area_m2=area;
      if(!v.type&&rm.class)v.type=CUBI_KEY[rm.class]||rm.class.toLowerCase();
      const nm=(v.name&&v.name.length>1)?v.name:(area!=null?area+'м²':(rm.class||'комната'));
      return {name:nm,area:area!=null?area+'м²':'',_v:v,_cx:c[0],_cz:c[1],_poly:poly};
    }).filter(Boolean);}
  // PRIORITY 2 — name-based OCR rooms
  if(!realRooms.length)realRooms=(d.rooms||[]).filter(r=>isRoom(r.name)).map(r=>({name:r.name,area:r.area,
    _v:vById[(r.name||'').toString().toLowerCase().trim()]||{},_cx:r.x+r.w/2,_cz:r.y+r.h/2}));
  // PRIORITY 3 — numeric area-label anchors + geodesic segmentation polygons
  const segW=d.seg_width||W,segH=d.seg_height||H,gsx=W/segW,gsy=H/segH;
  const geomById={};(d.room_geom||[]).forEach(g=>{geomById[g.id]=g;});
  if(!realRooms.length&&(d.area_labels||[]).length){
    const pool=vlist.slice();
    realRooms=(d.area_labels||[]).map((a,idx)=>{
      let bi=-1,bd=1e9;pool.forEach((v,i)=>{const av=parseFloat(v.area_m2);if(isFinite(av)){const dd=Math.abs(av-a.value);if(dd<bd){bd=dd;bi=i;}}});
      let v={area_m2:a.value};if(bi>=0&&bd<=Math.max(2.5,a.value*0.25)){v=pool[bi];pool.splice(bi,1);}
      const nm=(v.name&&v.name.length>1)?v.name:(v.type||'комната');
      const gm=geomById[idx];
      const cxp=gm?gm.cx*gsx:a.x*osx, czp=gm?gm.cy*gsy:a.y*osy;
      return {name:nm,area:a.value+'м²',_v:v,_cx:cxp,_cz:czp,_poly:gm?gm.polygon.map(p=>[p[0]*gsx,p[1]*gsy]):null};});}
  // ---- zones: TRUE room-region polygons (respect walls) when segmented, else Voronoi ----
  const haveGeom=realRooms.some(r=>r._poly&&r._poly.length>=3);
  if(haveGeom){
    realRooms.forEach(r=>{if(!(r._poly&&r._poly.length>=3))return;
      const col=typeColor((r._v&&r._v.type)||r.name);
      const sh=new THREE.Shape();r._poly.forEach((p,i)=>{const X=p[0]-cx,Z=p[1]-cz;if(i===0)sh.moveTo(X,Z);else sh.lineTo(X,Z);});
      const gm=new THREE.ShapeGeometry(sh);gm.rotateX(Math.PI/2);   // XY shape -> XZ floor plane
      const fl=new THREE.Mesh(gm,new THREE.MeshStandardMaterial({color:col,transparent:true,opacity:.42,roughness:.9,side:THREE.DoubleSide}));
      fl.position.y=1;fl.receiveShadow=true;G.zones.add(fl);});
  } else if(realRooms.length&&BB){
    const step=Math.max(BB.x1-BB.x0,BB.y1-BB.y0)/55,tiles={};
    for(let X=BB.x0;X<BB.x1;X+=step)for(let Y=BB.y0;Y<BB.y1;Y+=step){let bi=-1,bd=1e18;
      realRooms.forEach((r,ri)=>{const dx=r._cx-(X+step/2),dy=r._cz-(Y+step/2),dd=dx*dx+dy*dy;if(dd<bd){bd=dd;bi=ri;}});
      if(bi>=0)(tiles[bi]=tiles[bi]||[]).push([X+step/2-cx,Y+step/2-cz]);}
    Object.keys(tiles).forEach(ri=>{const r=realRooms[ri],col=typeColor((r._v.type)||r.name),cells=tiles[ri];
      const im=new THREE.InstancedMesh(new THREE.BoxGeometry(step*0.98,1,step*0.98),new THREE.MeshStandardMaterial({color:col,transparent:true,opacity:.3,roughness:.9}),cells.length);
      const m4=new THREE.Matrix4();cells.forEach((c,i)=>{m4.makeTranslation(c[0],1,c[1]);im.setMatrixAt(i,m4);});im.instanceMatrix.needsUpdate=true;G.zones.add(im);});}
  // ---- labels ----
  realRooms.forEach(r=>{const lx=r._cx-cx,lz=r._cz-cz,v=r._v||{};
    label((r.name||'')+(v.type&&v.type!=r.name?' · '+v.type:''),lx,lz,'#ffffff',M*0.038,WALL_H+M*0.05,G.labels);
    const sub=[(v.area_m2?v.area_m2+'м²':(r.area||'')),(v.dimensions||'')].filter(Boolean).join(' · ');
    if(sub)label(sub,lx,lz,'#7fd6a0',M*0.03,WALL_H+M*0.012,G.labels);});
  // ---- furniture from VLM fixtures, placed in matched rooms ----
  let U=1/mpp; if(BB){U=Math.min(U,Math.max(BB.x1-BB.x0,BB.y1-BB.y0)*0.03);}  // clamp size
  realRooms.forEach(r=>{const v=r._v||{},fx=v.fixtures||[];
    fx.slice(0,6).forEach((f,fi)=>{const fm=fixtureMesh(f,U*0.5);const ang=fi/Math.max(fx.length,1)*6.283,rad=U*0.7;
      fm.position.set(r._cx-cx+Math.cos(ang)*rad,0,r._cz-cz+Math.sin(ang)*rad);G.furn.add(fm);});});
  (d.stairs_pts||[]).forEach(s=>{const lx=s.x+s.w/2-cx,lz=s.y+s.h/2-cz;
    const st=new THREE.Mesh(new THREE.BoxGeometry(M*.05,WALL_H*.5,M*.05),matStair);st.position.set(lx,WALL_H*.25,lz);st.castShadow=true;G.struct.add(st);label('лестница',lx,lz,'#ff8a7a',M*.04,WALL_H*.7,G.labels);});
  // ---- APARTMENTS layer (multi-unit floor): tinted footprint + outline + label ----
  const apts=(cubi.apartments)||[];
  const APTC=[0xe06666,0x6fbf73,0x6f9bff,0xe0c24f,0xc06fe0,0x4fd0d0,0xe0934f,0x9ad06f,0xd06f9a,0x6fd0a0,0xb0884f,0x8f9fd0,0xd0b0d0,0x70c0c0,0xc0c070,0xd07070,0x70a0d0,0xa0d070];
  const aInfo=((d.apartments_info||{}).apartments)||[];
  if(apts.length){
    const wt=Math.max(M*0.008,WALL_H*0.12);
    apts.forEach((ap,k)=>{const pg=(ap.polygon||[]).map(p=>[p[0]*csx,p[1]*csy]);if(pg.length<3)return;
      const col=APTC[k%APTC.length];
      const sh=new THREE.Shape();pg.forEach((p,i)=>{const X=p[0]-cx,Z=p[1]-cz;i?sh.lineTo(X,Z):sh.moveTo(X,Z);});
      const fg=new THREE.ShapeGeometry(sh);fg.rotateX(Math.PI/2);
      const fl=new THREE.Mesh(fg,new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.22,side:THREE.DoubleSide}));
      fl.position.y=M*0.003;G.apt.add(fl);                       // floor tint at ground
      // boundary WALLS from the apartment contour (low, readable)
      for(let i=0;i<pg.length;i++){const a=pg[i],bb=pg[(i+1)%pg.length];
        const x1=a[0]-cx,z1=a[1]-cz,x2=bb[0]-cx,z2=bb[1]-cz,dx=x2-x1,dz=z2-z1,L=Math.hypot(dx,dz);if(L<M*0.008)continue;
        const wm=new THREE.Mesh(new THREE.BoxGeometry(L+wt,WALL_H,wt),matWall);
        wm.position.set((x1+x2)/2,WALL_H/2,(z1+z2)/2);wm.rotation.y=-Math.atan2(dz,dx);wm.castShadow=true;wm.receiveShadow=true;G.struct.add(wm);}
      const inf=aInfo[k]||{};
      const aM2=inf.area_m2||(ap._m2)||(ap.area_px*csx*csy*mpp*mpp);
      const rc=inf.rooms_count?(' · '+inf.rooms_count+'к'):'';
      label('Кв. '+(k+1)+(aM2>4?' · '+aM2.toFixed(1)+'м²':'')+rc,ap.cx*csx-cx,ap.cy*csy-cz,'#ffe08a',M*0.045,WALL_H+M*0.09,G.apt);});
  }
  // ---- SCENE-TYPE understanding (what is this object?) — universal VLM category ----
  (function(){const aptn=apts.length;let cat=d.category_label;
    if(!cat){const bt=(d.building_type||'').toString().toLowerCase();
      if(aptn>=3)cat='🏢 Этаж многоквартирного дома';
      else if(/office/.test(bt))cat='🏢 Офисное помещение';
      else if(/retail|commercial|shop|store|mall/.test(bt))cat='🏬 Коммерческая недвижимость';
      else if(/warehouse|industrial/.test(bt))cat='🏭 Склад / производство';
      else if(/house|townhouse|detached|cottage|villa/.test(bt))cat='🏡 Дом';
      else if(/apartment|flat|studio/.test(bt)||realRooms.length)cat='🏠 Квартира';
      else cat='🏠 Объект';}
    if(aptn>=3)cat+=' · '+aptn+' кв.';
    const tot=aptn>=3?apts.reduce((s,a)=>s+(a._m2||a.area_px*csx*csy*mpp*mpp),0):0;
    if(tot>10)cat+=' · ~'+Math.round(tot)+' м² жилой';
    const u=d.understand;
    if(u&&u.scale&&u.scale.method)cat+=' · масштаб: '+({dim_chain:'размеры ✓',area_vs_polygon:'площади',vlm_overall_dim:'габариты'}[u.scale.method]||u.scale.method);
    if(u&&u.north)cat+=' · 🧭 '+u.north;
    if(u&&u.needs_review)cat+=' · ⚠ проверить';
    const el=document.getElementById('scenetype');if(el)el.textContent=cat;})();
  fitView(W,H,WALL_H);applyLayers();
  const longm=BB?Math.max(BB.x1-BB.x0,BB.y1-BB.y0)*mpp:M*mpp;
  let info;
  if(useCubi){const wpz=cubi.wall_paths||[];const nb=wpz.filter(p=>(p.c||'bearing')!=='partition').length,np=wpz.length-nb;
    const wn=wpz.length?(nb+' несущих + '+np+' перегородок'):cubi.walls.length+' стен';
    const aptn=(cubi.apartments||[]).length;
    info=(aptn>1?'🏠 '+aptn+' квартир · ':'')+wn+' · '+(cubi.openings||[]).filter(o=>o.class==='Door').length+' дв · '+(cubi.openings||[]).filter(o=>o.class==='Window').length+' ок · '+(cubi.fixtures||[]).length+' сантех/меб · ~'+longm.toFixed(1)+'м ('+scaleSrc+') · CubiCasa5k+медиаль';}
  else{const wallN=useVec?vseg.length:segs.length;info=wallN+(useVec?' стен (векторные)':' стен')+' · '+doors.length+' дв · '+wins.length+' ок · комнат '+realRooms.length+' · ~'+longm.toFixed(1)+'м ('+scaleSrc+')';}
  document.getElementById('info').textContent=info;}
let _fit=null;
function fitView(W,H,WALL_H){_fit=[W,H,WALL_H];
  const cy=WALL_H/2,maxDim=Math.max(W,H,WALL_H),fov=camera.fov*Math.PI/180;
  let dist=(maxDim/2)/Math.tan(fov/2);
  const fovH=2*Math.atan(Math.tan(fov/2)*camera.aspect),distH=(W/2)/Math.tan(fovH/2);
  dist=Math.max(dist,distH)*1.3;
  camera.position.set(dist*0.55,cy+dist*0.85,dist*0.55);
  camera.near=Math.max(dist/300,0.1);camera.far=dist*300;camera.updateProjectionMatrix();
  controls.target.set(0,cy,0);controls.update();}
init();
['lstruct','lzones','lfurn','llabels','lapt'].forEach(id=>{const e=document.getElementById(id);if(e)e.addEventListener('change',applyLayers);});
fetch(BASE+'/api/scene/'+JID+'/'+FI).then(r=>r.json()).then(d=>{document.getElementById('title').textContent='3D-схема · '+(d.floor_name||'этаж');build(d);}).catch(e=>{document.getElementById('info').textContent='ошибка загрузки сцены: '+e;});
</script></body></html>"""

LIBRARY_PAGE = r"""<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Библиотека</title>
<style>body{margin:0;background:#0e1116;color:#e6edf3;font:14px system-ui;max-width:1080px;margin:auto;padding:18px}
a{color:#4f8cff}h1{font-size:19px}.proj{background:#171c24;border:1px solid #222a35;border-radius:12px;padding:14px;margin-bottom:14px}
.head{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}.chip{background:#0b0e13;border:1px solid #222a35;border-radius:8px;padding:4px 10px;font-size:12px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px}.g a{display:block;border:1px solid #222a35;border-radius:8px;overflow:hidden;position:relative}
.g img{width:100%;display:block;aspect-ratio:3/2;object-fit:cover}.g span{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.6);font-size:10px;padding:1px 4px}
.mut{color:#8b97a7;font-size:12px}.s3d{margin:6px 0}.s3d a{margin-right:8px}
.objgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px}
.objcard{background:#171c24;border:1px solid #222a35;border-radius:8px;overflow:hidden;font-size:11px}
.objcard img{width:100%;display:block;aspect-ratio:1;object-fit:cover;background:#0b0e13}
.objcard .noimg{width:100%;aspect-ratio:1;background:#0b0e13}
.objcard .on{font-weight:600;padding:4px 6px 0}.objcard .oa{color:#8b97a7;padding:0 6px 4px}
#objsearch{width:100%;margin:8px 0;padding:8px 10px;background:#0b0e13;border:1px solid #222a35;color:#e6edf3;border-radius:8px;box-sizing:border-box}
table.rt{width:100%;border-collapse:collapse;font-size:12px;margin:6px 0}table.rt th,table.rt td{border:1px solid #222a35;padding:4px 7px;text-align:left;vertical-align:top}table.rt th{color:#8b97a7;font-weight:600}
details.sec{margin:8px 0;border:1px solid #222a35;border-radius:8px;padding:6px 10px;background:#0e1116}details.sec>summary{cursor:pointer;font-weight:600;color:#cdd6e0}
.aptchip{background:#3a2f1a;color:#ffd479;border:1px solid #5a4a2a;border-radius:8px;padding:4px 9px;font-size:12px;margin:2px;display:inline-block}
.invrow{display:flex;gap:8px;align-items:flex-start;padding:5px 0;border-bottom:1px solid #1c232d}.invrow img{width:56px;height:56px;object-fit:cover;border-radius:6px;background:#0b0e13;flex:none}
.desc{color:#aeb8c4;font-size:12px;font-style:italic}</style></head><body>
<h1>📚 Библиотека проектов и рендеров</h1>
<div class=mut style=margin-bottom:12px><a href="./">← к очереди</a></div>
<div id=list class=mut>загрузка…</div>
<h1 style=margin-top:26px>🗄 Библиотека объектов <span id=objcount class=mut></span></h1>
<input id=objsearch placeholder="поиск: диван, дуб, белый, кухня…">
<div id=objgrid class=objgrid><div class=mut>загрузка…</div></div>
<script>
const BASE=location.pathname.replace(/\/library$/,'');
const esc=s=>(s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;');
const T={project:'Проект',interior:'Проект по комнатам',reference:'Фотореализация',render:'Рендер',furnish:'Меблировка'};
function fmt(s){if(s==null)return '';if(s<60)return s+'с';return Math.floor(s/60)+'м '+(s%60)+'с';}
fetch(BASE+'/api/library').then(r=>r.json()).then(d=>{
  if(!d.items.length){document.getElementById('list').textContent='Пока пусто — собери проект в очереди.';return;}
  renderLib(d);
}).catch(e=>{document.getElementById('list').textContent='ошибка: '+e;});
function renderLib(d){
  document.getElementById('list').innerHTML=d.items.map(j=>{
    const p=j.project||{}, imgs=j.images||[], invs=j.inventories||[];
    const floors=(p.floors||[]).map(f=>f.scene!=null?`<a href="${BASE}/scene/${j.id}/${f.scene}" target=_blank>🧊 3D: ${f.name}</a>`:'').join('');
    const g=imgs.map(im=>`<a href="${BASE}/api/jobs/${j.id}/room/${im.idx}" target=_blank><img loading=lazy src="${BASE}/api/jobs/${j.id}/room/${im.idx}"><span>${esc(im.room||'')}</span></a>`).join('')
      || (j.result?`<a href="${BASE}/api/jobs/${j.id}/result" target=_blank><img src="${BASE}/api/jobs/${j.id}/result"></a>`:'');
    const apt=p.apartments_total?`<span class=chip style="background:#3a2f1a;color:#ffd479">🏢 ${p.apartments_total} квартир</span>${p.living_area_m2?'<span class=chip>'+p.living_area_m2+' м² жилой</span>':''}`:'';
    const stats=p.building_type?`<span class=chip>${esc(p.building_type)}</span>${apt}<span class=chip>этажей ${p.levels||1}</span><span class=chip>комнат ${p.rooms_total||imgs.length}</span>${p.total_area_m2?'<span class=chip>'+p.total_area_m2+' м²</span>':''}`:`<span class=chip>${imgs.length||1} рендер(ов)</span>`;
    // apartments breakdown
    const aptBlock=(p.apartments&&p.apartments.length)?`<details class=sec open><summary>🏢 Квартиры (${p.apartments.length})</summary><div>${p.apartments.map(a=>`<span class=aptchip>Кв. ${a.id}${a.floor&&p.levels>1?' · '+esc(a.floor):''}${a.area_m2?' · '+a.area_m2+' м²':''}${a.rooms_count?' · '+a.rooms_count+'к':''}${(a.rooms&&a.rooms.length)?' ('+a.rooms.join(', ')+')':''}</span>`).join('')}</div></details>`:'';
    // full per-room records
    const rooms=p.rooms||[];
    const roomBlock=rooms.length?`<details class=sec open><summary>📐 Комнаты — полные данные (${rooms.length})</summary>
      <table class=rt><tr><th>Комната</th><th>Тип</th><th>Площадь</th><th>Размеры</th><th>Окна</th><th>Объекты в комнате</th><th>Описание</th></tr>
      ${rooms.map(r=>`<tr><td><b>${esc(r.name)}</b>${p.levels>1&&r.floor?'<div class=mut>'+esc(r.floor)+'</div>':''}</td><td>${esc(r.type||'')}</td><td>${r.area_m2?r.area_m2+' м²':'—'}</td><td>${esc(r.dimensions||'—')}</td><td>${r.windows!=null?r.windows:'—'}</td><td>${esc((Array.isArray(r.fixtures)?r.fixtures.join(', '):r.fixtures)||'—')}</td><td class=desc>${esc(r.description||'')}</td></tr>`).join('')}</table></details>`:'';
    // per-photo inventory (all identified objects + materials)
    const invBlock=invs.length?invs.map(iv=>{
      const objs=iv.objects||[],m=iv.materials||{};
      const ms=v=>!v?'?':(typeof v==='object'?[v.material,v.color,v.condition].filter(Boolean).join(' '):v);
      const rows=objs.map((o,oi)=>{
        const th=(o.crop!=null)?`<a href="${BASE}/api/jobs/${j.id}/inventory/${iv._pj}/object/${o.crop}" target=_blank><img loading=lazy src="${BASE}/api/jobs/${j.id}/inventory/${iv._pj}/object/${o.crop}"></a>`:'<div style="width:56px;height:56px;background:#0b0e13;border-radius:6px;flex:none"></div>';
        return `<div class=invrow>${th}<div><b>${esc(o.name)}</b>${o.qty>1?' ×'+o.qty:''}<div class=mut>${[o.category,o.material,o.color,o.condition].filter(Boolean).map(esc).join(' · ')}</div>${o.note?'<div class=desc>'+esc(o.note)+'</div>':''}</div></div>`;
      }).join('');
      const mat=(m.floor||m.walls||m.ceiling)?`<div class=mut style=margin:4px_0>Материалы: пол — ${esc(ms(m.floor))}, стены — ${esc(ms(m.walls))}, потолок — ${esc(ms(m.ceiling))}</div>`:'';
      return `<details class=sec open><summary>📋 Опись фото «${esc(iv.room_type||iv.room||('фото '+iv._pj))}» — ${objs.length} объектов</summary>${iv.summary?'<div class=desc>'+esc(iv.summary)+'</div>':''}${mat}${rows}</details>`;
    }).join(''):'';
    return `<div class=proj><div class=head><b>${T[j.type]||j.type}</b> ${stats}<span class=mut>· ${fmt(j.took)} · #${j.id}</span></div>
      ${p.description?`<div class=mut>📝 ${esc(p.description)}</div>`:''}
      ${floors?`<div class=s3d>${floors}</div>`:''}
      ${aptBlock}${roomBlock}${invBlock}
      ${g?`<details class=sec open><summary>🖼 Рендеры (${imgs.length})</summary><div class=g>${g}</div></details>`:''}
      ${j.reasoning?`<details class=sec><summary>🧠 Анализ нейросети</summary><div class=desc style=white-space:pre-wrap>${esc(j.reasoning)}</div></details>`:''}
      <div class=mut style=margin-top:6px><a href="${BASE}/api/project/${j.id}/export" target=_blank>⬇ Экспорт всех данных (JSON)</a></div>
      </div>`;
  }).join('');
}
// ---- unified object library ----
let ALLOBJ=[];
function renderObjects(q){
  q=(q||'').toLowerCase();
  const list=ALLOBJ.filter(o=>!q||[o.name,o.category,o.material,o.color,o.condition,o.room].filter(Boolean).join(' ').toLowerCase().includes(q));
  document.getElementById('objgrid').innerHTML=list.map(o=>{
    const th=o.crop_url?`<img loading=lazy src="${BASE+o.crop_url}">`:'<div class=noimg></div>';
    const attr=[o.category,o.material,o.color,o.condition].filter(Boolean).map(esc).join(' · ');
    return `<div class=objcard>${th}<div class=on>${esc(o.name||'?')}${o.qty>1?' ×'+o.qty:''}</div><div class=oa>${attr}</div>${o.room?'<div class=oa>'+esc(o.room)+'</div>':''}</div>`;
  }).join('')||'<div class=mut>ничего не найдено</div>';
}
fetch(BASE+'/api/objects').then(r=>r.json()).then(d=>{
  ALLOBJ=d.objects||[];
  document.getElementById('objcount').textContent='· '+(d.count||0);
  document.getElementById('objgrid').innerHTML = ALLOBJ.length?'':'<div class=mut>пока нет объектов — загрузите фотографии комнат в проект</div>';
  renderObjects('');
}).catch(e=>{document.getElementById('objgrid').textContent='ошибка: '+e;});
document.getElementById('objsearch').oninput=e=>renderObjects(e.target.value);
</script></body></html>"""


# --------------------------------------------------------------------------- #
# Model/network registry — operational status & control                       #
# --------------------------------------------------------------------------- #
MODELS = [
    {"name": "voice", "label": "Голос — STT + TTS (whisper + XTTS)", "container": "whisper-xtts-server",
     "image": "whisper-xtts-server:latest", "compose": "/home/deploy/whisper-xtts-server/docker-compose.yml",
     "device": "GPU", "vram": "~5.6 ГБ", "note": "STT можно на CPU, TTS→Piper на CPU"},
    {"name": "render", "label": "Фотореализм/Меблировка — SD+ControlNet", "container": RENDER_CONTAINER,
     "image": "interior-render:latest", "compose": RENDER_COMPOSE,
     "device": "GPU", "vram": "~5 ГБ", "note": "по запросу через очередь"},
    {"name": "floorplan", "label": "План→3D — Mask R-CNN", "container": "floorplan3d",
     "image": "floorplan3d:cpu", "compose": "/home/deploy/FloorPlanTo3D-API/docker-compose.yml",
     "device": "CPU", "vram": "0", "note": "всегда доступен, GPU не нужен"},
    {"name": "avatar", "label": "Аватар — MuseTalk (липсинк)", "container": "avatar-muse",
     "image": "musetalk:serve", "compose": "/home/deploy/avatar-muse/docker-compose.avatar.yml",
     "device": "GPU", "vram": "~4–5 ГБ", "note": "по запросу"},
    {"name": "animate", "label": "Текст+фото→видео — Wan 2.2", "container": "wan-comfy",
     "image": "wan-comfy:latest", "compose": "/home/deploy/wan22-i2v/docker-compose.wan.yml",
     "device": "GPU", "vram": "~14 ГБ", "note": "эксклюзивно весь T4"},
    {"name": "ollama", "label": "LLM — Ollama", "container": "ollama",
     "image": "ollama/ollama:latest", "compose": "/home/deploy/webui/docker-compose.yml",
     "device": "GPU", "vram": "~6–8 ГБ", "note": "грузит модель по запросу"},
    {"name": "iopaint", "label": "Удаление объектов — IOPaint", "container": "iopaint",
     "image": "iopaint:latest", "compose": "/home/deploy/iopaint/docker-compose.yml",
     "device": "GPU", "vram": "~2–3 ГБ", "note": ""},
    {"name": "vms", "label": "Видеоаналитика — DeepStream", "container": "vms",
     "image": "vms:latest", "compose": "/home/deploy/vms-platform/vms/docker-compose.yml",
     "device": "GPU", "vram": "—", "note": "RTSP-аналитика"},
]
_MBY = {m["name"]: m for m in MODELS}
_mcache = {"t": 0, "data": None}


def _docker_sets():
    imgs = set(_sh(["sudo", "docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], 20).stdout.split())
    running = set(_sh(["sudo", "docker", "ps", "--format", "{{.Names}}"], 20).stdout.split())
    allc = set(_sh(["sudo", "docker", "ps", "-a", "--format", "{{.Names}}"], 20).stdout.split())
    return imgs, running, allc


def _models_status():
    if _mcache["data"] and time.time() - _mcache["t"] < 4:
        return _mcache["data"]
    imgs, running, allc = _docker_sets()
    out = []
    for m in MODELS:
        out.append({**{k: m[k] for k in ("name", "label", "device", "vram", "note")},
                    "deployed": m["image"] in imgs,        # image built/pulled
                    "created": m["container"] in allc,     # container exists
                    "running": m["container"] in running})
    _mcache["data"] = out
    _mcache["t"] = time.time()
    return out


@app.get("/api/models")
def models():
    return JSONResponse({"models": _models_status()})


@app.post("/api/models/{name}/{action}")
def model_action(name: str, action: str):
    m = _MBY.get(name)
    if not m:
        raise HTTPException(404, "unknown model")
    if action == "start":
        r = _sh(["sudo", "docker", "compose", "-f", m["compose"], "up", "-d"], 300)
    elif action == "stop":
        r = _sh(["sudo", "docker", "stop", m["container"]], 180)
    else:
        raise HTTPException(400, "action must be start|stop")
    _mcache["t"] = 0  # invalidate cache
    return {"ok": r.returncode == 0, "log": (r.stdout + r.stderr)[-300:]}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


PAGE = r"""<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>GPU Очередь</title>
<style>
:root{--bg:#0e1116;--card:#171c24;--line:#222a35;--mut:#8b97a7;--acc:#4f8cff;--ok:#2ecc71;--warn:#f1c40f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#e6edf3;font:14px/1.5 system-ui;max-width:1080px;margin:auto;padding:18px}
h1{font-size:19px;margin:0 0 4px}.mut{color:var(--mut);font-size:12px}
.bar{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.chip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;font-size:13px}
.chip b{color:#fff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.card h3{margin:0 0 8px;font-size:14px}
input,select,textarea,button{background:#0b0e13;color:#e6edf3;border:1px solid var(--line);border-radius:9px;padding:9px;font-size:14px}
textarea{width:100%;height:54px}select,input[type=file]{width:100%}
button{background:var(--acc);border-color:var(--acc);color:#fff;cursor:pointer;font-weight:600}
.row{display:flex;gap:8px;margin:6px 0}.row>*{flex:1}
.prog{height:8px;background:#0b0e13;border-radius:5px;overflow:hidden;margin-top:6px}.prog>i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),#7aa7ff);transition:width .5s}
.q{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--line)}.q:last-child{border:0}
.pos{width:26px;height:26px;border-radius:50%;background:#0b0e13;border:1px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0}
.tag{font-size:11px;padding:1px 7px;border-radius:20px;background:#222a35}
.res{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.res a{display:block;border:1px solid var(--line);border-radius:8px;overflow:hidden;position:relative}
.res img{width:100%;display:block}.res span{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.6);font-size:11px;padding:2px 5px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px}
.swap{background:#2c2710;border:1px solid #5a4a12;color:var(--warn);border-radius:9px;padding:8px 11px;font-size:13px;margin:10px 0;display:none}
.stepline{color:var(--acc);font-size:13px;margin:4px 0}
.jr{background:#0f141b;border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
.jr>b{font-size:14px}
.meta{font-size:12px;color:var(--mut);margin:4px 0}
.rooms{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px;margin-top:6px}
.rm{background:#171c24;border:1px solid var(--line);border-radius:8px;padding:7px 9px;font-size:12px;line-height:1.35}
.projhead{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
.projhead .stat{background:#171c24;border:1px solid var(--line);border-radius:9px;padding:6px 13px;text-align:center;min-width:72px}
.projhead .stat b{font-size:15px;color:#fff}
.reason{margin:8px 0;background:#0b0e13;border:1px solid var(--line);border-radius:9px;padding:8px 11px;font-size:13px}
.reason summary{cursor:pointer;color:var(--acc);user-select:none}
.reason>div,.reason>pre{margin-top:6px;color:#c8d2dd;white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.5}
.roomgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin-top:12px}
.roomcard{background:#171c24;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.roomcard img{width:100%;display:block;aspect-ratio:3/2;object-fit:cover}
.rcbody{padding:8px 10px;font-size:12px;line-height:1.4}
.rcbody>b{font-size:13px}
.floorhead{margin:14px 0 6px;font-weight:600;font-size:14px;border-top:1px solid var(--line);padding-top:10px}
.badge{background:#1d2533;border:1px solid var(--line);border-radius:20px;padding:1px 7px;font-size:11px}
.inv{margin-top:6px;font-size:12px}.inv summary{cursor:pointer;color:var(--acc);user-select:none}
.invlist{margin-top:4px}.invrow{display:flex;gap:8px;align-items:center;padding:4px 0;border-bottom:1px solid var(--line)}.invrow:last-child{border:0}
.objthumb{width:46px;height:46px;border-radius:6px;object-fit:cover;background:#0b0e13;border:1px solid var(--line);flex-shrink:0}
</style></head><body>
<h1>🧮 GPU — очередь заданий</h1>
<div class=mut>Набивай очередь — она обрабатывается по одному заданию, модель подгружается автоматически по требованию.</div>
<div style=margin:8px_0><a href="library" style="color:var(--acc);font-weight:600">📚 Библиотека проектов и рендеров →</a></div>

<div class=bar>
  <div class=chip>В GPU: <b id=model>—</b></div>
  <div class=chip>VRAM: <b id=vram>—</b></div>
  <div class=chip>⏳ в очереди: <b id=cq>0</b></div>
  <div class=chip>▶ обрабатывается: <b id=cr>0</b></div>
  <div class=chip>✓ готово: <b id=cd>0</b></div>
</div>
<div class=swap id=swap></div>

<div class=card style=margin-bottom:14px>
  <h3>Сети (модели) — что развёрнуто и запущено</h3>
  <div id=nets class=mut>загрузка…</div>
</div>

<div class=grid>
  <div class=card>
    <h3>Добавить в очередь</h3>
    <input id=file type=file accept="image/*" multiple>
    <select id=type>
      <option value=project>🏗️ Проект: ВСЕ фото (планы этажей + референсы)</option>
      <option value=interior>🏠 Один план → анализ + рендер по комнатам</option>
      <option value=reference>📷 Референс-фото → фотореализм</option>
      <option value=render>🔼 Топ-даун рендер плана</option>
      <option value=furnish>🔼 Топ-даун меблировка</option>
    </select>
    <input id=style placeholder="Стиль" value="scandinavian, warm wood, natural daylight">
    <textarea id=desc placeholder="Описание объекта (необязательно): этажность, материалы, особенности…"></textarea>
    <label style="display:flex;gap:8px;align-items:center;margin:6px 0;font-size:13px"><input type=checkbox id=dorender> 🎨 Генерировать фото-рендеры комнат (нагружает GPU; по умолчанию выкл — только 3D-план)</label>
    <button id=add>В очередь →</button>
    <div id=addStatus class=mut style=margin-top:6px>«Проект» — выбери СРАЗУ ВСЁ: оба плана этажей и фото реальных комнат. Система сама разберёт что есть что.</div>
  </div>

  <div class=card>
    <h3>Сейчас обрабатывается</h3>
    <div id=running class=mut>—</div>
  </div>
</div>

<div class=card style=margin-top:14px>
  <h3>Очередь</h3>
  <div id=queue class=mut>пусто</div>
</div>

<div class=card style=margin-top:14px>
  <h3>Готово</h3>
  <div id=results></div>
</div>

<script>
const $=s=>document.querySelector(s);
const T={render:'Топ-даун рендер',furnish:'Топ-даун меблировка',interior:'Проект по комнатам',reference:'Фотореализация'};
function invHtml(inv,jid,pj){
  const e=s=>(s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;');
  const objs=inv.objects||[], m=inv.materials||{};
  const ms=v=>!v?'?':(typeof v==='object'?[v.material,v.color,v.condition].filter(Boolean).join(' '):v);
  const rows=objs.map(o=>{
    const th=(o.crop!=null&&jid!=null)?`<a href="api/jobs/${jid}/inventory/${pj}/object/${o.crop}" target=_blank><img class=objthumb src="api/jobs/${jid}/inventory/${pj}/object/${o.crop}"></a>`:'<div class=objthumb></div>';
    return `<div class=invrow>${th}<div><b>${e(o.name)}</b>${o.qty&&o.qty>1?' ×'+o.qty:''}<div class=mut>${[o.category,o.material,o.color,o.condition].filter(Boolean).map(e).join(' · ')}</div>${o.note?'<div class=mut>'+e(o.note)+'</div>':''}</div></div>`;
  }).join('');
  const mat=(m.floor||m.walls||m.ceiling)?`<div class=mut>Материалы: пол — ${e(ms(m.floor))}, стены — ${e(ms(m.walls))}, потолок — ${e(ms(m.ceiling))}</div>`:'';
  return `<details class=inv><summary>📋 Опись · ${objs.length} объектов (сегментировано)</summary>${inv.summary?'<div class=mut>'+e(inv.summary)+'</div>':''}${mat}<div class=invlist>${rows}</div></details>`;
}
function fmt(s){if(s==null)return '—';s=Math.round(s);if(s<60)return s+' с';return Math.floor(s/60)+' м '+(s%60)+' с';}
$('#add').onclick=async()=>{
  const files=$('#file').files; if(!files.length){$('#addStatus').textContent='Выбери файл(ы)';return;}
  $('#add').disabled=true; $('#addStatus').textContent='Добавляю…';
  try{
    let r;
    if($('#type').value==='project'){
      const fd=new FormData();
      for(const f of files) fd.append('files',f);
      fd.append('description',$('#desc').value); fd.append('style',$('#style').value);
      fd.append('render', $('#dorender').checked ? 'true' : 'false');
      r=await fetch('api/project',{method:'POST',body:fd});
    }else{
      const fd=new FormData(); fd.append('image',files[0]); fd.append('type',$('#type').value);
      fd.append('style',$('#style').value); fd.append('prompt',$('#style').value);
      r=await fetch('api/jobs',{method:'POST',body:fd});
    }
    const j=await r.json();
    $('#addStatus').textContent=r.ok?('✓ В очереди #'+j.id+(j.files?' · файлов '+j.files:'')):('Ошибка: '+JSON.stringify(j));
  }catch(e){$('#addStatus').textContent='Сбой: '+e;}
  $('#add').disabled=false;
};
async function tick(){
  let s; try{s=await (await fetch('api/state')).json();}catch(e){return;}
  $('#model').textContent = s.model==='render'?'Рендер (SD+ControlNet)':'STT/TTS (whisper)';
  $('#vram').textContent = s.gpu.vram_used+' / '+s.gpu.vram_total+' МБ · '+s.gpu.util+'%';
  $('#cq').textContent=s.counts.queued; $('#cr').textContent=s.counts.running; $('#cd').textContent=s.counts.done;
  const sw=$('#swap'); if(s.swapping){sw.style.display='block';sw.textContent='🔄 '+s.swapping;}else sw.style.display='none';
  const run=s.jobs.find(j=>j.status==='running');
  $('#running').innerHTML = run
    ? `<b>${T[run.type]||run.type}</b><div class=stepline>${run.step||'…'}</div>
       <div class=prog><i style="width:${Math.min(99,Math.round(run.elapsed/Math.max(run.est,1)*100))}%"></i></div>
       <div class=mut style=margin-top:4px>прошло ${fmt(run.elapsed)} · оценка ~${fmt(run.est)}</div>`
    : '<span class=mut>простаивает</span>';
  const q=s.jobs.filter(j=>j.status==='queued').sort((a,b)=>a.position-b.position);
  $('#queue').innerHTML = q.length ? q.map(j=>`<div class=q><div class=pos>${j.position}</div>
     <div style=flex:1><b>${T[j.type]||j.type}</b> <span class=tag>${j.prompt? j.prompt.slice(0,40):'—'}</span></div>
     <div class=mut>~${fmt(j.eta)}</div></div>`).join('') : '<span class=mut>пусто</span>';
  const done=s.jobs.filter(j=>j.status==='done'||j.status==='error');
  $('#results').innerHTML = done.map(j=>{
    if(j.status==='error') return `<div class=jr style="border-color:#5a1212"><b style=color:#e74c3c>⚠ ${T[j.type]||j.type}</b><div class=mut>${(j.error||'').slice(0,160)}</div></div>`;
    const esc0=s=>(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
    if(j.type==='project'){
      const p=j.project||{}, imgs=j.images||[];
      const byFloor={}; imgs.forEach(im=>{(byFloor[im.floor||'—']=byFloor[im.floor||'—']||[]).push(im);});
      const floorsHtml=Object.entries(byFloor).map(([fl,ims])=>{
        const cards=ims.map(im=>`<div class=roomcard>
          <a href="api/jobs/${j.id}/room/${im.idx}" target=_blank><img src="api/jobs/${j.id}/room/${im.idx}"></a>
          <div class=rcbody><b>${im.room}</b>${im.type?' · '+im.type:''}
            <div class=mut>${im.area?im.area+'м²':''}${im.dimensions?' · '+im.dimensions:''}${im.windows?' · окон '+im.windows:''} · <span class=badge>${im.source==='reference'?'📷 по референсу':'🎨 генерация'}</span></div>
            ${(im.fixtures&&im.fixtures.length)?`<div class=mut>🔧 ${(im.fixtures||[]).join(', ')}</div>`:''}
            ${im.description?`<div class=mut style=font-style:italic;margin-top:2px>${(im.description||'').toString().replace(/</g,'&lt;')}</div>`:''}
            ${(im.inventory&&im.inventory.objects&&im.inventory.objects.length)?invHtml(im.inventory,j.id,im.idx-1000):''}
          </div></div>`).join('');
        return `<div class=floorhead>🏢 ${fl} · ${ims.length} комнат</div><div class=roomgrid>${cards}</div>`;
      }).join('');
      return `<div class=jr>
        <b>🏗️ Готовый объект · ${fmt(j.took)}</b>
        <div class=projhead>
          <span class=stat><div class=mut>тип</div><b>${p.building_type||'—'}</b></span>
          ${p.apartments_total?'<span class=stat><div class=mut>квартир</div><b style=color:#ffd479>'+p.apartments_total+'</b></span>':''}
          ${p.living_area_m2?'<span class=stat><div class=mut>жилая площадь</div><b>'+p.living_area_m2+' м²</b></span>':''}
          <span class=stat><div class=mut>этажей</div><b>${p.levels||'—'}</b></span>
          <span class=stat><div class=mut>комнат</div><b>${p.rooms_total||0}</b></span>
          <span class=stat><div class=mut>площадь</div><b>${p.total_area_m2?p.total_area_m2+' м²':'—'}</b></span>
          ${p.stairs?'<span class=stat><div class=mut>лестница</div><b>есть</b></span>':''}
          ${p.photos?'<span class=stat><div class=mut>референсов</div><b>'+p.photos+'</b></span>':''}
        </div>
        ${(p.apartments&&p.apartments.length)?`<details class=reason open><summary>🏢 Квартиры (${p.apartments.length})</summary><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px">${p.apartments.map(a=>`<span class=chip>Кв. ${a.id}${a.floor&&p.levels>1?' ('+a.floor+')':''}${a.area_m2?' · '+a.area_m2+' м²':''}${a.rooms_count?' · '+a.rooms_count+'к':''}</span>`).join('')}</div></details>`:''}
        ${p.description?`<div class=mut style=margin:4px_0><b style=color:#e6edf3>Описание:</b> ${esc0(p.description)}</div>`:''}
        ${j.reasoning?`<details class=reason open><summary>🧠 Рассуждение нейросети (по этажам)</summary><div>${esc0(j.reasoning)}</div></details>`:''}
        ${j.raw?`<details class=reason><summary>📄 Полный ответ модели</summary><pre>${esc0(j.raw)}</pre></details>`:''}
        <div style=margin:10px_0>${(p.floors||[]).map(f=>f.scene!=null?`<a href="scene/${j.id}/${f.scene}" target=_blank style="color:var(--acc);margin-right:14px;font-weight:600">🧊 3D-схема: ${f.name}</a>`:'').join('')}</div>
        ${floorsHtml}
      </div>`;
    }
    if(j.type==='interior'){
      const ctx=j.context||{}, rooms=ctx.rooms||[], imgs=j.images||[];
      const esc=s=>(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
      const byName={}; rooms.forEach(r=>byName[(r.name||'').trim()]=r);
      // each room = ONE card: its render + its details together, clearly separated
      const cards=imgs.map(im=>{
        const r=byName[(im.room||'').trim()]||{};
        return `<div class=roomcard>
          <a href="api/jobs/${j.id}/room/${im.idx}" target=_blank><img src="api/jobs/${j.id}/room/${im.idx}"></a>
          <div class=rcbody><b>${im.room}</b>${r.type?' · '+r.type:''}
            <div class=mut>${r.area_m2?r.area_m2+'м²':''}${r.dimensions?' · '+r.dimensions:''}${r.windows?' · окон '+r.windows:''}</div>
            ${(r.fixtures&&r.fixtures.length)?`<div class=mut>🔧 ${r.fixtures.join(', ')}</div>`:''}
            ${r.description?`<div class=mut style=font-style:italic;margin-top:2px>${esc(r.description)}</div>`:''}
          </div></div>`;
      }).join('');
      const names=rooms.map(r=>r.name).filter(Boolean).join(', ');
      return `<div class=jr>
        <b>🏠 Проект · ${fmt(j.took)}</b>
        <div class=projhead>
          <span class=stat><div class=mut>тип</div><b>${ctx.building_type||'—'}</b></span>
          <span class=stat><div class=mut>этажей</div><b>${ctx.levels||'—'}</b></span>
          <span class=stat><div class=mut>комнат</div><b>${rooms.length}</b></span>
          <span class=stat><div class=mut>площадь</div><b>${ctx.total_area_m2?ctx.total_area_m2+' м²':'—'}</b></span>
          ${ctx.stairs?'<span class=stat><div class=mut>лестница</div><b>есть</b></span>':''}
        </div>
        <div class=mut style=margin:6px_0><b style=color:#e6edf3>Комнаты:</b> ${names}</div>
        ${ctx.summary?`<div class=mut style=margin:4px_0>${esc(ctx.summary)}</div>`:''}
        ${j.reasoning?`<details class=reason open><summary>🧠 Рассуждение нейросети</summary><div>${esc(j.reasoning)}</div></details>`:''}
        ${j.raw?`<details class=reason><summary>📄 Полный ответ модели (raw)</summary><pre>${esc(j.raw)}</pre></details>`:''}
        <div class=roomgrid>${cards}</div>
      </div>`;
    }
    return `<div class=jr><b>${T[j.type]||j.type} · ${fmt(j.took)}</b><div class=res style=margin-top:8px><a href="api/jobs/${j.id}/result" target=_blank><img src="api/jobs/${j.id}/result"></a></div></div>`;
  }).join('');
}
async function netAct(name,action){
  try{await fetch('api/models/'+name+'/'+action,{method:'POST'});}catch(e){}
  setTimeout(nets,800);
}
async function nets(){
  let s; try{s=await (await fetch('api/models')).json();}catch(e){return;}
  $('#nets').innerHTML = s.models.map(m=>{
    const color = m.running?'var(--ok)' : (m.deployed?'var(--mut)':'#e74c3c');
    const status = m.running?'запущена' : (m.deployed? (m.created?'остановлена':'развёрнута'):'НЕ развёрнута');
    const dev = m.device.includes('CPU')?'<span class=tag>CPU</span>':'<span class=tag style="background:#1d2a1d;color:#7fd6a0">'+m.device+'</span>';
    const btn = m.running
      ? `<button style="background:#3a1212;border-color:#5a1212;padding:5px 10px;font-weight:400" onclick="netAct('${m.name}','stop')">Стоп</button>`
      : (m.deployed?`<button style="padding:5px 10px;font-weight:400" onclick="netAct('${m.name}','start')">Старт</button>`:'');
    return `<div class=q><span class=dot style="background:${color}"></span>
      <div style=flex:1><b>${m.label}</b><div class=mut>${dev} ${m.vram!=='0'&&m.vram!=='—'?'VRAM '+m.vram:''} · ${status}${m.note?' · '+m.note:''}</div></div>
      ${btn}</div>`;
  }).join('');
}
setInterval(tick,1500); tick();
setInterval(nets,5000); nets();
</script></body></html>"""

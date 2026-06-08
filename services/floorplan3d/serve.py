"""Container entrypoint: load Mask R-CNN, add OCR, serve on 0.0.0.0:8204.

Routes:  POST / (walls/doors/windows) · POST /ocr (room names/areas/dims/stairs) · GET /health
Upstream fixes (kept out of application.py via monkeypatch):
  * load_model() is never called -> call it.
  * getClassNames mis-registered via @before_first_request -> drop the hook.
  * normalizePoints divides by doorCount with no guard -> ZeroDivisionError when a
    plan has no detected doors -> patch with a guarded version.
"""
import re

import cv2
import numpy as np
import PIL.Image
import pytesseract
from flask import jsonify, request

import application as A

try:
    A.application.before_first_request_funcs = []
except Exception:
    pass


def _safe_normalize(bbx, classNames):
    """Guarded reimplementation of application.normalizePoints (no div-by-zero)."""
    result, doorCount, doorDifference, index = [], 0, 0, -1
    for bb in bbx:
        index += 1
        if classNames[index] == 3:
            doorCount += 1
            doorDifference += max(abs(bb[3] - bb[1]), abs(bb[2] - bb[0]))
        result.append([bb[0], bb[1], bb[2], bb[3]])
    return result, (doorDifference / doorCount if doorCount else 0)


A.normalizePoints = _safe_normalize  # prediction() resolves the name at call time

ROOM_WORDS = {
    "salon", "salonik", "kuchnia", "jadalnia", "łazienka", "lazienka", "sypialnia",
    "garderoba", "hall", "gościnny", "goscinny", "gabinet", "pokój", "pokoj",
    "taras", "kominek", "spiżarnia", "pralnia", "garaż", "garaz", "kotłownia",
    "korytarz", "antresola", "przedpokój", "gospodarczy", "gospod",
    "спальня", "кухня", "гостиная", "ванная", "санузел", "прихожая", "холл",
}
STOP = {
    "parter", "piętro", "pietro", "osiedle", "chęciny", "checiny", "uwaga",
    "rysunek", "poglądowy", "pogladowy", "północ", "polnoc", "zejście", "zejscie",
    "garażu", "garazu", "do", "na", "rzut", "nr",
}
STAIRS_RE = re.compile(r"(schod|лестн|stair)", re.I)
AREA_INT_RE = re.compile(r"^\s*\d+\s*m\s*2?\s*$", re.I)        # 8m, 11m, 32m, 11m2
DIM_RE = re.compile(r"^\s*\d+[.,]\d+\s*m\s*$", re.I)           # 4.3m, 2.7m, 6.0m
NORTH_RE = re.compile(r"^(północ|polnoc|север|north)$", re.I)


def _kind(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "other"
    low = s.lower()
    if STAIRS_RE.search(low):
        return "stairs"
    if "²" in s or AREA_INT_RE.match(s):   # m² marker, or integer-metre = area
        return "area"
    if DIM_RE.match(s):                     # decimal-metre = dimension line
        return "dim"
    if NORTH_RE.match(low.strip(".")):
        return "north"
    if low.strip(".:") in STOP:
        return "other"
    if low in ROOM_WORDS:
        return "room"
    if re.search(r"[a-zżźćńółęąśа-я]", low) and len(re.sub(r"[^a-zа-я]", "", low)) >= 3:
        return "room?"
    return "other"


@A.application.route("/ocr", methods=["POST"])
def ocr():
    img = PIL.Image.open(request.files["image"].stream).convert("RGB")
    W, H = img.size
    big = img.resize((W * 2, H * 2), PIL.Image.LANCZOS)   # 2x: better small-text OCR
    d = pytesseract.image_to_data(big, lang="pol+rus+eng", config="--psm 11",
                                  output_type=pytesseract.Output.DICT)
    words = []
    for i, txt in enumerate(d["text"]):
        s = (txt or "").strip()
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if not s or conf < 40:
            continue
        k = _kind(s)
        if k == "other":
            continue
        words.append({"text": s, "kind": k, "conf": round(conf),
                      "x": int(d["left"][i] / 2), "y": int(d["top"][i] / 2),
                      "w": int(d["width"][i] / 2), "h": int(d["height"][i] / 2)})

    rooms = [w for w in words if w["kind"] in ("room", "room?")]
    areas = [w for w in words if w["kind"] == "area"]

    def center(w):
        return (w["x"] + w["w"] / 2, w["y"] + w["h"] / 2)

    for a in areas:
        ac = center(a)
        best, bd = None, 1e18
        for r in rooms:
            if "area" in r:
                continue
            rc = center(r)
            dist = (ac[0] - rc[0]) ** 2 + (ac[1] - rc[1]) ** 2
            if dist < bd:
                bd, best = dist, r
        if best is not None and bd < (max(W, H) * 0.18) ** 2:
            best["area"] = a["text"]

    # --- numeric pass: area labels (m²) and dimension lines (mm) ---------------
    # Real-estate plans often label rooms by AREA NUMBER (e.g. "11,82") and carry
    # overall sizes as mm DIMENSION numbers (e.g. "5100"). Capture both — they were
    # dropped by _kind (no letters / no "m" suffix) yet they carry the geometry.
    area_labels, dim_mm = [], []
    for i, txt in enumerate(d["text"]):
        s = (txt or "").strip().replace(" ", "")
        if not s:
            continue
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if conf < 30:
            continue
        cx = int(d["left"][i] / 2 + d["width"][i] / 4)
        cy = int(d["top"][i] / 2 + d["height"][i] / 4)
        m_dec = re.match(r"^(\d{1,2})[.,](\d{1,2})$", s)            # 6.95 / 11,82
        if m_dec:
            val = float(m_dec.group(1) + "." + m_dec.group(2))
            if 1.0 <= val <= 400.0:
                area_labels.append({"value": round(val, 2), "x": cx, "y": cy, "text": s})
            continue
        if re.match(r"^\d+$", s):                                   # pure integer
            n = int(s)
            if 1000 <= n <= 30000:                                  # 4-5 digit = mm dimension
                dim_mm.append({"mm": n, "x": cx, "y": cy})
            elif 100 <= n <= 999:                                   # 3 digit -> area like 616=6.16
                v = n / 100.0
                if 1.5 <= v <= 99.0:
                    area_labels.append({"value": round(v, 2), "x": cx, "y": cy, "text": s})

    return jsonify({
        "Width": W, "Height": H, "words": words,
        "rooms": [{"name": r["text"], "area": r.get("area"),
                   "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]} for r in rooms],
        "stairs": [w for w in words if w["kind"] == "stairs"],
        "dims": [w for w in words if w["kind"] == "dim"],
        "area_labels": area_labels, "dim_mm": dim_mm,
    })


def _merge_collinear(segs, ang_tol, off_tol):
    """Collapse fragmented Hough segments that lie on the same line into one wall.
    Clusters by orientation + signed perpendicular offset, then projects every
    endpoint onto the dominant direction and keeps the two extremes."""
    items = []
    for x1, y1, x2, y2 in segs:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        nx, ny = -np.sin(np.radians(a)), np.cos(np.radians(a))
        off = nx * x1 + ny * y1                       # distance of the line from origin
        length = float(np.hypot(x2 - x1, y2 - y1))
        items.append([a, off, (x1, y1, x2, y2), length])
    used = [False] * len(items)
    out = []
    for i in range(len(items)):
        if used[i]:
            continue
        ai, oi = items[i][0], items[i][1]
        group = [items[i][2]]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            aj, oj = items[j][0], items[j][1]
            da = min(abs(ai - aj), 180.0 - abs(ai - aj))
            if da < ang_tol and abs(oi - oj) < off_tol:
                group.append(items[j][2])
                used[j] = True
        dx, dy = np.cos(np.radians(ai)), np.sin(np.radians(ai))
        pts = []
        for x1, y1, x2, y2 in group:
            pts += [(x1, y1), (x2, y2)]
        t = [px * dx + py * dy for px, py in pts]
        lo, hi = int(np.argmin(t)), int(np.argmax(t))
        seg = (pts[lo][0], pts[lo][1], pts[hi][0], pts[hi][1])
        if np.hypot(seg[2] - seg[0], seg[3] - seg[1]) >= 1:
            out.append(seg)
    return out


@A.application.route("/vectorize", methods=["POST"])
def vectorize():
    """Classical wall vectorisation that PRESERVES ANY WALL ANGLE (unlike the
    axis-aligned Mask R-CNN boxes). Pipeline: Otsu binarize -> distance-transform
    thick-stroke isolation (this also drops thin text/dimension/legend strokes,
    acting as a region-free text filter) -> HoughLinesP (any orientation) ->
    collinear merge. Returns wall centre-line segments in processed-image space."""
    img = PIL.Image.open(request.files["image"].stream).convert("L")
    W0, H0 = img.size
    MAXL = 2400
    scale = 1.0
    if max(W0, H0) > MAXL:
        scale = MAXL / float(max(W0, H0))
        img = img.resize((max(1, int(W0 * scale)), max(1, int(H0 * scale))), PIL.Image.LANCZOS)
    g = np.array(img)
    H, W = g.shape
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink = float((bw > 0).mean())
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    nz = dist[dist > 0]
    if nz.size == 0:
        return jsonify({"Width": W, "Height": H, "scale": scale, "wall_segments": [],
                        "count": 0, "ink": ink, "note": "no ink"})
    # walls are the THICK strokes; thin text/dimension lines sit below this threshold
    T = max(2.0, float(np.percentile(nz, 60)))
    core = (dist >= T).astype(np.uint8) * 255
    walls = cv2.bitwise_and(cv2.dilate(core, np.ones((3, 3), np.uint8),
                                       iterations=int(round(T))), bw)
    minlen = max(18, int(0.025 * max(W, H)))
    lines = cv2.HoughLinesP(walls, 1, np.pi / 360, threshold=50,
                            minLineLength=minlen, maxLineGap=int(0.4 * minlen))
    segs = [tuple(int(v) for v in l[0]) for l in lines] if lines is not None else []
    merged = _merge_collinear(segs, ang_tol=6.0, off_tol=max(6.0, T * 0.9))
    # angle mix (diagnostics: how non-orthogonal is this plan)
    angs = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0 for x1, y1, x2, y2 in merged]
    diag = float(np.mean([(abs(a - 45) < 18 or abs(a - 135) < 18) for a in angs])) if angs else 0.0
    return jsonify({
        "Width": W, "Height": H, "scale": scale, "wall_thickness_px": round(T, 1),
        "ink": round(ink, 3), "raw_segments": len(segs), "count": len(merged),
        "diagonal_fraction": round(diag, 3),
        "wall_segments": [[int(a), int(b), int(c), int(d)] for a, b, c, d in merged],
    })


@A.application.route("/segment", methods=["POST"])
def segment():
    """Partition the floor into ACTUAL room regions that respect the walls, seeded
    by room-label anchor points (POST field 'seeds' = JSON [{"id","x","y"}], in the
    uploaded image's pixel space). Method: thick-stroke wall barriers -> heal thin
    furniture-line cuts -> geodesic flood from each seed (walls block it, a border
    seed absorbs the exterior) -> per-room contour polygon + centroid + area.
    Fixes 'spatial disorientation' (zones bleeding across walls)."""
    import json as _json
    img = PIL.Image.open(request.files["image"].stream).convert("L")
    W0, H0 = img.size
    MAXL = 1100
    scale = 1.0
    if max(W0, H0) > MAXL:
        scale = MAXL / float(max(W0, H0))
        img = img.resize((max(1, int(W0 * scale)), max(1, int(H0 * scale))), PIL.Image.LANCZOS)
    seeds = []
    try:
        for s in _json.loads(request.form.get("seeds", "[]")):
            seeds.append({"id": s.get("id"), "x": int(round(s["x"] * scale)), "y": int(round(s["y"] * scale))})
    except Exception:
        seeds = []
    g = np.array(img)
    H, W = g.shape
    if not seeds:
        return jsonify({"Width": W, "Height": H, "scale": scale, "rooms": []})
    ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    nz = dist[dist > 0]
    Tw = max(3.0, float(np.percentile(nz, 75))) if nz.size else 3.0
    ksz = int(2 * Tw) | 1
    walls = cv2.dilate((dist >= Tw).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    free = (walls == 0).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    free[walls > 0] = 0
    lab = np.zeros((H, W), np.float32)
    lab[0, :] = lab[-1, :] = lab[:, 0] = lab[:, -1] = 9999       # border = exterior basin
    for i, s in enumerate(seeds):
        if 0 <= s["y"] < H and 0 <= s["x"] < W:
            cv2.circle(lab, (s["x"], s["y"]), 4, i + 1, -1)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(1500):
        dil = cv2.dilate(lab, k)
        take = (lab == 0) & (free > 0) & (dil > 0)
        if not take.any():
            break
        lab[take] = dil[take]
    lab = lab.astype(np.int32)
    rooms = []
    for i, s in enumerate(seeds):
        reg = (lab == i + 1).astype(np.uint8)
        a = int(reg.sum())
        if a < (W * H) * 0.004:                                 # ignore specks
            continue
        cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
        M = cv2.moments(reg, binaryImage=True)
        cx = M["m10"] / M["m00"] if M["m00"] else s["x"]
        cy = M["m01"] / M["m00"] if M["m00"] else s["y"]
        rooms.append({"id": s["id"], "area_px": a, "cx": round(cx, 1), "cy": round(cy, 1),
                      "polygon": [[int(p[0]), int(p[1])] for p in poly]})
    return jsonify({"Width": W, "Height": H, "scale": scale,
                    "wall_thickness_px": round(Tw, 1), "rooms": rooms})


@A.application.route("/health")
def _health():
    return {"status": "ok", "model": "mask_rcnn floorplan + ocr"}, 200


print(">> loading Mask R-CNN weights...", flush=True)
A.load_model()
print(">> model loaded; serving on :8204", flush=True)
A.application.run(host="0.0.0.0", port=8204, threaded=False)

"""CubiCasa5k as an HTTP geometry service (CPU). Loads the model once.
POST /parse (multipart image) -> structured vector geometry:
  walls (quads), rooms (polygons + type), openings (doors/windows), icons (fixtures).
GET /health.
Coordinates are in the processed-image pixel space (Width/Height returned)."""
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, request
from skimage.morphology import medial_axis
import sknw
import networkx as nx
from skimage.morphology import skeletonize

# scipy>=1.9 mode returns scalar; CubiCasa wants the old array form
import scipy.stats as _ss
_orig_mode = _ss.mode
def _mode_compat(a, axis=0, **kw):
    kw.pop("keepdims", None)
    return _orig_mode(a, axis=axis, keepdims=True)
_ss.mode = _mode_compat

from floortrans.models import get_model
from floortrans.loaders import RotateNTurns
from floortrans.post_prosessing import split_prediction, get_polygons

ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
ICON_CLASSES = ["No Icon", "Window", "Door", "Closet", "Electrical Applience", "Toilet",
                "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
N_CLASSES, SPLIT = 44, [21, 12, 11]
MAXL = 1024

app = Flask(__name__)
print(">> loading CubiCasa5k weights...", flush=True)
_model = get_model("hg_furukawa_original", 51)
_model.conv4_ = torch.nn.Conv2d(256, N_CLASSES, bias=True, kernel_size=1)
_model.upsample = torch.nn.ConvTranspose2d(N_CLASSES, N_CLASSES, kernel_size=4, stride=4)
_ckpt = torch.load("/app/model/model_best_val_loss_var.pkl", map_location="cpu")
_model.load_state_dict(_ckpt["model_state"])
_model.eval()
_rot = RotateNTurns()
print(">> model loaded; serving on :8205", flush=True)


def _quad(p):
    return [[int(round(x)), int(round(y))] for x, y in np.asarray(p).reshape(-1, 2)]


def _wall_paths(gray, region=None, colored=None):
    """FULL wall network, 1:1 with the drawing. Captures EVERY wall line incl. thin
    interior partitions, classifies each as load-bearing vs partition by thickness, and
    drops furniture/text/dimension noise.
    Method: dark mask -> drop specks -> skeletonize ALL ink (so thin partitions survive)
    -> distance map for per-point thickness -> sknw graph -> keep only large connected
    sub-networks (furniture/text are small isolated islands) that overlap the apartments
    (drops external dimension bands) -> per-edge length filter -> classify by thickness.
    Returns [{"p":[[x,y,thick]...], "c":"bearing"|"partition"}]."""
    H, W = gray.shape[:2]
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thr = min(otsu, 120)
    dark = (gray < thr).astype(np.uint8)
    if region is not None:
        dark = cv2.bitwise_and(dark, (region > 0).astype(np.uint8))
    # drop tiny specks (text dots / noise) but keep line strokes
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    clean = np.zeros_like(dark)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= max(20, 0.00003 * W * H):
            clean[lab == i] = 1
    if int(clean.sum()) < 80:
        return []
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    skel = skeletonize(clean.astype(bool))
    try:
        G = sknw.build_sknw(skel.astype(np.uint16))
    except Exception:
        return []

    def edge_len(u, v):
        pts = G[u][v]["pts"]
        return float(sum(np.hypot(*(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)))

    col_mask = (colored > 0) if colored is not None else None
    # tight bounding box of the apartments (to reject external dimension/title bands)
    bbox = None
    if col_mask is not None and col_mask.any():
        ys, xs = np.where(col_mask)
        my = (ys.max() - ys.min()) * 0.04
        mx = (xs.max() - xs.min()) * 0.04
        bbox = (ys.min() - my, ys.max() + my, xs.min() - mx, xs.max() + mx)
    long_dim = max(H, W)
    keep = set()
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        tot = sum(edge_len(u, v) for u, v in sub.edges())
        if tot < 0.12 * long_dim:                        # furniture/text islands = small networks
            continue
        if sub.edges():
            pts = np.vstack([G[u][v]["pts"] for u, v in sub.edges()])
            if col_mask is not None:                     # must overlap apartments (drops dim bands)
                inside = col_mask[pts[:, 0].astype(int).clip(0, H - 1), pts[:, 1].astype(int).clip(0, W - 1)].mean()
                if inside < 0.30:
                    continue
            if bbox is not None:                         # and stay within the apartments' bbox
                cy, cx = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
                if not (bbox[0] <= cy <= bbox[1] and bbox[2] <= cx <= bbox[3]):
                    continue
        keep |= set(comp)

    minlen = max(10, 0.02 * long_dim)
    raw = []
    for s, e in G.edges():
        if s not in keep:
            continue
        pts = G[s][e]["pts"]
        if len(pts) < 2 or edge_len(s, e) < minlen:
            continue
        thick = [float(dist[int(r), int(c)]) * 2.0 for r, c in pts]
        step = max(3, int(np.median(thick) * 0.7))
        out, last = [], -999
        for idx, (r, c) in enumerate(pts):
            if idx == 0 or idx == len(pts) - 1 or idx - last >= step:
                out.append([int(c), int(r)])
                last = idx
        # ROBUST stroke thickness: low percentile ignores the distance-transform SPIKES
        # at junctions, so a wall keeps ONE uniform width and doesn't bulge at corners.
        tw = float(np.percentile(thick, 30))
        raw.append([out, tw])
    if not raw:
        return []
    # --- reject NON-wall artifacts: image frame/bezel and filled dark blobs ----------
    # A real wall is a thin stroke. The screenshot bezel / a dimension frame is either far
    # thicker than every real wall, or it hugs the image border. Drop both.
    med_tw = sorted(t for _, t in raw)[len(raw) // 2]
    tcap = max(5.0 * med_tw, 0.045 * long_dim)
    mxb, myb = 0.02 * W, 0.02 * H

    def _is_frame(out):
        edge = sum(1 for x, y in out if x < mxb or x > W - mxb or y < myb or y > H - myb)
        return edge / max(1, len(out)) > 0.6

    raw = [[o, t] for o, t in raw if t <= tcap and not _is_frame(o)]
    if not raw:
        return []
    # --- snap nearby endpoints so contour lines CLOSE and don't dangle in mid-air ---
    med_t = float(np.median([t for _, t in raw]))
    tol = max(med_t * 1.4, 9.0)
    ends = []                                            # [path_idx, end(0|-1), x, y]
    for ri, (out, _) in enumerate(raw):
        ends.append([ri, 0, out[0][0], out[0][1]])
        ends.append([ri, -1, out[-1][0], out[-1][1]])
    parent = list(range(len(ends)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            if abs(ends[i][2] - ends[j][2]) <= tol and abs(ends[i][3] - ends[j][3]) <= tol:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(ends)):
        groups.setdefault(find(i), []).append(i)
    for grp in groups.values():
        if len(grp) < 2:
            continue
        cx = int(round(sum(ends[i][2] for i in grp) / len(grp)))
        cy = int(round(sum(ends[i][3] for i in grp) / len(grp)))
        for i in grp:
            ri, ei = ends[i][0], ends[i][1]
            raw[ri][0][ei] = [cx, cy]
    # classify: load-bearing = thick. Cut from the thickness distribution, clamped.
    allt = sorted(t for _, t in raw)
    cut = min(max(allt[int(len(allt) * 0.6)], 4.0), 14.0)
    return [{"p": out, "t": round(t, 1), "c": "bearing" if t >= cut else "partition"} for out, t in raw]


def _apartments(bgr):
    """Detect SEPARATE apartments on a multi-unit floor plan. Developers fill each
    apartment with its own pastel colour, so an apartment = a connected region of one
    colour (k-means in Lab), cut at wall lines, internal partitions healed. The white/
    grey corridor and stair core are uncolored -> excluded. Returns apartment polygons."""
    H, W = bgr.shape[:2]
    b, g, r = bgr[:, :, 0].astype(int), bgr[:, :, 1].astype(int), bgr[:, :, 2].astype(int)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    colored = ((mx - mn > 16) & (mx > 90) & (mn < 245)).astype(np.uint8)
    frac = float(colored.mean())
    if frac < 0.12:                                  # not a colour-coded multi-unit plan
        return {"colored_fraction": round(frac, 3), "apartments": []}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 110
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    pts = lab[colored > 0].astype(np.float32)
    if len(pts) < 50:
        return {"colored_fraction": round(frac, 3), "apartments": []}
    K = 12
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(pts, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    full = np.full((H, W), -1, np.int32)
    full[colored > 0] = labels.flatten()
    apts = []
    for c in range(K):
        m = (full == c).astype(np.uint8)
        m[dark] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, llab, st, ce = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < 0.007 * W * H:
                continue
            reg = (llab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            poly = cv2.approxPolyDP(max(cnts, key=cv2.contourArea),
                                    0.01 * cv2.arcLength(max(cnts, key=cv2.contourArea), True), True).reshape(-1, 2)
            apts.append({"area_px": int(st[i, cv2.CC_STAT_AREA]),
                         "cx": float(round(ce[i][0], 1)), "cy": float(round(ce[i][1], 1)),
                         "polygon": [[int(p[0]), int(p[1])] for p in poly]})
    apts.sort(key=lambda a: -a["area_px"])
    return {"colored_fraction": round(frac, 3), "apartments": apts}


@app.route("/health")
def health():
    return {"status": "ok", "model": "cubicasa5k"}, 200


@app.route("/parse", methods=["POST"])
def parse():
    pil = Image.open(request.files["image"].stream).convert("RGB")
    W0, H0 = pil.size
    scale = 1.0
    if max(W0, H0) > MAXL:
        scale = MAXL / float(max(W0, H0))
        pil = pil.resize((max(1, int(W0 * scale)), max(1, int(H0 * scale))), Image.LANCZOS)
    arr = np.asarray(pil).astype(np.float32)
    arr = 2 * (arr / 255.0) - 1
    H, W = arr.shape[:2]
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    t = torch.tensor(np.moveaxis(arr, -1, 0))[None]
    t = F.pad(t, (0, pw, 0, ph))
    h2, w2 = t.shape[2], t.shape[3]
    with torch.no_grad():
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred = torch.zeros([len(rotations), N_CLASSES, h2, w2])
        for i, (f, b) in enumerate(rotations):
            p = _rot(_model(_rot(t, "tensor", f)), "tensor", b)
            p = _rot(p, "points", b)
            p = F.interpolate(p, size=(h2, w2), mode="bilinear", align_corners=True)
            pred[i] = p[0]
        prediction = torch.mean(pred, 0, True)[:, :, :H, :W]

    heatmaps, rooms, icons = split_prediction(prediction, (H, W), SPLIT)
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

    walls, openings, fixtures = [], [], []
    for poly, t_ in zip(polygons, types):
        if t_["type"] == "wall":
            walls.append(_quad(poly))
        else:  # icon
            cid = int(t_["class"])
            name = ICON_CLASSES[cid] if 0 <= cid < len(ICON_CLASSES) else str(cid)
            entry = {"class": name, "polygon": _quad(poly)}
            (openings if cid in (1, 2) else fixtures).append(entry)

    rooms_out = []
    for poly, t_ in zip(room_polygons, room_types):
        cid = int(t_["class"])
        name = ROOM_CLASSES[cid] if 0 <= cid < len(ROOM_CLASSES) else str(cid)
        try:
            coords = [[int(round(x)), int(round(y))] for x, y in poly.exterior.coords]
        except Exception:
            continue
        if len(coords) >= 4 and name not in ("Wall", "Background", "Railing"):
            rooms_out.append({"class": name, "polygon": coords})

    # arbitrary-shape FULL wall network (incl. partitions) via medial axis; restricted
    # to the building region (colour cluster) on multi-unit plans to drop external noise.
    try:
        gray_arr = np.asarray(pil.convert("L"))
        bgr_arr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        bb, gg, rr = bgr_arr[:, :, 0].astype(int), bgr_arr[:, :, 1].astype(int), bgr_arr[:, :, 2].astype(int)
        mxx = np.maximum(np.maximum(bb, gg), rr)
        mnn = np.minimum(np.minimum(bb, gg), rr)
        col = ((mxx - mnn > 16) & (mxx > 90) & (mnn < 245)).astype(np.uint8)
        region, col_for_filter = None, None
        if float(col.mean()) > 0.12:                 # colour-coded plan -> restrict to building
            ys, xs = np.where(col > 0)
            mh, mw = gray_arr.shape
            my = int((ys.max() - ys.min()) * 0.025); mx = int((xs.max() - xs.min()) * 0.025)
            region = np.zeros((mh, mw), np.uint8)     # TIGHT bbox rectangle severs external dim bands
            region[max(0, ys.min() - my):min(mh, ys.max() + my), max(0, xs.min() - mx):min(mw, xs.max() + mx)] = 1
            k = int(max(gray_arr.shape) * 0.035) | 1
            col_for_filter = cv2.dilate(col, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        wall_paths = _wall_paths(gray_arr, region, col_for_filter)
    except Exception:
        wall_paths = []
    # separate apartments on a multi-unit floor (by colour fill)
    try:
        apt = _apartments(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
    except Exception:
        apt = {"apartments": []}

    return jsonify({"Width": W, "Height": H, "scale": scale,
                    "walls": walls, "wall_paths": wall_paths, "rooms": rooms_out,
                    "openings": openings, "fixtures": fixtures,
                    "apartments": apt.get("apartments", []),
                    "colored_fraction": apt.get("colored_fraction", 0)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8205, threaded=False)

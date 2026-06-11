"""Visual attribute extraction for unique-object identification.

The appearance embedding (OSNet/ImageNet features) already separates most
distinct objects, but it is only weakly colour-discriminative. To reliably tell
*this red car* from *that blue car* — and to surface a human-readable attribute
("Car · red", "Dog · brown") — we compute a compact colour signature from each
object crop:

  * ``name`` / ``hex`` — a human-readable dominant colour for display.
  * ``hist`` — a 12-bin saturation-weighted hue histogram (plus achromatic
    handling) used as an extra matching gate so objects whose colours clearly
    disagree are never merged into the same identity.

Pure numpy + OpenCV; no model, negligible cost.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

HUE_BINS = 12  # 30°-wide hue buckets over OpenCV's 0..179 hue range

# Representative name per 30° hue bucket (OpenCV hue is 0..179 == 0..360°/2).
_HUE_NAMES = [
    "red", "orange", "yellow", "lime", "green", "teal",
    "cyan", "blue", "indigo", "violet", "magenta", "pink",
]


def color_signature(crop: "np.ndarray") -> tuple[str, str, np.ndarray]:
    """Return ``(name, hex, hist)`` for an object crop.

    ``hist`` is L1-normalized over ``HUE_BINS`` (zeros for an achromatic crop).
    Dark/low-value pixels are ignored; achromatic crops are named white/gray/
    black by their brightness.
    """
    empty = np.zeros(HUE_BINS, dtype=np.float32)
    if crop is None or getattr(crop, "size", 0) == 0:
        return ("unknown", "#000000", empty)
    try:
        import cv2  # noqa: WPS433
    except Exception:
        return ("unknown", "#000000", empty)

    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return ("unknown", "#000000", empty)

    # Centre crop a little to bias toward the object, then downsample.
    y0, y1 = int(h * 0.1), int(h * 0.9)
    x0, x1 = int(w * 0.1), int(w * 0.9)
    region = crop[y0:y1, x0:x1] if (y1 > y0 and x1 > x0) else crop
    small = cv2.resize(region, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)        # 0..179
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    lit = val > 0.12                              # drop near-black pixels
    chroma = lit & (sat > 0.28)                   # colourful pixels

    hist = np.zeros(HUE_BINS, dtype=np.float32)
    if chroma.sum() > 0:
        bins = ((hue[chroma] / 180.0) * HUE_BINS).astype(int) % HUE_BINS
        weights = sat[chroma]
        for b, wgt in zip(bins, weights):
            hist[int(b)] += float(wgt)
        total = float(hist.sum())
        if total > 0:
            hist /= total

    # Mean colour (for the swatch hex).
    pix = small.reshape(-1, 3)[lit.reshape(-1)] if lit.sum() > 0 else small.reshape(-1, 3)
    mean_bgr = pix.mean(axis=0)
    hexcol = "#%02x%02x%02x" % (
        int(np.clip(mean_bgr[2], 0, 255)),
        int(np.clip(mean_bgr[1], 0, 255)),
        int(np.clip(mean_bgr[0], 0, 255)),
    )

    # Name: chromatic -> dominant hue bucket; else brightness-based grey.
    chroma_frac = float(chroma.sum()) / float(max(1, lit.sum()))
    if chroma_frac >= 0.18 and hist.sum() > 0:
        name = _HUE_NAMES[int(np.argmax(hist))]
        # Brown is dark/low-sat orange — common for animals/objects.
        if name in ("orange", "red") and float(val[chroma].mean()) < 0.5:
            name = "brown"
    else:
        mean_v = float(val[lit].mean()) if lit.sum() > 0 else float(val.mean())
        name = "white" if mean_v > 0.72 else ("black" if mean_v < 0.3 else "gray")

    return (name, hexcol, hist)


def color_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Histogram-intersection similarity in [0, 1] of two hue histograms.

    Returns 1.0 when either side is achromatic (no hue info to disagree on), so
    the colour gate only ever *rejects* on a clear colour conflict."""
    if a is None or b is None:
        return 1.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.sum() <= 0 or b.sum() <= 0:
        return 1.0
    return float(np.minimum(a, b).sum())

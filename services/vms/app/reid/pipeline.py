"""Sighting feature extraction: turn YOLO person boxes into match features.

The camera worker already produces YOLO person ``Box``es (xyxy + score) on a
trigger frame it has copied for the recording. For each box we extract a
:class:`SightingFeature` carrying:

  * ``appearance_vec`` — a 512-d L2-normalized OSNet body embedding (always
    computed; a back-turned/masked person still yields a body vector), or
    ``None`` if the crop fails the quality gate / the embedder is unavailable.
  * ``face_vec`` — a 512-d L2-normalized ArcFace embedding when a face is
    visible inside the box (det_score + min pixel size gate), else ``None``.

Face assignment runs ArcFace ONCE on the whole frame and assigns each detected
face to the person box whose area best contains the face centroid (cheaper and
more robust than re-detecting inside every crop, and reuses the existing
:class:`app.faces.recognizer.FaceRecognizer` — no second face model on the GPU).

This module is pure orchestration over injected embedders; it imports nothing
heavy at module load and has no DB dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --- structural typing for the injected embedders / boxes --------------------

class _BoxLike(Protocol):
    """Whatever the detector yields; we only need xyxy + score."""

    @property
    def xyxy(self) -> tuple[int, int, int, int]: ...

    @property
    def score(self) -> float: ...


@dataclass
class BBox:
    """A normalized person box used internally by the pipeline.

    Accepts boxes from the detector in several shapes via :meth:`from_any`."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def contains_point(self, px: float, py: float) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    @classmethod
    def from_any(cls, box) -> "BBox":
        """Build from a detector Box (``.xyxy`` + ``.score``), a 4/5-tuple, or
        a dict with ``x1..y2``/``score``."""
        if isinstance(box, BBox):
            return box
        # Object with .xyxy and .score (the worker's detector Box).
        xyxy = getattr(box, "xyxy", None)
        if xyxy is not None:
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            return cls(x1, y1, x2, y2, float(getattr(box, "score", 0.0)))
        if isinstance(box, dict):
            return cls(
                int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]),
                float(box.get("score", 0.0)),
            )
        seq = tuple(box)
        x1, y1, x2, y2 = (int(round(float(v))) for v in seq[:4])
        score = float(seq[4]) if len(seq) > 4 else 0.0
        return cls(x1, y1, x2, y2, score)


@dataclass
class SightingFeature:
    """Per-detection feature bundle the manager consumes for assignment."""

    box: BBox
    appearance_vec: Optional[np.ndarray]
    face_vec: Optional[np.ndarray]
    face_det_score: float
    crop_quality: float  # 0..1 heuristic; higher == cleaner crop
    box_area_frac: float  # box area / frame area
    has_face: bool
    object_class: str = "person"  # COCO label this detection belongs to
    face_bbox: Optional[tuple] = None  # (x1,y1,x2,y2) of the assigned face, frame coords
    face_quality: float = 0.0  # det_score * frontalness (pose-aware face quality)
    color_name: str = "unknown"   # dominant colour ("red", "brown", "gray"…)
    color_hex: str = "#000000"
    color_hist: Optional[np.ndarray] = None  # 12-bin hue histogram
    # Vehicle attributes (NVIDIA TAO classifiers), set by the worker for vehicles.
    vehicle_make: Optional[str] = None
    vehicle_make_conf: float = 0.0
    vehicle_type: Optional[str] = None
    vehicle_type_conf: float = 0.0

    @property
    def usable_appearance(self) -> bool:
        return self.appearance_vec is not None


class _ReIDEmbedderLike(Protocol):
    def embed(self, crop: np.ndarray) -> np.ndarray: ...


class _FaceRecognizerLike(Protocol):
    def detect(self, frame: np.ndarray) -> list: ...


# --- crop quality helpers ----------------------------------------------------

def _clip_box(box: BBox, w: int, h: int) -> BBox:
    return BBox(
        x1=max(0, min(box.x1, w - 1)),
        y1=max(0, min(box.y1, h - 1)),
        x2=max(0, min(box.x2, w)),
        y2=max(0, min(box.y2, h)),
        score=box.score,
    )


def crop_quality(crop: np.ndarray) -> float:
    """Cheap 0..1 sharpness heuristic via variance of the Laplacian.

    A blurry/occluded crop yields a low value; we normalize a typical
    variance-of-Laplacian range (~0..500) into ``[0, 1]`` and clamp. Returns
    0.0 for empty crops. OpenCV is imported lazily so this stays importable in
    test environments without cv2."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    try:
        import cv2  # noqa: WPS433

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        # Fallback: intensity variance (still discriminates flat/occluded crops).
        lap_var = float(np.asarray(crop, dtype=np.float32).var())
    return float(max(0.0, min(1.0, lap_var / 500.0)))


def frontalness(kps) -> float:
    """0..1 frontal-ness from the 5 face landmarks (le, re, nose, lm, rm).

    A frontal face has the nose horizontally centred between the eyes and the
    mouth; a profile pushes it toward one side. Returns 1.0 when landmarks are
    unavailable (don't penalise). Cheap pose gate for surveillance crops."""
    try:
        import numpy as _np
        k = _np.asarray(kps, dtype=_np.float32).reshape(-1, 2)
        if k.shape[0] < 5:
            return 1.0
        le, re, nose = k[0], k[1], k[2]
        span = float(re[0] - le[0])
        if abs(span) < 1e-3:
            return 0.5
        ratio = (float(nose[0]) - float(le[0])) / span  # ~0.5 when frontal
        return float(max(0.0, min(1.0, 1.0 - 2.0 * abs(ratio - 0.5))))
    except Exception:
        return 1.0


def _aspect_ok(box: BBox, min_aspect: float = 1.1) -> bool:
    """A plausible standing/sitting body is taller than ~1.1x its width.

    Rejects horizontal slivers (false detections, partial occlusions)."""
    w = box.x2 - box.x1
    h = box.y2 - box.y1
    if w <= 0 or h <= 0:
        return False
    return (h / w) >= min_aspect


class IdentityPipeline:
    """Extracts :class:`SightingFeature`s from a frame + person boxes.

    Parameters
    ----------
    min_app_box_area_frac:
        Minimum box-area / frame-area to compute an appearance embedding /
        allow new-identity creation (tiny crops are noise).
    min_face_pixels:
        Minimum face bbox side length (px) to trust an ArcFace embedding.
    face_det_thresh:
        Minimum SCRFD detection score to accept a face for matching/exemplars.
    min_aspect:
        Minimum height/width ratio for a plausible body box.
    """

    def __init__(
        self,
        min_app_box_area_frac: float = 0.01,
        min_face_pixels: int = 24,
        face_det_thresh: float = 0.5,
        min_aspect: float = 1.1,
    ) -> None:
        self.min_app_box_area_frac = float(min_app_box_area_frac)
        self.min_face_pixels = int(min_face_pixels)
        self.face_det_thresh = float(face_det_thresh)
        self.min_aspect = float(min_aspect)

    def extract(
        self,
        frame: np.ndarray,
        boxes: Sequence,
        face_recognizer: Optional[_FaceRecognizerLike] = None,
        reid_embedder: Optional[_ReIDEmbedderLike] = None,
    ) -> list[SightingFeature]:
        """Produce one :class:`SightingFeature` per person box.

        ``face_recognizer`` / ``reid_embedder`` may each be ``None`` (graceful
        degradation: the corresponding vector is simply absent)."""
        if frame is None or getattr(frame, "size", 0) == 0 or not boxes:
            return []
        h, w = frame.shape[:2]
        frame_area = float(max(1, w * h))
        norm_boxes = [_clip_box(BBox.from_any(b), w, h) for b in boxes]
        # Carry each box's object class (detector Box has .label; default person).
        box_classes = [str(getattr(b, "label", "person") or "person") for b in boxes]
        is_person = [cls == "person" for cls in box_classes]

        # 1. Detect faces once on the whole frame, assign each to a *person* box
        #    (faces are meaningless for cars/animals/etc).
        face_for_box: dict[int, tuple[np.ndarray, float]] = {}
        if face_recognizer is not None and any(is_person):
            try:
                faces = face_recognizer.detect(frame)
            except Exception:
                logger.exception("Face detection failed on trigger frame.")
                faces = []
            for f in faces:
                fx1, fy1, fx2, fy2 = f.bbox
                fw, fh = (fx2 - fx1), (fy2 - fy1)
                if min(fw, fh) < self.min_face_pixels:
                    continue
                if float(getattr(f, "det_score", 0.0)) < self.face_det_thresh:
                    continue
                fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                # Smallest containing person box wins (handles overlapping people).
                best_i, best_area = -1, None
                for i, box in enumerate(norm_boxes):
                    if not is_person[i]:
                        continue
                    if box.contains_point(fcx, fcy):
                        if best_area is None or box.area < best_area:
                            best_i, best_area = i, box.area
                if best_i < 0:
                    continue
                emb = np.asarray(f.embedding, dtype=np.float32).reshape(-1)
                ds = float(getattr(f, "det_score", 0.0))
                fq = ds * frontalness(getattr(f, "kps", None))  # quality incl. pose
                # Keep the highest-QUALITY (score x frontalness) face per box.
                prev = face_for_box.get(best_i)
                if prev is None or fq > prev[3]:
                    face_for_box[best_i] = (emb, ds, (int(fx1), int(fy1), int(fx2), int(fy2)), fq)

        # 2. Per box: appearance embedding + attach assigned face.
        out: list[SightingFeature] = []
        for i, box in enumerate(norm_boxes):
            area_frac = box.area / frame_area
            crop = frame[box.y1 : box.y2, box.x1 : box.x2]
            quality = crop_quality(crop)
            # Dominant-colour signature for unique-object identification.
            from .attributes import color_signature

            color_name, color_hex, color_hist = color_signature(crop)

            # The tall-body aspect gate only makes sense for people; other
            # objects (cars are wider than tall, etc.) skip it.
            aspect_ok = _aspect_ok(box, self.min_aspect) if is_person[i] else True
            app_vec: Optional[np.ndarray] = None
            if (
                reid_embedder is not None
                and crop.size > 0
                and area_frac >= self.min_app_box_area_frac
                and aspect_ok
            ):
                try:
                    app_vec = np.asarray(
                        reid_embedder.embed(crop), dtype=np.float32
                    ).reshape(-1)
                except Exception:
                    logger.exception("ReID embedding failed for a crop.")
                    app_vec = None

            face_entry = face_for_box.get(i)
            face_vec = face_entry[0] if face_entry else None
            face_score = face_entry[1] if face_entry else 0.0
            face_bbox = face_entry[2] if face_entry and len(face_entry) > 2 else None
            face_quality = face_entry[3] if face_entry and len(face_entry) > 3 else face_score

            out.append(
                SightingFeature(
                    box=box,
                    appearance_vec=app_vec,
                    face_vec=face_vec,
                    face_det_score=face_score,
                    crop_quality=quality,
                    box_area_frac=area_frac,
                    has_face=face_vec is not None,
                    object_class=box_classes[i],
                    face_bbox=face_bbox,
                    face_quality=face_quality,
                    color_name=color_name,
                    color_hex=color_hex,
                    color_hist=color_hist,
                )
            )
        return out

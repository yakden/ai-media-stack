"""Per-camera multi-object tracker for presence / dwell-time accounting.

The camera worker detects objects every frame; this lightweight tracker links
those per-frame detections into *tracks* — one physical object instance present
in front of the camera over a span of time. A track records when the object
entered view and when it was last seen, which is exactly the dwell time we
accumulate onto its identity ("how long was this object in the camera").

Association is greedy IoU within the same object class (a car never inherits a
person's track). It is deliberately simple (no Kalman/Hungarian): cameras run
at a few FPS for detection and objects rarely cross, so IoU gating is enough and
cheap on CPU. Re-identification (the orientation-invariant "remember this exact
object across appearances/cameras") is layered on top by the worker, which
embeds a track's crop periodically and asks the IdentityManager to assign it —
the tracker only owns the short-term, single-camera continuity + timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Optional


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """A single object's continuous presence in one camera."""

    track_id: int
    object_class: str
    bbox: tuple[int, int, int, int]
    score: float
    enter_ts: float          # epoch seconds of first detection
    last_ts: float           # epoch seconds of most recent detection
    frames: int = 1
    peak_score: float = 0.0  # max detection confidence over the track's lifetime
    # Re-ID linkage (filled in by the worker once it embeds the crop).
    identity_id: Optional[int] = None
    identity_name: Optional[str] = None
    last_embed_ts: float = 0.0
    # Set when the track has been written to a PresenceSegment already.
    finalized: bool = False
    # Sighting row ids created for this track (to link them to the event on
    # close — exact, avoids the same-identity time-window over-claim race).
    sighting_ids: list = field(default_factory=list)
    # Temporal-fusion buffers (recent appearance embeddings + capture epoch +
    # crop quality), FIFO-capped by the worker; used to build one robust,
    # viewpoint-averaged query vector. Empty unless temporal fusion is enabled.
    appearance_vecs: list = field(default_factory=list)
    appearance_ts: list = field(default_factory=list)
    appearance_q: list = field(default_factory=list)
    # Face-embedding aggregation buffers (vectors + pose/quality weights) for
    # cross-angle face matching — averaged into one template before matching.
    face_vecs: list = field(default_factory=list)
    face_q: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_ts - self.enter_ts)


@dataclass
class ObjectTracker:
    """Greedy IoU tracker, class-aware, with a gap-based track timeout.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU to associate a detection with an existing track.
    max_gap_seconds:
        A track with no matching detection for longer than this is considered
        to have left view and is closed (its presence segment finalized).
    """

    iou_threshold: float = 0.3
    max_gap_seconds: float = 3.0
    _next_id: "count[int]" = field(default_factory=lambda: count(1))
    tracks: dict[int, Track] = field(default_factory=dict)

    def update(self, detections, now: float) -> tuple[list[Track], list[Track]]:
        """Associate this frame's detections to tracks.

        ``detections`` is a sequence of objects exposing ``.xyxy``, ``.score``
        and ``.label`` (the detector ``Box``). Returns ``(active, closed)`` where
        ``closed`` tracks have just timed out and need their presence finalized.
        """
        dets = []
        for d in detections:
            xyxy = tuple(int(v) for v in d.xyxy)
            dets.append((xyxy, float(getattr(d, "score", 0.0)), str(getattr(d, "label", "person") or "person")))

        unmatched_tracks = set(self.tracks.keys())
        used_dets: set[int] = set()

        # Greedy: for each track pick the best same-class IoU detection.
        for tid, tr in self.tracks.items():
            best_j, best_iou = -1, self.iou_threshold
            for j, (xyxy, _score, cls) in enumerate(dets):
                if j in used_dets or cls != tr.object_class:
                    continue
                iou = _iou(tr.bbox, xyxy)
                if iou >= best_iou:
                    best_j, best_iou = j, iou
            if best_j >= 0:
                xyxy, score, _cls = dets[best_j]
                tr.bbox = xyxy
                tr.score = score
                tr.peak_score = max(tr.peak_score, score)
                tr.last_ts = now
                tr.frames += 1
                used_dets.add(best_j)
                unmatched_tracks.discard(tid)

        # New tracks for unmatched detections.
        for j, (xyxy, score, cls) in enumerate(dets):
            if j in used_dets:
                continue
            tid = next(self._next_id)
            self.tracks[tid] = Track(
                track_id=tid, object_class=cls, bbox=xyxy, score=score,
                enter_ts=now, last_ts=now, peak_score=score,
            )

        # Close tracks that have gone stale.
        closed: list[Track] = []
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            if now - tr.last_ts > self.max_gap_seconds:
                closed.append(tr)
                del self.tracks[tid]

        return list(self.tracks.values()), closed

    def flush(self) -> list[Track]:
        """Close and return every remaining track (e.g. on worker shutdown)."""
        out = list(self.tracks.values())
        self.tracks.clear()
        return out

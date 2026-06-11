"""IdentityManager: the online cross-camera identity assignment engine.

DB-backed (single source of truth) over the ``identities`` /
``face_exemplars`` / ``appearance_exemplars`` / ``sightings`` tables, with the
in-memory :class:`app.reid.gallery.IdentityGallery` as derived state. For each
:class:`app.reid.pipeline.SightingFeature` it decides MATCH (existing identity)
vs NEW (auto-create "Person N") using the fusion / threshold / anti-explosion
rules in the architecture, persists a ``Sighting`` row, and updates the matched
identity's running face/appearance exemplars (rows + in-memory gallery).

The manager is constructed per camera worker (like the FaceIndex/FaceRecognizer
today): derived gallery rebuilt from the shared DB at startup and re-synced on a
timer; all writes go through to the DB so other workers converge. The API
process owns the authoritative gallery and exposes the merge/split helpers here.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .gallery import IdentityGallery, deserialize_vector, normalize, serialize_vector
from .pipeline import SightingFeature

logger = logging.getLogger(__name__)


@dataclass
class MatchConfig:
    """All matching thresholds / windows / caps (defaults from the architecture,
    tuned for buffalo_l ArcFace + OSNet x0_25 on a T4). The integration layer
    fills these from ``Settings``; defaults make the manager usable standalone."""

    # Face thresholds (ArcFace cosine).
    face_match: float = 0.42
    face_strong: float = 0.55
    face_reject_new: float = 0.32
    face_merge_threshold: float = 0.60
    # Appearance thresholds (OSNet cosine).
    app_match: float = 0.62
    app_match_cross: float = 0.66
    app_gate: float = 0.50
    # Windows / decay.
    app_window_seconds: float = 600.0
    app_decay_tau_seconds: float = 43_200.0
    # Temporal aggregation: fuse a track's recent appearance embeddings into one
    # quality+decay-weighted query vector before matching (off => per-frame vec).
    app_temporal_fusion: bool = False
    # Exemplar caps.
    max_face_exemplars: int = 8
    max_app_exemplars: int = 16
    # Face-exemplar "useful but not redundant" acceptance band.
    face_exemplar_lo: float = 0.45
    face_exemplar_hi: float = 0.90
    # Margin tests (best - second_best between DISTINCT identities).
    match_margin_face: float = 0.06
    match_margin_app: float = 0.05
    # Quality gates.
    min_face_pixels: int = 24
    min_app_box_area_frac: float = 0.01
    min_crop_quality_for_new: float = 0.10
    require_quality_for_new: bool = True
    # Colour gate: reject an appearance match between non-person objects whose
    # hue histograms intersect below this (a red car must not become a blue one).
    color_gate: float = 0.35
    # Anti-explosion.
    new_identity_rate_per_min: int = 20
    # Sticky / hysteresis: keep a continuous person on the same identity.
    sticky_iou: float = 0.30
    sticky_seconds: float = 2.0


@dataclass
class AssignResult:
    """Outcome of :meth:`IdentityManager.assign` for one sighting."""

    identity_id: Optional[int]
    sighting_id: Optional[int]
    match_kind: str  # 'face' | 'appearance' | 'new' | 'sticky' | 'dropped'
    face_score: Optional[float]
    appearance_score: Optional[float]
    is_new: bool
    created_identity: bool = False


@dataclass
class _StickyEntry:
    identity_id: int
    box: tuple[int, int, int, int]
    ts: float
    object_class: str = "person"


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IdentityManager:
    """Online per-sighting assignment over a DB session + a shared gallery.

    Parameters
    ----------
    gallery:
        The in-memory :class:`IdentityGallery` (derived state). The manager
        mutates it as it creates identities / adds exemplars so subsequent
        sightings in the same batch see the update immediately.
    config:
        :class:`MatchConfig` thresholds.
    """

    def __init__(
        self,
        gallery: IdentityGallery,
        config: Optional[MatchConfig] = None,
    ) -> None:
        self.gallery = gallery
        self.cfg = config or MatchConfig()
        self._lock = threading.RLock()
        # Per-camera sticky cache: camera_id -> list[_StickyEntry].
        self._sticky: dict[int, list[_StickyEntry]] = {}
        # Per-camera new-identity rate limiter: camera_id -> list[epoch].
        self._new_times: dict[int, list[float]] = {}

    # -- public API -----------------------------------------------------------

    def assign(
        self,
        session,
        feature: SightingFeature,
        camera_id: int,
        ts: Optional[datetime] = None,
        event_id: Optional[int] = None,
        thumb_path: Optional[str] = None,
    ) -> AssignResult:
        """Assign one sighting feature to an identity (existing or new),
        persist the ``Sighting`` row + exemplar updates, and return the outcome.

        The caller commits the session (so a whole trigger-frame batch persists
        atomically). All in-memory gallery updates are applied immediately."""
        ts = ts or datetime.utcnow()
        # DB datetime columns are naive UTC; normalise any tz-aware input so we
        # never compare offset-aware vs naive datetimes downstream.
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        with self._lock:
            # 0. Sticky / hysteresis: a continuous, overlapping person on the
            #    same camera keeps its last identity without re-evaluation.
            sticky_id = self._sticky_lookup(
                camera_id, feature.box.xyxy, now=time.time(),
                object_class=feature.object_class or "person",
            )
            if sticky_id is not None:
                return self._finalize(
                    session, feature, camera_id, ts, event_id, thumb_path,
                    identity_id=sticky_id, match_kind="sticky",
                    face_score=None, app_score=None, is_new=False,
                )

            face_decision = self._decide_by_face(feature)
            if face_decision is not None:
                identity_id, fscore = face_decision
                return self._finalize(
                    session, feature, camera_id, ts, event_id, thumb_path,
                    identity_id=identity_id, match_kind="face",
                    face_score=fscore, app_score=None, is_new=False,
                )

            # Borderline face may have set a "veto" identity to protect against
            # appearance-assigning to a different identity.
            veto_identity = self._face_veto_identity(feature)

            app_decision = self._decide_by_appearance(
                feature, camera_id, ts, veto_identity
            )
            if app_decision is not None:
                identity_id, ascore = app_decision
                return self._finalize(
                    session, feature, camera_id, ts, event_id, thumb_path,
                    identity_id=identity_id, match_kind="appearance",
                    face_score=None, app_score=ascore, is_new=False,
                )

            # NEW (subject to quality gate + rate limit).
            return self._create_new(
                session, feature, camera_id, ts, event_id, thumb_path
            )

    # -- decision steps -------------------------------------------------------

    def _decide_by_face(
        self, feature: SightingFeature
    ) -> Optional[tuple[int, float]]:
        """Step 1: confident face -> identity. Returns (identity_id, score) on a
        clean accept (STRONG or MATCH-with-margin) else None (caller falls
        through to appearance)."""
        if feature.face_vec is None or feature.face_det_score < self.cfg.min_face_pixels * 0:
            return None
        hits = self.gallery.best_face_per_identity(feature.face_vec, k=16)
        if not hits:
            return None
        best = hits[0]
        second = hits[1].score if len(hits) > 1 else -1.0
        margin = best.score - second

        if best.score >= self.cfg.face_strong:
            # Authoritative: ignore appearance, ignore a thin margin (a strong
            # face is conclusive even amid look-alikes).
            return (best.identity_id, best.score)
        if best.score >= self.cfg.face_match and margin >= self.cfg.match_margin_face:
            return (best.identity_id, best.score)
        return None

    def _face_veto_identity(self, feature: SightingFeature) -> Optional[int]:
        """If this sighting has a confident face for identity A (>= FACE_MATCH),
        appearance must never assign it to a different identity B. Returns A (the
        protected identity) when the face is at/above FACE_MATCH, else None."""
        if feature.face_vec is None:
            return None
        hits = self.gallery.best_face_per_identity(feature.face_vec, k=2)
        if hits and hits[0].score >= self.cfg.face_match:
            return hits[0].identity_id
        return None

    def _decide_by_appearance(
        self,
        feature: SightingFeature,
        camera_id: int,
        ts: datetime,
        veto_identity: Optional[int],
    ) -> Optional[tuple[int, float]]:
        """Step 2: appearance fallback within the time window. Returns
        (identity_id, decayed_score) on accept else None."""
        if feature.appearance_vec is None:
            return None
        # If the face provided borderline corroboration evidence, fuse it.
        face_corroborator = self._face_corroborator(feature)

        cands = self.gallery.appearance_candidates(
            feature.appearance_vec, now=ts, same_camera_id=camera_id,
            object_class=feature.object_class,
        )
        if not cands:
            return None
        best = cands[0]
        second = cands[1].score if len(cands) > 1 else -1.0
        margin = best.score - second

        # Face contradiction veto: a confident face for A forbids appearance->B.
        if veto_identity is not None and best.identity_id != veto_identity:
            # Try to fall to a candidate that IS the vetoed identity.
            best = next(
                (c for c in cands if c.identity_id == veto_identity), None
            )
            if best is None:
                return None
            second = max(
                (c.score for c in cands if c.identity_id != best.identity_id),
                default=-1.0,
            )
            margin = best.score - second

        # Borderline-face fusion: if a face weakly corroborates THIS candidate,
        # accept at the lower APP_GATE instead of the full APP_MATCH bar.
        if (
            face_corroborator is not None
            and face_corroborator[0] == best.identity_id
            and best.score >= self.cfg.app_gate
            and margin >= self.cfg.match_margin_app
        ):
            return (best.identity_id, best.score)

        # Colour gate: for non-person objects, a clear colour conflict forbids
        # the match (keeps unique instances — red car vs blue car — separate).
        if (feature.object_class or "person") != "person":
            from .attributes import color_similarity

            cand_hist = self.gallery.color_hist_of(best.identity_id)
            if color_similarity(getattr(feature, "color_hist", None), cand_hist) < self.cfg.color_gate:
                return None

        # Pure appearance link: same- vs cross-camera bar.
        seen_cameras = self.gallery.identity_cameras(best.identity_id)
        cross = bool(seen_cameras) and camera_id not in seen_cameras
        bar = self.cfg.app_match_cross if cross else self.cfg.app_match
        if best.score >= bar and margin >= self.cfg.match_margin_app:
            return (best.identity_id, best.score)
        return None

    def _face_corroborator(
        self, feature: SightingFeature
    ) -> Optional[tuple[int, float]]:
        """Borderline-face evidence: a face in [FACE_REJECT_NEW, FACE_MATCH)
        that points at an identity. Returns (identity_id, score) else None."""
        if feature.face_vec is None:
            return None
        hits = self.gallery.best_face_per_identity(feature.face_vec, k=2)
        if not hits:
            return None
        best = hits[0]
        if self.cfg.face_reject_new <= best.score < self.cfg.face_match:
            return (best.identity_id, best.score)
        return None

    # -- new-identity creation ------------------------------------------------

    def _quality_ok_for_new(self, feature: SightingFeature) -> bool:
        if not self.cfg.require_quality_for_new:
            return True
        # A usable face always qualifies (faces are strong, rare evidence).
        if feature.has_face and feature.face_vec is not None:
            return True
        if feature.appearance_vec is None:
            return False
        if feature.box_area_frac < self.cfg.min_app_box_area_frac:
            return False
        if feature.crop_quality < self.cfg.min_crop_quality_for_new:
            return False
        return True

    def _rate_limit_ok(self, camera_id: int, now: float) -> bool:
        window = self._new_times.setdefault(camera_id, [])
        cutoff = now - 60.0
        window[:] = [t for t in window if t >= cutoff]
        if len(window) >= self.cfg.new_identity_rate_per_min:
            return False
        return True

    def _create_new(
        self,
        session,
        feature: SightingFeature,
        camera_id: int,
        ts: datetime,
        event_id: Optional[int],
        thumb_path: Optional[str],
    ) -> AssignResult:
        if not self._quality_ok_for_new(feature):
            logger.debug(
                "Dropping unmatched low-quality crop (cam=%s) instead of "
                "spawning an identity.", camera_id,
            )
            return AssignResult(
                identity_id=None, sighting_id=None, match_kind="dropped",
                face_score=None, appearance_score=None, is_new=False,
            )
        if not self._rate_limit_ok(camera_id, time.time()):
            logger.warning(
                "New-identity rate limit hit on cam=%s; dropping crop.", camera_id
            )
            return AssignResult(
                identity_id=None, sighting_id=None, match_kind="dropped",
                face_score=None, appearance_score=None, is_new=False,
            )

        from app.db.models import Identity  # noqa: WPS433

        has_face = feature.has_face and feature.face_vec is not None
        obj_class = feature.object_class or "person"
        identity = Identity(
            name="",  # filled after we get the id
            is_named=False,
            object_class=obj_class,
            num_sightings=0,
            first_seen=ts,
            last_seen=ts,
            # Provisional: a single low-evidence, faceless sighting that
            # maintenance prunes if it never accrues a 2nd sighting / a face.
            # Non-person objects have no face, so don't hold them to that bar.
            is_provisional=(obj_class == "person" and not has_face),
        )
        session.add(identity)
        session.flush()  # assign identity.id
        # Auto-name by class: "Person 3", "Car 7", "Dog 2", ...
        identity.name = f"{obj_class.capitalize()} {identity.id}"
        self.gallery.register_identity(int(identity.id), obj_class)
        # Seed the colour gate immediately so the next sighting can be gated.
        hist = getattr(feature, "color_hist", None)
        if hist is not None:
            self.gallery.set_identity_color(int(identity.id), hist)
        self._new_times.setdefault(camera_id, []).append(time.time())

        result = self._finalize(
            session, feature, camera_id, ts, event_id, thumb_path,
            identity_id=int(identity.id), match_kind="new",
            face_score=None, app_score=None, is_new=True,
        )
        result.created_identity = True
        return result

    # -- persistence + exemplar update ---------------------------------------

    def _finalize(
        self,
        session,
        feature: SightingFeature,
        camera_id: int,
        ts: datetime,
        event_id: Optional[int],
        thumb_path: Optional[str],
        identity_id: int,
        match_kind: str,
        face_score: Optional[float],
        app_score: Optional[float],
        is_new: bool,
    ) -> AssignResult:
        from app.db.models import Identity, Sighting  # noqa: WPS433

        x1, y1, x2, y2 = feature.box.xyxy
        sighting = Sighting(
            identity_id=identity_id,
            camera_id=camera_id,
            event_id=event_id,
            ts=ts,
            object_class=feature.object_class or "person",
            bbox_x1=int(x1), bbox_y1=int(y1), bbox_x2=int(x2), bbox_y2=int(y2),
            det_score=float(feature.box.score),
            has_face=bool(feature.has_face),
            face_score=face_score,
            appearance_score=app_score,
            match_kind=match_kind if match_kind != "sticky" else "appearance",
            thumb_path=thumb_path,
        )
        session.add(sighting)
        session.flush()  # assign sighting.id

        # Update exemplars + identity bookkeeping (skip for pure sticky to keep
        # the hot continuous-track path cheap; the streak already enrolled).
        identity = session.get(Identity, identity_id)
        if identity is not None:
            self._maybe_add_face_exemplar(
                session, identity, feature, camera_id, sighting.id, face_score
            )
            self._maybe_add_appearance_exemplar(
                session, identity, feature, camera_id, sighting.id, ts
            )
            identity.num_sightings = int(identity.num_sightings or 0) + 1
            if identity.first_seen is None or ts < identity.first_seen:
                identity.first_seen = ts
            if identity.last_seen is None or ts > identity.last_seen:
                identity.last_seen = ts
            if identity.rep_sighting_id is None:
                identity.rep_sighting_id = sighting.id
            # A 2nd sighting (or a face) graduates a provisional identity.
            if getattr(identity, "is_provisional", False) and (
                identity.num_sightings >= 2 or feature.has_face
            ):
                identity.is_provisional = False
            # Merge/refresh visual attributes (colour + vehicle make/type).
            self._update_identity_attributes(identity, feature)

        # Refresh sticky entry for this continuous track.
        self._sticky_update(
            camera_id, feature.box.xyxy, identity_id, time.time(),
            object_class=feature.object_class or "person",
        )

        return AssignResult(
            identity_id=identity_id,
            sighting_id=int(sighting.id),
            match_kind=match_kind,
            face_score=face_score,
            appearance_score=app_score,
            is_new=is_new,
        )

    def _update_identity_attributes(self, identity, feature: SightingFeature) -> None:
        """Merge the sighting's visual attributes onto the identity (JSON).

        Colour is seeded once (then stable); vehicle make/type keep the
        highest-confidence value seen across the identity's sightings.
        """
        import json as _json

        attrs: dict = {}
        if getattr(identity, "attributes", None):
            try:
                attrs = _json.loads(identity.attributes) or {}
            except Exception:
                attrs = {}

        cname = getattr(feature, "color_name", None)
        if "color" not in attrs and cname and cname != "unknown":
            attrs["color"] = cname
            attrs["hex"] = getattr(feature, "color_hex", "#000000")
            hist = getattr(feature, "color_hist", None)
            if hist is not None:
                attrs["hist"] = np.asarray(hist, dtype=np.float32).tolist()

        mk, mkc = getattr(feature, "vehicle_make", None), float(getattr(feature, "vehicle_make_conf", 0.0) or 0.0)
        if mk and mkc >= float(attrs.get("make_conf", 0.0)):
            attrs["make"], attrs["make_conf"] = mk, round(mkc, 3)
        tp, tpc = getattr(feature, "vehicle_type", None), float(getattr(feature, "vehicle_type_conf", 0.0) or 0.0)
        if tp and tpc >= float(attrs.get("type_conf", 0.0)):
            attrs["type"], attrs["type_conf"] = tp, round(tpc, 3)

        if attrs:
            try:
                identity.attributes = _json.dumps(attrs)
            except Exception:
                pass

    def _maybe_add_face_exemplar(
        self, session, identity, feature: SightingFeature,
        camera_id: int, sighting_id: int, face_score: Optional[float],
    ) -> None:
        if feature.face_vec is None or not feature.has_face:
            return
        if feature.face_det_score < self.cfg.face_match * 0:  # always pass det
            pass
        # Useful-but-not-redundant band: a brand-new identity (no exemplars
        # yet, face_score None) always seeds its first face exemplar.
        if face_score is not None and not (
            self.cfg.face_exemplar_lo <= face_score <= self.cfg.face_exemplar_hi
        ):
            return

        from app.db.models import FaceExemplar  # noqa: WPS433

        vec = normalize(feature.face_vec)
        existing = list(identity.face_exemplars)
        if len(existing) >= self.cfg.max_face_exemplars:
            # Evict the lowest-quality (lowest det_score) exemplar.
            victim = min(existing, key=lambda e: (e.det_score or 0.0))
            session.delete(victim)
        ex = FaceExemplar(
            identity_id=identity.id,
            vector=serialize_vector(vec),
            det_score=float(feature.face_det_score),
            camera_id=camera_id,
            sighting_id=sighting_id,
        )
        session.add(ex)
        session.flush()
        self.gallery.add_face_exemplar(int(ex.id), int(identity.id), vec)
        identity.face_centroid = self._recompute_face_centroid(session, identity)

    def _maybe_add_appearance_exemplar(
        self, session, identity, feature: SightingFeature,
        camera_id: int, sighting_id: int, ts: datetime,
    ) -> None:
        if feature.appearance_vec is None:
            return
        if feature.box_area_frac < self.cfg.min_app_box_area_frac:
            return

        from app.db.models import AppearanceExemplar  # noqa: WPS433

        vec = normalize(feature.appearance_vec)
        existing = list(identity.appearance_exemplars)
        if len(existing) >= self.cfg.max_app_exemplars:
            # Evict oldest-lowest: prefer dropping the stalest, breaking ties by
            # lowest quality.
            victim = min(existing, key=lambda e: (e.ts, e.quality or 0.0))
            session.delete(victim)
        ex = AppearanceExemplar(
            identity_id=identity.id,
            vector=serialize_vector(vec),
            quality=float(feature.crop_quality),
            camera_id=camera_id,
            sighting_id=sighting_id,
            ts=ts,
        )
        session.add(ex)
        session.flush()
        self.gallery.add_appearance_exemplar(
            int(identity.id), vec, ts, camera_id
        )

    def _recompute_face_centroid(self, session, identity) -> Optional[bytes]:
        vecs = []
        for ex in identity.face_exemplars:
            try:
                vecs.append(deserialize_vector(ex.vector))
            except Exception:
                continue
        if not vecs:
            return None
        mean = normalize(np.mean(np.vstack(vecs), axis=0))
        return serialize_vector(mean)

    # -- sticky / hysteresis --------------------------------------------------

    def _sticky_lookup(
        self, camera_id: int, box: tuple[int, int, int, int], now: float,
        object_class: str = "person",
    ) -> Optional[int]:
        entries = self._sticky.get(camera_id)
        if not entries:
            return None
        cutoff = now - self.cfg.sticky_seconds
        entries[:] = [e for e in entries if e.ts >= cutoff]
        best_id, best_iou = None, 0.0
        for e in entries:
            # Only stick to the SAME object class (a car must not inherit a
            # person's identity just because their boxes overlap briefly).
            if e.object_class != object_class:
                continue
            iou = _iou(box, e.box)
            if iou >= self.cfg.sticky_iou and iou > best_iou:
                best_id, best_iou = e.identity_id, iou
        return best_id

    def _sticky_update(
        self, camera_id: int, box: tuple[int, int, int, int],
        identity_id: int, now: float, object_class: str = "person",
    ) -> None:
        entries = self._sticky.setdefault(camera_id, [])
        cutoff = now - self.cfg.sticky_seconds
        entries[:] = [e for e in entries if e.ts >= cutoff]
        entries.append(
            _StickyEntry(identity_id=identity_id, box=box, ts=now, object_class=object_class)
        )

    # -- offline merge / split (used by API + maintenance) -------------------

    def merge(self, session, target_id: int, source_ids: list[int]) -> int:
        """Merge ``source_ids`` into ``target_id``: reassign all sightings +
        exemplars, recompute centroids/counters, delete the sources. Returns the
        number of sightings moved. Face-priority; the caller is responsible for
        the (conservative, face-based) decision to merge. Commits NOT done here.

        ``is_named`` identities are protected from automatic merges upstream;
        this low-level helper performs whatever merge it is told to (operator
        MERGE may target a named identity intentionally)."""
        from app.db.models import (  # noqa: WPS433
            AppearanceExemplar,
            Event,
            FaceExemplar,
            Identity,
            Sighting,
        )

        target = session.get(Identity, target_id)
        if target is None:
            raise ValueError(f"merge target identity {target_id} not found")
        moved = 0
        for sid in source_ids:
            if sid == target_id:
                continue
            src = session.get(Identity, sid)
            if src is None:
                continue
            moved += (
                session.query(Sighting)
                .filter(Sighting.identity_id == sid)
                .update({Sighting.identity_id: target_id}, synchronize_session=False)
            )
            session.query(FaceExemplar).filter(
                FaceExemplar.identity_id == sid
            ).update({FaceExemplar.identity_id: target_id}, synchronize_session=False)
            session.query(AppearanceExemplar).filter(
                AppearanceExemplar.identity_id == sid
            ).update(
                {AppearanceExemplar.identity_id: target_id},
                synchronize_session=False,
            )
            # Repoint enriched Events (denormalized link).
            session.query(Event).filter(Event.identity_id == sid).update(
                {Event.identity_id: target_id}, synchronize_session=False
            )
            # Carry over first/last seen.
            if src.first_seen and (
                target.first_seen is None or src.first_seen < target.first_seen
            ):
                target.first_seen = src.first_seen
            if src.last_seen and (
                target.last_seen is None or src.last_seen > target.last_seen
            ):
                target.last_seen = src.last_seen
            session.delete(src)

        session.flush()
        self._refresh_identity_aggregates(session, target)
        return moved

    def split(
        self,
        session,
        identity_id: int,
        sighting_ids: Optional[list[int]] = None,
    ) -> int:
        """Split a chosen identity. If ``sighting_ids`` is given, those sightings
        (+ their exemplars) move to a NEW identity. If ``None``, re-cluster the
        identity's sightings into two by face (when available) else appearance,
        moving the minority cluster out. Returns the new identity id (or the
        original id when nothing was split). Commit NOT done here."""
        from app.db.models import (  # noqa: WPS433
            AppearanceExemplar,
            FaceExemplar,
            Identity,
            Sighting,
        )

        identity = session.get(Identity, identity_id)
        if identity is None:
            raise ValueError(f"split identity {identity_id} not found")

        if sighting_ids is None:
            sighting_ids = self._auto_split_sighting_ids(session, identity)
        if not sighting_ids:
            return identity_id

        all_sids = {
            s.id for s in session.query(Sighting.id)
            .filter(Sighting.identity_id == identity_id).all()
        }
        move = [sid for sid in sighting_ids if sid in all_sids]
        if not move or len(move) >= len(all_sids):
            # Nothing to do / would empty the original — refuse to split.
            return identity_id

        ts_now = datetime.utcnow()
        new_identity = Identity(
            name="",
            is_named=False,
            object_class=getattr(identity, "object_class", "person") or "person",
            num_sightings=0,
            first_seen=ts_now,
            last_seen=ts_now,
            is_provisional=False,
        )
        session.add(new_identity)
        session.flush()
        obj_class = getattr(identity, "object_class", "person") or "person"
        new_identity.name = f"{obj_class.capitalize()} {new_identity.id}"
        self.gallery.register_identity(int(new_identity.id), obj_class)

        session.query(Sighting).filter(Sighting.id.in_(move)).update(
            {Sighting.identity_id: new_identity.id}, synchronize_session=False
        )
        # Move exemplars tied to the moved sightings.
        for model in (FaceExemplar, AppearanceExemplar):
            session.query(model).filter(model.sighting_id.in_(move)).update(
                {model.identity_id: new_identity.id}, synchronize_session=False
            )
        session.flush()
        self._refresh_identity_aggregates(session, identity)
        self._refresh_identity_aggregates(session, new_identity)
        return int(new_identity.id)

    def _auto_split_sighting_ids(self, session, identity) -> list[int]:
        """2-means on the identity's face exemplar vectors (preferred) else its
        appearance exemplar vectors; returns the sighting ids of the minority
        cluster (the ones to move out). Empty when too few vectors / no split."""
        from app.db.models import (  # noqa: WPS433
            AppearanceExemplar,
            FaceExemplar,
        )

        rows = (
            session.query(FaceExemplar)
            .filter(FaceExemplar.identity_id == identity.id)
            .filter(FaceExemplar.sighting_id.isnot(None))
            .all()
        )
        if len(rows) < 4:
            rows = (
                session.query(AppearanceExemplar)
                .filter(AppearanceExemplar.identity_id == identity.id)
                .filter(AppearanceExemplar.sighting_id.isnot(None))
                .all()
            )
        if len(rows) < 4:
            return []

        vecs, sids = [], []
        for r in rows:
            try:
                vecs.append(normalize(deserialize_vector(r.vector)))
                sids.append(int(r.sighting_id))
            except Exception:
                continue
        if len(vecs) < 4:
            return []
        labels = _two_means(np.vstack(vecs))
        # Move the smaller cluster out (label with fewer members).
        ones = [sids[i] for i, l in enumerate(labels) if l == 1]
        zeros = [sids[i] for i, l in enumerate(labels) if l == 0]
        minority = ones if len(ones) <= len(zeros) else zeros
        # De-dup sighting ids.
        return sorted(set(minority))

    def _refresh_identity_aggregates(self, session, identity) -> None:
        """Recompute num_sightings, first/last_seen, face_centroid, rep_sighting."""
        from app.db.models import Sighting  # noqa: WPS433

        session.refresh(identity)
        sightings = (
            session.query(Sighting)
            .filter(Sighting.identity_id == identity.id)
            .order_by(Sighting.ts.asc())
            .all()
        )
        identity.num_sightings = len(sightings)
        if sightings:
            identity.first_seen = sightings[0].ts
            identity.last_seen = sightings[-1].ts
            if (
                identity.rep_sighting_id is None
                or identity.rep_sighting_id not in {s.id for s in sightings}
            ):
                identity.rep_sighting_id = sightings[0].id
        identity.face_centroid = self._recompute_face_centroid(session, identity)


def _two_means(mat: np.ndarray, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Tiny cosine 2-means (vectors are unit-norm -> cosine == dot). Returns a
    label array in {0,1}. Used only by the offline SPLIT helper."""
    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    # Seed centroids at the two most dissimilar points (cheap & deterministic-ish).
    sims = mat @ mat.T
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    c0, c1 = mat[i].copy(), mat[j].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        s0 = mat @ c0
        s1 = mat @ c1
        new_labels = (s1 > s0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for lbl, ref in ((0, "c0"), (1, "c1")):
            members = mat[labels == lbl]
            if members.shape[0] == 0:
                continue
            cen = members.mean(axis=0)
            norm = np.linalg.norm(cen)
            if norm > 0:
                cen = cen / norm
            if lbl == 0:
                c0 = cen
            else:
                c1 = cen
    return labels

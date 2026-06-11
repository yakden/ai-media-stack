"""Background maintenance for the auto-discovered identity library.

This module owns the *housekeeping* half of the Re-ID identity system. The
online assignment path (``app.reid.manager.IdentityManager``) creates and
updates identities on every sighting; this maintenance worker periodically
repairs and compacts that state so the gallery stays small, accurate and
cheap to query:

  * **Appearance time-decay + pruning** — each ``AppearanceExemplar`` carries a
    capture timestamp. We drop exemplars whose decay weight ``exp(-Δt/TAU)`` has
    fallen below a floor (older than ~2*TAU contributes nothing) and cap the
    per-identity count at ``max_app_exemplars`` keeping the freshest/highest
    quality. This encodes "people change clothes between days".
  * **Face exemplar pruning** — faces are time-stable (never decayed) but we
    still cap per identity at ``max_face_exemplars``, evicting lowest quality.
  * **Centroid + representative-thumbnail recompute** — ``face_centroid`` is the
    L2-normalized running mean of the face exemplars; ``appearance_centroid`` is
    the decay-weighted normalized mean of the appearance exemplars. The
    ``rep_sighting_id`` is refreshed to the best (face-bearing, high-score)
    recent sighting so the gallery has a good thumbnail.
  * **Provisional-identity cleanup** — a NEW identity with a single low-evidence
    sighting and no face is provisional; if it never accrued a second sighting
    or a face within ``provisional_grace_seconds`` it is deleted (detector
    noise, not a real person).
  * **Conservative face-only auto-merge** — repairs over-segmentation by merging
    two AUTO (un-named) identities whose face centroids are very close
    (``face_merge_threshold``) AND which were never seen at the same time on
    different cameras (can't be in two places at once). Appearance never merges
    (clothes are shared/ambiguous); ``is_named`` identities are frozen.

Design / conventions (mirrors the rest of the codebase):
  * The SQLite DB is the single source of truth. Vectors are 512-d little-endian
    float32 L2-normalized BLOBs, (de)serialized via ``app.faces.index``.
  * Heavy/optional deps (ORM models, the API-process gallery) are imported
    lazily so this module is importable and unit-testable in isolation.
  * ``run_once(session)`` does one full pass against a caller-supplied session
    and is pure/deterministic (no threads), so it is trivially unit-testable.
  * ``start(app_state)`` / ``stop()`` run ``run_once`` on a timer in a daemon
    thread; the integration component calls them from ``main.lifespan``.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger("vms.reid.maintenance")

EMBEDDING_DIM = 512

# Decay weight below which an appearance exemplar is considered dead and pruned.
# exp(-2) ~= 0.135, so anything older than ~2*TAU stops contributing.
_DECAY_PRUNE_FLOOR = math.exp(-2.0)


# ---------------------------------------------------------------------------
# Tunables (read from Settings with safe fallbacks so this module works even
# before the integration component lands the REID_* config).
# ---------------------------------------------------------------------------


@dataclass
class MaintenanceConfig:
    """Resolved knobs for one maintenance pass.

    Defaults match the authoritative matching_algorithm/config contract; the
    real values come from ``Settings`` via :meth:`from_settings`.
    """

    interval_seconds: float = 60.0
    app_decay_tau_seconds: float = 12 * 3600.0
    max_face_exemplars: int = 8
    max_app_exemplars: int = 16
    provisional_grace_seconds: float = 600.0
    face_merge_threshold: float = 0.6
    # Appearance exemplars with a decay weight below this floor are pruned.
    decay_prune_floor: float = _DECAY_PRUNE_FLOOR
    # Hard age cap (seconds) for appearance exemplars regardless of decay floor;
    # 0/None disables. Defaults to 7 days so the library cannot grow unbounded
    # for an identity that keeps re-matching at the very tail of the window.
    app_max_age_seconds: float = 7 * 24 * 3600.0
    enabled: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "MaintenanceConfig":
        def g(name: str, default):
            return getattr(settings, name, default)

        return cls(
            interval_seconds=float(g("reid_maintenance_interval_seconds", 60.0)),
            app_decay_tau_seconds=float(g("app_decay_tau_seconds", 12 * 3600.0)),
            max_face_exemplars=int(g("max_face_exemplars", 8)),
            max_app_exemplars=int(g("max_app_exemplars", 16)),
            provisional_grace_seconds=float(g("provisional_grace_seconds", 600.0)),
            face_merge_threshold=float(g("face_merge_threshold", 0.6)),
            app_max_age_seconds=float(g("reid_app_max_age_seconds", 7 * 24 * 3600.0)),
            enabled=bool(g("reid_enabled", True)),
        )


@dataclass
class MaintenanceStats:
    """Counters returned by :func:`run_once` (handy for logging/tests)."""

    identities_scanned: int = 0
    app_exemplars_pruned: int = 0
    face_exemplars_pruned: int = 0
    centroids_recomputed: int = 0
    thumbnails_updated: int = 0
    provisional_deleted: int = 0
    merges: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return {
            "identities_scanned": self.identities_scanned,
            "app_exemplars_pruned": self.app_exemplars_pruned,
            "face_exemplars_pruned": self.face_exemplars_pruned,
            "centroids_recomputed": self.centroids_recomputed,
            "thumbnails_updated": self.thumbnails_updated,
            "provisional_deleted": self.provisional_deleted,
            "merges": self.merges,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Vector / decay helpers (self-contained; tolerant of app.reid.decay absence).
# ---------------------------------------------------------------------------


def _deserialize(blob: bytes) -> Optional[np.ndarray]:
    """512-d float32 from a BLOB, or ``None`` if it is malformed."""
    try:
        from app.faces.index import deserialize_vector  # noqa: WPS433

        return deserialize_vector(blob)
    except Exception:
        # Fall back to a local decode so maintenance never hard-depends on the
        # faces package being importable.
        try:
            arr = np.frombuffer(blob, dtype="<f4")
            if arr.shape[0] != EMBEDDING_DIM:
                return None
            return np.array(arr, dtype=np.float32)
        except Exception:
            return None


def _serialize(vec: np.ndarray) -> bytes:
    try:
        from app.faces.index import serialize_vector  # noqa: WPS433

        return serialize_vector(vec)
    except Exception:
        return np.asarray(vec, dtype="<f4").reshape(-1).tobytes()


def _normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return (vec / norm).astype(np.float32)


def decay_weight(age_seconds: float, tau_seconds: float) -> float:
    """``exp(-Δt/TAU)`` clamped to [0, 1]. Negative ages treated as fresh."""
    if tau_seconds <= 0:
        return 1.0
    if age_seconds <= 0:
        return 1.0
    try:
        from app.reid.decay import decay_weight as _dw  # noqa: WPS433

        return float(_dw(age_seconds, tau_seconds))
    except Exception:
        return float(math.exp(-age_seconds / tau_seconds))


def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize to naive-UTC so arithmetic with ``datetime.utcnow()`` is safe.

    The ORM stores naive datetimes (SQLite). We defensively strip tzinfo if a
    caller passes an aware datetime.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ---------------------------------------------------------------------------
# Per-identity maintenance steps. Each takes the loaded ORM objects and mutates
# them on the session (the caller commits once per pass).
# ---------------------------------------------------------------------------


def _exemplar_ts(ex: Any) -> Optional[datetime]:
    """Best-available timestamp for an appearance exemplar (ts -> created_at)."""
    return _to_naive_utc(getattr(ex, "ts", None) or getattr(ex, "created_at", None))


def _prune_appearance_exemplars(
    session,
    identity: Any,
    cfg: MaintenanceConfig,
    now: datetime,
) -> int:
    """Drop dead/over-cap appearance exemplars. Returns number deleted.

    Eviction order: dead-by-decay (or dead-by-hard-age) first, then, if still
    over ``max_app_exemplars``, lowest combined (quality * decay) score.
    """
    exemplars = list(getattr(identity, "appearance_exemplars", []) or [])
    if not exemplars:
        return 0

    deleted = 0
    survivors: list[tuple[Any, float, float]] = []  # (ex, decay_w, score)
    for ex in exemplars:
        ts = _exemplar_ts(ex)
        age = (now - ts).total_seconds() if ts is not None else 0.0
        w = decay_weight(age, cfg.app_decay_tau_seconds)
        too_old = (
            cfg.app_max_age_seconds
            and cfg.app_max_age_seconds > 0
            and age > cfg.app_max_age_seconds
        )
        if w < cfg.decay_prune_floor or too_old:
            session.delete(ex)
            deleted += 1
            continue
        quality = float(getattr(ex, "quality", 0.0) or 0.0)
        survivors.append((ex, w, quality * max(w, 1e-6)))

    # Cap: keep the highest combined-score exemplars.
    if len(survivors) > cfg.max_app_exemplars:
        survivors.sort(key=lambda t: t[2], reverse=True)
        for ex, _w, _s in survivors[cfg.max_app_exemplars:]:
            session.delete(ex)
            deleted += 1
        survivors = survivors[: cfg.max_app_exemplars]

    return deleted


def _prune_face_exemplars(session, identity: Any, cfg: MaintenanceConfig) -> int:
    """Cap face exemplars at ``max_face_exemplars`` (evict lowest det_score).

    Faces are time-stable so they are NOT decayed — only count-capped. Returns
    the number deleted.
    """
    exemplars = list(getattr(identity, "face_exemplars", []) or [])
    if len(exemplars) <= cfg.max_face_exemplars:
        return 0
    # Keep highest-quality (det_score) exemplars.
    exemplars.sort(key=lambda e: float(getattr(e, "det_score", 0.0) or 0.0), reverse=True)
    deleted = 0
    for ex in exemplars[cfg.max_face_exemplars:]:
        session.delete(ex)
        deleted += 1
    return deleted


def _recompute_face_centroid(identity: Any) -> Optional[np.ndarray]:
    """L2-normalized mean of the (non-decayed) face exemplar vectors."""
    vecs: list[np.ndarray] = []
    for ex in getattr(identity, "face_exemplars", []) or []:
        v = _deserialize(getattr(ex, "vector", b""))
        if v is not None:
            vecs.append(_normalize(v))
    if not vecs:
        return None
    mean = np.mean(np.vstack(vecs), axis=0)
    return _normalize(mean)


def _recompute_appearance_centroid(
    identity: Any, cfg: MaintenanceConfig, now: datetime
) -> Optional[np.ndarray]:
    """Decay-weighted, L2-normalized mean of appearance exemplar vectors."""
    acc = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    total_w = 0.0
    for ex in getattr(identity, "appearance_exemplars", []) or []:
        v = _deserialize(getattr(ex, "vector", b""))
        if v is None:
            continue
        ts = _exemplar_ts(ex)
        age = (now - ts).total_seconds() if ts is not None else 0.0
        w = decay_weight(age, cfg.app_decay_tau_seconds)
        quality = float(getattr(ex, "quality", 1.0) or 1.0)
        weight = w * max(quality, 1e-6)
        if weight <= 0:
            continue
        acc += _normalize(v).astype(np.float64) * weight
        total_w += weight
    if total_w <= 0:
        return None
    return _normalize((acc / total_w).astype(np.float32))


def _pick_rep_sighting(identity: Any) -> Optional[int]:
    """Choose a representative sighting id: prefer face-bearing, high-score,
    most-recent crops that actually have a thumbnail."""
    best = None
    best_key: tuple = ()
    for s in getattr(identity, "sightings", []) or []:
        if not getattr(s, "thumb_path", None):
            continue
        ts = _to_naive_utc(getattr(s, "ts", None)) or datetime.min
        key = (
            1 if getattr(s, "has_face", False) else 0,
            float(getattr(s, "face_score", 0.0) or 0.0),
            float(getattr(s, "det_score", 0.0) or 0.0),
            ts,
        )
        if best is None or key > best_key:
            best = s
            best_key = key
    return int(best.id) if best is not None and getattr(best, "id", None) else None


def _maintain_identity(
    session,
    identity: Any,
    cfg: MaintenanceConfig,
    now: datetime,
    stats: MaintenanceStats,
) -> None:
    """Prune exemplars + recompute centroids/thumb/counters for one identity."""
    stats.app_exemplars_pruned += _prune_appearance_exemplars(session, identity, cfg, now)
    stats.face_exemplars_pruned += _prune_face_exemplars(session, identity, cfg)
    # Make deletions visible to the relationship collections before recompute.
    session.flush()

    face_c = _recompute_face_centroid(identity)
    app_c = _recompute_appearance_centroid(identity, cfg, now)
    if hasattr(identity, "face_centroid"):
        identity.face_centroid = _serialize(face_c) if face_c is not None else None
    if hasattr(identity, "appearance_centroid"):
        identity.appearance_centroid = (
            _serialize(app_c) if app_c is not None else None
        )
    stats.centroids_recomputed += 1

    rep = _pick_rep_sighting(identity)
    if rep is not None and getattr(identity, "rep_sighting_id", None) != rep:
        identity.rep_sighting_id = rep
        stats.thumbnails_updated += 1

    # Keep the denormalized counter + first/last_seen honest.
    sightings = list(getattr(identity, "sightings", []) or [])
    if hasattr(identity, "num_sightings"):
        identity.num_sightings = len(sightings)
    if sightings:
        ts_values = [_to_naive_utc(getattr(s, "ts", None)) for s in sightings]
        ts_values = [t for t in ts_values if t is not None]
        if ts_values:
            if hasattr(identity, "first_seen"):
                identity.first_seen = min(ts_values)
            if hasattr(identity, "last_seen"):
                identity.last_seen = max(ts_values)


# ---------------------------------------------------------------------------
# Provisional-identity cleanup.
# ---------------------------------------------------------------------------


def _is_named(identity: Any) -> bool:
    return bool(getattr(identity, "is_named", False))


def _has_face_evidence(identity: Any) -> bool:
    if getattr(identity, "face_exemplars", None):
        return len(list(identity.face_exemplars)) > 0
    if getattr(identity, "face_centroid", None):
        return True
    # Fall back to scanning sightings.
    for s in getattr(identity, "sightings", []) or []:
        if getattr(s, "has_face", False):
            return True
    return False


def _cleanup_provisional(
    session, identity: Any, cfg: MaintenanceConfig, now: datetime
) -> bool:
    """Delete an identity that never matured into a real person.

    A provisional identity has: not named, no face evidence, <= 1 sighting, and
    is older than the grace period. Returns True if deleted.
    """
    if _is_named(identity):
        return False
    if _has_face_evidence(identity):
        return False
    sightings = list(getattr(identity, "sightings", []) or [])
    if len(sightings) > 1:
        return False

    created = _to_naive_utc(getattr(identity, "created_at", None))
    last_seen = _to_naive_utc(getattr(identity, "last_seen", None))
    reference = last_seen or created
    if reference is None:
        # No timing info: leave it alone (safer than deleting blind).
        return False
    age = (now - reference).total_seconds()
    if age < cfg.provisional_grace_seconds:
        return False

    session.delete(identity)
    return True


# ---------------------------------------------------------------------------
# Conservative face-only auto-merge.
# ---------------------------------------------------------------------------


def _sighting_intervals(identity: Any) -> list[tuple[int, datetime]]:
    """List of (camera_id, ts) for an identity's sightings (naive-UTC ts)."""
    out: list[tuple[int, datetime]] = []
    for s in getattr(identity, "sightings", []) or []:
        ts = _to_naive_utc(getattr(s, "ts", None))
        cam = getattr(s, "camera_id", None)
        if ts is not None and cam is not None:
            out.append((int(cam), ts))
    return out


def _temporally_conflicting(
    a: Any, b: Any, window_seconds: float = 5.0
) -> bool:
    """True if A and B were seen on DIFFERENT cameras within ``window_seconds``
    of each other (can't be the same person in two places at once)."""
    sa = _sighting_intervals(a)
    sb = _sighting_intervals(b)
    if not sa or not sb:
        return False
    # O(n*m) is fine: per-identity sighting counts are small after pruning, and
    # this only runs in background maintenance.
    for cam_a, ts_a in sa:
        for cam_b, ts_b in sb:
            if cam_a != cam_b and abs((ts_a - ts_b).total_seconds()) <= window_seconds:
                return True
    return False


def _merge_into(session, target: Any, source: Any, stats: MaintenanceStats) -> None:
    """Reassign all of ``source``'s sightings/exemplars to ``target`` and delete
    ``source``. Centroids/counters are recomputed on the next per-identity pass;
    callers should re-run maintenance or recompute explicitly afterwards."""
    tid = target.id
    for s in list(getattr(source, "sightings", []) or []):
        s.identity_id = tid
    for fx in list(getattr(source, "face_exemplars", []) or []):
        fx.identity_id = tid
    for ax in list(getattr(source, "appearance_exemplars", []) or []):
        ax.identity_id = tid
    # If the target's rep thumbnail came from source, it stays valid (sighting
    # was reassigned, not deleted). Drop source's rep pointer to avoid a stale
    # FK if the source row's rep_sighting_id pointed at a now-reassigned row.
    if hasattr(source, "rep_sighting_id"):
        source.rep_sighting_id = None
    session.flush()
    session.delete(source)
    stats.merges += 1
    logger.info("Auto-merged identity %s into %s (face-only)", source.id, target.id)


def _auto_merge_pass(
    session, identities: Sequence[Any], cfg: MaintenanceConfig, now: datetime, stats: MaintenanceStats
) -> set[int]:
    """Conservative face-only merge of over-segmented AUTO identities.

    Two identities are merged when BOTH:
      * face centroids cosine >= ``face_merge_threshold``, and
      * they were never seen on different cameras at the same instant.
    Named identities are frozen (never auto-merged). Returns the set of merged
    (deleted) source identity ids so the caller skips them.
    """
    # Gather (identity, face_centroid) for un-named identities with a face.
    candidates: list[tuple[Any, np.ndarray]] = []
    for ident in identities:
        if _is_named(ident):
            continue
        c = getattr(ident, "face_centroid", None)
        vec = _deserialize(c) if c else None
        if vec is None:
            # Try to derive on the fly from exemplars.
            vec = _recompute_face_centroid(ident)
        if vec is not None:
            candidates.append((ident, _normalize(vec)))

    merged_away: set[int] = set()
    if len(candidates) < 2:
        return merged_away

    # Union-find of merges (target = earliest id, the canonical survivor).
    canonical: dict[int, Any] = {}

    def resolve(ident: Any) -> Any:
        cur = ident
        seen = set()
        while cur.id in canonical and cur.id not in seen:
            seen.add(cur.id)
            cur = canonical[cur.id]
        return cur

    n = len(candidates)
    for i in range(n):
        ident_a, vec_a = candidates[i]
        if ident_a.id in merged_away:
            continue
        for j in range(i + 1, n):
            ident_b, vec_b = candidates[j]
            if ident_b.id in merged_away:
                continue
            cos = float(np.dot(vec_a, vec_b))
            if cos < cfg.face_merge_threshold:
                continue
            tgt = resolve(ident_a)
            src = resolve(ident_b)
            if tgt.id == src.id:
                continue
            # Keep the lower id as the canonical survivor (stable "Person N").
            if src.id < tgt.id:
                tgt, src = src, tgt
            if _is_named(src):
                # Should not happen (filtered), but never merge away a named one.
                continue
            if _temporally_conflicting(tgt, src):
                continue
            _merge_into(session, tgt, src, stats)
            merged_away.add(src.id)
            canonical[src.id] = tgt

    return merged_away


# ---------------------------------------------------------------------------
# Public: a single full maintenance pass.
# ---------------------------------------------------------------------------


def run_once(
    session,
    settings: Any = None,
    cfg: Optional[MaintenanceConfig] = None,
    now: Optional[datetime] = None,
) -> MaintenanceStats:
    """Run one complete maintenance pass against ``session``.

    Steps, in order:
      1. Load all identities (eager enough via lazy relationship access).
      2. Provisional cleanup (delete noise identities).
      3. Conservative face-only auto-merge of remaining AUTO identities.
      4. Per-identity prune (appearance decay + caps, face caps) and recompute
         (centroids, rep thumbnail, counters, first/last_seen).
      5. Commit once.

    The caller owns the session lifecycle; on any error the transaction is
    rolled back and the error is counted. Returns :class:`MaintenanceStats`.
    """
    if cfg is None:
        cfg = MaintenanceConfig.from_settings(settings) if settings is not None else MaintenanceConfig()
    if now is None:
        now = datetime.utcnow()
    now = _to_naive_utc(now) or datetime.utcnow()

    stats = MaintenanceStats()

    try:
        from app.db.models import Identity  # noqa: WPS433
    except Exception:
        logger.debug("Identity model unavailable; maintenance is a no-op")
        return stats

    try:
        identities = list(session.query(Identity).order_by(Identity.id.asc()).all())
        stats.identities_scanned = len(identities)

        # 1) Provisional cleanup.
        survivors: list[Any] = []
        for ident in identities:
            try:
                if _cleanup_provisional(session, ident, cfg, now):
                    stats.provisional_deleted += 1
                else:
                    survivors.append(ident)
            except Exception:
                logger.exception("Provisional cleanup failed for identity %s", getattr(ident, "id", "?"))
                stats.errors += 1
                survivors.append(ident)
        session.flush()

        # 2) Conservative face-only auto-merge.
        try:
            merged_away = _auto_merge_pass(session, survivors, cfg, now, stats)
        except Exception:
            logger.exception("Auto-merge pass failed")
            stats.errors += 1
            merged_away = set()
        session.flush()

        # 3) Per-identity prune + recompute (skip merged-away sources).
        for ident in survivors:
            if getattr(ident, "id", None) in merged_away:
                continue
            try:
                _maintain_identity(session, ident, cfg, now, stats)
            except Exception:
                logger.exception("Maintenance failed for identity %s", getattr(ident, "id", "?"))
                stats.errors += 1

        session.commit()
    except Exception:
        logger.exception("Maintenance pass failed; rolling back")
        stats.errors += 1
        try:
            session.rollback()
        except Exception:
            logger.exception("Rollback after maintenance failure also failed")

    logger.info("Maintenance pass: %s", stats.as_dict())
    return stats


# ---------------------------------------------------------------------------
# Background thread: start(app_state) / stop().
# ---------------------------------------------------------------------------


@dataclass
class _MaintenanceThread:
    cfg: MaintenanceConfig
    app_state: Any
    settings: Any
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="reid-maintenance", daemon=True
        )
        self._thread.start()
        logger.info(
            "Re-ID maintenance thread started (interval=%.0fs)", self.cfg.interval_seconds
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        logger.info("Re-ID maintenance thread stopped")

    # -- internals -----------------------------------------------------------

    def _loop(self) -> None:
        # First pass after a short delay so startup isn't contended.
        if self._stop.wait(min(self.cfg.interval_seconds, 30.0)):
            return
        while not self._stop.is_set():
            self._tick()
            # Interruptible sleep until the next interval.
            if self._stop.wait(self.cfg.interval_seconds):
                break

    def _tick(self) -> None:
        session = None
        try:
            from app.db.database import SessionLocal  # noqa: WPS433

            session = SessionLocal()
            run_once(session, settings=self.settings, cfg=self.cfg)
        except Exception:
            logger.exception("Re-ID maintenance tick failed")
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            # After a pass, ask the API-process gallery to reload so operator
            # UIs and any in-process matching see the compacted state.
            self._reload_gallery()

    def _reload_gallery(self) -> None:
        gallery = getattr(self.app_state, "identity_gallery", None)
        if gallery is None:
            return
        reload_fn = getattr(gallery, "reload", None) or getattr(
            gallery, "rebuild_from_db", None
        )
        if reload_fn is None:
            return
        session = None
        try:
            from app.db.database import SessionLocal  # noqa: WPS433

            # rebuild_from_db needs a session; reload() may not. Try both shapes.
            try:
                session = SessionLocal()
                reload_fn(session)
                session.commit()
            except TypeError:
                reload_fn()
        except Exception:
            logger.exception("Identity gallery reload after maintenance failed")
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass


# Module-level handle so start/stop can be called functionally from lifespan.
_RUNNER: Optional[_MaintenanceThread] = None


def start(app_state: Any, settings: Any = None) -> Optional[_MaintenanceThread]:
    """Start the background maintenance thread.

    ``app_state`` is typically ``app.state`` (used to find ``identity_gallery``
    for post-pass reloads and ``settings`` if not passed explicitly). Returns
    the runner (or ``None`` when Re-ID is disabled). Idempotent.
    """
    global _RUNNER

    if settings is None:
        settings = getattr(app_state, "settings", None)
        if settings is None:
            try:
                from app.config import get_settings  # noqa: WPS433

                settings = get_settings()
            except Exception:
                settings = None

    cfg = MaintenanceConfig.from_settings(settings) if settings is not None else MaintenanceConfig()
    if not cfg.enabled:
        logger.info("Re-ID disabled; maintenance thread not started")
        return None

    if _RUNNER is not None:
        _RUNNER.stop()
    _RUNNER = _MaintenanceThread(cfg=cfg, app_state=app_state, settings=settings)
    _RUNNER.start()
    return _RUNNER


def stop() -> None:
    """Stop the background maintenance thread if running. Idempotent."""
    global _RUNNER
    if _RUNNER is not None:
        _RUNNER.stop()
        _RUNNER = None

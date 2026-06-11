"""Tests for the Re-ID maintenance worker (app.reid.maintenance).

These exercise the *logic* of one maintenance pass with no GPU, no DB engine
and no ORM dependency: the maintenance module is written to duck-type the ORM
(it only touches attributes/relationships and ``session.delete/flush/commit``),
so we drive it with lightweight in-memory fakes. The DB-backed ``run_once``
query path (``session.query(Identity)``) is covered with a fake session whose
``query(...).order_by(...).all()`` returns our fake identities, and the
``app.db.models.Identity`` import is satisfied with a stub module.

Covered:
  * decay_weight math (monotonic, exp shape, edge cases)
  * appearance exemplar pruning: dead-by-decay, hard-age cap, count cap
  * face exemplar count cap (lowest det_score evicted; no decay)
  * centroid recompute (face = plain mean; appearance = decay-weighted mean)
  * representative thumbnail selection (face-bearing / high score / recent)
  * provisional cleanup (deleted only when stale, no face, <=1 sighting, auto)
  * conservative face-only auto-merge (close centroids merge; named frozen;
    temporal/different-camera conflict vetoes a merge; appearance never merges)
  * run_once orchestration + counters
"""

from __future__ import annotations

import math
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# Make ``vms/`` importable so ``app.reid.maintenance`` resolves.
VMS_ROOT = Path(__file__).resolve().parents[1]
if str(VMS_ROOT) not in sys.path:
    sys.path.insert(0, str(VMS_ROOT))


maint = pytest.importorskip(
    "app.reid.maintenance", reason="app.reid.maintenance not available"
)

DIM = maint.EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Vector helpers + in-memory ORM fakes.
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    return (vec / n).astype(np.float32) if n else vec


def _rand_unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(DIM).astype(np.float32))


def _near(base: np.ndarray, jitter: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(base + rng.standard_normal(DIM).astype(np.float32) * jitter)


def _blob(vec: np.ndarray) -> bytes:
    return np.asarray(_unit(vec), dtype="<f4").reshape(-1).tobytes()


_NEXT_ID = [1]


def _new_id() -> int:
    v = _NEXT_ID[0]
    _NEXT_ID[0] += 1
    return v


class FaceExemplar:
    def __init__(self, vector, det_score=0.7):
        self.id = _new_id()
        self.identity_id = None
        self.vector = _blob(vector)
        self.det_score = det_score
        self.created_at = datetime.utcnow()


class AppearanceExemplar:
    def __init__(self, vector, ts, quality=1.0):
        self.id = _new_id()
        self.identity_id = None
        self.vector = _blob(vector)
        self.ts = ts
        self.quality = quality
        self.created_at = ts


class Sighting:
    def __init__(self, camera_id, ts, has_face=False, face_score=0.0, det_score=0.5, thumb_path="t.jpg"):
        self.id = _new_id()
        self.identity_id = None
        self.camera_id = camera_id
        self.ts = ts
        self.has_face = has_face
        self.face_score = face_score
        self.det_score = det_score
        self.thumb_path = thumb_path


class Identity:
    def __init__(self, name="Person", is_named=False, created_at=None):
        self.id = _new_id()
        self.name = name
        self.is_named = is_named
        self.notes = None
        self.rep_sighting_id = None
        self.face_centroid = None
        self.appearance_centroid = None
        self.num_sightings = 0
        self.first_seen = None
        self.last_seen = None
        self.created_at = created_at or datetime.utcnow()
        self.sightings = []
        self.face_exemplars = []
        self.appearance_exemplars = []

    def add_sighting(self, s):
        s.identity_id = self.id
        self.sightings.append(s)
        return s

    def add_face(self, fx):
        fx.identity_id = self.id
        self.face_exemplars.append(fx)
        return fx

    def add_app(self, ax):
        ax.identity_id = self.id
        self.appearance_exemplars.append(ax)
        return ax


class FakeSession:
    """Minimal SQLAlchemy-session surface used by maintenance.

    Supports ``delete`` (removes object from its parent identity collections and
    records it), ``flush``/``commit``/``rollback`` (no-ops) and a ``query``
    that returns the registered identities for the run_once path.
    """

    def __init__(self, identities):
        self._identities = list(identities)
        self.deleted = []

    # query(Identity).order_by(Identity.id.asc()).all()
    def query(self, model):
        return _FakeQuery(self._identities)

    def delete(self, obj):
        self.deleted.append(obj)
        if isinstance(obj, Identity):
            if obj in self._identities:
                self._identities.remove(obj)
            return
        # Exemplar: detach from any identity collection.
        for ident in self._identities:
            for coll in (ident.face_exemplars, ident.appearance_exemplars):
                if obj in coll:
                    coll.remove(obj)

    def flush(self):
        # Apply pending sighting/exemplar reassignments to collections so
        # relationship access reflects merges (maintenance reassigns by
        # identity_id then deletes the source; mimic by re-bucketing here).
        self._rebucket()

    def commit(self):
        self._rebucket()

    def rollback(self):
        pass

    def close(self):
        pass

    def _rebucket(self):
        by_id = {i.id: i for i in self._identities}
        for ident in list(self._identities):
            for s in list(ident.sightings):
                if s.identity_id != ident.id and s.identity_id in by_id:
                    ident.sightings.remove(s)
                    by_id[s.identity_id].sightings.append(s)
            for fx in list(ident.face_exemplars):
                if fx.identity_id != ident.id and fx.identity_id in by_id:
                    ident.face_exemplars.remove(fx)
                    by_id[fx.identity_id].face_exemplars.append(fx)
            for ax in list(ident.appearance_exemplars):
                if ax.identity_id != ident.id and ax.identity_id in by_id:
                    ident.appearance_exemplars.remove(ax)
                    by_id[ax.identity_id].appearance_exemplars.append(ax)


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def order_by(self, *_a, **_k):
        return self

    def all(self):
        return list(self._rows)


@pytest.fixture(autouse=True)
def _stub_identity_model(monkeypatch):
    """Provide a stub ``app.db.models`` with an ``Identity`` having ``id`` so
    run_once's ``query(Identity).order_by(Identity.id.asc())`` import works."""
    mod = types.ModuleType("app.db.models")

    class _Ident:
        id = types.SimpleNamespace(asc=lambda: None)

    mod.Identity = _Ident
    monkeypatch.setitem(sys.modules, "app.db.models", mod)
    yield


# ---------------------------------------------------------------------------
# decay_weight
# ---------------------------------------------------------------------------


def test_decay_weight_shape():
    tau = 1000.0
    assert maint.decay_weight(0.0, tau) == pytest.approx(1.0)
    assert maint.decay_weight(-5.0, tau) == pytest.approx(1.0)  # future -> fresh
    assert maint.decay_weight(tau, tau) == pytest.approx(math.exp(-1.0), rel=1e-6)
    assert maint.decay_weight(2 * tau, tau) == pytest.approx(math.exp(-2.0), rel=1e-6)
    # Monotonic decreasing.
    assert maint.decay_weight(10, tau) > maint.decay_weight(100, tau)


def test_decay_weight_zero_tau_is_one():
    assert maint.decay_weight(99999.0, 0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Appearance pruning
# ---------------------------------------------------------------------------


def test_prune_dead_appearance_exemplars():
    cfg = maint.MaintenanceConfig(app_decay_tau_seconds=3600.0, max_app_exemplars=16)
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(1)
    ident = Identity()
    fresh = ident.add_app(AppearanceExemplar(base, ts=now - timedelta(seconds=10)))
    # Older than 2*TAU -> decay weight below the prune floor -> deleted.
    dead = ident.add_app(AppearanceExemplar(base, ts=now - timedelta(hours=10)))

    sess = FakeSession([ident])
    deleted = maint._prune_appearance_exemplars(sess, ident, cfg, now)
    assert deleted == 1
    assert dead in sess.deleted
    assert fresh not in sess.deleted
    assert fresh in ident.appearance_exemplars
    assert dead not in ident.appearance_exemplars


def test_prune_appearance_hard_age_cap():
    cfg = maint.MaintenanceConfig(
        app_decay_tau_seconds=10 * 24 * 3600.0,  # huge TAU so decay floor never trips
        app_max_age_seconds=3600.0,
        max_app_exemplars=16,
    )
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(2)
    ident = Identity()
    young = ident.add_app(AppearanceExemplar(base, ts=now - timedelta(minutes=10)))
    old = ident.add_app(AppearanceExemplar(base, ts=now - timedelta(hours=2)))

    sess = FakeSession([ident])
    deleted = maint._prune_appearance_exemplars(sess, ident, cfg, now)
    assert deleted == 1
    assert old in sess.deleted
    assert young in ident.appearance_exemplars


def test_prune_appearance_count_cap_keeps_best():
    cfg = maint.MaintenanceConfig(
        app_decay_tau_seconds=10 * 24 * 3600.0, app_max_age_seconds=0, max_app_exemplars=2
    )
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(3)
    ident = Identity()
    # Three live exemplars with distinct quality; lowest quality evicted.
    hi = ident.add_app(AppearanceExemplar(base, ts=now - timedelta(minutes=1), quality=0.9))
    mid = ident.add_app(AppearanceExemplar(base, ts=now - timedelta(minutes=1), quality=0.5))
    lo = ident.add_app(AppearanceExemplar(base, ts=now - timedelta(minutes=1), quality=0.1))

    sess = FakeSession([ident])
    deleted = maint._prune_appearance_exemplars(sess, ident, cfg, now)
    assert deleted == 1
    assert lo in sess.deleted
    assert hi in ident.appearance_exemplars and mid in ident.appearance_exemplars


# ---------------------------------------------------------------------------
# Face pruning (count cap, no decay)
# ---------------------------------------------------------------------------


def test_prune_face_count_cap_evicts_low_score():
    cfg = maint.MaintenanceConfig(max_face_exemplars=2)
    ident = Identity()
    good1 = ident.add_face(FaceExemplar(_rand_unit(10), det_score=0.9))
    good2 = ident.add_face(FaceExemplar(_rand_unit(11), det_score=0.8))
    weak = ident.add_face(FaceExemplar(_rand_unit(12), det_score=0.3))

    sess = FakeSession([ident])
    deleted = maint._prune_face_exemplars(sess, ident, cfg)
    assert deleted == 1
    assert weak in sess.deleted
    assert good1 in ident.face_exemplars and good2 in ident.face_exemplars


def test_prune_face_under_cap_noop():
    cfg = maint.MaintenanceConfig(max_face_exemplars=8)
    ident = Identity()
    ident.add_face(FaceExemplar(_rand_unit(13)))
    sess = FakeSession([ident])
    assert maint._prune_face_exemplars(sess, ident, cfg) == 0


# ---------------------------------------------------------------------------
# Centroid recompute
# ---------------------------------------------------------------------------


def test_face_centroid_is_normalized_mean():
    ident = Identity()
    a = _rand_unit(20)
    b = _near(a, jitter=0.05, seed=21)
    ident.add_face(FaceExemplar(a))
    ident.add_face(FaceExemplar(b))
    c = maint._recompute_face_centroid(ident)
    assert c is not None
    assert float(np.linalg.norm(c)) == pytest.approx(1.0, abs=1e-5)
    # Mean of two near-duplicates stays close to both.
    assert float(c @ _unit(a)) > 0.9


def test_face_centroid_none_when_empty():
    assert maint._recompute_face_centroid(Identity()) is None


def test_appearance_centroid_decay_weighted():
    cfg = maint.MaintenanceConfig(app_decay_tau_seconds=3600.0)
    now = datetime(2026, 6, 6, 12, 0, 0)
    ident = Identity()
    recent = _rand_unit(30)
    old = _rand_unit(31)  # near-orthogonal to recent
    ident.add_app(AppearanceExemplar(recent, ts=now - timedelta(seconds=1)))
    ident.add_app(AppearanceExemplar(old, ts=now - timedelta(hours=3)))  # heavily decayed
    c = maint._recompute_appearance_centroid(ident, cfg, now)
    assert c is not None
    assert float(np.linalg.norm(c)) == pytest.approx(1.0, abs=1e-5)
    # Centroid dominated by the recent (high-weight) exemplar.
    assert float(c @ _unit(recent)) > float(c @ _unit(old))


# ---------------------------------------------------------------------------
# Representative thumbnail selection
# ---------------------------------------------------------------------------


def test_rep_sighting_prefers_face_bearing():
    ident = Identity()
    now = datetime(2026, 6, 6, 12, 0, 0)
    no_face = ident.add_sighting(Sighting(1, now, has_face=False, det_score=0.99))
    face = ident.add_sighting(Sighting(1, now - timedelta(minutes=5), has_face=True, face_score=0.7))
    rep = maint._pick_rep_sighting(ident)
    assert rep == face.id and rep != no_face.id


def test_rep_sighting_skips_missing_thumb():
    ident = Identity()
    now = datetime(2026, 6, 6, 12, 0, 0)
    ident.add_sighting(Sighting(1, now, has_face=True, face_score=0.9, thumb_path=None))
    assert maint._pick_rep_sighting(ident) is None


# ---------------------------------------------------------------------------
# Provisional cleanup
# ---------------------------------------------------------------------------


def test_provisional_deleted_when_stale_and_no_face():
    cfg = maint.MaintenanceConfig(provisional_grace_seconds=600.0)
    now = datetime(2026, 6, 6, 12, 0, 0)
    ident = Identity(created_at=now - timedelta(minutes=20))
    s = ident.add_sighting(Sighting(1, now - timedelta(minutes=20), has_face=False))
    ident.last_seen = s.ts
    sess = FakeSession([ident])
    assert maint._cleanup_provisional(sess, ident, cfg, now) is True
    assert ident in sess.deleted


def test_provisional_kept_when_within_grace():
    cfg = maint.MaintenanceConfig(provisional_grace_seconds=600.0)
    now = datetime(2026, 6, 6, 12, 0, 0)
    ident = Identity(created_at=now - timedelta(seconds=60))
    s = ident.add_sighting(Sighting(1, now - timedelta(seconds=60)))
    ident.last_seen = s.ts
    sess = FakeSession([ident])
    assert maint._cleanup_provisional(sess, ident, cfg, now) is False


def test_provisional_kept_with_face_or_multi_sighting():
    cfg = maint.MaintenanceConfig(provisional_grace_seconds=1.0)
    now = datetime(2026, 6, 6, 12, 0, 0)
    old = now - timedelta(hours=1)
    # Has a face -> real.
    withface = Identity(created_at=old)
    withface.last_seen = old
    withface.add_face(FaceExemplar(_rand_unit(40)))
    withface.add_sighting(Sighting(1, old, has_face=True))
    # Two sightings -> not provisional.
    multi = Identity(created_at=old)
    multi.last_seen = old
    multi.add_sighting(Sighting(1, old))
    multi.add_sighting(Sighting(2, old))
    # Named -> frozen.
    named = Identity(is_named=True, created_at=old)
    named.last_seen = old
    named.add_sighting(Sighting(1, old))

    sess = FakeSession([withface, multi, named])
    assert maint._cleanup_provisional(sess, withface, cfg, now) is False
    assert maint._cleanup_provisional(sess, multi, cfg, now) is False
    assert maint._cleanup_provisional(sess, named, cfg, now) is False


# ---------------------------------------------------------------------------
# Conservative face-only auto-merge
# ---------------------------------------------------------------------------


def test_auto_merge_close_face_centroids():
    cfg = maint.MaintenanceConfig(face_merge_threshold=0.6)
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(50)
    a = Identity()
    a.face_centroid = _blob(base)
    a.add_sighting(Sighting(1, now - timedelta(hours=2)))
    b = Identity()
    b.face_centroid = _blob(_near(base, jitter=0.02, seed=51))  # very close
    b.add_sighting(Sighting(2, now))  # different time -> no conflict

    sess = FakeSession([a, b])
    stats = maint.MaintenanceStats()
    merged = maint._auto_merge_pass(sess, [a, b], cfg, now, stats)
    # Lower id is the canonical survivor; the other is merged away.
    survivor, source = (a, b) if a.id < b.id else (b, a)
    assert source.id in merged
    assert source in sess.deleted
    assert stats.merges == 1
    # Source sighting reassigned to survivor.
    assert all(s.identity_id == survivor.id for s in survivor.sightings)


def test_auto_merge_vetoed_by_temporal_conflict():
    cfg = maint.MaintenanceConfig(face_merge_threshold=0.6)
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(60)
    a = Identity()
    a.face_centroid = _blob(base)
    a.add_sighting(Sighting(1, now))
    b = Identity()
    b.face_centroid = _blob(_near(base, jitter=0.02, seed=61))
    # Same instant, DIFFERENT camera -> cannot be the same person.
    b.add_sighting(Sighting(2, now + timedelta(seconds=1)))

    sess = FakeSession([a, b])
    stats = maint.MaintenanceStats()
    merged = maint._auto_merge_pass(sess, [a, b], cfg, now, stats)
    assert merged == set()
    assert stats.merges == 0


def test_auto_merge_skips_named():
    cfg = maint.MaintenanceConfig(face_merge_threshold=0.6)
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(70)
    a = Identity(is_named=True)
    a.face_centroid = _blob(base)
    b = Identity(is_named=True)
    b.face_centroid = _blob(_near(base, jitter=0.01, seed=71))
    sess = FakeSession([a, b])
    stats = maint.MaintenanceStats()
    merged = maint._auto_merge_pass(sess, [a, b], cfg, now, stats)
    assert merged == set() and stats.merges == 0


def test_auto_merge_not_for_distant_faces():
    cfg = maint.MaintenanceConfig(face_merge_threshold=0.6)
    now = datetime(2026, 6, 6, 12, 0, 0)
    a = Identity()
    a.face_centroid = _blob(_rand_unit(80))
    b = Identity()
    b.face_centroid = _blob(_rand_unit(999))  # near-orthogonal
    sess = FakeSession([a, b])
    stats = maint.MaintenanceStats()
    assert maint._auto_merge_pass(sess, [a, b], cfg, now, stats) == set()


# ---------------------------------------------------------------------------
# run_once orchestration
# ---------------------------------------------------------------------------


def test_run_once_full_pass():
    cfg = maint.MaintenanceConfig(
        app_decay_tau_seconds=3600.0,
        max_app_exemplars=2,
        max_face_exemplars=2,
        provisional_grace_seconds=600.0,
        face_merge_threshold=0.6,
        app_max_age_seconds=0,
    )
    now = datetime(2026, 6, 6, 12, 0, 0)

    # Identity 1: healthy, over-capped exemplars + needs centroid/thumb.
    ident = Identity()
    base = _rand_unit(100)
    ident.add_face(FaceExemplar(base, det_score=0.9))
    ident.add_face(FaceExemplar(_near(base, 0.05, 101), det_score=0.8))
    ident.add_face(FaceExemplar(_rand_unit(102), det_score=0.2))  # evicted
    ident.add_app(AppearanceExemplar(base, ts=now - timedelta(minutes=1), quality=0.9))
    ident.add_app(AppearanceExemplar(base, ts=now - timedelta(minutes=2), quality=0.8))
    ident.add_app(AppearanceExemplar(base, ts=now - timedelta(minutes=3), quality=0.1))  # evicted
    s = ident.add_sighting(Sighting(1, now, has_face=True, face_score=0.7))
    ident.last_seen = s.ts

    # Identity 2: provisional noise (stale, no face, single sighting) -> deleted.
    noise = Identity(created_at=now - timedelta(hours=1))
    ns = noise.add_sighting(Sighting(1, now - timedelta(hours=1), has_face=False))
    noise.last_seen = ns.ts

    sess = FakeSession([ident, noise])
    stats = maint.run_once(sess, cfg=cfg, now=now)

    assert stats.identities_scanned == 2
    assert stats.provisional_deleted == 1
    assert noise in sess.deleted
    assert stats.face_exemplars_pruned == 1
    assert stats.app_exemplars_pruned == 1
    assert stats.centroids_recomputed >= 1
    # Centroids + counters refreshed on the survivor.
    assert ident.face_centroid is not None
    assert ident.appearance_centroid is not None
    assert ident.num_sightings == 1
    assert ident.rep_sighting_id == s.id
    assert ident.first_seen == s.ts and ident.last_seen == s.ts


def test_run_once_empty_is_clean():
    sess = FakeSession([])
    stats = maint.run_once(sess, cfg=maint.MaintenanceConfig(), now=datetime(2026, 6, 6))
    assert stats.identities_scanned == 0
    assert stats.errors == 0


def test_config_from_settings_defaults_and_overrides():
    # Object missing fields -> defaults.
    empty = types.SimpleNamespace()
    cfg = maint.MaintenanceConfig.from_settings(empty)
    assert cfg.max_face_exemplars == 8
    assert cfg.max_app_exemplars == 16
    assert cfg.app_decay_tau_seconds == 12 * 3600.0
    assert cfg.enabled is True

    overridden = types.SimpleNamespace(
        max_face_exemplars=4,
        max_app_exemplars=10,
        app_decay_tau_seconds=999.0,
        face_merge_threshold=0.7,
        provisional_grace_seconds=120.0,
        reid_enabled=False,
    )
    cfg2 = maint.MaintenanceConfig.from_settings(overridden)
    assert cfg2.max_face_exemplars == 4
    assert cfg2.max_app_exemplars == 10
    assert cfg2.app_decay_tau_seconds == 999.0
    assert cfg2.face_merge_threshold == 0.7
    assert cfg2.provisional_grace_seconds == 120.0
    assert cfg2.enabled is False

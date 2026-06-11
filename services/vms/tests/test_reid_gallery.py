"""IdentityGallery + decay tests with synthetic embeddings (no GPU, no ONNX).

Exercises the real :class:`app.reid.gallery.IdentityGallery` contract:

  * face FAISS search collapsed to best-hit-per-identity (sorted, no threshold)
  * appearance candidate scoring with exponential time-decay + the temporal
    window gate
  * (de)serialization round-trip of the 512-d BLOB
  * decay math in :mod:`app.reid.decay`

A faiss stub (same surface as test_faces.py) is registered when faiss-cpu is
absent so the gallery logic still runs on CI.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# faiss stub (mirrors tests/test_faces.py) when faiss-cpu is unavailable.
# ---------------------------------------------------------------------------
if importlib.util.find_spec("faiss") is None:
    import types

    faiss = types.ModuleType("faiss")

    class _IndexFlatIP:
        def __init__(self, dim: int):
            self.d = dim
            self._vecs = np.zeros((0, dim), dtype=np.float32)

        @property
        def ntotal(self) -> int:
            return int(self._vecs.shape[0])

        def add(self, vecs):
            self._vecs = np.vstack([self._vecs, np.asarray(vecs, dtype=np.float32)])

        def search(self, q, k):
            q = np.asarray(q, dtype=np.float32)
            n = q.shape[0]
            if self._vecs.shape[0] == 0:
                return (
                    np.full((n, k), -1.0, dtype=np.float32),
                    np.full((n, k), -1, dtype=np.int64),
                )
            sims = q @ self._vecs.T
            k = min(k, sims.shape[1])
            idx = np.argsort(-sims, axis=1)[:, :k]
            dist = np.take_along_axis(sims, idx, axis=1)
            return dist.astype(np.float32), idx.astype(np.int64)

        def reset(self):
            self._vecs = np.zeros((0, self.d), dtype=np.float32)

    faiss.IndexFlatIP = _IndexFlatIP  # type: ignore[attr-defined]
    sys.modules["faiss"] = faiss


VMS_ROOT = Path(__file__).resolve().parents[1]
if str(VMS_ROOT) not in sys.path:
    sys.path.insert(0, str(VMS_ROOT))


gallery_mod = pytest.importorskip("app.reid.gallery")
decay_mod = pytest.importorskip("app.reid.decay")

IdentityGallery = gallery_mod.IdentityGallery
serialize_vector = gallery_mod.serialize_vector
deserialize_vector = gallery_mod.deserialize_vector
DIM = getattr(gallery_mod, "EMBEDDING_DIM", 512)


# ---------------------------------------------------------------------------
# synthetic-embedding helpers
# ---------------------------------------------------------------------------
def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    return (vec / n).astype(np.float32) if n else vec


def _rand_unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(DIM).astype(np.float32))


def _near(base: np.ndarray, jitter: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(base + rng.standard_normal(DIM).astype(np.float32) * jitter)


# ---------------------------------------------------------------------------
# (de)serialization
# ---------------------------------------------------------------------------
def test_serialize_roundtrip():
    v = _rand_unit(1)
    blob = serialize_vector(v)
    assert len(blob) == DIM * 4
    assert np.allclose(deserialize_vector(blob), v, atol=1e-6)


# ---------------------------------------------------------------------------
# face side
# ---------------------------------------------------------------------------
def test_empty_gallery():
    g = IdentityGallery(dim=DIM)
    assert g.num_identities == 0
    assert g.search_faces(_rand_unit(2)) == []
    assert g.best_face_per_identity(_rand_unit(2)) == []


def test_face_search_resolves_identity():
    g = IdentityGallery(dim=DIM)
    base = _rand_unit(10)
    g.add_face_exemplar(exemplar_id=1, identity_id=7, vector=base)
    g.add_face_exemplar(exemplar_id=2, identity_id=8, vector=_rand_unit(11))

    hit = g.best_face_per_identity(_near(base, 0.02, seed=12))[0]
    assert hit.identity_id == 7
    assert hit.score > 0.9


def test_best_face_per_identity_collapses_exemplars():
    g = IdentityGallery(dim=DIM)
    base = _rand_unit(20)
    # Two exemplars for the SAME identity -> collapsed to one (best) hit.
    g.add_face_exemplar(1, identity_id=5, vector=base)
    g.add_face_exemplar(2, identity_id=5, vector=_near(base, 0.05, 21))
    g.add_face_exemplar(3, identity_id=6, vector=_rand_unit(22))

    hits = g.best_face_per_identity(_near(base, 0.02, 23))
    ids = [h.identity_id for h in hits]
    assert ids.count(5) == 1  # collapsed
    assert hits[0].identity_id == 5


def test_face_centroid_for():
    g = IdentityGallery(dim=DIM)
    base = _rand_unit(30)
    g.add_face_exemplar(1, identity_id=9, vector=base)
    g.add_face_exemplar(2, identity_id=9, vector=_near(base, 0.05, 31))
    cen = g.face_centroid_for(9)
    assert cen is not None
    assert float(cen @ base) > 0.9
    assert g.face_centroid_for(404) is None


# ---------------------------------------------------------------------------
# appearance side + time decay
# ---------------------------------------------------------------------------
def test_appearance_recent_match_scores_high():
    now = datetime(2026, 6, 6, 12, 0, 0)
    g = IdentityGallery(dim=DIM, app_window_seconds=600, app_decay_tau_seconds=43_200)
    outfit = _rand_unit(40)
    g.add_appearance_exemplar(identity_id=3, vector=outfit, ts=now - timedelta(seconds=30), camera_id=1)

    cands = g.appearance_candidates(_near(outfit, 0.02, 41), now=now)
    assert cands
    assert cands[0].identity_id == 3
    assert cands[0].score > 0.85  # recent -> minimal decay


def test_appearance_outside_window_filtered():
    now = datetime(2026, 6, 6, 12, 0, 0)
    g = IdentityGallery(dim=DIM, app_window_seconds=600, app_decay_tau_seconds=43_200)
    outfit = _rand_unit(50)
    # Last seen 20 minutes ago -> outside the 10-minute window.
    g.add_appearance_exemplar(3, outfit, ts=now - timedelta(minutes=20), camera_id=1)
    assert g.appearance_candidates(outfit, now=now) == []


def test_appearance_decay_lowers_older_score():
    now = datetime(2026, 6, 6, 12, 0, 0)
    # Wide window so the old exemplar is still a candidate, but decayed.
    g = IdentityGallery(dim=DIM, app_window_seconds=86_400, app_decay_tau_seconds=43_200)
    outfit = _rand_unit(60)
    g.add_appearance_exemplar(1, outfit, ts=now - timedelta(hours=12), camera_id=1)  # ~1 tau
    cands = g.appearance_candidates(_near(outfit, 0.01, 61), now=now)
    assert cands
    # exp(-1) ~= 0.368 of the raw ~1.0 cosine.
    assert 0.3 < cands[0].score < 0.45
    assert cands[0].raw_score > 0.95


def test_identity_cameras_tracked():
    now = datetime(2026, 6, 6, 12, 0, 0)
    g = IdentityGallery(dim=DIM)
    g.add_appearance_exemplar(1, _rand_unit(70), ts=now, camera_id=2)
    g.add_appearance_exemplar(1, _rand_unit(71), ts=now, camera_id=5)
    assert g.identity_cameras(1) == {2, 5}


# ---------------------------------------------------------------------------
# decay module
# ---------------------------------------------------------------------------
def test_decay_weight_monotonic():
    w0 = decay_mod.decay_weight(0, 100)
    w1 = decay_mod.decay_weight(100, 100)
    w2 = decay_mod.decay_weight(1000, 100)
    assert w0 == pytest.approx(1.0)
    assert w1 == pytest.approx(np.exp(-1), abs=1e-6)
    assert w2 < w1 < w0


def test_decay_weight_future_clamped():
    assert decay_mod.decay_weight(-50, 100) == pytest.approx(1.0)


def test_effective_score_takes_max_weighted():
    now = datetime(2026, 6, 6, 12, 0, 0)
    cosines = [0.9, 0.95]
    tss = [now - timedelta(hours=24), now - timedelta(seconds=10)]
    # Recent 0.95 dominates the decayed older 0.9.
    s = decay_mod.effective_score(cosines, tss, now, tau_seconds=43_200)
    assert s == pytest.approx(0.95, abs=0.01)


def test_decayed_centroid_normalized():
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(80)
    vecs = [base, _near(base, 0.02, 81)]
    tss = [now, now - timedelta(seconds=60)]
    cen = decay_mod.decayed_centroid(vecs, tss, now, tau_seconds=43_200)
    assert cen is not None
    assert float(np.linalg.norm(cen)) == pytest.approx(1.0, abs=1e-5)
    assert float(cen @ base) > 0.95


def test_decayed_centroid_all_stale_returns_none():
    now = datetime(2026, 6, 6, 12, 0, 0)
    base = _rand_unit(90)
    # ts so old that decay weight < NEGLIGIBLE_WEIGHT.
    tss = [now - timedelta(days=30)]
    assert decay_mod.decayed_centroid([base], tss, now, tau_seconds=43_200) is None

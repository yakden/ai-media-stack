"""FaceIndex tests: add / search / match / threshold with synthetic embeddings.

These tests use *synthetic* L2-normalized embeddings (no ArcFace, no GPU, no
RTSP). They target the real :class:`app.faces.index.FaceIndex` contract:

  * ``FaceIndex(dim=512, match_threshold=0.4)``
  * ``add(embedding_id, person_id, vector) -> faiss_row``
  * ``search(vector, k) -> list[MatchResult]`` (sorted by cosine, no threshold)
  * ``match(vector, threshold=None) -> MatchResult | None`` (threshold applied)
  * ``size`` property, ``clear()``
  * ``serialize_vector`` / ``deserialize_vector`` round-trip the 512-d BLOB.

FAISS (``faiss-cpu``) is required by the index; if it is not installed a small
in-memory ``IndexFlatIP`` stub with the same ``add`` / ``search`` / ``ntotal``
surface is registered so the FaceIndex logic can still be exercised on CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Provide a faiss stub if faiss-cpu is not installed.
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


# Make ``vms/`` importable so ``app.faces.index`` resolves.
VMS_ROOT = Path(__file__).resolve().parents[1]
if str(VMS_ROOT) not in sys.path:
    sys.path.insert(0, str(VMS_ROOT))


index_mod = pytest.importorskip(
    "app.faces.index", reason="app.faces.index not available"
)
FaceIndex = index_mod.FaceIndex
serialize_vector = index_mod.serialize_vector
deserialize_vector = index_mod.deserialize_vector
EMBEDDING_DIM = getattr(index_mod, "EMBEDDING_DIM", 512)


# ---------------------------------------------------------------------------
# Synthetic-embedding helpers
# ---------------------------------------------------------------------------

DIM = EMBEDDING_DIM


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    return (vec / n).astype(np.float32) if n else vec


def _rand_unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(DIM).astype(np.float32))


def _near(base: np.ndarray, jitter: float, seed: int) -> np.ndarray:
    """A near-duplicate of ``base`` with controllable cosine closeness."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(DIM).astype(np.float32) * jitter
    return _unit(base + noise)


# ---------------------------------------------------------------------------
# (De)serialization round-trip — DB BLOB contract.
# ---------------------------------------------------------------------------


def test_serialize_roundtrip():
    v = _rand_unit(1)
    blob = serialize_vector(v)
    assert isinstance(blob, (bytes, bytearray))
    assert len(blob) == DIM * 4  # float32, little-endian
    back = deserialize_vector(blob)
    assert back.shape == (DIM,)
    assert np.allclose(back, v, atol=1e-6)


def test_serialize_rejects_wrong_dim():
    with pytest.raises(ValueError):
        serialize_vector(np.zeros(DIM + 1, dtype=np.float32))


# ---------------------------------------------------------------------------
# Synthetic embedding geometry sanity (independent of the index).
# ---------------------------------------------------------------------------


def test_normalized_self_similarity_is_one():
    v = _rand_unit(2)
    assert np.isclose(float(v @ v), 1.0, atol=1e-5)


def test_near_duplicate_high_similarity():
    base = _rand_unit(3)
    near = _near(base, jitter=0.01, seed=4)  # very small jitter -> ~1.0 cosine
    far = _rand_unit(99)
    assert float(base @ near) > 0.95
    # Two independent random 512-d unit vectors are near-orthogonal.
    assert abs(float(base @ far)) < 0.2


# ---------------------------------------------------------------------------
# FaceIndex: add / size / search / match / threshold / clear.
# ---------------------------------------------------------------------------


def test_index_starts_empty():
    idx = FaceIndex(dim=DIM)
    assert idx.size == 0
    assert idx.search(_rand_unit(5)) == []
    assert idx.match(_rand_unit(5)) is None


def test_add_returns_row_and_grows_size():
    idx = FaceIndex(dim=DIM)
    pos0 = idx.add(embedding_id=1, person_id=100, vector=_rand_unit(10))
    pos1 = idx.add(embedding_id=2, person_id=200, vector=_rand_unit(11))
    assert pos0 == 0
    assert pos1 == 1
    assert idx.size == 2


def test_add_normalizes_unnormalized_input():
    idx = FaceIndex(dim=DIM)
    base = _rand_unit(12)
    idx.add(embedding_id=1, person_id=100, vector=base * 7.5)  # arbitrary scale
    hit = idx.match(base, threshold=0.9)
    assert hit is not None
    assert hit.score == pytest.approx(1.0, abs=1e-4)


def test_search_returns_sorted_matches():
    idx = FaceIndex(dim=DIM, match_threshold=0.3)
    targets = {1: _rand_unit(20), 2: _rand_unit(21), 3: _rand_unit(22)}
    for eid, vec in targets.items():
        idx.add(embedding_id=eid, person_id=eid * 10, vector=vec)

    probe = _near(targets[2], jitter=0.02, seed=23)
    hits = idx.search(probe, k=3)
    assert len(hits) == 3
    # Sorted descending by cosine.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    # Nearest neighbour is person 2 / embedding 2.
    assert hits[0].embedding_id == 2
    assert hits[0].person_id == 20
    assert hits[0].score > 0.9


def test_match_above_threshold_returns_hit():
    idx = FaceIndex(dim=DIM, match_threshold=0.4)
    base = _rand_unit(30)
    idx.add(embedding_id=7, person_id=70, vector=base)

    probe = _near(base, jitter=0.02, seed=31)
    hit = idx.match(probe)
    assert hit is not None
    assert hit.embedding_id == 7
    assert hit.person_id == 70
    assert hit.score >= 0.4
    assert hit.matched is True


def test_match_below_threshold_returns_none():
    idx = FaceIndex(dim=DIM, match_threshold=0.9)
    enrolled = _rand_unit(40)
    idx.add(embedding_id=1, person_id=100, vector=enrolled)

    stranger = _rand_unit(987)  # ~orthogonal -> cosine well below 0.9
    # search still returns the (only) neighbour...
    assert len(idx.search(stranger, k=1)) == 1
    # ...but match applies the threshold and rejects it.
    assert idx.match(stranger) is None


def test_match_threshold_override():
    idx = FaceIndex(dim=DIM, match_threshold=0.99)
    base = _rand_unit(50)
    idx.add(embedding_id=1, person_id=100, vector=base)
    probe = _near(base, jitter=0.1, seed=51)  # moderate similarity

    score = idx.search(probe, k=1)[0].score
    # Default (very strict) threshold may reject; a lenient override accepts.
    assert idx.match(probe, threshold=min(0.3, score)) is not None


def test_clear_empties_index():
    idx = FaceIndex(dim=DIM)
    idx.add(embedding_id=1, person_id=100, vector=_rand_unit(60))
    idx.add(embedding_id=2, person_id=200, vector=_rand_unit(61))
    assert idx.size == 2
    idx.clear()
    assert idx.size == 0
    assert idx.search(_rand_unit(60)) == []


def test_search_k_capped_at_size():
    idx = FaceIndex(dim=DIM)
    idx.add(embedding_id=1, person_id=100, vector=_rand_unit(70))
    # Asking for more neighbours than exist must not error or pad with -1 rows.
    hits = idx.search(_rand_unit(70), k=10)
    assert len(hits) == 1
    assert hits[0].person_id == 100

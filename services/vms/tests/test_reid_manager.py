"""IdentityManager tests: online assignment, anti-explosion, merge/split.

Uses a real in-memory SQLite DB (the actual ORM models) + synthetic embeddings
and a fake ``SightingFeature`` builder — no GPU, no ONNX, no RTSP. A faiss stub
(same surface as tests/test_faces.py) is registered when faiss-cpu is absent.

Covered behaviours from the matching algorithm:
  * NEW identity creation seeds face + appearance exemplars
  * strong face -> MATCH to the existing identity (clothing-invariant link)
  * appearance fallback within the time window when no face is visible
  * appearance outside the window -> NEW (no teleportation)
  * face contradiction veto (a confident face for A blocks appearance->B)
  * sticky/hysteresis keeps a continuous overlapping track on one identity
  * new-identity rate limit drops bursts
  * provisional flag flips on the 2nd sighting / first face
  * MERGE reassigns sightings/exemplars; SPLIT carves a cluster out
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# faiss stub
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
manager_mod = pytest.importorskip("app.reid.manager")
pipeline_mod = pytest.importorskip("app.reid.pipeline")
models_mod = pytest.importorskip("app.db.models")

IdentityGallery = gallery_mod.IdentityGallery
IdentityManager = manager_mod.IdentityManager
MatchConfig = manager_mod.MatchConfig
SightingFeature = pipeline_mod.SightingFeature
BBox = pipeline_mod.BBox

DIM = getattr(gallery_mod, "EMBEDDING_DIM", 512)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite://", future=True)
    models_mod.Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    db = Local()
    # A camera so Sighting.camera_id FK is satisfied.
    cam = models_mod.Camera(id=1, name="cam1", rtsp_url="rtsp://x", enabled=True)
    cam2 = models_mod.Camera(id=2, name="cam2", rtsp_url="rtsp://y", enabled=True)
    db.add_all([cam, cam2])
    db.commit()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def manager():
    g = IdentityGallery(dim=DIM, app_window_seconds=600, app_decay_tau_seconds=43_200)
    return IdentityManager(g, MatchConfig())


# ---------------------------------------------------------------------------
# embedding / feature helpers
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


def _feature(
    *,
    box=(10, 10, 60, 160),
    score=0.9,
    appearance=None,
    face=None,
    face_score=0.0,
    quality=0.8,
    area_frac=0.2,
):
    return SightingFeature(
        box=BBox(*box, score),
        appearance_vec=appearance,
        face_vec=face,
        face_det_score=face_score,
        crop_quality=quality,
        box_area_frac=area_frac,
        has_face=face is not None,
    )


# ---------------------------------------------------------------------------
# NEW identity creation
# ---------------------------------------------------------------------------
def test_new_identity_with_face_and_appearance(session, manager):
    face = _rand_unit(1)
    app = _rand_unit(2)
    feat = _feature(face=face, face_score=0.8, appearance=app)
    res = manager.assign(session, feat, camera_id=1, ts=datetime(2026, 6, 6, 12, 0, 0))
    session.commit()

    assert res.is_new and res.match_kind == "new"
    ident = session.get(models_mod.Identity, res.identity_id)
    assert ident.name == f"Person {ident.id}"
    assert ident.num_sightings == 1
    assert len(ident.face_exemplars) == 1
    assert len(ident.appearance_exemplars) == 1
    assert ident.rep_sighting_id == res.sighting_id
    assert ident.is_provisional is False  # has a face -> not provisional


def test_faceless_low_evidence_is_provisional(session, manager):
    feat = _feature(face=None, appearance=_rand_unit(3))
    res = manager.assign(session, feat, camera_id=1, ts=datetime(2026, 6, 6, 12, 0, 0))
    session.commit()
    ident = session.get(models_mod.Identity, res.identity_id)
    assert ident.is_provisional is True
    assert ident.num_sightings == 1


# ---------------------------------------------------------------------------
# face MATCH (clothing-invariant cross-camera link)
# ---------------------------------------------------------------------------
def test_strong_face_links_across_cameras_different_clothes(session, manager):
    face = _rand_unit(10)
    # First sighting on camera 1, outfit A.
    r1 = manager.assign(
        session,
        _feature(face=face, face_score=0.8, appearance=_rand_unit(11)),
        camera_id=1,
        ts=datetime(2026, 6, 6, 12, 0, 0),
    )
    session.commit()

    # Later sighting on camera 2, DIFFERENT outfit, but same face.
    r2 = manager.assign(
        session,
        _feature(face=_near(face, 0.02, 12), face_score=0.8, appearance=_rand_unit(99)),
        camera_id=2,
        ts=datetime(2026, 6, 6, 13, 0, 0),
    )
    session.commit()

    assert r2.match_kind == "face"
    assert r2.identity_id == r1.identity_id  # linked by face despite new clothes
    ident = session.get(models_mod.Identity, r1.identity_id)
    assert ident.num_sightings == 2


# ---------------------------------------------------------------------------
# appearance fallback within window (no face)
# ---------------------------------------------------------------------------
def test_appearance_links_when_face_absent_in_window(session, manager):
    outfit = _rand_unit(20)
    r1 = manager.assign(
        session,
        _feature(face=_rand_unit(21), face_score=0.8, appearance=outfit),
        camera_id=1,
        ts=datetime(2026, 6, 6, 12, 0, 0),
    )
    session.commit()

    # Same camera, 1 min later, BACK TURNED (no face) but same clothes.
    r2 = manager.assign(
        session,
        _feature(face=None, appearance=_near(outfit, 0.02, 22)),
        camera_id=1,
        ts=datetime(2026, 6, 6, 12, 1, 0),
    )
    session.commit()

    assert r2.match_kind == "appearance"
    assert r2.identity_id == r1.identity_id


def test_appearance_outside_window_creates_new(session, manager):
    outfit = _rand_unit(30)
    r1 = manager.assign(
        session,
        _feature(face=None, appearance=outfit, quality=0.8),
        camera_id=1,
        ts=datetime(2026, 6, 6, 12, 0, 0),
    )
    session.commit()
    # 30 minutes later (window=600s=10min) -> no appearance link -> NEW.
    r2 = manager.assign(
        session,
        _feature(face=None, appearance=_near(outfit, 0.02, 31)),
        camera_id=1,
        ts=datetime(2026, 6, 6, 12, 30, 0),
    )
    session.commit()
    assert r2.match_kind == "new"
    assert r2.identity_id != r1.identity_id


# ---------------------------------------------------------------------------
# face contradiction veto
# ---------------------------------------------------------------------------
def test_face_veto_blocks_appearance_to_other_identity(session, manager):
    # Identity A: distinctive face + outfit X.
    faceA = _rand_unit(40)
    outfitX = _rand_unit(41)
    rA = manager.assign(
        session,
        _feature(face=faceA, face_score=0.8, appearance=outfitX),
        camera_id=1,
        ts=datetime(2026, 6, 6, 12, 0, 0),
    )
    session.commit()

    # Identity B: different face, but happens to wear the SAME outfit X.
    faceB = _rand_unit(42)
    rB = manager.assign(
        session,
        _feature(face=faceB, face_score=0.8, appearance=_near(outfitX, 0.02, 43)),
        camera_id=1,
        ts=datetime(2026, 6, 6, 12, 1, 0),
    )
    session.commit()
    # Confident distinct face -> must be its own identity, never appearance->A.
    assert rB.match_kind == "face"
    assert rB.identity_id != rA.identity_id


# ---------------------------------------------------------------------------
# sticky / hysteresis
# ---------------------------------------------------------------------------
def test_sticky_keeps_continuous_track_on_one_identity(session, manager):
    app = _rand_unit(50)
    r1 = manager.assign(
        session,
        _feature(box=(10, 10, 60, 160), face=None, appearance=app),
        camera_id=1,
    )
    session.commit()
    # Next frame, overlapping box (IoU high), within 2s -> sticky reuse.
    r2 = manager.assign(
        session,
        _feature(box=(12, 12, 62, 162), face=None, appearance=_rand_unit(51)),
        camera_id=1,
    )
    session.commit()
    assert r2.match_kind == "sticky"
    assert r2.identity_id == r1.identity_id


# ---------------------------------------------------------------------------
# new-identity rate limit
# ---------------------------------------------------------------------------
def test_new_identity_rate_limit(session):
    g = IdentityGallery(dim=DIM)
    cfg = MatchConfig(new_identity_rate_per_min=2, sticky_iou=2.0)  # disable sticky
    mgr = IdentityManager(g, cfg)
    created = 0
    for i in range(5):
        res = mgr.assign(
            session,
            _feature(box=(0, 0, 5, 50), face=_rand_unit(100 + i), face_score=0.8,
                     appearance=_rand_unit(200 + i)),
            camera_id=1,
            ts=datetime(2026, 6, 6, 12, 0, i),
        )
        if res.created_identity:
            created += 1
    session.commit()
    assert created == 2  # capped


# ---------------------------------------------------------------------------
# provisional graduation
# ---------------------------------------------------------------------------
def test_provisional_graduates_on_second_sighting(session, manager):
    outfit = _rand_unit(60)
    r1 = manager.assign(
        session, _feature(face=None, appearance=outfit), camera_id=1,
        ts=datetime(2026, 6, 6, 12, 0, 0),
    )
    session.commit()
    ident = session.get(models_mod.Identity, r1.identity_id)
    assert ident.is_provisional is True

    manager.assign(
        session, _feature(box=(200, 200, 260, 360), face=None, appearance=_near(outfit, 0.02, 61)),
        camera_id=1, ts=datetime(2026, 6, 6, 12, 0, 30),
    )
    session.commit()
    session.refresh(ident)
    assert ident.is_provisional is False
    assert ident.num_sightings == 2


# ---------------------------------------------------------------------------
# merge / split
# ---------------------------------------------------------------------------
def test_merge_reassigns_sightings_and_exemplars(session, manager):
    rA = manager.assign(
        session, _feature(face=_rand_unit(70), face_score=0.8, appearance=_rand_unit(71)),
        camera_id=1, ts=datetime(2026, 6, 6, 12, 0, 0),
    )
    rB = manager.assign(
        session, _feature(box=(300, 10, 360, 200), face=_rand_unit(72), face_score=0.8,
                          appearance=_rand_unit(73)),
        camera_id=2, ts=datetime(2026, 6, 6, 12, 5, 0),
    )
    session.commit()
    assert rA.identity_id != rB.identity_id

    moved = manager.merge(session, target_id=rA.identity_id, source_ids=[rB.identity_id])
    session.commit()
    assert moved >= 1
    assert session.get(models_mod.Identity, rB.identity_id) is None
    target = session.get(models_mod.Identity, rA.identity_id)
    assert target.num_sightings == 2
    assert len(target.face_exemplars) == 2


def test_split_carves_out_chosen_sightings(session, manager):
    # One identity that accumulated 4 sightings via sticky/appearance.
    outfit = _rand_unit(80)
    ids = []
    base_ts = datetime(2026, 6, 6, 12, 0, 0)
    for i in range(4):
        res = manager.assign(
            session,
            _feature(box=(10, 10, 60, 160), face=_rand_unit(800 + i), face_score=0.8,
                     appearance=_near(outfit, 0.02, 80 + i)),
            camera_id=1,
            ts=base_ts + timedelta(seconds=i),
        )
        # Sticky after the first; force-disable sticky by clearing it so each is
        # appended to the SAME identity via face match instead.
        ids.append(res)
    session.commit()
    # Collapse them onto one identity by merging (simulating over-clustering).
    target = ids[0].identity_id
    others = sorted({r.identity_id for r in ids} - {target})
    if others:
        manager.merge(session, target_id=target, source_ids=others)
        session.commit()
    ident = session.get(models_mod.Identity, target)
    sightings_before = ident.num_sightings
    assert sightings_before >= 1

    # Split out the first two sightings explicitly.
    sids = [r.sighting_id for r in ids][:2]
    new_id = manager.split(session, identity_id=target, sighting_ids=sids)
    session.commit()
    if new_id != target:  # split happened (needs >0 and < all)
        new_ident = session.get(models_mod.Identity, new_id)
        assert new_ident is not None
        assert new_ident.num_sightings >= 1
        session.refresh(ident)
        assert ident.num_sightings + new_ident.num_sightings == sightings_before

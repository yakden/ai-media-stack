"""Tests for the auto-discovered Identities API (app/api/identities.py).

The Identity / Sighting / FaceExemplar / AppearanceExemplar ORM models are
owned by the *integration* component (they get added to ``app.db.models``).
This suite is self-contained: if those models have not landed yet, it defines
equivalent ORM classes on the SAME declarative ``Base`` and attaches them to
``app.db.models`` so the router's lazy ``_models()`` lookup resolves and the
tables materialise — mirroring the cross-component, stays-green philosophy of
``tests/test_api.py``.

Native deps (cv2 / onnxruntime) and the WorkerManager are stubbed exactly as in
``tests/test_api.py`` so the suite runs on a plain CPU box with no models.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub native deps before importing the app package (only if truly missing).
# ---------------------------------------------------------------------------
def _missing(name: str) -> bool:
    """True if ``name`` is neither already imported/stubbed nor installed.

    Guard against a sibling test having already inserted a bare stub module
    (whose ``__spec__`` is None, which would make ``find_spec`` raise).
    """
    if name in sys.modules:
        return False
    try:
        return importlib.util.find_spec(name) is None
    except (ValueError, ModuleNotFoundError):
        return False


def _stub_native_deps() -> None:
    if _missing("cv2"):
        cv2 = types.ModuleType("cv2")
        cv2.IMREAD_COLOR = 1  # type: ignore[attr-defined]
        cv2.imdecode = lambda buf, flag: np.zeros((8, 8, 3), dtype=np.uint8)  # type: ignore[attr-defined]
        cv2.imencode = lambda ext, img, *a, **k: (True, np.frombuffer(b"\xff\xd8\xff\xd9", dtype=np.uint8))  # type: ignore[attr-defined]
        cv2.imwrite = lambda *a, **k: True  # type: ignore[attr-defined]
        sys.modules["cv2"] = cv2
    if _missing("onnxruntime"):
        ort = types.ModuleType("onnxruntime")
        ort.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
        sys.modules["onnxruntime"] = ort


_stub_native_deps()


VMS_ROOT = Path(__file__).resolve().parents[1]
if str(VMS_ROOT) not in sys.path:
    sys.path.insert(0, str(VMS_ROOT))


class _FakeWorkerManager:
    def __init__(self, settings=None, *a, **k):
        self.settings = settings

    def start_all(self, cameras=None, *a, **k):
        pass

    def shutdown(self, *a, **k):
        pass

    def all_status(self, *a, **k):
        return {}

    def get_status(self, camera_id, *a, **k):
        return None


def _ensure_reid_models(models_mod) -> None:
    """Define Identity/Sighting/(Face|Appearance)Exemplar on the shared Base if
    the integration component has not added them yet. Idempotent."""
    if hasattr(models_mod, "Identity"):
        return

    from sqlalchemy import (
        Boolean,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        LargeBinary,
        String,
        Text,
        func,
    )
    from sqlalchemy.orm import Mapped, mapped_column, relationship

    Base = models_mod.Base

    class Identity(Base):  # type: ignore[misc, valid-type]
        __tablename__ = "identities"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        name: Mapped[str] = mapped_column(String, nullable=False, default="Person")
        is_named: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
        notes: Mapped[str | None] = mapped_column(Text, nullable=True)
        rep_sighting_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
        face_centroid: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
        appearance_centroid: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
        num_sightings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
        first_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
        last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
        created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
        sightings = relationship("Sighting", cascade="all, delete-orphan", passive_deletes=True)
        face_exemplars = relationship("FaceExemplar", cascade="all, delete-orphan", passive_deletes=True)
        appearance_exemplars = relationship("AppearanceExemplar", cascade="all, delete-orphan", passive_deletes=True)

    class Sighting(Base):  # type: ignore[misc, valid-type]
        __tablename__ = "sightings"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        identity_id: Mapped[int] = mapped_column(
            ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True
        )
        camera_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
        event_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
        ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
        bbox_x1: Mapped[int] = mapped_column(Integer, default=0)
        bbox_y1: Mapped[int] = mapped_column(Integer, default=0)
        bbox_x2: Mapped[int] = mapped_column(Integer, default=0)
        bbox_y2: Mapped[int] = mapped_column(Integer, default=0)
        det_score: Mapped[float | None] = mapped_column(Float, nullable=True)
        has_face: Mapped[bool] = mapped_column(Boolean, default=False)
        face_score: Mapped[float | None] = mapped_column(Float, nullable=True)
        appearance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
        match_kind: Mapped[str | None] = mapped_column(String, nullable=True)
        thumb_path: Mapped[str | None] = mapped_column(String, nullable=True)
        created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    class FaceExemplar(Base):  # type: ignore[misc, valid-type]
        __tablename__ = "face_exemplars"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        identity_id: Mapped[int] = mapped_column(
            ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True
        )
        vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
        det_score: Mapped[float | None] = mapped_column(Float, nullable=True)
        camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
        sighting_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
        created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    class AppearanceExemplar(Base):  # type: ignore[misc, valid-type]
        __tablename__ = "appearance_exemplars"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        identity_id: Mapped[int] = mapped_column(
            ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True
        )
        vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
        quality: Mapped[float | None] = mapped_column(Float, nullable=True)
        camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
        sighting_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
        ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
        created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())

    models_mod.Identity = Identity
    models_mod.Sighting = Sighting
    models_mod.FaceExemplar = FaceExemplar
    models_mod.AppearanceExemplar = AppearanceExemplar


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("data_dir", str(data_dir))
    monkeypatch.setenv("models_dir", str(models_dir))
    monkeypatch.setenv("detector_backend", "cpu")
    monkeypatch.setenv("device", "cpu")
    monkeypatch.setenv("faces_enabled", "false")
    monkeypatch.setenv("auth_required", "false")

    config_mod = importlib.import_module("app.config")
    config_mod.get_settings.cache_clear()

    for name in list(sys.modules):
        if name.startswith("app.") and name not in {"app", "app.config"}:
            del sys.modules[name]

    db_mod = importlib.import_module("app.db.database")
    if hasattr(db_mod, "reset_engine"):
        db_mod.reset_engine()

    manager_mod = importlib.import_module("app.workers.manager")
    monkeypatch.setattr(manager_mod, "WorkerManager", _FakeWorkerManager)

    # Register reid models on the shared metadata BEFORE the app boots so the
    # router resolves them and create_all materialises the tables.
    models_mod = importlib.import_module("app.db.models")
    _ensure_reid_models(models_mod)

    main_mod = importlib.import_module("app.main")
    app = main_mod.create_app()

    # The integration component adds 'app.api.identities' to main._OPTIONAL_ROUTERS.
    # Until that lands, mount it here so the contract is actually exercised (rather
    # than skipped). Idempotent: skip if already mounted by create_app().
    if not any(getattr(r, "path", "") == "/api/identities" for r in app.routes):
        identities_mod = importlib.import_module("app.api.identities")
        app.include_router(identities_mod.router)

    engine = db_mod.get_engine()
    models_mod.Base.metadata.create_all(bind=engine)

    headers = {"X-Email": "tester@example.com", "X-User": "tester"}
    with TestClient(app, raise_server_exceptions=False) as tc:
        tc.headers.update(headers)
        yield tc

    config_mod.get_settings.cache_clear()
    if hasattr(db_mod, "reset_engine"):
        db_mod.reset_engine()


def _route_exists(client, path: str, method: str = "GET") -> bool:
    method = method.upper()
    for route in client.app.routes:
        methods = getattr(route, "methods", None) or set()
        tmpl = getattr(route, "path", "")
        if method in methods and (tmpl == path or tmpl.rstrip("/") == path.rstrip("/")):
            return True
    return False


def _require_route(client, path: str, method: str = "GET") -> None:
    if not _route_exists(client, path, method):
        pytest.skip(f"route {method} {path} not mounted (owning component absent)")


# ---------------------------------------------------------------------------
# DB seeding helpers (direct ORM, mirroring tests/test_api.py)
# ---------------------------------------------------------------------------
def _seed_camera(name="Cam1") -> int:
    db_mod = importlib.import_module("app.db.database")
    models = importlib.import_module("app.db.models")
    s = db_mod.SessionLocal()
    try:
        cam = models.Camera(name=name, rtsp_url="rtsp://x/y", enabled=True)
        s.add(cam)
        s.commit()
        s.refresh(cam)
        return cam.id
    finally:
        s.close()


def _seed_identity(
    name="Person 1",
    *,
    is_named=False,
    sightings=None,  # list of dicts: {camera_id, ts, thumb_path, has_face, ...}
):
    db_mod = importlib.import_module("app.db.database")
    models = importlib.import_module("app.db.models")
    s = db_mod.SessionLocal()
    try:
        ident = models.Identity(name=name, is_named=is_named)
        s.add(ident)
        s.flush()
        sighting_ids = []
        first = last = None
        for sd in sightings or []:
            sg = models.Sighting(
                identity_id=ident.id,
                camera_id=sd["camera_id"],
                ts=sd["ts"],
                bbox_x1=sd.get("bbox", (0, 0, 10, 20))[0],
                bbox_y1=sd.get("bbox", (0, 0, 10, 20))[1],
                bbox_x2=sd.get("bbox", (0, 0, 10, 20))[2],
                bbox_y2=sd.get("bbox", (0, 0, 10, 20))[3],
                det_score=sd.get("det_score", 0.9),
                has_face=sd.get("has_face", False),
                face_score=sd.get("face_score"),
                appearance_score=sd.get("appearance_score"),
                match_kind=sd.get("match_kind", "appearance"),
                thumb_path=sd.get("thumb_path"),
            )
            s.add(sg)
            s.flush()
            sighting_ids.append(sg.id)
            first = sg.ts if first is None or sg.ts < first else first
            last = sg.ts if last is None or sg.ts > last else last
        ident.num_sightings = len(sighting_ids)
        ident.first_seen = first
        ident.last_seen = last
        if sighting_ids:
            ident.rep_sighting_id = sighting_ids[-1]
        s.commit()
        s.refresh(ident)
        return ident.id, sighting_ids
    finally:
        s.close()


def _write_thumb(client, rel_path: str) -> str:
    """Create a fake jpg under the data dir; return the stored relative path."""
    settings = client.app.state.settings
    abs_path = Path(settings.data_dir) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(b"\xff\xd8\xff\xe0jpegdata\xff\xd9")
    return f"data/{rel_path}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_list_identities_empty(client):
    _require_route(client, "/api/identities", "GET")
    r = client.get("/api/identities")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_list_and_get_identity_with_sightings(client):
    _require_route(client, "/api/identities", "GET")
    cam_a = _seed_camera("CamA")
    cam_b = _seed_camera("CamB")
    now = datetime.now(timezone.utc)
    ident_id, sids = _seed_identity(
        name="Person 1",
        sightings=[
            {"camera_id": cam_a, "ts": now - timedelta(minutes=10), "has_face": True, "face_score": 0.6},
            {"camera_id": cam_b, "ts": now - timedelta(minutes=2), "match_kind": "appearance", "appearance_score": 0.7},
        ],
    )

    # List shows the identity with both cameras and a rep thumb (rep sighting set).
    r = client.get("/api/identities")
    assert r.status_code == 200, r.text
    item = next(it for it in r.json()["items"] if it["id"] == ident_id)
    assert item["num_sightings"] == 2
    assert sorted(item["cameras"]) == sorted([cam_a, cam_b])

    # Detail.
    r = client.get(f"/api/identities/{ident_id}")
    assert r.status_code == 200, r.text
    detail = r.json()
    assert detail["id"] == ident_id
    assert detail["num_sightings"] == 2
    assert len(detail["recent_sightings"]) == 2
    # Newest first.
    assert detail["recent_sightings"][0]["camera_name"] == "CamB"

    # Paginated sightings.
    r = client.get(f"/api/identities/{ident_id}/sightings", params={"limit": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["items"]) == 1
    sg = body["items"][0]
    assert sg["bbox"] == [0, 0, 10, 20]


def test_get_missing_identity_404(client):
    _require_route(client, "/api/identities/{identity_id}", "GET")
    r = client.get("/api/identities/999999")
    assert r.status_code == 404


def test_rename_sets_is_named(client):
    _require_route(client, "/api/identities/{identity_id}", "PUT")
    cam = _seed_camera()
    ident_id, _ = _seed_identity(
        sightings=[{"camera_id": cam, "ts": datetime.now(timezone.utc)}]
    )

    r = client.put(f"/api/identities/{ident_id}", json={"name": "Alice", "notes": "VIP"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "Alice"
    assert body["is_named"] is True
    assert body["notes"] == "VIP"

    # Explicit unfreeze.
    r = client.put(f"/api/identities/{ident_id}", json={"is_named": False})
    assert r.status_code == 200
    assert r.json()["is_named"] is False


def test_thumbnail_serving(client):
    _require_route(client, "/api/identities/{identity_id}/thumbnail", "GET")
    cam = _seed_camera()
    thumb = _write_thumb(client, "identities/1/1.jpg")
    db_mod = importlib.import_module("app.db.database")
    models = importlib.import_module("app.db.models")
    # Seed an identity whose rep sighting points at the on-disk thumb.
    s = db_mod.SessionLocal()
    try:
        ident = models.Identity(name="P", num_sightings=1)
        s.add(ident)
        s.flush()
        sg = models.Sighting(
            identity_id=ident.id, camera_id=cam, ts=datetime.now(timezone.utc),
            thumb_path=thumb,
        )
        s.add(sg)
        s.flush()
        ident.rep_sighting_id = sg.id
        ident.last_seen = sg.ts
        s.commit()
        ident_id, sid = ident.id, sg.id
    finally:
        s.close()

    r = client.get(f"/api/identities/{ident_id}/thumbnail")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg"

    r = client.get(f"/api/identities/sightings/{sid}/thumbnail")
    assert r.status_code == 200

    # Missing sighting thumb -> 404.
    r = client.get("/api/identities/sightings/999999/thumbnail")
    assert r.status_code == 404


def test_merge_fallback(client):
    _require_route(client, "/api/identities/merge", "POST")
    cam_a = _seed_camera("CamA")
    cam_b = _seed_camera("CamB")
    now = datetime.now(timezone.utc)
    target_id, _ = _seed_identity(
        name="Person 1", sightings=[{"camera_id": cam_a, "ts": now - timedelta(minutes=20)}]
    )
    source_id, _ = _seed_identity(
        name="Person 2", sightings=[{"camera_id": cam_b, "ts": now - timedelta(minutes=1)}]
    )

    r = client.post(
        "/api/identities/merge",
        json={"target_id": target_id, "source_ids": [source_id]},
    )
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["ok"] is True
    assert res["target_id"] == target_id
    assert res["moved_sightings"] == 1

    # Source is gone.
    r = client.get(f"/api/identities/{source_id}")
    assert r.status_code == 404

    # Target absorbed both sightings on both cameras.
    r = client.get(f"/api/identities/{target_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["num_sightings"] == 2
    assert sorted(detail["cameras"]) == sorted([cam_a, cam_b])


def test_merge_self_only_rejected(client):
    _require_route(client, "/api/identities/merge", "POST")
    cam = _seed_camera()
    ident_id, _ = _seed_identity(
        sightings=[{"camera_id": cam, "ts": datetime.now(timezone.utc)}]
    )
    r = client.post(
        "/api/identities/merge",
        json={"target_id": ident_id, "source_ids": [ident_id]},
    )
    assert r.status_code == 400


def test_split_explicit_fallback(client):
    _require_route(client, "/api/identities/{identity_id}/split", "POST")
    cam = _seed_camera()
    now = datetime.now(timezone.utc)
    ident_id, sids = _seed_identity(
        sightings=[
            {"camera_id": cam, "ts": now - timedelta(minutes=5)},
            {"camera_id": cam, "ts": now - timedelta(minutes=3)},
            {"camera_id": cam, "ts": now - timedelta(minutes=1)},
        ],
    )
    # Peel off one sighting into a new identity.
    r = client.post(
        f"/api/identities/{ident_id}/split",
        json={"sighting_ids": [sids[0]], "new_name": "Person split"},
    )
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["moved_sightings"] == 1
    new_id = res["new_id"]
    assert new_id and new_id != ident_id

    r = client.get(f"/api/identities/{ident_id}")
    assert r.json()["num_sightings"] == 2
    r = client.get(f"/api/identities/{new_id}")
    assert r.json()["num_sightings"] == 1
    assert r.json()["name"] == "Person split"


def test_split_all_sightings_rejected(client):
    _require_route(client, "/api/identities/{identity_id}/split", "POST")
    cam = _seed_camera()
    ident_id, sids = _seed_identity(
        sightings=[{"camera_id": cam, "ts": datetime.now(timezone.utc)}]
    )
    r = client.post(f"/api/identities/{ident_id}/split", json={"sighting_ids": sids})
    assert r.status_code == 400


def test_split_requires_sightings_or_auto(client):
    _require_route(client, "/api/identities/{identity_id}/split", "POST")
    cam = _seed_camera()
    ident_id, _ = _seed_identity(
        sightings=[{"camera_id": cam, "ts": datetime.now(timezone.utc)}]
    )
    r = client.post(f"/api/identities/{ident_id}/split", json={})
    assert r.status_code == 400


def test_delete_identity(client):
    _require_route(client, "/api/identities/{identity_id}", "DELETE")
    cam = _seed_camera()
    ident_id, _ = _seed_identity(
        sightings=[{"camera_id": cam, "ts": datetime.now(timezone.utc)}]
    )
    r = client.delete(f"/api/identities/{ident_id}")
    assert r.status_code == 204
    r = client.get(f"/api/identities/{ident_id}")
    assert r.status_code == 404

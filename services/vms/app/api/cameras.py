"""Camera management API.

CRUD over the ``cameras`` table plus status endpoints. Mutations drive the
WorkerManager so that enabling/disabling a camera or changing its RTSP URL
spawns, stops, or restarts the per-camera worker process. Online/offline
status is reported from the worker heartbeat (persisted by the worker onto the
Camera row, with a freshness check against ``last_seen``).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import require_user
from ..config import get_settings
from ..db.database import get_db
from ..db.models import Camera
from ..schemas import (
    CameraCreate,
    CameraOut,
    CameraStatusOut,
    CameraUpdate,
)

logger = logging.getLogger("vms.api.cameras")

router = APIRouter(prefix="/api/cameras", tags=["cameras"], dependencies=[Depends(require_user)])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _manager(request: Request):
    """Return the shared WorkerManager from app state (or None if absent)."""
    return getattr(request.app.state, "workers", None)


@router.get("/detect/classes")
def detect_classes() -> dict:
    """Return the detector's supported object classes (COCO) for the UI.

    The camera form uses this to render the multi-select of objects that can
    trigger recording. Returns the full ordered list plus the default.
    """
    from ..detect.base import COCO_CLASSES, PERSON_CLASS_ID

    return {
        "classes": COCO_CLASSES,
        "default": [COCO_CLASSES[PERSON_CLASS_ID]],
    }


def _live_status(manager, camera_id: int) -> tuple[str | None, datetime | None]:
    """Read the live worker heartbeat from the WorkerManager registry.

    The per-camera worker process publishes its ``state``/``last_seen`` into the
    manager's shared (cross-process) registry every ~1s — this is the source of
    truth for "is the camera connected right now". The DB row's ``status`` is
    only a coarse fallback (it is not updated on every heartbeat), which is why
    a connected camera could otherwise show "offline". Returns ``(state,
    last_seen)`` or ``(None, None)`` when no live info is available.
    """
    if manager is None:
        return None, None
    getter = getattr(manager, "get_status", None)
    if not callable(getter):
        return None, None
    try:
        info = getter(int(camera_id))
    except Exception:
        return None, None
    if not isinstance(info, dict):
        return None, None
    state = info.get("state")
    ls = info.get("last_seen")
    last_seen = None
    if isinstance(ls, (int, float)):
        last_seen = datetime.fromtimestamp(float(ls), tz=timezone.utc)
    return (state if isinstance(state, str) else None), last_seen


def _effective_status(camera: Camera, manager=None) -> str:
    """Derive the externally-reported status for a camera.

    Prefers the live worker heartbeat from the manager registry (the worker
    publishes there every ~1s); falls back to the persisted Camera row with a
    freshness check. A camera is offline when disabled, when no worker is
    running, or when the last heartbeat is older than the staleness window.
    """
    if not camera.enabled:
        return "offline"

    settings = get_settings()
    stale_after = timedelta(seconds=getattr(settings, "status_stale_seconds", 30))
    now = datetime.now(timezone.utc)

    # 1) Prefer the live worker registry — this is what actually reflects the
    #    current RTSP connection state.
    live_state, live_seen = _live_status(manager, camera.id)
    if live_state is not None:
        if live_state == "error":
            return "error"
        if live_state == "online":
            # Honour heartbeat freshness even on an "online" state.
            if live_seen is not None and now - live_seen > stale_after:
                return "offline"
            return "online"
        # starting / offline / stopped all surface as offline to the UI.
        return "offline"

    # 2) Fall back to the persisted row.
    stored = camera.status or "offline"
    if stored == "error":
        return "error"

    last_seen = camera.last_seen
    if last_seen is None:
        # Enabled but no frame ever seen yet: starting up / unreachable.
        return stored if stored in ("online", "offline", "error") else "offline"

    seen = last_seen if last_seen.tzinfo else last_seen.replace(tzinfo=timezone.utc)
    if now - seen > stale_after:
        return "offline"
    return stored


def _mask_rtsp(url: str | None) -> str | None:
    """Redact embedded credentials so the URL never leaves the server in the
    clear. 'rtsp://user:pass@host:554/s' -> 'rtsp://***@host:554/s'. The host is
    kept (operators need to identify the camera); only the userinfo is hidden."""
    if not url:
        return url
    try:
        scheme, rest = url.split("://", 1)
    except ValueError:
        return url
    if "@" in rest:
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


def _to_out(camera: Camera, manager=None) -> CameraOut:
    """Serialise an ORM Camera into its response schema with derived status."""
    data = CameraOut.model_validate(camera, from_attributes=True)
    # Override the persisted status with the live/freshness-checked value, and
    # surface the live last_seen when the worker has a fresher one than the row.
    _, live_seen = _live_status(manager, camera.id)
    # Never expose RTSP credentials in API responses (the worker reads the raw
    # URL straight from the DB, so masking here doesn't affect capture).
    update = {"status": _effective_status(camera, manager), "rtsp_url": _mask_rtsp(camera.rtsp_url)}
    if live_seen is not None and (camera.last_seen is None or live_seen.replace(tzinfo=None) > (camera.last_seen if camera.last_seen.tzinfo is None else camera.last_seen.replace(tzinfo=None))):
        update["last_seen"] = live_seen
    return data.model_copy(update=update)


def _get_or_404(db: Session, camera_id: int) -> Camera:
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return camera


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #

@router.get("", response_model=list[CameraOut])
def list_cameras(request: Request, db: Session = Depends(get_db)) -> list[CameraOut]:
    """List all configured cameras with their current status."""
    manager = _manager(request)
    cameras = db.execute(select(Camera).order_by(Camera.id)).scalars().all()
    return [_to_out(c, manager) for c in cameras]


@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreate,
    request: Request,
    db: Session = Depends(get_db),
) -> CameraOut:
    """Create a camera; start its worker immediately when enabled."""
    camera = Camera(
        name=payload.name,
        rtsp_url=payload.rtsp_url,
        enabled=payload.enabled if payload.enabled is not None else True,
        detect_conf=payload.detect_conf,
        pre_seconds=payload.pre_seconds,
        post_seconds=payload.post_seconds,
        trigger_classes=payload.trigger_classes,
        detect_iou=payload.detect_iou,
        detect_imgsz=payload.detect_imgsz,
        detect_interval=payload.detect_interval,
        trigger_cooldown=payload.trigger_cooldown,
        min_trigger_frames=payload.min_trigger_frames,
        rtsp_transport=payload.rtsp_transport,
        faces_enabled=payload.faces_enabled,
        reid_enabled=payload.reid_enabled,
        status="offline",
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)

    manager = _manager(request)
    if manager is not None and camera.enabled:
        manager.start_camera(camera)

    return _to_out(camera, manager)


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: int, request: Request, db: Session = Depends(get_db)) -> CameraOut:
    """Fetch a single camera."""
    return _to_out(_get_or_404(db, camera_id), _manager(request))


@router.put("/{camera_id}", response_model=CameraOut)
def update_camera(
    camera_id: int,
    payload: CameraUpdate,
    request: Request,
    db: Session = Depends(get_db),
) -> CameraOut:
    """Partial-update a camera.

    The worker is restarted when the RTSP URL changes, started when the camera
    transitions to enabled, and stopped when it transitions to disabled. Tuning
    fields (detect_conf / pre / post seconds) also trigger a restart of a
    running worker so the new values take effect.
    """
    camera = _get_or_404(db, camera_id)
    fields = payload.model_dump(exclude_unset=True)

    # The API only ever returns a MASKED rtsp_url, so the UI may echo that
    # redacted value back on an unrelated edit. Treat a blank or masked URL as
    # "keep the current one" — only a real, new URL replaces the stored secret.
    if "rtsp_url" in fields:
        v = (fields["rtsp_url"] or "").strip()
        if not v or "***" in v:
            fields.pop("rtsp_url")

    was_enabled = camera.enabled
    old_rtsp = camera.rtsp_url
    tuning_keys = {
        "detect_conf", "pre_seconds", "post_seconds", "trigger_classes",
        "detect_iou", "detect_imgsz", "detect_interval", "trigger_cooldown",
        "min_trigger_frames", "rtsp_transport", "faces_enabled", "reid_enabled",
    }
    tuning_changed = any(
        k in fields and fields[k] != getattr(camera, k) for k in tuning_keys
    )

    rtsp_changed = "rtsp_url" in fields and fields["rtsp_url"] != old_rtsp

    for key, value in fields.items():
        setattr(camera, key, value)

    # If the camera is being (re)enabled or disabled, reset transient status so
    # we don't report a stale 'online'.
    if "enabled" in fields and not camera.enabled:
        camera.status = "offline"
        camera.last_seen = None

    # Repointing the RTSP URL means this is effectively a different physical
    # camera now — purge the prior events + their files so we never show old
    # footage from the source that used to live here.
    if rtsp_changed:
        from ..db.models import Event
        from ..recording.cleanup import delete_event_artifacts

        old_events = db.scalars(select(Event).where(Event.camera_id == camera.id)).all()
        settings = get_settings()
        for ev in old_events:
            try:
                delete_event_artifacts(settings, camera.id, ev.id)
            except Exception:
                pass
            db.delete(ev)
        if old_events:
            logger.info("Camera %s repointed; purged %d stale events", camera.id, len(old_events))

    db.commit()
    db.refresh(camera)

    manager = _manager(request)
    if manager is not None:
        now_enabled = camera.enabled

        if not now_enabled and was_enabled:
            manager.stop_camera(camera.id)
        elif now_enabled and not was_enabled:
            manager.start_camera(camera)
        elif now_enabled and (rtsp_changed or tuning_changed):
            manager.restart_camera(camera)

    return _to_out(camera, manager)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    """Delete a camera. Stops its worker; events cascade per FK; files purged."""
    camera = _get_or_404(db, camera_id)

    manager = _manager(request)
    if manager is not None:
        manager.stop_camera(camera.id)
    hls = getattr(request.app.state, "hls", None)
    if hls is not None:
        hls.stop(camera.id)

    cam_id = int(camera.id)
    settings = get_settings()
    from ..db.models import (
        AppearanceExemplar, Event, FaceExemplar, FaceSample, Identity,
        PresenceSegment, Sighting,
    )
    from ..recording.cleanup import delete_camera_artifacts

    # On-disk artifacts (clips/thumbnails/segments/hls/frame slot) BEFORE the
    # cascade drops the event rows (so the file paths are still resolvable).
    event_ids = [int(e) for e in db.scalars(select(Event.id).where(Event.camera_id == cam_id)).all()]
    try:
        delete_camera_artifacts(settings, cam_id, event_ids)
    except Exception:
        logger.warning("artifact cleanup failed for camera %s", cam_id, exc_info=True)

    # Face-sample thumbnails on disk for this camera.
    for fs in db.scalars(select(FaceSample).where(FaceSample.camera_id == cam_id)).all():
        if fs.thumb_path:
            try:
                p = os.path.join(os.path.dirname(os.path.abspath(str(settings.data_dir))), fs.thumb_path)
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass

    # Rows derived from this camera that have NO camera FK (so the DB cascade
    # never touches them): face samples, presence segments, exemplars.
    for model in (FaceSample, PresenceSegment, FaceExemplar, AppearanceExemplar):
        db.query(model).filter(model.camera_id == cam_id).delete(synchronize_session=False)

    # Camera delete cascades its events + sightings (FK ON DELETE CASCADE).
    db.delete(camera)
    db.commit()

    # Identities are cross-camera and have no camera link, so after the cascade
    # removed this camera's sightings, drop any identity left with NO sightings
    # (i.e. it existed only because of this camera). Identities still seen on
    # other cameras survive.
    live_iids = set(db.scalars(select(Sighting.identity_id).distinct()).all())
    orphans = [
        i for i in db.scalars(select(Identity)).all()
        if int(i.id) not in {int(x) for x in live_iids if x is not None}
    ]
    orphan_ids = [int(i.id) for i in orphans]
    for ident in orphans:
        db.delete(ident)  # cascades its remaining exemplars/sightings
    if orphan_ids and hasattr(Event, "identity_id"):
        vals = {"identity_id": None}
        if hasattr(Event, "identity_name"):
            vals["identity_name"] = None
        from sqlalchemy import update as _update
        db.execute(_update(Event).where(Event.identity_id.in_(orphan_ids)).values(**vals))
    db.commit()

    # Converge derived state: identity gallery + FAISS face index.
    try:
        from ..api.identities import _reload_gallery
        _reload_gallery(request)
    except Exception:
        logger.warning("gallery reload after camera delete failed", exc_info=True)
    try:
        from ..api.people import _rebuild_face_index
        _rebuild_face_index(request, db)
    except Exception:
        logger.warning("face index rebuild after camera delete failed", exc_info=True)

    logger.info("Deleted camera %s + derived data (%d orphan identities)", cam_id, len(orphan_ids))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

@router.get("/{camera_id}/status", response_model=CameraStatusOut)
def camera_status(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> CameraStatusOut:
    """Live status for one camera: status, last_seen, fps, detector backend.

    ``fps`` and ``detector`` come from the live WorkerManager status registry
    when a worker is running; otherwise they fall back to None / the configured
    default backend.
    """
    camera = _get_or_404(db, camera_id)

    fps: float | None = None
    detector: str | None = None

    manager = _manager(request)
    if manager is not None:
        info = None
        getter = getattr(manager, "get_status", None)
        if callable(getter):
            info = getter(camera.id)
        if info:
            # Tolerate either a dict or an attribute-style status object.
            if isinstance(info, dict):
                fps = info.get("fps")
                detector = info.get("detector")
            else:
                fps = getattr(info, "fps", None)
                detector = getattr(info, "detector", None)

    if detector is None:
        settings = get_settings()
        detector = getattr(settings, "detector_backend", None)

    _, live_seen = _live_status(manager, camera.id)
    return CameraStatusOut(
        status=_effective_status(camera, manager),
        last_seen=live_seen or camera.last_seen,
        fps=fps,
        detector=detector,
    )

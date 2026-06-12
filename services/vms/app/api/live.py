"""Live monitoring API.

MJPEG (``multipart/x-mixed-replace``) streaming and single-snapshot JPEG
endpoints. Both read the latest *annotated* JPEG frame that each camera worker
publishes into the :class:`WorkerManager` frame slots (one slot per camera).

The MJPEG stream is the simplest reliable option for the live grid: each
``<img src="/api/live/{id}/stream">`` tag in the SPA holds one long-lived HTTP
connection and the browser swaps in each new JPEG part as it arrives. nginx is
configured with ``proxy_buffering off`` so parts flush per-frame.

No transcoding happens here — the worker already encodes annotated frames to
JPEG (boxes drawn). This module only multiplexes the bytes onto HTTP.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ..auth import require_auth
from ..config import get_settings
from ..db.database import get_db
from ..db.models import Camera, Event

router = APIRouter(prefix="/api/live", tags=["live"])

# Multipart boundary token used for the MJPEG stream.
_BOUNDARY = "vmsframe"

# 1x1 black JPEG placeholder served while a worker has not yet published a frame
# (camera starting up, reconnecting, or detector warming up). Keeps the
# ``<img>`` element from breaking and lets the browser keep the stream open.
_PLACEHOLDER_JPEG: bytes = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707"
    "07090908"  # quant table (truncated marker sequence below)
)


def _make_placeholder() -> bytes:
    """Build a tiny valid black JPEG at import time.

    We try to use OpenCV/numpy if available (always present in this image since
    workers depend on it); otherwise fall back to a hand-rolled minimal JPEG so
    the endpoint never hard-fails on a missing frame.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "no signal",
            (90, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (64, 64, 64),
            2,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    return _PLACEHOLDER_JPEG


_PLACEHOLDER = _make_placeholder()


def _manager(request: Request):
    """Return the WorkerManager from app state, or 503 if it is not booted."""
    manager = getattr(request.app.state, "workers", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="Worker manager not available")
    return manager


def _camera_or_404(db, camera_id: int) -> Camera:
    """Verify the camera exists in the DB (independent of worker state)."""
    cam = db.get(Camera, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return cam


def _get_frame(manager, camera_id: int) -> bytes | None:
    """Pull the latest annotated JPEG bytes for a camera from the manager.

    The WorkerManager owns the frame slots; we support a few accessor shapes so
    this endpoint integrates regardless of the exact method name the workers
    component settles on. Returns ``None`` when no frame is available yet.
    """
    # Preferred explicit accessor. WorkerManager exposes read_frame().
    getter = (
        getattr(manager, "get_frame", None)
        or getattr(manager, "get_latest_frame", None)
        or getattr(manager, "read_frame", None)
    )
    if callable(getter):
        try:
            frame = getter(camera_id)
        except KeyError:
            return None
        return frame if frame else None

    # Fallback: a dict-like mapping of camera_id -> jpeg bytes.
    frames = getattr(manager, "frames", None)
    if frames is not None:
        try:
            frame = frames.get(camera_id) if hasattr(frames, "get") else frames[camera_id]
        except (KeyError, TypeError):
            return None
        return frame if frame else None

    return None


@router.get("/{camera_id}/snapshot")
async def snapshot(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> Response:
    """Return the most recent annotated frame as a single ``image/jpeg``.

    404 if the camera does not exist; serves a placeholder (200) if the worker
    has not produced a frame yet so the UI degrades gracefully.
    """
    _camera_or_404(db, camera_id)
    manager = _manager(request)

    frame = _get_frame(manager, camera_id)
    if not frame:
        frame = _PLACEHOLDER

    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


async def _mjpeg_generator(manager, camera_id: int, fps: float):
    """Yield multipart MJPEG parts at up to ``fps``, one part per JPEG frame.

    Sends a frame only when it changes (cheap identity/length check) to avoid
    re-pushing identical bytes, but always pushes at least one initial part so
    the browser renders immediately. Stops when the client disconnects (the
    ``StreamingResponse`` cancels this coroutine and ``asyncio.sleep`` raises
    ``CancelledError``).
    """
    interval = 1.0 / fps if fps > 0 else 0.1
    last_id: int | None = None
    last_len: int = -1

    # Prime with an immediate frame (real or placeholder) so the <img> shows up.
    frame = _get_frame(manager, camera_id) or _PLACEHOLDER
    yield _part(frame)
    last_id = id(frame)
    last_len = len(frame)

    while True:
        await asyncio.sleep(interval)
        frame = _get_frame(manager, camera_id)
        if not frame:
            # Nothing new yet — keep the connection warm but don't spam bytes.
            continue
        fid, flen = id(frame), len(frame)
        if fid == last_id and flen == last_len:
            continue
        last_id, last_len = fid, flen
        yield _part(frame)


def _part(frame: bytes) -> bytes:
    """Format one MJPEG multipart chunk for a JPEG frame."""
    header = (
        f"--{_BOUNDARY}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(frame)}\r\n\r\n"
    ).encode("ascii")
    return header + frame + b"\r\n"


@router.get("/{camera_id}/stream")
async def stream(
    camera_id: int,
    request: Request,
    fps: float | None = None,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> StreamingResponse:
    """Stream annotated frames as ``multipart/x-mixed-replace`` MJPEG.

    Long-lived response; the browser renders each JPEG part in place. Frame rate
    is capped by ``settings.live_mjpeg_fps`` to bound bandwidth/CPU; an optional
    ``?fps=`` lets the grid request a LOWER rate (many tiles at once) while a
    focused/fullscreen viewer asks for the full rate. nginx must run with
    ``proxy_buffering off`` for per-frame flushing.
    """
    _camera_or_404(db, camera_id)
    manager = _manager(request)

    settings = get_settings()
    cap = float(getattr(settings, "live_mjpeg_fps", 10.0) or 10.0)
    use_fps = cap if fps is None else max(1.0, min(cap, float(fps)))

    return StreamingResponse(
        _mjpeg_generator(manager, camera_id, use_fps),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Connection": "close",
            "X-Accel-Buffering": "no",  # belt-and-suspenders: disable nginx buffering
        },
    )


# --------------------------------------------------------------------------- #
# MANUAL recording — operator presses ● REC in Live Monitoring. Stateless: the
# client holds the server-issued start time and sends it back on stop; the clip
# is assembled from the warm on-disk segment buffer (no worker round-trip).
# --------------------------------------------------------------------------- #

_MANUAL_PRE = 1   # seconds of pre-roll to include before the pressed-start
_MANUAL_POST = 1  # seconds of post-roll after stop


@router.post("/{camera_id}/record/start")
async def record_start(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    """Begin a manual recording. Returns the server-trusted start timestamp the
    client echoes back on stop (keeps the server stateless)."""
    _camera_or_404(db, camera_id)
    return {"camera_id": camera_id, "started_at": datetime.now(timezone.utc).isoformat()}


@router.post("/{camera_id}/record/stop")
async def record_stop(
    camera_id: int,
    payload: dict,
    request: Request,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
) -> dict:
    """Finish a manual recording: assemble a clip spanning [started_at, now] from
    the rolling segment buffer and persist it as a ``manual`` event."""
    _camera_or_404(db, camera_id)
    settings = get_settings()

    raw = str((payload or {}).get("started_at") or "")
    try:
        start_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid started_at")

    end_dt = datetime.now(timezone.utc)
    start_epoch = start_dt.timestamp()
    end_epoch = end_dt.timestamp()
    if end_epoch - start_epoch < 0.5:
        start_epoch = end_epoch - 1.0  # ensure a minimum clip length
    # Bound by the segment retention so we never ask for footage already pruned.
    retention = float(getattr(settings, "segment_retention_seconds", 120))
    if end_epoch - start_epoch > retention - 5:
        start_epoch = end_epoch - (retention - 5)

    # Persist the event up-front (naive-UTC ts, matching the worker convention).
    ev = Event(
        camera_id=camera_id,
        ts=datetime.utcfromtimestamp(start_epoch),
        label="manual",
        num_objects=1,
        object_classes="manual",
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    event_id = int(ev.id)

    # Assemble the clip off the event loop (it waits briefly for post-roll).
    from ..recording.clipper import build_clip_from_track

    seg_shim = SimpleNamespace(
        segments_dir=settings.segments_dir / str(camera_id),
        segment_seconds=int(getattr(settings, "segment_seconds", 2)),
        retention_seconds=retention,
        pre_seconds=_MANUAL_PRE,
        post_seconds=_MANUAL_POST,
    )
    built = await run_in_threadpool(
        build_clip_from_track,
        segmenter=seg_shim,
        camera_id=camera_id,
        event_id=event_id,
        enter_ts=start_epoch,
        last_ts=end_epoch,
        data_dir=settings.data_dir,
        pre_seconds=_MANUAL_PRE,
        post_seconds=_MANUAL_POST,
    )

    ev = db.get(Event, event_id)
    if ev is not None:
        if getattr(built, "clip_path", None):
            ev.clip_path = built.clip_path
        if getattr(built, "thumb_path", None):
            ev.thumb_path = built.thumb_path
        db.commit()

    return {
        "event_id": event_id,
        "clip": bool(getattr(built, "clip_path", None)),
        "duration": round(end_epoch - start_epoch, 1),
    }


# --------------------------------------------------------------------------- #
# Live WITH SOUND — on-demand RTSP -> HLS (MJPEG carries no audio).
# These are sync `def` endpoints so the brief startup poll runs in the
# threadpool without blocking the event loop. hls.js (or Safari native) plays
# the playlist; relative segment names resolve to the segment route below.
# --------------------------------------------------------------------------- #
_HLS_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def _hls(request: Request):
    return getattr(request.app.state, "hls", None)


@router.get("/{camera_id}/hls/index.m3u8")
def hls_playlist(
    camera_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
):
    """Start (idempotent) the camera's HLS session and serve the playlist."""
    cam = _camera_or_404(db, camera_id)
    hls = _hls(request)
    if hls is None:
        raise HTTPException(status_code=503, detail="Live-with-sound is disabled")
    if not hls.start(camera_id, cam.rtsp_url):
        raise HTTPException(status_code=503, detail="Live-with-sound at capacity; try again shortly")
    path = hls.playlist_path(camera_id)
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        try:
            if path.is_file() and path.stat().st_size > 0:
                break
        except OSError:
            pass
        time.sleep(0.2)
    if not (path.is_file() and path.stat().st_size > 0):
        raise HTTPException(status_code=503, detail="HLS starting; retry")
    return FileResponse(str(path), media_type="application/vnd.apple.mpegurl", headers=_HLS_HEADERS)


@router.get("/{camera_id}/hls/close")
@router.post("/{camera_id}/hls/close")
def hls_close(camera_id: int, request: Request, _auth=Depends(require_auth)):
    """Stop a camera's HLS session (called on player close / sendBeacon)."""
    hls = _hls(request)
    if hls is not None:
        hls.stop(camera_id)
    return {"ok": True}


@router.get("/{camera_id}/hls/{segment}")
def hls_segment(
    camera_id: int,
    segment: str,
    request: Request,
    _auth=Depends(require_auth),
):
    """Serve one HLS .ts segment (name validated against ^seg\\d{5}\\.ts$)."""
    hls = _hls(request)
    if hls is None:
        raise HTTPException(status_code=404, detail="Not found")
    p = hls.segment_path(camera_id, segment)
    if p is None or not p.is_file():
        raise HTTPException(status_code=404, detail="Segment not found")
    return FileResponse(str(p), media_type="video/mp2t", headers=_HLS_HEADERS)

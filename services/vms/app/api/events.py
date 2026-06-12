"""Events history API.

Read/serve side of recordings and matched identities:

  GET    /api/events                 list/filter (camera_id, person_id, from, to, label, limit, offset)
  GET    /api/events/{id}            full event
  DELETE /api/events/{id}            delete event row + clip + thumbnail files
  GET    /api/events/{id}/clip       Range-supported streaming of the recorded mp4 clip
  GET    /api/events/{id}/thumbnail  JPEG thumbnail

All routes are behind the trusted-header SSO dependency (see app.auth). The clip
and thumbnail endpoints additionally support the optional Bearer API_KEY path so
that an HTML5 <video>/<img> tag can fetch them through nginx with the SSO cookie.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..auth import require_auth
from ..config import get_settings
from ..db.database import get_db
from ..db.models import Camera, Event
from ..schemas import EventDetail, EventListItem, EventListResponse

logger = logging.getLogger("vms.api.events")

router = APIRouter(prefix="/api/events", tags=["events"])

settings = get_settings()

# 1 MiB chunks for Range / full-file streaming of clips.
_CHUNK_SIZE = 1024 * 1024
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _abs_data_path(rel_path: str) -> str:
    """Resolve a DB-stored relative path against the configured data dir.

    Stored paths look like ``data/recordings/<cam>/<id>.mp4`` (relative to the
    app working dir) or already-absolute paths. We resolve and then guard
    against path traversal escaping the data root.
    """
    data_root = os.path.abspath(str(settings.data_dir))
    if os.path.isabs(rel_path):
        candidate = os.path.abspath(rel_path)
    else:
        # DB paths are relative to the app working dir; they begin with the
        # data dir's basename (e.g. "data/..."). Join against the data root's
        # parent so "data/recordings/..." resolves correctly, and also accept
        # paths already relative to the data root.
        parent = os.path.dirname(data_root)
        candidate = os.path.abspath(os.path.join(parent, rel_path))
        if not os.path.exists(candidate):
            alt = os.path.abspath(os.path.join(data_root, rel_path))
            if os.path.exists(alt):
                candidate = alt
    # Containment guard.
    if os.path.commonpath([candidate, data_root]) != data_root:
        raise HTTPException(status_code=400, detail="Invalid stored path")
    return candidate


def _get_event_or_404(db: Session, event_id: int) -> Event:
    event = db.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def _thumb_url(event_id: int) -> str:
    return f"/api/events/{event_id}/thumbnail"


def _clip_url(event_id: int) -> str:
    return f"/api/events/{event_id}/clip"


def _to_list_item(event: Event, camera_name: Optional[str]) -> EventListItem:
    return EventListItem(
        id=event.id,
        camera_id=event.camera_id,
        camera_name=camera_name,
        ts=event.ts,
        end_ts=event.end_ts,
        label=event.label,
        person_id=event.person_id,
        person_name=event.person_name,
        match_score=event.match_score,
        identity_id=getattr(event, "identity_id", None),
        identity_name=getattr(event, "identity_name", None),
        num_objects=getattr(event, "num_objects", None),
        object_classes=getattr(event, "object_classes", None),
        peak_confidence=getattr(event, "peak_confidence", None),
        num_frames=getattr(event, "num_frames", None),
        duration_seconds=_duration_seconds(event),
        thumb_url=_thumb_url(event.id) if event.thumb_path else None,
        clip_url=_clip_url(event.id) if event.clip_path else None,
    )


def _duration_seconds(event) -> Optional[float]:
    ts, end = getattr(event, "ts", None), getattr(event, "end_ts", None)
    if ts is not None and end is not None:
        try:
            return round((end - ts).total_seconds(), 1)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# List / filter
# --------------------------------------------------------------------------- #
@router.get("", response_model=EventListResponse)
def list_events(
    camera_id: Optional[int] = Query(None, description="Filter by camera id"),
    person_id: Optional[int] = Query(None, description="Filter by recognized person id"),
    from_: Optional[datetime] = Query(None, alias="from", description="ISO start time (inclusive)"),
    to: Optional[datetime] = Query(None, description="ISO end time (exclusive)"),
    label: Optional[str] = Query(None, description="Detected class label (default 'person')"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: object = Depends(require_auth),
) -> EventListResponse:
    """List events newest-first with optional filtering and pagination."""
    filters = []
    if camera_id is not None:
        filters.append(Event.camera_id == camera_id)
    if person_id is not None:
        filters.append(Event.person_id == person_id)
    if label is not None:
        filters.append(Event.label == label)
    if from_ is not None:
        filters.append(Event.ts >= from_)
    if to is not None:
        filters.append(Event.ts < to)

    count_stmt = select(func.count()).select_from(Event)
    for f in filters:
        count_stmt = count_stmt.where(f)
    total = db.execute(count_stmt).scalar_one()

    stmt = select(Event, Camera.name)
    stmt = stmt.outerjoin(Camera, Camera.id == Event.camera_id)
    for f in filters:
        stmt = stmt.where(f)
    stmt = stmt.order_by(Event.ts.desc(), Event.id.desc()).limit(limit).offset(offset)

    rows = db.execute(stmt).all()
    items = [_to_list_item(event, camera_name) for event, camera_name in rows]
    return EventListResponse(total=total, items=items)


# --------------------------------------------------------------------------- #
# Get single
# --------------------------------------------------------------------------- #
@router.get("/{event_id}", response_model=EventDetail)
def get_event(
    event_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_auth),
) -> EventDetail:
    event = _get_event_or_404(db, event_id)
    camera = db.get(Camera, event.camera_id)
    return EventDetail(
        id=event.id,
        camera_id=event.camera_id,
        camera_name=camera.name if camera else None,
        ts=event.ts,
        end_ts=event.end_ts,
        label=event.label,
        person_id=event.person_id,
        person_name=event.person_name,
        match_score=event.match_score,
        identity_id=getattr(event, "identity_id", None),
        identity_name=getattr(event, "identity_name", None),
        num_objects=getattr(event, "num_objects", None),
        object_classes=getattr(event, "object_classes", None),
        peak_confidence=getattr(event, "peak_confidence", None),
        num_frames=getattr(event, "num_frames", None),
        duration_seconds=_duration_seconds(event),
        clip_path=event.clip_path,
        thumb_path=event.thumb_path,
        thumb_url=_thumb_url(event.id) if event.thumb_path else None,
        clip_url=_clip_url(event.id) if event.clip_path else None,
        created_at=event.created_at,
    )


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #
@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_event(
    event_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_auth),
) -> Response:
    event = _get_event_or_404(db, event_id)

    # Best-effort removal of on-disk artifacts before dropping the row.
    for rel in (event.clip_path, event.thumb_path):
        if not rel:
            continue
        try:
            path = _abs_data_path(rel)
            if os.path.isfile(path):
                os.remove(path)
        except HTTPException:
            logger.warning("Refusing to delete out-of-root artifact for event %s: %r", event_id, rel)
        except OSError as exc:
            logger.warning("Failed to remove artifact %r for event %s: %s", rel, event_id, exc)

    db.delete(event)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _purge_event_files(event) -> None:
    for rel in (event.clip_path, event.thumb_path):
        if not rel:
            continue
        try:
            path = _abs_data_path(rel)
            if os.path.isfile(path):
                os.remove(path)
        except HTTPException:
            pass
        except OSError:
            pass


@router.post("/bulk-delete")
def bulk_delete_events(
    ids: list[int] = Body(..., embed=True, max_length=1000),
    db: Session = Depends(get_db),
    _: object = Depends(require_auth),
) -> dict:
    """Delete the given events (+ their clip/thumbnail files)."""
    if not ids:
        return {"deleted": 0}
    n = 0
    for ev in db.query(Event).filter(Event.id.in_([int(i) for i in ids])).all():
        _purge_event_files(ev)
        db.delete(ev)
        n += 1
    db.commit()
    return {"deleted": n}


@router.post("/clear-all")
def clear_all_events(
    confirm: bool = Body(False, embed=True),
    camera_id: Optional[int] = Body(None, embed=True),
    label: Optional[str] = Body(None, embed=True),
    db: Session = Depends(get_db),
    _: object = Depends(require_auth),
) -> dict:
    """Delete ALL events (optionally filtered by camera/label) + their files.

    Requires ``confirm=true``. Sightings referencing the events are SET NULL by
    the FK, so they survive. Deletes in chunks to avoid long write locks."""
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true required")
    q = db.query(Event)
    if camera_id is not None:
        q = q.filter(Event.camera_id == camera_id)
    if label is not None:
        q = q.filter(Event.label == label)
    n = 0
    while True:
        batch = q.limit(200).all()
        if not batch:
            break
        for ev in batch:
            _purge_event_files(ev)
            db.delete(ev)
            n += 1
        db.commit()
        if len(batch) < 200:
            break
    return {"deleted": n}


# --------------------------------------------------------------------------- #
# Clip streaming (HTTP Range)
# --------------------------------------------------------------------------- #
def _iter_file_range(path: str, start: int, end: int):
    """Yield bytes [start, end] inclusive from a file in _CHUNK_SIZE chunks."""
    remaining = end - start + 1
    with open(path, "rb") as fh:
        fh.seek(start)
        while remaining > 0:
            chunk = fh.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.get("/{event_id}/clip")
def get_clip(
    event_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _: object = Depends(require_auth),
):
    """Stream the recorded mp4 clip with HTTP Range support for <video> seeking."""
    event = _get_event_or_404(db, event_id)
    if not event.clip_path:
        raise HTTPException(status_code=404, detail="Event has no clip")

    path = _abs_data_path(event.clip_path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Clip file missing on disk")

    file_size = os.path.getsize(path)
    range_header = request.headers.get("range") or request.headers.get("Range")

    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'inline; filename="event_{event_id}.mp4"',
        "Cache-Control": "private, max-age=3600",
    }

    if not range_header:
        # Full-file response; still streamed in chunks to keep memory flat.
        headers = {**base_headers, "Content-Length": str(file_size)}
        return StreamingResponse(
            _iter_file_range(path, 0, file_size - 1) if file_size else iter(()),
            media_type="video/mp4",
            headers=headers,
        )

    match = _RANGE_RE.fullmatch(range_header.strip())
    if not match:
        raise HTTPException(status_code=400, detail="Malformed Range header")

    start_s, end_s = match.group(1), match.group(2)
    if start_s == "" and end_s == "":
        raise HTTPException(status_code=400, detail="Malformed Range header")

    if start_s == "":
        # Suffix range: last N bytes.
        length = int(end_s)
        if length <= 0:
            raise HTTPException(status_code=416, detail="Invalid range")
        start = max(file_size - length, 0)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s != "" else file_size - 1

    if start >= file_size or start > end:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    end = min(end, file_size - 1)
    content_length = end - start + 1

    headers = {
        **base_headers,
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(content_length),
    }
    return StreamingResponse(
        _iter_file_range(path, start, end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type="video/mp4",
        headers=headers,
    )


# --------------------------------------------------------------------------- #
# Thumbnail
# --------------------------------------------------------------------------- #
@router.get("/{event_id}/thumbnail")
def get_thumbnail(
    event_id: int,
    db: Session = Depends(get_db),
    _: object = Depends(require_auth),
):
    """Serve the event thumbnail JPEG."""
    event = _get_event_or_404(db, event_id)
    if not event.thumb_path:
        raise HTTPException(status_code=404, detail="Event has no thumbnail")

    path = _abs_data_path(event.thumb_path)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Thumbnail file missing on disk")

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )

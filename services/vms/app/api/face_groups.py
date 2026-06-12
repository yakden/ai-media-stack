"""Face-grouping API — the dedicated face-recognition layer.

Clusters captured :class:`~app.db.models.FaceSample` rows into groups of the
same person by cosine similarity, with operator-tunable settings (similarity
threshold, clothing-fusion weight, minimum group size). Cross-angle robustness
comes from ArcFace + many samples per person; clothing is an optional extra
signal. Grouping is computed on demand (face counts are modest), so the settings
are just query parameters — no precomputed state to invalidate.

  GET    /api/face-groups                 cluster + list groups (settings in query)
  GET    /api/face-groups/samples/{id}/thumbnail   face crop JPEG
  POST   /api/face-groups/label           name a group (set label on its samples)
  DELETE /api/face-groups/samples/{id}     drop one captured face
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import require_auth
from ..config import Settings, get_settings
from fastapi import Request

from ..db.database import get_db
from ..db.models import Camera, Event, FaceEmbedding, FaceSample, Person, Sighting
from ..faces.grouping import FaceSampleVec, cluster_faces
from ..reid.gallery import deserialize_vector
from ..schemas import EventListItem, EventListResponse

logger = logging.getLogger("vms.api.face_groups")

router = APIRouter(
    prefix="/api/face-groups",
    tags=["face-groups"],
    dependencies=[Depends(require_auth)],
)


def _resolve_data_path(settings: Settings, stored: Optional[str]) -> Optional[str]:
    if not stored:
        return None
    data_root = os.path.abspath(str(settings.data_dir))
    candidate = (
        os.path.abspath(stored)
        if os.path.isabs(stored)
        else os.path.abspath(os.path.join(os.path.dirname(data_root), stored))
    )
    if not os.path.exists(candidate):
        alt = os.path.abspath(os.path.join(data_root, stored))
        if os.path.exists(alt):
            candidate = alt
    try:
        if os.path.commonpath([candidate, data_root]) != data_root:
            return None
    except ValueError:
        return None
    return candidate


def _thumb_url(sample_id: int) -> str:
    return f"/api/face-groups/samples/{sample_id}/thumbnail"


@router.get("")
def list_face_groups(
    face_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Min cosine to link two faces"),
    clothing_weight: float = Query(0.0, ge=0.0, le=1.0, description="Blend of clothing similarity"),
    min_size: int = Query(1, ge=1, le=100, description="Hide groups smaller than this"),
    max_members: int = Query(40, ge=1, le=500, description="Member thumbnails per group to return"),
    max_samples: int = Query(2000, ge=1, le=2000, description="Cap on samples clustered"),
    db: Session = Depends(get_db),
) -> dict:
    """Cluster captured faces and return groups (largest first)."""
    rows = db.scalars(
        select(FaceSample).order_by(FaceSample.id.desc()).limit(max_samples)
    ).all()

    samples: list[FaceSampleVec] = []
    for r in rows:
        try:
            vec = deserialize_vector(r.vector)
        except Exception:
            continue
        app = None
        if r.app_vector:
            try:
                app = deserialize_vector(r.app_vector)
            except Exception:
                app = None
        samples.append(FaceSampleVec(
            id=int(r.id), vec=vec, app=app, quality=float(r.quality or 0.0),
            camera_id=r.camera_id, ts=r.ts, identity_id=r.identity_id,
            label=r.label, thumb_path=r.thumb_path,
        ))

    groups = cluster_faces(
        samples, face_threshold=face_threshold,
        clothing_weight=clothing_weight, min_size=min_size,
    )

    out = []
    for g in groups:
        members = g.members
        labels = [m.label for m in members if m.label]
        cams = sorted({m.camera_id for m in members if m.camera_id is not None})
        ts_vals = [m.ts for m in members if m.ts is not None]
        rep = members[0]  # highest quality (grouping sorted)
        out.append({
            "group_id": int(rep.id),  # stable key = representative sample id
            "size": g.size,
            "label": (max(set(labels), key=labels.count) if labels else None),
            "cameras": cams,
            "first_seen": (min(ts_vals).isoformat() if ts_vals else None),
            "last_seen": (max(ts_vals).isoformat() if ts_vals else None),
            "representative": {"sample_id": int(rep.id), "thumb_url": _thumb_url(rep.id)},
            "member_sample_ids": [int(m.id) for m in members],
            "members": [
                {
                    "id": int(m.id),
                    "thumb_url": _thumb_url(m.id),
                    "camera_id": m.camera_id,
                    "ts": (m.ts.isoformat() if m.ts is not None else None),
                    "quality": round(float(m.quality), 3),
                    "identity_id": m.identity_id,
                }
                for m in members[:max_members]
            ],
        })

    return {
        "total_samples": len(samples),
        "total_groups": len(out),
        "settings": {
            "face_threshold": face_threshold,
            "clothing_weight": clothing_weight,
            "min_size": min_size,
        },
        "groups": out,
    }


@router.get("/samples/{sample_id}/thumbnail")
def get_face_thumbnail(sample_id: int, db: Session = Depends(get_db)):
    sample = db.get(FaceSample, sample_id)
    if sample is None or not sample.thumb_path:
        raise HTTPException(status_code=404, detail="Face sample/thumbnail not found")
    abs_path = _resolve_data_path(get_settings(), sample.thumb_path)
    if not abs_path or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Thumbnail missing on disk")
    return FileResponse(abs_path, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


@router.post("/enroll")
def enroll_group(
    sample_ids: list[int] = Body(..., embed=True, max_length=5000),
    name: str = Body(..., embed=True, max_length=200),
    request: Request = None,
    db: Session = Depends(get_db),
) -> dict:
    """Turn a face group into a known Person: create the Person and enrol its
    best face crops into the recognition DB (FaceEmbedding + FAISS rebuild), so
    future detections of this face are matched by name."""
    pname = (name or "").strip()
    if not pname:
        raise HTTPException(status_code=400, detail="name required")
    if not sample_ids:
        raise HTTPException(status_code=400, detail="sample_ids required")

    # Best (highest-quality) faces first; cap to avoid bloating the index.
    rows = db.scalars(
        select(FaceSample)
        .where(FaceSample.id.in_([int(s) for s in sample_ids]))
        .order_by(FaceSample.quality.desc())
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No such face samples")

    person = Person(name=pname)
    db.add(person)
    db.flush()  # assign person.id

    enrolled = 0
    for fs in rows[:12]:
        # FaceSample.vector uses the same 512-d little-endian f32 layout as
        # FaceEmbedding.vector, so the blob copies directly.
        db.add(FaceEmbedding(person_id=person.id, vector=fs.vector, image_path=fs.thumb_path))
        enrolled += 1
    # Label the group's samples with the new name.
    db.query(FaceSample).filter(FaceSample.id.in_([int(s) for s in sample_ids])).update(
        {FaceSample.label: pname}, synchronize_session=False
    )
    db.commit()

    # Rebuild the FAISS face index so live matching uses the new person.
    try:
        from .people import _rebuild_face_index
        _rebuild_face_index(request, db)
    except Exception:
        logger.exception("Face index rebuild after enroll failed")

    return {"person_id": int(person.id), "name": pname, "enrolled_faces": enrolled}


@router.post("/label")
def label_group(
    sample_ids: list[int] = Body(..., embed=True, max_length=5000),
    label: str = Body(..., embed=True, max_length=2000),
    db: Session = Depends(get_db),
) -> dict:
    """Name a group: set ``label`` on the given face samples."""
    name = (label or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="label required")
    if not sample_ids:
        raise HTTPException(status_code=400, detail="sample_ids required")
    n = (
        db.query(FaceSample)
        .filter(FaceSample.id.in_([int(s) for s in sample_ids]))
        .update({FaceSample.label: name}, synchronize_session=False)
    )
    db.commit()
    return {"updated": int(n), "label": name}


@router.post("/clips", response_model=EventListResponse)
def group_clips(
    sample_ids: list[int] = Body(..., embed=True),
    db: Session = Depends(get_db),
) -> EventListResponse:
    """Recorded clips for a face group: its samples -> sightings -> events.

    Newest-first, de-duplicated by event (many faces map to one clip). Returns
    only events that actually have a clip; an empty list when nothing links
    (vector-only/backfilled samples, or events deleted via SET NULL)."""
    from datetime import timedelta

    if not sample_ids:
        return EventListResponse(total=0, items=[])
    # FaceSample carries camera_id + ts directly. sightings.event_id is not
    # reliably set, so resolve clips by camera + time overlap: a clip-event
    # whose [ts-pre, end_ts+post] window contains one of these faces' captures.
    rows = db.execute(
        select(FaceSample.camera_id, FaceSample.ts, FaceSample.identity_id).where(
            FaceSample.id.in_([int(s) for s in sample_ids]),
        )
    ).all()
    by_cam: dict = {}
    identity_ids = set()
    for cam_id, ts, iid in rows:
        if cam_id is not None and ts is not None:
            by_cam.setdefault(cam_id, []).append(ts)
        if iid is not None:
            identity_ids.add(int(iid))
    # Also include the linked identities' sightings (camera+ts) so a face group
    # surfaces every clip its people appear in, not just the exact capture frame.
    if identity_ids:
        for cam_id, ts in db.execute(
            select(Sighting.camera_id, Sighting.ts).where(Sighting.identity_id.in_(identity_ids))
        ).all():
            if cam_id is not None and ts is not None:
                by_cam.setdefault(cam_id, []).append(ts)
    if not by_cam:
        return EventListResponse(total=0, items=[])
    PRE, POST = timedelta(seconds=90), timedelta(seconds=20)
    erows = db.execute(
        select(Event, Camera.name)
        .outerjoin(Camera, Camera.id == Event.camera_id)
        .where(Event.camera_id.in_(list(by_cam.keys())), Event.clip_path.is_not(None))
        .order_by(Event.ts.desc(), Event.id.desc())
    ).all()
    items, seen = [], set()
    for ev, cam_name in erows:
        if ev.id in seen:
            continue
        lo = ev.ts - PRE
        hi = (ev.end_ts or ev.ts) + POST
        if not any(lo <= t <= hi for t in by_cam.get(ev.camera_id, ())):
            continue
        seen.add(ev.id)
        items.append(EventListItem(
            id=ev.id, camera_id=ev.camera_id, camera_name=cam_name, ts=ev.ts,
            end_ts=ev.end_ts, label=ev.label, person_id=ev.person_id,
            person_name=ev.person_name, match_score=ev.match_score,
            identity_id=getattr(ev, "identity_id", None),
            identity_name=getattr(ev, "identity_name", None),
            thumb_url=f"/api/events/{ev.id}/thumbnail" if ev.thumb_path else None,
            clip_url=f"/api/events/{ev.id}/clip",
        ))
    return EventListResponse(total=len(items), items=items)


def _purge_sample_file(sample) -> None:
    abs_path = _resolve_data_path(get_settings(), sample.thumb_path)
    if abs_path and os.path.isfile(abs_path):
        try:
            os.remove(abs_path)
        except OSError:
            pass


@router.post("/samples/bulk-delete")
def bulk_delete_samples(
    ids: list[int] = Body(..., embed=True),
    db: Session = Depends(get_db),
) -> dict:
    if not ids:
        return {"deleted": 0}
    n = 0
    for fs in db.query(FaceSample).filter(FaceSample.id.in_([int(i) for i in ids])).all():
        _purge_sample_file(fs)
        db.delete(fs)
        n += 1
    db.commit()
    return {"deleted": n}


@router.post("/samples/clear-all")
def clear_all_samples(
    confirm: bool = Body(False, embed=True),
    label: Optional[str] = Body(None, embed=True),
    db: Session = Depends(get_db),
) -> dict:
    """Delete ALL captured face samples (optionally one label). Requires confirm."""
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true required")
    q = db.query(FaceSample)
    if label is not None:
        q = q.filter(FaceSample.label == label)
    n = 0
    while True:
        batch = q.limit(300).all()
        if not batch:
            break
        for fs in batch:
            _purge_sample_file(fs)
            db.delete(fs)
            n += 1
        db.commit()
        if len(batch) < 300:
            break
    return {"deleted": n}


@router.delete("/samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sample(sample_id: int, db: Session = Depends(get_db)) -> Response:
    sample = db.get(FaceSample, sample_id)
    if sample is not None:
        abs_path = _resolve_data_path(get_settings(), sample.thumb_path)
        if abs_path and os.path.isfile(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass
        db.delete(sample)
        db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

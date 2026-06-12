"""Identities API — the automatic, cross-camera person layer.

This router exposes the auto-discovered identity gallery built by the ReID
pipeline (``app/reid/*``). Unlike the manual ``/api/people`` face DB, identities
are created automatically on each sighting and corrected by operators after the
fact.

Routes (mounted under ``/api/identities``):

    GET    /api/identities                      list identities (rep thumb,
                                                 counts, first/last seen, cameras)
    GET    /api/identities/{id}                 one identity + recent sightings
    GET    /api/identities/{id}/sightings       paginated sightings (cross-cam/time)
    GET    /api/identities/{id}/thumbnail       representative body-crop JPEG
    GET    /api/identities/sightings/{sid}/thumbnail   one sighting's body crop
    PUT    /api/identities/{id}                 RENAME / annotate (sets is_named)
    POST   /api/identities/merge               MERGE source_ids -> target_id
    POST   /api/identities/{id}/split          SPLIT (explicit sighting_ids|auto)
    DELETE /api/identities/{id}                 delete identity (cascades sightings)

Merge / split delegate to ``app.reid.manager`` helpers (the reid-identity-core
component) when available; a conservative DB-level fallback keeps the operator
tools functional regardless of which sibling components have landed yet. Any
mutation triggers an identity-gallery reload so in-process workers re-sync.

All routes sit behind the trusted-header SSO dependency (see ``app.auth``). The
thumbnail endpoints additionally accept the Bearer ``API_KEY`` path so an
``<img>`` tag can fetch them through nginx with the SSO cookie.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ..auth import require_auth
from ..config import Settings, get_settings
from ..db.database import get_db
from ..db.models import Camera, Event
from ..reid import schemas

logger = logging.getLogger("vms.api.identities")

router = APIRouter(
    prefix="/api/identities",
    tags=["identities"],
    dependencies=[Depends(require_auth)],
)


# --------------------------------------------------------------------------- #
# ORM access (tolerant of the reid-identity-core models landing later)
# --------------------------------------------------------------------------- #
def _models():
    """Import the ReID ORM models lazily.

    The Identity/Sighting/FaceExemplar/AppearanceExemplar models are added to
    ``app.db.models`` by the integration component. Importing lazily (and
    failing soft with 503) keeps this router mountable before they land.
    """
    from ..db import models as m

    missing = [
        n
        for n in ("Identity", "Sighting", "FaceExemplar", "AppearanceExemplar")
        if not hasattr(m, n)
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Identity models not available: {', '.join(missing)}",
        )
    return m


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_identity_or_404(db: Session, identity_id: int):
    m = _models()
    identity = db.get(m.Identity, identity_id)
    if identity is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    return identity


def _identity_cameras(db: Session, identity_id: int) -> List[int]:
    m = _models()
    rows = db.execute(
        select(m.Sighting.camera_id)
        .where(m.Sighting.identity_id == identity_id)
        .distinct()
    ).all()
    return sorted({int(r[0]) for r in rows if r[0] is not None})


def _rep_thumb_url(identity) -> Optional[str]:
    """Representative thumbnail URL: served via the identity thumbnail route."""
    if getattr(identity, "rep_sighting_id", None):
        return f"/api/identities/{identity.id}/thumbnail"
    return None


def _best_face_url(db: Session, identity_id: int) -> Optional[str]:
    """URL of this identity's highest-quality face crop (face-first people gallery)."""
    try:
        from ..db.models import FaceSample
        fid = db.scalar(
            select(FaceSample.id).where(FaceSample.identity_id == identity_id)
            .order_by(FaceSample.quality.desc()).limit(1)
        )
        return f"/api/face-groups/samples/{int(fid)}/thumbnail" if fid else None
    except Exception:
        return None


def _identity_list_item(db: Session, identity) -> schemas.IdentityListItem:
    color = color_hex = make = vehicle_type = None
    attrs = getattr(identity, "attributes", None)
    if attrs:
        try:
            import json as _json
            a = _json.loads(attrs)
            color = a.get("color")
            color_hex = a.get("hex")
            make = a.get("make")
            vehicle_type = a.get("type")
        except Exception:
            pass
    return schemas.IdentityListItem(
        id=identity.id,
        name=identity.name,
        is_named=bool(getattr(identity, "is_named", False)),
        object_class=str(getattr(identity, "object_class", "person") or "person"),
        color=color,
        color_hex=color_hex,
        make=make,
        vehicle_type=vehicle_type,
        total_seconds=float(getattr(identity, "total_seconds", 0.0) or 0.0),
        num_sightings=int(getattr(identity, "num_sightings", 0) or 0),
        first_seen=getattr(identity, "first_seen", None),
        last_seen=getattr(identity, "last_seen", None),
        cameras=_identity_cameras(db, identity.id),
        rep_thumb_url=_rep_thumb_url(identity),
        face_thumb_url=_best_face_url(db, identity.id),
        created_at=getattr(identity, "created_at", None),
    )


def _sighting_item(sighting, camera_name: Optional[str], event: Optional[Event]) -> schemas.SightingItem:
    has_event = event is not None
    return schemas.SightingItem(
        id=sighting.id,
        identity_id=sighting.identity_id,
        camera_id=sighting.camera_id,
        camera_name=camera_name,
        event_id=getattr(sighting, "event_id", None),
        ts=sighting.ts,
        bbox=[
            int(getattr(sighting, "bbox_x1", 0) or 0),
            int(getattr(sighting, "bbox_y1", 0) or 0),
            int(getattr(sighting, "bbox_x2", 0) or 0),
            int(getattr(sighting, "bbox_y2", 0) or 0),
        ],
        det_score=getattr(sighting, "det_score", None),
        has_face=bool(getattr(sighting, "has_face", False)),
        face_score=getattr(sighting, "face_score", None),
        appearance_score=getattr(sighting, "appearance_score", None),
        match_kind=getattr(sighting, "match_kind", None),
        thumb_url=(
            f"/api/identities/sightings/{sighting.id}/thumbnail"
            if getattr(sighting, "thumb_path", None)
            else None
        ),
        event_thumb_url=(
            f"/api/events/{event.id}/thumbnail"
            if has_event and event.thumb_path
            else None
        ),
        event_clip_url=(
            f"/api/events/{event.id}/clip" if has_event and event.clip_path else None
        ),
    )


def _resolve_data_path(settings: Settings, stored: Optional[str]) -> Optional[str]:
    """Resolve a DB-stored (relative) thumbnail path to an on-disk absolute path,
    guarding against traversal outside the data root. Mirrors events._abs_data_path.
    """
    if not stored:
        return None
    data_root = os.path.abspath(str(settings.data_dir))
    if os.path.isabs(stored):
        candidate = os.path.abspath(stored)
    else:
        # Stored like ``data/identities/<id>/<sid>.jpg`` (relative to the app
        # working dir whose data dir basename is "data"); also accept paths
        # already relative to the data root.
        parent = os.path.dirname(data_root)
        candidate = os.path.abspath(os.path.join(parent, stored))
        if not os.path.exists(candidate):
            alt = os.path.abspath(os.path.join(data_root, stored))
            if os.path.exists(alt):
                candidate = alt
    if os.path.commonpath([candidate, data_root]) != data_root:
        raise HTTPException(status_code=400, detail="Invalid stored path")
    return candidate


def _reload_gallery(request: Request) -> None:
    """Trigger an identity-gallery reload after a mutation.

    The API process owns the authoritative gallery (``app.state.identity_gallery``);
    in-process workers re-sync on their own timer. We refresh the API-side gallery
    immediately so subsequent reads/assignments see the change.
    """
    gallery = getattr(request.app.state, "identity_gallery", None)
    if gallery is None:
        return
    db = None
    try:
        from ..db.database import SessionLocal

        db = SessionLocal()
        if hasattr(gallery, "rebuild_from_db"):
            gallery.rebuild_from_db(db)
        elif hasattr(gallery, "reload"):
            gallery.reload(db)
        db.commit()
    except Exception:  # noqa: BLE001 - gallery is derived state; never fatal
        logger.exception("Identity gallery reload failed (continuing)")
        if db is not None:
            db.rollback()
    finally:
        if db is not None:
            db.close()


# --------------------------------------------------------------------------- #
# Core merge/split delegation (with conservative DB-level fallback)
# --------------------------------------------------------------------------- #
def _core_manager(request: Request):
    """Return a reid-identity-core ``IdentityManager`` bound to the API-side
    gallery, or None when the component / gallery has not landed.

    The merge/split helpers live as IdentityManager methods (they recompute
    centroids and register split-off identities in the gallery), so we drive
    them through a manager bound to ``app.state.identity_gallery``.
    """
    gallery = getattr(request.app.state, "identity_gallery", None)
    if gallery is None:
        return None
    try:
        from ..reid.manager import IdentityManager
    except Exception:  # noqa: BLE001 - component may not have landed yet
        return None
    # Reuse a cached manager so its in-memory sticky/rate state is stable.
    mgr = getattr(request.app.state, "identity_manager", None)
    if mgr is not None and getattr(mgr, "gallery", None) is gallery:
        return mgr
    try:
        mgr = IdentityManager(gallery)
    except Exception:  # noqa: BLE001
        return None
    request.app.state.identity_manager = mgr
    return mgr


def _fallback_merge(db: Session, target_id: int, source_ids: List[int]) -> schemas.IdentityOpResult:
    """Reassign all sightings/exemplars of each source into the target, then
    delete the sources. Used only when the core ``merge_identities`` helper is
    absent. Recomputes the denormalized counters and first/last seen.
    """
    m = _models()
    target = db.get(m.Identity, target_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target identity not found")

    moved = 0
    affected = [target_id]
    for sid in source_ids:
        if sid == target_id:
            continue
        source = db.get(m.Identity, sid)
        if source is None:
            continue
        if bool(getattr(source, "is_named", False)) and not bool(
            getattr(target, "is_named", False)
        ):
            # Preserve the named identity's intent: don't silently lose a name.
            target.name = source.name
            target.is_named = True
        moved += db.execute(
            update(m.Sighting)
            .where(m.Sighting.identity_id == sid)
            .values(identity_id=target_id)
        ).rowcount or 0
        db.execute(
            update(m.FaceExemplar)
            .where(m.FaceExemplar.identity_id == sid)
            .values(identity_id=target_id)
        )
        db.execute(
            update(m.AppearanceExemplar)
            .where(m.AppearanceExemplar.identity_id == sid)
            .values(identity_id=target_id)
        )
        # Past Events denormalized onto the source identity follow it.
        if hasattr(Event, "identity_id"):
            values = {"identity_id": target_id}
            if hasattr(Event, "identity_name"):
                values["identity_name"] = target.name
            db.execute(
                update(Event).where(Event.identity_id == sid).values(**values)
            )
        db.delete(source)
        affected.append(sid)

    _recompute_identity_stats(db, m, target)
    db.commit()
    return schemas.IdentityOpResult(
        ok=True,
        target_id=target_id,
        affected_ids=affected,
        moved_sightings=moved,
        detail="fallback merge",
    )


def _fallback_split(
    db: Session,
    identity_id: int,
    sighting_ids: List[int],
    new_name: Optional[str],
) -> schemas.IdentityOpResult:
    """Move the named sightings (and any exemplars derived from them) into a
    freshly created identity. Used only when the core helper is absent.
    """
    m = _models()
    source = db.get(m.Identity, identity_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    if not sighting_ids:
        raise HTTPException(status_code=400, detail="No sightings to split off")

    # Validate the sightings belong to this identity.
    owned = db.scalars(
        select(m.Sighting.id).where(
            m.Sighting.identity_id == identity_id,
            m.Sighting.id.in_(sighting_ids),
        )
    ).all()
    owned_set = {int(s) for s in owned}
    bad = [s for s in sighting_ids if s not in owned_set]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Sightings not owned by identity {identity_id}: {bad}",
        )
    if len(owned_set) >= int(getattr(source, "num_sightings", 0) or 0):
        raise HTTPException(
            status_code=400,
            detail="Refusing to split off every sighting (use rename instead)",
        )

    new_identity = m.Identity(name=new_name or "Person new")
    db.add(new_identity)
    db.flush()  # assign new_identity.id
    if not new_name:
        new_identity.name = f"Person {new_identity.id}"

    moved = db.execute(
        update(m.Sighting)
        .where(m.Sighting.id.in_(owned_set))
        .values(identity_id=new_identity.id)
    ).rowcount or 0
    # Move exemplars whose source sighting moved.
    db.execute(
        update(m.FaceExemplar)
        .where(m.FaceExemplar.sighting_id.in_(owned_set))
        .values(identity_id=new_identity.id)
    )
    db.execute(
        update(m.AppearanceExemplar)
        .where(m.AppearanceExemplar.sighting_id.in_(owned_set))
        .values(identity_id=new_identity.id)
    )
    # If the source's representative sighting moved, clear it for recompute.
    if getattr(source, "rep_sighting_id", None) in owned_set:
        source.rep_sighting_id = None

    _recompute_identity_stats(db, m, source)
    _recompute_identity_stats(db, m, new_identity)
    db.commit()
    return schemas.IdentityOpResult(
        ok=True,
        target_id=identity_id,
        new_id=new_identity.id,
        affected_ids=[identity_id, new_identity.id],
        moved_sightings=moved,
        detail="fallback split",
    )


def _recompute_identity_stats(db: Session, m, identity) -> None:
    """Recompute the denormalized num_sightings / first_seen / last_seen / rep
    for an identity from its sightings. Centroid recompute is left to the core /
    maintenance pass (it owns the decay math); we only fix counters + rep here.
    """
    agg = db.execute(
        select(
            func.count(m.Sighting.id),
            func.min(m.Sighting.ts),
            func.max(m.Sighting.ts),
        ).where(m.Sighting.identity_id == identity.id)
    ).one()
    count, first_ts, last_ts = int(agg[0] or 0), agg[1], agg[2]
    identity.num_sightings = count
    if first_ts is not None:
        identity.first_seen = first_ts
    if last_ts is not None:
        identity.last_seen = last_ts
    if not getattr(identity, "rep_sighting_id", None) and count:
        # Pick the most recent sighting that has a stored thumbnail as the rep.
        rep = db.scalar(
            select(m.Sighting.id)
            .where(
                m.Sighting.identity_id == identity.id,
                m.Sighting.thumb_path.is_not(None),
            )
            .order_by(m.Sighting.ts.desc())
            .limit(1)
        )
        if rep is None:
            rep = db.scalar(
                select(m.Sighting.id)
                .where(m.Sighting.identity_id == identity.id)
                .order_by(m.Sighting.ts.desc())
                .limit(1)
            )
        identity.rep_sighting_id = rep


# --------------------------------------------------------------------------- #
# List / detail
# --------------------------------------------------------------------------- #
@router.get("", response_model=schemas.IdentityList)
def list_identities(
    named_only: bool = Query(False, description="Only operator-named identities"),
    min_sightings: int = Query(0, ge=0, description="Hide sparse/provisional clusters"),
    object_class: Optional[str] = Query(None, description="Filter by object class (person/car/dog/…)"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.IdentityList:
    """List auto-discovered identities, most-recently-seen first."""
    m = _models()

    filters = []
    if named_only:
        filters.append(m.Identity.is_named.is_(True))
    if min_sightings > 0:
        filters.append(m.Identity.num_sightings >= min_sightings)
    if object_class:
        filters.append(m.Identity.object_class == object_class)

    count_stmt = select(func.count()).select_from(m.Identity)
    for f in filters:
        count_stmt = count_stmt.where(f)
    total = int(db.execute(count_stmt).scalar_one())

    stmt = select(m.Identity)
    for f in filters:
        stmt = stmt.where(f)
    stmt = (
        stmt.order_by(m.Identity.last_seen.desc().nullslast(), m.Identity.id.desc())
        .limit(limit)
        .offset(offset)
    )
    identities = db.scalars(stmt).all()
    items = [_identity_list_item(db, ident) for ident in identities]
    return schemas.IdentityList(total=total, items=items)


@router.get("/analytics")
def identities_analytics(
    object_class: str = Query("person", description="Aggregate this object class"),
    db: Session = Depends(get_db),
):
    """Aggregates for the People analytics dashboard (read-only, indexed queries)."""
    import datetime as _dt
    m = _models()
    from ..db.models import Camera
    try:
        from ..db.models import PresenceSegment
    except Exception:
        PresenceSegment = None

    now = _dt.datetime.utcnow()
    today0 = _dt.datetime(now.year, now.month, now.day)
    day_start = now - _dt.timedelta(hours=24)
    week_start = now - _dt.timedelta(days=7)
    idf = m.Identity.object_class == object_class
    sf = m.Sighting.object_class == object_class

    def _c(stmt):
        return int(db.scalar(stmt) or 0)

    total = _c(select(func.count()).select_from(m.Identity).where(idf))
    named = _c(select(func.count()).select_from(m.Identity).where(idf, m.Identity.is_named.is_(True)))
    created_today = _c(select(func.count()).select_from(m.Identity).where(idf, m.Identity.created_at >= today0))
    total_sightings = _c(select(func.count()).select_from(m.Sighting).where(sf))
    seen_today = _c(select(func.count(func.distinct(m.Sighting.identity_id))).where(sf, m.Sighting.ts >= today0))
    seen_7d = _c(select(func.count(func.distinct(m.Sighting.identity_id))).where(sf, m.Sighting.ts >= week_start))

    cam_names = {c.id: c.name for c in db.scalars(select(Camera)).all()}
    dwell = {}
    if PresenceSegment is not None:
        try:
            for cid, sec in db.execute(
                select(PresenceSegment.camera_id, func.avg(PresenceSegment.seconds))
                .where(PresenceSegment.object_class == object_class)
                .group_by(PresenceSegment.camera_id)).all():
                dwell[cid] = float(sec or 0.0)
        except Exception:
            pass
    by_camera = []
    for cid, scount, icount in db.execute(
            select(m.Sighting.camera_id, func.count(), func.count(func.distinct(m.Sighting.identity_id)))
            .where(sf).group_by(m.Sighting.camera_id)).all():
        by_camera.append({"camera_id": cid, "camera_name": cam_names.get(cid, f"#{cid}"),
                          "sightings": int(scount), "people": int(icount),
                          "avg_dwell_s": round(dwell.get(cid, 0.0), 1)})
    by_camera.sort(key=lambda x: -x["sightings"])

    hours = [0] * 24
    for (ts,) in db.execute(select(m.Sighting.ts).where(sf, m.Sighting.ts >= day_start)).all():
        try:
            hours[ts.hour] += 1
        except Exception:
            pass
    by_hour = [{"hour": h, "sightings": hours[h]} for h in range(24)]

    by_match = {str(k): int(v) for k, v in db.execute(
        select(m.Sighting.match_kind, func.count()).where(sf)
        .group_by(m.Sighting.match_kind)).all() if k}

    return {
        "object_class": object_class,
        "summary": {"total_people": total, "named": named, "unnamed": total - named,
                    "created_today": created_today, "seen_today": seen_today,
                    "seen_7d": seen_7d, "total_sightings": total_sightings},
        "by_camera": by_camera, "by_hour": by_hour, "by_match_kind": by_match,
    }


@router.get("/{identity_id}", response_model=schemas.IdentityDetail)
def get_identity(
    identity_id: int,
    recent: int = Query(24, ge=0, le=200, description="Recent sightings to inline"),
    db: Session = Depends(get_db),
) -> schemas.IdentityDetail:
    """Fetch one identity with summary counts and its most-recent sightings."""
    m = _models()
    identity = _get_identity_or_404(db, identity_id)

    base = _identity_list_item(db, identity)

    num_face = int(
        db.scalar(
            select(func.count(m.FaceExemplar.id)).where(
                m.FaceExemplar.identity_id == identity_id
            )
        )
        or 0
    )
    num_app = int(
        db.scalar(
            select(func.count(m.AppearanceExemplar.id)).where(
                m.AppearanceExemplar.identity_id == identity_id
            )
        )
        or 0
    )

    recent_items: List[schemas.SightingItem] = []
    if recent:
        recent_items = _query_sightings(db, m, identity_id, limit=recent, offset=0)[1]

    return schemas.IdentityDetail(
        **base.model_dump(),
        notes=getattr(identity, "notes", None),
        num_face_exemplars=num_face,
        num_appearance_exemplars=num_app,
        recent_sightings=recent_items,
    )


def _query_sightings(
    db: Session, m, identity_id: int, *, limit: int, offset: int
) -> tuple[int, List[schemas.SightingItem]]:
    """Return (total, page-of-sightings) for an identity, newest first."""
    total = int(
        db.scalar(
            select(func.count(m.Sighting.id)).where(
                m.Sighting.identity_id == identity_id
            )
        )
        or 0
    )
    stmt = (
        select(m.Sighting, Camera.name)
        .outerjoin(Camera, Camera.id == m.Sighting.camera_id)
        .where(m.Sighting.identity_id == identity_id)
        .order_by(m.Sighting.ts.desc(), m.Sighting.id.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = db.execute(stmt).all()

    # Batch-load the linked events so we can expose clip/thumb links.
    event_ids = [
        getattr(s, "event_id", None) for s, _ in rows if getattr(s, "event_id", None)
    ]
    events: dict[int, Event] = {}
    if event_ids:
        for ev in db.scalars(
            select(Event).where(Event.id.in_(set(event_ids)))
        ).all():
            events[ev.id] = ev

    items = [
        _sighting_item(s, cam_name, events.get(getattr(s, "event_id", None)))
        for s, cam_name in rows
    ]
    return total, items


@router.get("/{identity_id}/sightings", response_model=schemas.SightingList)
def list_sightings(
    identity_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.SightingList:
    """Paginated, cross-camera/time list of an identity's sightings (newest first)."""
    m = _models()
    _get_identity_or_404(db, identity_id)
    total, items = _query_sightings(db, m, identity_id, limit=limit, offset=offset)
    return schemas.SightingList(total=total, items=items)


# --------------------------------------------------------------------------- #
# Thumbnails (also reachable via Bearer API key for <img> through nginx)
# --------------------------------------------------------------------------- #
def _events_overlapping_sightings(db, sighting_model, where_clause):
    """Return [(Event, camera_name)] whose clip window overlaps any sighting
    matched by ``where_clause`` (same camera + time), newest-first.

    Works around the unset Sighting.event_id FK by matching on (camera_id, ts).
    """
    from datetime import timedelta

    PRE = timedelta(seconds=90)   # clip pre-roll + sampling slack
    POST = timedelta(seconds=20)  # clip post-roll slack

    rows = db.execute(
        select(sighting_model.camera_id, sighting_model.ts).where(where_clause)
    ).all()
    if not rows:
        return []
    by_cam: dict = {}
    for cam_id, ts in rows:
        if cam_id is not None and ts is not None:
            by_cam.setdefault(cam_id, []).append(ts)
    if not by_cam:
        return []

    events = db.execute(
        select(Event, Camera.name)
        .outerjoin(Camera, Camera.id == Event.camera_id)
        .where(Event.camera_id.in_(list(by_cam.keys())), Event.clip_path.is_not(None))
        .order_by(Event.ts.desc(), Event.id.desc())
    ).all()
    matched = []
    for ev, cam_name in events:
        lo = ev.ts - PRE
        hi = (ev.end_ts or ev.ts) + POST
        if any(lo <= t <= hi for t in by_cam.get(ev.camera_id, ())):
            matched.append((ev, cam_name))
    return matched


@router.get("/{identity_id}/events")
def list_identity_events(
    identity_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Recorded clips for an identity: its sightings -> events with a clip.

    Newest-first, de-duplicated by event. Returns app.schemas EventListItem
    objects (same shape the events list/cards use) so the SPA can play them in
    the existing rich viewer."""
    from datetime import timedelta

    from ..api.events import _to_list_item  # app.schemas EventListItem builder

    m = _models()
    _get_identity_or_404(db, identity_id)

    # sightings.event_id is not reliably set (the tracker creates sightings
    # continuously, events fire on triggers), so resolve clips by camera + time
    # overlap: a clip-event whose [ts-pre, end_ts+post] window contains one of
    # this identity's sightings on the same camera.
    matched = _events_overlapping_sightings(
        db, m.Sighting, m.Sighting.identity_id == identity_id,
    )
    total = len(matched)
    page = matched[offset:offset + limit]
    items = [_to_list_item(ev, cam_name) for ev, cam_name in page]
    return {"total": total, "items": [i.model_dump() for i in items]}


@router.get("/{identity_id}/faces")
def list_identity_faces(identity_id: int, db: Session = Depends(get_db)):
    """Face samples captured for this identity (for the detail view)."""
    from ..db.models import FaceSample

    _get_identity_or_404(db, identity_id)
    rows = db.scalars(
        select(FaceSample).where(FaceSample.identity_id == identity_id)
        .order_by(FaceSample.quality.desc()).limit(60)
    ).all()
    return {
        "total": len(rows),
        "items": [
            {
                "id": int(f.id),
                "thumb_url": f"/api/face-groups/samples/{int(f.id)}/thumbnail",
                "camera_id": f.camera_id,
                "ts": (f.ts.isoformat() if f.ts is not None else None),
                "quality": round(float(f.quality or 0.0), 3),
            }
            for f in rows
        ],
    }


@router.get("/{identity_id}/thumbnail")
def get_identity_thumbnail(identity_id: int, db: Session = Depends(get_db)):
    """Serve the identity's representative body-crop JPEG."""
    m = _models()
    identity = _get_identity_or_404(db, identity_id)
    rep_id = getattr(identity, "rep_sighting_id", None)
    if not rep_id:
        raise HTTPException(status_code=404, detail="No representative thumbnail")
    sighting = db.get(m.Sighting, rep_id)
    if sighting is None or not getattr(sighting, "thumb_path", None):
        raise HTTPException(status_code=404, detail="Representative thumbnail missing")
    return _serve_thumb(sighting.thumb_path)


@router.get("/sightings/{sighting_id}/thumbnail")
def get_sighting_thumbnail(sighting_id: int, db: Session = Depends(get_db)):
    """Serve a single sighting's cropped body thumbnail."""
    m = _models()
    sighting = db.get(m.Sighting, sighting_id)
    if sighting is None or not getattr(sighting, "thumb_path", None):
        raise HTTPException(status_code=404, detail="Sighting thumbnail not found")
    return _serve_thumb(sighting.thumb_path)


def _serve_thumb(thumb_path: str):
    abs_path = _resolve_data_path(get_settings(), thumb_path)
    if not abs_path or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Thumbnail file missing on disk")
    return FileResponse(
        abs_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


# --------------------------------------------------------------------------- #
# Operator corrections: rename / merge / split / delete
# --------------------------------------------------------------------------- #
@router.put("/{identity_id}", response_model=schemas.IdentityDetail)
def rename_identity(
    identity_id: int,
    body: schemas.IdentityRename,
    request: Request,
    db: Session = Depends(get_db),
) -> schemas.IdentityDetail:
    """Rename / annotate an identity. Setting a name marks it ``is_named``
    (operator intent) so the maintenance auto-merge pass leaves it frozen.
    """
    identity = _get_identity_or_404(db, identity_id)
    data = body.model_dump(exclude_unset=True)

    if "name" in data and data["name"] is not None:
        identity.name = data["name"]
        # Naming implies confirmation unless explicitly overridden below.
        identity.is_named = True
    if "notes" in data:
        identity.notes = data["notes"]
    if "is_named" in data and data["is_named"] is not None:
        identity.is_named = bool(data["is_named"])

    db.commit()
    db.refresh(identity)
    _reload_gallery(request)
    return get_identity(identity_id, recent=24, db=db)


@router.post("/merge", response_model=schemas.IdentityOpResult)
def merge_identities(
    body: schemas.IdentityMergeRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> schemas.IdentityOpResult:
    """Merge ``source_ids`` into ``target_id`` (operator-confirmed)."""
    _models()  # 503 if identity models absent
    if body.target_id in body.source_ids:
        # Allow callers to be sloppy; just drop the no-op self-merge.
        sources = [s for s in body.source_ids if s != body.target_id]
        if not sources:
            raise HTTPException(
                status_code=400, detail="source_ids must differ from target_id"
            )
    else:
        sources = list(dict.fromkeys(body.source_ids))  # dedupe, keep order

    if db.get(_models().Identity, body.target_id) is None:
        raise HTTPException(status_code=404, detail="Target identity not found")

    mgr = _core_manager(request)
    if mgr is not None and hasattr(mgr, "merge"):
        try:
            moved = mgr.merge(db, target_id=body.target_id, source_ids=sources)
            db.commit()
            res = schemas.IdentityOpResult(
                ok=True,
                target_id=body.target_id,
                affected_ids=[body.target_id, *sources],
                moved_sightings=int(moved),
                detail="core merge",
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back rather than 500
            db.rollback()
            logger.warning("core merge failed (%s); using fallback", exc)
            res = _fallback_merge(db, body.target_id, sources)
    else:
        res = _fallback_merge(db, body.target_id, sources)

    _reload_gallery(request)
    return res


@router.post("/{identity_id}/split", response_model=schemas.IdentityOpResult)
def split_identity(
    identity_id: int,
    request: Request,
    body: schemas.IdentitySplitRequest = Body(default_factory=schemas.IdentitySplitRequest),
    db: Session = Depends(get_db),
) -> schemas.IdentityOpResult:
    """Split an identity: peel off explicit ``sighting_ids`` into a new identity,
    or ``auto`` re-cluster (face-priority) into two. Auto delegates to the core.
    """
    _get_identity_or_404(db, identity_id)

    explicit = list(dict.fromkeys(body.sighting_ids)) if body.sighting_ids else None
    if not explicit and not body.auto:
        raise HTTPException(
            status_code=400,
            detail="Provide sighting_ids to split off, or set auto=true",
        )

    mgr = _core_manager(request)
    if mgr is not None and hasattr(mgr, "split"):
        try:
            # Core's split: explicit sighting_ids, or None for auto re-cluster.
            new_id = mgr.split(db, identity_id=identity_id, sighting_ids=explicit)
            # No-op (refused / nothing to split) -> new_id == identity_id.
            if int(new_id) == int(identity_id):
                db.rollback()
                if body.auto:
                    res = schemas.IdentityOpResult(
                        ok=False,
                        target_id=identity_id,
                        affected_ids=[identity_id],
                        moved_sightings=0,
                        detail="auto-split found no separable cluster",
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Split would empty the identity or move no sightings",
                    )
            else:
                if body.new_name:
                    new_obj = db.get(_models().Identity, int(new_id))
                    if new_obj is not None:
                        new_obj.name = body.new_name
                moved = int(
                    db.scalar(
                        select(func.count(_models().Sighting.id)).where(
                            _models().Sighting.identity_id == int(new_id)
                        )
                    )
                    or 0
                )
                db.commit()
                res = schemas.IdentityOpResult(
                    ok=True,
                    target_id=identity_id,
                    new_id=int(new_id),
                    affected_ids=[identity_id, int(new_id)],
                    moved_sightings=moved,
                    detail="core split",
                )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            if explicit is None:
                raise HTTPException(status_code=500, detail=f"Auto-split failed: {exc}")
            logger.warning("core split failed (%s); using fallback", exc)
            res = _fallback_split(db, identity_id, explicit, body.new_name)
    else:
        if explicit is None:
            raise HTTPException(
                status_code=501,
                detail="Auto-split requires the reid-identity-core component",
            )
        res = _fallback_split(db, identity_id, explicit, body.new_name)

    _reload_gallery(request)
    return res


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _purge_identity_files(settings: Settings, db: Session, identity_id: int) -> None:
    """Remove on-disk artifacts OWNED by an identity so a delete leaves nothing
    behind: (1) the face-sample crops linked to it + their rows (FaceSample has
    no DB-level FK cascade), and (2) the whole per-identity crop directory where
    every sighting body-thumbnail lives (``data/identities/<id>/``)."""
    m = _models()
    fs_model = getattr(m, "FaceSample", None)
    if fs_model is not None:
        try:
            for s in db.query(fs_model).filter(fs_model.identity_id == identity_id).all():
                if getattr(s, "thumb_path", None):
                    _safe_remove(_resolve_data_path(settings, s.thumb_path))
                db.delete(s)
        except Exception:
            logger.exception("face-sample purge failed for identity %s", identity_id)
    try:
        d = os.path.join(os.path.abspath(str(settings.data_dir)), "identities", str(identity_id))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        logger.exception("identity crop-dir purge failed for identity %s", identity_id)


@router.delete("/{identity_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_identity(
    identity_id: int,
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    """Delete an identity recursively: cascades sightings + exemplars + presence
    in the DB, purges its on-disk crops (sighting thumbnails + face samples),
    detaches it from any Events (denormalized link nulled), reloads the gallery.
    """
    m = _models()
    identity = _get_identity_or_404(db, identity_id)

    # Detach from events explicitly so the denormalized link is cleared even
    # though the column has no DB-level FK cascade (per the migration shim).
    if hasattr(Event, "identity_id"):
        values = {"identity_id": None}
        if hasattr(Event, "identity_name"):
            values["identity_name"] = None
        db.execute(
            update(Event).where(Event.identity_id == identity_id).values(**values)
        )

    _purge_identity_files(get_settings(), db, identity_id)
    db.delete(identity)  # cascades sightings + face/appearance exemplars + presence
    db.commit()

    _reload_gallery(request)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _detach_identity_events(db, ids: list[int]) -> None:
    if hasattr(Event, "identity_id"):
        values = {"identity_id": None}
        if hasattr(Event, "identity_name"):
            values["identity_name"] = None
        db.execute(update(Event).where(Event.identity_id.in_(ids)).values(**values))


@router.post("/bulk-delete")
def bulk_delete_identities(
    ids: list[int] = Body(..., embed=True, max_length=1000),
    request: Request = None,
    db: Session = Depends(get_db),
) -> dict:
    """Delete the given identities (cascades sightings/exemplars)."""
    m = _models()
    if not ids:
        return {"deleted": 0}
    ids = [int(i) for i in ids]
    _detach_identity_events(db, ids)
    settings = get_settings()
    n = 0
    for ident in db.query(m.Identity).filter(m.Identity.id.in_(ids)).all():
        _purge_identity_files(settings, db, int(ident.id))
        db.delete(ident)
        n += 1
    db.commit()
    _reload_gallery(request)
    return {"deleted": n}


@router.post("/clear-all")
def clear_all_identities(
    confirm: bool = Body(False, embed=True),
    object_class: Optional[str] = Body(None, embed=True),
    request: Request = None,
    db: Session = Depends(get_db),
) -> dict:
    """Delete ALL identities (optionally one object_class). Requires confirm."""
    m = _models()
    if not confirm:
        raise HTTPException(status_code=400, detail="confirm=true required")
    q = db.query(m.Identity)
    if object_class:
        q = q.filter(m.Identity.object_class == object_class)
    settings = get_settings()
    n = 0
    while True:
        batch = q.limit(200).all()
        if not batch:
            break
        _detach_identity_events(db, [int(i.id) for i in batch])
        for ident in batch:
            _purge_identity_files(settings, db, int(ident.id))
            db.delete(ident)
            n += 1
        db.commit()
        if len(batch) < 200:
            break
    _reload_gallery(request)
    return {"deleted": n}

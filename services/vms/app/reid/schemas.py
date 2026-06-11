"""Pydantic v2 schemas for the auto-discovered Identities API.

Kept in a dedicated module (rather than ``app/schemas.py``) so the ReID layer
owns its own contracts and parallel work never collides on the shared schema
file. Mirrors the conventions of ``app/schemas.py`` (``from_attributes`` where
the shape matches an ORM row, denormalized URL helpers for the frontend, raw
embedding blobs never serialized to the client).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Identities (auto-discovered person clusters)
# ---------------------------------------------------------------------------
class IdentityListItem(BaseModel):
    """Item shape inside GET /api/identities.

    ``rep_thumb_url`` points at the representative sighting's body crop;
    ``cameras`` is the set of camera ids the identity has been seen on.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_named: bool = False
    object_class: str = "person"
    color: Optional[str] = None
    color_hex: Optional[str] = None
    make: Optional[str] = None          # vehicle brand (TAO VehicleMakeNet)
    vehicle_type: Optional[str] = None  # body type (TAO VehicleTypeNet)
    total_seconds: float = 0.0
    num_sightings: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    cameras: List[int] = Field(default_factory=list)
    rep_thumb_url: Optional[str] = None
    created_at: Optional[datetime] = None


class IdentityList(BaseModel):
    """Response for GET /api/identities."""

    total: int
    items: List[IdentityListItem]


class SightingItem(BaseModel):
    """One identified person-detection, with thumbnail + source-event links."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    identity_id: int
    camera_id: int
    camera_name: Optional[str] = None
    event_id: Optional[int] = None
    ts: datetime
    bbox: List[int] = Field(default_factory=list, description="[x1, y1, x2, y2]")
    det_score: Optional[float] = None
    has_face: bool = False
    face_score: Optional[float] = None
    appearance_score: Optional[float] = None
    match_kind: Optional[str] = None  # 'face' | 'appearance' | 'new'
    thumb_url: Optional[str] = None
    # Link back to the triggered clip when the sighting produced an Event.
    event_thumb_url: Optional[str] = None
    event_clip_url: Optional[str] = None


class SightingList(BaseModel):
    """Paginated list of sightings (GET /api/identities/{id}/sightings)."""

    total: int
    items: List[SightingItem]


class IdentityDetail(IdentityListItem):
    """Full identity resource (GET /api/identities/{id})."""

    notes: Optional[str] = None
    num_face_exemplars: int = 0
    num_appearance_exemplars: int = 0
    # First page of the identity's sightings for convenience; the paginated
    # endpoint serves the rest.
    recent_sightings: List[SightingItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Operator correction tools
# ---------------------------------------------------------------------------
class IdentityRename(BaseModel):
    """Body for PUT /api/identities/{id} — rename / annotate / confirm."""

    name: Optional[str] = Field(None, min_length=1)
    notes: Optional[str] = None
    # Renaming defaults is_named=True (operator intent freezes auto-merge); a
    # client may toggle it explicitly.
    is_named: Optional[bool] = None


class IdentityMergeRequest(BaseModel):
    """Body for POST /api/identities/merge.

    All ``source_ids`` are folded into ``target_id``: their sightings and
    exemplars are reassigned, then the sources are deleted. Face-priority
    centroid recompute is handled by the core helper.
    """

    target_id: int = Field(..., description="Surviving identity id")
    source_ids: List[int] = Field(
        ..., min_length=1, description="Identities to merge into the target"
    )


class IdentitySplitRequest(BaseModel):
    """Body for POST /api/identities/{id}/split.

    Either pass an explicit ``sighting_ids`` list to peel off into a new
    identity, or set ``auto=True`` to let the core re-cluster the identity's
    sightings (by face when available, else appearance) into two identities.
    """

    sighting_ids: Optional[List[int]] = Field(
        None, description="Sightings to move to a new identity"
    )
    auto: bool = Field(
        False, description="Auto re-cluster this identity into two"
    )
    new_name: Optional[str] = Field(
        None, min_length=1, description="Optional name for the split-off identity"
    )


class IdentityOpResult(BaseModel):
    """Generic result of a merge/split operation."""

    ok: bool = True
    target_id: Optional[int] = None
    new_id: Optional[int] = None
    affected_ids: List[int] = Field(default_factory=list)
    moved_sightings: int = 0
    detail: Optional[str] = None

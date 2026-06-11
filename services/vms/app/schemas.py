"""Pydantic v2 request/response schemas shared across all VMS routers.

These mirror the API contracts in the architecture spec. ORM objects are
converted via ``model_config = ConfigDict(from_attributes=True)`` so routers
can return SQLAlchemy instances directly where the shape matches.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def _normalize_trigger_classes(v):
    """Accept a list[str] or a comma-separated string, store as CSV (or None)."""
    if v is None:
        return None
    if isinstance(v, str):
        items = [p.strip() for p in v.split(",")]
    else:
        items = [str(p).strip() for p in v]
    items = [p for p in items if p]
    # de-dup preserving order
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return ",".join(out) if out else None


class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, description="Human-friendly camera name")
    rtsp_url: str = Field(..., min_length=1, description="RTSP source URL")
    enabled: bool = True
    detect_conf: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Detection confidence threshold override"
    )
    pre_seconds: Optional[int] = Field(
        None, ge=0, le=120, description="Pre-roll buffer seconds override"
    )
    post_seconds: Optional[int] = Field(
        None, ge=0, le=300, description="Post-roll buffer seconds override"
    )
    trigger_classes: Optional[str] = Field(
        None,
        description="Comma-separated COCO classes that trigger recording "
        "(e.g. 'person,car,dog'). Empty/None -> 'person'.",
    )
    detect_iou: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="NMS IoU threshold override"
    )
    detect_imgsz: Optional[int] = Field(
        None, ge=128, le=1920, description="Detector input size override (px)"
    )
    detect_interval: Optional[float] = Field(
        None, ge=0.0, le=60.0, description="Seconds between detections (0 = every frame)"
    )
    trigger_cooldown: Optional[float] = Field(
        None, ge=0.0, le=3600.0, description="Min seconds between event triggers"
    )
    min_trigger_frames: Optional[int] = Field(
        None, ge=1, le=100, description="Consecutive detection frames before a trigger"
    )
    rtsp_transport: Optional[str] = Field(
        None, description="RTSP transport: 'tcp' or 'udp'"
    )
    faces_enabled: Optional[bool] = Field(
        None, description="Run face recognition on this camera"
    )
    reid_enabled: Optional[bool] = Field(
        None, description="Run cross-camera re-identification on this camera"
    )

    @field_validator("trigger_classes", mode="before")
    @classmethod
    def _csv_classes(cls, v):
        return _normalize_trigger_classes(v)

    @field_validator("rtsp_transport")
    @classmethod
    def _check_transport(cls, v):
        if v is not None and v not in ("tcp", "udp"):
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
        return v


class CameraCreate(CameraBase):
    """Body for POST /api/cameras."""


class CameraUpdate(BaseModel):
    """Body for PUT /api/cameras/{id} — all fields optional (partial update)."""

    name: Optional[str] = Field(None, min_length=1)
    rtsp_url: Optional[str] = Field(None, min_length=1)
    enabled: Optional[bool] = None
    detect_conf: Optional[float] = Field(None, ge=0.0, le=1.0)
    pre_seconds: Optional[int] = Field(None, ge=0, le=120)
    post_seconds: Optional[int] = Field(None, ge=0, le=300)
    trigger_classes: Optional[str] = None
    detect_iou: Optional[float] = Field(None, ge=0.0, le=1.0)
    detect_imgsz: Optional[int] = Field(None, ge=128, le=1920)
    detect_interval: Optional[float] = Field(None, ge=0.0, le=60.0)
    trigger_cooldown: Optional[float] = Field(None, ge=0.0, le=3600.0)
    min_trigger_frames: Optional[int] = Field(None, ge=1, le=100)
    rtsp_transport: Optional[str] = None
    faces_enabled: Optional[bool] = None
    reid_enabled: Optional[bool] = None

    @field_validator("trigger_classes", mode="before")
    @classmethod
    def _csv_classes(cls, v):
        return _normalize_trigger_classes(v)

    @field_validator("rtsp_transport")
    @classmethod
    def _check_transport(cls, v):
        if v is not None and v not in ("tcp", "udp"):
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
        return v


class Camera(CameraBase):
    """Full camera resource returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str  # 'online' | 'offline' | 'error'
    last_seen: Optional[datetime] = None
    created_at: datetime


class CameraStatus(BaseModel):
    """Response for GET /api/cameras/{id}/status."""

    status: str
    last_seen: Optional[datetime] = None
    fps: Optional[float] = None
    detector: Optional[str] = None


# ---------------------------------------------------------------------------
# Events / history
# ---------------------------------------------------------------------------
class EventBase(BaseModel):
    camera_id: int
    ts: datetime
    end_ts: Optional[datetime] = None
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    match_score: Optional[float] = None
    label: str = "person"
    # Detection metadata (track-driven recording).
    num_objects: Optional[int] = None
    object_classes: Optional[str] = None
    peak_confidence: Optional[float] = None
    num_frames: Optional[int] = None
    duration_seconds: Optional[float] = None  # derived (end_ts - ts), not stored


class Event(EventBase):
    """Full event resource (GET /api/events/{id})."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    clip_path: Optional[str] = None
    thumb_path: Optional[str] = None
    # Denormalized helpers for the SPA detail viewer (previously stripped by
    # Pydantic extra-ignore, which broke the detail clip player).
    camera_name: Optional[str] = None
    thumb_url: Optional[str] = None
    clip_url: Optional[str] = None
    identity_id: Optional[int] = None
    identity_name: Optional[str] = None
    created_at: datetime


class EventListItem(BaseModel):
    """Item shape inside GET /api/events list response.

    Adds denormalized ``camera_name`` and URL helpers for the frontend
    instead of raw on-disk paths.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: int
    camera_name: Optional[str] = None
    ts: datetime
    end_ts: Optional[datetime] = None
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    match_score: Optional[float] = None
    label: str = "person"
    identity_id: Optional[int] = None
    identity_name: Optional[str] = None
    num_objects: Optional[int] = None
    object_classes: Optional[str] = None
    peak_confidence: Optional[float] = None
    num_frames: Optional[int] = None
    duration_seconds: Optional[float] = None
    thumb_url: Optional[str] = None
    clip_url: Optional[str] = None


class EventList(BaseModel):
    """Response for GET /api/events."""

    total: int
    items: List[EventListItem]


# ---------------------------------------------------------------------------
# People / face DB
# ---------------------------------------------------------------------------
class PersonBase(BaseModel):
    name: str = Field(..., min_length=1)
    notes: Optional[str] = None


class PersonCreate(PersonBase):
    """Body for POST /api/people."""


class PersonUpdate(BaseModel):
    """Body for PUT /api/people/{id} — partial update."""

    name: Optional[str] = Field(None, min_length=1)
    notes: Optional[str] = None


class Person(PersonBase):
    """Full person resource. ``num_faces`` is computed by the router."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    num_faces: int = 0


class FaceEmbedding(BaseModel):
    """Face row returned by GET /api/people/{id}/faces.

    ``image_url`` is a frontend-facing URL derived from the stored
    ``image_path``; the raw vector blob is never serialized to the client.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    person_id: int
    image_url: Optional[str] = None
    created_at: datetime


class FaceEnrollResult(BaseModel):
    """Response for POST /api/people/{id}/faces."""

    embedding_id: Optional[int] = None
    faces_detected: int
    image_path: Optional[str] = None


# ---------------------------------------------------------------------------
# System / health
# ---------------------------------------------------------------------------
class WorkerState(BaseModel):
    camera_id: int
    state: str  # e.g. 'running' | 'starting' | 'stopped' | 'error'
    fps: Optional[float] = None
    last_seen: Optional[datetime] = None


class GpuInfo(BaseModel):
    used_mb: Optional[float] = None
    total_mb: Optional[float] = None


class HealthResponse(BaseModel):
    """Response for GET /health (no auth)."""

    status: str = "ok"
    version: str
    gpu: GpuInfo = Field(default_factory=GpuInfo)
    workers: List[WorkerState] = Field(default_factory=list)


class SystemInfo(BaseModel):
    """Response for GET /api/system — detailed worker + model + backend info."""

    status: str = "ok"
    version: str
    detector_backend: str
    device: str
    models: dict = Field(default_factory=dict)
    gpu: GpuInfo = Field(default_factory=GpuInfo)
    workers: List[WorkerState] = Field(default_factory=list)


# --- Cross-component name aliases (router contracts) ----------------------
# The camera/event routers were written against these names; map them to the
# existing response models so all routers import cleanly.
CameraOut = Camera
CameraStatusOut = CameraStatus
EventDetail = Event
EventListResponse = EventList

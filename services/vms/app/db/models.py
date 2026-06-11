"""SQLAlchemy 2.x ORM models for the VMS metadata DB (SQLite, WAL).

Core tables: Camera, Event, Person, FaceEmbedding.

Automatic cross-camera identification adds four more: Identity, Sighting,
FaceExemplar, AppearanceExemplar (auto-discovered people, their per-detection
sightings, and the per-identity face/appearance exemplars the gallery matches
against).

Design notes / single-source-of-truth:
  * The DB is the authoritative store for everything, including face
    embeddings. The FAISS index is *derived* state, rebuilt at startup by
    streaming ``FaceEmbedding.vector`` blobs into an ``IndexFlatIP``.
  * ``FaceEmbedding.vector`` holds 512 little-endian float32 values
    (np.float32 array, L2-normalized) serialized via ``ndarray.tobytes()``.
    The identity exemplar vectors follow the exact same convention.
  * Cascades match the architecture contract:
      - delete a Camera  -> its Events cascade-delete
      - delete a Person  -> its FaceEmbeddings cascade-delete, and any
        Event.person_id referencing it is SET NULL (event history kept).
      - delete an Identity -> its Sightings + exemplars cascade-delete; any
        Event.identity_id referencing it is nulled in application code
        (the column is a plain INTEGER on SQLite, see ``ensure_reid_schema``).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Declarative base for all VMS ORM models."""


class Camera(Base):
    """An RTSP camera managed by the VMS.

    ``status`` / ``last_seen`` are updated by the camera worker heartbeat.
    The per-camera tunables are nullable; when NULL the worker falls back to
    the global env defaults from Settings.
    """

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    rtsp_url: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # 'online' | 'offline' | 'error'
    status: Mapped[str] = mapped_column(String, nullable=False, default="offline")
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Per-camera tunables (nullable -> fall back to global env defaults).
    detect_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    pre_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    post_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Comma-separated COCO class names that trigger recording on this camera.
    # NULL -> the worker default ("person"). e.g. "person,car,dog".
    trigger_classes: Mapped[str | None] = mapped_column(String, nullable=True)
    # Additional per-camera detection/trigger overrides (NULL -> global default).
    detect_iou: Mapped[float | None] = mapped_column(Float, nullable=True)
    detect_imgsz: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detect_interval: Mapped[float | None] = mapped_column(Float, nullable=True)
    trigger_cooldown: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_trigger_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rtsp_transport: Mapped[str | None] = mapped_column(String, nullable=True)
    faces_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reid_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    events: Mapped[list["Event"]] = relationship(
        back_populates="camera",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Camera id={self.id} name={self.name!r} status={self.status}>"


class Person(Base):
    """A known person enrolled in the face DB."""

    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    faces: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    # Events reference person via SET NULL on delete; expose the read side.
    events: Mapped[list["Event"]] = relationship(
        back_populates="person",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Person id={self.id} name={self.name!r}>"


class FaceEmbedding(Base):
    """A single 512-d ArcFace embedding for a person.

    ``vector`` is 512 float32 little-endian, L2-normalized (np.tobytes()).
    ``faiss_id`` is the position id in the in-memory FAISS index, used for
    remove/rebuild mapping. It is derived state and may be re-assigned on a
    full index rebuild.
    """

    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    faiss_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    person: Mapped["Person"] = relationship(back_populates="faces")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<FaceEmbedding id={self.id} person_id={self.person_id}>"


class Event(Base):
    """A person-appeared event with its recorded clip + thumbnail + match.

    ``ts`` is the event (person-appeared) time; ``end_ts`` is when the clip
    recording finished. ``person_id`` / ``person_name`` / ``match_score``
    capture the best face match (if any); ``person_name`` is a denormalized
    snapshot so history stays meaningful even if the Person is renamed or
    deleted later. ``clip_path`` / ``thumb_path`` are paths relative to the
    project root (e.g. ``data/recordings/<camera_id>/<event_id>.mp4``).
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(String, nullable=True)
    thumb_path: Mapped[str | None] = mapped_column(String, nullable=True)
    person_id: Mapped[int | None] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    person_name: Mapped[str | None] = mapped_column(String, nullable=True)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    label: Mapped[str] = mapped_column(String, nullable=False, default="person")
    # Auto-discovered identity attached to this event's dominant person (the
    # automatic cross-camera layer, separate from the manual ``person_id``).
    # On SQLite this is materialised as a plain INTEGER column by
    # ``database.ensure_reid_schema`` (create_all can't ALTER an existing
    # table); the FK/relationship is declared here at the ORM level only.
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("identities.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    identity_name: Mapped[str | None] = mapped_column(String, nullable=True)
    identity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Detection metadata (track-driven recording). Materialised by
    # ``database.ensure_event_track_schema`` on existing DBs.
    num_objects: Mapped[int | None] = mapped_column(Integer, nullable=True)
    object_classes: Mapped[str | None] = mapped_column(String, nullable=True)
    peak_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clip_start_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    camera: Mapped["Camera"] = relationship(back_populates="events")
    person: Mapped["Person | None"] = relationship(back_populates="events")
    identity: Mapped["Identity | None"] = relationship(
        back_populates="events", foreign_keys=[identity_id]
    )

    __table_args__ = (
        # Composite index for history filtering (camera + time range).
        Index("ix_events_camera_id_ts", "camera_id", "ts"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Event id={self.id} camera_id={self.camera_id} ts={self.ts}>"


# ---------------------------------------------------------------------------
# Automatic cross-camera identification
# ---------------------------------------------------------------------------


class Identity(Base):
    """An auto-discovered person, built online from sightings (no enrollment).

    ``face_centroid`` / ``appearance_centroid`` are *derived* caches: the
    running (face) / time-decayed (appearance) means of the per-identity
    exemplars, L2-normalized, serialized as 512 little-endian float32 like
    ``FaceEmbedding.vector``. They are recomputed by the matcher on update and
    by maintenance; the exemplar rows remain the source of truth.

    ``is_named`` flips to True once an operator renames/confirms the identity;
    named identities are frozen against automatic merges.
    """

    __tablename__ = "identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, default="")
    is_named: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # The COCO object class this identity represents ("person", "car", "dog"…).
    # Matching is scoped to a single class — a car never merges with a dog.
    object_class: Mapped[str] = mapped_column(String, nullable=False, default="person", index=True)
    # Total seconds this object has been present in front of any camera,
    # accumulated from PresenceSegment rows (the "how long was it here" total).
    total_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Visual attributes for unique-object identification, JSON:
    # {"color": "red", "hex": "#c81e1e", "hist": [..12..]}. The hue histogram
    # also gates appearance matching (a red car never merges with a blue one).
    attributes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Provisional identities (single low-evidence faceless sighting) are pruned
    # by maintenance if they never accrue a second sighting or a face.
    is_provisional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    rep_sighting_id: Mapped[int | None] = mapped_column(
        # use_alter breaks the identities<->sightings FK cycle at create time
        # (the constraint is added after both tables exist).
        ForeignKey(
            "sightings.id", ondelete="SET NULL", use_alter=True,
            name="fk_identities_rep_sighting",
        ),
        nullable=True,
    )
    face_centroid: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    appearance_centroid: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True
    )
    num_sightings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    sightings: Mapped[list["Sighting"]] = relationship(
        back_populates="identity",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Sighting.identity_id",
    )
    face_exemplars: Mapped[list["FaceExemplar"]] = relationship(
        back_populates="identity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    appearance_exemplars: Mapped[list["AppearanceExemplar"]] = relationship(
        back_populates="identity",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    # Read side of the denormalized Event.identity_id link.
    events: Mapped[list["Event"]] = relationship(
        back_populates="identity",
        foreign_keys="Event.identity_id",
        passive_deletes=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Identity id={self.id} name={self.name!r} n={self.num_sightings}>"


class Sighting(Base):
    """One identified person-detection on a trigger frame.

    Persisted for every assigned crop. ``thumb_path`` is the cropped body
    thumbnail (``data/identities/<identity_id>/<sighting_id>.jpg``).
    ``match_kind`` records how the assignment was made: ``'face'`` (linked by
    ArcFace), ``'appearance'`` (linked by OSNet within the time window), or
    ``'new'`` (seeded a brand-new identity).
    """

    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int] = mapped_column(
        ForeignKey("identities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_id: Mapped[int | None] = mapped_column(
        ForeignKey("events.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    object_class: Mapped[str] = mapped_column(String, nullable=False, default="person")
    bbox_x1: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bbox_y1: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bbox_x2: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bbox_y2: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    det_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    has_face: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    face_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    appearance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_kind: Mapped[str] = mapped_column(String, nullable=False, default="new")
    thumb_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    identity: Mapped["Identity"] = relationship(
        back_populates="sightings", foreign_keys=[identity_id]
    )
    camera: Mapped["Camera"] = relationship()

    __table_args__ = (
        Index("ix_sightings_identity_ts", "identity_id", "ts"),
        Index("ix_sightings_camera_ts", "camera_id", "ts"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Sighting id={self.id} identity_id={self.identity_id} "
            f"camera_id={self.camera_id} kind={self.match_kind}>"
        )


class FaceExemplar(Base):
    """A representative ArcFace vector kept for an identity (cap ~8).

    ``vector`` is 512 float32 little-endian, L2-normalized — identical layout
    to ``FaceEmbedding.vector``. Face exemplars are NOT time-decayed (faces are
    time-stable); they are quality-pruned when the cap is exceeded.
    """

    __tablename__ = "face_exemplars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int] = mapped_column(
        ForeignKey("identities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    det_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sighting_id: Mapped[int | None] = mapped_column(
        ForeignKey("sightings.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    identity: Mapped["Identity"] = relationship(back_populates="face_exemplars")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<FaceExemplar id={self.id} identity_id={self.identity_id}>"


class AppearanceExemplar(Base):
    """A per-identity OSNet appearance vector with a capture timestamp (cap ~16).

    ``vector`` is 512 float32 little-endian, L2-normalized. ``ts`` is the
    capture time used by the time-decay weighting (people change clothes
    between days), so exemplars are pruned/decayed by recency + quality.
    """

    __tablename__ = "appearance_exemplars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int] = mapped_column(
        ForeignKey("identities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sighting_id: Mapped[int | None] = mapped_column(
        ForeignKey("sightings.id", ondelete="SET NULL"),
        nullable=True,
    )
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    identity: Mapped["Identity"] = relationship(
        back_populates="appearance_exemplars"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<AppearanceExemplar id={self.id} identity_id={self.identity_id}>"


class FaceSample(Base):
    """A single captured face crop + its ArcFace embedding for face grouping.

    Independent of the body Re-ID identities: the dedicated face-recognition
    layer captures one of these per person-track-with-a-face (throttled), stores
    the aligned-ish face thumbnail, the 512-d ArcFace ``vector`` (L2-normed) and
    — for the optional clothing signal — the same sighting's OSNet appearance
    ``app_vector``. Unsupervised grouping clusters these by cosine (tunable),
    optionally fused with appearance. ``label`` is set when an operator names a
    group; ``identity_id`` links back to the auto Re-ID identity when known.
    """

    __tablename__ = "face_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)        # ArcFace 512
    app_vector: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # OSNet 512 (clothing)
    quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    identity_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    sighting_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    thumb_path: Mapped[str | None] = mapped_column(String, nullable=True)
    label: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<FaceSample id={self.id} cam={self.camera_id} label={self.label!r}>"


class PresenceSegment(Base):
    """One continuous appearance of an identity in front of one camera.

    A camera worker tracks each object frame-to-frame; when the object leaves
    (not seen for the track-gap window) it closes the segment with the elapsed
    ``seconds``. Summing an identity's segments gives the total time it spent in
    view (``Identity.total_seconds``). This is the audit trail behind the dwell
    total and supports per-camera / per-day breakdowns.
    """

    __tablename__ = "presence_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identity_id: Mapped[int] = mapped_column(
        ForeignKey("identities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    object_class: Mapped[str] = mapped_column(String, nullable=False, default="person")
    enter_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    exit_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<PresenceSegment id={self.id} identity_id={self.identity_id} {self.seconds:.1f}s>"

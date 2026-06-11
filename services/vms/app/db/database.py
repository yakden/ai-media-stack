"""SQLAlchemy 2.x engine, session factory, and DB initialisation.

SQLite in WAL mode — zero-ops, single-box, plenty for the MVP. This module owns
the declarative ``Base`` (so ``init_db`` can ``create_all`` and so models can
inherit from it) plus the ``get_db`` FastAPI dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ..config import Settings, get_settings

logger = logging.getLogger("vms.db")


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models (see ``app.db.models``)."""


# Built lazily so that get_settings() (and thus data_dir) is resolved first and
# so tests can point at a temp DB before the engine is created.
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _enable_sqlite_pragmas(dbapi_conn, _conn_record) -> None:
    """Per-connection pragmas: WAL journal + sane durability + FK enforcement."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


def get_engine(settings: Settings | None = None) -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        settings = settings or get_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            settings.database_url,
            # SQLite + multithreaded uvicorn / multiprocessing workers.
            connect_args={"check_same_thread": False},
            future=True,
        )
        event.listen(_engine, "connect", _enable_sqlite_pragmas)
        _SessionLocal = sessionmaker(
            bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False,
            class_=Session, future=True,
        )
        logger.info("DB engine created: %s", settings.database_url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def SessionLocal() -> Session:
    """Create a new Session. Usable by worker subprocesses (own engine/connection)."""
    return get_session_factory()()


def get_db() -> Iterator[Session]:
    """FastAPI request-scoped session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(settings: Settings | None = None) -> None:
    """Create directories and all tables. Idempotent — safe on every startup.

    Importing ``app.db.models`` registers the ORM classes on ``Base.metadata``
    so ``create_all`` materialises the full schema.
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    engine = get_engine(settings)

    # Imported for the side effect of registering tables on Base.metadata.
    from . import models  # noqa: F401

    # models.py declares its OWN DeclarativeBase, so create tables from that
    # metadata (database.Base here is empty and would create nothing).
    models.Base.metadata.create_all(bind=engine)

    # create_all builds the four new ReID tables but never ALTERs existing
    # ones, so the events.identity_* columns are added by an idempotent shim.
    ensure_reid_schema(engine)
    ensure_camera_schema(engine)
    ensure_identity_object_schema(engine)
    ensure_event_track_schema(engine)

    logger.info("DB initialised (tables created if absent) at %s", settings.db_path)


def ensure_reid_schema(engine: Engine) -> None:
    """Add the ReID columns to the existing ``events`` table, idempotently.

    SQLite can't add a column with an inline FK constraint to an existing
    table, so ``events.identity_id`` is materialised as a plain INTEGER column
    (the FK/relationship is declared at the ORM level; the ON DELETE behaviour
    for this denormalized link is enforced in application code on identity
    delete, mirroring how ``delete_person`` nulls ``Event.person_id``). Guarded
    by a PRAGMA table_info check so re-runs are no-ops.

    The four new tables (identities, sightings, face_exemplars,
    appearance_exemplars) are created by ``create_all`` and need no shim.
    """
    new_cols = {
        "identity_id": "INTEGER",
        "identity_name": "VARCHAR",
        "identity_score": "FLOAT",
    }
    with engine.begin() as conn:
        existing = {
            row[1]  # PRAGMA table_info columns: (cid, name, type, ...)
            for row in conn.exec_driver_sql("PRAGMA table_info(events)").fetchall()
        }
        for col, col_type in new_cols.items():
            if col not in existing:
                conn.exec_driver_sql(
                    f"ALTER TABLE events ADD COLUMN {col} {col_type}"
                )
                logger.info("Added events.%s column (ReID schema)", col)


def ensure_event_track_schema(engine: Engine) -> None:
    """Add track-driven-recording / detection-metadata columns to ``events``.

    Idempotent (PRAGMA table_info guard). Runs in the API process at startup
    BEFORE any worker subprocess writes a track-mode Event.
    """
    new_cols = {
        "num_objects": "INTEGER",
        "object_classes": "VARCHAR",
        "peak_confidence": "FLOAT",
        "num_frames": "INTEGER",
        "clip_start_ts": "DATETIME",
    }
    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(events)").fetchall()
        }
        for col, col_type in new_cols.items():
            if col not in existing:
                conn.exec_driver_sql(f"ALTER TABLE events ADD COLUMN {col} {col_type}")
                logger.info("Added events.%s column (event track schema)", col)


def ensure_identity_object_schema(engine: Engine) -> None:
    """Add object-class + dwell columns to ``identities``/``sightings``.

    Generalises the identity gallery from person-only to any object class and
    records per-identity total dwell seconds. The ``presence_segments`` table is
    created by ``create_all``; only the column adds need a shim. Idempotent.
    """
    table_cols = {
        "identities": {
            "object_class": ("VARCHAR", "'person'"),
            "total_seconds": ("FLOAT", "0"),
            "attributes": ("TEXT", "NULL"),
        },
        "sightings": {"object_class": ("VARCHAR", "'person'")},
    }
    with engine.begin() as conn:
        for table, cols in table_cols.items():
            existing = {
                row[1]
                for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            }
            for col, (col_type, default) in cols.items():
                if col not in existing:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default}"
                    )
                    logger.info("Added %s.%s column (identity object schema)", table, col)


def ensure_camera_schema(engine: Engine) -> None:
    """Add the per-camera trigger/tuning columns to ``cameras``, idempotently.

    Mirrors :func:`ensure_reid_schema`: guarded by a PRAGMA table_info check so
    re-runs are no-ops on databases created before these columns existed.
    """
    new_cols = {
        "trigger_classes": "VARCHAR",
        "detect_iou": "FLOAT",
        "detect_imgsz": "INTEGER",
        "detect_interval": "FLOAT",
        "trigger_cooldown": "FLOAT",
        "min_trigger_frames": "INTEGER",
        "rtsp_transport": "VARCHAR",
        "faces_enabled": "BOOLEAN",
        "reid_enabled": "BOOLEAN",
    }
    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(cameras)").fetchall()
        }
        for col, col_type in new_cols.items():
            if col not in existing:
                conn.exec_driver_sql(
                    f"ALTER TABLE cameras ADD COLUMN {col} {col_type}"
                )
                logger.info("Added cameras.%s column (camera schema)", col)


def reset_engine() -> None:
    """Dispose and forget the cached engine. Used by tests between runs."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None

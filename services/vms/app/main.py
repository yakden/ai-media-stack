"""FastAPI application factory and lifespan.

The integration glue: initialises the DB, boots the FaceIndex and the
per-camera WorkerManager on startup, tears them down on shutdown, mounts the
vanilla-JS SPA, wires every router, and serves the unauthenticated ``/health``
probe consumed by the control-plane.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.system import gpu_memory, router as system_router, worker_states
from .config import get_settings
from .db.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("vms")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Routers owned by sibling components. Imported defensively so the app remains
# buildable/bootable as components land; each is mounted only if importable.
_OPTIONAL_ROUTERS = (
    "app.api.cameras",
    "app.api.live",
    "app.api.events",
    "app.api.people",
    "app.api.identities",
    "app.api.face_groups",
)


def _boot_face_index(app: FastAPI, settings) -> None:
    """Instantiate the FAISS-backed FaceIndex (rebuilt from the DB)."""
    if not settings.faces_enabled:
        app.state.face_index = None
        return
    try:
        from .db.database import SessionLocal
        from .faces.index import FaceIndex

        db = SessionLocal()
        try:
            index = FaceIndex.from_db(
                db, match_threshold=settings.face_match_threshold
            )
            db.commit()  # from_db assigns faiss_id positions on the rows
        finally:
            db.close()
        app.state.face_index = index
        logger.info("FaceIndex loaded (faces=%s)", index.size)
    except Exception:
        logger.exception("FaceIndex failed to load; face matching disabled")
        app.state.face_index = None


def _boot_recognizer(app: FastAPI, settings) -> None:
    """Instantiate the shared insightface recognizer used for enrollment.

    Construction is cheap (the heavy model is loaded lazily on first use), so
    this never blocks startup; if construction fails enrollment returns 503.
    """
    if not settings.faces_enabled:
        app.state.recognizer = None
        return
    try:
        from .faces.recognizer import FaceRecognizer

        app.state.recognizer = FaceRecognizer(
            models_dir=str(settings.insightface_root),
            pack_name=settings.insightface_pack,
            device=settings.device,
            det_size=(settings.face_det_size, settings.face_det_size),
        )
        logger.info("FaceRecognizer instantiated (lazy model load)")
    except Exception:
        logger.exception("FaceRecognizer init failed; enrollment disabled")
        app.state.recognizer = None


def _boot_identity_gallery(app: FastAPI, settings) -> None:
    """Instantiate the authoritative identity gallery (rebuilt from the DB).

    The API process owns the authoritative gallery; camera workers each rebuild
    their own and re-sync on a timer. Degrades gracefully when ReID is disabled
    or the OSNet model / module isn't present yet.
    """
    if not getattr(settings, "reid_enabled", False):
        app.state.identity_gallery = None
        return
    try:
        from .db.database import SessionLocal
        from .reid.gallery import IdentityGallery

        db = SessionLocal()
        try:
            gallery = IdentityGallery(settings=settings)
            gallery.rebuild_from_db(db)
            db.commit()
        finally:
            db.close()
        app.state.identity_gallery = gallery
        logger.info("IdentityGallery loaded")
    except Exception:
        logger.exception("IdentityGallery failed to load; identities API degraded")
        app.state.identity_gallery = None


def _boot_reid_maintenance(app: FastAPI, settings) -> None:
    """Start the background ReID maintenance thread (decay/prune/merge)."""
    if not getattr(settings, "reid_enabled", False):
        app.state.reid_maintenance = None
        return
    try:
        from .reid import maintenance as reid_maintenance

        reid_maintenance.start(app.state)
        app.state.reid_maintenance = reid_maintenance
        logger.info("ReID maintenance thread started")
    except Exception:
        logger.exception("ReID maintenance failed to start")
        app.state.reid_maintenance = None


def _enabled_cameras(settings):
    """Load enabled Camera rows from the DB to seed the workers."""
    from .db.database import SessionLocal
    from .db.models import Camera

    db = SessionLocal()
    try:
        return list(db.query(Camera).filter(Camera.enabled.is_(True)).all())
    finally:
        db.close()


def _boot_hls(app: FastAPI, settings) -> None:
    """Start the on-demand HLS (live-with-sound) manager."""
    if not getattr(settings, "hls_enabled", True):
        app.state.hls = None
        return
    try:
        from .recording.hls import HlsManager

        app.state.hls = HlsManager(settings)
        logger.info("HLS (live-with-sound) manager started")
    except Exception:
        logger.exception("HLS manager failed to start; live-with-sound disabled")
        app.state.hls = None


def _boot_workers(app: FastAPI, settings) -> None:
    """Spawn per-camera worker processes for all enabled cameras."""
    try:
        from .workers.manager import WorkerManager

        manager = WorkerManager(settings)
        manager.start_all(_enabled_cameras(settings))
        manager.start_reconcile()  # keep running workers == enabled DB cameras
        app.state.workers = manager
        logger.info("WorkerManager started")
    except Exception:
        logger.exception("WorkerManager failed to start; live/recording disabled")
        app.state.workers = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    init_db(settings)
    # Remove on-disk artifacts (thumbnails/clips/segments/frame slots) left by
    # deleted cameras/events so stale footage from a gone camera is never shown.
    try:
        from .recording.cleanup import purge_orphans
        purge_orphans(settings)
    except Exception:
        logger.exception("Orphan artifact purge failed (non-fatal)")
    _boot_face_index(app, settings)
    _boot_recognizer(app, settings)
    _boot_identity_gallery(app, settings)
    _boot_reid_maintenance(app, settings)
    _boot_hls(app, settings)
    _boot_workers(app, settings)

    logger.info(
        "Startup complete (backend=%s device=%s port=%s)",
        settings.detector_backend,
        settings.device,
        settings.port,
    )
    try:
        yield
    finally:
        manager = getattr(app.state, "workers", None)
        if manager is not None:
            try:
                manager.shutdown()
                logger.info("WorkerManager stopped")
            except Exception:
                logger.exception("Error stopping WorkerManager")
        hls = getattr(app.state, "hls", None)
        if hls is not None:
            try:
                hls.shutdown()
                logger.info("HLS manager stopped")
            except Exception:
                logger.exception("Error stopping HLS manager")
        reid_maintenance = getattr(app.state, "reid_maintenance", None)
        if reid_maintenance is not None:
            try:
                reid_maintenance.stop()
                logger.info("ReID maintenance thread stopped")
            except Exception:
                logger.exception("Error stopping ReID maintenance")


def create_app() -> FastAPI:
    app = FastAPI(title="vms", version=__version__, lifespan=lifespan)

    # System router (/api/system). /health is defined below (no auth).
    app.include_router(system_router)

    for module_path in _OPTIONAL_ROUTERS:
        try:
            module = importlib.import_module(module_path)
            app.include_router(module.router)
            logger.info("Mounted router %s", module_path)
        except Exception as exc:  # ImportError until the component lands, etc.
            logger.warning("Router %s not mounted: %s", module_path, exc)

    @app.get("/health", include_in_schema=False)
    async def health(request: Request) -> dict:
        """Unauthenticated liveness + GPU/worker snapshot for the control-plane."""
        gpu = gpu_memory()
        return {
            "status": "ok",
            "version": __version__,
            "gpu": {"used_mb": gpu["used_mb"], "total_mb": gpu["total_mb"]},
            "workers": worker_states(request),
        }

    # Serve the SPA. index.html at "/", assets under their own paths. Mounting
    # StaticFiles(html=True) at root keeps /app.js, /style.css etc. working while
    # the API routers (registered above) take precedence for /api and /health.
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:  # pragma: no cover - frontend component not present yet
        logger.warning("Static dir %s missing; SPA not served", STATIC_DIR)

        @app.get("/", include_in_schema=False)
        async def _root() -> dict:
            return {"service": "vms", "version": __version__}

    return app


app = create_app()

"""WorkerManager: spawn/stop per-camera detection processes and expose status.

The API process owns a single :class:`WorkerManager`. On startup (and whenever a
camera is created / enabled) it spawns one detection subprocess per enabled
camera via :func:`app.workers.camera_worker.run_worker`. Each worker:

* reads RTSP frames, runs detection, triggers clip recording + face matching;
* writes its latest annotated JPEG to a per-camera *frame slot* so the live
  MJPEG / snapshot endpoints can serve it;
* publishes heartbeat status (state, fps, last_seen) into a shared registry.

Cross-process communication uses a :class:`multiprocessing.Manager` for the
status registry (a plain dict-of-dicts) and a small on-disk frames directory for
the JPEG slots (one file per camera, written atomically). The frames dir lives
on ``/dev/shm`` when available (it's a tmpfs, so this is effectively shared
memory) and falls back to ``<data>/frames``. This avoids the fragility of
fixed-size ``shared_memory`` blocks for variable-size JPEGs while remaining
fast.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("vms.workers.manager")

# State values published by workers and surfaced via the API.
STATE_STARTING = "starting"
STATE_ONLINE = "online"
STATE_OFFLINE = "offline"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"


def _parse_trigger_classes(raw: Any) -> list[str]:
    """Parse the camera's stored trigger classes (CSV string) into a list.

    Empty / NULL falls back to ``["person"]`` so a camera always records people
    unless explicitly reconfigured. Picklable plain list for the worker cfg.
    """
    if not raw:
        return ["person"]
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw]
    else:
        items = [p.strip() for p in str(raw).split(",")]
    items = [p for p in items if p]
    return items or ["person"]


@dataclass
class WorkerHandle:
    camera_id: int
    process: mp.Process
    started_at: float = field(default_factory=time.time)
    # Identity used to detect stale config and decide whether a restart is needed.
    rtsp_url: str = ""
    enabled: bool = True


class WorkerManager:
    """Owns the lifecycle of all per-camera worker subprocesses."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        # Spawn (not fork): CUDA / onnxruntime contexts are not fork-safe.
        try:
            self._ctx = mp.get_context("spawn")
        except ValueError:  # pragma: no cover - spawn always available on Linux
            self._ctx = mp.get_context()

        self._mp_manager = self._ctx.Manager()
        # camera_id -> {state, fps, last_seen, detector, error, updated_at, pid}
        self.status: Any = self._mp_manager.dict()
        self._handles: dict[int, WorkerHandle] = {}
        # Guards _handles against the API mutations vs the reconcile thread.
        self._lock = threading.RLock()
        self._reconcile_stop = threading.Event()
        self._reconcile_thread: threading.Thread | None = None
        self._reconcile_interval = float(getattr(settings, "worker_reconcile_seconds", 20.0))

        self.frames_dir = self._resolve_frames_dir(settings)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Worker frame slots dir: %s", self.frames_dir)

    # -- frame slot paths ---------------------------------------------------

    @staticmethod
    def _resolve_frames_dir(settings: Any) -> Path:
        configured = getattr(settings, "frames_dir", None)
        if configured:
            return Path(str(configured))
        shm = Path("/dev/shm")
        if shm.is_dir() and os.access(shm, os.W_OK):
            return shm / "vms_frames"
        data_dir = Path(str(getattr(settings, "data_dir", "data")))
        return data_dir / "frames"

    def frame_path(self, camera_id: int) -> Path:
        return self.frames_dir / f"cam_{camera_id}.jpg"

    def read_frame(self, camera_id: int) -> bytes | None:
        """Return the latest annotated JPEG bytes for a camera, or ``None``."""
        path = self.frame_path(camera_id)
        try:
            return path.read_bytes()
        except (FileNotFoundError, OSError):
            return None

    # -- lifecycle ----------------------------------------------------------

    def start_camera(self, camera: Any) -> None:
        """Spawn (or restart) the worker for ``camera`` if it is enabled.

        ``camera`` is an ORM ``Camera`` row (or any object exposing
        ``id``, ``rtsp_url``, ``enabled`` and the optional per-camera tunables).
        """
        with self._lock:
            self._start_camera_locked(camera)

    def _start_camera_locked(self, camera: Any) -> None:
        cam_id = int(camera.id)
        enabled = bool(getattr(camera, "enabled", True))
        rtsp_url = str(getattr(camera, "rtsp_url", "") or "")

        if not enabled:
            self._stop_camera_locked(cam_id)
            self._set_status(cam_id, state=STATE_STOPPED)
            return

        existing = self._handles.get(cam_id)
        if existing and existing.process.is_alive():
            if existing.rtsp_url == rtsp_url and existing.enabled == enabled:
                return  # already running with the same config
            # Config changed -> restart.
            self._stop_camera_locked(cam_id)

        cfg = self._camera_config(camera)
        self._set_status(cam_id, state=STATE_STARTING, error=None)

        proc = self._ctx.Process(
            target=_worker_entrypoint,
            name=f"vms-cam-{cam_id}",
            args=(cfg, self.status, str(self.frame_path(cam_id))),
            daemon=True,
        )
        proc.start()
        self._handles[cam_id] = WorkerHandle(
            camera_id=cam_id, process=proc, rtsp_url=rtsp_url, enabled=enabled
        )
        logger.info("Started worker for camera %s (pid=%s)", cam_id, proc.pid)

    def stop_camera(self, camera_id: int, timeout: float = 8.0) -> None:
        """Stop a camera worker and clean up its frame slot."""
        with self._lock:
            self._stop_camera_locked(camera_id, timeout)

    def _stop_camera_locked(self, camera_id: int, timeout: float = 8.0) -> None:
        handle = self._handles.pop(int(camera_id), None)
        if handle is None:
            self._set_status(int(camera_id), state=STATE_STOPPED)
            return
        proc = handle.process
        if proc.is_alive():
            logger.info("Stopping worker for camera %s (pid=%s)", camera_id, proc.pid)
            proc.terminate()
            proc.join(timeout)
            if proc.is_alive():
                logger.warning("Worker %s did not exit; killing", camera_id)
                proc.kill()
                proc.join(2.0)
        self._set_status(int(camera_id), state=STATE_STOPPED)
        try:
            self.frame_path(int(camera_id)).unlink(missing_ok=True)
        except OSError:
            pass

    def restart_camera(self, camera: Any) -> None:
        with self._lock:
            self._stop_camera_locked(int(camera.id))
            self._start_camera_locked(camera)

    def sync(self, cameras: list[Any]) -> None:
        """Reconcile running workers with the desired set of cameras.

        Starts/updates workers for enabled cameras and stops workers for
        cameras that are disabled or no longer present. Pass ENABLED cameras as
        the desired set (their workers must run; everything else is stopped)."""
        with self._lock:
            desired_ids = {int(c.id) for c in cameras if bool(getattr(c, "enabled", True))}
            for cam in cameras:
                try:
                    self._start_camera_locked(cam)
                except Exception:  # never let one bad camera abort the sync
                    logger.exception("Failed to start worker for camera %s", getattr(cam, "id", "?"))
                    self._set_status(int(cam.id), state=STATE_ERROR, error="spawn failed")
            # Stop any worker whose camera is no longer enabled/present — this is
            # the architectural guarantee that no orphaned/stale session keeps
            # generating data after a camera is deleted or disabled.
            for cam_id in list(self._handles.keys()):
                if cam_id not in desired_ids:
                    logger.info("Reconcile: stopping orphaned worker for camera %s", cam_id)
                    self._stop_camera_locked(cam_id)

    # -- periodic reconcile -------------------------------------------------

    def start_reconcile(self) -> None:
        """Start the background reconcile loop (running workers == enabled cams)."""
        if self._reconcile_thread is not None:
            return
        self._reconcile_stop.clear()
        self._reconcile_thread = threading.Thread(
            target=self._reconcile_loop, name="worker-reconcile", daemon=True
        )
        self._reconcile_thread.start()
        logger.info("Worker reconcile loop started (every %.0fs)", self._reconcile_interval)

    def _reconcile_loop(self) -> None:
        while not self._reconcile_stop.wait(self._reconcile_interval):
            try:
                self.sync(self._load_cameras())
            except Exception:
                logger.exception("Worker reconcile pass failed")

    def _load_cameras(self) -> list[Any]:
        """Load all cameras from the DB (the source of truth for reconcile)."""
        from ..db.database import SessionLocal
        from ..db.models import Camera

        session = SessionLocal()
        try:
            return list(session.query(Camera).all())
        finally:
            session.close()

    def start_all(self, cameras: list[Any]) -> None:
        for cam in cameras:
            if bool(getattr(cam, "enabled", True)):
                try:
                    self.start_camera(cam)
                except Exception:
                    logger.exception("Failed to start worker for camera %s", cam.id)

    def shutdown(self) -> None:
        """Stop every worker (called from the app lifespan teardown)."""
        self._reconcile_stop.set()
        if self._reconcile_thread is not None:
            self._reconcile_thread.join(timeout=5.0)
            self._reconcile_thread = None
        with self._lock:
            for cam_id in list(self._handles.keys()):
                self._stop_camera_locked(cam_id)
        try:
            self._mp_manager.shutdown()
        except Exception:  # pragma: no cover
            pass

    # -- status -------------------------------------------------------------

    def _set_status(self, camera_id: int, **fields: Any) -> None:
        cur = dict(self.status.get(camera_id, {}))
        cur.update(fields)
        cur["updated_at"] = time.time()
        self.status[camera_id] = cur

    def get_status(self, camera_id: int) -> dict[str, Any]:
        """Return a snapshot of a camera worker's status.

        Reconciles the published heartbeat with liveness: if the process died
        without publishing an error, report it as offline/error.
        """
        cam_id = int(camera_id)
        st = dict(self.status.get(cam_id, {}))
        handle = self._handles.get(cam_id)
        if handle is None:
            st.setdefault("state", STATE_STOPPED)
            return st
        if not handle.process.is_alive() and st.get("state") not in (STATE_STOPPED,):
            st["state"] = STATE_ERROR
            st.setdefault("error", "worker process exited")
        # Heartbeat staleness -> treat a silent worker as offline.
        updated = st.get("updated_at", 0)
        if st.get("state") == STATE_ONLINE and time.time() - updated > 15.0:
            st["state"] = STATE_OFFLINE
        st["pid"] = handle.process.pid
        return st

    def all_status(self) -> dict[int, dict[str, Any]]:
        return {cam_id: self.get_status(cam_id) for cam_id in self._handles}

    def is_running(self, camera_id: int) -> bool:
        handle = self._handles.get(int(camera_id))
        return bool(handle and handle.process.is_alive())

    # -- config marshalling -------------------------------------------------

    def _camera_config(self, camera: Any) -> dict[str, Any]:
        """Build a plain, picklable config dict passed to the worker process.

        We resolve per-camera tunables against global settings defaults here so
        the worker doesn't need to import the settings object across the spawn
        boundary for those values (it still loads settings for model paths etc.).
        """
        s = self.settings

        def cam_or_default(attr: str, default_attr: str, fallback: Any) -> Any:
            val = getattr(camera, attr, None)
            if val is not None:
                return val
            return getattr(s, default_attr, fallback)

        return {
            "camera_id": int(camera.id),
            "name": str(getattr(camera, "name", f"camera-{camera.id}")),
            "rtsp_url": str(getattr(camera, "rtsp_url", "")),
            "detect_conf": float(cam_or_default("detect_conf", "detect_conf", 0.4)),
            "pre_seconds": int(cam_or_default("pre_seconds", "pre_seconds", 5)),
            "post_seconds": int(cam_or_default("post_seconds", "post_seconds", 5)),
            # Objects that trigger recording on this camera (parsed CSV -> list).
            "trigger_classes": _parse_trigger_classes(getattr(camera, "trigger_classes", None)),
            # Per-camera detection/trigger overrides (fall back to globals).
            "detect_imgsz": int(cam_or_default("detect_imgsz", "detect_imgsz", 640)),
            "detect_iou": float(cam_or_default("detect_iou", "detect_iou", 0.45)),
            "detect_interval": float(cam_or_default("detect_interval", "detect_interval", 0.0)),
            # Adaptive detection cadence (active vs idle) + adaptive preview fps.
            "detect_interval_idle": float(getattr(s, "detect_interval_idle", 0.5)),
            "active_grace_seconds": float(getattr(s, "active_grace_seconds", 3.0)),
            "active_preview_fps": float(getattr(s, "active_preview_fps", 10.0)),
            "idle_preview_fps": float(getattr(s, "idle_preview_fps", 2.0)),
            "trigger_cooldown": float(cam_or_default("trigger_cooldown", "trigger_cooldown", 30.0)),
            "min_trigger_frames": int(cam_or_default("min_trigger_frames", "min_trigger_frames", 3)),
            "rtsp_transport": str(cam_or_default("rtsp_transport", "rtsp_transport", "tcp")),
            "faces_enabled": bool(cam_or_default("faces_enabled", "faces_enabled", True)),
            "reid_enabled": bool(cam_or_default("reid_enabled", "reid_enabled", True)),
            # Object tracking / dwell-time accounting.
            "track_iou": float(getattr(s, "track_iou", 0.3)),
            "track_gap_seconds": float(getattr(s, "track_gap_seconds", 3.0)),
            "reid_sample_seconds": float(getattr(s, "reid_sample_seconds", 3.0)),
            "reid_confident_sample_seconds": float(getattr(s, "reid_confident_sample_seconds", 9.0)),
            "max_reid_per_frame": int(getattr(s, "max_reid_per_frame", 4)),
            "min_track_frames": int(getattr(s, "min_track_frames", 2)),
            "reid_app_temporal_fusion": bool(getattr(s, "reid_app_temporal_fusion", False)),
            "face_pose_min": float(getattr(s, "face_pose_min", 0.3)),
            "recording_mode": str(getattr(s, "recording_mode", "track")),
            "segment_retention_seconds": int(getattr(s, "segment_retention_seconds", 120)),
            "min_track_seconds": float(getattr(s, "min_track_seconds", 1.0)),
            # Globals the worker needs but that aren't per-camera:
            "detector_backend": str(getattr(s, "detector_backend", "yolo")),
            "detector_device": str(getattr(s, "detector_device", getattr(s, "device", "cuda"))),
            "data_dir": str(getattr(s, "data_dir", "data")),
            "models_dir": str(getattr(s, "models_dir", "models")),
            "db_url": str(getattr(s, "db_url", getattr(s, "database_url", ""))),
            "face_match_threshold": float(getattr(s, "face_match_threshold", 0.35)),
            "frame_jpeg_quality": int(getattr(s, "frame_jpeg_quality", 75)),
            "reconnect_delay": float(getattr(s, "reconnect_delay", 3.0)),
            "deepstream_endpoint": str(getattr(s, "deepstream_endpoint", "")),
            # --- Face recognizer construction (real signature) ---
            "insightface_root": str(getattr(s, "insightface_root", "")),
            "insightface_pack": str(getattr(s, "insightface_pack", "buffalo_l")),
            "face_det_size": int(getattr(s, "face_det_size", 640)),
            # NB: faces_enabled / reid_enabled are resolved per-camera above.
            # --- Automatic identification (ReID) tunables ---
            "reid_model_path": str(getattr(s, "reid_model_path", "")),
            "reid_input_w": int(getattr(s, "reid_input_w", 128)),
            "reid_input_h": int(getattr(s, "reid_input_h", 256)),
            "reid_embedding_dim": int(getattr(s, "reid_embedding_dim", 512)),
            "identities_dir": str(getattr(s, "identities_dir", "data/identities")),
            "reid_face_match": float(getattr(s, "reid_face_match", 0.42)),
            "reid_face_strong": float(getattr(s, "reid_face_strong", 0.55)),
            "reid_face_reject_new": float(getattr(s, "reid_face_reject_new", 0.32)),
            "reid_app_match": float(getattr(s, "reid_app_match", 0.62)),
            "reid_app_match_cross": float(getattr(s, "reid_app_match_cross", 0.66)),
            "reid_app_gate": float(getattr(s, "reid_app_gate", 0.50)),
            "reid_app_window_seconds": int(getattr(s, "reid_app_window_seconds", 600)),
            "reid_app_decay_tau_seconds": int(
                getattr(s, "reid_app_decay_tau_seconds", 43_200)
            ),
            "reid_max_face_exemplars": int(getattr(s, "reid_max_face_exemplars", 8)),
            "reid_max_app_exemplars": int(getattr(s, "reid_max_app_exemplars", 16)),
            "reid_new_identity_rate_per_min": int(
                getattr(s, "reid_new_identity_rate_per_min", 30)
            ),
            "reid_provisional_grace_seconds": int(
                getattr(s, "reid_provisional_grace_seconds", 600)
            ),
            "reid_match_margin_face": float(getattr(s, "reid_match_margin_face", 0.06)),
            "reid_match_margin_app": float(getattr(s, "reid_match_margin_app", 0.05)),
            "reid_min_face_pixels": int(getattr(s, "reid_min_face_pixels", 24)),
            "reid_min_app_box_area_frac": float(
                getattr(s, "reid_min_app_box_area_frac", 0.01)
            ),
            "reid_require_quality_for_new": bool(
                getattr(s, "reid_require_quality_for_new", True)
            ),
            "reid_require_face_for_new_person": bool(
                getattr(s, "reid_require_face_for_new_person", True)
            ),
            "reid_face_exemplar_min_quality": float(
                getattr(s, "reid_face_exemplar_min_quality", 0.35)
            ),
            "reid_gallery_reload_seconds": int(
                getattr(s, "reid_gallery_reload_seconds", 30)
            ),
            "reid_device": str(getattr(s, "reid_device", getattr(s, "device", "cuda"))),
        }


def _worker_entrypoint(cfg: dict[str, Any], status: Any, frame_path: str) -> None:
    """Top-level (picklable) target for the spawned worker process."""
    # Import inside the child so the parent process never imports cv2/onnxruntime.
    from .camera_worker import run_worker

    run_worker(cfg, status, frame_path)

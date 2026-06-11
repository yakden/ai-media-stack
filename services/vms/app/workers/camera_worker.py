"""Per-camera worker process.

Runs in its own spawned subprocess (one per enabled camera). Responsibilities:

1. Pull frames from the camera's RTSP URL via OpenCV (FFMPEG backend, RTSP over
   TCP) and keep a warm rolling segment buffer (``recording.segmenter``) so
   pre-roll footage exists the instant a person appears.
2. Run the person detector on frames, draw boxes, and publish the latest
   annotated JPEG to the camera's frame slot for the live MJPEG/snapshot API.
3. Debounce person detections into discrete *triggers*; on a trigger, assemble a
   pre/post clip (``recording.clipper``), run face recognition on the clip
   thumbnail (``faces.recognizer`` + ``faces.index``), and write an ``Event``
   row to the DB.
4. Publish heartbeat status (state, fps, last_seen) into the shared registry so
   the manager / API can report online/offline.

Everything is wrapped so transient RTSP/decoder failures lead to reconnects
rather than a dead worker, and so a missing optional dependency (e.g. face
models not downloaded) degrades gracefully (events still recorded, just without
a recognized identity).
"""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..reid.decay import decayed_centroid

logger = logging.getLogger("vms.workers.camera")

# Mirror the manager's state constants (kept local to avoid an import cycle and
# to keep the worker self-contained across the spawn boundary).
STATE_STARTING = "starting"
STATE_ONLINE = "online"
STATE_OFFLINE = "offline"
STATE_ERROR = "error"
STATE_STOPPED = "stopped"

_BOX_COLOR = (0, 200, 0)
_MATCH_COLOR = (0, 165, 255)


class _Stop(Exception):
    """Raised to break out of the loop on SIGTERM/SIGINT."""


def run_worker(cfg: dict[str, Any], status: Any, frame_path_str: str) -> None:
    """Entry point invoked by the manager in the child process."""
    worker = CameraWorker(cfg, status, Path(frame_path_str))
    worker.run()


class CameraWorker:
    def __init__(self, cfg: dict[str, Any], status: Any, frame_path: Path) -> None:
        self.cfg = cfg
        self.camera_id = int(cfg["camera_id"])
        self.name = cfg.get("name", f"camera-{self.camera_id}")
        self.rtsp_url = cfg["rtsp_url"]
        self.status = status
        self.frame_path = frame_path
        self.tmp_frame_path = frame_path.with_suffix(".tmp")

        self.detect_conf = float(cfg.get("detect_conf", 0.4))
        self.pre_seconds = int(cfg.get("pre_seconds", 5))
        self.post_seconds = int(cfg.get("post_seconds", 5))
        self.detect_interval = float(cfg.get("detect_interval", 0.0))
        self.trigger_cooldown = float(cfg.get("trigger_cooldown", 30.0))
        self.min_trigger_frames = int(cfg.get("min_trigger_frames", 3))
        self.face_threshold = float(cfg.get("face_match_threshold", 0.35))
        self.jpeg_quality = int(cfg.get("frame_jpeg_quality", 75))
        self.reconnect_delay = float(cfg.get("reconnect_delay", 3.0))
        self.rtsp_transport = str(cfg.get("rtsp_transport", "tcp"))

        self.faces_enabled = bool(cfg.get("faces_enabled", True))
        self.reid_enabled = bool(cfg.get("reid_enabled", True))
        # Objects that trigger recording on this camera (COCO labels). Defaults
        # to people only. The detector reports every class; the worker filters.
        tc = cfg.get("trigger_classes") or ["person"]
        self.trigger_classes = {str(x).strip() for x in tc if str(x).strip()} or {"person"}
        # Object tracking / dwell-time accounting.
        self.track_iou = float(cfg.get("track_iou", 0.3))
        self.track_gap_seconds = float(cfg.get("track_gap_seconds", 3.0))
        self.reid_sample_seconds = float(cfg.get("reid_sample_seconds", 3.0))
        self.min_track_frames = int(cfg.get("min_track_frames", 2))
        # Min face quality (det_score * frontalness) to feed a face into the
        # per-track face template (drops hard profiles / blurry faces).
        self.face_pose_min = float(cfg.get("face_pose_min", 0.3))
        # Recording mode: 'track' = record exactly while the object is in view
        # (one event per presence, clip = [enter-pre, last+post]); 'trigger' =
        # legacy fixed-window debounced clip.
        self.recording_mode = str(cfg.get("recording_mode", "track"))
        self.retention_seconds = int(cfg.get("segment_retention_seconds", 120))
        self.min_track_seconds = float(cfg.get("min_track_seconds", 1.0))
        # Background clip-assembly: a single daemon thread drains a queue so the
        # detection loop never blocks on ffmpeg/post-roll; ALL clip-thread DB
        # writes go through this one thread (serialised, avoids SQLite locks).
        self._clip_queue = None
        self._clip_thread = None
        self._tracker = None
        self._vehicle_attrs = None
        self.gallery_reload_seconds = float(cfg.get("reid_gallery_reload_seconds", 30.0))

        self._running = True
        self._consecutive_person_frames = 0
        self._last_trigger_ts = 0.0
        self._active_trigger = False

        # Lazily-built components (constructed inside run() so import/CUDA cost
        # is paid in the child process).
        self._detector = None
        self._segmenter = None
        self._recognizer = None
        self._face_index = None
        self._reid_embedder = None
        self._identity_gallery = None
        self._identity_manager = None
        self._pipeline = None
        self._last_gallery_reload = 0.0

        # FPS accounting.
        self._fps = 0.0
        self._fps_window_start = time.time()
        self._fps_count = 0
        self._last_heartbeat = 0.0

    # -- status -------------------------------------------------------------

    def _publish(self, **fields: Any) -> None:
        try:
            cur = dict(self.status.get(self.camera_id, {}))
        except Exception:
            cur = {}
        cur.update(fields)
        cur["updated_at"] = time.time()
        cur["fps"] = round(self._fps, 1)
        cur.setdefault("detector", getattr(self._detector, "backend", self.cfg.get("detector_backend")))
        try:
            self.status[self.camera_id] = cur
        except Exception:
            pass

    def _heartbeat(self, state: str, *, last_seen: float | None = None, error: str | None = None) -> None:
        now = time.time()
        if state == STATE_ONLINE and now - self._last_heartbeat < 1.0:
            return  # throttle online heartbeats to ~1 Hz
        self._last_heartbeat = now
        fields: dict[str, Any] = {"state": state}
        if last_seen is not None:
            fields["last_seen"] = last_seen
        if error is not None:
            fields["error"] = error
        elif state in (STATE_ONLINE, STATE_STARTING):
            fields["error"] = None
        self._publish(**fields)

    # -- lifecycle ----------------------------------------------------------

    def _install_signals(self) -> None:
        def _handler(signum, _frame):  # noqa: ANN001
            self._running = False
            raise _Stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # pragma: no cover
                pass

    def _build_components(self) -> None:
        # Build a fresh settings object inside the child for model paths etc.
        settings = self._load_settings()

        # Detector (required).
        from ..detect.base import create_detector

        self._detector = create_detector(settings)
        # Apply this camera's detection thresholds to the detector instance.
        # conf/iou are pure post-processing thresholds and safe to override.
        # imgsz is only overridden when the model's input is dynamic (a fixed
        # export bakes in its size, so changing it would break inference).
        cam_conf = self.cfg.get("detect_conf")
        cam_iou = self.cfg.get("detect_iou")
        cam_imgsz = self.cfg.get("detect_imgsz")
        if cam_conf is not None and hasattr(self._detector, "conf"):
            self._detector.conf = float(cam_conf)
        if cam_iou is not None and hasattr(self._detector, "iou"):
            self._detector.iou = float(cam_iou)
        if cam_imgsz is not None and hasattr(self._detector, "session"):
            try:
                shape = self._detector.session.get_inputs()[0].shape
                dynamic = not (len(shape) == 4 and isinstance(shape[2], int))
                if dynamic:
                    self._detector.imgsz = int(cam_imgsz)
            except Exception:
                pass
        # If using the optional DeepStream backend, bind this worker's source.
        if getattr(self._detector, "backend", "") == "deepstream":
            try:
                self._detector.bind_source(str(self.camera_id))  # type: ignore[attr-defined]
            except Exception:
                logger.exception("Failed to bind DeepStream source for camera %s", self.camera_id)

        # Recording segmenter (warm rolling buffer). Best-effort: if it can't
        # start we still publish live frames and events (without a clip).
        try:
            from ..recording.segmenter import Segmenter

            self._segmenter = Segmenter(
                camera_id=self.camera_id,
                rtsp_url=self.rtsp_url,
                settings=settings,
                pre_seconds=self.pre_seconds,
                post_seconds=self.post_seconds,
            )
            self._segmenter.start()
        except Exception:
            logger.exception("Segmenter unavailable for camera %s; clips disabled", self.camera_id)
            self._segmenter = None

        # Face recognition (optional; degrades to no-match if unavailable).
        # Construct with the recognizer's *real* signature and rebuild the
        # FAISS index from the DB (it is derived state).
        if self.faces_enabled:
            try:
                from ..faces.index import FaceIndex
                from ..faces.recognizer import FaceRecognizer

                self._recognizer = FaceRecognizer(
                    models_dir=str(
                        self.cfg.get("insightface_root")
                        or getattr(settings, "insightface_root", "")
                    ),
                    pack_name=str(
                        self.cfg.get("insightface_pack")
                        or getattr(settings, "insightface_pack", "buffalo_l")
                    ),
                    device=str(
                        self.cfg.get("reid_device")
                        or getattr(settings, "device", "cuda")
                    ),
                    det_size=(
                        int(self.cfg.get("face_det_size", 640)),
                        int(self.cfg.get("face_det_size", 640)),
                    ),
                )
                self._face_index = self._load_face_index(settings)
            except Exception:
                logger.exception("Face recognition unavailable for camera %s", self.camera_id)
                self._recognizer = None
                self._face_index = None

        # Automatic cross-camera identification (optional; degrades like faces).
        #
        # Construct the real component contracts:
        #   * ReIDEmbedder(model_path, input_w, input_h, device, embedding_dim)
        #   * IdentityGallery(settings=...) rebuilt from the shared DB (derived
        #     state, like FaceIndex) and re-synced on a timer.
        #   * IdentityManager(gallery, config) — the brain; ``assign(session,
        #     feature, camera_id, ...)`` persists the Sighting + exemplars itself.
        #   * IdentityPipeline(min_app_box_area_frac, min_face_pixels,
        #     face_det_thresh, min_aspect).
        if self.reid_enabled:
            try:
                from ..reid.embedder import ReIDEmbedder
                from ..reid.manager import IdentityManager
                from ..reid.pipeline import IdentityPipeline

                self._reid_embedder = ReIDEmbedder(
                    model_path=str(
                        self.cfg.get("reid_model_path")
                        or getattr(settings, "reid_model_path", "")
                    ),
                    input_w=int(self.cfg.get("reid_input_w", 128)),
                    input_h=int(self.cfg.get("reid_input_h", 256)),
                    device=str(
                        self.cfg.get("reid_device")
                        or getattr(settings, "device", "cuda")
                    ),
                    embedding_dim=int(self.cfg.get("reid_embedding_dim", 512)),
                )
                self._identity_gallery = self._load_identity_gallery(settings)
                self._identity_manager = IdentityManager(
                    self._identity_gallery,
                    self._build_match_config(),
                )
                self._pipeline = IdentityPipeline(
                    min_app_box_area_frac=float(
                        self.cfg.get("reid_min_app_box_area_frac", 0.01)
                    ),
                    min_face_pixels=int(self.cfg.get("reid_min_face_pixels", 24)),
                    face_det_thresh=float(self.cfg.get("face_match_threshold", 0.5)),
                )
                from ..reid.tracker import ObjectTracker

                self._tracker = ObjectTracker(
                    iou_threshold=self.track_iou,
                    max_gap_seconds=self.track_gap_seconds,
                )
                # Background clip-assembly drain thread (non-blocking recording).
                self._clip_queue = queue.Queue(maxsize=64)
                self._clip_thread = threading.Thread(
                    target=self._clip_drain_loop, name=f"clip-cam{self.camera_id}", daemon=True,
                )
                self._clip_thread.start()
                # Vehicle make / body-type classifiers (NVIDIA TAO). Best-effort.
                try:
                    from ..detect.vehicle_attrs import VehicleAttributeClassifier

                    vac = VehicleAttributeClassifier(
                        models_dir=str(self.cfg.get("models_dir", "models")),
                        device=str(self.cfg.get("reid_device", "cuda")),
                    )
                    self._vehicle_attrs = vac if vac.available else None
                except Exception:
                    logger.exception("Vehicle attribute classifier unavailable for camera %s", self.camera_id)
                    self._vehicle_attrs = None
                self._last_gallery_reload = time.time()
                logger.info("ReID identification + dwell tracking enabled for camera %s", self.camera_id)
            except Exception:
                logger.exception("ReID unavailable for camera %s; identities disabled", self.camera_id)
                self._reid_embedder = None
                self._identity_gallery = None
                self._identity_manager = None
                self._pipeline = None
                self._tracker = None

    def _build_match_config(self):
        """Marshal the picklable cfg tunables into a MatchConfig for the manager."""
        from ..reid.manager import MatchConfig

        c = self.cfg
        return MatchConfig(
            face_match=float(c.get("reid_face_match", 0.42)),
            face_strong=float(c.get("reid_face_strong", 0.55)),
            face_reject_new=float(c.get("reid_face_reject_new", 0.32)),
            face_merge_threshold=float(c.get("reid_face_merge_threshold", 0.60)),
            app_match=float(c.get("reid_app_match", 0.62)),
            app_match_cross=float(c.get("reid_app_match_cross", 0.66)),
            app_gate=float(c.get("reid_app_gate", 0.50)),
            app_window_seconds=float(c.get("reid_app_window_seconds", 600.0)),
            app_decay_tau_seconds=float(c.get("reid_app_decay_tau_seconds", 43_200.0)),
            max_face_exemplars=int(c.get("reid_max_face_exemplars", 8)),
            max_app_exemplars=int(c.get("reid_max_app_exemplars", 16)),
            match_margin_face=float(c.get("reid_match_margin_face", 0.06)),
            match_margin_app=float(c.get("reid_match_margin_app", 0.05)),
            min_face_pixels=int(c.get("reid_min_face_pixels", 24)),
            min_app_box_area_frac=float(c.get("reid_min_app_box_area_frac", 0.01)),
            require_quality_for_new=bool(c.get("reid_require_quality_for_new", True)),
            require_face_for_new_person=bool(c.get("reid_require_face_for_new_person", True)),
            new_identity_rate_per_min=int(c.get("reid_new_identity_rate_per_min", 20)),
            app_temporal_fusion=bool(c.get("reid_app_temporal_fusion", False)),
        )

    def _load_identity_gallery(self, settings: Any):
        """Build the per-worker identity gallery from the DB (derived state)."""
        from ..db.database import SessionLocal
        from ..reid.gallery import IdentityGallery

        gallery = IdentityGallery(settings=settings)
        session = SessionLocal()
        try:
            gallery.rebuild_from_db(session)
            session.commit()
        finally:
            session.close()
        return gallery

    @staticmethod
    def _load_face_index(settings: Any):
        """Build the per-worker FAISS face index from the DB (derived state)."""
        from ..db.database import SessionLocal
        from ..faces.index import FaceIndex

        session = SessionLocal()
        try:
            index = FaceIndex.from_db(
                session,
                match_threshold=float(getattr(settings, "face_match_threshold", 0.4)),
            )
            session.commit()  # from_db assigns faiss_id positions
            return index
        finally:
            session.close()

    @staticmethod
    def _load_settings() -> Any:
        try:
            from ..config import get_settings

            return get_settings()
        except Exception:  # pragma: no cover - config not yet wired
            return object()

    def run(self) -> None:
        self._install_signals()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._heartbeat(STATE_STARTING)
        try:
            self._build_components()
        except Exception:
            logger.exception("Camera %s failed to initialize", self.camera_id)
            self._heartbeat(STATE_ERROR, error="initialization failed")
            self._teardown()
            return

        try:
            self._loop()
        except _Stop:
            logger.info("Camera %s worker stopping (signal)", self.camera_id)
        except Exception:
            logger.exception("Camera %s worker crashed", self.camera_id)
            self._heartbeat(STATE_ERROR, error="worker crashed")
        finally:
            self._teardown()
            self._heartbeat(STATE_STOPPED)

    def _teardown(self) -> None:
        # Close any still-open tracks so their dwell time + clips aren't lost.
        if self._tracker is not None:
            try:
                for tr in self._tracker.flush():
                    self._finalize_presence(tr)
            except Exception:
                logger.exception("Camera %s: track flush on teardown failed", self.camera_id)
        # Drain in-flight clip jobs WHILE the segmenter is still alive, then
        # stop the drain thread, then close components. Only wait if the
        # consumer thread is actually alive (else join() would block forever).
        if self._clip_queue is not None and self._clip_thread is not None and self._clip_thread.is_alive():
            try:
                self._clip_queue.join()  # wait for enqueued clips to assemble
            except Exception:
                pass
            try:
                self._clip_queue.put_nowait(None)  # stop sentinel
            except Exception:
                pass
            self._clip_thread.join(timeout=20.0)
        for comp in (
            self._segmenter,
            self._detector,
            self._recognizer,
            self._reid_embedder,
            self._identity_manager,
        ):
            try:
                if comp is not None and hasattr(comp, "close"):
                    comp.close()
            except Exception:
                logger.exception("Error closing component for camera %s", self.camera_id)
        try:
            self.tmp_frame_path.unlink(missing_ok=True)
        except OSError:
            pass

    # -- main loop ----------------------------------------------------------

    def _open_capture(self):
        import os

        import cv2

        # Force RTSP over the configured transport via FFMPEG options.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{self.rtsp_transport}|stimeout;5000000",
        )
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Keep latency low: small internal buffer.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _loop(self) -> None:
        import cv2

        cap = None
        last_detect = 0.0
        while self._running:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                self._heartbeat(STATE_OFFLINE, error="connecting")
                cap = self._open_capture()
                if not cap.isOpened():
                    logger.warning("Camera %s: cannot open RTSP; retrying", self.camera_id)
                    self._sleep(self.reconnect_delay)
                    cap = None
                    continue
                logger.info("Camera %s: RTSP connected", self.camera_id)

            ok, frame = cap.read()
            now = time.time()
            if not ok or frame is None:
                logger.warning("Camera %s: frame read failed; reconnecting", self.camera_id)
                cap.release()
                cap = None
                self._heartbeat(STATE_OFFLINE, error="stream interrupted")
                self._sleep(self.reconnect_delay)
                continue

            self._tick_fps()
            self._heartbeat(STATE_ONLINE, last_seen=now)

            # Throttle detection if configured (saves GPU on high-fps streams).
            boxes = []
            did_detect = False
            if self.detect_interval <= 0 or (now - last_detect) >= self.detect_interval:
                last_detect = now
                boxes = self._safe_detect(frame)
                did_detect = True

            # Keep only the object classes this camera records on.
            trig_boxes = [b for b in boxes if b.label in self.trigger_classes]
            # Track objects + accrue dwell time + (re-)identify (only on frames
            # where detection actually ran, so the gap timing matches cadence).
            if did_detect and self._tracker is not None:
                try:
                    self._track_and_identify(frame, trig_boxes, now)
                except Exception:
                    logger.exception("Camera %s: tracking/identify failed", self.camera_id)
            self._handle_detections(frame, trig_boxes, now)
            self._write_frame_slot(frame, trig_boxes)

        if cap is not None:
            cap.release()

    def _sleep(self, seconds: float) -> None:
        # Interruptible sleep that respects stop requests.
        end = time.time() + seconds
        while self._running and time.time() < end:
            time.sleep(min(0.2, end - time.time()))

    def _tick_fps(self) -> None:
        self._fps_count += 1
        elapsed = time.time() - self._fps_window_start
        if elapsed >= 2.0:
            self._fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_window_start = time.time()

    def _safe_detect(self, frame) -> list:
        try:
            boxes = self._detector.detect(frame)  # type: ignore[union-attr]
            return [b for b in boxes if b.score >= self.detect_conf]
        except Exception:
            logger.exception("Detection error on camera %s", self.camera_id)
            return []

    # -- trigger logic ------------------------------------------------------

    def _handle_detections(self, frame, boxes: list, now: float) -> None:
        # In 'track' mode events are born at track close (_finalize_presence),
        # so the legacy fixed-window trigger path is disabled to avoid double
        # recording. Frame counting is harmless to keep.
        has_trigger = len(boxes) > 0
        if has_trigger:
            self._consecutive_person_frames += 1
        else:
            self._consecutive_person_frames = 0

        if self.recording_mode != "trigger":
            return

        in_cooldown = (now - self._last_trigger_ts) < self.trigger_cooldown
        if (
            has_trigger
            and not self._active_trigger
            and not in_cooldown
            and self._consecutive_person_frames >= self.min_trigger_frames
        ):
            self._trigger_event(frame, boxes, now)

    def _trigger_event(self, frame, boxes: list, now: float) -> None:
        """Begin an event: create the row, schedule clip assembly + matching."""
        self._active_trigger = True
        self._last_trigger_ts = now
        ts = datetime.now(timezone.utc)
        best_score = max((b.score for b in boxes), default=0.0)
        labels = sorted({b.label for b in boxes})
        logger.info(
            "Camera %s: trigger [%s] (n=%d, conf=%.2f)",
            self.camera_id, ",".join(labels), len(boxes), best_score,
        )

        try:
            self._record_and_persist(frame.copy(), boxes, ts)
        except Exception:
            logger.exception("Camera %s: failed to record/persist event", self.camera_id)
        finally:
            self._active_trigger = False

    def _record_and_persist(self, snapshot, boxes: list, ts: datetime) -> None:
        """Assemble the clip, derive a thumbnail, match a face, write Event.

        Runs inline in the worker loop. Clip assembly waits for post-roll, so
        the loop pauses briefly; for the MVP a short pause per trigger (bounded
        by the cooldown) is acceptable and keeps the design simple.
        """
        # Event label = the dominant (highest-score) triggering class, so the
        # history shows what was actually detected (person / car / dog / ...).
        dominant = max(boxes, key=lambda b: b.score, default=None)
        label = dominant.label if dominant is not None else "person"

        event_id = self._create_event_row(ts, label)
        if event_id is None:
            return

        clip_path: str | None = None
        thumb_path: str | None = None
        person_id: int | None = None
        person_name: str | None = None
        match_score: float | None = None
        end_ts: datetime | None = None

        # Assemble the pre/post clip + thumbnail from the warm segment buffer.
        if self._segmenter is not None:
            try:
                from ..recording.clipper import build_clip

                result = build_clip(
                    segmenter=self._segmenter,
                    camera_id=self.camera_id,
                    event_id=event_id,
                    trigger_time=ts,
                    fallback_frame=snapshot,
                    data_dir=Path(self.cfg.get("data_dir", "data")),
                    pre_seconds=self.cfg.get("pre_seconds"),
                    post_seconds=self.cfg.get("post_seconds"),
                )
                clip_path = getattr(result, "clip_path", None) or (result.get("clip_path") if isinstance(result, dict) else None)
                thumb_path = getattr(result, "thumb_path", None) or (result.get("thumb_path") if isinstance(result, dict) else None)
                end_ts = datetime.now(timezone.utc)
            except Exception:
                logger.exception("Camera %s: clip assembly failed (event %s)", self.camera_id, event_id)

        # Always ensure a thumbnail exists (fall back to the trigger frame).
        if not thumb_path:
            thumb_path = self._write_thumbnail(snapshot, event_id)

        # Manual face DB match on the trigger frame (legacy "People" layer).
        if self._recognizer is not None and self._face_index is not None:
            try:
                person_id, person_name, match_score = self._match_face(snapshot)
            except Exception:
                logger.exception("Camera %s: face matching failed (event %s)", self.camera_id, event_id)

        # Automatic identification is driven continuously by the tracker
        # (_track_and_identify). Here we just denormalize the dominant object's
        # current identity onto the event so history shows who/what it was.
        identity_id, identity_name, identity_score = self._identity_for_boxes(boxes)

        self._update_event_row(
            event_id,
            clip_path=clip_path,
            thumb_path=thumb_path,
            person_id=person_id,
            person_name=person_name,
            match_score=match_score,
            identity_id=identity_id,
            identity_name=identity_name,
            identity_score=identity_score,
            end_ts=end_ts,
        )
        logger.info(
            "Camera %s: event %s saved (clip=%s person=%s identity=%s)",
            self.camera_id,
            event_id,
            bool(clip_path),
            person_name,
            identity_name,
        )

    def _match_face(self, frame) -> tuple[int | None, str | None, float | None]:
        """Match faces in the frame against the manual face DB (FAISS index)."""
        faces = self._recognizer.detect(frame)  # type: ignore[union-attr]
        best_id: int | None = None
        best_name: str | None = None
        best_score = -1.0
        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            match = self._face_index.match(embedding, threshold=self.face_threshold)  # type: ignore[union-attr]
            if match is None:
                continue
            if match.score > best_score:
                best_score = match.score
                best_id = int(match.person_id)
        if best_id is None:
            return None, None, None
        best_name = self._lookup_person_name(best_id)
        return best_id, best_name, best_score

    @staticmethod
    def _lookup_person_name(person_id: int) -> str | None:
        """Resolve a person's display name for the denormalized Event snapshot."""
        try:
            from ..db.database import SessionLocal
            from ..db.models import Person

            session = SessionLocal()
            try:
                person = session.get(Person, person_id)
                return person.name if person is not None else None
            finally:
                session.close()
        except Exception:
            return None

    # -- automatic identification -------------------------------------------

    def _identify(
        self, snapshot, boxes: list, ts: datetime, event_id: int
    ) -> tuple[int | None, str | None, float | None]:
        """Run the ReID pipeline + IdentityManager over the trigger boxes.

        The IdentityManager is the single source of truth: ``assign(session,
        feature, camera_id, ...)`` persists the Sighting row + exemplar updates
        and (via the gallery) mints/links identities. We commit the whole
        trigger-frame batch atomically, then write each sighting's body-crop
        thumbnail keyed on the returned sighting id, and return the dominant
        box's identity (id, name, fused score) to denormalize onto the Event.

        Hysteresis (sticky per-track assignment) and the new-identity rate limit
        live inside the manager, so the worker just feeds it features.
        """
        now = time.time()
        self._maybe_reload_gallery(now)

        features = self._pipeline.extract(  # type: ignore[union-attr]
            snapshot, boxes, self._recognizer, self._reid_embedder
        )
        if not features:
            return None, None, None

        # The dominant box is the largest-area feature (most prominent person).
        dominant_feat = max(
            features, key=lambda f: f.box.area, default=None
        )
        dom_identity: tuple[int | None, str | None, float | None] = (None, None, None)
        # (sighting_id, identity_id, bbox) for the post-commit thumbnail pass.
        thumbs: list[tuple[int, int, tuple[int, int, int, int]]] = []

        session = self._session()
        try:
            for feat in features:
                result = self._identity_manager.assign(  # type: ignore[union-attr]
                    session,
                    feat,
                    camera_id=self.camera_id,
                    ts=ts,
                    event_id=event_id,
                )
                identity_id = getattr(result, "identity_id", None)
                if identity_id is None:
                    continue  # dropped (quality gate / rate limit)
                sighting_id = getattr(result, "sighting_id", None)
                fused = (
                    getattr(result, "face_score", None)
                    if getattr(result, "match_kind", "") == "face"
                    else getattr(result, "appearance_score", None)
                )
                name = self._identity_name(session, int(identity_id))

                if sighting_id is not None:
                    thumbs.append(
                        (int(sighting_id), int(identity_id), feat.box.xyxy)
                    )

                if feat is dominant_feat:
                    dom_identity = (int(identity_id), name, fused)
                elif dom_identity[0] is None:
                    dom_identity = (int(identity_id), name, fused)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Camera %s: identity assignment failed", self.camera_id)
            return None, None, None
        finally:
            session.close()

        # Write the body-crop thumbnails after commit (the sighting ids exist).
        for sighting_id, identity_id, bbox in thumbs:
            self._write_sighting_thumb(snapshot, bbox, identity_id, sighting_id)

        return dom_identity

    # -- tracking + dwell-time accounting -----------------------------------

    def _track_and_identify(self, frame, boxes: list, now: float) -> None:
        """Advance the per-camera tracker, accrue dwell time, (re-)identify.

        Runs every detection frame. Closed tracks (object left view) are
        finalized into a PresenceSegment so their time counts toward the
        identity's total. Active tracks are embedded + matched to an identity
        every ``reid_sample_seconds`` so the same physical object is *remembered*
        across appearances/cameras regardless of orientation."""
        self._maybe_reload_gallery(now)
        active, closed = self._tracker.update(boxes, now)

        for tr in closed:
            self._finalize_presence(tr)

        if self._pipeline is None or self._identity_manager is None:
            return
        for tr in active:
            due = tr.identity_id is None or (now - tr.last_embed_ts) >= self.reid_sample_seconds
            if tr.frames >= self.min_track_frames and due:
                self._assign_track_identity(frame, tr, now)

    def _assign_track_identity(self, frame, tr, now: float) -> None:
        """Embed a track's current crop and assign/refresh its identity."""
        from ..detect.base import Box

        box = Box(
            x1=float(tr.bbox[0]), y1=float(tr.bbox[1]),
            x2=float(tr.bbox[2]), y2=float(tr.bbox[3]),
            score=float(tr.score), label=tr.object_class,
        )
        # Faces only help people; skip the face model for other classes.
        recognizer = self._recognizer if tr.object_class == "person" else None
        features = self._pipeline.extract(frame, [box], recognizer, self._reid_embedder)
        if not features:
            tr.last_embed_ts = now
            return

        # Vehicle make/body-type (NVIDIA TAO) for vehicle crops.
        if self._vehicle_attrs is not None:
            from ..detect.vehicle_attrs import VEHICLE_CLASSES

            if tr.object_class in VEHICLE_CLASSES:
                x1, y1, x2, y2 = tr.bbox
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                try:
                    attrs = self._vehicle_attrs.classify(crop)
                    features[0].vehicle_make = attrs.get("make")
                    features[0].vehicle_make_conf = float(attrs.get("make_conf", 0.0))
                    features[0].vehicle_type = attrs.get("type")
                    features[0].vehicle_type_conf = float(attrs.get("type_conf", 0.0))
                except Exception:
                    logger.debug("vehicle attr classify failed", exc_info=True)

        ts = datetime.now(timezone.utc)

        # Temporal aggregation: fuse this track's recent appearance embeddings
        # into one quality + time-decay weighted query vector. A single-sample
        # track returns that same vector, so flag-off / short tracks are
        # unchanged. Buffer the RAW vector BEFORE overwriting it.
        if (
            getattr(self._identity_manager.cfg, "app_temporal_fusion", False)
            and features[0].appearance_vec is not None
        ):
            raw = features[0].appearance_vec
            tr.appearance_vecs.append(raw)
            tr.appearance_ts.append(now)
            tr.appearance_q.append(float(getattr(features[0], "crop_quality", 1.0) or 1.0))
            cap = 8
            if len(tr.appearance_vecs) > cap:
                tr.appearance_vecs = tr.appearance_vecs[-cap:]
                tr.appearance_ts = tr.appearance_ts[-cap:]
                tr.appearance_q = tr.appearance_q[-cap:]
            # Use tz-aware UTC consistently with `ts` (fromtimestamp, NOT
            # utcfromtimestamp — the latter would inject the box's UTC offset).
            ts_list = [datetime.fromtimestamp(t, timezone.utc) for t in tr.appearance_ts]
            fused = decayed_centroid(
                tr.appearance_vecs, ts_list, now=ts,
                tau_seconds=float(self._identity_manager.cfg.app_decay_tau_seconds),
                weights=tr.appearance_q,
            )
            if fused is not None:
                features[0].appearance_vec = fused

            # Same idea for the FACE embedding: aggregate the track's recent
            # faces into one pose/quality-weighted template so a person matches
            # across angles. Faces aren't time-decayed (stable), so all
            # timestamps == now -> decayed_centroid becomes a pure quality mean.
            # Pose gate: only buffer reasonably frontal faces (face_quality).
            fvec = features[0].face_vec
            fq = float(getattr(features[0], "face_quality", 0.0) or 0.0)
            if fvec is not None and fq >= self.face_pose_min:
                tr.face_vecs.append(fvec)
                tr.face_q.append(max(fq, 0.05))
                if len(tr.face_vecs) > 8:
                    tr.face_vecs = tr.face_vecs[-8:]
                    tr.face_q = tr.face_q[-8:]
                fused_face = decayed_centroid(
                    tr.face_vecs, [ts] * len(tr.face_vecs), now=ts,
                    tau_seconds=1e12, weights=tr.face_q,  # tau huge => no decay
                )
                if fused_face is not None:
                    features[0].face_vec = fused_face

        prev_id = tr.identity_id
        session = self._session()
        try:
            result = self._identity_manager.assign(
                session, features[0], camera_id=self.camera_id, ts=ts,
            )
            identity_id = getattr(result, "identity_id", None)
            sighting_id = getattr(result, "sighting_id", None)
            # Remember this track's sightings so we can link them to the event
            # exactly on close (avoids the same-identity time-window race).
            if sighting_id is not None:
                tr.sighting_ids.append(int(sighting_id))
            if identity_id is not None:
                name = self._identity_name(session, int(identity_id))
                tr.identity_id = int(identity_id)
                tr.identity_name = name
                # Identity flip on this track -> drop the fusion buffer so we
                # never blend embeddings across two different identities.
                if prev_id is not None and prev_id != tr.identity_id:
                    tr.appearance_vecs.clear()
                    tr.appearance_ts.clear()
                    tr.appearance_q.clear()
                    tr.face_vecs.clear()
                    tr.face_q.clear()
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Camera %s: track identity assign failed", self.camera_id)
            session.close()
            tr.last_embed_ts = now
            return
        finally:
            tr.last_embed_ts = now

        if identity_id is not None and sighting_id is not None:
            self._write_sighting_thumb(frame, tr.bbox, int(identity_id), int(sighting_id))

        # Dedicated face-recognition layer: capture a face sample (crop +
        # ArcFace vector + the clothing appearance vector) for grouping. Only
        # for people with a visible face; throttled by the embed cadence.
        feat0 = features[0]
        if (
            tr.object_class == "person"
            and getattr(feat0, "has_face", False)
            and feat0.face_vec is not None
            and getattr(feat0, "face_bbox", None) is not None
        ):
            try:
                self._save_face_sample(frame, feat0, identity_id, sighting_id, ts)
            except Exception:
                logger.debug("face sample capture failed", exc_info=True)

        try:
            session.close()
        except Exception:
            pass

    def _save_face_sample(self, frame, feat, identity_id, sighting_id, ts) -> None:
        """Persist one captured face: thumbnail + ArcFace + clothing vectors."""
        import cv2

        from ..db.models import FaceSample
        from ..reid.gallery import serialize_vector

        data_dir = Path(self.cfg.get("data_dir", "data"))
        fx1, fy1, fx2, fy2 = (int(v) for v in feat.face_bbox)
        h, w = frame.shape[:2]
        # Pad the face box ~20% for a nicer thumbnail.
        pw, ph = int((fx2 - fx1) * 0.2), int((fy2 - fy1) * 0.2)
        fx1, fy1 = max(0, fx1 - pw), max(0, fy1 - ph)
        fx2, fy2 = min(w, fx2 + pw), min(h, fy2 + ph)
        if fx2 <= fx1 or fy2 <= fy1:
            return
        crop = frame[fy1:fy2, fx1:fx2]

        try:
            app_blob = serialize_vector(feat.appearance_vec) if feat.appearance_vec is not None else None
        except Exception:
            app_blob = None

        session = self._session()
        try:
            fs = FaceSample(
                camera_id=self.camera_id,
                ts=ts.replace(tzinfo=None) if ts.tzinfo else ts,
                vector=serialize_vector(feat.face_vec),
                app_vector=app_blob,
                quality=float(getattr(feat, "face_det_score", 0.0) or 0.0),
                identity_id=int(identity_id) if identity_id is not None else None,
                sighting_id=int(sighting_id) if sighting_id is not None else None,
            )
            session.add(fs)
            session.commit()
            fid = int(fs.id)
            samples_dir = data_dir / "face_samples"
            samples_dir.mkdir(parents=True, exist_ok=True)
            abs_path = samples_dir / f"{fid}.jpg"
            if cv2.imwrite(str(abs_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
                fs.thumb_path = str(Path("data") / "face_samples" / f"{fid}.jpg")
                session.commit()
        except Exception:
            session.rollback()
            logger.debug("face sample persist failed", exc_info=True)
        finally:
            session.close()

    def _finalize_presence(self, tr) -> None:
        """On track close: (track mode) birth ONE event for the whole presence
        and enqueue a non-blocking clip job; always accrue dwell for identified
        tracks. Cheap DB writes only — clip assembly runs on the drain thread."""
        if tr.finalized:
            return
        tr.finalized = True
        seconds = float(tr.duration)

        # 1. Track-mode event: record exactly while the object was in view.
        #    Independent of identity (reid-disabled / short / faceless tracks
        #    still record), gated by a min-frames/min-duration debounce.
        if (
            self.recording_mode == "track"
            and self._clip_queue is not None
            and tr.frames >= self.min_track_frames
            and seconds >= self.min_track_seconds
        ):
            ev_ts = datetime.utcfromtimestamp(tr.enter_ts)  # naive UTC
            event_id = self._create_event_row(
                ev_ts, tr.object_class,
                identity_id=tr.identity_id, identity_name=tr.identity_name,
                num_objects=1, object_classes=tr.object_class,
                num_frames=int(tr.frames), peak_confidence=float(tr.peak_score),
            )
            if event_id is not None:
                # Link this track's sightings to the event (exact ids).
                if tr.identity_id is not None and tr.sighting_ids:
                    self._link_sightings(event_id, tr.sighting_ids)
                try:
                    self._clip_queue.put_nowait(
                        {"event_id": int(event_id), "enter_ts": float(tr.enter_ts), "last_ts": float(tr.last_ts)}
                    )
                except queue.Full:
                    logger.warning("camera %s: clip queue full; dropping clip for event %s",
                                   self.camera_id, event_id)

        # 2. Dwell accrual (identity-gated, unchanged).
        if tr.identity_id is None or seconds <= 0.0:
            return
        from ..db.models import Identity, PresenceSegment

        session = self._session()
        try:
            session.add(PresenceSegment(
                identity_id=int(tr.identity_id), camera_id=self.camera_id,
                object_class=tr.object_class,
                enter_ts=datetime.utcfromtimestamp(tr.enter_ts),
                exit_ts=datetime.utcfromtimestamp(tr.last_ts), seconds=seconds,
            ))
            ident = session.get(Identity, int(tr.identity_id))
            if ident is not None:
                ident.total_seconds = float(ident.total_seconds or 0.0) + seconds
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Camera %s: presence finalize failed", self.camera_id)
        finally:
            session.close()

    def _link_sightings(self, event_id: int, sighting_ids: list) -> None:
        """Set Sighting.event_id for this track's sightings (idempotent)."""
        from sqlalchemy import update
        from ..db.models import Sighting

        ids = [int(s) for s in sighting_ids]
        if not ids:
            return
        session = self._session()
        try:
            session.execute(
                update(Sighting).where(
                    Sighting.id.in_(ids), Sighting.event_id.is_(None)
                ).values(event_id=int(event_id))
            )
            session.commit()
        except Exception:
            session.rollback()
            logger.debug("sighting link failed", exc_info=True)
        finally:
            session.close()

    def _clip_drain_loop(self) -> None:
        """Daemon: assemble track clips off the detection loop, serialised."""
        from ..recording.clipper import build_clip_from_track

        while True:
            job = self._clip_queue.get()
            try:
                if job is None:  # shutdown sentinel
                    return
                if self._segmenter is None:
                    continue
                try:
                    r = build_clip_from_track(
                        segmenter=self._segmenter, camera_id=self.camera_id,
                        event_id=job["event_id"], enter_ts=job["enter_ts"],
                        last_ts=job["last_ts"], data_dir=Path(self.cfg.get("data_dir", "data")),
                        pre_seconds=self.pre_seconds, post_seconds=self.post_seconds,
                    )
                    self._update_event_row(
                        job["event_id"], clip_path=r.clip_path, thumb_path=r.thumb_path,
                        end_ts=datetime.utcfromtimestamp(job["last_ts"] + self.post_seconds),
                        clip_start_ts=datetime.utcfromtimestamp(job["enter_ts"] - self.pre_seconds),
                    )
                except Exception:
                    logger.exception("camera %s: track clip assembly failed (event %s)",
                                     self.camera_id, job.get("event_id"))
            finally:
                self._clip_queue.task_done()

    def _identity_for_boxes(self, boxes: list) -> tuple[int | None, str | None, float | None]:
        """Best-effort: identity of the dominant box from the active tracker.

        Matches the largest trigger box to a live track by IoU so an event can
        be denormalized with the object's identity without a second assignment."""
        if self._tracker is None or not boxes:
            return None, None, None
        from ..reid.tracker import _iou

        dom = max(boxes, key=lambda b: b.area, default=None)
        if dom is None:
            return None, None, None
        dom_xyxy = tuple(int(v) for v in dom.xyxy)
        best, best_iou = None, 0.1
        for tr in self._tracker.tracks.values():
            if tr.object_class != getattr(dom, "label", "person"):
                continue
            iou = _iou(tr.bbox, dom_xyxy)
            if iou >= best_iou and tr.identity_id is not None:
                best, best_iou = tr, iou
        if best is None:
            return None, None, None
        return best.identity_id, best.identity_name, None

    @staticmethod
    def _identity_name(session, identity_id: int) -> str | None:
        from ..db.models import Identity

        ident = session.get(Identity, identity_id)
        return ident.name if ident is not None else None

    def _maybe_reload_gallery(self, now: float) -> None:
        """Periodically re-sync the per-worker gallery from the shared DB so
        identities created by other workers / the API process converge."""
        if self._identity_gallery is None:
            return
        if now - self._last_gallery_reload < self.gallery_reload_seconds:
            return
        self._last_gallery_reload = now
        session = self._session()
        try:
            self._identity_gallery.rebuild_from_db(session)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Camera %s: gallery reload failed", self.camera_id)
        finally:
            session.close()

    # -- frame slot ---------------------------------------------------------

    def _annotate(self, frame, boxes: list):
        import cv2

        annotated = frame
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
            label = f"{b.label} {b.score:.2f}"
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR, 1, cv2.LINE_AA,
            )
        return annotated

    def _write_frame_slot(self, frame, boxes: list) -> None:
        import cv2

        try:
            annotated = self._annotate(frame, boxes)
            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                return
            # Atomic write: write tmp then rename so readers never see a partial.
            self.tmp_frame_path.write_bytes(buf.tobytes())
            self.tmp_frame_path.replace(self.frame_path)
        except Exception:
            logger.debug("Camera %s: failed to write frame slot", self.camera_id, exc_info=True)

    def _write_thumbnail(self, frame, event_id: int) -> str | None:
        import cv2

        data_dir = Path(self.cfg.get("data_dir", "data"))
        thumb_dir = data_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        rel = Path("data") / "thumbnails" / f"{event_id}.jpg"
        abs_path = thumb_dir / f"{event_id}.jpg"
        try:
            cv2.imwrite(str(abs_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return str(rel)
        except Exception:
            logger.exception("Camera %s: failed to write thumbnail for event %s", self.camera_id, event_id)
            return None

    def _write_sighting_thumb(
        self, snapshot, bbox: tuple[int, int, int, int], identity_id: int, sighting_id: int
    ) -> None:
        """Crop the body box and save it as the sighting thumbnail, then record
        the relative path on the Sighting row."""
        import cv2

        x1, y1, x2, y2 = bbox
        h, w = snapshot.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        crop = snapshot[y1:y2, x1:x2]
        data_dir = Path(self.cfg.get("data_dir", "data"))
        ident_dir = data_dir / "identities" / str(identity_id)
        ident_dir.mkdir(parents=True, exist_ok=True)
        rel = Path("data") / "identities" / str(identity_id) / f"{sighting_id}.jpg"
        abs_path = ident_dir / f"{sighting_id}.jpg"
        try:
            cv2.imwrite(str(abs_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        except Exception:
            logger.exception("Camera %s: failed to write sighting thumb %s", self.camera_id, sighting_id)
            return
        try:
            from ..db.models import Sighting

            session = self._session()
            try:
                s = session.get(Sighting, sighting_id)
                if s is not None:
                    s.thumb_path = str(rel)
                    session.commit()
            finally:
                session.close()
        except Exception:
            logger.exception("Camera %s: failed to record sighting thumb path", self.camera_id)

    # -- DB access ----------------------------------------------------------

    def _session(self):
        """Yield a DB session usable inside the worker process.

        The DB component owns ``SessionLocal``/engine; we import it lazily so the
        worker stays decoupled and a misconfigured DB degrades to no events
        rather than crashing the detection loop.
        """
        from ..db.database import SessionLocal

        return SessionLocal()

    def _create_event_row(self, ts: datetime, label: str = "person", **meta: Any) -> int | None:
        try:
            from ..db.models import Event

            session = self._session()
            try:
                # Only pass columns the Event model knows about.
                allowed = {
                    k: v for k, v in meta.items()
                    if v is not None and hasattr(Event, k)
                }
                event = Event(camera_id=self.camera_id, ts=ts, label=label, **allowed)
                session.add(event)
                session.commit()
                session.refresh(event)
                return int(event.id)
            finally:
                session.close()
        except Exception:
            logger.exception("Camera %s: could not create event row", self.camera_id)
            return None

    def _update_event_row(self, event_id: int, **fields: Any) -> None:
        try:
            from ..db.models import Event

            session = self._session()
            try:
                event = session.get(Event, event_id)
                if event is None:
                    return
                for key, value in fields.items():
                    if value is not None:
                        setattr(event, key, value)
                session.commit()
            finally:
                session.close()
        except Exception:
            logger.exception("Camera %s: could not update event row %s", self.camera_id, event_id)

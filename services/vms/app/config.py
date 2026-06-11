"""Runtime configuration, loaded from environment variables / .env.

Single source of truth for ports, on-disk paths, detection/recording defaults,
the device/backend selection, and the SSO/auth knobs. Every other component
reads its tunables from here via ``get_settings()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore", case_sensitive=False
    )

    # --- HTTP server ---
    # Bind to loopback only; the box reaches us via the nginx SSO gateway.
    host: str = "127.0.0.1"
    port: int = 8120

    # --- Auth ---
    # nginx sets a trusted header (default X-Email) after cookie-SSO. Presence of
    # the header == authenticated. The user-name header is read for display.
    sso_email_header: str = "X-Email"
    sso_user_header: str = "X-User"
    # When true, requests without an SSO header AND without a valid API key are
    # rejected. Set false for local dev where neither nginx nor a key is present.
    auth_required: bool = False
    # Optional bearer token for SSH-tunnel / CLI use that bypasses nginx SSO.
    api_key: str | None = None

    # --- Storage layout ---
    # All mutable state lives under data/ (bind-mounted); models under models/.
    data_dir: Path = Path("/app/data")
    models_dir: Path = Path("/app/models")

    # --- Detection ---
    # "yolo"/"onnx" (onnxruntime YOLOv8n, default), "cpu" (force onnxruntime CPU
    # EP), or "deepstream" (optional high-throughput backend).
    detector_backend: str = "yolo"
    # Compute device for onnxruntime models. "cuda" or "cpu". ``detector_device``
    # is an env alias; ``device`` remains the canonical field both detector and
    # face recognizer read.
    device: str = "cuda"
    detector_device: str | None = None  # env alias -> resolved into ``device``
    yolo_model: str = "yolov8n.onnx"
    # Default person-confidence threshold (per-camera value overrides this).
    detect_conf: float = 0.4
    # NMS IoU threshold for the YOLO post-process.
    detect_iou: float = 0.45
    # Only the COCO "person" class (id 0) is acted on in the MVP.
    person_class_id: int = 0
    # Detector inference square size.
    detect_imgsz: int = 640
    # Run detection every Nth decoded frame to keep GPU/CPU modest.
    detect_every_n: int = 3
    # Minimum seconds between detection passes (0 = every frame the loop reads).
    detect_interval: float = 0.0
    # Optional DeepStream metadata endpoint (only used when backend=deepstream).
    deepstream_endpoint: str = ""

    # --- Recording (segmenter + clipper) ---
    # Length of each rolling ffmpeg segment, in seconds.
    segment_seconds: int = 2
    # Default pre/post-roll around a trigger (per-camera values override these).
    pre_seconds: int = 5
    post_seconds: int = 5
    # Minimum gap between two clip triggers on the same camera (debounce).
    trigger_cooldown: float = 30.0
    # Recording mode: 'track' = record exactly while an object is in view (one
    # event per presence, clip = [enter-pre, last+post]); 'trigger' = legacy
    # fixed-window debounced clip around each trigger.
    recording_mode: str = "track"
    # Track mode debounce: minimum presence duration to mint an event.
    min_track_seconds: float = 1.0
    # How often the manager reconciles running workers against the DB (stops
    # orphaned/disabled/deleted-camera sessions; restarts crashed ones).
    worker_reconcile_seconds: float = 20.0
    # Consecutive person frames required before a trigger fires (debounce).
    min_trigger_frames: int = 3
    # How many rolling segments to keep on disk per camera before pruning.
    max_segments_per_camera: int = 60
    # Seconds of rolling segment buffer to retain per camera (>= largest pre).
    segment_retention_seconds: int = 120

    # --- Face recognition ---
    faces_enabled: bool = True
    # insightface model pack directory under models_dir.
    insightface_pack: str = "buffalo_l"
    # Cosine-similarity threshold for declaring a face match.
    face_match_threshold: float = 0.45
    # Face detector minimum size (SCRFD det_size square).
    face_det_size: int = 640

    # --- Automatic cross-camera identification (ReID) ---
    # Master switch; degrades gracefully (like faces) when off/unavailable.
    reid_enabled: bool = True
    # OSNet appearance model exported to ONNX, under models_dir. Default is the
    # AIN x1.0 variant (domain-generalizable → best cross-camera/angle accuracy).
    # Lighter fallback for CPU-bound boxes: osnet_x0_25_msmt17.onnx.
    reid_model: str = "osnet_ain_x1_0_msmt17.onnx"
    # OSNet input geometry (WxH) — the torchreid ReID standard. 512-d output.
    reid_input_w: int = 128
    reid_input_h: int = 256
    reid_embedding_dim: int = 512
    # Matching thresholds (cosine; vectors L2-normalized). Tuned for buffalo_l
    # ArcFace + OSNet on a T4 — see matching_algorithm.
    reid_face_match: float = 0.42
    reid_face_strong: float = 0.55
    reid_face_reject_new: float = 0.32
    reid_app_match: float = 0.62
    reid_app_match_cross: float = 0.66
    reid_app_gate: float = 0.50
    # Appearance-only links allowed only within this temporal window.
    reid_app_window_seconds: int = 600
    # Appearance time-decay constant (people change clothes between days).
    reid_app_decay_tau_seconds: int = 43_200  # 12h
    # Fuse a track's recent appearance + FACE embeddings into one query vector
    # (quality/pose + time-decay weighted) before matching — robust to viewpoint.
    # Default ON; set REID_APP_TEMPORAL_FUSION=false to revert to per-frame.
    reid_app_temporal_fusion: bool = True
    # Min face quality (det_score x frontalness) to use a face for the per-track
    # face template — a cheap pose/quality gate (drops profiles/blur).
    face_pose_min: float = 0.3
    # Per-identity exemplar caps.
    reid_max_face_exemplars: int = 12  # multi-view gallery: frontal + L/R profiles
    reid_max_app_exemplars: int = 16
    # Anti-explosion: cap new-identity creation per camera per minute.
    reid_new_identity_rate_per_min: int = 30
    # Provisional (faceless single-sighting) identities are reaped after this.
    reid_provisional_grace_seconds: int = 600
    # Face-only conservative auto-merge threshold (maintenance).
    reid_face_merge_threshold: float = 0.60
    # Match margin (best - second_best) required to accept; ambiguous -> NEW.
    reid_match_margin_face: float = 0.06
    reid_match_margin_app: float = 0.05
    # Quality gates.
    reid_min_face_pixels: int = 24
    reid_min_app_box_area_frac: float = 0.01
    # Require a quality crop to CREATE a new identity (drop blurry singletons).
    reid_require_quality_for_new: bool = True
    # Identity is anchored on the FACE: a faceless person (back/side view) never
    # spawns a NEW identity — it only attaches to an existing one by appearance
    # within a session, else is dropped. THE fix for back-view duplicate explosion.
    reid_require_face_for_new_person: bool = True
    # How often the worker reloads the shared gallery from the DB.
    reid_gallery_reload_seconds: int = 30
    # How often the maintenance thread runs its decay/prune/merge pass.
    reid_maintenance_interval_seconds: int = 120

    # --- Live monitoring ---
    # Target JPEG quality for the MJPEG/snapshot frames (1-100). ``frame_jpeg_quality``
    # is the name the camera worker reads; ``mjpeg_quality`` kept as an alias.
    frame_jpeg_quality: int = 70
    # Max frames per second pushed to an MJPEG client.
    live_mjpeg_fps: float = 10.0
    # --- Live-with-sound (on-demand RTSP->HLS, AAC audio; MJPEG carries none) ---
    hls_enabled: bool = True
    hls_segment_seconds: int = 2
    hls_list_size: int = 6
    hls_idle_timeout: int = 60   # stop a session this many seconds after last access
    hls_max_sessions: int = 2    # hard cap on concurrent live-with-sound transcodes

    # --- Workers ---
    # RTSP transport for OpenCV/ffmpeg ("tcp" recommended over lossy links).
    rtsp_transport: str = "tcp"
    # RTSP read timeout (microseconds passed to OpenCV/FFMPEG) and reconnect delay.
    rtsp_timeout_us: int = 5_000_000
    reconnect_delay: float = 5.0
    # Seconds without a fresh frame before a camera is marked "offline".
    status_stale_seconds: int = 15

    @property
    def mjpeg_quality(self) -> int:
        return self.frame_jpeg_quality

    @property
    def mjpeg_fps(self) -> float:
        return self.live_mjpeg_fps

    # --- Derived paths (created at startup by init) ---
    @property
    def db_path(self) -> Path:
        return self.data_dir / "vms.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    # The WorkerManager passes ``db_url`` to subprocess workers; alias it.
    @property
    def db_url(self) -> str:
        return self.database_url

    @property
    def recordings_dir(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"

    @property
    def face_samples_dir(self) -> Path:
        return self.data_dir / "face_samples"

    @property
    def segments_dir(self) -> Path:
        return self.data_dir / "segments"

    @property
    def hls_dir(self) -> Path:
        return self.data_dir / "hls"

    @property
    def identities_dir(self) -> Path:
        return self.data_dir / "identities"

    @property
    def yolo_model_path(self) -> Path:
        return self.models_dir / self.yolo_model

    @property
    def reid_model_path(self) -> Path:
        return self.models_dir / self.reid_model

    # The OSNet device follows the canonical detector/face device.
    @property
    def reid_device(self) -> str:
        return self.device

    @property
    def insightface_root(self) -> Path:
        # insightface expects <root>/models/<pack>. The download script and the
        # Dockerfile (INSIGHTFACE_HOME) both use <models_dir>/insightface as the
        # root, so the pack resolves to <models_dir>/insightface/models/<pack>.
        return self.models_dir / "insightface"

    @model_validator(mode="after")
    def _resolve_device_alias(self) -> "Settings":
        # DETECTOR_DEVICE env (if set) overrides ``device`` for the detector
        # and, via ``device``, the face recognizer. "cpu" backend forces CPU.
        if self.detector_device:
            object.__setattr__(self, "device", self.detector_device)
        if str(self.detector_backend).lower() == "cpu":
            object.__setattr__(self, "device", "cpu")
            object.__setattr__(self, "detector_backend", "yolo")
        return self

    def ensure_dirs(self) -> None:
        """Create all on-disk directories the app writes to."""
        for d in (
            self.data_dir,
            self.recordings_dir,
            self.thumbnails_dir,
            self.faces_dir,
            self.face_samples_dir,
            self.segments_dir,
            self.hls_dir,
            self.identities_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

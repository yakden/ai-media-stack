"""insightface SCRFD + ArcFace wrapper.

Loads the ``buffalo_l`` model pack (SCRFD-10GF detector + ArcFace r50
recognition head) under onnxruntime, runs face detection + 512-d embedding
extraction on a BGR frame, and L2-normalizes the embeddings so the FAISS
``IndexFlatIP`` yields cosine similarity directly.

Design notes
------------
* Models live in ``<MODELS_DIR>/insightface`` (insightface expects a ``root``
  dir that contains a ``models/<pack_name>/`` subtree). We point ``root`` at
  ``MODELS_DIR`` so the pack resolves to ``MODELS_DIR/models/buffalo_l``.
* GPU (CUDAExecutionProvider) is used when ``FACE_DEVICE`` is ``cuda`` and the
  CUDA provider is actually available in the onnxruntime build; otherwise we
  fall back to CPU. This keeps the worker alive when the control-plane evicts
  GPU memory for a heavy generative job.
* The recognizer is process-local and lazily initialized via
  :func:`get_recognizer` so each camera worker subprocess gets its own
  onnxruntime session (CUDA contexts are not fork-safe to share).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
DEFAULT_PACK = "buffalo_l"
# SCRFD input size (w, h). 640x640 is the buffalo_l default and a good
# accuracy/latency tradeoff on a T4.
DEFAULT_DET_SIZE = (640, 640)


@dataclass
class DetectedFace:
    """A single detected face in a frame.

    Attributes
    ----------
    bbox:
        ``(x1, y1, x2, y2)`` integer pixel coordinates in the source frame.
    det_score:
        SCRFD detection confidence in ``[0, 1]``.
    embedding:
        512-d float32 vector, L2-normalized (unit length). Ready to push into
        an ``IndexFlatIP`` for cosine matching.
    kps:
        Optional 5-point facial landmarks ``(5, 2)`` float32 array.
    """

    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: np.ndarray
    kps: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm as float32."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return (vec / norm).astype(np.float32)


def _resolve_providers(device: str) -> list[str]:
    """Pick onnxruntime providers, honoring the requested device but degrading
    gracefully if CUDA isn't actually available in this build/runtime."""
    try:
        import onnxruntime as ort  # noqa: WPS433 (local import keeps import light)

        available = set(ort.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime always present in prod
        available = set()

    want_gpu = str(device).lower() in {"cuda", "gpu"}
    if want_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if want_gpu:
        logger.warning(
            "FACE_DEVICE=%s requested but CUDAExecutionProvider unavailable "
            "(have %s); falling back to CPU.",
            device,
            sorted(available),
        )
    return ["CPUExecutionProvider"]


class FaceRecognizer:
    """Wraps an insightface ``FaceAnalysis`` app for detect + embed.

    Parameters
    ----------
    models_dir:
        Root directory holding the insightface pack. The pack is expected at
        ``<models_dir>/models/<pack_name>``.
    pack_name:
        insightface model pack name. Defaults to ``buffalo_l``.
    device:
        ``"cuda"`` or ``"cpu"``. Falls back to CPU automatically if CUDA is
        unavailable.
    det_size:
        SCRFD detector input size ``(w, h)``.
    det_thresh:
        Minimum detection confidence to keep a face.
    """

    def __init__(
        self,
        models_dir: str,
        pack_name: str = DEFAULT_PACK,
        device: str = "cuda",
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        det_thresh: float = 0.5,
    ) -> None:
        self.models_dir = models_dir
        self.pack_name = pack_name
        self.device = device
        self.det_size = det_size
        self.det_thresh = float(det_thresh)
        self._app = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            # Imported lazily so the rest of the package (and tests) don't
            # require the heavy insightface/onnxruntime stack to be installed.
            from insightface.app import FaceAnalysis  # noqa: WPS433

            providers = _resolve_providers(self.device)
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
            logger.info(
                "Loading insightface pack '%s' from '%s' (providers=%s, ctx_id=%d)",
                self.pack_name,
                self.models_dir,
                providers,
                ctx_id,
            )
            app = FaceAnalysis(
                name=self.pack_name,
                root=self.models_dir,
                providers=providers,
            )
            app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
            self._app = app
            logger.info("insightface recognizer ready.")

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces in a BGR frame and return them with embeddings.

        Faces are returned sorted by detection score (descending).
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        self._ensure_loaded()
        assert self._app is not None

        faces = self._app.get(frame)
        results: list[DetectedFace] = []
        for f in faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                raw = getattr(f, "embedding", None)
                if raw is None:
                    continue
                emb = l2_normalize(raw)
            else:
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.shape[0] != EMBEDDING_DIM:
                logger.warning(
                    "Unexpected embedding dim %d (expected %d); skipping face.",
                    emb.shape[0],
                    EMBEDDING_DIM,
                )
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in f.bbox)
            results.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    det_score=float(getattr(f, "det_score", 0.0)),
                    embedding=emb,
                    kps=getattr(f, "kps", None),
                )
            )
        results.sort(key=lambda d: d.det_score, reverse=True)
        return results

    def embed_single(self, frame: np.ndarray) -> Optional[DetectedFace]:
        """Convenience for enrollment: return the largest detected face (by
        bbox area), or ``None`` if no face is found.

        The largest face is used because enrollment photos are expected to be
        portraits of a single subject; picking by area is more robust to small
        background faces than picking by detection score alone.
        """
        faces = self.detect(frame)
        if not faces:
            return None
        return max(faces, key=lambda d: d.area)


# --- process-local singleton -------------------------------------------------

_RECOGNIZER: Optional[FaceRecognizer] = None
_RECOGNIZER_LOCK = threading.Lock()


def get_recognizer(
    models_dir: Optional[str] = None,
    pack_name: str = DEFAULT_PACK,
    device: Optional[str] = None,
    det_thresh: float = 0.5,
) -> FaceRecognizer:
    """Return a process-local :class:`FaceRecognizer`, creating it on first use.

    If ``models_dir``/``device`` are not supplied they are read from the app
    :class:`~app.config.Settings` when available, with sane fallbacks so this
    module stays importable in isolation (e.g. unit tests).
    """
    global _RECOGNIZER
    if _RECOGNIZER is not None:
        return _RECOGNIZER
    with _RECOGNIZER_LOCK:
        if _RECOGNIZER is not None:
            return _RECOGNIZER

        if models_dir is None or device is None:
            try:
                from app.config import settings  # noqa: WPS433

                if models_dir is None:
                    models_dir = getattr(settings, "models_dir", "models")
                if device is None:
                    device = getattr(settings, "face_device", None) or getattr(
                        settings, "device", "cuda"
                    )
                pack_name = getattr(settings, "face_pack", pack_name)
                det_thresh = float(getattr(settings, "face_det_thresh", det_thresh))
            except Exception:  # pragma: no cover - config optional in tests
                models_dir = models_dir or "models"
                device = device or "cuda"

        _RECOGNIZER = FaceRecognizer(
            models_dir=models_dir,
            pack_name=pack_name,
            device=device or "cuda",
            det_thresh=det_thresh,
        )
        return _RECOGNIZER

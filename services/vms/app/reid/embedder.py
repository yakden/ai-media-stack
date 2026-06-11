"""OSNet appearance Re-ID model wrapper (ONNX under onnxruntime-gpu).

Produces a 512-d L2-normalized *appearance* embedding (clothing + body shape)
from a full person crop. This is the clothing-variant counterpart to the
ArcFace face embedding: it works when the face is not visible (back turned,
masked) and links a person across cameras within a time window.

Model
-----
OSNet ``osnet_x0_25`` trained on MSMT17 (Torchreid zoo key
``osnet_x0_25_msmt17``), exported to ONNX by ``scripts/export_reid_onnx.py``.
The output is the 512-d global feature (the layer before the classifier),
which we L2-normalize so an ``IndexFlatIP`` / inner product yields cosine
similarity directly — the same convention as :mod:`app.faces.index`.

Input
-----
BGR person crop -> RGB, resize to 128x256 (WxH, the torchreid ReID standard),
scale to ``[0, 1]``, ImageNet-normalize (mean ``[0.485, 0.456, 0.406]``, std
``[0.229, 0.224, 0.225]``), NCHW. The ONNX session's declared input dtype is
honoured (an fp16 export takes a float16 tensor).

Design notes
------------
* GPU (CUDAExecutionProvider) is used when ``device`` is ``cuda``/``gpu`` and
  the CUDA provider is actually available in this onnxruntime build; otherwise
  we fall back to CPU. This keeps the worker alive when the box is GPU-contended
  — identical degradation to the YOLO/face wrappers.
* The embedder is process-local and lazily initialized via
  :func:`get_embedder` so each camera worker subprocess gets its own
  onnxruntime session (CUDA contexts are not fork-safe to share).
* The ONNX model is exported with a dynamic batch axis; :meth:`embed_batch`
  packs several per-frame crops into a single inference call to amortize
  launch overhead on the T4.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
DEFAULT_MODEL = "osnet_x0_25_msmt17.onnx"
# torchreid ReID standard input size (W, H). OSNet was trained at 128x256.
DEFAULT_INPUT_W = 128
DEFAULT_INPUT_H = 256

# ImageNet normalization (torchreid default for OSNet).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


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
            "REID device=%s requested but CUDAExecutionProvider unavailable "
            "(have %s); falling back to CPU.",
            device,
            sorted(available),
        )
    return ["CPUExecutionProvider"]


# --- crop quality ------------------------------------------------------------


def crop_quality(crop: np.ndarray) -> dict:
    """Compute cheap quality stats for a candidate person crop.

    Returns a dict with:
      * ``ok``        — bool, all gates passed (usable for matching/exemplars)
      * ``area``      — int pixel area (w*h)
      * ``width``     — int
      * ``height``    — int
      * ``aspect``    — float height/width (a standing/sitting body is tall)
      * ``blur``      — float variance-of-Laplacian sharpness (higher = sharper)

    The thresholds themselves live in config / the matcher; this helper just
    reports the raw measurements plus a conservative ``ok`` flag using sane
    defaults so it is useful standalone (e.g. in tests). Mirrors the
    self-contained style of :mod:`app.faces.recognizer`.
    """
    out = {
        "ok": False,
        "area": 0,
        "width": 0,
        "height": 0,
        "aspect": 0.0,
        "blur": 0.0,
    }
    if crop is None or getattr(crop, "size", 0) == 0:
        return out
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return out
    out["width"] = int(w)
    out["height"] = int(h)
    out["area"] = int(w * h)
    out["aspect"] = float(h) / float(w)
    out["blur"] = _blur_score(crop)
    return out


def _blur_score(crop: np.ndarray) -> float:
    """Variance of the Laplacian — a standard, cheap sharpness proxy.

    Falls back to a NumPy gradient-energy estimate if OpenCV is unavailable so
    the helper never hard-depends on cv2.
    """
    try:
        import cv2  # noqa: WPS433

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        arr = crop.astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        gx = np.diff(arr, axis=1)
        gy = np.diff(arr, axis=0)
        return float(gx.var() + gy.var())


def is_quality_crop(
    crop: np.ndarray,
    frame_area: Optional[int] = None,
    min_area_frac: float = 0.01,
    min_aspect: float = 1.0,
    min_blur: float = 0.0,
) -> bool:
    """Gate a crop for matching / new-identity creation.

    * ``min_area_frac`` — crop area must be >= this fraction of ``frame_area``
      (skipped when ``frame_area`` is None).
    * ``min_aspect``    — height/width must be >= this (reject slivers; a
      standing/sitting body is taller than it is wide).
    * ``min_blur``      — variance-of-Laplacian must be >= this (reject smears).
    """
    q = crop_quality(crop)
    if q["area"] <= 0:
        return False
    if q["aspect"] < float(min_aspect):
        return False
    if min_blur > 0.0 and q["blur"] < float(min_blur):
        return False
    if frame_area and min_area_frac > 0.0:
        if q["area"] < float(min_area_frac) * float(frame_area):
            return False
    return True


class ReIDEmbedder:
    """OSNet appearance embedder.

    Parameters
    ----------
    model_path:
        Path to the exported OSNet ONNX file.
    input_w, input_h:
        Network input size (W, H). Defaults to the torchreid 128x256.
    device:
        ``"cuda"`` or ``"cpu"``. Falls back to CPU automatically if CUDA is
        unavailable.
    embedding_dim:
        Expected output dimensionality (512 for OSNet global feature). Used as
        a sanity check on the model's output.
    """

    def __init__(
        self,
        model_path: str,
        input_w: int = DEFAULT_INPUT_W,
        input_h: int = DEFAULT_INPUT_H,
        device: str = "cuda",
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        self.model_path = str(model_path)
        self.input_w = int(input_w)
        self.input_h = int(input_h)
        self.device = device
        self.embedding_dim = int(embedding_dim)
        self._session = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._np_dtype = np.float32
        self._lock = threading.Lock()

    # -- session lifecycle ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort  # noqa: WPS433 (lazy: keep import light)

            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"OSNet Re-ID ONNX model not found at {self.model_path}. "
                    "Run scripts/download_models.py (or scripts/export_reid_onnx.py) "
                    "to obtain it."
                )

            providers = _resolve_providers(self.device)
            so = ort.SessionOptions()
            so.log_severity_level = 3  # warnings+; keep worker logs quiet
            session = ort.InferenceSession(
                self.model_path, sess_options=so, providers=providers
            )

            active = session.get_providers()
            self.device = "cuda" if "CUDAExecutionProvider" in active else "cpu"

            inp = session.get_inputs()[0]
            self._input_name = inp.name
            # fp16 export -> float16 input tensor.
            self._np_dtype = np.float16 if "float16" in str(inp.type) else np.float32
            # Honour a baked-in spatial input shape if the export fixed one.
            shape = inp.shape  # e.g. [batch, 3, 256, 128] (NCHW, H then W)
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_h = int(shape[2])
                self.input_w = int(shape[3])
            self._output_name = session.get_outputs()[0].name
            self._session = session
            logger.info(
                "OSNet Re-ID ONNX loaded (providers=%s, device=%s, input=%dx%d, dtype=%s)",
                active,
                self.device,
                self.input_w,
                self.input_h,
                self._np_dtype.__name__,
            )

    # -- preprocessing -------------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR crop -> normalized CHW float tensor (no batch dim)."""
        import cv2  # noqa: WPS433

        # Resize to (W, H). cv2.resize takes (width, height).
        resized = cv2.resize(
            crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        # BGR -> RGB, [0,1], ImageNet normalize (HWC).
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC -> CHW.
        chw = np.transpose(rgb, (2, 0, 1))
        return np.ascontiguousarray(chw, dtype=self._np_dtype)

    def _postprocess(self, raw: np.ndarray) -> np.ndarray:
        """A single raw output row -> L2-normalized float32 embedding."""
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.embedding_dim:
            logger.warning(
                "Unexpected Re-ID embedding dim %d (expected %d); using as-is.",
                vec.shape[0],
                self.embedding_dim,
            )
        return l2_normalize(vec)

    # -- inference -----------------------------------------------------------

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Return a 512-d L2-normalized appearance embedding for a BGR crop.

        Returns ``None`` for an empty/degenerate crop.
        """
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return None
        self._ensure_loaded()
        assert self._session is not None and self._input_name is not None

        blob = self._preprocess(crop)[None]  # add batch dim
        outputs = self._session.run([self._output_name], {self._input_name: blob})
        return self._postprocess(outputs[0][0])

    def embed_batch(
        self, crops: Sequence[np.ndarray]
    ) -> list[Optional[np.ndarray]]:
        """Embed several crops in one inference call (amortizes launch cost).

        Returns a list aligned with ``crops``; entries for empty/degenerate
        crops are ``None``. If every crop is invalid, returns all ``None``
        without touching the GPU.
        """
        if not crops:
            return []

        valid_idx: list[int] = []
        blobs: list[np.ndarray] = []
        for i, crop in enumerate(crops):
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            h, w = crop.shape[:2]
            if h <= 0 or w <= 0:
                continue
            valid_idx.append(i)

        results: list[Optional[np.ndarray]] = [None] * len(crops)
        if not valid_idx:
            return results

        self._ensure_loaded()
        assert self._session is not None and self._input_name is not None

        for i in valid_idx:
            blobs.append(self._preprocess(crops[i]))
        batch = np.ascontiguousarray(np.stack(blobs, axis=0), dtype=self._np_dtype)
        outputs = self._session.run([self._output_name], {self._input_name: batch})
        out_mat = outputs[0]
        for slot, i in enumerate(valid_idx):
            results[i] = self._postprocess(out_mat[slot])
        return results


# --- process-local singleton -------------------------------------------------

_EMBEDDER: Optional[ReIDEmbedder] = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(
    model_path: Optional[str] = None,
    input_w: Optional[int] = None,
    input_h: Optional[int] = None,
    device: Optional[str] = None,
    embedding_dim: int = EMBEDDING_DIM,
) -> ReIDEmbedder:
    """Return a process-local :class:`ReIDEmbedder`, creating it on first use.

    If arguments are not supplied they are read from the app
    :class:`~app.config.Settings` when available, with sane fallbacks so this
    module stays importable in isolation (e.g. unit tests).
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER

        if model_path is None or device is None or input_w is None or input_h is None:
            try:
                from app.config import settings  # noqa: WPS433

                if model_path is None:
                    rp = getattr(settings, "reid_model_path", None)
                    model_path = str(rp) if rp is not None else DEFAULT_MODEL
                if device is None:
                    device = getattr(settings, "reid_device", None) or getattr(
                        settings, "device", "cuda"
                    )
                if input_w is None:
                    input_w = int(getattr(settings, "reid_input_w", DEFAULT_INPUT_W))
                if input_h is None:
                    input_h = int(getattr(settings, "reid_input_h", DEFAULT_INPUT_H))
                embedding_dim = int(
                    getattr(settings, "reid_embedding_dim", embedding_dim)
                )
            except Exception:  # pragma: no cover - config optional in tests
                model_path = model_path or DEFAULT_MODEL
                device = device or "cuda"
                input_w = input_w or DEFAULT_INPUT_W
                input_h = input_h or DEFAULT_INPUT_H

        _EMBEDDER = ReIDEmbedder(
            model_path=model_path or DEFAULT_MODEL,
            input_w=int(input_w or DEFAULT_INPUT_W),
            input_h=int(input_h or DEFAULT_INPUT_H),
            device=device or "cuda",
            embedding_dim=embedding_dim,
        )
        return _EMBEDDER

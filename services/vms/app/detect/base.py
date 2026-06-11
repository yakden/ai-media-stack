"""Detector abstraction and factory.

The worker only ever needs to know how to turn a BGR frame (``numpy.ndarray``)
into a list of :class:`Box` results. Concrete backends (YOLOv8n ONNX,
DeepStream) implement :class:`Detector`; :func:`create_detector` picks one from
the ``DETECTOR_BACKEND`` setting.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger("vms.detect")

# COCO class id for "person".
PERSON_CLASS_ID = 0

# The 80 COCO classes, in the canonical order the YOLOv8 export emits them
# (output channel ``4 + class_id``). The UI lets the user pick any subset of
# these as per-camera recording triggers; the detector labels boxes from here.
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# name -> class id, for resolving user-selected trigger classes.
COCO_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}


def class_name(class_id: int) -> str:
    """Return the COCO label for a class id (or ``classN`` if out of range)."""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class{class_id}"


@dataclass(slots=True)
class Box:
    """A single detection in absolute pixel coordinates of the source frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "person"
    class_id: int = PERSON_CLASS_ID

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class Detector(ABC):
    """Abstract person detector.

    Implementations must be safe to construct inside a freshly spawned
    subprocess (no shared CUDA context across ``fork``) and must be cheap to
    call repeatedly on a hot loop.
    """

    #: Human-readable backend name, surfaced via /api/system.
    backend: str = "base"
    #: Effective device the detector ended up running on ("cuda" or "cpu").
    device: str = "cpu"

    @abstractmethod
    def detect(self, frame: "np.ndarray") -> list[Box]:
        """Return person detections for a single BGR frame."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release any held resources (GPU sessions, sockets)."""


def create_detector(settings: Any | None = None) -> Detector:
    """Build the detector selected by ``DETECTOR_BACKEND``.

    Parameters
    ----------
    settings:
        A ``Settings`` instance (or anything exposing the relevant attributes).
        When ``None`` the global settings are loaded lazily. We read attributes
        defensively with ``getattr`` so this component stays decoupled from the
        exact field names chosen by the config component.
    """

    if settings is None:
        try:
            from ..config import get_settings  # local import: avoids cycles

            settings = get_settings()
        except Exception:  # pragma: no cover - config not yet wired
            settings = object()

    backend = str(getattr(settings, "detector_backend", "yolo") or "yolo").lower()
    conf = float(getattr(settings, "detect_conf", 0.4) or 0.4)
    iou = float(getattr(settings, "detect_iou", 0.45) or 0.45)
    imgsz = int(getattr(settings, "detect_imgsz", 640) or 640)
    device = str(getattr(settings, "detector_device", getattr(settings, "device", "cuda")) or "cuda")

    if backend in ("deepstream", "ds"):
        from .deepstream_client import DeepStreamDetector

        endpoint = str(getattr(settings, "deepstream_endpoint", "") or "")
        logger.info("Detector backend: deepstream (endpoint=%s)", endpoint or "<unset>")
        return DeepStreamDetector(endpoint=endpoint, conf=conf)

    # Default / fallback: YOLOv8n ONNX.
    from .yolo_onnx import YoloOnnxDetector

    model_path = getattr(settings, "yolo_model_path", None)
    if model_path is None:
        # Fall back to <models_dir>/yolov8n.onnx.
        from pathlib import Path

        models_dir = Path(str(getattr(settings, "models_dir", "models")))
        model_path = models_dir / "yolov8n.onnx"

    logger.info("Detector backend: yolo_onnx (model=%s, device=%s)", model_path, device)
    return YoloOnnxDetector(
        model_path=str(model_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
    )

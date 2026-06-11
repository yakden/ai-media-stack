"""Vehicle attribute classifiers (NVIDIA TAO VehicleMakeNet / VehicleTypeNet).

Secondary classifiers that run on a detected vehicle crop and return its make
(brand) and body type. The ONNX models ship with NVIDIA DeepStream
(``Secondary_VehicleMake`` / ``Secondary_VehicleTypes``); we run them under
onnxruntime (already a dependency) so no DeepStream runtime is needed.

Preprocessing matches the DeepStream nvinfer config for these models:
``net-scale-factor=1``, ``offsets`` unset (0), ``model-color-format=1`` (BGR) —
i.e. a plain resize to 224x224 of the BGR crop, raw 0..255 values, NCHW.

NOTE: these give *brand* and *body type* only. Exact model and year are not
predictable by any TAO pretrained model and need a custom-trained classifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("vms.detect.vehicle_attrs")

INPUT_H = 224
INPUT_W = 224

# COCO classes for which vehicle make/type classification is meaningful.
VEHICLE_CLASSES = {"car", "truck", "bus"}


def _read_labels(path: Path) -> list[str]:
    """TAO label files are a single ';'-separated line."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    sep = ";" if ";" in text else "\n"
    return [t.strip() for t in text.split(sep) if t.strip()]


class _OnnxClassifier:
    """One TAO ResNet-18 softmax classifier over a fixed label set."""

    def __init__(self, model_path: str, labels: list[str], device: str = "cuda") -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.lower() != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.labels = labels

    def classify(self, crop_bgr: np.ndarray) -> tuple[Optional[str], float]:
        import cv2

        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0
        img = cv2.resize(crop_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        # BGR, raw 0..255, NCHW (net-scale-factor=1, offsets=0).
        blob = np.ascontiguousarray(
            img.transpose(2, 0, 1)[None].astype(np.float32)
        )
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        probs = np.asarray(out).reshape(-1)
        if probs.size == 0:
            return None, 0.0
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else f"class{idx}"
        return label, float(probs[idx])


class VehicleAttributeClassifier:
    """Make + body-type classifier pair for vehicle crops.

    Construct from the models dir (expects ``vehiclemakenet/`` and
    ``vehicletypenet/`` subdirs with the ``.onnx`` + ``labels.txt``). Missing
    models degrade gracefully (that attribute is simply absent).
    """

    def __init__(self, models_dir: str, device: str = "cuda", threshold: float = 0.51) -> None:
        self.threshold = float(threshold)
        self._make: Optional[_OnnxClassifier] = None
        self._type: Optional[_OnnxClassifier] = None
        root = Path(models_dir)

        make_dir = root / "vehiclemakenet"
        make_onnx = next(iter(make_dir.glob("*.onnx")), None)
        if make_onnx is not None:
            try:
                self._make = _OnnxClassifier(
                    str(make_onnx), _read_labels(make_dir / "labels.txt"), device
                )
                logger.info("VehicleMakeNet loaded (%d makes)", len(self._make.labels))
            except Exception:
                logger.exception("Failed to load VehicleMakeNet")

        type_dir = root / "vehicletypenet"
        type_onnx = next(iter(type_dir.glob("*.onnx")), None)
        if type_onnx is not None:
            try:
                self._type = _OnnxClassifier(
                    str(type_onnx), _read_labels(type_dir / "labels.txt"), device
                )
                logger.info("VehicleTypeNet loaded (%d types)", len(self._type.labels))
            except Exception:
                logger.exception("Failed to load VehicleTypeNet")

    @property
    def available(self) -> bool:
        return self._make is not None or self._type is not None

    def classify(self, crop_bgr: np.ndarray) -> dict:
        """Return ``{make, make_conf, type, type_conf}`` (keys present only when
        the model is loaded and clears the confidence threshold)."""
        out: dict = {}
        if self._make is not None:
            make, conf = self._make.classify(crop_bgr)
            if make is not None and conf >= self.threshold:
                out["make"], out["make_conf"] = make, conf
        if self._type is not None:
            vtype, conf = self._type.classify(crop_bgr)
            if vtype is not None and conf >= self.threshold:
                out["type"], out["type_conf"] = vtype, conf
        return out

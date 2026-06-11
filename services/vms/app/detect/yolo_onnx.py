"""YOLOv8n person detector running under onnxruntime.

Chosen over the ultralytics PyTorch package to keep the image small and VRAM
modest. Tries the CUDA execution provider first (fp16 friendly on the T4) and
falls back to CPU automatically -- both when ``device=cpu`` is requested and
when the CUDA provider is unavailable / the GPU is contended.

The exported model is assumed to be a standard ``ultralytics`` YOLOv8 export
(``yolo export model=yolov8n.pt format=onnx``), which produces a single output
of shape ``(1, 84, 8400)`` -- 4 box coords + 80 class scores per anchor, in
``xywh`` (center) format scaled to the network input (default 640x640).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import Box, Detector, class_name

logger = logging.getLogger("vms.detect.yolo")


class YoloOnnxDetector(Detector):
    backend = "yolo_onnx"

    def __init__(
        self,
        model_path: str,
        conf: float = 0.4,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cuda",
    ) -> None:
        import onnxruntime as ort  # imported here so the module loads even w/o ORT

        self.model_path = str(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLO ONNX model not found at {self.model_path}. "
                "Run scripts/download_models.py to fetch it."
            )

        providers = self._select_providers(ort, device)
        so = ort.SessionOptions()
        so.log_severity_level = 3  # warnings+; keep the worker logs quiet
        self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=providers)

        active = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in active else "cpu"
        logger.info("YOLOv8n ONNX loaded (providers=%s, device=%s)", active, self.device)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Detect expected dtype (fp16 export -> float16 input tensor).
        self._np_dtype = np.float16 if "float16" in str(inp.type) else np.float32
        # Honour a fixed spatial input shape if the export baked one in.
        shape = inp.shape  # e.g. [1, 3, 640, 640] or [1, 3, 'height', 'width']
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.imgsz = int(shape[2])
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _select_providers(ort, device: str) -> list:
        available = set(ort.get_available_providers())
        if device.lower() == "cpu":
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.warning("CUDAExecutionProvider unavailable; running YOLO on CPU")
        return ["CPUExecutionProvider"]

    # -- preprocessing ------------------------------------------------------

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize keeping aspect ratio, pad to a square ``imgsz`` canvas."""
        import cv2

        h, w = frame.shape[:2]
        s = self.imgsz
        scale = min(s / h, s / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        top = (s - nh) // 2
        left = (s - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, scale, left, top

    # -- inference ----------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Box]:
        if frame is None or frame.size == 0:
            return []

        canvas, scale, pad_x, pad_y = self._letterbox(frame)
        # BGR -> RGB, HWC -> CHW, normalize to [0,1], add batch dim.
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None], dtype=self._np_dtype)

        outputs = self.session.run([self.output_name], {self.input_name: blob})
        pred = outputs[0]

        boxes = self._postprocess(pred, scale, pad_x, pad_y)
        return boxes

    def _postprocess(self, pred: np.ndarray, scale: float, pad_x: int, pad_y: int) -> list[Box]:
        pred = np.asarray(pred, dtype=np.float32)
        # Squeeze batch dim -> (84, 8400) or (8400, 84). Normalize to (N, 84).
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1]:
            # (84, 8400) -> (8400, 84)
            pred = pred.transpose(1, 0)

        n_cols = pred.shape[1]
        n_classes = n_cols - 4
        if n_classes <= 0:
            return []

        # Multi-class: take the best class per anchor (argmax over class scores).
        # The worker decides which classes actually trigger recording, so we
        # surface every class the model knows about here.
        cls_scores = pred[:, 4:4 + n_classes]
        best_cls = np.argmax(cls_scores, axis=1)
        best_score = cls_scores[np.arange(cls_scores.shape[0]), best_cls]

        keep = best_score >= self.conf
        if not np.any(keep):
            return []

        cand = pred[keep]
        cand_scores = best_score[keep]
        cand_cls = best_cls[keep]

        # xywh (center) at network scale -> xyxy.
        cx, cy, bw, bh = cand[:, 0], cand[:, 1], cand[:, 2], cand[:, 3]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Per-class NMS so an overlapping car + person aren't suppressed into one.
        boxes: list[Box] = []
        for cid in np.unique(cand_cls):
            m = cand_cls == cid
            cls_xyxy = xyxy[m]
            cls_sc = cand_scores[m]
            for i in self._nms(cls_xyxy, cls_sc, self.iou):
                bx1, by1, bx2, by2 = cls_xyxy[i]
                # Undo letterbox: remove padding, then divide by scale.
                bx1 = (bx1 - pad_x) / scale
                by1 = (by1 - pad_y) / scale
                bx2 = (bx2 - pad_x) / scale
                by2 = (by2 - pad_y) / scale
                boxes.append(
                    Box(
                        x1=float(max(0.0, bx1)),
                        y1=float(max(0.0, by1)),
                        x2=float(max(0.0, bx2)),
                        y2=float(max(0.0, by2)),
                        score=float(cls_sc[i]),
                        label=class_name(int(cid)),
                        class_id=int(cid),
                    )
                )
        return boxes

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            order = order[1:][iou <= iou_thresh]
        return keep

    def close(self) -> None:
        # onnxruntime releases the session on GC; drop the reference explicitly.
        self.session = None  # type: ignore[assignment]

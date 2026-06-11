"""Person-detection backends.

A small abstraction (:class:`app.detect.base.Detector`) plus implementations:

* :mod:`app.detect.yolo_onnx` -- YOLOv8n exported to ONNX, run under
  onnxruntime (CUDA fp16, CPU fallback). The default MVP backend.
* :mod:`app.detect.deepstream_client` -- OPTIONAL high-throughput backend that
  consumes detection metadata produced by a DeepStream pipeline. Not required
  to build or run the MVP.

Use :func:`app.detect.base.create_detector` to obtain a detector chosen by the
``DETECTOR_BACKEND`` setting.
"""

from __future__ import annotations

from .base import Box, Detector, create_detector

__all__ = ["Box", "Detector", "create_detector"]

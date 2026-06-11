"""OPTIONAL DeepStream detection backend.

This is *not* required to build or run the MVP. It is wired so that setting
``DETECTOR_BACKEND=deepstream`` swaps the in-process ONNX detector for a thin
client that consumes person-detection metadata produced by a separate
DeepStream 7.1 pipeline (``nvcr.io/nvidia/deepstream:7.1-triton-multiarch``),
which benchmarks ~700 FPS for person detection + tracking on the T4.

Integration model (kept deliberately simple and decoupled): the DeepStream
container runs the nvinfer person detector + nvtracker and publishes per-frame
metadata as newline-delimited JSON over a small HTTP/TCP endpoint, keyed by a
camera/source id. Each line looks like::

    {"source": "<camera_id>", "frame": 1234, "objects": [
        {"label": "person", "confidence": 0.91,
         "bbox": [x1, y1, x2, y2], "object_id": 7}
    ]}

The VMS camera worker still reads frames from RTSP (for the live MJPEG slot and
for clip thumbnails), but delegates detection to the freshest metadata line for
its source. Because the two pipelines are not frame-locked, we always return the
most recent objects seen for this source -- adequate for event triggering at
human time-scales.

If the endpoint is unreachable, ``detect`` returns an empty list rather than
raising, so a misconfigured optional backend never crashes the worker loop.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .base import PERSON_CLASS_ID, Box, Detector

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger("vms.detect.deepstream")


class DeepStreamDetector(Detector):
    """Client for a DeepStream metadata stream.

    Parameters
    ----------
    endpoint:
        Base URL of the DeepStream metadata service, e.g.
        ``http://deepstream:8060``. The client polls
        ``<endpoint>/meta?source=<source>``-style by streaming the NDJSON body.
    source:
        The DeepStream source id this detector represents (set per camera by
        the worker via :meth:`bind_source`).
    conf:
        Minimum confidence to report a detection.
    """

    backend = "deepstream"
    device = "deepstream"

    def __init__(self, endpoint: str = "", source: str | None = None, conf: float = 0.4) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.source = source
        self.conf = float(conf)
        self._latest: list[Box] = []
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.endpoint and self.source is not None:
            self._start()

    def bind_source(self, source: str) -> None:
        """Bind this detector to a DeepStream source id and start polling."""
        self.source = str(source)
        if self.endpoint and self._thread is None:
            self._start()

    # -- background reader --------------------------------------------------

    def _start(self) -> None:
        self._thread = threading.Thread(
            target=self._reader_loop, name=f"ds-reader-{self.source}", daemon=True
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        url = self._meta_url()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._consume(url)
                backoff = 1.0
            except Exception as exc:  # network hiccup -> retry with backoff
                logger.warning("DeepStream metadata stream error (%s); retrying", exc)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 15.0)

    def _meta_url(self) -> str:
        parsed = urlparse(self.endpoint)
        if not parsed.scheme:
            base = f"http://{self.endpoint}"
        else:
            base = self.endpoint
        return f"{base}/meta?source={self.source}"

    def _consume(self, url: str) -> None:
        # Lazy import so the optional backend doesn't add a hard dependency.
        import urllib.request

        req = urllib.request.Request(url, headers={"Accept": "application/x-ndjson"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted internal)
            for raw in resp:
                if self._stop.is_set():
                    return
                line = raw.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._ingest(msg)

    def _ingest(self, msg: dict) -> None:
        if str(msg.get("source")) != str(self.source):
            return
        boxes: list[Box] = []
        for obj in msg.get("objects", []):
            label = str(obj.get("label", "")).lower()
            if label and label != "person":
                continue
            score = float(obj.get("confidence", 0.0))
            if score < self.conf:
                continue
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            boxes.append(
                Box(x1=x1, y1=y1, x2=x2, y2=y2, score=score, label="person", class_id=PERSON_CLASS_ID)
            )
        with self._lock:
            self._latest = boxes
            self._latest_ts = time.time()

    # -- Detector API -------------------------------------------------------

    def detect(self, frame: "np.ndarray | None" = None) -> list[Box]:  # noqa: D401
        """Return the most recent person detections for this source.

        The ``frame`` argument is accepted for interface compatibility but is
        ignored: detection happens inside the DeepStream pipeline.
        """
        # Stale metadata (no fresh frame in a while) -> report nothing.
        with self._lock:
            if time.time() - self._latest_ts > 5.0:
                return []
            return list(self._latest)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

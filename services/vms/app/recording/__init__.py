"""Recording subsystem: warm rolling segment buffer + clip assembly.

This package contains the *mechanics* of recording, with no knowledge of
cameras, detection, or the database. It is driven entirely by the per-camera
worker:

* ``Segmenter`` keeps a warm ``ffmpeg -f segment`` process alive per camera,
  continuously writing short ``.mp4`` segments to ``data/segments/<camera_id>/``.
  Because the buffer is always running, pre-roll footage is on disk the instant
  a person is detected — no need to start recording reactively (and miss the
  approach).

* ``clip_event`` (in :mod:`clipper`) is called on a person-trigger. It selects
  the segments covering ``[trigger - pre, trigger + post]``, stream-copies them
  (no re-encode) into a single ``<event_id>.mp4`` clip, and extracts a thumbnail
  JPEG.

Neither module touches the GPU; everything here is ffmpeg + filesystem.
"""

from __future__ import annotations

from .clipper import ClipResult, clip_event, extract_thumbnail
from .segmenter import Segmenter, SegmentInfo

__all__ = [
    "Segmenter",
    "SegmentInfo",
    "clip_event",
    "extract_thumbnail",
    "ClipResult",
]

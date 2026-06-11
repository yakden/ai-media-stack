"""Camera workers: per-camera detection processes and their manager.

The :class:`WorkerManager` (see :mod:`app.workers.manager`) spawns one
:func:`app.workers.camera_worker.run_worker` subprocess per enabled camera. It
keeps a status registry and a shared latest-annotated-frame slot per camera
that the live MJPEG/snapshot endpoints read from.
"""

from __future__ import annotations

from .manager import WorkerManager, WorkerHandle

__all__ = ["WorkerManager", "WorkerHandle"]

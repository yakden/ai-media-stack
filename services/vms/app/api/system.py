"""System / introspection endpoints: detailed worker, model, and backend info.

``/health`` itself lives in ``main.py`` (no auth, used by the control-plane).
This router exposes the richer authenticated ``/api/system`` view.
"""

from __future__ import annotations

import logging
import sys

from fastapi import APIRouter, Depends, Request

from .. import __version__
from ..auth import Principal, require_user
from ..config import Settings, get_settings

logger = logging.getLogger("vms.system")

router = APIRouter(prefix="/api", tags=["system"])


def gpu_memory() -> dict:
    """Best-effort GPU memory snapshot via NVML; empty/zeros if unavailable."""
    info = {"available": False, "used_mb": 0, "total_mb": 0, "free_mb": 0}
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info.update(
                available=True,
                used_mb=int(mem.used / 1_048_576),
                total_mb=int(mem.total / 1_048_576),
                free_mb=int(mem.free / 1_048_576),
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # NVML not installed or no GPU visible — report unavailable, never raise.
        pass
    return info


def worker_states(request: Request) -> list[dict]:
    """Read the WorkerManager status registry off app.state, if it booted."""
    manager = getattr(request.app.state, "workers", None)
    if manager is None:
        return []
    try:
        # WorkerManager.all_status() -> {camera_id: {state, fps, last_seen, ...}}.
        status = manager.all_status()
        return [
            {"camera_id": cid, **(v if isinstance(v, dict) else {"state": v})}
            for cid, v in status.items()
        ]
    except Exception:  # pragma: no cover - defensive against partial wiring
        logger.exception("Failed to read worker status")
        return []


def onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort  # type: ignore

        return list(ort.get_available_providers())
    except Exception:
        return []


@router.get("/system")
def system_info(
    request: Request,
    settings: Settings = Depends(get_settings),
    user: Principal = Depends(require_user),
) -> dict:
    faces = getattr(request.app.state, "face_index", None)
    faces_loaded = faces is not None
    faces_count = None
    if faces_loaded:
        try:
            faces_count = faces.size()  # FaceIndex is expected to expose size()
        except Exception:
            faces_count = None

    return {
        "service": "vms",
        "version": __version__,
        "python": sys.version.split()[0],
        "user": {"email": user.email, "name": user.user, "via": user.via},
        "backend": {
            "detector": settings.detector_backend,
            "device": settings.device,
            "onnx_providers": onnx_providers(),
            "faces_enabled": settings.faces_enabled,
        },
        "models": {
            "yolo": str(settings.yolo_model_path),
            "insightface_pack": settings.insightface_pack,
        },
        "faces": {"index_loaded": faces_loaded, "count": faces_count},
        "gpu": gpu_memory(),
        "workers": worker_states(request),
    }

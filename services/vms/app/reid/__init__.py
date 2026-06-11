"""Automatic cross-camera person Re-Identification (Re-ID) subpackage.

This package adds an *appearance* (body/clothing) embedding alongside the
existing ArcFace face embedding, plus the identity gallery + online clustering
that links sightings of the same person across cameras and over time.

Modules
-------
embedder:
    :class:`~app.reid.embedder.ReIDEmbedder` — OSNet appearance model wrapper
    (ONNX under onnxruntime-gpu) producing 512-d L2-normalized body embeddings,
    plus crop-quality helpers. Mirrors :mod:`app.faces.recognizer`.

The matching/gallery/maintenance/API/UI pieces live in sibling modules owned by
other components; this ``__init__`` keeps the package importable without pulling
in the heavy onnxruntime stack (the embedder imports it lazily).
"""

from __future__ import annotations

__all__ = ["ReIDEmbedder", "get_embedder", "crop_quality", "is_quality_crop"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export shim
    # Lazily forward the common embedder symbols so ``from app.reid import
    # ReIDEmbedder`` works without importing onnxruntime/opencv at package
    # import time (matches how the worker imports these lazily).
    if name in __all__:
        from . import embedder as _embedder

        return getattr(_embedder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

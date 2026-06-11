"""Face recognition & index package.

Provides:
  - ``FaceRecognizer``: insightface SCRFD detector + ArcFace embedder wrapper
    that turns a BGR frame into a list of detected faces with 512-d
    L2-normalized embeddings.
  - ``FaceIndex``: a FAISS ``IndexFlatIP`` over those normalized embeddings
    (inner product == cosine similarity) that is rebuilt from the SQLite
    ``face_embeddings`` table on startup and kept in sync on add/remove.

The DB is the single source of truth; the FAISS index is derived state.
"""

from __future__ import annotations

from .index import (
    EMBEDDING_DIM,
    FaceIndex,
    MatchResult,
    deserialize_vector,
    serialize_vector,
)
from .recognizer import DetectedFace, FaceRecognizer, get_recognizer

__all__ = [
    "EMBEDDING_DIM",
    "FaceIndex",
    "MatchResult",
    "DetectedFace",
    "FaceRecognizer",
    "get_recognizer",
    "serialize_vector",
    "deserialize_vector",
]

"""FAISS-backed face index (cosine similarity over normalized ArcFace vectors).

The index is *derived state*: the SQLite ``face_embeddings`` table is the
single source of truth. On startup we stream every stored vector into a
``faiss.IndexFlatIP`` (inner product == cosine similarity because vectors are
L2-normalized) and remember a parallel mapping from FAISS row position ->
``(embedding_id, person_id)``.

FlatIP does not support stable per-vector deletion, so removals rebuild the
index from the current set of embeddings. The index is tiny (hundreds of
people), so a full rebuild is cheap and avoids id-bookkeeping bugs.

Thread-safety: a single ``RLock`` guards all mutating/searching operations so
the API process (enrollment, deletion) and any in-process matching share the
index safely. (Camera worker subprocesses build their own index instance.)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512


# --- (de)serialization helpers ----------------------------------------------

def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a 512-d float32 L2-normalized vector to little-endian bytes
    for storage in the ``face_embeddings.vector`` BLOB column."""
    arr = np.asarray(vec, dtype="<f4").reshape(-1)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EMBEDDING_DIM}-d vector, got shape {arr.shape}"
        )
    return arr.tobytes()


def deserialize_vector(blob: bytes) -> np.ndarray:
    """Inverse of :func:`serialize_vector`."""
    arr = np.frombuffer(blob, dtype="<f4")
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"BLOB decodes to {arr.shape[0]} floats, expected {EMBEDDING_DIM}"
        )
    # Copy out of the read-only buffer so callers can mutate freely.
    return np.array(arr, dtype=np.float32)


def _normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return (vec / norm).astype(np.float32)


@dataclass
class MatchResult:
    """Result of a face match against the index."""

    embedding_id: int
    person_id: int
    score: float  # cosine similarity in [-1, 1]

    @property
    def matched(self) -> bool:
        return self.person_id is not None


class FaceIndex:
    """In-memory FAISS ``IndexFlatIP`` over L2-normalized 512-d embeddings.

    Parameters
    ----------
    dim:
        Embedding dimensionality (512 for ArcFace r50).
    match_threshold:
        Minimum cosine similarity for :meth:`search` to consider a hit a match
        (used by :meth:`match`). Typical buffalo_l ArcFace threshold ~0.35-0.5.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, match_threshold: float = 0.4) -> None:
        self.dim = dim
        self.match_threshold = float(match_threshold)
        self._lock = threading.RLock()
        self._index = self._new_index()
        # FAISS row position -> stored embedding metadata.
        self._emb_ids: list[int] = []
        self._person_ids: list[int] = []

    # -- construction ---------------------------------------------------------

    def _new_index(self):
        import faiss  # noqa: WPS433 (lazy import keeps module import light)

        return faiss.IndexFlatIP(self.dim)

    @classmethod
    def from_db(
        cls,
        session,
        dim: int = EMBEDDING_DIM,
        match_threshold: float = 0.4,
    ) -> "FaceIndex":
        """Build a fresh index by streaming all ``FaceEmbedding`` rows from the
        given SQLAlchemy session. Assigns ``faiss_id`` to each row to reflect
        its position in the rebuilt index (caller should commit the session)."""
        idx = cls(dim=dim, match_threshold=match_threshold)
        idx.rebuild_from_db(session)
        return idx

    # -- mutation -------------------------------------------------------------

    def rebuild_from_db(self, session) -> int:
        """Drop and rebuild the index from the DB. Updates each embedding's
        ``faiss_id`` to its new row position. Returns the number of vectors
        loaded. The caller is responsible for committing the session."""
        from app.db.models import FaceEmbedding  # noqa: WPS433

        rows = (
            session.query(FaceEmbedding)
            .order_by(FaceEmbedding.id.asc())
            .all()
        )
        with self._lock:
            self._index = self._new_index()
            self._emb_ids = []
            self._person_ids = []
            vectors: list[np.ndarray] = []
            for pos, row in enumerate(rows):
                try:
                    vec = deserialize_vector(row.vector)
                except Exception:
                    logger.exception(
                        "Skipping FaceEmbedding id=%s with bad vector blob", row.id
                    )
                    continue
                vec = _normalize(vec)
                vectors.append(vec)
                self._emb_ids.append(int(row.id))
                self._person_ids.append(int(row.person_id))
                row.faiss_id = len(self._emb_ids) - 1
            if vectors:
                mat = np.vstack(vectors).astype(np.float32)
                self._index.add(mat)
            logger.info("FaceIndex rebuilt: %d vectors loaded.", len(self._emb_ids))
            return len(self._emb_ids)

    def add(self, embedding_id: int, person_id: int, vector: np.ndarray) -> int:
        """Append a single embedding to the index. Returns its FAISS row
        position (which the caller may persist as ``faiss_id``)."""
        vec = _normalize(vector)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim}-d vector, got {vec.shape[0]}")
        with self._lock:
            self._index.add(vec.reshape(1, -1).astype(np.float32))
            self._emb_ids.append(int(embedding_id))
            self._person_ids.append(int(person_id))
            return len(self._emb_ids) - 1

    def remove(self, session, embedding_id: int) -> bool:
        """Remove an embedding by id and rebuild the index from the DB.

        Note: this relies on the row already having been deleted from the DB
        (or about to be) — it rebuilds from whatever the session currently
        sees. For correctness, delete the ``FaceEmbedding`` row first, then
        call ``rebuild_from_db``; this helper is provided for symmetry and
        simply rebuilds. Returns True if the id was present before rebuild."""
        with self._lock:
            present = embedding_id in self._emb_ids
            self.rebuild_from_db(session)
            return present

    def clear(self) -> None:
        with self._lock:
            self._index = self._new_index()
            self._emb_ids = []
            self._person_ids = []

    # -- query ----------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._emb_ids)

    def search(
        self, vector: np.ndarray, k: int = 5
    ) -> list[MatchResult]:
        """Return up to ``k`` nearest neighbors by cosine similarity, sorted
        descending. Does not apply the match threshold."""
        with self._lock:
            n = len(self._emb_ids)
            if n == 0:
                return []
            vec = _normalize(vector).reshape(1, -1).astype(np.float32)
            k_eff = min(k, n)
            scores, idxs = self._index.search(vec, k_eff)
            out: list[MatchResult] = []
            for score, pos in zip(scores[0], idxs[0]):
                if pos < 0:  # FAISS pads with -1 when fewer than k results
                    continue
                out.append(
                    MatchResult(
                        embedding_id=self._emb_ids[pos],
                        person_id=self._person_ids[pos],
                        score=float(score),
                    )
                )
            return out

    def match(
        self, vector: np.ndarray, threshold: Optional[float] = None
    ) -> Optional[MatchResult]:
        """Return the single best match if its cosine similarity is at or above
        the threshold, else ``None``."""
        thr = self.match_threshold if threshold is None else float(threshold)
        hits = self.search(vector, k=1)
        if not hits:
            return None
        best = hits[0]
        if best.score >= thr:
            return best
        return None

    def match_many(
        self,
        vectors: Sequence[np.ndarray] | Iterable[np.ndarray],
        threshold: Optional[float] = None,
    ) -> list[Optional[MatchResult]]:
        """Match a batch of query vectors; returns a list aligned with input,
        each element a :class:`MatchResult` or ``None``."""
        return [self.match(v, threshold=threshold) for v in vectors]

"""In-memory identity gallery: derived state over the identity tables.

The SQLite ``identities`` / ``face_exemplars`` / ``appearance_exemplars``
tables are the single source of truth (mirroring the ``FaceIndex`` /
``face_embeddings`` contract). The gallery is *derived state* rebuilt from the
DB at startup and re-synced on a timer:

  * FACE side: a FAISS ``IndexFlatIP`` over every per-identity ArcFace exemplar
    (inner product == cosine, vectors L2-normalized). Each FAISS row maps back
    to ``(exemplar_id, identity_id)``. Faces are time-stable -> not decayed.
  * APPEARANCE side: a per-identity list of OSNet exemplars with capture
    timestamps. Appearance scoring is time-decayed (see :mod:`app.reid.decay`),
    so we keep the raw vectors + timestamps in memory and let the manager apply
    the decay/window logic at query time. We also keep a small bookkeeping of
    each identity's freshest appearance timestamp + camera ids for the
    time/space window gate.

Thread-safety mirrors :class:`app.faces.index.FaceIndex`: a single ``RLock``
guards all mutating/searching operations so the API process and any in-process
matching share the gallery safely. (Camera worker subprocesses build their own
gallery instance and reload from the shared DB on a timer.)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from .decay import effective_score, max_decay_weight

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512


# --- (de)serialization helpers (same little-endian f32 contract as faces) ----

def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a 512-d float32 L2-normalized vector to little-endian bytes."""
    arr = np.asarray(vec, dtype="<f4").reshape(-1)
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-d vector, got shape {arr.shape}")
    return arr.tobytes()


def deserialize_vector(blob: bytes) -> np.ndarray:
    """Inverse of :func:`serialize_vector` (returns a writable copy)."""
    arr = np.frombuffer(blob, dtype="<f4")
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"BLOB decodes to {arr.shape[0]} floats, expected {EMBEDDING_DIM}"
        )
    return np.array(arr, dtype=np.float32)


def normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm as float32."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return v
    return (v / norm).astype(np.float32)


@dataclass
class FaceHit:
    """A nearest face-exemplar match resolved to its owning identity."""

    identity_id: int
    exemplar_id: int
    score: float  # cosine similarity in [-1, 1]


@dataclass
class AppearanceHit:
    """A decay-weighted appearance match to an identity."""

    identity_id: int
    score: float  # decay-weighted cosine
    raw_score: float  # un-decayed cosine of the best exemplar


@dataclass
class _AppearanceStore:
    """In-memory appearance exemplars for one identity."""

    vectors: list[np.ndarray] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    camera_ids: list[Optional[int]] = field(default_factory=list)

    def add(self, vec: np.ndarray, ts: datetime, camera_id: Optional[int]) -> None:
        self.vectors.append(normalize(vec))
        self.timestamps.append(ts)
        self.camera_ids.append(camera_id)

    @property
    def cameras_seen(self) -> set[int]:
        return {c for c in self.camera_ids if c is not None}


class IdentityGallery:
    """Derived face FAISS index + appearance store over the identity tables.

    Parameters
    ----------
    dim:
        Embedding dimensionality (512 for both ArcFace and OSNet x0_25).
    app_window_seconds:
        Temporal window for appearance-only candidacy: an identity is an
        appearance candidate only if its freshest appearance exemplar is within
        this window of the query time (bounds "teleportation" across cameras).
    app_decay_tau_seconds:
        Time-decay constant for appearance scoring.
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        app_window_seconds: float = 600.0,
        app_decay_tau_seconds: float = 43_200.0,
        settings: object | None = None,
    ) -> None:
        # Callers (camera_worker, main) construct with settings=...; pull tunables
        # from it via getattr (keeps explicit kwargs working too).
        if settings is not None:
            dim = int(getattr(settings, "reid_embedding_dim", dim))
            app_window_seconds = float(getattr(settings, "reid_app_window_seconds", app_window_seconds))
            app_decay_tau_seconds = float(getattr(settings, "reid_app_decay_tau_seconds", app_decay_tau_seconds))
        self.dim = dim
        self.app_window_seconds = float(app_window_seconds)
        self.app_decay_tau_seconds = float(app_decay_tau_seconds)
        self._lock = threading.RLock()
        # Face FAISS index + parallel row -> (exemplar_id, identity_id) maps.
        self._face_index = self._new_index()
        self._face_exemplar_ids: list[int] = []
        self._face_identity_ids: list[int] = []
        # identity_id -> appearance exemplars.
        self._appearance: dict[int, _AppearanceStore] = {}
        # Cached set of all known identity ids (so the manager can mint
        # "Person N" without a DB round-trip on the hot path).
        self._identity_ids: set[int] = set()
        # identity_id -> object class ("person", "car", ...). Matching is scoped
        # to one class so different object types never merge into one identity.
        self._identity_class: dict[int, str] = {}
        # identity_id -> 12-bin hue histogram, the colour gate for matching.
        self._identity_color: dict[int, np.ndarray] = {}

    # -- construction ---------------------------------------------------------

    def _new_index(self):
        import faiss  # noqa: WPS433 (lazy import keeps module import light)

        return faiss.IndexFlatIP(self.dim)

    @classmethod
    def from_db(
        cls,
        session,
        dim: int = EMBEDDING_DIM,
        app_window_seconds: float = 600.0,
        app_decay_tau_seconds: float = 43_200.0,
    ) -> "IdentityGallery":
        g = cls(
            dim=dim,
            app_window_seconds=app_window_seconds,
            app_decay_tau_seconds=app_decay_tau_seconds,
        )
        g.rebuild_from_db(session)
        return g

    # -- mutation -------------------------------------------------------------

    def rebuild_from_db(self, session) -> int:
        """Drop and rebuild the gallery from the DB. Returns the identity count.

        Streams every ``FaceExemplar`` into the face FAISS index and every
        ``AppearanceExemplar`` into the per-identity appearance store. Cheap to
        run on a timer (galleries are small: hundreds of identities, a handful
        of exemplars each)."""
        from app.db.models import (  # noqa: WPS433
            AppearanceExemplar,
            FaceExemplar,
            Identity,
        )

        with self._lock:
            self._face_index = self._new_index()
            self._face_exemplar_ids = []
            self._face_identity_ids = []
            self._appearance = {}
            self._identity_ids = set()
            self._identity_class = {}
            self._identity_color = {}

            import json as _json

            for ident in session.query(
                Identity.id, Identity.object_class, Identity.attributes
            ).all():
                iid = int(ident.id)
                self._identity_ids.add(iid)
                self._identity_class[iid] = str(ident.object_class or "person")
                if ident.attributes:
                    try:
                        hist = _json.loads(ident.attributes).get("hist")
                        if hist:
                            self._identity_color[iid] = np.asarray(hist, dtype=np.float32)
                    except Exception:
                        pass

            face_vectors: list[np.ndarray] = []
            face_rows = (
                session.query(FaceExemplar)
                .order_by(FaceExemplar.id.asc())
                .all()
            )
            for row in face_rows:
                try:
                    vec = normalize(deserialize_vector(row.vector))
                except Exception:
                    logger.exception(
                        "Skipping FaceExemplar id=%s with bad vector blob", row.id
                    )
                    continue
                face_vectors.append(vec)
                self._face_exemplar_ids.append(int(row.id))
                self._face_identity_ids.append(int(row.identity_id))
            if face_vectors:
                mat = np.vstack(face_vectors).astype(np.float32)
                self._face_index.add(mat)

            app_rows = (
                session.query(AppearanceExemplar)
                .order_by(AppearanceExemplar.id.asc())
                .all()
            )
            for row in app_rows:
                try:
                    vec = normalize(deserialize_vector(row.vector))
                except Exception:
                    logger.exception(
                        "Skipping AppearanceExemplar id=%s with bad vector blob",
                        row.id,
                    )
                    continue
                store = self._appearance.setdefault(
                    int(row.identity_id), _AppearanceStore()
                )
                store.add(vec, row.ts, row.camera_id)

            logger.info(
                "IdentityGallery rebuilt: %d identities, %d face exemplars, "
                "%d appearance exemplars.",
                len(self._identity_ids),
                len(self._face_exemplar_ids),
                len(app_rows),
            )
            return len(self._identity_ids)

    def add_face_exemplar(
        self, exemplar_id: int, identity_id: int, vector: np.ndarray
    ) -> int:
        """Append one face exemplar to the FAISS index. Returns its row pos."""
        vec = normalize(vector)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim}-d vector, got {vec.shape[0]}")
        with self._lock:
            self._face_index.add(vec.reshape(1, -1).astype(np.float32))
            self._face_exemplar_ids.append(int(exemplar_id))
            self._face_identity_ids.append(int(identity_id))
            self._identity_ids.add(int(identity_id))
            return len(self._face_exemplar_ids) - 1

    def add_appearance_exemplar(
        self,
        identity_id: int,
        vector: np.ndarray,
        ts: datetime,
        camera_id: Optional[int] = None,
    ) -> None:
        """Append one appearance exemplar to the in-memory store."""
        with self._lock:
            store = self._appearance.setdefault(int(identity_id), _AppearanceStore())
            store.add(vector, ts, camera_id)
            self._identity_ids.add(int(identity_id))

    def register_identity(self, identity_id: int, object_class: str = "person") -> None:
        """Record a freshly-created identity id (before any exemplar lands)."""
        with self._lock:
            self._identity_ids.add(int(identity_id))
            self._identity_class[int(identity_id)] = str(object_class or "person")

    def class_of(self, identity_id: int) -> str:
        """Return the object class of an identity (default 'person')."""
        with self._lock:
            return self._identity_class.get(int(identity_id), "person")

    def set_identity_color(self, identity_id: int, hist) -> None:
        """Record an identity's colour histogram (the matching colour gate)."""
        if hist is None:
            return
        with self._lock:
            self._identity_color[int(identity_id)] = np.asarray(hist, dtype=np.float32)

    def color_hist_of(self, identity_id: int):
        """Return an identity's colour histogram, or ``None``."""
        with self._lock:
            return self._identity_color.get(int(identity_id))

    def clear(self) -> None:
        with self._lock:
            self._face_index = self._new_index()
            self._face_exemplar_ids = []
            self._face_identity_ids = []
            self._appearance = {}
            self._identity_ids = set()

    # -- introspection --------------------------------------------------------

    @property
    def num_identities(self) -> int:
        with self._lock:
            return len(self._identity_ids)

    @property
    def num_face_exemplars(self) -> int:
        with self._lock:
            return len(self._face_exemplar_ids)

    def identity_ids(self) -> list[int]:
        with self._lock:
            return sorted(self._identity_ids)

    def face_centroid_for(self, identity_id: int) -> Optional[np.ndarray]:
        """L2-normalized mean of an identity's in-memory face exemplars.

        Used by the manager's face-contradiction veto (does identity B's face
        strongly disagree with a query face) without a DB round-trip."""
        with self._lock:
            vecs = [
                self._face_index.reconstruct(pos)
                for pos, iid in enumerate(self._face_identity_ids)
                if iid == identity_id
            ] if hasattr(self._face_index, "reconstruct") else None
        # IndexFlatIP supports reconstruct, but the test stub may not; fall back
        # to "no centroid" rather than erroring.
        if not vecs:
            return None
        mean = np.mean(np.vstack(vecs), axis=0)
        return normalize(mean)

    # -- query: faces ---------------------------------------------------------

    def search_faces(self, vector: np.ndarray, k: int = 8) -> list[FaceHit]:
        """Up to ``k`` nearest face exemplars by cosine, sorted descending,
        each resolved to its owning identity (no threshold applied)."""
        with self._lock:
            n = len(self._face_exemplar_ids)
            if n == 0:
                return []
            vec = normalize(vector).reshape(1, -1).astype(np.float32)
            k_eff = min(k, n)
            scores, idxs = self._face_index.search(vec, k_eff)
            out: list[FaceHit] = []
            for score, pos in zip(scores[0], idxs[0]):
                if pos < 0:
                    continue
                out.append(
                    FaceHit(
                        identity_id=self._face_identity_ids[pos],
                        exemplar_id=self._face_exemplar_ids[pos],
                        score=float(score),
                    )
                )
            return out

    def best_face_per_identity(
        self, vector: np.ndarray, k: int = 16, object_class: Optional[str] = None
    ) -> list[FaceHit]:
        """Like :meth:`search_faces` but collapsed to the best hit per identity,
        sorted descending. Lets the manager apply a margin test between the top
        two *distinct* identities rather than two exemplars of the same person.

        When ``object_class`` is given, only identities of that class are
        considered (faces only ever belong to people, but the filter keeps the
        contract uniform)."""
        hits = self.search_faces(vector, k=k)
        best: dict[int, FaceHit] = {}
        for h in hits:
            if object_class is not None and self._identity_class.get(h.identity_id, "person") != object_class:
                continue
            cur = best.get(h.identity_id)
            if cur is None or h.score > cur.score:
                best[h.identity_id] = h
        ordered = sorted(best.values(), key=lambda h: h.score, reverse=True)
        return ordered

    # -- query: appearance ----------------------------------------------------

    def appearance_candidates(
        self,
        vector: np.ndarray,
        now: datetime,
        same_camera_id: Optional[int] = None,
        object_class: Optional[str] = None,
    ) -> list[AppearanceHit]:
        """Decay-weighted appearance scores for identities inside the time window.

        Only identities whose freshest appearance exemplar is within
        ``app_window_seconds`` of ``now`` are considered. Each candidate's score
        is ``max_i(cosine_i * decay_w_i)`` per :func:`app.reid.decay.effective_score`.
        Sorted descending. ``same_camera_id`` is recorded so the caller can apply
        the stricter cross-camera bar; we do not filter on it here."""
        q = normalize(vector)
        out: list[AppearanceHit] = []
        with self._lock:
            for iid, store in self._appearance.items():
                if not store.vectors:
                    continue
                # Class scope: only match within the same object type.
                if object_class is not None and self._identity_class.get(iid, "person") != object_class:
                    continue
                # Window gate: the freshest exemplar must still carry weight
                # corresponding to app_window_seconds.
                fresh_w = max_decay_weight(
                    store.timestamps, now, self.app_decay_tau_seconds
                )
                window_w = max_decay_weight(
                    [now], now, self.app_decay_tau_seconds
                )  # == 1.0
                # Translate the hard time window into a min-weight floor.
                import math

                window_floor = math.exp(
                    -self.app_window_seconds / self.app_decay_tau_seconds
                )
                if fresh_w < window_floor:
                    continue
                cosines = [float(q @ v) for v in store.vectors]
                raw_best = max(cosines) if cosines else 0.0
                score = effective_score(
                    cosines, store.timestamps, now, self.app_decay_tau_seconds
                )
                out.append(
                    AppearanceHit(
                        identity_id=iid, score=score, raw_score=raw_best
                    )
                )
        out.sort(key=lambda h: h.score, reverse=True)
        return out

    def identity_cameras(self, identity_id: int) -> set[int]:
        """Cameras on which this identity has recent appearance exemplars."""
        with self._lock:
            store = self._appearance.get(int(identity_id))
            return set(store.cameras_seen) if store else set()

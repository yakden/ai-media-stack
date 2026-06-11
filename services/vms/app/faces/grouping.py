"""Unsupervised face grouping (clustering of similar faces).

Given the captured :class:`~app.db.models.FaceSample` rows (each an ArcFace
512-d embedding + optional OSNet appearance/clothing embedding), group them into
clusters of the *same person* by cosine similarity. This is the dedicated
face-recognition layer, separate from the online body-Re-ID identities.

Cross-angle robustness: ArcFace is fairly pose-tolerant, and because we capture
many samples per person across a track (different angles), single-linkage at a
tunable threshold lets different-angle crops of one person join the same group
transitively (A~B, B~C => one group) without needing every pair to match.

Clothing fusion: when ``clothing_weight > 0`` and both samples carry an
appearance vector, the pairwise score is
``(1-w)*face_cos + w*appearance_cos`` — clothing reinforces face and helps when
the face signal is weak (oblique/low-res), exactly the "плюс к одежде" ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FaceSampleVec:
    id: int
    vec: np.ndarray                 # L2-normed ArcFace 512
    app: Optional[np.ndarray]       # L2-normed OSNet 512 (clothing) or None
    quality: float
    camera_id: Optional[int]
    ts: object                      # datetime
    identity_id: Optional[int]
    label: Optional[str]
    thumb_path: Optional[str]


@dataclass
class FaceGroup:
    members: list = field(default_factory=list)   # list[FaceSampleVec]

    @property
    def size(self) -> int:
        return len(self.members)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_faces(
    samples: list[FaceSampleVec],
    face_threshold: float = 0.5,
    clothing_weight: float = 0.0,
    min_size: int = 1,
) -> list[FaceGroup]:
    """Cluster face samples into groups (single-linkage at ``face_threshold``).

    Parameters
    ----------
    face_threshold:
        Minimum fused cosine similarity to link two samples (0..1). Higher =>
        tighter, more groups; lower => looser, fewer/larger groups.
    clothing_weight:
        0..1 blend of appearance (clothing) cosine into the link score.
    min_size:
        Drop groups smaller than this many samples.
    """
    n = len(samples)
    if n == 0:
        return []
    w = max(0.0, min(1.0, float(clothing_weight)))
    thr = float(face_threshold)

    mat = np.vstack([s.vec for s in samples]).astype(np.float32)  # (n, 512), L2-normed
    face_sim = mat @ mat.T  # cosine since rows are unit-norm

    if w > 0.0:
        app_rows = [s.app if s.app is not None else None for s in samples]
        have_app = np.array([a is not None for a in app_rows])
        appmat = np.vstack([
            (a if a is not None else np.zeros(mat.shape[1], dtype=np.float32))
            for a in app_rows
        ]).astype(np.float32)
        app_sim = appmat @ appmat.T
        # Only blend where BOTH samples have an appearance vector; else face-only.
        both = np.outer(have_app, have_app)
        fused = np.where(both, (1.0 - w) * face_sim + w * app_sim, face_sim)
    else:
        fused = face_sim

    uf = _UnionFind(n)
    # Link every pair at/above threshold (single-linkage connected components).
    iu = np.triu_indices(n, k=1)
    for i, j in zip(iu[0].tolist(), iu[1].tolist()):
        if fused[i, j] >= thr:
            uf.union(i, j)

    groups: dict[int, FaceGroup] = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, FaceGroup()).members.append(samples[idx])

    out = [g for g in groups.values() if g.size >= max(1, int(min_size))]
    # Largest groups first; within a group, highest-quality sample first.
    for g in out:
        g.members.sort(key=lambda s: s.quality, reverse=True)
    out.sort(key=lambda g: g.size, reverse=True)
    return out

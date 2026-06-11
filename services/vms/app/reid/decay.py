"""Time-decay math for appearance (body Re-ID) exemplars.

People change clothes between days, so an appearance embedding captured
yesterday should barely link to a sighting today, while one captured minutes
ago is a strong link. We model this with an exponential decay weight

    w(Δt) = exp(-Δt / TAU)

applied to the cosine similarity of each stored appearance exemplar. With the
default ``TAU = 12h`` an exemplar older than ~2*TAU contributes negligibly
(w < ~0.14), matching the "outfit is stale by tomorrow" intuition.

Two derived quantities use this:

  * The *effective* appearance score of a candidate identity is the maximum,
    over its stored appearance exemplars, of ``cosine * w(Δt)`` — a single
    recent, similar outfit is enough to link.
  * The identity's appearance *centroid* (a derived cache) is the
    decay-weighted, L2-normalized mean of its exemplar vectors at recompute
    time.

Face exemplars are explicitly NOT decayed (faces are time-stable); this module
is appearance-only.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

# Below this weight an exemplar is treated as contributing nothing (avoids
# carrying ancient, irrelevant outfits in centroid math / scoring).
NEGLIGIBLE_WEIGHT = 1e-3


def _to_epoch(ts: datetime) -> float:
    return ts.timestamp()


def decay_weight(age_seconds: float, tau_seconds: float) -> float:
    """Exponential decay weight ``exp(-age/tau)`` clamped to ``[0, 1]``.

    A negative ``age`` (exemplar timestamp slightly in the future relative to
    ``now`` due to clock skew) is treated as age 0 -> weight 1.0.
    """
    if tau_seconds <= 0:
        return 1.0
    age = max(0.0, float(age_seconds))
    return math.exp(-age / float(tau_seconds))


def decay_weight_at(
    exemplar_ts: datetime, now: datetime, tau_seconds: float
) -> float:
    """Decay weight of an exemplar captured at ``exemplar_ts`` evaluated at ``now``."""
    return decay_weight(_to_epoch(now) - _to_epoch(exemplar_ts), tau_seconds)


def effective_score(
    cosines: Sequence[float],
    timestamps: Sequence[datetime],
    now: datetime,
    tau_seconds: float,
) -> float:
    """Decay-weighted appearance score for a candidate identity.

    ``cosines[i]`` is the raw cosine similarity between the query appearance
    vector and the identity's i-th appearance exemplar, captured at
    ``timestamps[i]``. Returns the maximum of ``cosine_i * w(Δt_i)``; an empty
    input returns ``0.0``.
    """
    best = 0.0
    for cos, ts in zip(cosines, timestamps):
        w = decay_weight_at(ts, now, tau_seconds)
        score = float(cos) * w
        if score > best:
            best = score
    return best


def decayed_centroid(
    vectors: Sequence[np.ndarray],
    timestamps: Sequence[datetime],
    now: datetime,
    tau_seconds: float,
    eps: float = 1e-10,
    weights: Sequence[float] | None = None,
) -> np.ndarray | None:
    """Decay-weighted, L2-normalized mean of appearance exemplar vectors.

    Exemplars whose decay weight falls below :data:`NEGLIGIBLE_WEIGHT` are
    dropped. Returns ``None`` when no exemplar carries meaningful weight (the
    caller should leave the cached centroid untouched / clear it).

    ``weights`` optionally scales each sample by a quality factor (e.g. crop
    sharpness) on top of the time decay; omitting it reproduces the pure
    time-decayed mean exactly (parity-preserving).
    """
    acc: np.ndarray | None = None
    total_w = 0.0
    for i, (vec, ts) in enumerate(zip(vectors, timestamps)):
        w = decay_weight_at(ts, now, tau_seconds)
        if w < NEGLIGIBLE_WEIGHT:
            continue
        if weights is not None and i < len(weights):
            q = float(weights[i])
            if q > 0.0:
                w *= q
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if acc is None:
            acc = np.zeros_like(v)
        acc += v * w
        total_w += w
    if acc is None or total_w <= 0.0:
        return None
    mean = acc / total_w
    norm = float(np.linalg.norm(mean))
    if norm < eps:
        return None
    return (mean / norm).astype(np.float32)


def is_stale(
    exemplar_ts: datetime,
    now: datetime,
    tau_seconds: float,
    stale_multiple: float = 2.0,
) -> bool:
    """Return True when an appearance exemplar is older than ``stale_multiple*TAU``.

    Used by maintenance to prune exemplars that can no longer meaningfully
    contribute to a match (their decay weight is below ~exp(-stale_multiple)).
    """
    age = _to_epoch(now) - _to_epoch(exemplar_ts)
    return age > stale_multiple * float(tau_seconds)


def max_decay_weight(
    timestamps: Iterable[datetime], now: datetime, tau_seconds: float
) -> float:
    """Largest decay weight among ``timestamps`` (i.e. the most recent exemplar).

    Handy for the appearance time-window gate: an identity is an appearance
    candidate only if its freshest exemplar is recent enough.
    """
    best = 0.0
    for ts in timestamps:
        w = decay_weight_at(ts, now, tau_seconds)
        if w > best:
            best = w
    return best

"""Non-learned boundary rules. Each returns a 0/1 array b of length T with b[0] = 1.

All rules are causal: b[t] uses only information available at step t, so they
can be fed to the chunked model without leaking the future.
"""
from __future__ import annotations

import numpy as np


def fixed_boundaries(T, N):
    b = np.zeros(T, dtype=np.int8)
    b[::N] = 1
    return b


def random_boundaries(T, N, episode_seed):
    """Bernoulli boundaries whose expected count, including the forced step-0 boundary, is T / N."""
    rng = np.random.default_rng(episode_seed)
    q = max(0.0, (T / N - 1) / (T - 1)) if T > 1 else 0.0
    b = (rng.random(T) < q).astype(np.int8)
    b[0] = 1
    return b


def local_minima_score(speed):
    """Score for a causal velocity-minimum boundary at t: speed[t-1] if t-1 was a local minimum
    (speed[t-2] >= speed[t-1] < speed[t]), else +inf. The minimum is only known one step later."""
    s = np.asarray(speed, dtype=np.float64)
    score = np.full(len(s), np.inf)
    if len(s) >= 3:
        is_min = (s[1:-1] <= s[:-2]) & (s[1:-1] < s[2:])
        score[2:][is_min] = s[1:-1][is_min]
    return score


def threshold_for_rate(scores, lengths, N, higher=True):
    """Threshold on pooled per-step scores so that the boundary rate (counting the forced
    step-0 boundary) is 1/N. `higher`: boundary where score > thr; else score < thr."""
    pooled = np.concatenate([s[1:] for s in scores])
    total = int(np.sum(lengths))
    k = int(round(total / N)) - len(lengths)  # boundaries left after the forced ones
    k = max(0, min(k, len(pooled)))
    srt = np.sort(pooled)
    if higher:
        srt = srt[::-1]
        thr = srt[k - 1] if k > 0 else np.inf
        return thr
    thr = srt[k - 1] if k > 0 else -np.inf
    return thr


def apply_threshold(score, thr, higher=True):
    score = np.asarray(score)
    b = (score >= thr) if higher else (score <= thr)
    b = b & np.isfinite(score)
    b = b.astype(np.int8)
    b[0] = 1
    return b


def velocity_boundaries(speeds, N, thr=None):
    """Velocity-minima rule. Pass `thr` fitted on the training split, or None to fit on `speeds`."""
    scores = [local_minima_score(s) for s in speeds]
    if thr is None:
        thr = threshold_for_rate(scores, [len(s) for s in speeds], N, higher=False)
    return [apply_threshold(sc, thr, higher=False) for sc in scores], thr


def entropy_boundaries(nlls, N, thr=None):
    """BLT-like rule: boundary at t when the proxy model's NLL of x(t) given x(<t) is above thr."""
    if thr is None:
        thr = threshold_for_rate(nlls, [len(s) for s in nlls], N, higher=True)
    return [apply_threshold(s, thr, higher=True) for s in nlls], thr

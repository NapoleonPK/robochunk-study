"""Boundary and alignment metrics, plus bootstrap and permutation helpers."""
from __future__ import annotations

import numpy as np


def true_boundaries(phase):
    """Steps where the phase label changes (step 0 is never a boundary)."""
    phase = np.asarray(phase)
    return np.flatnonzero(phase[1:] != phase[:-1]) + 1


def match_boundaries(pred, true, tol):
    """One-to-one matching, nearest pairs first, within ±tol steps. Returns number of matches."""
    pred = np.asarray([p for p in pred if p != 0])
    true = np.asarray(true)
    if len(pred) == 0 or len(true) == 0:
        return 0
    d = np.abs(pred[:, None] - true[None, :])
    pi, ti = np.nonzero(d <= tol)
    order = np.argsort(d[pi, ti], kind="stable")
    used_p, used_t, m = set(), set(), 0
    for k in order:
        a, b = pi[k], ti[k]
        if a not in used_p and b not in used_t:
            used_p.add(a)
            used_t.add(b)
            m += 1
    return m


def boundary_prf(pred, true, tol=3):
    """Precision, recall, F1 for one episode. The forced boundary at step 0 is ignored.
    Precision is 0 when nothing is predicted; recall is 0 when nothing matches."""
    pred = np.asarray([p for p in pred if p != 0])
    m = match_boundaries(pred, true, tol)
    p = m / len(pred) if len(pred) else 0.0
    r = m / len(true) if len(true) else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def episode_f1(preds, phases, tol=3):
    """Per-episode (P, R, F1) array of shape (n_episodes, 3)."""
    return np.array([boundary_prf(p, true_boundaries(ph), tol) for p, ph in zip(preds, phases)])


def bootstrap_ci(values, n=10_000, seed=0, stat=np.mean, alpha=0.05):
    """Percentile bootstrap over the first axis (episodes)."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n, len(values)))
    stats = np.array([stat(values[i]) for i in idx]) if stat is not np.mean else values[idx].mean(1)
    return float(stat(values)), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def near_event_share(bounds, events, tol=3):
    """Fraction of boundaries (step 0 excluded) within ±tol of any event, pooled over episodes."""
    hit = tot = 0
    for b, e in zip(bounds, events):
        b = np.asarray([x for x in b if x != 0])
        e = np.asarray(e)
        tot += len(b)
        if len(b) and len(e):
            hit += int((np.abs(b[:, None] - e[None, :]).min(1) <= tol).sum())
    return hit / tot if tot else 0.0


def shuffle_test(bounds, events, lengths, tol=3, n=1000, seed=0):
    """Within-episode shuffle of boundary positions (same count per episode, positions 1..T-1).
    Returns observed share, mean shuffled share, ratio and one-sided permutation p-value."""
    rng = np.random.default_rng(seed)
    obs = near_event_share(bounds, events, tol)
    null = np.empty(n)
    counts = [len([x for x in b if x != 0]) for b in bounds]
    for i in range(n):
        sb = [rng.choice(np.arange(1, T), size=min(c, T - 1), replace=False) if c else [] for c, T in zip(counts, lengths)]
        null[i] = near_event_share(sb, events, tol)
    p = (1 + (null >= obs).sum()) / (n + 1)
    return dict(share=obs, null_share=float(null.mean()), ratio=obs / null.mean() if null.mean() > 0 else np.inf, p=float(p),
                null=null)

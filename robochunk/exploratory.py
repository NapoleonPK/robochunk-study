"""Exploratory boundary analyses (not pre-registered; requested 2026-10-29, see DECISIONS.md).

- Chance-corrected F1: F1 minus the mean F1 of the same boundaries shuffled within each episode; for the Fixed
  rule also minus a grid chance (the same every-N grid from a random offset per episode).
- Burst-merged F1: boundaries closer than `gap` steps to the last kept one are dropped, then F1 again.
- Boundary density by phase: boundaries per step inside each of the 7 phases.
"""
from __future__ import annotations

import numpy as np

from .data_sim import PHASES
from .metrics import boundary_prf, true_boundaries


def mean_f1(bounds, phases, tol):
    return float(np.mean([boundary_prf(b, true_boundaries(ph), tol)[2] for b, ph in zip(bounds, phases)]))


def chance_f1(bounds, phases, tol, n=1000, seed=0):
    """Mean F1 over n within-episode shuffles: each episode keeps its number of boundaries (step 0 excluded),
    placed uniformly without replacement on steps 1..T-1. Returns (mean, 2.5%, 97.5%) over shuffles."""
    rng = np.random.default_rng(seed)
    trues = [true_boundaries(ph) for ph in phases]
    counts = [int(np.sum(np.asarray(b) != 0)) for b in bounds]
    lengths = [len(ph) for ph in phases]
    null = np.empty(n)
    for i in range(n):
        f = [boundary_prf(rng.choice(np.arange(1, T), size=min(c, T - 1), replace=False) if c else [], t, tol)[2]
             for c, T, t in zip(counts, lengths, trues)]
        null[i] = np.mean(f)
    return float(null.mean()), float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))


def grid_chance_f1(phases, N, tol, n=1000, seed=0):
    """Chance floor for a regular grid: boundaries every N steps from a random offset U{0..N-1} per episode
    (step 0 excluded). Returns (mean, 2.5%, 97.5%) over n draws."""
    rng = np.random.default_rng(seed)
    trues = [true_boundaries(ph) for ph in phases]
    null = np.empty(n)
    for i in range(n):
        null[i] = np.mean([boundary_prf(np.arange(rng.integers(0, N), len(ph), N), t, tol)[2]
                           for ph, t in zip(phases, trues)])
    return float(null.mean()), float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))


def merge_bursts(b, gap=3):
    """Keep a boundary only if it is at least `gap` steps after the last kept one (step 0 counts as kept)."""
    kept, last = [], 0
    for t in sorted(int(x) for x in b):
        if t == 0 or t - last >= gap:
            kept.append(t)
            last = t
    return np.asarray(kept, dtype=np.int64)


def phase_density(bounds, phases):
    """Boundaries per step in each phase, pooled over episodes; a boundary at step t counts for phase[t].
    Step 0 is excluded (forced boundary, and not a step any rule chooses)."""
    hits, steps = np.zeros(len(PHASES)), np.zeros(len(PHASES))
    for b, ph in zip(bounds, phases):
        ph = np.asarray(ph)
        steps += np.bincount(ph[1:], minlength=len(PHASES))
        b = np.asarray([x for x in b if x != 0], dtype=np.int64)
        if len(b):
            hits += np.bincount(ph[b], minlength=len(PHASES))
    return {p: float(h / s) if s else float("nan") for p, h, s in zip(PHASES, hits, steps)}


def run_report(bounds, phases, tols=(1, 3), n_shuffle=1000, merge_gap=3, grid_N=None):
    out = {"rate": float(sum(int(np.sum(np.asarray(b) != 0)) + 1 for b in bounds) / sum(len(p) for p in phases))}
    merged = [merge_bursts(b, merge_gap) for b in bounds]
    for tol in tols:
        f1 = mean_f1(bounds, phases, tol)
        ch, lo, hi = chance_f1(bounds, phases, tol, n_shuffle)
        out[f"F1_tol{tol}"] = f1
        out[f"chance_F1_tol{tol}"] = ch
        out[f"chance_F1_tol{tol}_95"] = [lo, hi]
        out[f"F1_minus_chance_tol{tol}"] = f1 - ch
        out[f"merged_F1_tol{tol}"] = mean_f1(merged, phases, tol)
        if grid_N:
            g, glo, ghi = grid_chance_f1(phases, grid_N, tol, n_shuffle)
            out[f"grid_chance_F1_tol{tol}"] = g
            out[f"grid_chance_F1_tol{tol}_95"] = [glo, ghi]
            out[f"F1_minus_grid_chance_tol{tol}"] = f1 - g
    out["merged_rate"] = float(sum(len(b) for b in merged) / sum(len(p) for p in phases))
    out["phase_density"] = phase_density(bounds, phases)
    return out

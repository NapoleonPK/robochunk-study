"""Pre-registered analyses for H1–H4. Every number in the report is produced here from run outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .metrics import near_event_share, shuffle_test

RUNS = Path("runs")
N_BOOT = 10_000


def load_run(run_id):
    d = RUNS / run_id
    m = json.load(open(d / "metrics.json"))
    z = np.load(d / "test_eval.npz")
    ev = {k: z[k] for k in z.files}
    if "bounds" in ev:
        off = ev["bounds_off"]
        ev["bounds_list"] = [ev["bounds"][off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return m, ev


def paired_bootstrap(diff, n=N_BOOT, seed=0):
    """Mean of per-episode differences and its 95% percentile CI over episodes."""
    diff = np.asarray(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    bs = diff[idx].mean(1)
    return float(diff.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def h1(run_ids, tol=3, margin=0.10):
    """run_ids: {method: [run_id per seed]} at N_main, 'learned' = RoboChunk.
    Per-episode F1 averaged over seeds; for each baseline, the paired bootstrap of RoboChunk − baseline."""
    f1 = {m: np.mean([load_run(r)[1][f"prf_tol{tol}"][:, 2] for r in rs], 0) for m, rs in run_ids.items()}
    per_seed = {m: [float(load_run(r)[1][f"prf_tol{tol}"][:, 2].mean()) for r in rs] for m, rs in run_ids.items()}
    out = dict(tol=tol, mean_f1={m: float(v.mean()) for m, v in f1.items()}, per_seed=per_seed, diffs={})
    ok = True
    for m in run_ids:
        if m == "learned":
            continue
        mean, lo, hi = paired_bootstrap(f1["learned"] - f1[m])
        passed = mean >= margin and lo > 0
        out["diffs"][m] = dict(mean=mean, ci=[lo, hi], pass_=passed)
        ok &= passed
    out["pass"] = bool(ok)
    return out


def h2(learned, fixed, Ns, min_compression_frac=0.8):
    """learned/fixed: {N: [run_id per seed]}. Pass at N if mean test bits(learned) < mean bits(fixed)
    and RoboChunk's realised compression is at least 0.8 N (otherwise compute is not matched: counts as a fail)."""
    rows, wins = [], 0
    for N in Ns:
        lm = [load_run(r)[0] for r in learned[N]]
        fm = [load_run(r)[0] for r in fixed[N]]
        lb, fb = np.mean([m["test_bits"] for m in lm]), np.mean([m["test_bits"] for m in fm])
        comp = np.mean([m["test_compression"] for m in lm])
        matched = comp >= min_compression_frac * N
        win = bool(lb < fb and matched)
        wins += win
        rows.append(dict(N=N, learned_bits=float(lb), fixed_bits=float(fb), learned_bits_seeds=[m["test_bits"] for m in lm],
                         fixed_bits_seeds=[m["test_bits"] for m in fm], learned_compression=float(comp),
                         learned_flops=float(np.mean([m["flops_total_per_step"] for m in lm])),
                         fixed_flops=float(np.mean([m["flops_total_per_step"] for m in fm])),
                         compression_matched=bool(matched), learned_wins=win))
    return dict(rows=rows, wins=wins, pass_=wins >= 2)


def h3(run_ids, events, lengths, tol=3, n_shuffle=1000, ratio_needed=2.0, alpha=0.01):
    """Seed-pooled: each (seed, test episode) is one unit; boundaries shuffled within each unit.
    events/lengths: per test episode. Pass if share / shuffled share >= 2 and permutation p < 0.01."""
    bounds, ev, L = [], [], []
    for r in run_ids:
        b = load_run(r)[1]["bounds_list"]
        bounds += b
        ev += list(events)
        L += list(lengths)
    res = shuffle_test(bounds, ev, L, tol, n_shuffle, seed=0)
    per_seed = []
    for r in run_ids:
        b = load_run(r)[1]["bounds_list"]
        per_seed.append(near_event_share(b, events, tol))
    out = dict(share=res["share"], null_share=res["null_share"], ratio=res["ratio"], p=res["p"],
               per_seed_share=per_seed, pass_=bool(res["ratio"] >= ratio_needed and res["p"] < alpha))
    return out


def gain_ci(succ_rc, succ_r, n=N_BOOT, seed=0):
    """succ_*: (seeds, episodes) 0/1 success. Bootstrap resamples episodes (paired) and seeds independently."""
    succ_rc, succ_r = np.asarray(succ_rc), np.asarray(succ_r)
    rng = np.random.default_rng(seed)
    S, E = succ_rc.shape
    g = np.empty(n)
    for i in range(n):
        e = rng.integers(0, E, E)
        s1, s0 = rng.integers(0, S, S), rng.integers(0, S, S)
        g[i] = succ_rc[s1][:, e].mean() - succ_r[s0][:, e].mean()
    return float(succ_rc.mean() - succ_r.mean()), float(np.quantile(g, 0.025)), float(np.quantile(g, 0.975))


def h4(novelty, gain, clean, corrupt, rho_needed=0.6):
    """novelty/gain: {candidate: value}. Spearman over clean candidates; failure-mode check on corrupted ones:
    novelty in the top half of all candidates and gain CI upper bound <= 0 or CI containing 0."""
    rho, p = stats.spearmanr([novelty[c] for c in clean], [gain[c]["mean"] for c in clean])
    allc = clean + corrupt
    median = np.median([novelty[c] for c in allc])
    fm = {x: dict(top_half=bool(novelty[x] > median), no_gain=bool(gain[x]["ci"][0] <= 0)) for x in corrupt}
    return dict(rho=float(rho), p=float(p), pass_=bool(rho >= rho_needed), failure_mode=fm,
                failure_mode_count=sum(v["top_half"] and v["no_gain"] for v in fm.values()))

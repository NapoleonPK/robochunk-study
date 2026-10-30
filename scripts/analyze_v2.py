"""v2 confirmatory analysis (PREREGISTRATION_v2.md). Run once, after all v2 runs: PYTHONPATH=. python scripts/analyze_v2.py
Verifies the frozen hashes, then writes runs/results_v2.json. Every reported v2 number comes from that file."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from robochunk import analysis as A
from robochunk import baselines as BL
from robochunk import exploratory as X
from robochunk.data import Normalizer
from robochunk.data_real import load_real
from robochunk.datasets import load_dataset
from robochunk.metrics import boundary_prf, shuffle_test, true_boundaries
from robochunk.train import DEFAULTS, build_model, episode_seed, evaluate, proxy_surprise

sys.path.insert(0, "scripts")
from real_bursts import describe, event_dirs  # noqa: E402

RUNS, DEV = Path("runs"), torch.device("cpu")
N, N_REAL, SEEDS, TOLS = 10, 12, [0, 1, 2], (1, 3, 5)
VEL_THR, ENT_THR = 0.0019115220056846738, -7.098278999328613  # v1 thresholds, stated in PREREGISTRATION_v2.md §4
torch.set_num_threads(1)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def verify_hashes():
    bad = []
    for f in ("runs/v2/frozen_artifacts.sha256", "runs/v2/fresh_sha256.txt"):
        for line in open(f):
            h, p = line.split()
            if sha(p) != h:
                bad.append(p)
    assert not bad, f"frozen files changed: {bad}"
    for m, thr in (("velocity", VEL_THR), ("entropy", ENT_THR)):
        assert json.load(open(RUNS / f"sim_{m}_N10_s0" / "metrics.json"))["threshold"] == thr


def run_model(run_id, eps, bnds=None):
    d = RUNS / run_id
    cfg = {**DEFAULTS, **yaml.safe_load(open(d / "config.yaml"))}
    norm = Normalizer.load(d / "norm.json")
    model = build_model(cfg, eps[0]["x"].shape[1])
    model.load_state_dict(torch.load(d / "best.pt", map_location=DEV))
    ev = evaluate(model, cfg, eps, bnds, norm, DEV, keep=True)
    if bnds is not None:
        ev["bounds"] = [np.flatnonzero(b) for b in bnds]
    return ev


def per_episode_f1(bounds, phases, tol):
    return np.array([boundary_prf(b, true_boundaries(p), tol)[2] for b, p in zip(bounds, phases)])


def sim_part():
    eps = load_dataset({**DEFAULTS, "dataset": "sim", "splits": {"test": "fresh"}})["test"]
    phases = [e["phase"] for e in eps]
    T = [len(e["x"]) for e in eps]
    # boundary sets per method: list over seeds of per-episode boundary arrays (seed-independent rules: one set)
    B, bits = {}, {}
    for name, pre in (("refractory", "exp_refractory_N10"), ("learned", "sim_learned_N10"), ("secondary", "v2_sim_refractory_N10")):
        seeds = [3, 4, 5] if name == "secondary" else SEEDS
        evs = [run_model(f"{pre}_s{s}", eps) for s in seeds]
        B[name] = [e["bounds"] for e in evs]
        bits[name] = dict(seeds=[e["bits"] for e in evs], rate=[e["rate"] for e in evs])
    fixed = [BL.fixed_boundaries(t, N) for t in T]
    B["fixed"] = [[np.flatnonzero(b) for b in fixed]]
    fevs = [run_model(f"sim_fixed_N10_s{s}", eps, fixed) for s in SEEDS]
    bits["fixed"] = dict(seeds=[e["bits"] for e in fevs], rate=[e["rate"] for e in fevs])
    B["random"] = [[np.flatnonzero(BL.random_boundaries(t, N, episode_seed("fresh", i) + s)) for i, t in enumerate(T)]
                   for s in SEEDS]
    vel, _ = BL.velocity_boundaries([e["speed"] for e in eps], N, VEL_THR)
    B["velocity"] = [[np.flatnonzero(b) for b in vel]]
    ent, _ = BL.entropy_boundaries(proxy_surprise("runs/proxy_sim", eps, DEV, "robot"), N, ENT_THR)
    B["entropy"] = [[np.flatnonzero(b) for b in ent]]

    res = dict(n_episodes=len(eps), methods={})
    f1 = {m: {t: np.mean([per_episode_f1(b, phases, t) for b in bs], 0) for t in TOLS} for m, bs in B.items()}
    chance = {}
    for m, bs in B.items():  # circular-shift chance (primary for H1b-v2), a new rng(0) per run
        chance[m] = {t: np.mean([X.circular_chance_f1(b, phases, t, 1000, 0) for b in bs], 0) for t in (1, 3)}
    for m, bs in B.items():
        r = dict(F1={t: float(f1[m][t].mean()) for t in TOLS},
                 circ_chance={t: float(chance[m][t].mean()) for t in (1, 3)},
                 cf_circ={t: float((f1[m][t] - chance[m][t]).mean()) for t in (1, 3)},
                 rate=float(np.mean([sum(len(x) for x in b) / sum(T) for b in bs])))
        # secondary, v1 chance definitions (uniform shuffle; grid offsets for Fixed)
        r["uniform_chance_3"] = float(np.mean([X.chance_f1(b, phases, 3, 1000)[0] for b in bs]))
        r["cf_uniform_3"] = r["F1"][3] - r["uniform_chance_3"]
        if m == "fixed":
            r["grid_chance_3"] = X.grid_chance_f1(phases, N, 3, 1000)[0]
            r["cf_grid_3"] = r["F1"][3] - r["grid_chance_3"]
        r["merged_F1_3"] = float(np.mean([X.mean_f1([X.merge_bursts(x) for x in b], phases, 3) for b in bs]))
        r["phase_density"] = {k: float(np.mean([X.phase_density(b, phases)[k] for b in bs])) for k in X.PHASES}
        r["per_seed_F1_3"] = [float(per_episode_f1(b, phases, 3).mean()) for b in bs]
        res["methods"][m] = r
    base = ["fixed", "random", "entropy", "velocity"]
    h1, h1b = {}, {}
    for b in base:
        mean, lo, hi = A.paired_bootstrap(f1["refractory"][3] - f1[b][3])
        h1[b] = dict(mean=mean, ci=[lo, hi], pass_=bool(mean >= 0.10 and lo > 0))
        d = (f1["refractory"][3] - chance["refractory"][3]) - (f1[b][3] - chance[b][3])
        mean, lo, hi = A.paired_bootstrap(d)
        h1b[b] = dict(mean=mean, ci=[lo, hi], pass_=bool(mean > 0 and lo > 0))
    res["H1_v2"] = dict(diffs=h1, pass_=all(v["pass_"] for v in h1.values()))
    res["H1b_v2"] = dict(diffs=h1b, pass_=all(v["pass_"] for v in h1b.values()))
    comp = 1 / np.mean(bits["refractory"]["rate"])
    res["bits"] = dict(bits, refractory_compression=float(comp), compression_matched=bool(comp >= 0.8 * N),
                       refractory_better=bool(np.mean(bits["refractory"]["seeds"]) < np.mean(bits["fixed"]["seeds"])))
    return res


def real_part():
    res = {}
    for ds in ("so100", "koch", "so101"):
        sp = load_real(ds)
        te = sp["test"]
        ev, L = [e["events"] for e in te], [len(e["x"]) for e in te]
        ids = [f"v2_real_{ds}_refractory_N{N_REAL}_s{s}" for s in SEEDS]
        kinds, _ = event_dirs(te)
        r = dict(test=A.h3(ids, ev, L), n_test_episodes=len(te), n_events=int(sum(len(e) for e in ev)),
                 runs={i: {k: v for k, v in describe(i, te, kinds).items() if k != "bursts"} for i in ids},
                 stops={i: json.load(open(RUNS / i / "metrics.json"))["stop"] for i in ids},
                 test_rate={i: json.load(open(RUNS / i / "metrics.json"))["test_rate"] for i in ids})
        if ds == "so101":  # secondary: entropy rule on the fresh dataset (proxy + val-fitted threshold, as in v1)
            sur = {k: proxy_surprise("runs/v2_proxy_so101", v, DEV) for k, v in sp.items() if k in ("val", "test")}
            _, thr = BL.entropy_boundaries(sur["val"], N_REAL)
            eb = [np.flatnonzero(b) for b in BL.entropy_boundaries(sur["test"], N_REAL, thr)[0]]
            st = shuffle_test(eb, ev, L, 3, 1000, seed=0)
            hits = [bool(len(b[b != 0]) and np.abs(b[b != 0] - t).min() <= 3) for b, e in zip(eb, ev) for t in e]
            r["entropy_secondary"] = dict(share=st["share"], null_share=st["null_share"], ratio=st["ratio"], p=st["p"],
                                          event_recall=float(np.mean(hits)), threshold=float(thr),
                                          rate=float(sum(len(b) for b in eb) / sum(L)))
        res[ds] = r
    passes = [res[d]["test"]["pass_"] for d in ("so100", "koch")]
    res["H3_v2"] = {2: "supported", 1: "mixed", 0: "not supported"}[sum(passes)]
    res["H3b_v2"] = "pass" if res["so101"]["test"]["pass_"] else "fail"
    return res


if __name__ == "__main__":
    verify_hashes()
    out = dict(sim=sim_part(), real=real_part())
    json.dump(out, open(RUNS / "results_v2.json", "w"), indent=1, default=float)
    s, r = out["sim"], out["real"]
    print(json.dumps(dict(H1_v2=s["H1_v2"], H1b_v2=s["H1b_v2"], H3_v2=r["H3_v2"], H3b_v2=r["H3b_v2"],
                          H3_ratios={d: (r[d]["test"]["ratio"], r[d]["test"]["p"]) for d in ("so100", "koch", "so101")}),
                     indent=1, default=float))

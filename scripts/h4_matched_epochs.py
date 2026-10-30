"""POST-HOC diagnostic for the negative in-distribution gains (C2, C3). Not pre-registered; does not change H4.

Hypothesis: BC trains for a fixed 20,000 updates, so adding new unique data cuts passes per sample from ~260 (R) to ~186
(R ∪ C) and under-fits; duplicates (C1) add no unique samples and stay neutral. Test: retrain R ∪ C at updates scaled so
passes match R's (20,000 × samples(R ∪ C) / samples(R)), same seeds and test set. If the hypothesis holds, C2 and C3 gains
move to about zero or above. Writes runs/partb_policy_matched/<name>_s<seed>.npy and runs/results_partB_matched.json.
usage: PYTHONPATH=. python scripts/h4_matched_epochs.py --workers 2
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path("runs/partb_policy_matched")
NAMES = ["C1", "C2", "C3", "C6"]


def job(args):
    name, seed = args
    import torch
    torch.set_num_threads(3)
    from robochunk.data_sim import load
    from robochunk.partb import broad_specs
    from robochunk.policy import evaluate_policy, train_bc
    path = OUT / f"{name}_s{seed}.npy"
    if path.exists():
        return name, seed, float(np.load(path).mean())
    r = load("data/partb/R.npz")
    eps = r + load(f"data/partb/{name}.npz")
    steps = round(train_bc.__defaults__[0] * sum(len(e["action"]) for e in eps) / sum(len(e["action"]) for e in r))
    succ = evaluate_policy(train_bc(eps, seed, steps=steps), broad_specs(200))
    np.save(path, succ)
    return name, seed, float(succ.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    with Pool(a.workers) as p:
        for name, seed, sr in p.imap_unordered(job, [(n, s) for s in range(3) for n in NAMES]):
            print(f"{name} seed {seed}: success {sr:.3f}", flush=True)
    from robochunk.analysis import gain_ci
    base = np.stack([np.load(f"runs/partb_policy/R_s{s}.npy") for s in range(3)])
    res = {"label": "POST-HOC, matched passes over the data"}
    for n in NAMES:
        sc = np.stack([np.load(OUT / f"{n}_s{s}.npy") for s in range(3)])
        g, lo, hi = gain_ci(sc, base)
        res[n] = dict(gain=g, ci=[lo, hi], success=float(sc.mean()))
    json.dump(res, open("runs/results_partB_matched.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

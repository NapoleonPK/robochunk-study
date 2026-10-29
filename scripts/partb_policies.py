"""Part B usefulness: BC on R and on R ∪ C for every candidate, 3 seeds, evaluated on the broad test set.
Writes runs/partb_policy/<name>_s<seed>.npy (0/1 success per broad test episode). Resumable."""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path("runs/partb_policy")


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
    eps = load("data/partb/R.npz")
    if name != "R":
        eps = eps + load(f"data/partb/{name}.npz")
    pol = train_bc(eps, seed)
    succ = evaluate_policy(pol, broad_specs(200))
    np.save(path, succ)
    return name, seed, float(succ.mean())


if __name__ == "__main__":
    from robochunk.partb import CLEAN, CORRUPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s) for s in range(a.seeds) for n in ["R"] + CLEAN + CORRUPT]
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    with Pool(a.workers) as p:
        for name, seed, sr in p.imap_unordered(job, jobs):
            print(f"{name} seed {seed}: success {sr:.3f}", flush=True)

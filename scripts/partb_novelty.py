"""Part B novelty scores for every candidate.

- RoboChunk / Flat score: mean bits per step on the candidate minus mean bits per step on held-out R,
  using models trained on R only (runs given by --runs, one per seed); averaged over seeds.
- Nearest-neighbour score: mean distance from candidate steps to the nearest R step, in R-normalised
  robot channels.
Writes runs/partb_novelty.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from robochunk.data import Normalizer
from robochunk.datasets import _sim_split
from robochunk.partb import CLEAN, CORRUPT
from robochunk.train import DEFAULTS, build_model, evaluate


def bits_on(run_dir, eps):
    cfg = {**DEFAULTS, **yaml.safe_load(open(run_dir / "config.yaml"))}
    norm = Normalizer.load(run_dir / "norm.json")
    model = build_model(cfg, len(norm.mean))
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    bnds = None
    if cfg["method"] == "flat":
        bnds = [np.ones(len(e["x"]), np.int8) for e in eps]
    elif cfg["method"] != "learned":
        raise ValueError("novelty is defined for learned and flat models only")
    return evaluate(model, cfg, eps, bnds, norm, "cpu")["bits"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robochunk", nargs="+", required=True, help="run ids trained on R")
    ap.add_argument("--flat", nargs="+", required=True)
    ap.add_argument("--inputs", default="robot")
    a = ap.parse_args()
    torch.set_num_threads(6)
    names = CLEAN + CORRUPT
    data = {n: _sim_split(f"data/partb/{n}.npz", a.inputs) for n in names + ["R", "R_heldout"]}
    out = {}
    for kind, runs in [("robochunk", a.robochunk), ("flat", a.flat)]:
        per_seed = []
        for r in runs:
            ref = bits_on(Path("runs") / r, data["R_heldout"])
            per_seed.append({n: bits_on(Path("runs") / r, data[n]) - ref for n in names})
        out[kind] = {n: float(np.mean([s[n] for s in per_seed])) for n in names}
        out[kind + "_per_seed"] = per_seed
    norm = Normalizer.fit([e["x"] for e in data["R"]])
    tree = cKDTree(np.concatenate([norm(e["x"]) for e in data["R"]]))
    out["nn"] = {}
    for n in names:
        q = np.concatenate([norm(e["x"]) for e in data[n]])
        out["nn"][n] = float(tree.query(q, k=1)[0].mean())
    json.dump(out, open("runs/partb_novelty.json", "w"), indent=1)
    for n in names:
        print(f"{n:4s} robochunk {out['robochunk'][n]:8.3f}  flat {out['flat'][n]:8.3f}  nn {out['nn'][n]:.3f}")

"""Run the pre-registered tests H1–H4 and secondary measures from run outputs.

usage: python scripts/analyze.py [partA|real|partB|all]
Writes runs/results_<part>.json and runs/summary_<part>.csv (the README/report numbers come from these).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from robochunk import analysis as A
from robochunk.metrics import shuffle_test

FROZEN = yaml.safe_load(open("configs/frozen.yaml"))
NM = FROZEN["N_main"]
NS = [NM // 2, NM, 2 * NM]
SEEDS = [0, 1, 2]
METHODS = ["learned", "fixed", "random", "entropy", "velocity"]
OUT = Path("runs")


def _dump(name, obj):
    json.dump(obj, open(OUT / f"results_{name}.json", "w"), indent=1, default=float)


def run_table(prefix):
    rows = [json.load(open(p)) for p in sorted(OUT.glob(f"{prefix}*/metrics.json"))]
    return pd.DataFrame(rows)


def part_a():
    df = run_table("sim_")
    df.to_csv(OUT / "summary_sim.csv", index=False)
    res = {}
    res["H1"] = A.h1({m: [f"sim_{m}_N{NM}_s{s}" for s in SEEDS] for m in METHODS}, tol=3)
    res["H1_sensitivity"] = {t: A.h1({m: [f"sim_{m}_N{NM}_s{s}" for s in SEEDS] for m in METHODS}, tol=t)
                             for t in (1, 5)}
    res["F1_all_N"] = {m: {N: A.h1({"learned": [f"sim_learned_N{N}_s{s}" for s in SEEDS],
                                    m: [f"sim_{m}_N{N}_s{s}" for s in SEEDS]})["mean_f1"][m] for N in NS}
                       for m in METHODS}
    res["H2"] = A.h2({N: [f"sim_learned_N{N}_s{s}" for s in SEEDS] for N in NS},
                     {N: [f"sim_fixed_N{N}_s{s}" for s in SEEDS] for N in NS}, NS)
    res["flat_bits"] = [A.load_run(f"sim_flat_s{s}")[0]["test_bits"] for s in SEEDS]
    # budget diagnostics: did runs stop on the cap rather than converge?
    res["stops"] = df.groupby("method")["stop"].value_counts().unstack(fill_value=0).to_dict()
    res["realised_rates"] = df.groupby(["method", "N"])["test_rate"].mean().reset_index().to_dict("records")
    for grp, pre in [("ablations", "abl_"), ("exploratory", "exp_")]:
        t = run_table(pre)
        if len(t):
            ids = {r: list(t[t.run_id.str.startswith(r)].run_id) for r in sorted({x.rsplit("_s", 1)[0] for x in t.run_id})}
            res[grp] = {k: A.h1({"learned": v, "fixed": [f"sim_fixed_N{NM}_s{s}" for s in SEEDS]}) for k, v in ids.items()}
            res[grp + "_bits"] = {k: float(t[t.run_id.isin(v)].test_bits.mean()) for k, v in ids.items()}
    _dump("partA", res)
    print(json.dumps({"H1": res["H1"]["pass"], "H1_f1": res["H1"]["mean_f1"], "H2": res["H2"]["pass_"],
                      "H2_rows": [(r["N"], round(r["learned_bits"], 3), round(r["fixed_bits"], 3)) for r in res["H2"]["rows"]]},
                     indent=1, default=float))


def part_real():
    from robochunk.data_real import load_real
    res = {}
    verdicts = []
    for name, nr in FROZEN["real_datasets"].items():
        sp = load_real(name)
        r = {}
        for split in ("test",):
            ev, L = [e["events"] for e in sp[split]], [len(e["x"]) for e in sp[split]]
            r["H3"] = A.h3([f"real_{name}_learned_N{nr}_s{s}" for s in SEEDS], ev, L)
            for m in ("fixed", "entropy"):
                r[f"{m}_alignment"] = A.h3([f"real_{name}_{m}_N{nr}_s{s}" for s in SEEDS], ev, L)
        r["tolerance_seconds"] = 3 / 30
        res[name] = r
        verdicts.append(r["H3"]["pass_"])
    res["H3_verdict"] = {2: "supported", 1: "mixed", 0: "not supported"}[sum(verdicts)]
    _dump("real", res)
    print(json.dumps({k: (v["H3"] if isinstance(v, dict) else v) for k, v in res.items()}, indent=1, default=float))


def part_b():
    from robochunk.partb import CLEAN, CORRUPT
    nov = json.load(open(OUT / "partb_novelty.json"))
    pol = Path("runs/partb_policy")
    base = np.stack([np.load(pol / f"R_s{s}.npy") for s in SEEDS])
    cands = {}
    for c in CLEAN + CORRUPT:
        sc = np.stack([np.load(pol / f"{c}_s{s}.npy") for s in SEEDS])
        g, lo, hi = A.gain_ci(sc, base)
        cands[c] = dict(novelty=nov["robochunk"][c], novelty_flat=nov["flat"][c], novelty_nn=nov["nn"][c],
                        gain=g, gain_ci=[lo, hi], success=float(sc.mean()))
    gain = {c: dict(mean=v["gain"], ci=v["gain_ci"]) for c, v in cands.items()}
    h4 = A.h4({c: v["novelty"] for c, v in cands.items()}, gain, CLEAN, CORRUPT)
    sec = {k: A.h4({c: v[f"novelty_{k}"] for c, v in cands.items()}, gain, CLEAN, CORRUPT) for k in ("flat", "nn")}
    res = dict(candidates=cands, h4=h4, secondary=sec, base_success=float(base.mean()))
    json.dump(res, open(OUT / "partb_results.json", "w"), indent=1, default=float)
    _dump("partB", res)
    print(json.dumps(dict(h4=h4, flat_rho=sec["flat"]["rho"], nn_rho=sec["nn"]["rho"]), indent=1, default=float))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("partA", part_a), ("real", part_real), ("partB", part_b)]:
        if what in (name, "all"):
            fn()

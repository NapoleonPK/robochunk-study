"""Exploratory: where do boundaries fall on the real-robot test episodes, per run (burst locations),
and how well does each rule mark gripper events (share near events, event recall).

usage: PYTHONPATH=. python scripts/real_bursts.py  ->  runs/results_real_bursts.json + printed tables
A burst = a maximal run of boundaries whose consecutive gaps are < 3 steps (a lone boundary is a burst of 1).
Event direction: episodes start with the gripper open, so the most common direction of each episode's first
event in a dataset is labelled 'grasp' and the opposite direction 'release'.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from robochunk import analysis as A
from robochunk.data_real import load_real

FROZEN = yaml.safe_load(open("configs/frozen.yaml"))
SEEDS = [0, 1, 2]
TOL, GAP, START = 3, 3, 5
REGIONS = ["start", "reach", "at_grasp", "transport", "at_release", "retreat"]


def bursts(b, gap=GAP):
    b = sorted(int(x) for x in b if x != 0)
    out = []
    for t in b:
        if out and t - out[-1][1] < gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def event_dirs(eps):
    dirs = [[int(np.sign(e["gripper"][min(t, len(e["gripper"]) - 1)] - 0.5)) for t in e["events"]] for e in eps]
    first = Counter(d[0] for d in dirs if d).most_common(1)[0][0]
    return [["grasp" if x == first else "release" for x in d] for d in dirs], first


def region(s, e, ev, kinds):
    for t, k in zip(ev, kinds):
        if s - TOL <= t <= e + TOL:
            return f"at_{k}"
    if e <= START:
        return "start"
    g = [t for t, k in zip(ev, kinds) if k == "grasp"]
    r = [t for t, k in zip(ev, kinds) if k == "release"]
    if not g or s < g[0]:
        return "reach"
    if not r or s < r[-1]:
        return "transport"
    return "retreat"


def describe(run_id, eps, kinds):
    bl = A.load_run(run_id)[1]["bounds_list"]
    reg_b, reg_steps, lens, gaps1, hit, n_ev, rows = Counter(), Counter(), [], [], 0, 0, []
    for i, (b, e, k) in enumerate(zip(bl, eps, kinds)):
        ev = list(e["events"])
        bb = np.asarray([x for x in b if x != 0])
        if len(bb) > 1:
            gaps1.append(float(np.mean(np.diff(bb) == 1)))
        for t in ev:
            n_ev += 1
            hit += bool(len(bb) and np.abs(bb - t).min() <= TOL)
        for s, en in bursts(b):
            rg = region(s, en, ev, k)
            reg_b[rg] += 1
            reg_steps[rg] += int(((bb >= s) & (bb <= en)).sum())
            lens.append(int(((bb >= s) & (bb <= en)).sum()))
            rows.append(dict(episode=i, start=s, end=en, n=lens[-1], region=rg))
    nb, ns = sum(reg_b.values()), sum(reg_steps.values())
    return dict(n_bursts=nb, n_boundaries=ns, mean_burst_len=float(np.mean(lens)) if lens else 0.0,
                share_1step_gaps=float(np.mean(gaps1)) if gaps1 else float("nan"),
                bursts_by_region={r: reg_b[r] for r in REGIONS},
                boundary_share_by_region={r: reg_steps[r] / ns if ns else 0.0 for r in REGIONS},
                event_recall=hit / n_ev if n_ev else float("nan"), bursts=rows)


def main():
    res = {}
    for name, nr in FROZEN["real_datasets"].items():
        eps = load_real(name)["test"]
        kinds, first_dir = event_dirs(eps)
        res[name] = dict(grasp_direction=first_dir, n_test_episodes=len(eps), n_events=int(sum(len(e["events"]) for e in eps)),
                         event_kinds=Counter(k for ks in kinds for k in ks), runs={})
        for m in ("learned", "entropy", "fixed"):
            for s in SEEDS:
                rid = f"real_{name}_{m}_N{nr}_s{s}"
                if (Path("runs") / rid / "test_eval.npz").exists():
                    res[name]["runs"][rid] = describe(rid, eps, kinds)
    json.dump(res, open("runs/results_real_bursts.json", "w"), indent=1, default=int)
    for name, r in res.items():
        print(f"\n## {name}: {r['n_test_episodes']} test episodes, {r['n_events']} events {dict(r['event_kinds'])}")
        print("| run | bursts | boundaries | mean burst len | 1-step gaps | event recall ±3 | "
              + " | ".join(f"% boundaries {x}" for x in REGIONS) + " |")
        print("|" + "---|" * (6 + len(REGIONS)))
        for rid, d in r["runs"].items():
            sh = d["boundary_share_by_region"]
            print(f"| {rid} | {d['n_bursts']} | {d['n_boundaries']} | {d['mean_burst_len']:.1f} | "
                  f"{d['share_1step_gaps']:.2f} | {d['event_recall']:.2f} | "
                  + " | ".join(f"{100 * sh[x]:.0f}" for x in REGIONS) + " |")


if __name__ == "__main__":
    main()

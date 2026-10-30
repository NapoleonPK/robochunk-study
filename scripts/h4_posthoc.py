"""POST-HOC H4 diagnostics (not pre-registered; requested after H4 was scored). Writes runs/results_partB_posthoc.json.

1. Why in-distribution candidates (C2, C3) lower broad-test success: gains of C1–C3, the BC training budget (fixed number
   of updates, so passes over the data shrink as data grows), and success split by test subgroup (object inside or
   outside the core square that R covers; mass; friction).
2. Sensitivity: Spearman rho(novelty, gain) over C4–C11 only (C1–C3 excluded).
3. 95% CI for the pre-registered rho over C1–C11 (n = 11): Fisher z with the Bonett–Wright standard error for Spearman,
   and a percentile bootstrap over candidates (10,000 resamples).
usage: PYTHONPATH=. python scripts/h4_posthoc.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from robochunk.data_sim import load, sample_object_xy
from robochunk.partb import CLEAN, broad_specs
from robochunk.policy import train_bc

R_ = Path("runs")
SEEDS = [0, 1, 2]
res = json.load(open(R_ / "results_partB.json"))
cand = res["candidates"]
out = {"label": "POST-HOC, not pre-registered"}

# 1a. gains of the in-distribution candidates
out["gains"] = {c: dict(gain=cand[c]["gain"], ci=cand[c]["gain_ci"]) for c in ("C1", "C2", "C3")}

# 1b. BC budget: fixed updates (train_bc defaults), so epochs = updates * batch / samples
defaults = train_bc.__defaults__  # (steps, batch, lr)
steps, batch = defaults[0], defaults[1]
n_r = sum(len(e["action"]) for e in load("data/partb/R.npz"))
budget = dict(updates=steps, batch=batch, unit="fixed number of gradient updates (not epochs)",
              samples={"R": n_r}, epochs={"R": steps * batch / n_r})
for c in ("C1", "C2", "C3", "C6"):
    n = n_r + sum(len(e["action"]) for e in load(f"data/partb/{c}.npz"))
    budget["samples"][f"R+{c}"] = n
    budget["epochs"][f"R+{c}"] = steps * batch / n
out["bc_budget"] = budget

# 1c. success by broad-test subgroup (object position reproduced from the evaluation RNG; no simulation needed)
specs = broad_specs(200)
core = np.array([np.all(np.abs(sample_object_xy(np.random.default_rng(s.seed + 7919), s.region, np.zeros(3))) <= 0.075)
                 for s in specs])
heavy = np.array([s.mass_mult > 1 for s in specs])
slippy = np.array([s.friction_mult < 1 for s in specs])
groups = {"object in core square": core, "object outside core": ~core, "mass x1": ~heavy, "mass x3": heavy,
          "friction x1": ~slippy, "low friction": slippy}
pol = R_ / "partb_policy"
succ = {n: np.stack([np.load(pol / f"{n}_s{s}.npy") for s in SEEDS]) for n in ("R", "C1", "C2", "C3", "C6")}
out["subgroups"] = {g: dict(n=int(m.sum()), **{n: float(v[:, m].mean()) for n, v in succ.items()})
                    for g, m in groups.items()}

# 2 + 3. rho sensitivity and confidence intervals
nov = {c: cand[c]["novelty"] for c in CLEAN}
gain = {c: cand[c]["gain"] for c in CLEAN}


def rho(cs):
    r, p = stats.spearmanr([nov[c] for c in cs], [gain[c] for c in cs])
    return float(r), float(p)


sub = [c for c in CLEAN if c not in ("C1", "C2", "C3")]
r_sub, p_sub = rho(sub)
out["rho_excluding_C1_C3"] = dict(candidates=sub, n=len(sub), rho=r_sub, p=p_sub)

r, p = rho(CLEAN)
n = len(CLEAN)
se = np.sqrt((1 + r ** 2 / 2) / (n - 3))
z = np.arctanh(r)
fisher = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
rng = np.random.default_rng(0)
boot = []
for _ in range(10_000):
    cs = [CLEAN[i] for i in rng.integers(0, n, n)]
    x, y = [nov[c] for c in cs], [gain[c] for c in cs]
    if len(set(x)) > 1 and len(set(y)) > 1:
        boot.append(stats.spearmanr(x, y)[0])
out["rho_ci"] = dict(rho=r, n=n, fisher_bonett_wright_95=fisher,
                     bootstrap_candidates_95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                     bootstrap_valid=len(boot))

json.dump(out, open(R_ / "results_partB_posthoc.json", "w"), indent=1)
print(json.dumps(out, indent=1))

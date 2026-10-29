"""Write the pilot v2 (Amendment 1) section of docs/PILOT.md from runs/p2_*/metrics.json."""
import json
from pathlib import Path

import pandas as pd
import yaml

rows = []
for p in sorted(Path("runs").glob("p2_*/metrics.json")):
    cfg = yaml.safe_load(open(p.parent / "config.yaml"))
    rows.append({**json.load(open(p)), "alpha": cfg.get("alpha"), "router_norm": cfg.get("router_norm")})
df = pd.DataFrame(rows).set_index("run_id")
df.reset_index().to_csv("runs/summary_pilot2.csv", index=False)


def tab(ids, cols, names):
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    for i in ids:
        r = df.loc[i]
        out.append("| " + " | ".join([i] + [str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.4g}" for c in cols]) + " |")
    return "\n".join(out)


router = [i for i in df.index if i.startswith("p2_router_")]
lr = [i for i in df.index if i.startswith("p2_lr_") and "entropy_lr" not in i]
health = [i for i in df.index if i.startswith("p2_health_lr3e-4")]
txt = f"""

## Pilot v2 — Amendment 1 (σ floor 0.1, regenerated data, 2,000 epochs, patience 150)
Pilot v1 above is superseded; its runs are archived in `runs/_v1_before_amendment/`. Table values: `runs/summary_pilot2.csv`.

Router sweep (N = 10, lr 1e-3):
{tab(router, ["alpha", "router_norm", "val_rate", "test_F1_tol3", "val_bits"], ["run", "α", "LayerNorm", "val rate", "F1 ±3", "val bits"])}

Learning-rate search (N = 10; learned at α = 0.3; entropy thresholds fitted on the validation split, proxy = p2_lr_proxy_lr1e-3):
{tab(lr, ["method", "lr", "val_bits", "best_epoch", "val_rate", "test_F1_tol3"], ["run", "method", "lr", "val bits", "best epoch", "val rate", "F1 ±3"])}

Router health at the frozen lr 3e-4:
{tab(health, ["N", "alpha", "val_rate", "test_F1_tol3"], ["run", "N", "α", "val rate", "F1 ±3"])}

Notes: on the pilot, validation and test are the same 40 episodes, so pilot F1 is not a held-out number.
The velocity runs above used a threshold fitted on the training split (rate 0.103); main runs fit it on validation.
The entropy runs `p2_lr_entropy_lr*` (training-split threshold, val rate 0.136) are superseded by `p2_lr_entropy_v2_lr*`.
Proxies at lr 1e-3 and 3e-4 reached the 2,000-epoch pilot cap still improving slightly.
"""
p = Path("docs/PILOT.md")
s = p.read_text().split("\n\n## Pilot v2")[0]
p.write_text(s.rstrip("\n") + "\n" + txt)

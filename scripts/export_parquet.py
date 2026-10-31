"""Export simulated splits to Parquet, one row per step, for the Hugging Face dataset viewer.
usage: python scripts/export_parquet.py OUT_DIR [split ...]   (default splits: test fresh)"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robochunk.data_sim import OBS, PHASES, load

AXES = "xyz"
OBS_NAMES = []
for name, s in OBS.items():
    n = s.stop - s.start
    OBS_NAMES += [f"obs_{name}_{AXES[i]}" if n == 3 else f"obs_{name}_{i}" for i in range(n)]
ACT_NAMES = ["action_dx", "action_dy", "action_dz", "action_gripper"]
GOAL_NAMES = ["goal_x", "goal_y", "goal_z"]


def table(episodes):
    cols = {k: [] for k in ["episode_index", "episode_seed", "step", "phase", "phase_name", "phase_start", "contact",
                            "success", "episode_retried", "episode_truncated"]}
    obs, act, goal = [], [], []
    for i, e in enumerate(episodes):
        T = len(e["action"])
        ph = e["phase"].astype(np.int8)
        cols["episode_index"] += [i] * T
        cols["episode_seed"] += [int(e["spec"]["seed"])] * T
        cols["step"] += list(range(T))
        cols["phase"] += ph.tolist()
        cols["phase_name"] += [PHASES[p] for p in ph]
        cols["phase_start"] += [bool(t > 0 and ph[t] != ph[t - 1]) for t in range(T)]
        cols["contact"] += e["contact"].tolist()
        cols["success"] += e["success"].tolist()
        cols["episode_retried"] += [bool(e["retried"])] * T
        cols["episode_truncated"] += [bool(e["truncated"])] * T
        obs.append(e["obs"])
        act.append(e["action"])
        goal.append(np.repeat(e["goal"][None], T, 0))
    data = {k: pa.array(v) for k, v in cols.items()}
    for names, arr in ((OBS_NAMES, np.concatenate(obs)), (ACT_NAMES, np.concatenate(act)),
                       (GOAL_NAMES, np.concatenate(goal))):
        for j, n in enumerate(names):
            data[n] = pa.array(arr[:, j].astype(np.float32))
    return pa.table(data)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for split in sys.argv[2:] or ["test", "fresh"]:
        t = table(load(f"data/sim/{split}.npz"))
        pq.write_table(t, out / f"{split}.parquet", compression="zstd")
        print(split, t.num_rows, "rows", t.num_columns, "columns")

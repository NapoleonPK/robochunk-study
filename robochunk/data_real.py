"""Real teleoperated LeRobot datasets (parquet tables only, no video).

Channels: observation.state (6 joints incl. gripper) + action (6). Native frame rate, no resampling.
Gripper events: crossings of the normalised gripper command with hysteresis (H3 reference events).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    # name: (HF repo id, gripper index in action)
    "so100": ("lerobot/svla_so100_pickplace", 5),
    "koch": ("lerobot/koch_pick_place_1_lego", 5),
    "so101": ("lerobot/svla_so101_pickplace", 5),  # v2 only (H3b-v2)
}
ROOT = Path("downloads/real")
HYST_LO, HYST_HI = 0.35, 0.65
SPLIT_SEED = 0


def revision(name):
    return json.load(open(ROOT / "revisions.json"))[DATASETS[name][0]]


def fps(name):
    rid = DATASETS[name][0]
    return json.load(open(ROOT / rid.replace("/", "__") / "meta" / "info.json"))["fps"]


def gripper_events(g, lo=HYST_LO, hi=HYST_HI):
    """Steps where a [0,1]-normalised gripper command crosses hi going up or lo going down.
    The state only flips after crossing the opposite threshold, which removes chatter."""
    events, state = [], None
    for t, v in enumerate(g):
        if state is None:
            state = "high" if v >= 0.5 else "low"
        elif state == "low" and v >= hi:
            state = "high"
            events.append(t)
        elif state == "high" and v <= lo:
            state = "low"
            events.append(t)
    return np.array(events, dtype=int)


def load_episodes(name):
    rid, gi = DATASETS[name]
    d = ROOT / rid.replace("/", "__") / "data"
    df = pd.concat([pd.read_parquet(p) for p in sorted(d.rglob("*.parquet"))]).sort_values(["episode_index", "frame_index"])
    state = np.stack(df["observation.state"].values).astype(np.float32)
    action = np.stack(df["action"].values).astype(np.float32)
    g = action[:, gi]
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)  # dataset-level normalisation of the gripper command
    gn = np.clip((g - lo) / (hi - lo), 0, 1)
    eps = []
    for e, idx in df.groupby("episode_index").indices.items():
        idx = np.sort(idx)
        eps.append(dict(episode=int(e), x=np.concatenate([state[idx], action[idx]], 1), gripper=gn[idx],
                        events=gripper_events(gn[idx])))
    return eps


def split_episodes(eps, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(eps))
    n_tr, n_va = int(round(0.8 * len(eps))), int(round(0.1 * len(eps)))
    return dict(train=[eps[i] for i in sorted(perm[:n_tr])], val=[eps[i] for i in sorted(perm[n_tr:n_tr + n_va])],
                test=[eps[i] for i in sorted(perm[n_tr + n_va:])])


def load_real(name, cfg=None):
    return split_episodes(load_episodes(name))

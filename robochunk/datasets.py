"""Dataset registry: each loader returns {split: [episode dict]} with keys
x (T,C raw float32), and when available phase, speed, events, contact."""
from __future__ import annotations

from . import data_sim
from .data import sim_channels, sim_speed


def _sim_split(path, inputs):
    """`path` may end in a slice, e.g. data/sim/pilot.npz[0:160]."""
    sl = slice(None)
    if path.endswith("]"):
        path, rng = path[:-1].split("[")
        a, b = rng.split(":")
        sl = slice(int(a) if a else None, int(b) if b else None)
    out = []
    for ep in data_sim.load(path)[sl]:
        out.append(dict(x=sim_channels(ep, inputs), phase=ep["phase"], speed=sim_speed(ep), contact=ep["contact"],
                        obs=ep["obs"], action=ep["action"], spec=ep["spec"]))
    return out


def load_dataset(cfg):
    ds = cfg["dataset"]
    if ds == "sim":
        return {role: _sim_split(f"data/sim/{name}.npz" if "[" not in name else "data/sim/" + name.replace("[", ".npz["), cfg["inputs"]) for role, name in cfg["splits"].items()}
    if ds.startswith("partb:"):
        # Part B: arbitrary npz files under data/partb, e.g. partb:R
        return {role: _sim_split(f"data/partb/{name}.npz", cfg["inputs"]) for role, name in cfg["splits"].items()}
    if ds.startswith("real:"):
        from .data_real import load_real
        return load_real(ds.split(":", 1)[1], cfg)
    raise ValueError(ds)

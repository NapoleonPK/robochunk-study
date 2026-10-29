"""Part B datasets: reference R, candidates, corruptions, and the broad test distribution."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_sim import OBS, EpisodeSpec, generate, load, save

OUT = Path("data/partb")
N_CAND = 200
LOW_FRICTION = 0.2
SEED = dict(R=300_000, R_heldout=310_000, R_val=320_000, C3=330_000, C4=340_000, C5=350_000, C6=360_000,
            C7=370_000, C8=380_000, C9=390_000, C10=400_000, C11=410_000, X=420_000, broad=500_000)
CLEAN = [f"C{i}" for i in range(1, 12)]
CORRUPT = ["X1", "X2", "X3", "X4"]


def core(seed, **kw):
    return EpisodeSpec(seed=seed, region="core", **kw)


def broad_specs(n=200, seed0=None):
    """Policy evaluation distribution: whole table, mass x1 or x3, friction x1 or low, each with prob 1/2."""
    seed0 = SEED["broad"] if seed0 is None else seed0
    rng = np.random.default_rng(seed0)
    return [EpisodeSpec(seed=seed0 + i, region="whole", mass_mult=float(rng.choice([1.0, 3.0])),
                        friction_mult=float(rng.choice([1.0, LOW_FRICTION]))) for i in range(n)]


def _gen(name, specs):
    eps = generate(specs)
    save(OUT / f"{name}.npz", eps, specs)
    return eps


def _lag_state(ep, k=3):
    """X1: the logged observation lags the action by k steps."""
    o = ep["obs"]
    idx = np.maximum(np.arange(len(o)) - k, 0)
    return dict(ep, obs=o[idx])


def _drop_frames(ep, rng, frac=0.2):
    keep = np.sort(rng.choice(len(ep["phase"]), int(round(len(ep["phase"]) * (1 - frac))), replace=False))
    return dict(ep, obs=ep["obs"][keep], action=ep["action"][keep], phase=ep["phase"][keep],
                contact=ep["contact"][keep], success=ep["success"][keep])


def _invert_gripper(ep):
    a = ep["action"].copy()
    a[:, 3] *= -1
    return dict(ep, action=a)


def build_all(only=None, r_size=500):
    OUT.mkdir(parents=True, exist_ok=True)
    want = lambda n: only is None or n in only  # noqa: E731
    R_specs = [core(SEED["R"] + i) for i in range(r_size)]
    if want("R"):
        R = _gen("R", R_specs)
    else:
        R = load(OUT / "R.npz")
    if want("R_heldout"):
        _gen("R_heldout", [core(SEED["R_heldout"] + i) for i in range(100)])
    if want("R_val"):
        _gen("R_val", [core(SEED["R_val"] + i) for i in range(50)])
    n = N_CAND
    if want("C1"):  # exact duplicates of R episodes
        save(OUT / "C1.npz", R[:n], R_specs[:n])
    if want("C2"):  # same seeds, new small action noise
        _gen("C2", [core(SEED["R"] + i, noise_seed=10_000_000 + i) for i in range(n)])
    if want("C3"):
        _gen("C3", [core(SEED["C3"] + i) for i in range(n)])
    if want("C4"):
        _gen("C4", [EpisodeSpec(seed=SEED["C4"] + i, region="left") for i in range(n)])
    if want("C5"):
        _gen("C5", [EpisodeSpec(seed=SEED["C5"] + i, region="far") for i in range(n)])
    if want("C6"):
        _gen("C6", [EpisodeSpec(seed=SEED["C6"] + i, region="whole") for i in range(n)])
    if want("C7"):
        _gen("C7", [core(SEED["C7"] + i, mass_mult=3.0) for i in range(n)])
    if want("C8"):
        _gen("C8", [core(SEED["C8"] + i, friction_mult=LOW_FRICTION) for i in range(n)])
    if want("C9"):
        _gen("C9", [EpisodeSpec(seed=SEED["C9"] + i, region="whole", mass_mult=3.0) for i in range(n)])
    if want("C10"):
        _gen("C10", [core(SEED["C10"] + i, speed_mult=0.6) for i in range(n)])
    if want("C11"):  # half new left-side episodes, half duplicates of R
        specs = [EpisodeSpec(seed=SEED["C11"] + i, region="left") for i in range(n // 2)]
        eps = generate(specs)
        save(OUT / "C11.npz", eps + R[n // 2:n], specs + R_specs[n // 2:n])
    # corruptions start from fresh core episodes (the same base for X1, X3, X4)
    if any(want(x) for x in CORRUPT):
        base_specs = [core(SEED["X"] + i) for i in range(n)]
        base = generate(base_specs)
        rng = np.random.default_rng(SEED["X"])
        if want("X1"):
            save(OUT / "X1.npz", [_lag_state(e) for e in base], base_specs)
        if want("X2"):  # heavy noise on executed actions: the demonstrations themselves are degraded
            _gen("X2", [core(SEED["X"] + i, action_noise=0.5) for i in range(n)])
        if want("X3"):
            save(OUT / "X3.npz", [_drop_frames(e, rng) for e in base], base_specs)
        if want("X4"):
            inv = rng.random(n) < 0.3
            save(OUT / "X4.npz", [_invert_gripper(e) if f else e for e, f in zip(base, inv)], base_specs)

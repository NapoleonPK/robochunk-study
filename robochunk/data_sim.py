"""Simulated Fetch pick-and-place episodes with per-step phase labels.

A scripted expert runs seven phases (approach, descend, close, lift, transport,
open, retreat). Phase boundaries are deliberately hard to see: minimum-jerk
profiles that overlap across phases (superposition), random per-phase speed,
pauses placed inside phases, Gaussian action noise and occasional failed
grasps followed by a retry.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool

import numpy as np

PHASES = ["approach", "descend", "close", "lift", "transport", "open", "retreat"]
ENV_ID = "FetchPickAndPlace-v4"
MAX_STEPS = 150
# Base phase durations in steps at speed 1 (25 Hz control).
BASE_DUR = {"approach": 13, "descend": 8, "close": 5, "lift": 7, "transport": 15, "open": 4, "retreat": 7}
HOVER = 0.10      # approach height above object centre
LIFT = 0.08       # lift height
RETREAT = 0.10
MAX_STEP_M = 0.05  # Fetch moves the mocap at most 5 cm per action
# Observation layout of Fetch (25 dims).
OBS = dict(grip_pos=slice(0, 3), object_pos=slice(3, 6), object_rel=slice(6, 9), fingers=slice(9, 11),
           object_rot=slice(11, 14), object_velp=slice(14, 17), object_velr=slice(17, 20),
           grip_velp=slice(20, 23), finger_vel=slice(23, 25))


@dataclass
class EpisodeSpec:
    """Everything that varies between episodes; fully determined by `seed` and the distribution."""
    seed: int
    region: str = "core"          # core | left | far | whole
    mass_mult: float = 1.0
    friction_mult: float = 1.0
    speed_mult: float = 1.0       # global expert speed (C10 uses 0.6)
    action_noise: float = 0.05    # std of Gaussian noise on executed actions
    retry_prob: float = 0.10      # probability of an intentionally failed first grasp
    pause_max: int = 5
    overlap_max: int = 3
    speed_range: tuple = (0.5, 1.5)
    noise_seed: int | None = None  # separate RNG for action noise (near-duplicates: same seed, new noise)


def _min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


@dataclass
class _Profile:
    """One phase's minimum-jerk displacement with an optional pause inside it."""
    start: int
    move_steps: int
    delta: np.ndarray        # displacement of gripper target (3,) plus gripper command (1,)
    pause_at: int = 0        # steps into the phase at which the pause starts
    pause_len: int = 0
    total: int = field(init=False)

    def __post_init__(self):
        self.total = self.move_steps + self.pause_len

    def value(self, t):
        k = t - self.start
        if k <= 0:
            return np.zeros_like(self.delta)
        if self.pause_len and k > self.pause_at:
            k = max(self.pause_at, k - self.pause_len)
        return self.delta * _min_jerk(k / self.move_steps)


def sample_object_xy(rng, region, center):
    r = 0.15
    while True:
        if region == "core":
            d = rng.uniform(-r / 2, r / 2, 2)
        elif region == "whole":
            d = rng.uniform(-r, r, 2)
        elif region == "left":   # +y side of the table, outside the core square
            d = np.array([rng.uniform(-r, r), rng.uniform(r / 2, r)])
        elif region == "far":    # +x side, away from the robot base
            d = np.array([rng.uniform(r / 2, r), rng.uniform(-r, r)])
        else:
            raise ValueError(region)
        if np.linalg.norm(d) >= 0.03 or region != "core":
            return center[:2] + d


class Expert:
    def __init__(self, spec: EpisodeSpec, rng):
        self.spec, self.rng = spec, rng
        self.profiles: list[_Profile] = []
        self.phase_starts: list[tuple[int, int]] = []  # (t, phase index)
        self.waypoint = None  # planned (x, y, z, grip) after the last started phase

    def _duration(self, name, dist):
        lo, hi = self.spec.speed_range
        speed = self.rng.uniform(lo, hi) * self.spec.speed_mult
        base = BASE_DUR[name] / speed
        # min-jerk peak speed is 1.875 d / T; keep it under the per-step limit
        need = 1.875 * dist / (0.8 * MAX_STEP_M)
        return int(max(3, round(max(base, need))))

    def start_phase(self, t, name, target):
        target = np.asarray(target, dtype=np.float64)
        delta = target - self.waypoint
        steps = self._duration(name, np.linalg.norm(delta[:3]))
        pause_len = int(self.rng.integers(0, self.spec.pause_max + 1))
        pause_at = int(self.rng.integers(1, steps)) if pause_len else 0
        p = _Profile(t, steps, delta, pause_at, pause_len)
        self.profiles.append(p)
        self.phase_starts.append((t, PHASES.index(name)))
        self.waypoint = target
        overlap = int(self.rng.integers(0, self.spec.overlap_max + 1))
        return t + max(1, p.total - overlap)  # time the next phase starts

    def target(self, t, origin):
        return origin + sum((p.value(t) for p in self.profiles), np.zeros(4))


def _contact(model, data, finger_geoms, obj_geom):
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == obj_geom and c.geom2 in finger_geoms) or (c.geom2 == obj_geom and c.geom1 in finger_geoms):
            return True
    return False


_ENV = None


def _get_env():
    global _ENV
    if _ENV is None:
        warnings.filterwarnings("ignore")
        import gymnasium as gym
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
        _ENV = gym.make(ENV_ID, max_episode_steps=MAX_STEPS).unwrapped
        import mujoco
        m = _ENV.model
        _ENV._rc = dict(
            body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object0"),
            geom=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object0"),
            fingers={mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"robot0:{s}_gripper_finger_link") for s in "lr"},
        )
        _ENV._rc["mass0"] = float(m.body_mass[_ENV._rc["body"]])
        _ENV._rc["fric0"] = m.geom_friction[_ENV._rc["geom"]].copy()
        _ENV.target_in_the_air = False  # goals on the table: the expert places, not drops
    return _ENV


def reset_env(env, spec: EpisodeSpec, rng):
    """Reset with the spec's object region, physics and a table goal. Returns the observation dict."""
    rc = env._rc
    env.model.body_mass[rc["body"]] = rc["mass0"] * spec.mass_mult
    env.model.geom_friction[rc["geom"]] = rc["fric0"] * np.array([spec.friction_mult, 1, 1])
    env.reset(seed=int(spec.seed))
    center = env.initial_gripper_xpos
    obj_xy = sample_object_xy(rng, spec.region, center)
    qpos = env._utils.get_joint_qpos(env.model, env.data, "object0:joint").copy()
    qpos[:2] = obj_xy
    env._utils.set_joint_qpos(env.model, env.data, "object0:joint", qpos)
    while True:
        goal = center[:2] + rng.uniform(-0.15, 0.15, 2)
        if np.linalg.norm(goal - obj_xy) > 0.10:
            break
    env.goal = np.array([goal[0], goal[1], env.height_offset])
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    return env._get_obs()


def run_episode(spec: EpisodeSpec):
    env = _get_env()
    rng = np.random.default_rng(spec.seed + 7919)
    o = reset_env(env, spec, rng)
    rc = env._rc
    obs0 = o["observation"]
    ex = Expert(spec, rng)
    origin = np.concatenate([obs0[OBS["grip_pos"]], [1.0]])
    ex.waypoint = origin.copy()

    def obj():
        return env._get_obs()["observation"][OBS["object_pos"]].copy()

    retry_planned = rng.random() < spec.retry_prob
    plan = ["approach", "descend", "close", "lift", "transport", "open", "retreat"]
    retried = False
    grasp_z = None

    def target_for(name):
        w = ex.waypoint
        op = obj()
        if name == "approach":
            return [op[0], op[1], op[2] + HOVER, 1.0]
        if name == "descend":
            off = np.zeros(2)
            if retry_planned and not retried:
                off = np.array([rng.choice([-1, 1]) * rng.uniform(0.045, 0.06), 0.0])  # miss along x
            return [op[0] + off[0], op[1] + off[1], op[2], 1.0]
        if name == "close":
            return [w[0], w[1], w[2], -1.0]
        if name == "lift":
            return [w[0], w[1], w[2] + LIFT, -1.0]
        if name == "transport":
            g = env.goal
            return [g[0], g[1], g[2] + 0.01, -1.0]
        if name == "open":
            return [w[0], w[1], w[2], 1.0]
        if name == "retreat":
            return [w[0], w[1], w[2] + RETREAT, 1.0]
        raise ValueError(name)

    noise_rng = rng if spec.noise_seed is None else np.random.default_rng(spec.noise_seed)
    rec = {k: [] for k in ["obs", "action", "phase", "contact", "success"]}
    queue = list(plan)
    next_start = 0
    phase = 0
    finished_at = None
    for t in range(MAX_STEPS):
        while queue and t >= next_start:
            name = queue.pop(0)
            if name == "transport" and not retried:
                # check the grasp: object must have risen with the gripper
                if obj()[2] < grasp_z + 0.02:
                    retried = True
                    queue = ["open", "approach", "descend", "close", "lift", "transport", "open", "retreat"]
                    name = queue.pop(0)
            if name == "descend":
                grasp_z = obj()[2]
            next_start = ex.start_phase(t, name, target_for(name))
            phase = PHASES.index(name)
            if not queue:
                finished_at = t + ex.profiles[-1].total
        if finished_at is not None and t >= finished_at:
            break
        cur = env._get_obs()["observation"]
        tgt = ex.target(t + 1, origin)
        a = np.empty(4)
        a[:3] = (tgt[:3] - cur[OBS["grip_pos"]]) / MAX_STEP_M
        a[3] = tgt[3]
        a += noise_rng.normal(0, spec.action_noise, 4)
        a = np.clip(a, -1, 1)
        rec["obs"].append(cur.copy())
        rec["action"].append(a.copy())
        rec["phase"].append(phase)
        rec["contact"].append(_contact(env.model, env.data, rc["fingers"], rc["geom"]))
        env.step(a)
        rec["success"].append(bool(np.linalg.norm(obj() - env.goal) < 0.05))
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["goal"] = env.goal.copy()
    out["retried"] = retried
    out["truncated"] = finished_at is None or len(rec["obs"]) >= MAX_STEPS
    return out


def _worker(spec_dict):
    return run_episode(EpisodeSpec(**spec_dict))


def generate(specs: list[EpisodeSpec], workers: int | None = None):
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with Pool(workers) as pool:
        return pool.map(_worker, [asdict(s) for s in specs], chunksize=4)


def save(path, episodes, specs):
    """Concatenate episodes into one npz with offsets; spec metadata goes in a JSON string."""
    lens = np.array([len(e["phase"]) for e in episodes])
    np.savez_compressed(
        path,
        offsets=np.concatenate([[0], np.cumsum(lens)]),
        obs=np.concatenate([e["obs"] for e in episodes]).astype(np.float32),
        action=np.concatenate([e["action"] for e in episodes]).astype(np.float32),
        phase=np.concatenate([e["phase"] for e in episodes]).astype(np.int8),
        contact=np.concatenate([e["contact"] for e in episodes]).astype(bool),
        success=np.concatenate([e["success"] for e in episodes]).astype(bool),
        goal=np.stack([e["goal"] for e in episodes]).astype(np.float32),
        retried=np.array([e["retried"] for e in episodes]),
        truncated=np.array([e["truncated"] for e in episodes]),
        specs=json.dumps([asdict(s) for s in specs]),
    )


def load(path):
    """Returns a list of episode dicts (obs, action, phase, contact, success, goal, retried, truncated, spec)."""
    with np.load(path, allow_pickle=False) as npz:
        z = {k: npz[k] for k in npz.files}  # decompress each array once
    off = z["offsets"]
    specs = json.loads(str(z["specs"]))
    eps = []
    for i in range(len(off) - 1):
        s = slice(off[i], off[i + 1])
        eps.append(dict(obs=z["obs"][s], action=z["action"][s], phase=z["phase"][s], contact=z["contact"][s],
                        success=z["success"][s], goal=z["goal"][i], retried=bool(z["retried"][i]),
                        truncated=bool(z["truncated"][i]), spec=specs[i]))
    return eps

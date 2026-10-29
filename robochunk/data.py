"""Channel selection, normalisation and padded batching shared by sim and real data."""
from __future__ import annotations

import json

import numpy as np
import torch

from .data_sim import OBS


def sim_channels(ep, inputs="robot"):
    """Robot-only channels: gripper position (3), gripper velocity (3), finger width (1), action (4).
    inputs='robot+object' appends object position (3)."""
    o = ep["obs"]
    cols = [o[:, OBS["grip_pos"]], o[:, OBS["grip_velp"]], o[:, OBS["fingers"]].sum(1, keepdims=True), ep["action"]]
    if inputs == "robot+object":
        cols.append(o[:, OBS["object_pos"]])
    return np.concatenate(cols, 1).astype(np.float32)


def sim_speed(ep):
    """End-effector speed per step, for the velocity-minima baseline."""
    return np.linalg.norm(ep["obs"][:, OBS["grip_velp"]], axis=1)


class Normalizer:
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    @classmethod
    def fit(cls, seqs):
        cat = np.concatenate(seqs)
        return cls(cat.mean(0), np.maximum(cat.std(0), 1e-6))

    def __call__(self, x):
        return (x - self.mean) / self.std

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(mean=self.mean.tolist(), std=self.std.tolist()), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["mean"], d["std"])


def pad_batch(seqs, boundaries=None, device="cpu"):
    """Right-pad a list of (T_i, C) arrays. Returns x (B,T,C), mask (B,T), b (B,T) or None."""
    T = max(len(s) for s in seqs)
    B, C = len(seqs), seqs[0].shape[1]
    x = np.zeros((B, T, C), np.float32)
    m = np.zeros((B, T), bool)
    bb = np.zeros((B, T), np.int8) if boundaries is not None else None
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        m[i, :len(s)] = True
        if boundaries is not None:
            bb[i, :len(s)] = boundaries[i]
    out = (torch.from_numpy(x).to(device), torch.from_numpy(m).to(device))
    return out + ((torch.from_numpy(bb).to(device) if bb is not None else None),)

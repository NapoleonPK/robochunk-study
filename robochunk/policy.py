"""Behaviour-cloning MLP policy (full observation + goal -> action) and simulator evaluation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data_sim import MAX_STEPS, _get_env, reset_env

HIDDEN = 256


class MLP(nn.Module):
    def __init__(self, d_in=28, d_out=4, h=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, d_out), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def bc_arrays(episodes):
    X = np.concatenate([np.concatenate([e["obs"], np.repeat(e["goal"][None], len(e["obs"]), 0)], 1) for e in episodes])
    Y = np.concatenate([e["action"] for e in episodes])
    return X.astype(np.float32), Y.astype(np.float32)


def train_bc(episodes, seed, steps=20_000, batch=512, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, Y = bc_arrays(episodes)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = torch.from_numpy((X - mu) / sd)
    Yt = torch.from_numpy(Y)
    model = MLP(X.shape[1], Y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.from_numpy(rng.integers(0, len(X), batch))
        loss = ((model(Xn[i]) - Yt[i]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return dict(model=model, mu=mu, sd=sd, final_loss=float(loss))


def evaluate_policy(pol, specs):
    """Roll out on each spec (same object/goal placement as the expert would get). Returns 0/1 successes
    at the last step (object within 5 cm of the goal)."""
    env = _get_env()
    out = []
    for spec in specs:
        rng = np.random.default_rng(spec.seed + 7919)
        o = reset_env(env, spec, rng)
        for _ in range(MAX_STEPS):
            x = np.concatenate([o["observation"], env.goal]).astype(np.float32)
            with torch.no_grad():
                a = pol["model"](torch.from_numpy((x - pol["mu"]) / pol["sd"])[None])[0].numpy()
            o, *_ = env.step(a)
        out.append(float(np.linalg.norm(o["achieved_goal"] - env.goal) < 0.05))
    return np.array(out)

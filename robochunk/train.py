"""Train and evaluate one run from a YAML config. Resumable; checkpoints every 10 minutes.

usage: python -m robochunk.train configs/<run>.yaml
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml

from . import baselines as BL
from .data import Normalizer, pad_batch
from .datasets import load_dataset
from .metrics import episode_f1
from .model import Proxy, RoboChunk, gaussian_nll, ratio_loss, step_surprise

DEFAULTS = dict(
    method="learned",        # learned | fixed | random | entropy | velocity | flat | proxy
    dataset="sim", inputs="robot", encoder="gru", N=10, seed=0,
    lr=1e-3, weight_decay=0.01, batch=64, warmup_steps=200, max_epochs=300, patience=20,
    max_minutes=30, alpha=0.03, router_norm=False, identity_init=True, device="cpu",
    d0=128, d1=256, main_layers=4, heads=4, proxy_run=None, splits=dict(train="train", val="val", test="test"),
    eval_only=False,
)
CKPT_EVERY_S = 600
LN2 = math.log(2)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def episode_seed(split, i):
    return zlib.crc32(f"{split}:{i}".encode())


def build_model(cfg, n_ch):
    if cfg["method"] == "proxy":
        return Proxy(n_ch)
    return RoboChunk(n_ch, cfg["d0"], cfg["d1"], cfg["main_layers"], cfg["heads"], cfg["encoder"],
                     cfg["router_norm"], cfg["identity_init"])


def proxy_surprise(proxy_dir, split_eps, device):
    """Per-episode NLL of x(t) | x(<t) from a trained proxy run (in nats, step 0 = inf)."""
    pcfg = yaml.safe_load(open(Path(proxy_dir) / "config.yaml"))
    norm = Normalizer.load(Path(proxy_dir) / "norm.json")
    model = Proxy(len(norm.mean)).to(device)
    model.load_state_dict(torch.load(Path(proxy_dir) / "best.pt", map_location=device))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(split_eps), 128):
            seqs = [norm(e["x"]) for e in split_eps[i:i + 128]]
            x, m, _ = pad_batch(seqs, device=device)
            s = step_surprise(model, x, m).cpu().numpy()
            out += [s[j, :len(q)] for j, q in enumerate(seqs)]
    assert pcfg["inputs"] == pcfg["inputs"]
    return out


def make_boundaries(cfg, data, device):
    """Boundary arrays per split for non-learned rules (None for learned / proxy).
    Thresholds for velocity and entropy are fitted on the training split only."""
    method, N = cfg["method"], cfg["N"]
    if method in ("learned", "proxy"):
        return {k: None for k in data}, {}
    out, info = {}, {}
    if method == "fixed":
        out = {k: [BL.fixed_boundaries(len(e["x"]), N) for e in v] for k, v in data.items()}
    elif method == "flat":
        out = {k: [np.ones(len(e["x"]), np.int8) for e in v] for k, v in data.items()}
    elif method == "random":
        out = {k: [BL.random_boundaries(len(e["x"]), N, episode_seed(k, i) + cfg["seed"]) for i, e in enumerate(v)]
               for k, v in data.items()}
    elif method == "velocity":
        _, thr = BL.velocity_boundaries([e["speed"] for e in data["train"]], N)
        out = {k: BL.velocity_boundaries([e["speed"] for e in v], N, thr)[0] for k, v in data.items()}
        info["threshold"] = float(thr)
    elif method == "entropy":
        sur = {k: proxy_surprise(cfg["proxy_run"], v, device) for k, v in data.items()}
        _, thr = BL.entropy_boundaries(sur["train"], N)
        out = {k: BL.entropy_boundaries(s, N, thr)[0] for k, s in sur.items()}
        info["threshold"] = float(thr)
    else:
        raise ValueError(method)
    return out, info


def batch_loss(model, cfg, x, m, b):
    out = model(x, m, b)
    nll, valid = gaussian_nll(out, x, m)
    loss = nll[valid].mean()
    rl = torch.zeros((), device=x.device)
    if cfg["method"] == "learned":
        rl = ratio_loss(out["b"], out["p"], m, cfg["N"])
        loss = loss + cfg["alpha"] * rl
    return loss, nll, valid, out, rl


def evaluate(model, cfg, eps, bnds, norm, device, keep=False):
    """Mean NLL (bits/step), MSE, per-episode NLL, boundaries (as index arrays), boundary rate."""
    model.eval()
    tot_nll = tot_se = n = 0.0
    ep_bits, bounds, nb = [], [], 0
    with torch.no_grad():
        for i in range(0, len(eps), 128):
            seqs = [norm(e["x"]) for e in eps[i:i + 128]]
            bb = None if bnds is None else bnds[i:i + 128]
            x, m, b = pad_batch(seqs, bb, device)
            out = model(x, m, b)
            nll, valid = gaussian_nll(out, x, m)
            se = ((out["mu"][:, :-1] - x[:, 1:]) ** 2).mean(-1)
            tot_nll += nll[valid].sum().item()
            tot_se += se[valid].sum().item()
            n += valid.sum().item()
            for j, s in enumerate(seqs):
                v = valid[j]
                ep_bits.append(nll[j][v].mean().item() / LN2 if v.any() else float("nan"))
                if keep and "b" in out:
                    bounds.append(np.flatnonzero(out["b"][j, :len(s)].cpu().numpy()))
                if "b" in out:
                    nb += int(out["b"][j, :len(s)].sum())
    steps = sum(len(e["x"]) for e in eps)
    return dict(bits=tot_nll / n / LN2, mse=tot_se / n, ep_bits=ep_bits, bounds=bounds,
                rate=nb / steps if nb else None)


def flops_per_step(model, cfg, eps, bnds, norm, device):
    """Measured FLOPs per valid step (torch.utils.flop_counter) plus analytic GRU FLOPs,
    which the counter does not see."""
    from torch.utils.flop_counter import FlopCounterMode
    seqs = [norm(e["x"]) for e in eps[:64]]
    x, m, b = pad_batch(seqs, None if bnds is None else bnds[:64], device)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x, m, b)
    counted = fc.get_total_flops()
    gru = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.GRU):
            for layer in range(mod.num_layers):
                d_in = mod.input_size if layer == 0 else mod.hidden_size
                gru += 6 * mod.hidden_size * (d_in + mod.hidden_size)
    padded_steps = x.shape[0] * x.shape[1]
    valid = int(m.sum())
    return dict(counted_per_step=counted / valid, gru_per_step=gru * padded_steps / valid,
                total_per_step=(counted + gru * padded_steps) / valid)


def log(run_dir, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(run_dir / "log.txt", "a") as f:
        f.write(line + "\n")


def main(cfg_path):
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 6)))
    cfg = {**DEFAULTS, **yaml.safe_load(open(cfg_path))}
    for k in ("lr", "weight_decay", "alpha", "max_minutes"):
        cfg[k] = float(cfg[k])  # PyYAML reads "3e-4" as a string
    run_dir = Path("runs") / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))
    if not (run_dir / "git.txt").exists():
        (run_dir / "git.txt").write_text(git_hash() + "\n")
    if (run_dir / "metrics.json").exists() and not cfg["eval_only"]:
        print("already finished:", run_dir)
        return
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    data = load_dataset(cfg)
    norm = Normalizer.fit([e["x"] for e in data["train"]])
    norm.save(run_dir / "norm.json")
    bnds, binfo = make_boundaries(cfg, data, device)
    model = build_model(cfg, data["train"][0]["x"].shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_train = len(data["train"])
    steps_per_epoch = math.ceil(n_train / cfg["batch"])
    total_steps = cfg["max_epochs"] * steps_per_epoch

    def lr_at(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        q = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, q)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    state = dict(epoch=0, step=0, best=float("inf"), best_epoch=-1, elapsed=0.0, history=[])
    ck = run_dir / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        state = s["state"]
        np.random.set_state(s["np_rng"])
        torch.set_rng_state(s["torch_rng"])
        log(run_dir, f"resumed at epoch {state['epoch']}")
    else:
        log(run_dir, f"start {cfg['run_id']} params={n_params} device={device} train={n_train} {binfo}")

    def save_ckpt():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=state,
                        np_rng=np.random.get_state(), torch_rng=torch.get_rng_state()), ck)

    t0, last_ck = time.time() - state["elapsed"], time.time()
    rng = np.random.default_rng(cfg["seed"] + 1000 * state["epoch"])
    stop_reason = "max_epochs"
    while state["epoch"] < cfg["max_epochs"] and not cfg["eval_only"]:
        model.train()
        perm = rng.permutation(n_train)
        tr_loss = tr_ratio = 0.0
        for bi in range(steps_per_epoch):
            ids = perm[bi * cfg["batch"]:(bi + 1) * cfg["batch"]]
            seqs = [norm(data["train"][i]["x"]) for i in ids]
            bb = None if bnds["train"] is None else [bnds["train"][i] for i in ids]
            x, m, b = pad_batch(seqs, bb, device)
            loss, _, _, _, rl = batch_loss(model, cfg, x, m, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            state["step"] += 1
            tr_loss += loss.item()
            tr_ratio += rl.item()
        state["epoch"] += 1
        ev = evaluate(model, cfg, data["val"], bnds["val"], norm, device)
        state["elapsed"] = time.time() - t0
        state["history"].append(dict(epoch=state["epoch"], train=tr_loss / steps_per_epoch, val_bits=ev["bits"],
                                     rate=ev["rate"], ratio=tr_ratio / steps_per_epoch, t=state["elapsed"]))
        if ev["bits"] < state["best"]:
            state["best"], state["best_epoch"] = ev["bits"], state["epoch"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        if state["epoch"] % 5 == 0 or state["epoch"] == 1:
            r = f"{ev['rate']:.4f}" if ev["rate"] else "-"
            log(run_dir, f"ep {state['epoch']} train {tr_loss / steps_per_epoch:.4f} val_bits {ev['bits']:.4f} "
                         f"rate {r} best {state['best']:.4f}@{state['best_epoch']} t={state['elapsed']:.0f}s")
        if time.time() - last_ck > CKPT_EVERY_S:
            save_ckpt()
            last_ck = time.time()
        if state["epoch"] - state["best_epoch"] >= cfg["patience"]:
            stop_reason = "early_stop"
            break
        if state["elapsed"] > cfg["max_minutes"] * 60:
            stop_reason = "time_budget"
            break
    save_ckpt()
    # final evaluation with the best checkpoint
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    res = dict(run_id=cfg["run_id"], method=cfg["method"], N=cfg["N"], seed=cfg["seed"], dataset=cfg["dataset"],
               inputs=cfg["inputs"], encoder=cfg["encoder"], lr=cfg["lr"], params=n_params,
               epochs=state["epoch"], best_epoch=state["best_epoch"], stop=stop_reason,
               minutes=state["elapsed"] / 60, git=(run_dir / "git.txt").read_text().strip(), **binfo)
    for split in ("val", "test"):
        ev = evaluate(model, cfg, data[split], bnds[split], norm, device, keep=True)
        res[f"{split}_bits"], res[f"{split}_mse"] = ev["bits"], ev["mse"]
        if cfg["method"] != "proxy":
            res[f"{split}_rate"] = ev["rate"]
            res[f"{split}_compression"] = 1 / ev["rate"]
        if split == "test":
            arr = dict(ep_bits=np.array(ev["ep_bits"]))
            if ev["bounds"]:
                lens = np.array([len(e["x"]) for e in data[split]])
                flat = np.concatenate(ev["bounds"])
                arr.update(bounds=flat, bounds_off=np.concatenate([[0], np.cumsum([len(b) for b in ev["bounds"]])]),
                           lengths=lens)
                if "phase" in data[split][0]:
                    for tol in (1, 3, 5):
                        f = episode_f1(ev["bounds"], [e["phase"] for e in data[split]], tol)
                        arr[f"prf_tol{tol}"] = f
                        res[f"test_P_tol{tol}"], res[f"test_R_tol{tol}"], res[f"test_F1_tol{tol}"] = map(float, f.mean(0))
            np.savez(run_dir / "test_eval.npz", **arr)
    if cfg["method"] != "proxy":
        res.update({f"flops_{k}": v for k, v in flops_per_step(model, cfg, data["test"], bnds["test"], norm, device).items()})
    json.dump(res, open(run_dir / "metrics.json", "w"), indent=1)
    json.dump(state["history"], open(run_dir / "history.json", "w"))
    log(run_dir, "done " + json.dumps({k: v for k, v in res.items() if k.startswith("test_") or k == "stop"}))


if __name__ == "__main__":
    main(sys.argv[1])

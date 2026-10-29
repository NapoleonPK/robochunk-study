"""Figures 1–6 from run outputs. usage: python scripts/make_figures.py [fig ...]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from robochunk.analysis import load_run, paired_bootstrap
from robochunk.data_sim import PHASES, load
from robochunk.metrics import true_boundaries

FIG = Path("figures")
FROZEN = yaml.safe_load(open("configs/frozen.yaml"))
NM = FROZEN["N_main"]
NS = [NM // 2, NM, 2 * NM]
SEEDS = [0, 1, 2]
# categorical slots in fixed order (validated reference palette, light mode)
COL = dict(learned="#2a78d6", fixed="#eb6834", random="#1baf7a", entropy="#eda100", velocity="#e87ba4", flat="#52514e")
NAME = dict(learned="RoboChunk (learned)", fixed="Fixed", random="Random", entropy="Entropy (BLT-like)",
            velocity="Velocity minima", flat="Flat (no chunking)")
METHODS = ["learned", "fixed", "random", "entropy", "velocity"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False})


def fig1(n_eps=6):
    """Six test episodes: true phases as bands, each method's boundaries as ticks (seed 0, N_main)."""
    test = load("data/sim/test.npz")
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(test), n_eps, replace=False))
    bnd = {m: load_run(f"sim_{m}_N{NM}_s0")[1]["bounds_list"] for m in METHODS}
    fig, axes = plt.subplots(n_eps, 1, figsize=(7.5, 1.25 * n_eps), sharex=False)
    for ax, i in zip(axes, pick):
        ph = test[i]["phase"]
        T = len(ph)
        starts = np.concatenate([[0], true_boundaries(ph), [T]])
        for k in range(len(starts) - 1):
            a, b = starts[k], starts[k + 1]
            ax.axvspan(a - 0.5, b - 0.5, ymin=0.78, ymax=1.0, color=["#e6e5e0", "#cfcdc6"][k % 2], lw=0)
            if b - a >= 6:
                ax.text((a + b - 1) / 2, len(METHODS) + 0.55, PHASES[ph[a]][:5], ha="center", va="center", fontsize=6.5,
                        color=INK)
        for tb in starts[1:-1]:
            ax.axvline(tb - 0.5, color=MUTED, lw=0.5, ls=":")
        for j, m in enumerate(METHODS):
            y = len(METHODS) - 1 - j
            b = [x for x in bnd[m][i] if x != 0]
            ax.vlines(b, y - 0.35, y + 0.35, color=COL[m], lw=1.6)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([NAME[m].split(" (")[0] for m in METHODS[::-1]], fontsize=6.5)
        ax.set_ylim(-0.6, len(METHODS) + 1.0)
        ax.set_xlim(-1, T)
        ax.grid(False)
        ax.set_title(f"test episode {i} ({T} steps)", fontsize=7, loc="left", color=MUTED, pad=2)
    axes[-1].set_xlabel("step (25 Hz)")
    fig.suptitle(f"True phases (bands) and boundaries placed by each rule, N = {NM}", fontsize=9, x=0.02, ha="left")
    fig.savefig(FIG / "fig1_episodes.png")


def fig2():
    """Boundary F1 (±3) by method and N, mean of seeds with 95% bootstrap CI over test episodes."""
    fig, ax = plt.subplots(figsize=(6.5, 3))
    w = 0.16
    for j, m in enumerate(METHODS):
        for k, N in enumerate(NS):
            f = np.mean([load_run(f"sim_{m}_N{N}_s{s}")[1]["prf_tol3"][:, 2] for s in SEEDS], 0)
            mean, lo, hi = paired_bootstrap(f)
            x = k + (j - 2) * w
            ax.bar(x, mean, w * 0.88, color=COL[m], label=NAME[m] if k == 0 else None)
            ax.errorbar(x, mean, [[mean - lo], [hi - mean]], color=INK, lw=0.8, capsize=0)
    ax.set_xticks(range(len(NS)))
    ax.set_xticklabels([f"N = {N}" for N in NS])
    ax.set_ylabel("boundary F1 (±3 steps)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    fig.savefig(FIG / "fig2_f1.png")


def fig3():
    """Test bits per step against N for RoboChunk, Fixed and the Flat reference (each seed + mean)."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for m in ["learned", "fixed"]:
        vals = np.array([[load_run(f"sim_{m}_N{N}_s{s}")[0]["test_bits"] for s in SEEDS] for N in NS])
        ax.plot(NS, vals.mean(1), color=COL[m], lw=2, marker="o", ms=5, label=NAME[m])
        for s in range(len(SEEDS)):
            ax.scatter(NS, vals[:, s], color=COL[m], s=8, alpha=0.5)
    flat = [load_run(f"sim_flat_s{s}")[0]["test_bits"] for s in SEEDS]
    ax.axhline(np.mean(flat), color=COL["flat"], lw=1.2, ls="--", label=NAME["flat"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("target compression N (steps per chunk)")
    ax.set_ylabel("test NLL (bits / step, lower is better)")
    ax.legend(fontsize=7)
    fig.savefig(FIG / "fig3_loss_vs_N.png")


def fig4(n_eps=4):
    """Real episodes: normalised gripper command with RoboChunk boundaries (seed 0)."""
    from robochunk.data_real import load_real
    names = list(FROZEN["real_datasets"])
    fig, axes = plt.subplots(n_eps, len(names), figsize=(7.5, 1.3 * n_eps), squeeze=False)
    for c, name in enumerate(names):
        nr = FROZEN["real_datasets"][name]
        test = load_real(name)["test"]
        b = load_run(f"real_{name}_learned_N{nr}_s0")[1]["bounds_list"]
        for r in range(min(n_eps, len(test))):
            ax = axes[r, c]
            ep = test[r]
            ax.plot(ep["gripper"], color=INK, lw=1)
            ax.vlines([x for x in b[r] if x != 0], 0, 1, color=COL["learned"], lw=0.8, alpha=0.8)
            for e in ep["events"]:
                ax.axvspan(e - 3, e + 3, color=COL["fixed"], alpha=0.25, lw=0)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.grid(False)
            if r == 0:
                ax.set_title(f"{name}: gripper command (black), learned boundaries (blue), events ±3 (orange)",
                             fontsize=6.5, loc="left")
        axes[-1, c].set_xlabel("step (30 Hz)")
    fig.savefig(FIG / "fig4_real_episodes.png")


def fig5(window=30):
    """Peri-event histogram: boundary density around gripper events, real vs within-episode shuffle."""
    from robochunk.data_real import load_real
    names = list(FROZEN["real_datasets"])
    fig, axes = plt.subplots(1, len(names), figsize=(7.5, 2.6), squeeze=False)
    rng = np.random.default_rng(0)
    for c, name in enumerate(names):
        nr = FROZEN["real_datasets"][name]
        test = load_real(name)["test"]
        lags = np.arange(-window, window + 1)
        obs, shuf = np.zeros(len(lags)), np.zeros(len(lags))
        n_ev = 0
        for s in SEEDS:
            b = load_run(f"real_{name}_learned_N{nr}_s{s}")[1]["bounds_list"]
            for ep, bb in zip(test, b):
                bb = np.array([x for x in bb if x != 0])
                T = len(ep["x"])
                for _ in range(100):
                    sb = rng.choice(np.arange(1, T), len(bb), replace=False)
                    for e in ep["events"]:
                        shuf += np.isin(e + lags, sb) / 100
                for e in ep["events"]:
                    obs += np.isin(e + lags, bb)
                    n_ev += 1
        ax = axes[0, c]
        ax.bar(lags, obs / n_ev, 0.9, color=COL["learned"], label="learned boundaries")
        ax.plot(lags, shuf / n_ev, color=INK, lw=1.2, label="within-episode shuffle")
        ax.axvspan(-3.5, 3.5, color=GRID, zorder=0)
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_xlabel("steps from gripper event")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("boundaries per event per lag")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(FIG / "fig5_peri_event.png")


def fig6():
    """Part B: novelty against policy gain; clean candidates as dots, corrupted as crosses."""
    res = json.load(open("runs/partb_results.json"))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for c, v in res["candidates"].items():
        corrupt = c.startswith("X")
        ax.errorbar(v["novelty"], v["gain"], yerr=[[v["gain"] - v["gain_ci"][0]], [v["gain_ci"][1] - v["gain"]]],
                    fmt="x" if corrupt else "o", color=COL["fixed"] if corrupt else COL["learned"], ms=6, lw=0.8,
                    mew=1.8)
        ax.annotate(c, (v["novelty"], v["gain"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("novelty (bits / step above held-out R)")
    ax.set_ylabel("success gain over R alone")
    ax.set_title(f"clean (dots): Spearman ρ = {res['h4']['rho']:.2f}; corrupted (crosses)", fontsize=8, loc="left")
    fig.savefig(FIG / "fig6_novelty_gain.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
    for f in todo:
        globals()[f]()
        print("wrote", f)

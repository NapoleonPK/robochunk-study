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
COL = dict(refractory="#6250d6", learned="#2a78d6", fixed="#eb6834", random="#1baf7a", entropy="#eda100", velocity="#e87ba4", flat="#52514e")
NAME = dict(refractory="Refractory router", learned="RoboChunk (learned)", fixed="Fixed", random="Random", entropy="Entropy (BLT-like)",
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
    fig, axes = plt.subplots(n_eps, 1, figsize=(7.5, 1.5 * n_eps), sharex=False, gridspec_kw=dict(hspace=0.75))
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
    fig, axes = plt.subplots(n_eps, len(names), figsize=(7.5, 1.5 * n_eps), squeeze=False,
                             gridspec_kw=dict(hspace=0.55, wspace=0.15))
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
                ax.set_title({"so100": "SO-100", "koch": "Koch"}.get(name, name), fontsize=8, loc="left")
        axes[-1, c].set_xlabel("step (30 Hz)")
    fig.suptitle("Test episodes: gripper command (black), learned boundaries (blue), gripper events ±3 steps (orange)",
                 fontsize=8, x=0.02, ha="left", y=0.99)
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
        ax.set_title({"so100": "SO-100", "koch": "Koch"}.get(name, name), fontsize=8, loc="left")
        ax.set_xlabel("steps from gripper event")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("boundaries per event per lag")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=7, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.08))
    fig.savefig(FIG / "fig5_peri_event.png")


def fig6():
    """Part B: novelty against policy gain. Left: clean candidates (the H4 test). Right: all 15 on a symmetric-log axis."""
    res = json.load(open("runs/partb_results.json"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.3))
    for ax, which in ((a1, "clean"), (a2, "all")):
        for c, v in res["candidates"].items():
            corrupt = c.startswith("X")
            if which == "clean" and corrupt:
                continue
            ax.errorbar(v["novelty"], v["gain"], yerr=[[v["gain"] - v["gain_ci"][0]], [v["gain_ci"][1] - v["gain"]]],
                        fmt="x" if corrupt else "o", color=COL["fixed"] if corrupt else COL["learned"], ms=6, lw=0.8,
                        mew=1.8, ecolor=GRID if not corrupt else None, zorder=3)
            off = {"C1": (-14, 2), "C2": (-14, -4), "C3": (5, -6), "C9": (-12, -10), "C8": (5, 4)}.get(c, (5, 3))
            ax.annotate(c, (v["novelty"], v["gain"]), xytext=off, textcoords="offset points", fontsize=7, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel("success gain over R alone")
    a1.set_xlabel("novelty (bits / step above held-out R)")
    a1.set_title(f"clean candidates C1–C11: Spearman ρ = {res['h4']['rho']:.2f}", fontsize=8, loc="left")
    a2.set_xscale("symlog", linthresh=1)
    a2.set_xlim(-1.2, 600)
    a2.set_xlabel("novelty (symmetric log scale)")
    a2.set_title("all 15: corrupted X1–X4 (crosses) look most novel", fontsize=8, loc="left")
    fig.savefig(FIG / "fig6_novelty_gain.png")


def fig7():
    """v2, fresh simulated test set: raw and chance-corrected (circular shift) boundary F1 at ±3 per method."""
    r = json.load(open("runs/results_v2.json"))["sim"]["methods"]
    order = ["refractory", "fixed", "velocity", "entropy", "random", "learned"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw=dict(wspace=0.3))
    x = np.arange(len(order))
    for ax, key, title in ((a1, "F1", "raw F1 (±3 steps)"), (a2, "cf_circ", "F1 − circular-shift chance (±3)")):
        vals = [r[m][key]["3"] for m in order]
        ax.bar(x, vals, color=[COL[m] for m in order], width=0.7, edgecolor="white", linewidth=2)
        top = max(vals)
        for xi, v in zip(x, vals):
            ax.text(xi, max(v, 0) + 0.02 * top, f"{v:+.3f}" if key == "cf_circ" else f"{v:.3f}", ha="center",
                    fontsize=7, color=INK)
        ax.set_ylim(min(0, min(vals)) * 1.3, top * 1.18)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x)
        ax.set_xticklabels(["Refractory router", "Fixed", "Velocity minima", "Entropy", "Random", "Learned (plain)"],
                           rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=8, loc="left")
    a1.set_ylabel("boundary F1, 300 fresh episodes")
    fig.suptitle("v2 confirmatory test: H1-v2 (raw, vs Fixed) failed; H1b-v2 (chance-corrected) passed", fontsize=9,
                 x=0.02, ha="left", y=1.04)
    fig.savefig(FIG / "fig7_v2_f1.png")


def fig8():
    """Post hoc: policy gain with the original fixed-update BC budget vs passes matched to R."""
    orig = json.load(open("runs/results_partB.json"))["candidates"]
    mat = json.load(open("runs/results_partB_matched.json"))
    cs = ["C1", "C2", "C3", "C6"]
    fig, ax = plt.subplots(figsize=(5, 3.0))
    for k, (src, lab, col, dx) in enumerate(((orig, "fixed 20k updates (as run)", COL["fixed"], -0.12),
                                             (mat, "passes matched to R (post hoc)", COL["learned"], 0.12))):
        for i, c in enumerate(cs):
            g = src[c]["gain"]
            lo, hi = src[c]["gain_ci"] if "gain_ci" in src[c] else src[c]["ci"]
            ax.errorbar(i + dx, g, yerr=[[g - lo], [hi - g]], fmt="o", color=col, ms=6, lw=1,
                        label=lab if i == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels(["C1 duplicates", "C2 near-dup.", "C3 same region", "C6 whole table"], fontsize=7)
    ax.set_ylabel("success gain over R")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Negative in-distribution gains were a training-budget artefact", fontsize=8, loc="left")
    fig.savefig(FIG / "fig8_bc_budget.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8"]
    for f in todo:
        globals()[f]()
        print("wrote", f)

"""Stage the Hugging Face upload in hf_stage/ (git-ignored): one model repo (checkpoints of every simulated-data and
Part B run) and one dataset repo (the simulated data generated here, Parquet copies of the test and fresh splits, and the
result files). Real-robot checkpoints and the LeRobot datasets are not included. Card sections (results, hashes,
timeline) come from scripts/make_cards.py; every staged text file is scanned with scripts/check_sensitive.py rules.
usage: PYTHONPATH=. python scripts/stage_hf.py"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from check_sensitive import scan_text  # noqa: E402
from export_parquet import ACT_NAMES, GOAL_NAMES, OBS_NAMES  # noqa: E402
from make_cards import (DATA, FIG1, GH, MODEL, NOT_AFFILIATED, fig_url, prereg_hashes, results_table,  # noqa: E402
                        sim_counts, timeline)

ST = Path("hf_stage")
RUN_PREFIX = ("sim_", "exp_", "abl_", "v2_sim_", "partb_", "proxy_sim")
KEEP = ("best.pt", "config.yaml", "norm.json", "metrics.json", "history.json", "git.txt")
CREDIT = ("The model follows **H-Net** (Sukjun Hwang, Brandon Wang, Albert Gu, *Dynamic Chunking for End-to-End "
          "Hierarchical Sequence Modeling*, arXiv:2507.07955) and Cartesia's posts on dynamic chunking. Simulation: "
          "Gymnasium-Robotics and MuJoCo. Author: Téo Tran Van Lieu. Code written with Claude (Anthropic) as a coding agent.")
SNIPPET = Path("scripts/hf_snippet.py").read_text().split('"""', 2)[2].strip()  # drop the docstring

PHASES_MD = """| id | phase | what the scripted expert does |
|---|---|---|
| 0 | approach | move the open gripper to 10 cm above the object |
| 1 | descend | lower the open gripper to the object (on planned retries, offset 4.5–6 cm along x so the grasp misses) |
| 2 | close | close the fingers in place |
| 3 | lift | raise the gripper 8 cm with the fingers closed |
| 4 | transport | carry the object to 1 cm above the goal |
| 5 | open | open the fingers in place |
| 6 | retreat | raise the open gripper 10 cm |

The phase label at a step is the most recently started phase. Each phase follows a minimum-jerk profile with a random speed
factor (0.5–1.5), one pause of 0–5 steps, and may start 0–3 steps before the previous one ends. If the object has not
risen with the gripper when transport is due to start, the expert opens and repeats approach → retreat (a *retry*;
10% of episodes plan a missed first grasp)."""

FIELDS_MD = "\n".join([
    "| field | meaning |", "|---|---|",
    "| `episode_index` | episode number within the split (0-based) |",
    "| `episode_seed` | seed that regenerates the episode exactly |",
    "| `step` | time step within the episode (25 Hz, at most 150 steps) |",
    "| `phase`, `phase_name` | ground-truth motion phase (table below) |",
    "| `phase_start` | true if a new phase starts at this step: the boundaries scored by H1 (step 0 excluded) |",
    "| `contact` | a finger touches the object |",
    "| `success` | object within 5 cm of the goal at this step |",
    "| `episode_retried`, `episode_truncated` | the episode contains a retry / hit the 150-step limit |",
    f"| `{OBS_NAMES[0]}` … `{OBS_NAMES[-1]}` | the 25 FetchPickAndPlace observation channels: gripper position, object "
    "position, object relative to gripper, finger positions, object rotation, object linear/angular velocity, gripper "
    "velocity, finger velocities (simulator units) |",
    f"| `{ACT_NAMES[0]}` … `{ACT_NAMES[-1]}` | the expert's action: end-effector displacement command (x, y, z) and "
    "gripper command (positive opens, negative closes), with Gaussian noise std 0.05 |",
    f"| `{GOAL_NAMES[0]}` … `{GOAL_NAMES[-1]}` | goal position for the object (constant within an episode) |"])


def results(repo, typ):
    return "\n\n".join([
        "## Results", results_table(),
        f"![Episode boundaries]({fig_url(repo, typ)})",
        "*True motion phases (bands) and the boundaries each rule places on six simulated test episodes, N = 10.*",
        "Full report with all numbers, caveats and deviations: `REPORT.md` in the GitHub repository."])


def evidence():
    tl = timeline().replace("Source details: `docs/timeline_evidence.json`.",
                            f"Source details: `docs/timeline_evidence.json` in {GH}.")
    return "\n\n".join(["## Pre-registration hashes", prereg_hashes(), "## Timeline evidence", tl, "## Credits", CREDIT])


def model_card():
    return f"""---
license: mit
tags: [robotics, h-net, dynamic-chunking, behaviour-cloning, pre-registered]
library_name: pytorch
---
# RoboChunk checkpoints

{NOT_AFFILIATED}

Checkpoints from **RoboChunk**, a pre-registered study of H-Net-style dynamic chunking on robot trajectories
(FetchPickAndPlace-v4, scripted expert with labelled motion phases). Code, pre-registrations, decision log and the full
report: **{GH}**. Simulated data and result files: https://huggingface.co/datasets/{DATA}

{results(MODEL, "model")}

## Load a checkpoint and plot its boundaries

```python
# Needs the code: git clone {GH} && cd robochunk-study && pip install -r requirements.txt
{SNIPPET}
```

## Contents
`runs/<run_id>/` for every simulated-data and Part B run:
- `best.pt` — PyTorch state dict at the best validation epoch; `config.yaml` — run settings; `norm.json` — input
  normalisation; `metrics.json` — test metrics and stop reason; `history.json` — training curve; `git.txt` — code commit
  at training time (an old hash: see *History rewrite* below).
- Run ids: `sim_<method>_N<N>_s<seed>` (main study), `exp_refractory_N10_s*` (refractory router, exploratory in v1,
  tested in v2), `v2_sim_refractory_N10_s3–5` (v2 secondary seeds), `abl_*` (ablations), `partb_*` (models trained on
  the Part B reference set for novelty scoring), `proxy_sim` (entropy-baseline proxy). Real-robot checkpoints are not
  published.

{evidence()}
"""


def data_card():
    counts, pb = sim_counts()
    return f"""---
license: cc-by-4.0
tags: [robotics, simulation, pick-and-place, motion-segmentation, mujoco]
pretty_name: RoboChunk simulated pick-and-place trajectories
size_categories: [10K<n<100K]
configs:
- config_name: default
  data_files:
  - split: test
    path: sim/parquet/test.parquet
  - split: fresh
    path: sim/parquet/fresh.parquet
---
# RoboChunk simulated data and results

{NOT_AFFILIATED}

Simulated pick-and-place trajectories with ground-truth motion-phase labels, generated for **RoboChunk** ({GH}), plus the
study's result files. All data here was generated by the study's own scripted expert; no third-party data is included.
Checkpoints: https://huggingface.co/{MODEL}

{results(DATA, "dataset")}

## Splits
FetchPickAndPlace-v4 (gymnasium-robotics 1.4.2, MuJoCo 3.3.7), 25 Hz. The viewer shows the `test` split (H1–H2) and the
`fresh` split (generated after the v2 pre-registration, used by H1-v2/H1b-v2) as Parquet, one row per step. All splits are
also in `sim/*.npz` (read with `robochunk.data_sim.load`; regenerate with `python scripts/make_sim_data.py`).

{counts}

Part B (`partb/*.npz`): reference set `R`, `R_heldout`, `R_val`, clean candidates `C1–C11`, corrupted candidates `X1–X4`
(what each one is: `DECISIONS.md` in the GitHub repository).

{pb}

`results/`: every result file the report is generated from, and the per-episode policy successes behind H4
(`partb_policy*/`).

## Fields (Parquet)

{FIELDS_MD}

## Phases

{PHASES_MD}

```python
from datasets import load_dataset
ds = load_dataset("{DATA}", split="test")
ep = ds.filter(lambda r: r["episode_index"] == 0)
print(ep["phase_name"][:20])
```

{evidence()}
"""


def main():
    if ST.exists():
        shutil.rmtree(ST)
    m, d = ST / "model", ST / "dataset"
    for run in sorted(Path("runs").iterdir()):
        if run.is_dir() and run.name.startswith(RUN_PREFIX) and (run / "best.pt").exists():
            for f in KEEP:
                if (run / f).exists():
                    (m / "runs" / run.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(run / f, m / "runs" / run.name / f)
    for sub in ("sim", "partb"):
        shutil.copytree(Path("data") / sub, d / sub)
    subprocess.run([sys.executable, "scripts/export_parquet.py", str(d / "sim" / "parquet"), "test", "fresh"], check=True)
    (d / "results").mkdir(parents=True)
    for f in list(Path("runs").glob("results_*.json")) + list(Path("runs").glob("partb_*.json")) + \
            list(Path("runs").glob("summary*.csv")):
        shutil.copy2(f, d / "results" / f.name)
    for sub in ("partb_policy", "partb_policy_matched"):  # per-episode 0/1 policy successes behind H4
        shutil.copytree(Path("runs") / sub, d / "results" / sub)
    for repo in (m, d):
        (repo / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIG1, repo / FIG1)
    (m / "README.md").write_text(model_card())
    (d / "README.md").write_text(data_card())
    bad = []
    for f in ST.rglob("*"):
        if f.is_file() and f.suffix in (".md", ".yaml", ".json", ".txt", ".csv"):
            txt = f.read_text(errors="replace")
            bad += scan_text(str(f), txt)
            if "/Users/" in txt:
                bad.append(f"{f}: local path")
    assert not bad, bad
    size = sum(f.stat().st_size for f in ST.rglob("*") if f.is_file()) / 1e9
    print(f"staged {len(list((m / 'runs').iterdir()))} runs + dataset; {size:.2f} GB; sensitive scan ok")


if __name__ == "__main__":
    main()

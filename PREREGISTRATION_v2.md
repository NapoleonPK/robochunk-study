# RoboChunk v2 — confirmatory test of the refractory router (pre-registration)

**Status: DRAFT rev. 2, 2026-10-30. Awaiting Téo's approval. Nothing below has been generated, trained or evaluated.**
On approval this file, `runs/v2/frozen_artifacts.sha256`, the v2 configs and `scripts/analyze_v2.py` are committed and
tagged `prereg-v2` *before* the fresh episodes are generated, the new dataset is downloaded or any v2 run starts.
Author: Téo Tran Van Lieu. Builds on `PREREGISTRATION.md` (v1).

## 1. Background, provenance and what was seen

**v1 timeline (local time, UTC+1).** v1 pre-registration `1f1faa0` committed 29 Oct 14:08:32. A first main queue started
14:08:43 and was stopped about 15 minutes later; only the three entropy proxies had finished and nothing from it was used
(archived in `runs/_v1_before_amendment/`). **Amendment 1** was committed as `fa40b3b` (19:28:48) and `f61d0f3` (19:28:57),
5 h 20 min after that aborted start and **5 seconds before the first run of the v1 main queue that produced the results**
(19:29:02). Each v1 main run records the commit current when it started (`f61d0f3`, `73e3735`, `9768bb7`, `dad5768` or `6b22c38`); between `f61d0f3` and all of these, the only change under `robochunk/` is the new analysis-only file `exploratory.py`, which training does not import.
The first v1 main run finished at 19:38:56 (an entropy proxy); the first hypothesis-relevant run at 20:00:18.

**The chance-floor issue was raised before any v1 result.** Téo's request for chance-corrected F1 arrived on 29 Oct at
19:32:39, 3.6 minutes after the v1 main queue started and before any v1 main run had finished. It was added to v1 as an
exploratory analysis (`73e3735`, 19:35:31), H1–H4 unchanged.

**What was seen before this file was written.** All v1 verdicts (H1 fail, H2 pass, H3 significant but below threshold),
and the v1 exploratory analyses in `docs/EXPLORATORY.md`, including the chance-corrected F1 of every method on the v1 test
split. On that split the refractory router had the highest chance-corrected F1 at ±1 and ±3. **H1b-v2 below was written
after seeing those numbers**; its test data are new for that reason.

**The refractory rule, precisely** (`robochunk/model.py`, `RoboChunk._suppress`; unchanged since `1f1faa0`). The router
computes p(t) as in H-Net and proposes b(t) = 1[p(t) ≥ 0.5] (with b(0) = 1 forced). Scanning t = 1, 2, …, a proposed
boundary is **kept only if t − (last kept boundary) > 3**; step 0 counts as kept, so steps 1–3 can never be boundaries.
The scan is causal (it uses only kept boundaries before t). It is applied **identically in training, validation and test**:
the same forward pass is used in all three. Suppressed steps are treated as non-boundaries everywhere downstream:
downsampling keeps only kept boundaries; the EMA dechunk uses P at kept boundaries only; the straight-through factor uses
c = p at kept boundaries and c = 1 − p elsewhere (including suppressed steps). **The ratio loss uses kept boundaries for
F** (the boundary fraction) **and the unmodified p at every valid step for G** (mean p), so suppressed proposals still
count in G. The window (3 steps) was set once in v1 with no sweep and no pilot run (`docs/EXPLORATORY.md` §3).

## 2. What is tested and what is fixed

- Model, inputs, training recipe, lr 3e-4, router (LayerNorm, α = 0.3, identity init), N, budgets, metrics, matching
  rule, bootstrap and permutation procedures: **identical to v1 after Amendment 1**. r = 3 only.
- H1-v2 and H3-v2 use the **v1 thresholds unchanged**. H1b-v2 and H3b-v2 are new tests with thresholds stated below.

## 3. Data

**Fresh simulated test set (never generated before the tag).** 300 episodes, seeds 700000–700299 (no v1 split uses this
range: v1 uses 0–2,399, 100,000–100,299, 200,000–200,299, 300,000–429,999, 500,000–510,199, 900,000–900,199), same
generator and settings as v1, whole table, default mass and friction. Generated once after the tag by
`scripts/make_sim_data.py` (new split `fresh`); its SHA-256 is recorded in `runs/v2/fresh_sha256.txt` before any model
sees it. Ground-truth boundaries = phase-label changes.

**v1 real data (not fresh).** `lerobot/svla_so100_pickplace` @ `728583b5…` (5 test episodes) and
`lerobot/koch_pick_place_1_lego` @ `afc0b3e5…` (10 test episodes), same 80/10/10 split (seed 0) as v1. These test
episodes were used for the v1 H3 verdicts of other methods, and their burst analysis is in `docs/EXPLORATORY.md`. The
refractory router has never been trained or evaluated on real data.

**New real dataset (never used): `lerobot/svla_so101_pickplace` @ `f641879e22172be7e8161d5e6c1503c2d2feb657`.** Found by
searching the Hugging Face `lerobot` datasets for pick-and-place. It meets the v1 criteria: real single SO-101 arm (driver
`so100_follower`), pick-and-place ("pink lego brick into the transparent box"), LeRobot v3.0, 50 episodes (≥ 50), 30 fps,
6 joint states + 6 joint actions with the gripper at action index 5 (`gripper.pos`), one 370 KB parquet file. Like the
v1 SO-100 dataset, its card does not state the control mode; LeRobot records this robot type with a leader arm. Only its
metadata (`meta/info.json`, `meta/tasks.parquet`, README, file list) has been read, to check eligibility; no episode data
has been downloaded. After the tag: `data/` and `meta/` only, same loader, same 80/10/10 split (seed 0, so 5 test
episodes), same gripper normalisation and hysteresis. Another eligible unused dataset exists
(`lerobot/koch_pick_place_1_lego_raph`, 100 episodes) but repeats the v1 Koch robot and task, so SO-101 was chosen.

## 4. Models and frozen artifacts

**Simulation, primary (no new training).** The three refractory models trained in v1, `exp_refractory_N10_s{0,1,2}`
(trained 30 Oct 04:06–04:52 on the v1 training split, best-validation checkpoints), evaluated unchanged on the fresh
episodes. Baselines at N = 10 with their v1 settings: Fixed (every 10 steps); Random (Bernoulli, rate 1/N counting step 0,
per-episode seed = crc32("fresh:index") + run seed, seeds 0–2); Velocity minima with the v1 threshold 0.0019115220056846738;
Entropy with the v1 proxy `proxy_sim` and the v1 threshold −7.098278999328613 (both fitted on the v1 validation split).
The plain learned router `sim_learned_N10_s{0,1,2}` is evaluated as a reference.
SHA-256 of every file the v2 simulation test reads (checkpoints, norms, configs, proxy, v1 data) is in
`runs/v2/frozen_artifacts.sha256`, committed with the tag; `scripts/analyze_v2.py` checks them before running. The three
refractory checkpoints:

| file | SHA-256 |
|---|---|
| `runs/exp_refractory_N10_s0/best.pt` | `db37510cf428b54f1571c3d3caa2960decc8b216dafd7e6ad4d01c54a7e37ba5` |
| `runs/exp_refractory_N10_s1/best.pt` | `36cd62372104089944025e22b6e68751bd76621c654ec6e46357978e8a3d1c3f` |
| `runs/exp_refractory_N10_s2/best.pt` | `120b11df9cb6fa953077931d1cda0876c11366b5d5be88c51b74ff2b384a26a8` |

**Simulation, secondary (new training).** Refractory runs with seeds 3, 4, 5, v1 recipe and sim budget (600 epochs,
patience 30, 50 min), evaluated on the fresh episodes. Not part of any pass/fail.

**Real data (new training).** Refractory router (r = 3 steps = 0.10 s at 30 fps), N_real = 12, seeds 0–2, on SO-100,
Koch and SO-101, v1 real-data recipe and budget (6,000 epochs, patience 400, 45 min). 9 runs. In v1 the Koch learned runs
stopped on the 45-min cap; the cap is kept, and stop reasons are reported. **Secondary on SO-101:** an entropy proxy
(v1 proxy recipe) and the entropy rule's boundaries on the SO-101 test split (threshold fitted on its validation split,
as in v1), a fresh check of the v1 entropy gripper-event finding.

## 5. Hypotheses and pass criteria

Chance F1 used by H1b-v2, per method, run and fresh episode: **shuffle chance** = mean F1 (±3) over 1,000 within-episode
shuffles of that run's boundaries in that episode (same count, uniform without replacement over steps 1…T−1; RNG seed 0);
for **Fixed**, **grid chance** = the exact mean F1 (±3) over the 10 grid offsets o = 0…9 of the grid {o, o+10, …} (step 0
excluded). Chance-corrected F1 = F1 − chance, per episode, averaged over the method's seeds.

| ID | Test | Pass criterion |
|---|---|---|
| H1-v2 | Fresh sim test set (300 episodes), N = 10. Boundary F1 at ±3, one-to-one nearest-first matching, step-0 boundary ignored, per episode then averaged; per-episode F1 averaged over each method's 3 seeds. | **Unchanged from v1.** For **every** baseline (Fixed, Random, Entropy, Velocity minima): mean over episodes of [refractory F1 − baseline F1] ≥ 0.10, **and** the 95% paired percentile bootstrap CI (10,000 resamples of episodes, seed 0) excludes 0. |
| H1b-v2 (co-primary) | Same episodes and runs. Chance-corrected F1 at ±3 as defined above (shuffle chance for every method except Fixed, grid chance for Fixed). | For **every** baseline: mean over episodes of [refractory chance-corrected F1 − baseline chance-corrected F1] > 0, **and** the 95% paired percentile bootstrap CI (10,000 resamples of episodes, seed 0) excludes 0. |
| H3-v2 | Real test splits of SO-100 and Koch, N_real = 12, gripper events as in v1 (hysteresis 0.65 / 0.35 on the gripper command normalised by its 2nd/98th percentiles). | **Unchanged from v1.** Share of refractory boundaries (step 0 excluded) within ±3 steps of a gripper event, pooling the 3 seeds (each seed × episode one unit), ≥ 2× the mean share under 1,000 within-unit shuffles, **and** one-sided permutation p < 0.01. Per dataset; *supported* if both pass, *mixed* if one, *not supported* if none. |
| H3b-v2 | Fresh: SO-101 test split, same N_real, events, pooling and shuffle test. | Same rule as H3: ratio ≥ 2× **and** p < 0.01 → pass; otherwise fail. Reported separately from H3-v2. |

H1-v2 and H1b-v2 are co-primary and reported side by side. Neither verdict is replaced or reinterpreted by the other.

## 6. What v1 predicts (stated in advance)

- **H1-v2:** on the v1 test split the refractory router's F1 at ±3 exceeded Random by about 0.21, Entropy by 0.17 and
  Velocity by 0.12, but **Fixed by only about 0.036**. H1-v2 is expected to **fail** on the Fixed comparison. The v1
  margin is kept anyway.
- **H1b-v2:** on the v1 test split (v1 exploratory numbers, grid chance by 1,000 random offsets there), chance-corrected F1
  at ±3 was about +0.176 for the refractory router vs +0.019 for Fixed (grid chance), +0.083 Velocity, +0.032 Entropy,
  −0.003 Random. H1b-v2 is expected to **pass** if the effect is real; the closest comparison is Velocity.
- **H3-v2 / H3b-v2:** no prediction (the refractory router was never run on real data). The plain router reached 1.89×
  (SO-100) and 1.77× (Koch) in v1.

## 7. Secondary measures (reported, no pass/fail)

Fresh sim set: F1 at ±1 and ±5; chance-corrected F1 at ±1; burst-merged F1; boundary density by phase; realised rate;
next-step bits per step of the refractory models vs Fixed at N = 10 (with the v1 compression rule); the secondary seeds
3–5. Real data: event recall (share of gripper events with a boundary within ±3), burst locations, realised rate and stop
reason per run; SO-101 entropy rule with the same H3 statistic.

## 8. Procedure and order

1. Commit this file, `runs/v2/frozen_artifacts.sha256`, `scripts/make_sim_data.py` (with the `fresh` split), the SO-101
   loader entry, the v2 configs and `scripts/analyze_v2.py`; tag `prereg-v2`.
2. Generate the fresh episodes and record their SHA-256; download SO-101 (`data/`, `meta/`) at the pinned revision.
3. Train the 9 real-data runs, the SO-101 proxy and the 3 secondary sim runs (at most 2 jobs at a time).
4. Run `scripts/analyze_v2.py` once (it first verifies the frozen hashes); it writes `runs/results_v2.json`. Every
   reported v2 number comes from that file.
5. Any deviation is logged with its reason in a dated amendment at the end of this file before results are reported.

Estimated compute: about 5–6 hours (9 real runs at up to 45 min, 3 sim runs at up to 50 min, one proxy, evaluation).

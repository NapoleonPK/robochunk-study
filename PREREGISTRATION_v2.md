# RoboChunk v2 — confirmatory test of the refractory router (pre-registration)

**Status: DRAFT, 2026-10-30. Sent to Téo for approval. Nothing below has been generated, trained or evaluated.**
On approval this file is committed and tagged `prereg-v2` *before* the fresh test episodes are generated and before any
v2 run starts. Author: Téo Tran Van Lieu. Builds on `PREREGISTRATION.md` (v1, commit `1f1faa0`, Amendment 1 `f61d0f3`).

## 1. Why a v2

In v1 the learned H-Net router failed H1 (boundary F1 0.358 vs Fixed 0.594) and H3 was significant but below the 2×
threshold on both real datasets. Its boundaries come in bursts (1-step gaps between consecutive boundaries are the norm).
An **exploratory** variant listed in v1 §7 — the *refractory router*, identical except that after a kept boundary at t,
boundaries at t+1…t+3 are dropped — had the best chance-corrected F1 of every method on the v1 test split
(see `docs/EXPLORATORY.md`). That result is post hoc in two ways: it is one exploratory variant among several, and it was
seen on the same test episodes used for all v1 verdicts. v2 asks whether it holds on data nobody has looked at.

## 2. What is tested and what is not

- The **refractory window r = 3 is fixed** (it was set once in v1, with no sweep; see `docs/EXPLORATORY.md` §3). No other r is tested.
- Model, inputs, training recipe, learning rate (3e-4), router (LayerNorm, α = 0.3, identity init), N, budgets, metrics,
  matching rule, bootstrap and permutation procedures: **identical to v1 after Amendment 1**.
- **Thresholds identical to v1** (margin 0.10 for H1, 2× and p < 0.01 for H3). They are not adjusted to the v1 exploratory
  numbers (see §6 for what v1 predicts).

## 3. Data

**Fresh simulated test set (never generated before this commit).** 300 episodes, seeds 700000–700299 (a range used by no
v1 split), same generator and code as v1 (`robochunk/data_sim.py` at the commit tagged `prereg-v2`), whole table,
default mass and friction, same scripted expert settings. Generated once, after the tag, by `scripts/make_sim_data.py`
with a new split `fresh`. Ground-truth boundaries = phase-label changes, as in v1.

**Real data.** The same two datasets, revisions and 80/10/10 split (seed 0) as v1: `lerobot/svla_so100_pickplace`
(5 test episodes) and `lerobot/koch_pick_place_1_lego` (10 test episodes). **Disclosure:** these test episodes are not
fresh. They were used for the v1 H3 verdicts of the plain learned router, Fixed and Entropy, and their burst analysis is
in `docs/EXPLORATORY.md`. The refractory router has never been trained or evaluated on real data. No unused episodes
exist in these datasets.

## 4. Models

**Simulation (no new training for the primary test).** The three refractory models already trained in v1
(`exp_refractory_N10_s{0,1,2}`, trained on the v1 training split, best-validation checkpoints, unchanged) are evaluated on
the fresh episodes. Baselines on the fresh episodes, all at N = 10 and with their v1 settings: Fixed (every 10 steps),
Random (Bernoulli, rate 1/N counting step 0, per-episode seed = crc32("fresh:index") + run seed, seeds 0–2), Velocity
minima and Entropy with the **v1 thresholds** (fitted on the v1 validation split; the Entropy rule uses the v1 proxy
`proxy_sim`). The plain learned router (`sim_learned_N10_s{0,1,2}`) is evaluated as a reference.

**Simulation, secondary (new training).** Three new refractory runs with seeds 3, 4, 5, same recipe and v1 sim budget
(600 epochs, patience 30, 50 min), evaluated on the fresh episodes. They test whether the effect belongs to the training
procedure rather than to the three v1 models. Not part of any pass/fail.

**Real data (new training).** Refractory router (r = 3 steps = 0.10 s at 30 fps), N_real = 12, seeds 0–2, on each
dataset, with the v1 real-data recipe and budget (6,000 epochs, patience 400, 45 min). 6 runs.

## 5. Hypotheses and pass criteria

| ID | Test | Pass criterion (same as v1) |
|---|---|---|
| H1-v2 | Fresh simulated test set (300 episodes), N = 10. Boundary F1 at ±3 steps, one-to-one nearest-first matching, forced step-0 boundary ignored, per episode then averaged; each method's per-episode F1 averaged over its 3 seeds (Fixed, Velocity and Entropy are seed-independent). | For **every** baseline (Fixed, Random, Entropy, Velocity minima): mean over fresh episodes of [refractory F1 − baseline F1] ≥ 0.10, **and** the 95% paired percentile bootstrap CI (10,000 resamples of episodes) of that difference excludes 0. |
| H3-v2 | Real test split of each dataset, N_real = 12, gripper events as in v1 (hysteresis 0.65 / 0.35 on the gripper command normalised by its 2nd/98th percentiles). | Share of refractory-router boundaries (step 0 excluded) within ±3 steps of a gripper event, pooling the 3 seeds (each seed × episode one unit), is ≥ 2× the mean share under 1,000 within-unit shuffles, **and** one-sided permutation p < 0.01. Per dataset; *supported* if both pass, *mixed* if one, *not supported* if none. |

## 6. What v1 predicts (stated before the test, so the outcome cannot be reframed afterwards)

On the v1 test split the refractory router's F1 at ±3 exceeded Random by about 0.21, Entropy by 0.17 and Velocity by 0.12,
but **Fixed by only about 0.036, below the 0.10 margin**. Unless the fresh episodes behave very differently, **H1-v2 is
expected to fail on the Fixed comparison** while passing the other three. We keep the v1 margin anyway: lowering it after
seeing v1 numbers is exactly what this confirmatory test is meant to avoid. A failure on Fixed alone will be reported as
"H1-v2 fails", with the per-baseline results shown. For H3-v2 v1 gives no prediction (the refractory router was not run on
real data); the plain router reached 1.89× (SO-100) and 1.77× (Koch).

## 7. Secondary measures (reported, no pass/fail)

On the fresh simulated set: F1 at ±1 and ±5; chance-corrected F1 (within-episode shuffle, 1,000 shuffles; grid chance for
Fixed); burst-merged F1; boundary density by phase; realised boundary rate; next-step bits per step of the refractory
models vs Fixed at N = 10 (with the v1 compression rule: realised compression ≥ 0.8 N); the secondary seeds 3–5. On real
data: event recall (share of gripper events with a boundary within ±3), burst locations, realised rate per seed.

## 8. Procedure and order

1. Commit this file, `scripts/make_sim_data.py` (with the `fresh` split), the v2 configs and `scripts/analyze_v2.py`; tag `prereg-v2`.
2. Generate the fresh episodes; record their SHA-256 in `runs/v2/fresh_sha256.txt`.
3. Train the 6 real-data runs and the 3 secondary sim runs (at most 2 jobs at a time).
4. Run `scripts/analyze_v2.py` once; it writes `runs/results_v2.json`. Every reported number comes from that file.
5. Any deviation from this plan is logged with its reason in a dated amendment at the end of this file, before results
   are reported.

Estimated compute: about 3–4 hours (6 real runs at up to 45 min, 3 sim runs at up to 50 min, evaluation in minutes).

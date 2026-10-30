# Decisions

Choices the brief left open, one line each, with the reason. Dated entries; newest last.

## Setup (2026-10-29)
- Python 3.12.15 via `uv`, venv in `.venv/`; versions pinned in `requirements.txt` (`uv pip freeze`).
- `mujoco==3.3.7`, not the latest 3.17.0: with 3.17 `gymnasium-robotics==1.4.2` Fetch envs fail an assertion in `set_joint_qpos` at construction.
- Environment `FetchPickAndPlace-v4` (newest Fetch pick-and-place registered by gymnasium-robotics 1.4.2; v1 is the old mujoco-py version).
- Device: CPU, 6 torch threads per job, 2 jobs at a time. Pilot timing, batch 64 pilot episodes, one training step: learned 89–98 ms CPU vs 116 ms MPS; fixed 111 vs 118 ms; flat 287–346 vs 207–231 ms. Two concurrent CPU jobs run at 135 ms/step each (1.4x throughput). One device for every run avoids mixing MPS/CPU numerics across methods.
- Git author is "Téo Tran Van Lieu" with the GitHub noreply address, so his email never enters the public history.
- Run outputs stay in `runs/` (git-ignored); data in `data/` (regenerable, git-ignored); downloads in `downloads/` (git-ignored).

## Simulated data
- Goals always on the table (`target_in_the_air = False`), so "open gripper" places the object instead of dropping it in the air.
- Main Part A data uses the whole table (Fetch's default ±15 cm object range); Part B's core region is the central ±7.5 cm square.
- Split seeds: train 0–2399, val 100000–100299, test 200000–200299, pilot 900000–900199. Part B seeds start at 300000 (see `robochunk/partb.py`).
- Base phase durations at speed 1 (steps): approach 13, descend 8, close 5, lift 7, transport 15, open 4, retreat 7; per-phase duration = base / U(0.5, 1.5), raised if needed so min-jerk peak speed stays under 80% of Fetch's 5 cm/step limit.
- "Speed factor 0.5–1.5x" is implemented as speed (duration divided by it), not as a duration multiplier.
- Overlap: the next phase's min-jerk profile starts 0–3 steps before the current one ends and the two displacements superpose, so velocity is smooth and non-zero across boundaries; the label switches when the next phase starts.
- Pause: 0–5 steps at a uniform random point inside every phase (the profile is frozen during the pause).
- Action noise: Gaussian, std 0.05 in Fetch action units (2.5 mm of commanded motion) on all 4 action dims.
- Failed grasp: in 10% of episodes the first descend is offset 4.5–6 cm along x, so the fingers miss; after the lift the expert checks the object height and, if the grasp failed (planned or not), runs open -> approach -> descend -> close -> lift again (one retry at most). Phase labels repeat.
- Episodes end when the retreat finishes, or at 150 steps (truncated). Episode lengths vary; no padding is stored.
- Mass x3 and friction x0.1–0.3 do not change the scripted expert's success (48/48 each in a pilot check): the Fetch gripper holds a 6 kg cube. They do lower the BC policy's success (low friction 40% vs 50%).
- Model input channels (robot-only, 11): gripper position (3), gripper velocity (3, Fetch reports it as displacement per step), finger width (sum of the two finger joints, 1), action (4). Object-state ablation appends object position (3).

## Model
- Matches H-Net §2 as read on 2026-10-29: α = 0.03 (fixed in all H-Net experiments), a norm at the end of each encoder / decoder / main network, the residual projection initialised near zero (weights x 1e-3, bias 0).
- W_q and W_k initialised to the identity (suggested in the brief; with it, p starts near 0, and the ratio loss pushes it up).
- LayerNorm (not RMSNorm; PyTorch 2.15 has both, LayerNorm is the common default for small models).
- Main network: learned chunk-index position embedding (a causal Transformer needs some position signal; chunk index rather than time keeps it identical across boundary rules).
- Log-variance clamped to [-10, 5].
- Non-learned rules feed b(t) and p(t) = b(t); the EMA is then the identity and is skipped (mathematically identical, faster).
- Bits per step = Gaussian NLL in nats summed over channels, divided by ln 2, in normalised (z-scored) units. It is a differential (continuous) quantity, so it can be negative; only differences between models on the same data are meaningful.
- FLOPs per step = `torch.utils.flop_counter` total on a 64-episode test batch plus an analytic 6·H·(D_in+H) per GRU layer per padded step (the counter does not see `aten::gru`), divided by valid steps.

## Baselines
- Velocity minima are causal: a local minimum at t-1 is only known at t, so the boundary is placed at t. Threshold on the minimum's speed, fitted on the training split to give rate 1/N (counting the forced step-0 boundary).
- Entropy rule: boundary at t where the proxy's NLL of x(t) given x(<t) is above a threshold fitted on the training split for rate 1/N. Proxy = 2-layer GRU, width 64, same training recipe, best validation checkpoint.
- Random rule: Bernoulli(1/N) with a per-episode seed = crc32("split:index") + run seed, frozen.
- Boundary F1: precision is 0 for an episode with no predicted boundary besides step 0.

## Real data
- Datasets (both LeRobot v3.0, 30 fps, teleoperated, single arm, 6 joints incl. gripper, parquet < 1.2 MB):
  `lerobot/svla_so100_pickplace` @ 728583b5eaf9e739a7f119e2def466fa1d552402 (50 episodes, "Pick up the cube and place it in the box"),
  `lerobot/koch_pick_place_1_lego` @ afc0b3e54679cb8a94283fbab5a4ccf0d405f10b (102 episodes, "Pick the lego blocks and place them in the box").
  Alternatives rejected: `svla_so101_pickplace` (fine, but one SO arm is enough), `koch_pick_place_5_lego*` (5 picks per episode, 2+ GB videos irrelevant but 760-step episodes).
- Channels: observation.state (6) + action (6) = 12. Gripper = action index 5, normalised per dataset to [0, 1] with its 2nd/98th percentiles; events = hysteresis crossings at 0.65 (up) / 0.35 (down).
- Episode split 80/10/10 with `numpy.random.default_rng(0).permutation`.

## Part B
- R = 500 core episodes; held-out R = 100 core episodes (novelty reference); R_val = 50 core episodes (early stopping for the R-trained sequence models).
- C2 near-duplicates: same seeds as R[0:200] (same object, goal, durations, pauses) with a new action-noise RNG.
- C8 "low friction": sliding friction x0.2. C7/C9 "mass x3".
- X1/X3/X4 corrupt the same 200 fresh core episodes; X2 re-runs those seeds with executed action noise std 0.5 (degraded demonstrations).
- X3 drops 20% of frames per episode uniformly at random (obs, action and labels together).
- Broad test = 200 episodes, seeds 500000–500199, whole table, mass x1 or x3 and friction x1 or x0.2, each with probability 1/2.
- Calibration on a separate broad-pilot set (seeds 510000–510199), so the evaluation seeds stay untouched: BC on R (500) succeeds on 45.0% -> within 20–70%, R size unchanged.
- BC policy: MLP 28 -> 256 -> 256 -> 256 -> 4 (tanh), inputs z-scored, MSE, AdamW lr 1e-3, batch 512, 20,000 steps (fixed number of updates for R and R ∪ C), success = object within 5 cm of goal at step 150.

## Pilot outcomes frozen in configs/frozen.yaml (2026-10-29)
- Router: LayerNorm before W_q/W_k and α = 3.0 (α ≤ 1 left the rate at 0.14–0.50 for target 0.10; α = 2 gave 0.120, on the edge; α = 3 gave 0.099). Same α at every N; pilot rates 0.207 (N=5), 0.099 (N=10), 0.045 (N=20), all within 20% of 1/N.
- Learning rates by pilot validation bits: learned 1e-3, fixed 3e-4, random 1e-3, velocity 1e-3, entropy 3e-4, flat 3e-4, proxy 3e-3. The entropy pilot used the best pilot proxy (lr 3e-3).
- Budgets: sim 200 epochs / patience 20 / 30 min; Flat 45 min and run alone (it is ~3x slower per step; +50% is the allowed margin); real 2000 epochs / patience 100 / 30 min (few, long episodes: 1–2 batches per epoch); Part B 800 epochs / patience 50 / 30 min.
- Conv-encoder ablation: depthwise causal conv written as a sum of k=4 shifted products (identical maths, 10x faster than grouped Conv1d on CPU).
- Exploratory refractory router: after a kept boundary at t, boundaries at t+1..t+3 are dropped (causal, applied before downsampling; p unchanged).

## Amendment 1 (2026-10-29) — see PREREGISTRATION.md
- Likelihood floor σ = 0.1 (normalised units) in every NLL; 0.01 was tried on pilot models and barely differs from the old clamp (σ ≈ 0.0067), so 0.1 was chosen (≈ the expert's action-noise scale).
- Action noise has its own RNG (seed + 104729, or `noise_seed`); all data regenerated; C2 now a true near-duplicate.
- Random rule counts the forced step-0 boundary towards the 1/N rate.
- Pilot runs: 2000 epochs, patience 150, ≤ 45 min (pilot only).
- Old pilot and the aborted main start are archived in `runs/_v1_before_amendment/` (not used for anything).
- Policy calibration on regenerated R (broad-pilot seeds 510000+): 41.5% success, within 20–70%; R stays at 500.
- Pilot v2 choices: α = 0.3 (smallest healthy α; checked at lr 3e-4 for N = 5/10/20); lr learned/fixed/random/velocity/flat 3e-4, entropy 1e-3, proxy 1e-3. The pilot v1 decisions above ("Pilot outcomes frozen") are superseded.
- Velocity and entropy thresholds are fitted on the validation split (the proxy's NLL on its own training data gave a held-out rate of 0.136 for target 0.10).
- Main-run budget set from two probe runs on the main train/val splits (test pointed at val, so test is untouched): see PREREGISTRATION.md Amendment 1.
- Téo delegated all decisions to me (2026-10-29); going public stays with him.

## Exploratory analyses added 2026-10-29 (after the main runs started, before any main result) — NOT pre-registered
Requested by Téo after the pilot report. H1–H4, their metrics and thresholds are unchanged; these are reported in a
separate "Exploratory" section of the report and never used for a pass/fail verdict. Code: `robochunk/exploratory.py`,
`scripts/analyze.py exploratory` → `runs/results_exploratory.json`.
- **Chance-corrected F1** (every method, N = 5/10/20, ±1 and ±3): F1 minus the mean F1 of the same boundaries shuffled
  within each episode (same count, steps 1..T−1, 1,000 shuffles, seed 0). For Fixed also minus a **grid chance**: the same
  every-N grid from a random offset per episode (1,000 draws). Why: at N = 10 a ±3 window covers 7 of every 10 steps, so a
  regular grid scores high by coverage alone (pilot: grid chance 0.561 vs Fixed 0.580 at ±3).
- **±1-step F1** (pre-registered as a sensitivity tolerance) is shown next to ±3 in the H1 table, not only in an appendix.
- **Burst-merged F1**: a boundary is kept only if it is ≥ 3 steps after the last kept one (step 0 counts as kept); F1 recomputed.
- **Boundary density by phase**: boundaries per step inside each of the 7 phases (a boundary at t counts for phase[t];
  step 0 excluded), per method, with the true phase starts as a reference row.
- Checked on pilot runs only (`scripts/analyze.py exploratory_pilot`, pilot val = test, not a result).

## Flat model sanity check (2026-10-29)
Question: Flat (full attention over every step) has more compute than any chunked model but had worse pilot validation
bits (−10.80 vs learned −12.09). Findings:
- Same learning-rate search as every method (3e-4 / 1e-3 / 3e-3, 2,000 epochs, patience 150); best 3e-4.
- Code path: with b = 1 everywhere the main network sees every step, the EMA dechunk is the identity, STE factor is 1;
  covered by `test_batch_equals_alone[flat]` and `test_fixed_rule_is_identity_dechunk`. No bug found.
- It overfits the 160 pilot episodes: lowest training loss of all models (−13.07 bits/step at 3e-4, −15.29 at 1e-3)
  while validation degrades (+0.79 at the last epoch at 1e-3). Train−val gap at the best epoch: Flat 1.12, Fixed 1.15,
  Random 0.58, learned 0.33. The small GRU proxy (52k parameters) has the best pilot validation bits (−12.50).
- Conclusion: a data-size effect on the pilot, not a bug. The main Flat runs (2,400 training episodes, 3 seeds) are the
  real test; if Flat still loses to Fixed there, that is reported as a result, with this diagnosis (train/val gap per method).

## Speed: threads per job (2026-10-29 20:45, during the main runs)
- Profile (main training data, 2 other jobs running): a training step is thousands of small GRU ops run step by step;
  each job used ~1.25 cores of its 6 threads. ms/step at 1 / 2 / 6 threads: learned N10 108 / 134 / 150, Fixed N5
  176 / – / 191, conv ablation 148 / 152 / 153, Flat 603 / 537 / 429.
- Loss after 12–15 training steps is bit-identical at every thread count for all four, so the change cannot alter results.
- From sim_fixed_N20 onwards: 1 thread per job, 6 for Flat (`scripts/run.sh`). Still at most 2 jobs; Flat still alone.
  Runs before the switch used 6 threads. The queue was restarted after the two running jobs finished (no run interrupted).
- Not done: Part B policies next to the Flat runs (extra load could push Flat into its 75-min time cap).

## v2 confirmatory test (2026-10-30) — see PREREGISTRATION_v2.md
- Approved by Téo (rev. 3) with the circular-shift chance for H1b-v2. Implementation choice within his spec "RNG seed 0":
  a new `default_rng(0)` per run (common random numbers across methods, independent of processing order).
- `scripts/analyze_v2.py` was smoke-tested on 5 pilot episodes with outputs discarded before the tag; no circular-shift
  chance was computed on the v1 test split or on any evaluation data.
- New fresh split `fresh` (seeds 700000–700299), not generated by default; SO-101 added to `robochunk/data_real.py`,
  downloaded only after the tag by `scripts/download_real.py` at the pinned revision.

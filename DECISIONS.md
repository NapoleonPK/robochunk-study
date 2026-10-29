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

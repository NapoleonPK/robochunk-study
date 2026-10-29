# RoboChunk — Pre-registration

Author: Téo Tran Van Lieu. Written 2026-10-29, before any main run. Only the pilot split (seeds 900000–900199) has been used so far; the pilot set learning rates, the router-health settings and budgets (see `docs/PILOT.md`), never hypotheses or thresholds. The commit that adds this file is quoted in the README; nothing below changes after it. Any deviation will be listed in `REPORT.md` with its reason. Every result is reported, including null and negative ones.

## 1. Questions

RoboChunk is a 1-stage H-Net (Hwang, Wang & Gu 2025) ported to continuous robot trajectories: a causal per-step encoder, a learned router that places chunk boundaries, a causal Transformer over chunks, an EMA + straight-through dechunker and a causal decoder trained to predict the next step with a Gaussian likelihood.

- **Part A.** Do the learned boundaries match real motion phases, and does learned chunking predict better than fixed chunking at matched compression?
- **Part B.** Does the same model's surprise on a new dataset (bits per step relative to its training distribution) predict how much that dataset improves a policy?

## 2. Hypotheses and pass criteria

| ID | Test | Pass criterion |
|---|---|---|
| H1 | Simulation, test split (300 episodes), N = N_main = 10. Boundary F1 at ±3 steps, one-to-one nearest-first matching, forced step-0 boundary ignored, per episode then averaged. | For **every** baseline (Fixed, Random, Entropy, Velocity minima): mean over test episodes of [RoboChunk F1 − baseline F1], each method's per-episode F1 averaged over its 3 seeds, is ≥ 0.10, **and** the 95% paired percentile bootstrap CI (10,000 resamples of test episodes) of that difference excludes 0. |
| H2 | Simulation, test split. Gaussian NLL in bits per step (sum over channels, normalised units), N ∈ {5, 10, 20}. | RoboChunk's mean test bits (3 seeds) is lower than Fixed's at the same N for at least 2 of the 3 values of N. At an N where RoboChunk's realised test compression (steps per chunk) is below 0.8 N, compute is not matched and that N counts as a fail. |
| H3 | Real teleoperated data, test split of each dataset, N_real = 12. Gripper events = hysteresis crossings (0.65 up / 0.35 down) of the gripper command normalised to [0, 1] by its 2nd/98th percentiles. | Share of RoboChunk boundaries (step 0 excluded) within ±3 steps of a gripper event, pooling the 3 seeds (each seed × episode is one unit), is ≥ 2× the mean share under 1,000 within-unit shuffles of the same number of boundaries over steps 1…T−1, **and** the one-sided permutation p < 0.01. Evaluated per dataset; H3 is *supported* if it passes on both datasets, *mixed* if on one, *not supported* if on none. |
| H4 | Part B, clean candidates C1–C11. Novelty = mean bits per step on the candidate − mean bits per step on held-out R, from RoboChunk (N = 10) trained on R only, averaged over 3 seeds. Gain = success(R ∪ C) − success(R) of a BC policy on 200 broad test episodes, averaged over 3 seeds. | Spearman ρ(novelty, gain) over C1–C11 ≥ 0.6. **Pre-registered expected failure mode:** each corrupted candidate X1–X4 has novelty above the median of all 15 candidates **and** a gain whose 95% CI (bootstrap over test episodes and seeds) has lower bound ≤ 0. We report for how many of the 4 this holds. |

## 3. Data (frozen; generator at this commit)

**Simulation.** `FetchPickAndPlace-v4` (gymnasium-robotics 1.4.2, MuJoCo 3.3.7), 25 Hz, ≤ 150 steps, goals on the table. Scripted expert with 7 phases (approach, descend, close, lift, transport, open, retreat); minimum-jerk profiles that overlap 0–3 steps across phases; per-phase speed U(0.5, 1.5); one pause of 0–5 steps inside every phase; Gaussian action noise std 0.05; intentionally failed first grasp + retry in 10% of episodes (and a retry after any natural failure). Ground-truth boundaries = steps where the phase label changes. Splits by seed: train 0–2399, val 100000–100299, test 200000–200299, pilot 900000–900199. Full details in `DECISIONS.md` and `robochunk/data_sim.py`.

**Model inputs.** Robot-only (11 channels): gripper position, gripper velocity, finger width, action. Ablation adds object position. Per-channel z-score fitted on the training split.

**Real data** (parquet only, native 30 fps, episode split 80/10/10 with seed 0):
`lerobot/svla_so100_pickplace` @ `728583b5eaf9e739a7f119e2def466fa1d552402` (50 episodes) and `lerobot/koch_pick_place_1_lego` @ `afc0b3e54679cb8a94283fbab5a4ccf0d405f10b` (102 episodes). Channels: 6 joint states + 6 joint actions. N_real = round(N_main × 30 / 25) = 12. The ±3-step tolerance is 0.10 s at 30 fps (0.12 s in simulation).

**Part B.** R = 500 core-region episodes (central ±7.5 cm, standard mass and friction). Held-out R = 100, R_val = 50. Candidates of 200 episodes: C1 exact duplicates of R; C2 same seeds as R with new action noise; C3 core, new seeds; C4 left strip; C5 far strip; C6 whole table; C7 mass ×3; C8 friction ×0.2; C9 whole table + mass ×3; C10 expert speed ×0.6; C11 100 left-strip + 100 R duplicates; X1 logged state lags action by 3 steps; X2 executed action noise std 0.5; X3 20% of frames dropped at random; X4 gripper action sign inverted in 30% of episodes. Broad test: 200 episodes (seeds 500000–500199), whole table, mass ×1 or ×3 and friction ×1 or ×0.2 with probability ½ each. Calibration on separate seeds (510000+): BC on R alone succeeds on 45.0% (target 20–70%), R size kept at 500.

## 4. Model and training (frozen)

- D0 = 128, 2-layer causal GRU encoder (ablation: 4 causal depthwise-conv k=4 + gated-MLP blocks), main network 4-layer causal pre-norm Transformer, D1 = 256, 4 heads, learned chunk-index positions; 2-layer GRU decoder; 3.81 M parameters.
- Router exactly as H-Net: p_t = ½(1 − cos(W_q x̂_t, W_k x̂_{t−1})), p_1 = 1, b_t = 1[p_t ≥ 0.5]; W_q = W_k = I at init; **LayerNorm before the routing projections; ratio-loss weight α = 3.0.** The paper's α = 0.03 did not hold the boundary rate here (rate 0.47 at N = 10); following the brief's fix order (LayerNorm, then higher α), α = 3.0 is the smallest tested value that put the pilot rate within 20% of 1/N (0.099 at N = 10). The same α is used at every N.
- Non-learned rules feed b_t with p_t = b_t into the identical model. Velocity minima: causal (a minimum at t−1 becomes a boundary at t), threshold on the training split for rate 1/N. Entropy: separate 2-layer GRU (width 64) proxy trained on the same training split; boundary where its NLL of x_t given x_<t exceeds a training-split threshold for rate 1/N. Random: Bernoulli(1/N) per episode, frozen. Flat: boundary at every step.
- AdamW, weight decay 0.01, batch 64 episodes, 200 warm-up steps then cosine, gradient clipping 1.0, early stopping on validation bits (best checkpoint), float32, CPU.
- Learning rate per method, chosen on the pilot split from {3e-4, 1e-3, 3e-3} by validation bits: RoboChunk 1e-3, Fixed 3e-4, Random 1e-3, Velocity minima 1e-3, Entropy 3e-4, Flat 3e-4, entropy proxy 3e-3 (pilot validation bits in `docs/PILOT.md`).
- Budgets: simulation max 200 epochs (7,600 updates), patience 20, ≤ 30 min (Flat: ≤ 45 min, run alone; it is a reference only). Real data: max 2,000 epochs, patience 100, ≤ 30 min. Part B sequence models: max 800 epochs, patience 50, ≤ 30 min. Seeds 0, 1, 2 for everything.

## 5. Run list

45 simulation runs (5 methods × N ∈ {5, 10, 20} × 3 seeds), 3 Flat, 6 ablations at N = 10 (object-state inputs; conv encoder), 18 real-data runs (RoboChunk, Fixed, Entropy × 2 datasets × 3 seeds), 3 entropy proxies (one per dataset), 6 Part B sequence models (RoboChunk N = 10 and Flat on R, 3 seeds), 48 BC policies (R and R ∪ each of 15 candidates, 3 seeds). Configs are generated by `scripts/make_configs.py` from `configs/frozen.yaml`.

## 6. Secondary measures (reported, not tested)

P, R and F1 at ±1 and ±5 steps; per-seed values; MSE; realised compression and measured FLOPs per step for every run; H3 on all episodes of each dataset and the same alignment for Fixed and Entropy boundaries; Flat-model novelty and nearest-neighbour novelty as comparison scores for H4, with their Spearman ρ.

## 7. Exploratory (labelled as exploratory in the report)

A **refractory router** (identical model; after a boundary at t, b is forced to 0 for steps t+1…t+3, p unchanged), N = 10, 3 seeds, scored with the H1 metric — motivated by the pilot burst pattern, reported as exploratory only and never counted towards H1; chunk length vs phase length; whether chunks of a repeated motion (the retry) merge within an episode; the burst structure of learned boundaries (share of 1-step gaps, already visible in the pilot); a 2-stage hierarchy (second router inside the main network, N2 = 3).

## 8. Known pilot observations disclosed up front

On the pilot split, the learned router fires in bursts (58–67% of gaps between consecutive boundaries are 1 step) around the close → lift → transport transitions, and its pilot F1 at ±3 (0.26–0.36) was below the Fixed rule's (0.61). These observations motivated no change to the hypotheses, which come from the brief written before any data existed.

# Pilot log (pilot split only: seeds 900000–900199, 160 train / 40 val)

All numbers come from `runs/summary_pilot_*.csv` (written by `scripts/summarize.py`).

## Phase statistics
- Mean steps per phase on the pilot split: 10.46 -> **N_main = 10**, sweep N = 5, 10, 20.
- Non-learned rules on the pilot split, ±3 tolerance (F1): N=5 fixed 0.598, velocity 0.539, random 0.455; N=10 fixed 0.613, velocity 0.509, random 0.441; N=20 fixed 0.427, velocity 0.314, random 0.327. Velocity minima are far below the 0.9 "too easy" warning level.
  (Computed before the generator was frozen, on 200 core-region episodes with seeds 900000+; the frozen pilot split uses the whole table.)

## Short pilot (300 epochs = 900 updates) — too short, superseded
- Learned router, lr 1e-3: boundary rate 0.47 (target 0.10), heading for all-boundaries.

## Router health sweep (N = 10, lr 1e-3, 1000 epochs, identity init on)
| LayerNorm before routing | α | val rate | test F1 ±3 | P | R |
|---|---|---|---|---|---|
| no | 0.03 | 0.498 | 0.237 | 0.140 | 0.802 |
| no | 0.1 | 0.370 | 0.255 | 0.161 | 0.658 |
| no | 0.3 | 0.210 | 0.334 | 0.253 | 0.552 |
| no | 1.0 | 0.140 | 0.313 | 0.296 | 0.405 |
| yes | 0.03 | 0.471 | 0.258 | 0.155 | 0.815 |
| yes | 0.1 | 0.391 | 0.278 | 0.175 | 0.744 |
| yes | 0.3 | 0.239 | 0.363 | 0.259 | 0.650 |
| yes | 1.0 | 0.143 | 0.296 | 0.282 | 0.393 |

- None within 20% of 1/N at α ≤ 1. The NLL here is summed over 11 continuous channels, so its gradients are much larger than the ratio loss's; α = 0.03 (the paper's value for bytes) does not hold the rate.
- Learned boundaries come in bursts: at α = 1, 67% of gaps between consecutive boundaries are 1 step. Bursts sit where the encoder state changes fast (around close -> lift -> transport). The H-Net router compares adjacent steps only; in smooth continuous signals a fast change lasts several steps.


## Pilot v2 — Amendment 1 (σ floor 0.1, regenerated data, 2,000 epochs, patience 150)
Pilot v1 above is superseded; its runs are archived in `runs/_v1_before_amendment/`. Table values: `runs/summary_pilot2.csv`.

Router sweep (N = 10, lr 1e-3):
| run | α | LayerNorm | val rate | F1 ±3 | val bits |
|---|---|---|---|---|---|
| p2_router_ln0_a0.03 | 0.03 | False | 0.257 | 0.3488 | -11.96 |
| p2_router_ln1_a0.03 | 0.03 | True | 0.2838 | 0.343 | -11.81 |
| p2_router_ln1_a0.3 | 0.3 | True | 0.0988 | 0.3218 | -11.87 |
| p2_router_ln1_a1.0 | 1 | True | 0.09511 | 0.3526 | -11.91 |
| p2_router_ln1_a10.0 | 10 | True | 0.1145 | 0.3909 | -12.05 |
| p2_router_ln1_a3.0 | 3 | True | 0.1053 | 0.3438 | -11.96 |

Learning-rate search (N = 10; learned at α = 0.3; entropy thresholds fitted on the validation split, proxy = p2_lr_proxy_lr1e-3):
| run | method | lr | val bits | best epoch | val rate | F1 ±3 |
|---|---|---|---|---|---|---|
| p2_lr_entropy_v2_lr1e-3 | entropy | 0.001 | -11.18 | 321 | 0.1 | 0.4456 |
| p2_lr_entropy_v2_lr3e-3 | entropy | 0.003 | -10.73 | 292 | 0.1 | 0.4456 |
| p2_lr_entropy_v2_lr3e-4 | entropy | 0.0003 | -11.05 | 774 | 0.1 | 0.4456 |
| p2_lr_fixed_lr1e-3 | fixed | 0.001 | -10.47 | 262 | 0.1059 | 0.5796 |
| p2_lr_fixed_lr3e-3 | fixed | 0.003 | -9.707 | 274 | 0.1059 | 0.5796 |
| p2_lr_fixed_lr3e-4 | fixed | 0.0003 | -10.71 | 638 | 0.1059 | 0.5796 |
| p2_lr_flat_lr1e-3 | flat | 0.001 | -10.39 | 232 | 1 | 0.1568 |
| p2_lr_flat_lr3e-3 | flat | 0.003 | -9.461 | 149 | 1 | 0.1568 |
| p2_lr_flat_lr3e-4 | flat | 0.0003 | -10.8 | 397 | 1 | 0.1568 |
| p2_lr_learned_lr1e-3 | learned | 0.001 | -11.87 | 568 | 0.0988 | 0.3218 |
| p2_lr_learned_lr3e-3 | learned | 0.003 | -11.17 | 342 | 0.1247 | 0.3482 |
| p2_lr_learned_lr3e-4 | learned | 0.0003 | -12.09 | 1205 | 0.1053 | 0.358 |
| p2_lr_proxy_lr1e-3 | proxy | 0.001 | -12.5 | 1964 | nan | nan |
| p2_lr_proxy_lr3e-3 | proxy | 0.003 | -12.42 | 1408 | nan | nan |
| p2_lr_proxy_lr3e-4 | proxy | 0.0003 | -11.99 | 1964 | nan | nan |
| p2_lr_random_lr1e-3 | random | 0.001 | -11.31 | 400 | 0.09972 | 0.4337 |
| p2_lr_random_lr3e-3 | random | 0.003 | -10.68 | 315 | 0.09972 | 0.4337 |
| p2_lr_random_lr3e-4 | random | 0.0003 | -11.46 | 767 | 0.09972 | 0.4337 |
| p2_lr_velocity_lr1e-3 | velocity | 0.001 | -11.2 | 406 | 0.1031 | 0.4861 |
| p2_lr_velocity_lr3e-3 | velocity | 0.003 | -10.34 | 286 | 0.1031 | 0.4861 |
| p2_lr_velocity_lr3e-4 | velocity | 0.0003 | -11.36 | 808 | 0.1031 | 0.4861 |

Router health at the frozen lr 3e-4:
| run | N | α | val rate | F1 ±3 |
|---|---|---|---|---|
| p2_health_lr3e-4_N20_a0.3 | 20 | 0.3 | 0.05971 | 0.4056 |
| p2_health_lr3e-4_N5_a0.3 | 5 | 0.3 | 0.237 | 0.3362 |
| p2_health_lr3e-4_N5_a1.0 | 5 | 1 | 0.2225 | 0.3455 |
| p2_health_lr3e-4_N5_a3.0 | 5 | 3 | 0.1761 | 0.403 |

Notes: on the pilot, validation and test are the same 40 episodes, so pilot F1 is not a held-out number.
The velocity runs above used a threshold fitted on the training split (rate 0.103); main runs fit it on validation.
The entropy runs `p2_lr_entropy_lr*` (training-split threshold, val rate 0.136) are superseded by `p2_lr_entropy_v2_lr*`.
Proxies at lr 1e-3 and 3e-4 reached the 2,000-epoch pilot cap still improving slightly.

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

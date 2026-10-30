# History note and commit-hash map

This repository was made public on 2026-10-30. Before publication the git history was rewritten **once**, only to
replace one string — the author's personal email address, which appeared in `docs/BRIEF.md` — with `[email redacted]`
(`git filter-repo --replace-text`). Nothing else changed: same files, same content, same author and commit dates, same
commit messages, same tags. Rewriting changes every commit hash, so documents written before publication
(`PREREGISTRATION.md`, `PREREGISTRATION_v2.md`, `DECISIONS.md`, run `git.txt` files) cite the **old** hashes.
Use this table to translate them. The earlier private repository is not public; this map is the bridge.

| commit date (author) | old hash | new hash | tag | message |
|---|---|---|---|---|
| 2026-10-29 11:29:55 +0100 | `e3a0e32` | `1724dd6` |  | Scaffold: sim data generator, RoboChunk model, baselines, metrics, training loop, tests |
| 2026-10-29 11:30:09 +0100 | `765a350` | `518e5ae` |  | Cast float config fields |
| 2026-10-29 11:30:39 +0100 | `2ae720e` | `ecb2a21` |  | Match H-Net: end-of-stack norms, near-zero residual init |
| 2026-10-29 11:38:11 +0100 | `9e210bf` | `b3d7d16` |  | Real-data loader, Part B datasets and BC policy, decisions log, source notes |
| 2026-10-29 14:08:32 +0100 | `1f1faa0` | `7ffc024` | prereg | Pre-registration: hypotheses, frozen pilot settings, run list |
| 2026-10-29 19:28:48 +0100 | `fa40b3b` | `7fff994` |  | Amendment 1: likelihood floor, separate noise RNG, val-fitted thresholds, re-run pilot, new budgets |
| 2026-10-29 19:28:57 +0100 | `f61d0f3` | `fb5e2ad` | prereg-amend1 | Amendment 1, item 8: training budgets from main-split probes |
| 2026-10-29 19:35:31 +0100 | `73e3735` | `e27cf64` |  | Exploratory boundary analyses (labelled, not pre-registered); Flat sanity check log |
| 2026-10-29 20:37:41 +0100 | `9768bb7` | `8bdff38` |  | Run GRU-model jobs on 1 thread (bit-identical, ~28% faster for learned); Flat keeps 6 |
| 2026-10-30 09:27:24 +0100 | `dad5768` | `56b2ea3` |  | Report notes: real-data caveats and interim observations |
| 2026-10-30 09:41:08 +0100 | `6b22c38` | `6ceae62` |  | Exploratory analyses (chance-corrected/burst-merged F1, real burst locations, entropy event finding); DRAFT v2 pre-registration |
| 2026-10-30 09:51:46 +0100 | `5b0466a` | `b358f3b` |  | DRAFT v2 pre-registration rev. 2: H1b-v2 co-primary, H3b-v2 on unused SO-101, refractory rule spelled out, frozen hashes |
| 2026-10-30 09:57:49 +0100 | `d9b86c7` | `9e99328` | prereg-v2 | Pre-registration v2 (approved): confirmatory test of the refractory router |
| 2026-10-30 09:58:28 +0100 | `7930e64` | `ac72fec` |  | v2: fresh sim split generated after tag prereg-v2 (SHA-256 recorded); SO-101 downloaded at pinned revision |

Full 40-character hashes: `docs/commit-map.txt` (output of git filter-repo).

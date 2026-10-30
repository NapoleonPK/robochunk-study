# Proposed follow-up: can novelty plus a simple quality check catch subtle corruptions? (DRAFT, not run, not pre-registered)

## Why
H4 showed that compression novelty ranks clean data by policy gain, but all four corrupted candidates (X1–X4) were scored as
highly novel while hurting the policy. Those corruptions were blatant (novelty up to 327 against at most 1.9 for clean data), so
any threshold would catch them. Real teleoperation faults are subtler. The question for a data-evaluation service: **with
corruptions that do not stand out on novelty, does a simple quality check still separate them from clean data?**

## Design (paired, small)
Two clean *base* sets of 200 fresh episodes each: **B-core** (same distribution as R, low novelty) and **B-whole** (whole table,
like C6, high novelty and positive gain). Each base also gets three subtly corrupted copies built from **the same episodes**,
so the corruption is the only difference:

| ID | Corruption | Applied to |
|---|---|---|
| L1 | 1-step lag: action[t] replaced by action[t−1] (first step repeats) | every episode |
| G5 | 5% of gripper events mislabelled: for a random 5% of open/close events, the gripper command is inverted for the 3 steps after the event | events across the set |
| N2 | mild extra action noise, std 0.1 executed (2× the expert's 0.05; X2 used 0.5) | every episode |

That gives 2 bases × (clean + 3) = **8 candidates**. All other settings are as in Part B (R = 500 core episodes, same broad
test set, BC 3 seeds).

## Scores (all from models trained on R only; nothing trained on candidates)
- **Novelty**: as in H4 (RoboChunk N = 10, mean bits per step on candidate − on held-out R, 3 seeds).
- **Confound to avoid**: anything calibrated on R also reacts to clean-but-novel data (B-whole), so a naive "surprise vs R"
  check would just re-measure novelty. Both checks are therefore built to be insensitive to *where* the object is:
- **Quality check A, robot-only action consistency**: an inverse-dynamics MLP trained on R predicts action[t] from the
  *robot* channels only (gripper position, velocity, finger state at t and t+1; object and goal channels excluded). The arm's
  response to a command does not depend on the object's location, so clean novel data should fit as well as R. Score = mean
  squared residual on the candidate ÷ on held-out R. Lag and inverted gripper commands break this link directly.
- **Quality check B, within-candidate spikiness**: ratio of the 99th percentile to the median of per-step bits (RoboChunk,
  trained on R), computed on the candidate and divided by the same ratio on held-out R. A shift in *level* (novelty) cancels;
  isolated spikes (mislabelled events) do not.
- **Gate**: a candidate fails quality if either score exceeds its 95th percentile over 200 bootstrap batches of held-out R
  (threshold fixed on R only, before any candidate is scored). F2 checks explicitly that B-whole is not flagged.

## Proposed hypotheses (to pre-register before running)
- **F1 (novelty is not enough)**: within each base, the three corrupted copies differ in novelty from their clean base by less
  than the gap between B-core and B-whole. Expected: true.
- **F2 (quality check separates)**: episode-level AUROC (clean vs corrupted episodes of the same base) ≥ 0.8 for the better
  of checks A and B, for each of L1, G5 and N2 (for G5, over the episodes that contain a mislabelled event; the other ~95% are
  untouched); and the batch-level gate flags all 6 corrupted candidates and neither clean base. Expected hard case: N2 noise
  is *executed*, so states stay consistent with actions and check A should not see it; only a worse demonstrator remains.
  If N2 is not separated and also does not reduce gain (F3), that is the right answer, not a miss.
- **F3 (does it matter for policies)**: report gain (95% CI) for all 8 candidates. Lag and mild noise may barely hurt BC at
  25 Hz; if a corruption does not reduce gain, flagging it is a false alarm for policy value and the report says so.
- **F4 (combined ranking)**: Spearman ρ(score, gain) over the 8 candidates, score = novelty with gate-failed candidates
  placed last, versus novelty alone. Descriptive (n = 8 is too small for a pass/fail threshold).

## Lessons carried over from Part B
- **BC budget**: decided after the post-hoc matched-passes test (`scripts/h4_matched_epochs.py`). If fixed updates explain the
  negative C2/C3 gains, the follow-up trains for a fixed number of passes over the data instead.
- Report CIs for every ρ (n is small) and keep any re-analysis labelled post hoc.

## Cost on the Mac mini
Data generation about 10 min; inverse-dynamics model 1 small training job; novelty and checks need no new training of the
sequence model; 24 BC policies at about 35 s each with 2 workers. **Under 1 hour, at most 2 jobs, under 1 GB of disk.**

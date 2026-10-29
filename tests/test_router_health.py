"""Router health on the pilot split: the learned boundary rate is within 20% of 1/N, no collapse.
Reads the Amendment-1 pilot runs at the frozen alpha 0.3 and lr 3e-4
(run `scripts/run.sh configs/pilot_v2/lr_learned_lr3e-4.yaml configs/pilot_v2/health_lr3e-4_N*_a0.3.yaml` first)."""
import json
from pathlib import Path

import pytest

RUNS = {10: "p2_lr_learned_lr3e-4", 5: "p2_health_lr3e-4_N5_a0.3", 20: "p2_health_lr3e-4_N20_a0.3"}


@pytest.mark.parametrize("N", sorted(RUNS))
def test_rate_within_20_percent(N):
    p = Path("runs") / RUNS[N] / "metrics.json"
    if not p.exists():
        pytest.skip(f"pilot run {RUNS[N]} not found")
    m = json.load(open(p))
    for split in ("val", "test"):
        rate = m[f"{split}_rate"]
        assert abs(rate - 1 / N) <= 0.2 / N, (split, rate)

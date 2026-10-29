"""Router health on the pilot split: the learned boundary rate is within 20% of 1/N, no collapse.
Reads the pilot runs (run `scripts/run.sh configs/router/router_lntrue_a3.0.yaml configs/router/health_N*.yaml` first)."""
import json
from pathlib import Path

import pytest

RUNS = {10: "router_lntrue_a3.0", 5: "health_learned_N5", 20: "health_learned_N20"}


@pytest.mark.parametrize("N", sorted(RUNS))
def test_rate_within_20_percent(N):
    p = Path("runs") / RUNS[N] / "metrics.json"
    if not p.exists():
        pytest.skip(f"pilot run {RUNS[N]} not found")
    m = json.load(open(p))
    for split in ("val", "test"):
        rate = m[f"{split}_rate"]
        assert abs(rate - 1 / N) <= 0.2 / N, (split, rate)

"""Generate the Part A simulated splits. Seeds define splits; see DECISIONS.md."""
import argparse
import time
from pathlib import Path

from robochunk.data_sim import EpisodeSpec, generate, save

SPLITS = {  # name: (first seed, count)
    "pilot": (900_000, 200),
    "train": (0, 2400),
    "val": (100_000, 300),
    "test": (200_000, 300),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--out", default="data/sim")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name in args.splits:
        s0, n = SPLITS[name]
        specs = [EpisodeSpec(seed=s0 + i, region="whole") for i in range(n)]
        t = time.time()
        eps = generate(specs)
        save(f"{args.out}/{name}.npz", eps, specs)
        print(f"{name}: {n} episodes in {time.time() - t:.0f}s")

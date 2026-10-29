"""Collect runs/*/metrics.json into one CSV. usage: python scripts/summarize.py [prefix] [out.csv]"""
import json
import sys
from pathlib import Path

import pandas as pd

prefix = sys.argv[1] if len(sys.argv) > 1 else ""
rows = [json.load(open(p)) for p in sorted(Path("runs").glob(f"{prefix}*/metrics.json"))]
df = pd.DataFrame(rows)
if len(sys.argv) > 2:
    df.to_csv(sys.argv[2], index=False)
cols = [c for c in ["run_id", "lr", "epochs", "best_epoch", "stop", "minutes", "val_bits", "val_rate", "test_F1_tol3",
                    "test_compression"] if c in df]
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(df[cols].round(4).to_string(index=False))

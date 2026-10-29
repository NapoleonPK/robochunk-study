"""Generate Part B: reference R, held-out R, candidates C1–C11 and X1–X4. Specs in robochunk/partb.py."""
import argparse
import time

from robochunk.partb import build_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--r-size", type=int, default=500)
    args = ap.parse_args()
    t = time.time()
    build_all(args.only, args.r_size)
    print(f"done in {time.time() - t:.0f}s")

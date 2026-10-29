#!/bin/bash
# Run one or more configs, at most $JOBS at a time (default 2), keeping the Mac awake.
# usage: scripts/run.sh configs/a.yaml [configs/b.yaml ...]
#        JOBS=1 scripts/run.sh configs/pilot/*.yaml
set -u
cd "$(dirname "$0")/.."
# Threads: 1 per job (the GRU stacks run step by step; extra threads only add overhead: learned 108 vs 150 ms/step),
# 6 for Flat (attention over every step uses them: 429 vs 603 ms/step). Losses are bit-identical across thread counts.
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONWARNINGS=ignore
JOBS=${JOBS:-2}
run_one() {
  id=$(grep '^run_id:' "$1" | awk '{print $2}')
  mkdir -p "runs/$id"
  th=${THREADS:-1}
  grep -q '^method: flat' "$1" && th=${FLAT_THREADS:-6}
  OMP_NUM_THREADS=$th .venv/bin/python -m robochunk.train "$1" >> "runs/$id/stdout.txt" 2>&1 \
    && echo "ok   $id" || echo "FAIL $id (see runs/$id/stdout.txt)"
}
export -f run_one
printf '%s\n' "$@" | caffeinate -i xargs -P "$JOBS" -I{} bash -c 'run_one {}'

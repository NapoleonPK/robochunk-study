#!/bin/bash
# Run one or more configs, at most $JOBS at a time (default 2), keeping the Mac awake.
# usage: scripts/run.sh configs/a.yaml [configs/b.yaml ...]
#        JOBS=1 scripts/run.sh configs/pilot/*.yaml
set -u
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=${THREADS:-6} PYTHONWARNINGS=ignore
JOBS=${JOBS:-2}
run_one() {
  id=$(grep '^run_id:' "$1" | awk '{print $2}')
  mkdir -p "runs/$id"
  .venv/bin/python -m robochunk.train "$1" >> "runs/$id/stdout.txt" 2>&1 \
    && echo "ok   $id" || echo "FAIL $id (see runs/$id/stdout.txt)"
}
export -f run_one
printf '%s\n' "$@" | caffeinate -i xargs -P "$JOBS" -I{} bash -c 'run_one {}'

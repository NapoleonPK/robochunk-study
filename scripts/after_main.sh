#!/bin/bash
# After the v1 main queue: Part B policies (H4), then the v2 runs. At most 2 jobs at any time.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONWARNINGS=ignore
until grep -q ALL_DONE runs/main_queue.log; do sleep 60; done
echo "MAIN_DONE $(date +%H:%M)"
caffeinate -i .venv/bin/python scripts/partb_policies.py --workers 2 > runs/partb_policies.log 2>&1 && echo "POLICIES_DONE $(date +%H:%M)" || echo "POLICIES_FAIL $(date +%H:%M)"
JOBS=2 scripts/run.sh configs/v2/*.yaml
echo "V2_DONE $(date +%H:%M)"

#!/bin/bash
# Wait until no training job is running, then run scripts/run.sh with the given configs.
while pgrep -f "robochunk.train" > /dev/null; do sleep 20; done
exec "$(dirname "$0")/run.sh" "$@"

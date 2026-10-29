#!/bin/bash
# usage: scripts/run_after_pid.sh PID configs...  — wait for process PID to exit, then run the configs.
pid=$1; shift
while kill -0 "$pid" 2>/dev/null; do sleep 20; done
exec "$(dirname "$0")/run.sh" "$@"

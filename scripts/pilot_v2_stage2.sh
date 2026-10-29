#!/bin/bash
# Wait for stage 1, pick the best entropy proxy by validation bits, then run learned/health/entropy pilots.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 20; done
best=$(.venv/bin/python -c "
import json,glob; r=[json.load(open(p)) for p in glob.glob('runs/p2_lr_proxy_*/metrics.json')]
print(min(r,key=lambda m:m['val_bits'])['run_id'])")
echo "best proxy: $best"
for lr in 3e-4 1e-3 3e-3; do
cat > configs/pilot_v2/lr_entropy_lr$lr.yaml <<EOC
run_id: p2_lr_entropy_lr$lr
seed: 0
splits: {train: "pilot[0:160]", val: "pilot[160:200]", test: "pilot[160:200]"}
max_epochs: 2000
patience: 150
max_minutes: 45
method: entropy
N: 10
lr: $lr
proxy_run: runs/$best
EOC
done
exec scripts/run.sh configs/pilot_v2/lr_learned_*.yaml configs/pilot_v2/health_*.yaml configs/pilot_v2/lr_entropy_*.yaml

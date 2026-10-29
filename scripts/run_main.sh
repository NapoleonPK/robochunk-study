#!/bin/bash
# All pre-registered sequence-model runs, in dependency order (entropy runs need their proxies).
cd "$(dirname "$0")/.."
ls configs/proxy/*.yaml > runs/main_order.txt
ls configs/sim/*.yaml | grep -v -e _entropy_ -e _flat_ >> runs/main_order.txt
ls configs/sim/*_entropy_*.yaml configs/ablation/*.yaml configs/exploratory/*.yaml configs/real/*.yaml >> runs/main_order.txt
ls configs/partb/*robochunk*.yaml configs/partb/*flat*.yaml >> runs/main_order.txt
JOBS=2 scripts/run.sh $(cat runs/main_order.txt)
JOBS=1 scripts/run.sh configs/sim/*_flat_*.yaml   # Flat runs alone (75 min budget)
echo ALL_DONE

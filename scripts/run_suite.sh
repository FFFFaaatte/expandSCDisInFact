#!/usr/bin/env bash
set -euo pipefail
# Use PYTHON to select the Python environment on the GPU machine.
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT="${OUTPUT:-runs}"
for seed in ${SEEDS:-0 1 2}; do
  for experiment in gbm gbm_celltype sim_is sim_oos; do
    if [[ -f "$OUTPUT/$experiment/seed_$seed/timing.json" && -f "$OUTPUT/$experiment/seed_$seed/metrics.csv" ]]; then
      continue
    fi
    "$PYTHON" -m cellshift.run --experiment "$experiment" --seed "$seed" --device cuda --data-root "$DATA_ROOT" --output "$OUTPUT"
  done
done
"$PYTHON" -m cellshift.report --root "$OUTPUT"

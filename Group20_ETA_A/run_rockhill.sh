#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Activate your GROUP'S environment first. Install requirements there if needed.
# Set ETA_INPUT_GLOB explicitly to the months agreed with the group.
# Default January only; do not silently change the assessed data scope.
ETA_INPUT_GLOB="${ETA_INPUT_GLOB:-/home/cs4207/data/yellow_tripdata_2026-01.parquet}"
ETA_RUN_NAME="${ETA_RUN_NAME:-rockhill_$(date -u +%Y%m%dT%H%M%SZ)}"
# Confirm cs4207 permits 4 CPUs; never request a specific GPU or the student partition.
# Keeping library threads at one avoids nested process/thread oversubscription.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --partition=cs4207 --cpus-per-task=4 python eta_data.py \
  --input-glob "$ETA_INPUT_GLOB" --workers 4 --output "outputs/$ETA_RUN_NAME"
srun --partition=cs4207 --cpus-per-task=4 python eta_data.py \
  --input-glob "$ETA_INPUT_GLOB" --benchmark --repeats 3 --output "outputs/${ETA_RUN_NAME}_benchmark"
printf '\nSaved outputs/%s and outputs/%s_benchmark\n' "$ETA_RUN_NAME" "$ETA_RUN_NAME"

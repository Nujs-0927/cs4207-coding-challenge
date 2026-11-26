#!/usr/bin/env bash
# Group 20 - Part C on Rockhill. Run from a Lab PC session, inside Group20_ETA_C,
# with Group20_ETA_A and Group20_ETA_B next to it and the group's environment activated.
#   bash run_rockhill_c.sh                                                   # A's 100k sample
#   ETA_DATA=../Group20_ETA_A/outputs/<A_run> bash run_rockhill_c.sh         # full month
set -euo pipefail
cd -- "$(dirname -- "$0")"
export ETA_A_DIR="${ETA_A_DIR:-../Group20_ETA_A}"
export ETA_B_DIR="${ETA_B_DIR:-../Group20_ETA_B}"
export ETA_DATA="${ETA_DATA:-$ETA_A_DIR/outputs/local_jan2026/sample_100k.parquet}"
export ETA_WORKERS="${ETA_WORKERS:-1 2 4}"
export ETA_RUN="${ETA_RUN:-rockhill_$(date -u +%Y%m%dT%H%M%SZ)}"
# one BLAS thread per process: avoids oversubscription inside the process pool
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun --partition=cs4207 --cpus-per-task=4 python -m unittest discover -s tests
srun --partition=cs4207 --cpus-per-task=4 jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=7200 C_Baseline_Models.ipynb --output "C_Baseline_Models_${ETA_RUN}.ipynb"
python make_c_materials.py --run "$ETA_RUN"
echo "Done: outputs/$ETA_RUN, C_Baseline_Models_${ETA_RUN}.ipynb, C_Poster_Section_EN.html, C_Speaking_EN.txt"

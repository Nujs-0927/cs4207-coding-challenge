#!/usr/bin/env bash
# Group 20 - Part B on Rockhill. Run from a Lab PC session, inside Group20_ETA_B,
# with Group20_ETA_A next to it and the environment on: source ~/envs/myenv/bin/activate
#   bash run_rockhill_b.sh                      # uses A's 100k sample
#   ETA_DATA=../Group20_ETA_A/outputs/<A_run> bash run_rockhill_b.sh   # full data (A's cleaned_parts)
set -euo pipefail
cd -- "$(dirname -- "$0")"
export ETA_A_DIR="${ETA_A_DIR:-../Group20_ETA_A}"
export ETA_DATA="${ETA_DATA:-$ETA_A_DIR/outputs/local_jan2026/sample_100k.parquet}"
export ETA_WORKERS="${ETA_WORKERS:-1 2 4}"
export ETA_RUN="${ETA_RUN:-rockhill_$(date -u +%Y%m%dT%H%M%SZ)}"
# cs4207 partition (23 Nov announcement), no GPU request. --mem is required:
# without it Slurm reserves the whole machine's memory and the job waits.
SRUN=(srun --partition=cs4207 --cpus-per-task=4 --mem=16G --time=01:00:00)
# one BLAS thread per process: avoids oversubscription inside the process pool
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${SRUN[@]}" python -m unittest discover -s tests
"${SRUN[@]}" jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=3600 B_EDA_Features.ipynb --output "B_EDA_Features_${ETA_RUN}.ipynb"
srun --partition=cs4207 --mem=1G python make_b_materials.py --run "$ETA_RUN"
echo "Done: outputs/$ETA_RUN, B_EDA_Features_${ETA_RUN}.ipynb, B_Poster_Section_EN.html, B_Speaking_EN.txt"

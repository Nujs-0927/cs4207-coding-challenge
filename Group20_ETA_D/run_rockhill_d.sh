#!/usr/bin/env bash
# Group 20 - Part D on Rockhill. Run from a Lab PC session, inside Group20_ETA_D,
# with Group20_ETA_B and Group20_ETA_C next to it and the group's environment activated.
#   bash run_rockhill_d.sh                                    # B's / C's local_sample outputs (quick check)
#   ETA_B_RUN=<B_rockhill_run> ETA_C_RUN=<C_rockhill_run> bash run_rockhill_d.sh   # full month (poster numbers)
# B and C must have been run on the SAME data first: D reads B's features and C's scoreboard.
set -euo pipefail
cd -- "$(dirname -- "$0")"
export ETA_B_DIR="${ETA_B_DIR:-../Group20_ETA_B}"
export ETA_C_DIR="${ETA_C_DIR:-../Group20_ETA_C}"
export ETA_B_RUN="${ETA_B_RUN:-local_sample}"
export ETA_C_RUN="${ETA_C_RUN:-local_sample}"
export ETA_RUN="${ETA_RUN:-rockhill_$(date -u +%Y%m%dT%H%M%SZ)}"
export ETA_THREADS="${ETA_THREADS:-4}"
# Pin BLAS/OpenMP to the thread count we ask Slurm for, so timings and results are comparable
export OMP_NUM_THREADS="$ETA_THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --partition=cs4207 --cpus-per-task="$ETA_THREADS" python -m unittest discover -s tests
srun --partition=cs4207 --cpus-per-task="$ETA_THREADS" jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=10800 D_Advanced_Models.ipynb --output "D_Advanced_Models_${ETA_RUN}.ipynb"
python make_d_materials.py --run "$ETA_RUN"
echo "Done: outputs/$ETA_RUN, D_Advanced_Models_${ETA_RUN}.ipynb, D_Poster_Section_EN.html, D_Speaking_EN.txt"

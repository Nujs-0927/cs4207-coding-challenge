#!/usr/bin/env bash
# Local development run (no Slurm): python make_d_notebook.py first if you changed it.
set -euo pipefail
cd -- "$(dirname -- "$0")"
export ETA_RUN="${ETA_RUN:-local_sample}" ETA_THREADS="${ETA_THREADS:-2}"
python -m unittest discover -s tests
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 D_Advanced_Models.ipynb --output D_Advanced_Models.ipynb
python make_d_materials.py --run "$ETA_RUN"

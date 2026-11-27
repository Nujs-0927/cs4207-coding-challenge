# Group 20 — Part D (Advanced models, results, conclusion)

## Files
| File | Purpose |
|---|---|
| `D_Advanced_Models.ipynb` | **Part D notebook**, executed (local run on A's 100k sample). C can merge it into the final notebook. |
| `d_models.py` | Loading B's features, Random Forest / LightGBM / XGBoost, validation tuning, deterministic prediction, reproducibility checks |
| `d_figures.py` | Model comparison, feature importance, predicted vs actual (same style as B/C) |
| `make_d_notebook.py` | Regenerates the (unexecuted) notebook from code |
| `make_d_materials.py` | Builds `D_Poster_Section_EN.html` + `D_Speaking_EN.txt` from a run's `d_summary.json` |
| `tests/test_d_models.py` | 5 tests: no target/leakage/day_of_month in X, fixed zone categories, deterministic forest prediction, seeded LightGBM same on 1 and 2 threads |
| `run_rockhill_d.sh` / `run_local_d.sh` | Tests + notebook + poster/script (Rockhill uses `srun --partition=cs4207`) |
| `outputs/<run>/` | `d_summary.json`, `scoreboard_all.csv` (C's rows + D's rows), tuning trials, feature importance, figures |

## Depends on
- **B**: `outputs/<run>/features_{train,validation,test}.parquet` (33 columns; D drops `day_of_month`, uses the zone codes as categories).
- **C**: `c_models.evaluate / to_minutes / Scoreboard` and `outputs/<run>/C_scoreboard.csv` for the **same** data (the notebook asserts the row counts match).
- Split: train 1–21 Jan, validation 22–26 Jan, test 27–31 Jan. Tuning and early stopping use validation only; test is scored once.

## Run on Rockhill (full month, poster numbers)
Put `Group20_ETA_A`, `_B`, `_C`, `_D` side by side, activate the group environment, then:
```bash
pip install -r requirements.txt       # lightgbm, xgboost, scikit-learn, nbconvert ...
# 1. A cleans the full month, 2. B builds full-month features, 3. C scores its models on them
cd Group20_ETA_D
ETA_B_RUN=<B_rockhill_run> ETA_C_RUN=<C_rockhill_run> bash run_rockhill_d.sh
```
`ETA_THREADS` (default 4) sets threads per model and the Slurm `--cpus-per-task`. The Random Forest uses 25 % of the training rows per tree, so the full month stays tractable.

## Rubric contribution
- **Non-determinism (criterion 5):** a real source found in Part D — scikit-learn's Random Forest `predict()` sums tree outputs from threads in completion order (4 different outputs in 4 calls). Fixed by one prediction thread; re-run shows 1 of 4. Seeded refits of all three models are identical across reruns and 1 vs 2 threads.
- **Poster (criterion 7):** results table, model comparison, feature importance, predicted vs actual, limitations, conclusion.

**Status:** local development run on the 100k sample, not Rockhill evidence.

# Group 20 — Part C (Time split, shared evaluation, baseline models)

## Files
| File | Purpose |
|---|---|
| `C_Baseline_Models.ipynb` | **Part C notebook**, executed on the full January file (local run) |
| `c_models.py` | Split check, **shared `evaluate()`**, `Scoreboard`, baselines, timing helper |
| `c_parallel.py` | Ridge, hyperparameter search, all-reduce gradient, non-determinism experiments |
| `c_figures.py` | Figures used in the notebook and on the poster |
| `tests/test_c_models.py` | 18 tests (metrics = scikit-learn, parallel = sequential, no leakage, back-off) |
| `run_rockhill_c.sh` | Tests + notebook on Rockhill (`cs4207` partition), then regenerates poster section and script |
| `make_c_materials.py` | Builds `C_Poster_Section_EN.html` and `C_Speaking_EN.txt` from a run's `c_summary.json` |
| `outputs/<run>/` | `C_scoreboard.csv`, `c_summary.json`, figures |

## Confirmed split (same dates as A's example and B's features — nobody has to rerun for this)
Train 1–21 Jan · validation 22–26 Jan (`>= 2026-01-22`) · test 27–31 Jan (`>= 2026-01-27`).
Fit on train only, choose on validation, score test **once**.

## For D: score your models with the same function
```python
import sys; sys.path.insert(0, '../Group20_ETA_C')
from c_models import Scoreboard, evaluate, to_minutes

board = Scoreboard.load('../Group20_ETA_C/outputs/<run>/C_scoreboard.csv')   # C's rows
pred_val = to_minutes(model.predict(X_val))      # only if you trained on log1p(duration)
board.add('LightGBM', 'validation', y_val, pred_val, owner='D')
board.add('LightGBM', 'test', y_test, pred_test, owner='D')                  # final model only
board.table()                                    # one row per model, ready for the poster
```
`evaluate(y_true, y_pred)` → `{'n', 'MAE', 'RMSE', 'R2', 'bias'}`, always in minutes. It rejects NaN and warns if predictions still look like logs.

Use `c_models.time_workers(...)` for speedup tables (warm-up + 3 repeats, checks that outputs match).
`c_parallel.train_gd` and `c_parallel.run_search` already cover two task-parallel patterns, so D's search is an extra, not a requirement.

## Things D and A should know
- **Test is harder than validation.** After the 25 Jan snowstorm, 27–30 Jan trips average 21.1 min; no training day is above 17.8. Every model gets worse on test. Do not move the dates after seeing this; report it as a limitation.
- **`day_of_month`:** train only has days 1–21, so it cannot generalise to 22–31. C drops it; D should check trees without it.
- **Decision for B (affects C and D): the smoothing m = 20 in `FeatureBuilder` looks too strong.** Validation MAE on the full month, only `smoothing` changed:

  | smoothing m | 20 (now) | 10 | 5 | 2 | 1 | 0.5 |
  |---|---|---|---|---|---|---|
  | Linear/ridge on encodings | 4.152 | 4.081 | 4.009 | 3.928 | 3.884 | 3.856 |
  | Linear + interactions | 4.022 | 3.973 | 3.917 | 3.846 | 3.808 | 3.790 |

  Part C's results use B's current default (20). If B changes the default, C's notebook picks it up on the next run and D must use the same features. Choose the value on validation, not test, and add the sweep to B's notebook so the poster can be traced.
- **B's poster numbers come from the 100k sample** and should be regenerated on the full month.
- **Speedup depends on size.** On 100k rows process start-up dominates and nothing gets faster. B's encoding tasks are still slower in parallel on the full month (0.75× with 4 workers, outputs identical): each job ships large arrays for a very short computation. Say so on the poster; the search and the gradient in Part C do speed up.
- `pyarrow` and `scikit-learn` must be installed (`requirements.txt`).

## Run on Rockhill
Put `Group20_ETA_A`, `_B`, `_C` side by side, activate the group environment, then:
```bash
pip install -r requirements.txt                                            # once
cd Group20_ETA_C
bash run_rockhill_c.sh                                                     # A's 100k sample (quick check)
ETA_DATA=../Group20_ETA_A/outputs/<A_rockhill_run> bash run_rockhill_c.sh  # full month (poster numbers)
```
The full run needs about 6 GB RAM and 1.5 GB of temporary disk (`ETA_CACHE=/tmp/eta_cache` moves it; it is deleted at the end).

## Rubric map
| Criterion | Where in Part C |
|---|---|
| 1 time split, leakage | C1, C4 |
| 2 sequential vs parallel, Amdahl, overheads | C5, C7.1 |
| 3 data parallelism, sharding, merge check | C7 |
| 4 task parallelism: hyperparameter search + gradient | C5, C7 |
| 5 non-determinism: 3 sources, fixes, limits | C8 |

**Status:** local development run, not Rockhill evidence. Re-run before the poster is final.

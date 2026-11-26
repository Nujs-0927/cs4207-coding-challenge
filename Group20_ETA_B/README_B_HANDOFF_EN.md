# Group 20 — Part B (EDA + Feature Engineering)

## Files
| File | Purpose |
|---|---|
| `B_EDA_Features.ipynb` | **Final Part B notebook** (self-contained, executed). C can merge it into the group notebook as-is. |
| `b_features.py` | Data loader, feature builder, task-parallel encoding, benchmark |
| `tests/test_b_features.py` | 8 tests: leakage guard, out-of-fold, unseen zones, parallel == serial, loader |
| `run_rockhill_b.sh` | Runs tests + notebook on Rockhill (`cs4207` partition), then regenerates the poster/speech |
| `make_b_materials.py` | Builds `B_Poster_Section_EN.html` + `B_Speaking_EN.txt` from a run's `summary.json` |
| `outputs/<run>/features_{train,validation,test}.parquet` | Feature matrices for C/D |
| `outputs/<run>/summary.json`, `figures/` | All numbers and figures used on the poster |

## For C/D: use the features
```python
import pandas as pd, numpy as np
tr = pd.read_parquet('outputs/local_sample/features_train.parquet')
X_tr = tr.drop(columns=['duration_minutes', 'pickup_datetime'])   # 31 numeric features
y_tr = np.log1p(tr['duration_minutes'])                            # predict with expm1(...)
```
Tree models can use all features directly. For linear models, standardise with train statistics only.
`*_count` columns tell the model how reliable an encoding is.

## Run on Rockhill (Lab PC)
Put `Group20_ETA_A` and `Group20_ETA_B` side by side (GitHub clone), activate the group environment, then:
```bash
pip install -r requirements.txt          # once, if nbconvert/ipykernel are missing
cd Group20_ETA_B
bash run_rockhill_b.sh                                                   # A's 100k sample
ETA_DATA=../Group20_ETA_A/outputs/<A_rockhill_run> bash run_rockhill_b.sh  # full data from A's cleaned_parts
```
Outputs: `outputs/rockhill_<time>/` (features, figures, summary), the executed notebook `B_EDA_Features_rockhill_<time>.ipynb`, and a refreshed `B_Poster_Section_EN.html` / `B_Speaking_EN.txt`.
Other settings: `ETA_WORKERS="1 2 4"`, `ETA_VAL_START`, `ETA_TEST_START` (must match C's split).

## Team decisions this depends on
- **Trip distance excluded**: realised distance is known only after the trip ends (leakage at pickup).
- **Split**: validation ≥ 2026-01-22, test ≥ 2026-01-27. If C changes the dates, rerun with `ETA_VAL_START` / `ETA_TEST_START`.
- Results included in this package are a **local development run** on A's sample. The assessed numbers come from the Rockhill run.

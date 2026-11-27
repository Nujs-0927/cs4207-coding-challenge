"""Writes D_Advanced_Models.ipynb (unexecuted). Execute it with run_local_d.sh or run_rockhill_d.sh."""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md('''
# Group 20 — Part D: Advanced models, results and conclusion
**Task:** predict taxi trip duration (minutes) at pickup — regression.
**Input:** Member B's feature matrices (built on Member A's cleaned NYC TLC Yellow Taxi data).
**Scoring:** Member C's shared `evaluate()` and scoreboard, so every number is comparable with C's baselines.

| Setting | Meaning |
|---|---|
| `ETA_B_RUN` | folder with B's `features_{train,validation,test}.parquet` |
| `ETA_C_RUN` | folder with C's `C_scoreboard.csv` for the **same** data |
| `ETA_THREADS` | threads per model (fixed, recorded) |
| `ETA_RUN` | output folder under `outputs/` |

On Rockhill: `bash run_rockhill_d.sh` (uses the `cs4207` partition).
''')

code('''
import os, sys, json, time, platform, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
warnings.filterwarnings('ignore', category=UserWarning)

B_DIR = Path(os.environ.get('ETA_B_DIR', '../Group20_ETA_B'))
C_DIR = Path(os.environ.get('ETA_C_DIR', '../Group20_ETA_C'))
B_RUN = B_DIR / 'outputs' / os.environ.get('ETA_B_RUN', 'local_sample')
C_RUN = C_DIR / 'outputs' / os.environ.get('ETA_C_RUN', 'local_sample')
RUN = os.environ.get('ETA_RUN', 'local_sample')
OUT = Path('outputs') / RUN; FIG = OUT / 'figures'; FIG.mkdir(parents=True, exist_ok=True)

sys.path[:0] = ['.', str(C_DIR)]
import d_models as dm, d_figures as dfig
from c_models import evaluate, to_minutes, Scoreboard          # C's shared scoring
THREADS = dm.threads()
T_START = time.perf_counter()

import sklearn, lightgbm, xgboost
summary = {'run': RUN, 'features_from': str(B_RUN), 'baselines_from': str(C_RUN),
           'environment': {'host': platform.node(), 'platform': platform.platform(), 'python': platform.python_version(),
                           'cpus_visible': os.cpu_count(), 'slurm_cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
                           'threads_per_model': THREADS, 'gpu': 'not used',
                           'pandas': pd.__version__, 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                           'lightgbm': lightgbm.__version__, 'xgboost': xgboost.__version__},
           'config': {'seed': dm.SEED, 'grids': dm.GRIDS, 'rf_trees': dm.RF_TREES, 'rf_max_samples': dm.RF_MAX_SAMPLES,
                      'boost_max_rounds': dm.BOOST_ROUNDS, 'early_stopping_rounds': dm.EARLY_STOP,
                      'learning_rate': dm.LEARNING_RATE, 'not_features': dm.NOT_FEATURES, 'categorical': dm.CATEGORICAL}}
def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=180, bbox_inches='tight'); plt.close(fig); display(Image(FIG / f'{name}.png'))
summary['environment']
''')

md('''
## 1. Data and inputs
B's features only use information available at pickup (no distance, fare, tip, payment or dropoff time).
D additionally drops `day_of_month`: training covers days 1–21 and test covers 27–31, so a tree would treat every test day as day 21.
B's zone codes come from a fixed vocabulary, so the boosters use them as **categorical** features.
''')

code('''
data = dm.load_features(B_RUN)
X_example, _ = dm.split_xy(data['train'])
summary['rows'] = {s: len(data[s]) for s in dm.SPLITS}
summary['n_features'] = X_example.shape[1]
summary['features'] = list(X_example.columns)
print({s: f"{len(data[s]):,} trips, {data[s].pickup_datetime.min():%d %b} to {data[s].pickup_datetime.max():%d %b}" for s in dm.SPLITS})
print(f'{X_example.shape[1]} model inputs; dropped:', list(dm.NOT_FEATURES))
''')

md('''
## 2. Hyperparameter tuning on validation
Each model is trained on `log1p(duration)` (right-skewed target). For every grid point we fit on **train** and score on **validation** (MAE in minutes).
The boosters use early stopping on the validation loss; the test period is not touched until section 3.
''')

code('''
trials, chosen = [], {}
for name in ['Random Forest', 'LightGBM', 'XGBoost']:
    t0 = time.perf_counter()
    table, chosen[name] = dm.tune(name, data, evaluate, to_minutes, THREADS)
    trials.append(table)
    print(f'{name}: {len(table)} configurations in {time.perf_counter() - t0:.0f} s; best {chosen[name]["params"]}')
trials = pd.concat(trials, ignore_index=True)
trials.to_csv(OUT / 'd_tuning_trials.csv', index=False)
summary['tuning'] = trials.round(4).to_dict('records')
trials.round(3)
''')

md('''
## 3. Chosen models on validation and test
Each chosen configuration is scored once on test. The rows are added to **C's scoreboard for the same data**, so baselines, linear models and tree models share one table.
''')

code('''
board = Scoreboard.load(C_RUN / 'C_scoreboard.csv')
c_n = {r['split']: r['n'] for r in board.rows}
assert c_n['validation'] == len(data['validation']) and c_n['test'] == len(data['test']), 'C scoreboard is from different data'
preds = {}
for name, ch in chosen.items():
    M = dm.matrices(data, ch['categorical'])
    for split in ('validation', 'test'):
        X, y = M[split]
        p = dm.predict_minutes(ch['model'], X, to_minutes)
        preds[(name, split)] = p
        board.add(name, split, y, p, owner='D', group='advanced')
board.save(OUT / 'scoreboard_all.csv')
table = board.table()
best = table.loc[list(chosen), 'validation_MAE'].idxmin()          # chosen on VALIDATION
summary['scoreboard'] = table.round(4).reset_index().to_dict('records')
summary['best_model'] = {'name': best, 'params': chosen[best]['params'],
                         **{k: float(v) for k, v in table.loc[best].items()}}
table.round(2)
''')

code('''
fig = dfig.model_comparison(table, highlight=list(chosen))
save(fig, 'D_model_comparison')
lin = [m for m in table.index if m not in chosen and not m.startswith(('Constant', 'Same', 'Zone'))]
ref = {'constant': 'Constant (train median)', 'zone_pair': 'Zone-pair median',
       'best_linear': table.loc[lin, 'validation_MAE'].idxmin()}
gain = {}
for k, m in ref.items():
    for s in ('validation', 'test'):
        gain[f'{s}_vs_{k}_pct'] = round(100 * (1 - table.loc[best, f'{s}_MAE'] / table.loc[m, f'{s}_MAE']), 1)
summary['reference_models'] = ref; summary['improvement'] = gain
gain
''')

md('''
## 4. What the best model uses
Gain importance: the share of the total loss reduction from splits on each feature.
''')

code('''
X_cols = dm.matrices(data, chosen[best]['categorical'])['train'][0].columns
imp = dm.importance(best, chosen[best]['model'], X_cols)
imp.to_csv(OUT / 'd_feature_importance.csv', header=['gain_share'])
save(dfig.feature_importance(imp, f'{best}: top features by gain'), 'D_feature_importance')
zone = imp[[c for c in imp.index if c.startswith('te_') or 'zone' in c or 'airport' in c]].sum()
summary['importance_top'] = imp.head(8).round(4).to_dict()
summary['importance_location_share'] = round(float(zone), 3)
print(f'Location-based features (zone encodings, zone codes, airport/same-zone flags): {100*zone:.1f}% of total gain')
''')

md('''
## 5. Errors: predicted vs actual, trip length and day
''')

code('''
y_te = data['test']['duration_minutes'].to_numpy(); p_te = preds[(best, 'test')]
save(dfig.predicted_vs_actual(y_te, p_te, f'{best}: predicted vs actual (test)'), 'D_pred_vs_actual')
buckets = dm.error_by_bucket(y_te, p_te)
summary['test_error_by_duration'] = buckets.reset_index().to_dict('records')
daily = {}
for split in ('validation', 'test'):
    d = data[split].assign(p=preds[(best, split)])
    for day, g in d.groupby(d.pickup_datetime.dt.date):
        daily[str(day)] = round(float(np.abs(g.p - g.duration_minutes).mean()), 2)
summary['daily_MAE_best'] = daily
display(buckets); daily
''')

md('''
## 6. Non-determinism in the tree models
**Found in Part D:** with a fixed seed, two refits of the Random Forest gave test predictions that differed in the last digits.
Training is reproducible (each tree has its own seed), but `predict()` adds the 300 tree outputs from several threads
**in whatever order the threads finish**. Floating-point addition is not associative, so the sum changes from call to call.
**Fix:** predict with one thread (fixed summation order); training stays multi-threaded.
''')

code('''
rf = chosen['Random Forest']['model']
X_rf = dm.matrices(data, False)['test'][0]
rf.set_params(n_jobs=THREADS)
order = dm.prediction_order_experiment(rf, X_rf, to_minutes, repeats=4)
summary['nondeterminism_rf_prediction'] = order.to_dict('records')
order
''')

md('''
**Reproducibility check** for every chosen model: refit twice with the fixed seed, once with a single thread, and twice with `seed=None`.
With the fix, the fixed-seed fingerprints must be identical. `seed=None` shows how much the seed matters
(scikit-learn then draws a fresh seed each time; LightGBM and XGBoost fall back to their own default seed).
''')

code('''
repro = {}
for name in chosen:
    r = dm.reproducibility_check(name, chosen[name]['params'], data, to_minutes, THREADS)
    repro[name] = r.to_dict('records')
    print(name); display(r)
summary['reproducibility'] = repro
''')

md('''
## 7. Summary of Part D
''')

code('''
summary['runtime_s'] = round(time.perf_counter() - T_START, 1)
(OUT / 'd_summary.json').write_text(json.dumps(summary, indent=2, default=str))
b, g = summary['best_model'], summary['improvement']
r = repro[best]
same_seed = all(v[0]['fingerprint'] == v[1]['fingerprint'] == v[2]['fingerprint'] for v in repro.values())
o = summary['nondeterminism_rf_prediction']
display(Markdown(f"""
- **Best model (chosen on validation):** {best} {b['params']} — validation MAE **{b['validation_MAE']:.2f} min**, test MAE **{b['test_MAE']:.2f} min** (R² {b['validation_R2']:.2f} / {b['test_R2']:.2f}).
- **Against C:** {g['validation_vs_best_linear_pct']}% lower validation MAE than the best linear model, {g['validation_vs_zone_pair_pct']}% lower than the zone-pair baseline, {g['validation_vs_constant_pct']}% lower than a constant.
- **Location dominates:** {100*summary['importance_location_share']:.0f}% of the split gain comes from zone-based features.
- **Test is harder:** test bias {b['test_bias']:.2f} min — every model under-predicts after the 25 Jan snowstorm (daily MAE up to {max(v for k, v in summary['daily_MAE_best'].items() if k >= '2026-01-27'):.1f} min).
- **Non-determinism found:** Random Forest parallel prediction gave {o[0]['distinct_outputs']} different outputs in {o[0]['calls']} calls (up to {o[0]['max_abs_diff_min']:.1e} min); with one prediction thread {o[1]['distinct_outputs']} of {o[1]['calls']}.
- **Reproducible:** fixed seed → {'identical' if same_seed else 'DIFFERENT'} test predictions for all three models across reruns and thread counts.
- Run time {summary['runtime_s']:.0f} s on {THREADS} threads; outputs in `{OUT}`.
"""))
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
Path(__file__).with_name('D_Advanced_Models.ipynb').write_text(nbf.writes(nb), encoding='utf-8')
print('Wrote D_Advanced_Models.ipynb')

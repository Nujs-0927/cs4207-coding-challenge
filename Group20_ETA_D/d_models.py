"""Group 20 - Member D: advanced models (Random Forest, LightGBM, XGBoost) for
taxi trip-duration (ETA) regression.

Inputs  : B's feature matrices  outputs/<run>/features_{train,validation,test}.parquet
Scoring : C's shared evaluate() / Scoreboard (minutes; log models converted with expm1)

Rules the whole group follows
- Fit on TRAIN only. Hyperparameters and early stopping use VALIDATION only.
- TEST is scored once, for the configuration chosen on validation.
- Every model is trained on log1p(duration) and converted back with C's to_minutes().
- Seeds and thread counts are fixed, so a rerun gives bit-identical predictions.
"""
from __future__ import annotations

import hashlib
import itertools
import os
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', message='.*eval_set.*')   # LightGBM 4.7 API rename notice

TARGET = 'duration_minutes'
SEED = 420720
SPLITS = ('train', 'validation', 'test')

# Columns that are not model inputs, and why
NOT_FEATURES = {
    TARGET: 'regression target',
    'pickup_datetime': 'only used for ordering and plots; calendar features already derived by B',
    'day_of_month': 'train covers days 1-21, test days 27-31: a tree would treat every test day as day 21',
}
# B's zone codes come from a FIXED vocabulary (stable across runs) -> safe as categories
CATEGORICAL = ['pickup_zone_code', 'dropoff_zone_code']

# Small grids: tuned on validation, chosen by validation MAE (minutes)
GRIDS = {
    'Random Forest': {'min_samples_leaf': [5, 20], 'max_features': [0.33, 0.6]},
    'LightGBM': {'num_leaves': [31, 127], 'min_child_samples': [20, 100]},
    'XGBoost': {'max_depth': [6, 10], 'min_child_weight': [1, 20]},
}
RF_TREES = 300
RF_MAX_SAMPLES = 0.25          # fraction of training rows per tree (bootstrap); same on sample and full data
BOOST_ROUNDS, EARLY_STOP, LEARNING_RATE = 3000, 100, 0.05


def threads() -> int:
    """Fixed thread count from ETA_THREADS (default: CPUs given by Slurm, else 4)."""
    n = os.environ.get('ETA_THREADS') or os.environ.get('SLURM_CPUS_PER_TASK') or '4'
    return max(1, int(n))


# --------------------------------------------------------------------- data
def load_features(b_run_dir) -> dict:
    """B's three feature files; checks they share one column list and do not overlap in time."""
    d = Path(b_run_dir)
    data = {s: pd.read_parquet(d / f'features_{s}.parquet') for s in SPLITS}
    cols = [list(df.columns) for df in data.values()]
    if not all(c == cols[0] for c in cols):
        raise ValueError('train/validation/test feature columns differ')
    t = {s: data[s]['pickup_datetime'] for s in SPLITS}
    if not (t['train'].max() < t['validation'].min() and t['validation'].max() < t['test'].min()):
        raise ValueError('splits are not in time order')
    return data


def split_xy(df: pd.DataFrame, categorical=True):
    X = df.drop(columns=[c for c in NOT_FEATURES if c in df.columns])
    leaky = {'trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'dropoff_datetime',
             'rate_code', 'payment_type', 'row_id'} & set(X.columns)
    if leaky:
        raise ValueError(f'leaky columns in X: {sorted(leaky)}')
    if categorical:
        for c in CATEGORICAL:
            if c in X:
                # fixed category list -> identical integer codes in every run and split
                X[c] = pd.Categorical(X[c].astype('int64'), categories=range(0, 267))
    return X, df[TARGET].to_numpy(dtype='float64')


# ------------------------------------------------------------------- models
def make_model(name: str, params: dict, n_threads: int, seed=SEED):
    if name == 'Random Forest':
        from sklearn.ensemble import RandomForestRegressor
        # each tree sees a bootstrap of at most RF_MAX_SAMPLES rows: keeps the full month tractable
        return RandomForestRegressor(n_estimators=RF_TREES, n_jobs=n_threads, random_state=seed,
                                     max_samples=RF_MAX_SAMPLES, **params)
    if name == 'LightGBM':
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=BOOST_ROUNDS, learning_rate=LEARNING_RATE, subsample=1.0,
                                 colsample_bytree=0.8, n_jobs=n_threads, random_state=seed,
                                 deterministic=True, force_row_wise=True, verbose=-1, **params)
    if name == 'XGBoost':
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=BOOST_ROUNDS, learning_rate=LEARNING_RATE, tree_method='hist',
                                enable_categorical=True, max_cat_to_onehot=1, colsample_bytree=0.8,
                                early_stopping_rounds=EARLY_STOP, n_jobs=n_threads, random_state=seed, **params)
    raise ValueError(name)


def fit(name, params, Xtr, ytr, Xva, yva, n_threads, seed=SEED):
    """Fit on train (log target). Boosters stop early on VALIDATION loss, never test."""
    model = make_model(name, params, n_threads, seed)
    t, tv = np.log1p(ytr), np.log1p(yva)
    t0 = time.perf_counter()
    if name == 'LightGBM':
        import lightgbm as lgb
        model.fit(Xtr, t, eval_set=[(Xva, tv)], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    elif name == 'XGBoost':
        model.fit(Xtr, t, eval_set=[(Xva, tv)], verbose=False)
    else:
        model.fit(Xtr, t)
    return model, time.perf_counter() - t0


def best_rounds(name, model):
    if name == 'LightGBM':
        return int(model.best_iteration_)
    if name == 'XGBoost':
        return int(model.best_iteration) + 1
    return RF_TREES


def predict_minutes(model, X, to_minutes, deterministic=True):
    """sklearn's forest adds the per-tree predictions from several threads in whatever
    order the threads finish, so the last bits change from call to call. A single
    prediction thread fixes the summation order (training stays multi-threaded)."""
    if deterministic and hasattr(model, 'estimators_'):
        n = model.n_jobs
        model.set_params(n_jobs=1)
        try:
            return to_minutes(model.predict(X))
        finally:
            model.set_params(n_jobs=n)
    return to_minutes(model.predict(X))


def grid(name):
    keys = list(GRIDS[name])
    return [dict(zip(keys, v)) for v in itertools.product(*(GRIDS[name][k] for k in keys))]


def tune(name, data, evaluate, to_minutes, n_threads):
    """Every configuration of the grid, scored on validation. Returns (trials, best model)."""
    cat = name != 'Random Forest'           # RF in sklearn has no categorical splits: codes stay numeric
    Xtr, ytr = split_xy(data['train'], categorical=cat)
    Xva, yva = split_xy(data['validation'], categorical=cat)
    if not cat:
        Xtr, Xva = Xtr.astype('float64'), Xva.astype('float64')
    rows, best = [], None
    for params in grid(name):
        model, sec = fit(name, params, Xtr, ytr, Xva, yva, n_threads)
        m = evaluate(yva, predict_minutes(model, Xva, to_minutes))
        rows.append({'model': name, **params, 'rounds': best_rounds(name, model),
                     'fit_s': round(sec, 2), **{k: m[k] for k in ('MAE', 'RMSE', 'R2', 'bias')}})
        if best is None or (m['MAE'], len(rows)) < (best[0], best[1]):
            best = (m['MAE'], len(rows), model, params)
    return pd.DataFrame(rows), {'model': best[2], 'params': best[3], 'categorical': cat}


def matrices(data, categorical):
    out = {}
    for s in SPLITS:
        X, y = split_xy(data[s], categorical=categorical)
        out[s] = (X if categorical else X.astype('float64'), y)
    return out


# --------------------------------------------------------- reproducibility
def fingerprint(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype='float64')).tobytes()).hexdigest()[:12]


def prediction_order_experiment(model, X, to_minutes, repeats=4):
    """Non-determinism found in Part D: the Random Forest's parallel prediction.
    Same fitted model, same input, repeated predict() calls."""
    rows = []
    for mode in ('parallel (threads finish in any order)', 'fixed order (1 prediction thread)'):
        prints, ref, diff = [], None, 0.0
        for _ in range(repeats):
            p = predict_minutes(model, X, to_minutes, deterministic=mode.startswith('fixed'))
            ref = p if ref is None else ref
            diff = max(diff, float(np.abs(p - ref).max()))
            prints.append(fingerprint(p))
        rows.append({'prediction': mode, 'calls': repeats, 'distinct_outputs': len(set(prints)),
                     'fingerprints': ' '.join(prints), 'max_abs_diff_min': diff})
    return pd.DataFrame(rows)


def reproducibility_check(name, params, data, to_minutes, n_threads):
    """Refit the chosen model twice with the fixed seed, once with 1 thread, and twice
    with no seed. Fingerprints of the test predictions show what changes the output."""
    cat = name != 'Random Forest'
    M = matrices(data, cat)
    (Xtr, ytr), (Xva, yva), (Xte, _) = M['train'], M['validation'], M['test']
    runs = [('seed fixed, run 1', n_threads, SEED), ('seed fixed, run 2', n_threads, SEED),
            ('seed fixed, 1 thread', 1, SEED), ('seed=None, run 1', n_threads, None), ('seed=None, run 2', n_threads, None)]
    rows, ref = [], None
    for label, nt, seed in runs:
        model, _ = fit(name, params, Xtr, ytr, Xva, yva, nt, seed=seed)
        p = predict_minutes(model, Xte, to_minutes)
        ref = p if ref is None else ref
        rows.append({'run': label, 'threads': nt, 'fingerprint': fingerprint(p),
                     'max_abs_diff_vs_run1_min': float(np.abs(p - ref).max())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- analysis
def importance(name, model, columns):
    if name == 'LightGBM':
        v = model.booster_.feature_importance(importance_type='gain')
    elif name == 'XGBoost':
        g = model.get_booster().get_score(importance_type='total_gain')
        v = np.array([g.get(c, 0.0) for c in columns])
    else:
        v = model.feature_importances_
    s = pd.Series(v, index=columns, dtype='float64')
    return (s / s.sum()).sort_values(ascending=False)


def error_by_bucket(y, p, edges=(0, 5, 10, 20, 40, 180)):
    b = pd.cut(pd.Series(y), list(edges), right=True, include_lowest=True)
    df = pd.DataFrame({'bucket': b.astype(str), 'abs_err': np.abs(p - y), 'err': p - y})
    g = df.groupby('bucket', sort=False).agg(trips=('err', 'size'), MAE=('abs_err', 'mean'), bias=('err', 'mean'))
    return g.loc[[str(x) for x in b.cat.categories if str(x) in g.index]].round(2)

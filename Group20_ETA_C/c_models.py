"""Group 20 - Member C: time-based split, shared evaluation and baseline models
for taxi trip-duration (ETA) regression.

Rules the whole group follows
- Train on earlier trips, choose on validation, score test once at the end.
- Everything learned here (lookup tables, constants) comes from TRAIN only.
- evaluate() is the one scoring function for C and D. It works in MINUTES, so a
  model trained on log1p(duration) must be converted back with np.expm1 first.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = 'duration_minutes'
VAL_START, TEST_START = '2026-01-22', '2026-01-27'
PRED_MIN, PRED_MAX = 1.0, 180.0        # A keeps trips of 1-180 minutes
SPLITS = ('train', 'validation', 'test')


# -------------------------------------------------------------------- 1. split
def check_split(train, val, test, total_rows=None):
    """Prove the split is chronological, disjoint and complete; describe each part.

    The cut itself is A's handoff.chronological_split(); C owns the two dates.
    21 training days are exactly three weeks, so every weekday is seen 3 times.
    """
    parts = dict(zip(SPLITS, (train, val, test)))
    if not train['pickup_datetime'].max() < val['pickup_datetime'].min():
        raise ValueError('train overlaps validation in time')
    if not val['pickup_datetime'].max() < test['pickup_datetime'].min():
        raise ValueError('validation overlaps test in time')
    ids = pd.concat([p['row_id'] for p in parts.values()])
    if not ids.is_unique:
        raise ValueError('a trip appears in more than one split')
    if total_rows is not None and len(ids) != total_rows:
        raise ValueError(f'split covers {len(ids)} of {total_rows} rows')
    rows = []
    for name, p in parts.items():
        t = p['pickup_datetime']
        rows.append({'split': name, 'rows': len(p), 'share_%': round(100 * len(p) / len(ids), 1),
                     'days': t.dt.normalize().nunique(), 'first_pickup': t.min(), 'last_pickup': t.max(),
                     'mean_min': round(p[TARGET].mean(), 2), 'median_min': round(p[TARGET].median(), 2)})
    return pd.DataFrame(rows).set_index('split')


def boundary_trips(part, boundary):
    """Trips picked up before `boundary` that end after it (label not yet known then)."""
    end = part['pickup_datetime'] + pd.to_timedelta(part[TARGET], unit='m')
    return int((end >= pd.Timestamp(boundary)).sum())


# --------------------------------------------------------- 2. shared evaluation
def evaluate(y_true, y_pred):
    """MAE, RMSE and R2 in minutes. bias > 0 means we predict too long a trip."""
    y = np.asarray(y_true, dtype='float64').ravel()
    p = np.asarray(y_pred, dtype='float64').ravel()
    if y.shape != p.shape:
        raise ValueError(f'{len(y)} targets but {len(p)} predictions')
    if y.size == 0:
        raise ValueError('nothing to evaluate')
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError('NaN or infinite value in targets or predictions')
    if p.max() < np.log1p(PRED_MAX + 1) and np.median(y) > np.log1p(PRED_MAX + 1):
        warnings.warn('Predictions look like log1p(minutes): apply np.expm1 before evaluate()')
    err = p - y
    sst = np.square(y - y.mean()).sum()
    return {'n': int(y.size),
            'MAE': float(np.abs(err).mean()),
            'RMSE': float(np.sqrt(np.square(err).mean())),
            'R2': float(1 - np.square(err).sum() / sst) if sst > 0 else float('nan'),
            'bias': float(err.mean())}


def to_minutes(pred_log):
    """log1p(minutes) -> minutes, clipped to the range A's cleaning allows."""
    return np.clip(np.expm1(pred_log), PRED_MIN, PRED_MAX)


class Scoreboard:
    """One results table for every model. D adds rows to the same object or file."""

    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def add(self, model, split, y_true, y_pred, owner='C', group='model'):
        if split not in SPLITS:
            raise ValueError(f'split must be one of {SPLITS}')
        self.rows = [r for r in self.rows if (r['model'], r['split']) != (model, split)]
        row = {'owner': owner, 'group': group, 'model': model, 'split': split, **evaluate(y_true, y_pred)}
        self.rows.append(row)
        return row

    def table(self):
        """One line per model, validation and test side by side, in the order added."""
        df = pd.DataFrame(self.rows)
        order = list(dict.fromkeys(df['model']))
        wide = df.pivot(index='model', columns='split', values=['MAE', 'RMSE', 'R2', 'bias'])
        wide.columns = [f'{s}_{m}' for m, s in wide.columns]
        cols = [f'{s}_{m}' for s in SPLITS for m in ('MAE', 'RMSE', 'R2', 'bias') if f'{s}_{m}' in wide]
        return wide.loc[order, cols]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(path, index=False)

    @classmethod
    def load(cls, path):
        return cls(pd.read_csv(path).to_dict('records'))


# ------------------------------------------------------------------ 3. baselines
def with_calendar(X):
    t = pd.to_datetime(X['pickup_datetime'])
    return X.assign(weekday=t.dt.day_of_week, hour=t.dt.hour)


class ConstantBaseline:
    """Same answer for every trip: the median (best constant for MAE) or mean."""

    def __init__(self, stat='median'):
        self.stat = stat

    def fit(self, X, y):
        self.value_ = float(getattr(np, self.stat)(np.asarray(y, dtype='float64')))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class LookupBaseline:
    """Average past duration of similar trips. A key with fewer than `min_count`
    training trips backs off to the next, coarser key, and finally to a constant."""

    def __init__(self, levels, stat='median', min_count=1):
        self.levels, self.stat, self.min_count = levels, stat, min_count

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype='float64'), index=X.index)
        self.constant_ = float(getattr(y, self.stat)())
        self.tables_ = []
        for keys in self.levels:
            g = y.groupby([X[k] for k in keys]).agg([self.stat, 'count'])
            self.tables_.append(g.loc[g['count'] >= self.min_count, self.stat])
        return self

    def predict(self, X):
        out = pd.Series(np.nan, index=range(len(X)))
        for keys, table in zip(self.levels, self.tables_):
            if len(keys) == 1:
                idx = pd.Index(X[keys[0]])
            else:
                idx = pd.MultiIndex.from_frame(X[keys].reset_index(drop=True))
            found = pd.Series(table.reindex(idx).to_numpy(dtype='float64'))
            out = out.fillna(found)
        return out.fillna(self.constant_).to_numpy()


def make_baselines():
    """From naive to strong. Each one only needs A's pickup-time columns."""
    return {
        'Constant (train median)': ConstantBaseline('median'),
        'Constant (train mean)': ConstantBaseline('mean'),
        'Same weekday & hour': LookupBaseline([['weekday', 'hour']]),
        'Zone-pair median': LookupBaseline([['pickup_zone', 'dropoff_zone'], ['pickup_zone']]),
    }


# ------------------------------------------------------ 4. timing (criterion 2)
def time_workers(run, check, workers=(1, 2, 4), repeats=3):
    """Same workload for every worker count: one warm-up, then `repeats` timed runs.

    run(w) returns a dict with 'total', 'compute' and 'startup' seconds; check(out)
    returns a fingerprint. Raises if any run gives a different output.
    """
    rows, prints, last = [], set(), {}
    for w in workers:
        run(w)                                                   # warm-up, not timed
        for r in range(repeats):
            out = run(w)
            prints.add(check(out))
            last[w] = out
            rows.append({'workers': w, 'repeat': r, 'total': out['total'],
                         'compute': out['compute'], 'startup': out['startup']})
    if len(prints) != 1:
        raise RuntimeError(f'outputs differ between runs or worker counts: {sorted(prints)}')
    raw = pd.DataFrame(rows)
    t = raw.groupby('workers', as_index=False).agg(mean_s=('total', 'mean'), std_s=('total', 'std'),
                                                   compute_s=('compute', 'mean'), startup_s=('startup', 'mean'))
    t['speedup'] = t['mean_s'].iloc[0] / t['mean_s']
    t['efficiency'] = t['speedup'] / t['workers']
    t['speedup_no_startup'] = t['compute_s'].iloc[0] / t['compute_s']
    t['outputs_equal'] = True
    return t, last, prints.pop()

"""Group 20 - Member B: data loading, feature engineering and task-parallel encoding
for taxi trip-duration (ETA) regression.

Input contract comes from Member A (Group20_ETA_A/handoff.py): only pickup-time
information - pickup_datetime, pickup_zone, dropoff_zone*, passenger_count,
passenger_count_missing. (*dropoff_zone stands in for the intended destination,
assumed known at pickup.)

Design rules
- Stateless features (calendar, zone flags) need no fitting.
- Learned features (target encodings, counts, passenger imputation) are fitted on
  the TRAINING split only. Training rows receive out-of-fold (OOF) encodings so a
  row never sees its own label; validation/test rows use encodings fitted on the
  whole training split.
- Encodings are computed on log1p(duration) because the target is right-skewed.
- Every (fold, table) encoding is an independent task -> task parallelism with a
  process pool. Keys are integer-coded once in the parent so each task ships only
  small integer arrays. A fixed fold seed and order-preserving map make parallel
  output bit-identical to serial output.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd

TARGET = 'duration_minutes'
FORBIDDEN_INPUTS = {
    TARGET, 'row_id', 'source_file', 'rate_code', 'vendor_id',
    'dropoff_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'fare_amount',
    'tip_amount', 'tolls_amount', 'total_amount', 'payment_type', 'speed_mph',
}

# TLC zone IDs of the three airports (taxi_zone_lookup.csv)
AIRPORT_ZONES = {'1': 'EWR', '132': 'JFK', '138': 'LGA'}
# US federal holidays in the data window (extend if more months are used)
HOLIDAYS = pd.to_datetime(['2026-01-01', '2026-01-19', '2026-02-16'])

# Our own time-band definition (the brief asks groups to define peak times)
TIME_BANDS = [  # (name, start hour inclusive, end hour exclusive)
    ('night', 0, 6),
    ('morning_peak', 6, 10),
    ('midday', 10, 16),
    ('afternoon_peak', 16, 20),
    ('evening', 20, 24),
]

# Fixed, sorted category vocabulary for zone IDs (TLC zones 1-265 + unknowns).
ZONE_VOCAB = [str(i) for i in range(1, 266)] + ['__MISSING__', '__OTHER__']


def zone_codes(zones: pd.Series) -> np.ndarray:
    """Category codes from the FIXED sorted vocabulary: identical in every run/process."""
    idx = pd.Index(ZONE_VOCAB).get_indexer(zones.astype(str))
    return np.where(idx < 0, ZONE_VOCAB.index('__OTHER__'), idx).astype('int16')


def zone_codes_naive(zones: pd.Series) -> np.ndarray:
    """NON-DETERMINISTIC version kept for the demonstration: vocabulary taken from a
    Python set, whose iteration order depends on per-process string hashing
    (PYTHONHASHSEED), so the same zone gets a different code in every process."""
    vocab = {z: i for i, z in enumerate(set(zones.astype(str)))}
    return zones.astype(str).map(vocab).to_numpy()


ENCODED_KEYS = {  # learned table -> key columns
    'te_pickup': ['pickup_zone'],
    'te_dropoff': ['dropoff_zone'],
    'te_pair': ['pickup_zone', 'dropoff_zone'],
    'te_pair_band': ['pickup_zone', 'dropoff_zone', 'time_band'],
    'te_pickup_hour': ['pickup_zone', 'hour'],
}


# --------------------------------------------------------------------------- data
def load_cleaned(source) -> pd.DataFrame:
    """Load A's output: a single sample Parquet file OR an A run directory that
    contains cleaned_parts/ (full month on Rockhill). Sorted by pickup time."""
    p = Path(source)
    if p.is_dir():
        parts = sorted((p / 'cleaned_parts').glob('part-*.parquet')) or sorted(p.glob('part-*.parquet'))
        if not parts:
            raise FileNotFoundError(f'No cleaned_parts/part-*.parquet under {p}')
        df = pd.concat((pd.read_parquet(f) for f in parts), ignore_index=True)
    else:
        df = pd.read_parquet(p)
    if not df['row_id'].is_unique:
        raise ValueError('Non-unique row IDs')
    return df.sort_values(['pickup_datetime', 'row_id'], kind='stable').reset_index(drop=True)


def time_band(hour: pd.Series) -> pd.Series:
    out = pd.Series('night', index=hour.index, dtype='object')
    for name, lo, hi in TIME_BANDS:
        out[(hour >= lo) & (hour < hi)] = name
    return out


def check_inputs(X: pd.DataFrame) -> None:
    bad = FORBIDDEN_INPUTS.intersection(X.columns)
    if bad:
        raise ValueError(f'Leakage risk: forbidden columns in X: {sorted(bad)}')


def stateless_features(X: pd.DataFrame) -> pd.DataFrame:
    """Calendar and zone features computable for any single trip at pickup."""
    check_inputs(X)
    t = pd.to_datetime(X['pickup_datetime'])
    hour, dow = t.dt.hour, t.dt.dayofweek
    minute_of_day = hour * 60 + t.dt.minute
    pu = X['pickup_zone'].astype('string').fillna('__MISSING__')
    f = pd.DataFrame(index=X.index)
    f['hour'] = hour.astype('int16')
    f['day_of_week'] = dow.astype('int8')
    f['is_weekend'] = (dow >= 5).astype('int8')
    f['is_holiday'] = t.dt.normalize().isin(HOLIDAYS).astype('int8')
    f['day_of_month'] = t.dt.day.astype('int8')
    f['tod_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    f['tod_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
    f['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    f['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    f['time_band'] = time_band(hour)
    f['is_peak'] = f['time_band'].isin(['morning_peak', 'afternoon_peak']).astype('int8')
    f['pickup_zone'] = pu
    f['pickup_zone_code'] = zone_codes(pu)
    f['pickup_airport'] = pu.isin(AIRPORT_ZONES).astype('int8')
    if 'dropoff_zone' in X.columns:
        do = X['dropoff_zone'].astype('string').fillna('__MISSING__')
        f['dropoff_zone'] = do
        f['dropoff_zone_code'] = zone_codes(do)
        f['dropoff_airport'] = do.isin(AIRPORT_ZONES).astype('int8')
        f['same_zone'] = (pu == do).astype('int8')
    f['passenger_count'] = X['passenger_count'].astype('float64')
    f['passenger_count_missing'] = X['passenger_count_missing'].astype('int8')
    return f


# ------------------------------------------------------------------ encoding core
def _key_strings(frame: pd.DataFrame, cols) -> np.ndarray:
    key = frame[cols[0]].astype(str)
    for c in cols[1:]:
        key = key + '|' + frame[c].astype(str)
    return key.to_numpy()


def _encode_task(args):
    """One independent task: fit one encoding table on the fitting rows and apply it.
    Inputs are integer key codes, so the task is cheap to send to a worker."""
    tag, fit_codes, fit_y, apply_codes, n_keys, m = args
    t0 = time.perf_counter()
    prior = float(fit_y.mean())                               # from fitting rows only
    sums = np.bincount(fit_codes, weights=fit_y, minlength=n_keys)
    counts = np.bincount(fit_codes, minlength=n_keys).astype('float64')
    s, n = sums[apply_codes], counts[apply_codes]             # unseen key -> 0, 0
    return tag, (s + m * prior) / (n + m), n, time.perf_counter() - t0


TASK_LOG = []   # (tag, seconds) of the most recent run, for load-imbalance analysis


def run_tasks(tasks, n_jobs):
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            out = list(pool.map(_encode_task, tasks))         # map keeps task order
    else:
        out = [_encode_task(t) for t in tasks]
    TASK_LOG[:] = [(tag, sec) for tag, _, _, sec in out]
    return [(tag, e, n) for tag, e, n, _ in out]


@dataclass
class FeatureBuilder:
    smoothing: float = 20.0   # m: shrink rare keys toward the prior
    n_folds: int = 5          # out-of-fold folds for training rows
    seed: int | None = 420720  # fixed -> deterministic folds; None = unseeded (demo only)
    n_jobs: int = 1           # >1 -> encoding tasks run in a process pool
    fitted_: dict = field(default_factory=dict, repr=False)

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        t0 = time.perf_counter()
        base = stateless_features(X_train)
        y_log = np.log1p(y_train.to_numpy(dtype='float64'))
        self.passenger_median_ = float(np.nanmedian(base['passenger_count']))
        self.names_ = [n for n in ENCODED_KEYS if all(c in base for c in ENCODED_KEYS[n])]
        # integer-code every key once; keep the vocabulary to code new data later
        self.vocab_, codes = {}, {}
        for name in self.names_:
            c, uniq = pd.factorize(_key_strings(base, ENCODED_KEYS[name]), sort=True)
            self.vocab_[name], codes[name] = pd.Index(uniq), c
        self.train_codes_, self.train_y_log_ = codes, y_log
        folds = np.random.default_rng(self.seed).permutation(np.arange(len(base)) % self.n_folds)
        tasks = []
        for k in range(self.n_folds):
            fit, app = folds != k, folds == k
            for name in self.names_:
                tasks.append(((k, name), codes[name][fit], y_log[fit], codes[name][app],
                              len(self.vocab_[name]), self.smoothing))
        t1 = time.perf_counter()
        enc = {tag: (e, n) for tag, e, n in run_tasks(tasks, self.n_jobs)}
        t2 = time.perf_counter()
        masks = [folds == k for k in range(self.n_folds)]
        self.fitted_ = {'n_train': len(base)}
        out = self._assemble(base, enc, masks)
        self.phase_seconds_ = {'serial_prepare': t1 - t0, 'parallel_tasks': t2 - t1,
                               'serial_assemble': time.perf_counter() - t2}
        self.task_seconds_ = list(TASK_LOG)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError('Call fit_transform on the TRAINING split first')
        base = stateless_features(X)
        tasks = []
        for name in self.names_:
            vocab = self.vocab_[name]
            code = vocab.get_indexer(_key_strings(base, ENCODED_KEYS[name]))
            code = np.where(code < 0, len(vocab), code)       # unseen key -> extra empty slot
            tasks.append((('all', name), self.train_codes_[name], self.train_y_log_, code,
                          len(vocab) + 1, self.smoothing))
        enc = {tag: (e, n) for tag, e, n in run_tasks(tasks, self.n_jobs)}
        return self._assemble(base, enc, None)

    def _assemble(self, base, enc, masks):
        out = base.drop(columns=['pickup_zone', 'dropoff_zone', 'time_band'], errors='ignore').copy()
        out['passenger_count'] = out['passenger_count'].fillna(self.passenger_median_)
        for name in self.names_:
            if masks is None:
                e, n = enc[('all', name)]
            else:
                e, n = np.empty(len(base)), np.empty(len(base))
                for k, mask in enumerate(masks):
                    e[mask], n[mask] = enc[(k, name)]
            out[name], out[f'{name}_count'] = e, np.log1p(n)
        if {'te_pair', 'te_pickup', 'te_dropoff'}.issubset(out.columns):
            n = np.expm1(out['te_pair_count'])
            w = n / (n + self.smoothing)            # rare pair -> lean on zone encodings
            out['te_pair_blend'] = w * out['te_pair'] + (1 - w) * 0.5 * (out['te_pickup'] + out['te_dropoff'])
        for name, _, _ in TIME_BANDS:
            out[f'band_{name}'] = (base['time_band'] == name).astype('int8')
        return out


# --------------------------------------------------------- determinism + timing
def fingerprint(df: pd.DataFrame) -> str:
    """Order-sensitive hash proving two feature matrices are identical."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]


def benchmark(X_train, y_train, X_eval, jobs=(1, 2, 4), repeats=3, **kw):
    """Same workload (fit_transform on train + transform on eval) per worker count.
    One warm-up per configuration; timing includes pool start-up and data transfer."""
    rows, prints = [], set()
    for j in jobs:
        FeatureBuilder(n_jobs=j, **kw).fit_transform(X_train, y_train)
        for r in range(repeats):
            t0 = time.perf_counter()
            fb = FeatureBuilder(n_jobs=j, **kw)
            a = fb.fit_transform(X_train, y_train)
            b = fb.transform(X_eval)
            rows.append({'workers': j, 'repeat': r, 'seconds': time.perf_counter() - t0})
            prints.add(fingerprint(a) + fingerprint(b))
    res = pd.DataFrame(rows).groupby('workers')['seconds'].agg(['mean', 'std']).reset_index()
    res['speedup'] = res['mean'].iloc[0] / res['mean']
    res['efficiency'] = res['speedup'] / res['workers']
    res['all_outputs_equal'] = len(prints) == 1
    return res

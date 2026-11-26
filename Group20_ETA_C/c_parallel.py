"""Group 20 - Member C: parallel ridge regression, hyperparameter search and
non-determinism experiments. Only numpy is imported, so worker processes start fast.

Two parallel patterns
1. Gradient computation (all-reduce style). Training rows are cut into a FIXED
   number of equal shards. Each iteration every shard returns its gradient SUM and
   row count; the parent adds them in shard order, divides by n and updates one
   shared weight vector.
2. Hyperparameter search. Each (feature set, target scale, history) trial fits on
   train and is scored on validation without talking to any other trial.

Large arrays are never pickled: they sit in .npy files that workers memory-map, so
all workers share one copy in the page cache. Every worker uses one BLAS thread,
otherwise "1 worker" would already be multi-threaded and the speedup meaningless.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

SEED = 420720
N_SHARDS = 8                 # fixed: does NOT depend on the number of workers
BLOCK = 4096                 # rows per block for the gradient (28 columns, fits in L2 cache)
FIT_BLOCK = 512              # rows per block for the wider fitting matrices (up to 115 columns)
ALPHAS = (0.0, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)      # 0 = plain linear regression
PRED_MIN, PRED_MAX = 1.0, 180.0

# B's columns that a LINEAR model should not see, and why
NOT_FOR_LINEAR = {
    'day_of_month': 'train covers days 1-21, so days 22-31 are pure extrapolation',
    'band_night': 'dummy trap: the five band columns always sum to 1',
    'is_peak': 'exactly band_morning_peak + band_afternoon_peak',
}
INTERACT_WITH = ('tod_sin', 'tod_cos', 'is_weekend', 'is_holiday', 'pickup_airport', 'dropoff_airport',
                 'same_zone', 'band_morning_peak', 'band_midday', 'band_afternoon_peak', 'band_evening')


# --------------------------------------------------------------- shared helpers
def fingerprint(*arrays):
    """Hash of the exact bytes: one flipped bit gives a different fingerprint."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def shard_bounds(n, k=N_SHARDS):
    """k contiguous shards whose sizes differ by at most one row."""
    edges = [n * i // k for i in range(k + 1)]
    return list(zip(edges[:-1], edges[1:]))


def save_arrays(folder, **arrays):
    close_store()                 # never overwrite a file this process still has mapped
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, a in arrays.items():
        np.save(folder / f'{name}.npy', np.ascontiguousarray(a))
        paths[name] = str(folder / f'{name}.npy')
    return paths


_STORE = {}


def open_store(paths):
    """Memory-map the shared arrays (also runs once in every worker)."""
    _STORE.clear()
    _STORE.update({k: np.asarray(np.load(p, mmap_mode='r')) for k, p in paths.items()})


def close_store():
    """Release the memory maps so the cache files can be deleted."""
    _STORE.clear()


def _ping(_):
    time.sleep(0.02)
    return os.getpid()


def make_pool(workers, paths):
    """Start a pool and wait until every worker is up. Returns (pool, start-up seconds)."""
    t0 = time.perf_counter()
    # 'spawn' behaves the same on Windows and Linux and is safe inside Jupyter
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                               initializer=open_store, initargs=(paths,))
    seen, rounds = set(), 0
    while len(seen) < workers and rounds < 200:
        seen |= set(pool.map(_ping, range(workers)))
        rounds += 1
    return pool, time.perf_counter() - t0


def amdahl_p(speedup, workers):
    """Parallel fraction p implied by one measured speedup (Amdahl's law solved for p)."""
    return (1 - 1 / speedup) / (1 - 1 / workers) if workers > 1 else float('nan')


def amdahl_fit(workers, speedups):
    """Least-squares p over all worker counts: 1/S = (1 - p) + p / W."""
    w, s = np.asarray(workers, float), np.asarray(speedups, float)
    x, y = 1 - 1 / w[w > 1], 1 - 1 / s[w > 1]
    return float((x @ y) / (x @ x))


# --------------------------------------------------------- feature sets for ridge
def feature_sets(columns):
    """name -> (column indices into B's feature matrix, interaction index pairs)."""
    columns = list(columns)
    cols = [c for c in columns if c not in NOT_FOR_LINEAR]
    plain = [c for c in cols if not c.startswith('te_')]
    te = [c for c in cols if c.startswith('te_') and not c.endswith('_count')]
    cal = [c for c in INTERACT_WITH if c in cols]
    pairs = [(a, b) for a in te for b in cal]
    pairs += [(a, b) for i, a in enumerate(te) for b in te[i:]]
    at = {c: i for i, c in enumerate(columns)}
    # pairs point into the SELECTED columns, so a block only has to be sliced once
    sel = {c: i for i, c in enumerate(cols)}
    idx = [at[c] for c in cols]
    return {'calendar': ([at[c] for c in plain], []),
            'encoded': (idx, []),
            'interactions': (idx, [(sel[a], sel[b]) for a, b in pairs])}


class BlockDesign:
    """Builds the model matrix for a few rows at a time in ONE reused buffer, so memory
    stays small and workers do not slow each other down with page faults. Columns are
    contiguous (Fortran order) because every operation below works column by column."""

    def __init__(self, idx, pairs, block=FIT_BLOCK):
        self.idx, self.pairs = list(idx), list(pairs)
        self.p, self.d, self.block = len(idx), len(idx) + len(pairs), block
        self.buf = np.empty((block, self.d), order='F')

    def __call__(self, X, s, mean=None, scale=None):
        raw = X[s:s + self.block]
        Z = self.buf[:len(raw)]
        Z[:, :self.p] = raw[:, self.idx]
        for j, (a, b) in enumerate(self.pairs, start=self.p):      # products of column pairs
            np.multiply(Z[:, a], Z[:, b], out=Z[:, j])
        if mean is not None:
            Z -= mean
            Z /= scale
        return Z


def design(X, idx, pairs):
    """Whole model matrix at once (small data and tests only)."""
    return BlockDesign(idx, pairs, block=len(X))(X, 0)


def fit_ridge_path(X, t, idx, pairs, alphas=ALPHAS, block=FIT_BLOCK):
    """Ridge for every alpha from one Gram matrix, reading X block by block.

    Minimises ||t - Zw||^2 + alpha ||w||^2 on standardised columns with an
    unpenalised intercept - the same objective as sklearn's Ridge.
    """
    rows = BlockDesign(idx, pairs, block)
    n, d = len(X), rows.d
    s1, s2 = np.zeros(d), np.zeros(d)
    for s in range(0, n, block):                       # pass 1: scaling, train only
        Z = rows(X, s)
        s1 += Z.sum(0)
        s2 += np.square(Z).sum(0)
    mean = s1 / n
    scale = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
    scale[scale < 1e-9] = 1.0                          # constant column -> contributes 0
    t_mean = float(t.mean())
    G, b, step = np.zeros((d, d)), np.zeros(d), np.empty((d, d))
    for s in range(0, n, block):                       # pass 2: Z'Z and Z't
        Z = rows(X, s, mean, scale)
        G += np.matmul(Z.T, Z, out=step)
        b += Z.T @ (t[s:s + block] - t_mean)
    lam, Q = np.linalg.eigh(G)
    qb, cut = Q.T @ b, lam[-1] * 1e-12
    W = np.empty((d, len(alphas)))
    for j, a in enumerate(alphas):                     # drop null directions when alpha = 0
        W[:, j] = Q @ np.where(lam + a > cut, qb / np.maximum(lam + a, cut), 0.0)
    return {'W': W, 'mean': mean, 'scale': scale, 't_mean': t_mean, 'alphas': list(alphas),
            'eig_min': float(lam[0] / n), 'eig_max': float(lam[-1] / n), 'n': n}


def predict_path(X, model, idx, pairs, target, block=FIT_BLOCK):
    """Predictions in MINUTES, one column per alpha."""
    rows = BlockDesign(idx, pairs, block)
    out = np.empty((len(X), model['W'].shape[1]))
    for s in range(0, len(X), block):
        out[s:s + block] = rows(X, s, model['mean'], model['scale']) @ model['W'] + model['t_mean']
    if target == 'log':
        np.expm1(out, out=out)
    return np.clip(out, PRED_MIN, PRED_MAX, out=out)


# ----------------------------------------------- pattern 1: gradient computation
def _shard_gradient(task):
    """Gradient SUM of the squared loss over one shard (no division yet)."""
    lo, hi, w = task
    X, t = _STORE['Z'], _STORE['t']
    g = np.zeros_like(w)
    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        B = X[s:e]
        g += B.T @ (B @ w - t[s:e])
    return g, hi - lo


def _reduce(parts):
    """Combine partial results: add the sums and the counts, divide later."""
    g, n = parts[0][0].copy(), parts[0][1]
    for gi, ni in parts[1:]:
        g += gi
        n += ni
    return g, n


def train_gd(paths, alpha, lam_max, lam_min, iters, workers=1, shards=N_SHARDS, reduce='ordered', trace=None):
    """Full-batch accelerated gradient descent for ridge on standardised data.

    reduce='ordered'    add shard gradients 0,1,2,... (deterministic)
    reduce='completion' add them as workers finish    (the bug we demonstrate)
    """
    open_store(paths)
    n, d = _STORE['Z'].shape
    bounds = shard_bounds(n, shards)
    L, mu = lam_max + alpha / n, max(lam_min + alpha / n, 1e-12)
    beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    w = v = np.zeros(d)
    pool, startup = (None, 0.0) if workers == 1 else make_pool(workers, paths)
    t0 = time.perf_counter()
    try:
        for it in range(iters):
            tasks = [(lo, hi, v) for lo, hi in bounds]
            if pool is None:
                parts = [_shard_gradient(x) for x in tasks]
            elif reduce == 'ordered':
                parts = list(pool.map(_shard_gradient, tasks))        # submission order
            else:
                parts = [f.result() for f in as_completed([pool.submit(_shard_gradient, x) for x in tasks])]
            g, seen = _reduce(parts)
            grad = g / seen + (alpha / n) * v
            w_new = v - grad / L
            v = w_new + beta * (w_new - w)
            w = w_new
            if trace is not None:
                trace.append(w.copy())
        compute = time.perf_counter() - t0
    finally:
        if pool is not None:
            pool.shutdown()
    return {'w': w, 'startup': startup, 'compute': compute, 'total': startup + compute}


def roundtrip_seconds(paths, workers, shards=N_SHARDS, repeats=100):
    """Cost of one all-reduce step with no maths in it: send, schedule, receive."""
    open_store(paths)
    d = _STORE['Z'].shape[1]
    pool, _ = make_pool(workers, paths)
    try:
        empty = [(0, 0, np.zeros(d))] * shards
        list(pool.map(_shard_gradient, empty))
        t0 = time.perf_counter()
        for _ in range(repeats):
            list(pool.map(_shard_gradient, empty))
        return (time.perf_counter() - t0) / repeats
    finally:
        pool.shutdown()


def shard_seconds(paths, shards=N_SHARDS, repeats=5):
    """Compute time of every shard on its own: shows how balanced the shards are."""
    open_store(paths)
    n, d = _STORE['Z'].shape
    out = []
    for lo, hi in shard_bounds(n, shards):
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            _shard_gradient((lo, hi, np.ones(d)))
            best = min(best, time.perf_counter() - t0)
        out.append(best)
    return np.array(out)


# ----------------------------------------------- pattern 2: hyperparameter search
def make_trials(sets, history_days=(7, 14, 21), targets=('log', 'raw')):
    trials = []
    for fs in sets:
        for tgt in targets:
            for days in history_days:
                trials.append({'trial': len(trials), 'features': fs, 'target': tgt, 'history_days': days})
    return trials


def _run_trial(task):
    """One independent job: fit on recent training rows, score all alphas on validation."""
    trial, (idx, pairs), start = task
    began, t0 = time.time(), time.perf_counter()
    X, y, Xv, yv = _STORE['Xtr'], _STORE['ytr'], _STORE['Xva'], _STORE['yva']
    X, y = X[start:], y[start:]
    t = np.log1p(y) if trial['target'] == 'log' else np.asarray(y)
    model = fit_ridge_path(X, t, idx, pairs)
    err = predict_path(Xv, model, idx, pairs, trial['target']) - yv[:, None]
    sst = np.square(yv - yv.mean()).sum()
    rows = [{**trial, 'alpha': a, 'MAE': float(np.abs(err[:, j]).mean()),
             'RMSE': float(np.sqrt(np.square(err[:, j]).mean())),
             'R2': float(1 - np.square(err[:, j]).sum() / sst)} for j, a in enumerate(model['alphas'])]
    return {'trial': trial['trial'], 'rows': rows, 'model': model, 'seconds': time.perf_counter() - t0,
            'began': began, 'pid': os.getpid(), 'n_rows': len(X), 'n_features': len(idx) + len(pairs)}


def trial_cost(task):
    """Rough work estimate (rows x features^2) used to start the longest jobs first."""
    _, (idx, pairs), start = task
    return (_STORE['Xtr'].shape[0] - start) * (len(idx) + len(pairs)) ** 2


def run_search(paths, tasks, workers=1, collect='by_trial', longest_first=True):
    """collect='by_trial'   sort results by trial id  (deterministic)
    collect='completion' keep arrival order         (the bug we demonstrate)"""
    open_store(paths)
    queue = sorted(tasks, key=lambda x: (-trial_cost(x), x[0]['trial'])) if longest_first else list(tasks)
    startup = 0.0
    if workers == 1:
        t0 = time.perf_counter()
        results = [_run_trial(x) for x in queue]
        compute = time.perf_counter() - t0
    else:
        pool, startup = make_pool(workers, paths)
        t0 = time.perf_counter()
        try:
            results = [f.result() for f in as_completed([pool.submit(_run_trial, x) for x in queue])]
            compute = time.perf_counter() - t0
        finally:
            pool.shutdown()
    if collect == 'by_trial':
        results.sort(key=lambda r: r['trial'])
    return {'results': results, 'rows': [row for r in results for row in r['rows']],
            'startup': startup, 'compute': compute, 'total': startup + compute}


def best_row(rows):
    """Lowest validation MAE; ties go to the lower trial id, then the smaller alpha."""
    return min(rows, key=lambda r: (r['MAE'], r['trial'], r['alpha']))


def rows_fingerprint(rows):
    text = '\n'.join(f"{r['trial']}|{r['alpha']!r}|{r['MAE']!r}|{r['RMSE']!r}" for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ------------------------------------------------- unseeded randomness experiment
def _random_trial(task):
    """Random search: draw alpha and a training subsample, fit, score on validation."""
    k, seed, n_sub = task
    rng = np.random.default_rng(None if seed is None else [seed, k])
    alpha = float(10 ** rng.uniform(-1, 5))
    Z, t = _STORE['Z'], _STORE['t']
    rows = np.sort(rng.choice(len(Z), size=min(n_sub, len(Z)), replace=False))
    A, b = Z[rows], t[rows]
    b_mean = b.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (b - b_mean))
    pred = np.clip(np.expm1(_STORE['Zva'] @ w + b_mean + _STORE['t_mean'][0]), PRED_MIN, PRED_MAX)
    return k, alpha, float(np.abs(pred - _STORE['yva']).mean())


def random_search(paths, n_trials=12, seed=SEED, workers=2, n_sub=20000):
    """seed=None reproduces the bug; an integer seeds each TRIAL (not each worker),
    so a draw does not depend on which worker happens to run the trial."""
    open_store(paths)
    tasks = [(k, seed, n_sub) for k in range(n_trials)]
    if workers == 1:
        out = [_random_trial(x) for x in tasks]
    else:
        pool, _ = make_pool(workers, paths)
        try:
            out = list(pool.map(_random_trial, tasks))
        finally:
            pool.shutdown()
    best = min(out, key=lambda r: (r[2], r[0]))
    return {'trials': out, 'best_trial': best[0], 'best_alpha': best[1], 'best_MAE': best[2],
            'fingerprint': fingerprint(np.array(out))}

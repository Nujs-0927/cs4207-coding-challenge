import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import c_parallel as cp                                                            # noqa: E402
from c_models import (ConstantBaseline, LookupBaseline, Scoreboard, boundary_trips,  # noqa: E402
                      check_split, evaluate, to_minutes, with_calendar)


def trips(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.Timestamp('2026-01-01') + pd.to_timedelta(np.sort(rng.integers(0, 30 * 86400, n)), unit='s')
    return pd.DataFrame({'row_id': [f'r{i}' for i in range(n)], 'pickup_datetime': t,
                         'pickup_zone': rng.choice(['1', '2', '3'], n), 'dropoff_zone': rng.choice(['1', '2', '3'], n),
                         'duration_minutes': rng.uniform(2, 60, n)})


def cut(df):
    t = df['pickup_datetime']
    a, b = pd.Timestamp('2026-01-22'), pd.Timestamp('2026-01-27')
    return df[t < a], df[(t >= a) & (t < b)], df[t >= b]


class TestSplitAndEvaluation(unittest.TestCase):
    def test_split_is_chronological_and_complete(self):
        df = trips()
        rep = check_split(*cut(df), total_rows=len(df))
        self.assertEqual(rep['rows'].sum(), len(df))
        self.assertLess(rep.loc['train', 'last_pickup'], rep.loc['validation', 'first_pickup'])

    def test_random_split_is_rejected(self):
        df = trips().sample(frac=1, random_state=1)
        with self.assertRaises(ValueError):
            check_split(df.iloc[:400], df.iloc[400:500], df.iloc[500:])

    def test_boundary_trips(self):
        df = trips(3).assign(pickup_datetime=pd.Timestamp('2026-01-21 23:50'), duration_minutes=[5.0, 10.0, 30.0])
        self.assertEqual(boundary_trips(df, '2026-01-22'), 2)

    def test_metrics_match_sklearn(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        rng = np.random.default_rng(0)
        y, p = rng.uniform(1, 60, 500), rng.uniform(1, 60, 500)
        m = evaluate(y, p)
        self.assertAlmostEqual(m['MAE'], mean_absolute_error(y, p))
        self.assertAlmostEqual(m['RMSE'], mean_squared_error(y, p) ** 0.5)
        self.assertAlmostEqual(m['R2'], r2_score(y, p))
        self.assertAlmostEqual(m['bias'], (p - y).mean())

    def test_evaluate_rejects_bad_input(self):
        for y, p in [([1, 2], [1]), ([], []), ([1.0, np.nan], [1, 2]), ([1, 2], [np.inf, 1])]:
            with self.assertRaises(ValueError):
                evaluate(y, p)

    def test_evaluate_warns_on_log_scale(self):
        y = np.linspace(5, 60, 50)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            evaluate(y, np.log1p(y))
        self.assertTrue(any('log1p' in str(x.message) for x in w))
        self.assertAlmostEqual(evaluate(y, to_minutes(np.log1p(y)))['MAE'], 0)

    def test_scoreboard_round_trip_and_overwrite(self):
        b = Scoreboard()
        b.add('m', 'validation', [1, 2, 3], [1, 2, 4])
        b.add('m', 'validation', [1, 2, 3], [1, 2, 3])          # same key replaces the old row
        b.add('m', 'test', [1, 2, 3], [2, 2, 3])
        with tempfile.TemporaryDirectory() as d:
            b.save(Path(d) / 's.csv')
            again = Scoreboard.load(Path(d) / 's.csv')
        self.assertEqual(len(again.rows), 2)
        self.assertEqual(again.table().loc['m', 'validation_MAE'], 0)


class TestBaselines(unittest.TestCase):
    def test_constant(self):
        self.assertTrue((ConstantBaseline().fit(None, [1, 2, 9]).predict(range(4)) == 2).all())

    def test_lookup_and_backoff(self):
        X = pd.DataFrame({'a': ['x', 'x', 'y'], 'b': ['1', '1', '2']})
        m = LookupBaseline([['a', 'b'], ['a']]).fit(X, [10, 20, 60])
        new = pd.DataFrame({'a': ['x', 'x', 'q'], 'b': ['1', '9', '9']}, index=[7, 8, 9])
        np.testing.assert_allclose(m.predict(new), [15, 15, 20])   # pair, then zone, then overall median

    def test_lookup_ignores_validation_labels(self):
        tr, va, _ = cut(trips())
        m = LookupBaseline([['weekday', 'hour']]).fit(with_calendar(tr), tr['duration_minutes'])
        a = m.predict(with_calendar(va))
        b = m.predict(with_calendar(va.assign(duration_minutes=999.0)))
        np.testing.assert_array_equal(a, b)


class TestRidge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        n = 3000
        cls.X = rng.normal(size=(n, 6)) * np.array([1, 5, .1, 2, 1, 3]) + np.array([0, 10, 1, -4, 0, 2])
        cls.y = np.exp(2 + 0.2 * cls.X[:, 0] + 0.05 * cls.X[:, 1] - 0.5 * cls.X[:, 2] + rng.normal(0, .2, n))
        cls.idx, cls.pairs = list(range(6)), [(0, 1), (2, 2)]

    def tearDown(self):
        cp.close_store()

    def test_matches_sklearn_ridge_and_ols(self):
        from sklearn.linear_model import LinearRegression, Ridge
        t = np.log1p(self.y)
        m = cp.fit_ridge_path(self.X, t, self.idx, self.pairs, alphas=(0.0, 10.0), block=500)
        Z = (cp.design(self.X, self.idx, self.pairs) - m['mean']) / m['scale']
        np.testing.assert_allclose(m['W'][:, 0], LinearRegression().fit(Z, t).coef_, atol=1e-8)
        np.testing.assert_allclose(m['W'][:, 1], Ridge(alpha=10.0).fit(Z, t).coef_, atol=1e-8)

    def _gd_inputs(self, folder):
        Z = (self.X - self.X.mean(0)) / self.X.std(0)
        t = np.log1p(self.y)
        lam = np.linalg.eigvalsh(Z.T @ Z / len(Z))
        paths = cp.save_arrays(folder, Z=Z, t=t - t.mean(), Zva=Z[:500], yva=self.y[:500], t_mean=np.array([t.mean()]))
        return Z, t, lam, paths

    def test_gradient_descent_reaches_closed_form(self):
        from sklearn.linear_model import Ridge
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            Z, t, lam, paths = self._gd_inputs(d)
            w = cp.train_gd(paths, 5.0, lam[-1], lam[0], iters=400)['w']
            cp.close_store()
        np.testing.assert_allclose(w, Ridge(alpha=5.0).fit(Z, t).coef_, atol=1e-8)

    def test_parallel_gradient_is_bit_identical(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            _, _, lam, paths = self._gd_inputs(d)
            a = cp.train_gd(paths, 5.0, lam[-1], lam[0], iters=10, workers=1)['w']
            b = cp.train_gd(paths, 5.0, lam[-1], lam[0], iters=10, workers=2)['w']
            cp.close_store()
        self.assertEqual(cp.fingerprint(a), cp.fingerprint(b))

    def test_seeded_random_search_ignores_worker_count(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            *_, paths = self._gd_inputs(d)
            a = cp.random_search(paths, n_trials=4, workers=1, n_sub=300)
            b = cp.random_search(paths, n_trials=4, workers=2, n_sub=300)
            cp.close_store()
        self.assertEqual(a['fingerprint'], b['fingerprint'])

    def test_parallel_search_equals_sequential(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            paths = cp.save_arrays(d, Xtr=self.X[:2500], ytr=self.y[:2500], Xva=self.X[2500:], yva=self.y[2500:])
            sets = {'small': (self.idx[:3], []), 'big': (self.idx, self.pairs)}
            trials = cp.make_trials(sets, history_days=(7, 21), targets=('log', 'raw'))
            tasks = [(t, sets[t['features']], 1000 if t['history_days'] == 7 else 0) for t in trials]
            a, b = cp.run_search(paths, tasks, 1), cp.run_search(paths, tasks, 2)
            cp.close_store()
        self.assertEqual(cp.rows_fingerprint(a['rows']), cp.rows_fingerprint(b['rows']))
        self.assertEqual(cp.best_row(a['rows']), cp.best_row(b['rows']))


class TestHelpers(unittest.TestCase):
    def test_shards_cover_every_row_and_are_balanced(self):
        b = cp.shard_bounds(1003, 8)
        self.assertEqual((b[0][0], b[-1][1]), (0, 1003))
        self.assertTrue(all(x[1] == y[0] for x, y in zip(b, b[1:])))
        sizes = [hi - lo for lo, hi in b]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_amdahl(self):
        p = 0.9
        s = [1 / ((1 - p) + p / w) for w in (1, 2, 4)]
        self.assertAlmostEqual(cp.amdahl_p(s[2], 4), p)
        self.assertAlmostEqual(cp.amdahl_fit([1, 2, 4], s), p)

    def test_linear_feature_sets_drop_unsafe_columns(self):
        cols = ['hour', 'day_of_month', 'is_peak', 'band_night', 'band_midday', 'tod_sin', 'te_pair', 'te_pair_count']
        sets = cp.feature_sets(cols)
        for idx, _ in sets.values():
            self.assertFalse({cols[i] for i in idx} & set(cp.NOT_FOR_LINEAR))
        self.assertNotIn(cols.index('te_pair'), sets['calendar'][0])
        idx, pairs = sets['interactions']
        names = {(cols[idx[a]], cols[idx[b]]) for a, b in pairs}
        self.assertIn(('te_pair', 'band_midday'), names)
        self.assertIn(('te_pair', 'te_pair'), names)


if __name__ == '__main__':
    unittest.main()

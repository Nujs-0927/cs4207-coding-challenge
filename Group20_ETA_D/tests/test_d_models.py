import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / 'Group20_ETA_C')]
import d_models as dm  # noqa: E402


def toy(n=600, seed=0, start='2026-01-01'):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'pickup_datetime': pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.integers(0, 5 * 86400, n)), unit='s'),
        'te_pair': rng.normal(2.5, 0.5, n), 'hour': rng.integers(0, 24, n), 'day_of_month': rng.integers(1, 22, n),
        'pickup_zone_code': rng.integers(0, 265, n), 'dropoff_zone_code': rng.integers(0, 265, n),
    })
    df['duration_minutes'] = np.clip(np.expm1(df['te_pair'] + rng.normal(0, 0.2, n)), 1, 180)
    return df


def to_minutes(p):
    return np.clip(np.expm1(p), 1, 180)


class TestD(unittest.TestCase):
    def test_target_and_day_of_month_not_inputs(self):
        X, y = dm.split_xy(toy())
        self.assertNotIn('duration_minutes', X)
        self.assertNotIn('day_of_month', X)
        self.assertNotIn('pickup_datetime', X)
        self.assertEqual(len(X), len(y))

    def test_rejects_leaky_columns(self):
        with self.assertRaises(ValueError):
            dm.split_xy(toy().assign(trip_distance=1.0))

    def test_zone_codes_are_fixed_categories(self):
        a, _ = dm.split_xy(toy(seed=1))
        b, _ = dm.split_xy(toy(seed=2).head(5))
        self.assertListEqual(list(a['pickup_zone_code'].cat.categories), list(b['pickup_zone_code'].cat.categories))

    def test_forest_prediction_is_deterministic_after_fix(self):
        tr, va = toy(seed=3), toy(seed=4, start='2026-01-10')
        (Xtr, ytr), (Xva, yva) = [(dm.split_xy(d, categorical=False)[0].astype('float64'), d['duration_minutes'].to_numpy()) for d in (tr, va)]
        m, _ = dm.fit('Random Forest', {'min_samples_leaf': 5, 'max_features': 0.6}, Xtr, ytr, Xva, yva, n_threads=2)
        prints = {dm.fingerprint(dm.predict_minutes(m, Xva, to_minutes)) for _ in range(5)}
        self.assertEqual(len(prints), 1)
        self.assertEqual(m.n_jobs, 2)          # training thread count restored

    def test_seeded_lightgbm_is_reproducible(self):
        tr, va = toy(seed=5), toy(seed=6, start='2026-01-10')
        (Xtr, ytr), (Xva, yva) = [dm.split_xy(d) for d in (tr, va)]
        p = [dm.fingerprint(dm.predict_minutes(dm.fit('LightGBM', {'num_leaves': 15, 'min_child_samples': 20},
                                                       Xtr, ytr, Xva, yva, n_threads=t)[0], Xva, to_minutes)) for t in (1, 2)]
        self.assertEqual(p[0], p[1])


if __name__ == '__main__':
    unittest.main()

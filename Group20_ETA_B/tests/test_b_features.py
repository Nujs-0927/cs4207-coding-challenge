import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b_features import FeatureBuilder, fingerprint, load_cleaned, stateless_features, time_band, zone_codes  # noqa: E402


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.Timestamp('2026-01-05') + pd.to_timedelta(rng.integers(0, 14 * 86400, n), unit='s')
    X = pd.DataFrame({
        'pickup_datetime': t,
        'pickup_zone': rng.choice(['132', '161', '237'], n).astype(object),
        'dropoff_zone': rng.choice(['138', '161', '236'], n).astype(object),
        'passenger_count': np.where(rng.random(n) < 0.2, np.nan, rng.integers(1, 4, n)),
    })
    X['passenger_count_missing'] = X['passenger_count'].isna()
    y = pd.Series(rng.uniform(2, 60, n))
    return X, y


class TestFeatures(unittest.TestCase):
    def test_rejects_leaky_columns(self):
        X, _ = toy()
        for col in ['duration_minutes', 'trip_distance', 'fare_amount', 'rate_code']:
            with self.assertRaises(ValueError):
                stateless_features(X.assign(**{col: 1.0}))

    def test_time_bands_cover_every_hour(self):
        b = time_band(pd.Series(range(24)))
        self.assertEqual(b.tolist()[:6], ['night'] * 6)
        self.assertEqual(b[8], 'morning_peak'); self.assertEqual(b[17], 'afternoon_peak')

    def test_transform_ignores_eval_labels(self):
        X, y = toy()
        Xe, _ = toy(seed=1)
        fb = FeatureBuilder(n_folds=3)
        fb.fit_transform(X, y)
        a = fb.transform(Xe)
        Xe2 = Xe.copy()  # evaluation target never enters transform
        self.assertEqual(fingerprint(a), fingerprint(fb.transform(Xe2)))

    def test_oof_row_does_not_see_own_label(self):
        X, y = toy()
        base = FeatureBuilder(n_folds=5).fit_transform(X, y)
        y2 = y.copy(); y2.iloc[0] = 10_000.0          # change one training label
        moved = FeatureBuilder(n_folds=5).fit_transform(X, y2)
        self.assertAlmostEqual(base['te_pair'].iloc[0], moved['te_pair'].iloc[0])

    def test_parallel_equals_serial(self):
        X, y = toy()
        a = FeatureBuilder(n_jobs=1).fit_transform(X, y)
        b = FeatureBuilder(n_jobs=2).fit_transform(X, y)
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_unseen_zone_gets_prior(self):
        X, y = toy()
        fb = FeatureBuilder(); fb.fit_transform(X, y)
        Xn = X.head(3).copy(); Xn['pickup_zone'] = '999'; Xn['dropoff_zone'] = '998'
        out = fb.transform(Xn)
        self.assertTrue(np.allclose(out['te_pair'], np.log1p(y).mean()))
        self.assertTrue((out['te_pair_count'] == 0).all())

    def test_load_sample_file_and_parts_dir(self):
        import tempfile
        X, y = toy(50)
        df = X.assign(duration_minutes=y, row_id=[f'r{i}' for i in range(50)])
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'cleaned_parts').mkdir()
            df.iloc[:20].to_parquet(Path(d) / 'cleaned_parts' / 'part-00001.parquet')
            df.iloc[20:].to_parquet(Path(d) / 'cleaned_parts' / 'part-00000.parquet')
            df.to_parquet(Path(d) / 's.parquet')
            a, b = load_cleaned(d), load_cleaned(Path(d) / 's.parquet')
            self.assertEqual(len(a), 50); self.assertTrue(a['pickup_datetime'].is_monotonic_increasing)
            self.assertEqual(a['row_id'].tolist(), b['row_id'].tolist())

    def test_zone_codes_fixed_vocabulary(self):
        codes = zone_codes(pd.Series(['1', '132', '265', 'bogus']))
        self.assertEqual(codes.tolist(), [0, 131, 264, 266])

    def test_unseeded_folds_differ_seeded_match(self):
        X, y = toy()
        a = FeatureBuilder(seed=None).fit_transform(X, y); b = FeatureBuilder(seed=None).fit_transform(X, y)
        c = FeatureBuilder(seed=1).fit_transform(X, y); d = FeatureBuilder(seed=1).fit_transform(X, y)
        self.assertNotEqual(fingerprint(a), fingerprint(b))
        self.assertEqual(fingerprint(c), fingerprint(d))

    def test_no_missing_and_numeric(self):
        X, y = toy()
        out = FeatureBuilder().fit_transform(X, y)
        self.assertFalse(out.isna().any().any())
        self.assertTrue(all(np.issubdtype(t, np.number) for t in out.dtypes))


if __name__ == '__main__':
    unittest.main()

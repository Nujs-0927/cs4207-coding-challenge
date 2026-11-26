"""Meaningful boundary, leakage and multiprocessing checks; no invented project data."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eta_data as eta
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def fixtures():
    rows = []
    def row(**changes):
        record = dict(tpep_pickup_datetime=pd.Timestamp('2026-01-15 12:00'),
                      tpep_dropoff_datetime=pd.Timestamp('2026-01-15 12:10'),
                      trip_distance=2.0, fare_amount=10.0, PULocationID=1,
                      DOLocationID=2, passenger_count=1.0, RatecodeID=1.0,
                      payment_type=1, VendorID=2)
        record.update(changes)
        rows.append(record)
    row()  # valid
    row(passenger_count=np.nan, RatecodeID=np.nan, payment_type=0)  # keep Flex Fare
    row(tpep_pickup_datetime=pd.NaT)
    row(tpep_pickup_datetime=pd.Timestamp('2025-12-31 23:55'),
        tpep_dropoff_datetime=pd.Timestamp('2026-01-01 00:05'))
    row(tpep_dropoff_datetime=pd.Timestamp('2026-01-15 12:00'))
    row(tpep_dropoff_datetime=pd.Timestamp('2026-01-15 12:00:30'))
    row(tpep_dropoff_datetime=pd.Timestamp('2026-01-15 15:00:01'))
    row(trip_distance=np.nan)
    row(trip_distance=0)
    row(trip_distance=101)
    row(fare_amount=np.inf)
    row(fare_amount=-1)
    row(trip_distance=20)  # 120mph
    row(PULocationID=999, DOLocationID=np.nan, passenger_count=99, RatecodeID=99)
    row(tpep_pickup_datetime=pd.Timestamp('2026-01-31 23:55'),
        tpep_dropoff_datetime=pd.Timestamp('2026-02-01 00:05'))  # pickup month controls
    return pd.DataFrame(rows)


class CleaningTests(unittest.TestCase):
    def test_each_filter_and_retained_missing_categories(self):
        clean, audit = eta.clean_frame(fixtures(), eta.CleaningConfig(), 'yellow_tripdata_2026-01.parquet')
        self.assertEqual(audit['rows_in'], 15)
        self.assertEqual(audit['rows_kept'], 4)
        self.assertEqual(sum(audit['removed'].values()), 11)
        for key, _ in eta.RULES:
            self.assertEqual(audit['removed'][key], 2 if key == 'duration_outside_scope' else 1)
        self.assertEqual(audit['flex_fare_rows_kept'], 1)
        self.assertEqual(audit['missing_passenger_rows_kept'], 2)
        self.assertEqual(clean.loc[2, 'pickup_zone'], '__UNKNOWN__')
        self.assertEqual(clean.loc[2, 'dropoff_zone'], '__MISSING__')
        self.assertEqual(clean.loc[2, 'rate_code'], '__UNKNOWN__')

    def test_no_post_trip_or_label_inputs_silently_exported(self):
        clean, _ = eta.clean_frame(fixtures(), eta.CleaningConfig(), 'yellow_tripdata_2026-01.parquet')
        self.assertFalse(set(['tpep_dropoff_datetime', 'fare_amount', 'trip_distance', 'payment_type', 'tip_amount']).intersection(clean.columns))
        self.assertIn('duration_minutes', clean)  # explicit target
        self.assertEqual(list(clean.columns), eta.OUTPUT_SCHEMA.names)

    def test_duration_and_speed_boundaries(self):
        frame = fixtures().iloc[[0]].copy()
        frame['tpep_dropoff_datetime'] = pd.Timestamp('2026-01-15 12:01')
        frame['trip_distance'] = 1.0
        clean, _ = eta.clean_frame(frame, eta.CleaningConfig(), 'yellow_tripdata_2026-01.parquet')
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean['duration_minutes'].iloc[0], 1)
        self.assertEqual(eta.month_bounds('yellow_tripdata_2026-12.parquet')[1], pd.Timestamp('2027-01-01'))

    def test_missing_required_column_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'fare_amount'):
            eta.clean_frame(fixtures().drop(columns='fare_amount'), eta.CleaningConfig(), 'yellow_tripdata_2026-01.parquet')

    def test_parallel_export_sample_and_conservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'yellow_tripdata_2026-01.parquet'
            pq.write_table(pa.Table.from_pandas(fixtures(), preserve_index=False), source)
            config = eta.CleaningConfig(batch_size=3, sample_size=3)
            serial = eta.run_pipeline([source], config, workers=1, output=base / 'serial')
            parallel = eta.run_pipeline([source], config, workers=2, output=base / 'parallel')
            self.assertEqual(serial['ordered_result_sha256'], parallel['ordered_result_sha256'])
            self.assertEqual(serial['removed'], parallel['removed'])
            self.assertEqual((base / 'serial/sample_100k.csv').read_bytes(), (base / 'parallel/sample_100k.csv').read_bytes())
            for folder in ['serial', 'parallel']:
                sample = pd.read_parquet(base / folder / 'sample_100k.parquet')
                self.assertEqual(len(sample), 3)
                self.assertTrue(sample['pickup_datetime'].is_monotonic_increasing)
                self.assertTrue(sample['row_id'].is_unique)
            with self.assertRaises(FileExistsError):
                eta.run_pipeline([source], config, output=base / 'serial')


if __name__ == '__main__':
    unittest.main()

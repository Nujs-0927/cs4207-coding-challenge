"""Group 20 / A: bounded-memory, deterministic NYC Yellow Taxi cleaning.

Uses only row-wise rules. No imputer, scaler, encoder or model is fitted here.
Raw distance, fare and dropoff are quality/label inputs, NOT prediction features.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
# The workspace runtime is a convenience for this delivery, not a server requirement.
LOCAL_RUNTIME = ROOT.parent.parent / 'tmp' / 'group20-runtime'
if LOCAL_RUNTIME.is_dir():
    sys.path.insert(0, str(LOCAL_RUNTIME))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_PAGE = 'https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page'
DATA_DICTIONARY = 'https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf'
REQUIRED = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance',
            'fare_amount', 'PULocationID', 'DOLocationID']
OPTIONAL = ['passenger_count', 'RatecodeID', 'payment_type', 'VendorID']
RULES = [
    ('missing_timestamps', 'Missing or invalid pickup/dropoff timestamp'),
    ('pickup_outside_file_month', 'Pickup outside the month in source filename'),
    ('non_positive_duration', 'Duration <= 0 minutes'),
    ('duration_outside_scope', 'Duration outside configured [min, max] minutes'),
    ('missing_or_nonfinite_distance', 'Distance missing or non-finite'),
    ('non_positive_distance', 'Distance <= 0 miles'),
    ('distance_above_limit', 'Distance exceeds configured maximum'),
    ('missing_or_nonfinite_fare', 'Fare missing or non-finite'),
    ('non_positive_fare', 'Fare <= 0 dollars'),
    ('speed_above_limit', 'Observed average speed exceeds configured maximum'),
]
STRING_COLUMNS = ['row_id', 'source_file', 'pickup_zone', 'dropoff_zone', 'rate_code', 'vendor_id']
OUTPUT_SCHEMA = pa.schema([
    ('row_id', pa.string()), ('source_file', pa.string()),
    ('pickup_datetime', pa.timestamp('us')), ('duration_minutes', pa.float64()),
    ('pickup_zone', pa.string()), ('dropoff_zone', pa.string()),
    ('passenger_count', pa.float64()), ('passenger_count_missing', pa.bool_()),
    ('rate_code', pa.string()), ('vendor_id', pa.string()),
])


@dataclass(frozen=True)
class CleaningConfig:
    batch_size: int = 100000
    sample_size: int = 100000
    sample_seed: int = 420720
    min_duration_minutes: float = 1.0
    max_duration_minutes: float = 180.0
    max_distance_miles: float = 100.0
    max_speed_mph: float = 100.0

    def validate(self):
        if not (self.batch_size > 0 and self.sample_size > 0):
            raise ValueError('batch_size and sample_size must be positive')
        if not (0 < self.min_duration_minutes <= self.max_duration_minutes):
            raise ValueError('Invalid duration bounds')
        if self.max_distance_miles <= 0 or self.max_speed_mph <= 0:
            raise ValueError('Distance/speed bounds must be positive')


def read_config(path=ROOT / 'config.json'):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    fields = CleaningConfig.__dataclass_fields__
    config = CleaningConfig(**{k: v for k, v in data.items() if k in fields})
    config.validate()
    return config, data


def month_bounds(source_file):
    match = re.fullmatch(r'yellow_tripdata_(\d{4})-(\d{2})\.parquet', source_file)
    if not match:
        raise ValueError(f'Expected yellow_tripdata_YYYY-MM.parquet, got {source_file}')
    year, month = map(int, match.groups())
    start = pd.Timestamp(year=year, month=month, day=1)
    return start, start + pd.offsets.MonthBegin(1)


def code_category(series, valid_codes=None, unknown_code=None):
    number = pd.to_numeric(series, errors='coerce')
    valid = number.notna() & np.isfinite(number) & (number % 1 == 0)
    if valid_codes is not None:
        valid &= number.isin(valid_codes)
    out = pd.Series('__UNKNOWN__', index=series.index, dtype='string')
    out.loc[number.isna()] = '__MISSING__'
    out.loc[valid] = number.loc[valid].astype('int64').astype('string')
    if unknown_code is not None:
        out.loc[number.eq(unknown_code)] = '__UNKNOWN__'
    return out


def clean_frame(frame, config, source_file, row_start=0):
    """Return clean rows plus a non-overlapping, ordered filter audit."""
    missing = set(REQUIRED).difference(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = frame.reset_index(drop=True).copy()
    for col in OPTIONAL:
        if col not in df:
            df[col] = np.nan
    pickup = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    duration = (dropoff - pickup).dt.total_seconds() / 60.0
    distance = pd.to_numeric(df['trip_distance'], errors='coerce')
    fare = pd.to_numeric(df['fare_amount'], errors='coerce')
    start, end = month_bounds(source_file)
    with np.errstate(divide='ignore', invalid='ignore'):
        speed = distance / (duration / 60.0)
    bad = [
        pickup.isna() | dropoff.isna(),
        ~((pickup >= start) & (pickup < end)),
        duration <= 0,
        (duration < config.min_duration_minutes) | (duration > config.max_duration_minutes),
        ~np.isfinite(distance), distance <= 0, distance > config.max_distance_miles,
        ~np.isfinite(fare), fare <= 0, speed > config.max_speed_mph,
    ]
    keep = pd.Series(True, index=df.index)
    counts = {}
    for (name, _), mask in zip(RULES, bad):
        removed = keep & mask.fillna(True)
        counts[name] = int(removed.sum())
        keep &= ~removed
    passenger = pd.to_numeric(df['passenger_count'], errors='coerce').astype('float64')
    # Missing passenger count is common in Flex Fare; retain these trips.
    passenger = passenger.where(np.isfinite(passenger) & (passenger >= 0) & (passenger <= 9))
    clean = pd.DataFrame({
        'row_id': pd.Series([f'{source_file}:{row_start + i:012d}' for i in range(len(df))], dtype='string'),
        'source_file': pd.Series(source_file, index=df.index, dtype='string'),
        'pickup_datetime': pickup,
        'duration_minutes': duration.astype('float64'),
        'pickup_zone': code_category(df['PULocationID'], range(1, 266)),
        'dropoff_zone': code_category(df['DOLocationID'], range(1, 266)),
        'passenger_count': passenger,
        'passenger_count_missing': passenger.isna(),
        # Kept as audit/category metadata; final rate code is not a pickup-time feature.
        'rate_code': code_category(df['RatecodeID'], [1, 2, 3, 4, 5, 6, 99], unknown_code=99),
        'vendor_id': code_category(df['VendorID'], [1, 2, 6, 7]),
    }).loc[keep].reset_index(drop=True)
    flex = pd.to_numeric(df['payment_type'], errors='coerce').eq(0)
    audit = {
        'rows_in': len(df), 'rows_kept': len(clean), 'removed': counts,
        'flex_fare_rows_in': int(flex.sum()),
        'flex_fare_rows_kept': int((flex & keep).sum()),
        'missing_passenger_rows_in': int(df['passenger_count'].isna().sum()),
        'missing_passenger_rows_kept': int(clean['passenger_count'].isna().sum()),
        'missing_rate_rows_in': int(df['RatecodeID'].isna().sum()),
        'missing_rate_rows_kept': int(clean['rate_code'].eq('__MISSING__').sum()),
        'invalid_passenger_values_to_missing': int((df['passenger_count'].notna() & passenger.isna()).sum()),
    }
    assert audit['rows_in'] == audit['rows_kept'] + sum(counts.values())
    return clean, audit


def iter_batches(files, config):
    for path in files:
        pf = pq.ParquetFile(path)
        missing = set(REQUIRED).difference(pf.schema_arrow.names)
        if missing:
            raise ValueError(f'{path.name}: missing required columns {sorted(missing)}')
        columns = REQUIRED + [c for c in OPTIONAL if c in pf.schema_arrow.names]
        offset = 0
        for batch in pf.iter_batches(batch_size=config.batch_size, columns=columns, use_threads=False):
            frame = batch.to_pandas()
            yield frame, config, path.name, offset
            offset += len(frame)


def _clean_job(args):
    return clean_frame(*args)


def ordered_results(jobs, workers):
    if workers == 1:
        for job in jobs:
            yield _clean_job(job)
        return
    # Bounded queue prevents whole-file loading; FIFO collection fixes shard order.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for job in jobs:
            pending.append(pool.submit(_clean_job, job))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def sample_priority(ids, seed):
    # Stable row-key hash, independent of worker completion/batch order.
    values = pd.util.hash_pandas_object(ids.astype('string'), index=False).to_numpy(dtype='uint64')
    with np.errstate(over='ignore'):
        values = values ^ np.uint64(seed)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        return values ^ (values >> np.uint64(31))


def run_pipeline(files, config, workers=1, output=None):
    """Same measured cleaning workload for serial and parallel runs.

    Timing includes read, conversion, dispatch, cleaning, fixed-order merge and
    fingerprinting. Excludes writing exports and selecting the handoff sample.
    """
    if workers < 1:
        raise ValueError('workers must be >= 1')
    config.validate()
    if not files:
        raise ValueError('No input Parquet files')
    if len({p.name for p in files}) != len(files):
        raise ValueError('Input basenames must be unique for stable row IDs')
    output = Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'cleaned_parts').mkdir()
    counts, meta = Counter(), Counter()
    digest = hashlib.sha256()
    candidate = None
    measured = 0.0
    start_wall = time.perf_counter()
    iterator = iter(ordered_results(iter_batches(files, config), workers))
    part = 0
    while True:
        start = time.perf_counter()
        try:
            clean, audit = next(iterator)
        except StopIteration:
            measured += time.perf_counter() - start
            break
        values = pd.util.hash_pandas_object(clean, index=False).to_numpy(dtype='uint64')
        digest.update(values.astype('<u8', copy=False).tobytes())
        counts.update(audit['removed'])
        meta.update({k: v for k, v in audit.items() if k != 'removed'})
        measured += time.perf_counter() - start
        if output is not None:
            table = pa.Table.from_pandas(clean, schema=OUTPUT_SCHEMA, preserve_index=False)
            pq.write_table(table, output / 'cleaned_parts' / f'part-{part:05d}.parquet', compression='snappy')
            part += 1
            block = clean.copy()
            block['_priority'] = sample_priority(block['row_id'], config.sample_seed)
            candidate = block if candidate is None else pd.concat([candidate, block], ignore_index=True)
            candidate = candidate.sort_values(['_priority', 'row_id'], kind='stable').head(config.sample_size).reset_index(drop=True)
    result = {
        'workers': workers, 'timed_cleaning_seconds': measured,
        'wall_seconds_including_exports': time.perf_counter() - start_wall,
        **dict(meta), 'removed': {k: int(counts[k]) for k, _ in RULES},
        'ordered_result_sha256': digest.hexdigest(),
    }
    assert result['rows_in'] == result['rows_kept'] + sum(result['removed'].values())
    if output is not None:
        if candidate is None or candidate.empty:
            raise ValueError('No rows survived the configured cleaning rules; no development sample can be exported.')
        candidate = candidate.drop(columns='_priority').sort_values(['pickup_datetime', 'row_id'], kind='stable').reset_index(drop=True)
        candidate.to_csv(output / 'sample_100k.csv', index=False, date_format='%Y-%m-%d %H:%M:%S.%f')
        pq.write_table(pa.Table.from_pandas(candidate, schema=OUTPUT_SCHEMA, preserve_index=False), output / 'sample_100k.parquet')
        result['sample_rows'] = len(candidate)
        manifest = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'execution_location': ('local_development_only; rerun on Rockhill for assessment'
                                   if platform.system() == 'Windows' else
                                   f'host={platform.node()}; verify this is the allocated Rockhill environment before assessment use'),
            'data_source': SOURCE_PAGE, 'data_dictionary': DATA_DICTIONARY,
            'config': asdict(config), 'inputs': [], 'result': result,
            'prediction_features': ['pickup_datetime (derive calendar features)', 'pickup_zone', 'dropoff_zone (intended destination assumption)', 'passenger_count', 'passenger_count_missing'],
            'not_features': ['row_id', 'source_file', 'duration_minutes (target)', 'rate_code (final rate)', 'vendor_id (metadata; only add with an explicit availability argument)'],
            'data_order': 'Full parts retain source-file/source-row order, NOT global chronological order. Split on pickup_datetime, never on part number.',
            'sample_method': 'Seeded bottom-k stable row-key hash across ALL retained rows, then sorted by pickup time; representative development sample, NOT a test holdout. Stable within recorded pandas version.',
            'scope_warning': 'These fixed cleaning thresholds define a restricted valid-trip population. Future dropoff, distance and fare are used only for retrospective quality/label checks. This introduces selection limitations; do not claim deployment-wide performance.',
            'duplicate_policy': 'No removal by matching time/zone/distance alone: those fields are not a unique trip ID.',
        }
        for p in files:
            with p.open('rb') as stream:
                h = hashlib.file_digest(stream, 'sha256').hexdigest()
            pf = pq.ParquetFile(p)
            manifest['inputs'].append({'file': p.name, 'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': h, 'rows': pf.metadata.num_rows, 'row_groups': pf.metadata.num_row_groups, 'available_columns': pf.schema_arrow.names})
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        rows, remaining = [], result['rows_in']
        rows.append({'step': 'raw', 'rule': 'Input rows', 'removed': 0, 'remaining': remaining})
        for key, label in RULES:
            remaining -= result['removed'][key]
            rows.append({'step': key, 'rule': label, 'removed': result['removed'][key], 'remaining': remaining})
        pd.DataFrame(rows).to_csv(output / 'cleaning_audit.csv', index=False)
    return result


def benchmark(files, config, repeats=3):
    if repeats < 3:
        raise ValueError('Use at least 3 measured repeats')
    rows = []
    for workers in (1, 2, 4):
        print(f'Warm-up: {workers} worker(s)', flush=True)
        warm = run_pipeline(files, config, workers)
        for repeat in range(1, repeats + 1):
            r = run_pipeline(files, config, workers)
            rows.append({'workers': workers, 'repeat': repeat,
                         'seconds': r['timed_cleaning_seconds'],
                         'rows_kept': r['rows_kept'], 'fingerprint': r['ordered_result_sha256'],
                         'warmup_fingerprint_equal': r['ordered_result_sha256'] == warm['ordered_result_sha256']})
            print(f'{workers} worker(s), repeat {repeat}: {r["timed_cleaning_seconds"]:.3f}s', flush=True)
    table = pd.DataFrame(rows)
    if table['fingerprint'].nunique() != 1 or not table['warmup_fingerprint_equal'].all():
        raise RuntimeError('Serial/parallel/repeat fingerprints differ')
    summary = table.groupby('workers', as_index=False).agg(
        mean_seconds=('seconds', 'mean'), std_seconds=('seconds', 'std'))
    baseline = float(summary.loc[summary['workers'] == 1, 'mean_seconds'].iloc[0])
    summary['speedup'] = baseline / summary['mean_seconds']
    summary['efficiency'] = summary['speedup'] / summary['workers']
    summary['all_outputs_equal'] = True
    return table, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--input-glob', help='Quoted file glob, e.g. /home/cs4207/data/yellow_tripdata_*.parquet')
    parser.add_argument('--output', default='outputs/local_jan2026')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    config, settings = read_config(args.config)
    pattern = args.input_glob or str(ROOT / settings['input_glob'])
    files = sorted((Path(p).resolve() for p in glob.glob(pattern)), key=lambda p: p.name)
    if not files:
        parser.error(f'No input files match {pattern}')
    for p in files:
        month_bounds(p.name)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    if args.benchmark:
        out.mkdir(parents=True, exist_ok=True)
        raw, summary = benchmark(files, config, args.repeats)
        raw.to_csv(out / 'benchmark_repeats.csv', index=False)
        summary.to_csv(out / 'benchmark_summary.csv', index=False)
        metadata = {'execution_location': 'local_development_only' if platform.system() == 'Windows' else 'Record hostname and confirm Rockhill before using in assessment',
                    'hostname': platform.node(), 'platform': platform.platform(), 'logical_cores': os.cpu_count(),
                    'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'pyarrow': pa.__version__,
                    'config': asdict(config), 'files': [p.name for p in files],
                    'timing_scope': 'Full raw read + batch conversion + pool startup + cleaning + ordered aggregation + output hashing. No clean export/sample writes. One full warm-up per worker count. Sequential order of benchmark configurations may leave cache/load bias.',
                    'interpretation': 'Cleaning/data-parallel benchmark only; NOT model speedup and NOT evidence of the two required task-parallel patterns or two nondeterminism sources.'}
        (out / 'benchmark_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(summary.to_string(index=False))
    else:
        r = run_pipeline(files, config, args.workers, out)
        (out / 'environment.json').write_text(json.dumps({'python': sys.version, 'platform': platform.platform(), 'hostname': platform.node(), 'logical_cores': os.cpu_count(), 'pandas': pd.__version__, 'numpy': np.__version__, 'pyarrow': pa.__version__}, indent=2), encoding='utf-8')
        print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()

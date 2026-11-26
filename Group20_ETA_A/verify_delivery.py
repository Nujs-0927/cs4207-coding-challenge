"""Verify delivered data and execute our own pure-Python notebook cells.

No external/untrusted notebooks are executed. No browser/server modifications.
"""
from pathlib import Path
import contextlib
import io
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import eta_data as eta
from handoff import load_sample, chronological_split, inputs_and_target
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def verify():
    out = ROOT / 'outputs/local_jan2026'
    manifest = json.loads((out / 'manifest.json').read_text())
    audit = pd.read_csv(out / 'cleaning_audit.csv')
    sample = load_sample()
    assert len(sample) == 100000
    assert sample['row_id'].is_unique
    assert sample['pickup_datetime'].is_monotonic_increasing
    assert sample['duration_minutes'].between(1, 180).all()
    assert sample['pickup_datetime'].between(pd.Timestamp('2026-01-01'), pd.Timestamp('2026-02-01'), inclusive='left').all()
    train, val, test = chronological_split(sample, '2026-01-22', '2026-01-27')
    assert train['pickup_datetime'].max() < val['pickup_datetime'].min()
    assert val['pickup_datetime'].max() < test['pickup_datetime'].min()
    X, y = inputs_and_target(train)
    assert not set(['duration_minutes', 'rate_code', 'fare_amount', 'trip_distance', 'row_id', 'source_file']).intersection(X.columns)
    assert len(X) == len(y)
    parts = sorted((out / 'cleaned_parts').glob('*.parquet'))
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    assert rows == manifest['result']['rows_kept']
    assert int(audit['removed'].sum()) + rows == manifest['result']['rows_in']
    bench = pd.read_csv(ROOT / 'outputs/local_benchmark/benchmark_repeats.csv')
    assert bench['fingerprint'].nunique() == 1
    assert bench['fingerprint'].iloc[0] == manifest['result']['ordered_result_sha256']
    assert bench.groupby('workers').size().to_dict() == {1: 3, 2: 3, 4: 3}
    assert bench['warmup_fingerprint_equal'].all()
    notebook = json.loads((ROOT / 'A_Data_Module.ipynb').read_text(encoding='utf-8'))
    scope = {'__name__': '__notebook__', 'DELIVERY_ROOT': ROOT}
    execution_count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        execution_count += 1
        stream = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(compile(''.join(cell['source']), f'A_Data_Module.ipynb:cell{execution_count}', 'exec'), scope)
        cell['execution_count'] = execution_count
        cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue().splitlines(keepends=True)}]
        cell['metadata']['verified_execution_seconds'] = round(time.perf_counter() - start, 4)
    (ROOT / 'A_Data_Module.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    report = {'status': 'PASS', 'sample_rows': len(sample), 'cleaned_rows': rows,
              'cleaned_part_files': len(parts), 'time_split_sample_rows': {'train': len(train), 'validation': len(val), 'test': len(test)},
              'benchmark_configurations': [1, 2, 4], 'repeats_per_configuration': 3,
              'all_cleaning_fingerprints_equal': True,
              'notebook_code_cells_executed_from_fresh_scope': execution_count,
              'scope': 'Local development verification, not Rockhill assessment evidence'}
    (ROOT / 'outputs/verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    verify()

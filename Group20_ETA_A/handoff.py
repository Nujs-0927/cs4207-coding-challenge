"""B/C/D: explicit input/target contract. Do not use df.drop(target) blindly."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LOCAL_RUNTIME = HERE.parent.parent / 'tmp' / 'group20-runtime'
if LOCAL_RUNTIME.is_dir():
    sys.path.insert(0, str(LOCAL_RUNTIME))
import pandas as pd

TARGET = 'duration_minutes'
ALLOWED_RAW_INPUTS = ['pickup_datetime', 'pickup_zone', 'dropoff_zone',
                      'passenger_count', 'passenger_count_missing']
NEVER_AUTO_INCLUDE = ['row_id', 'source_file', TARGET, 'rate_code', 'vendor_id']


def load_sample(path=None):
    """Parquet preserves categories/timestamps better than inferred CSV types."""
    path = Path(path) if path else HERE / 'outputs/local_jan2026/sample_100k.parquet'
    df = pd.read_parquet(path)
    if not df['row_id'].is_unique:
        raise ValueError('Non-unique row IDs')
    return df.sort_values(['pickup_datetime', 'row_id'], kind='stable').reset_index(drop=True)


def chronological_split(df, validation_start, test_start):
    """Explicit date boundaries, independent of CSV/Parquet part order."""
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    if validation_start >= test_start:
        raise ValueError('validation_start must precede test_start')
    time = pd.to_datetime(df['pickup_datetime'], errors='raise')
    if time.isna().any():
        raise ValueError('Missing pickup time')
    pieces = [df.loc[time < validation_start],
              df.loc[(time >= validation_start) & (time < test_start)],
              df.loc[time >= test_start]]
    if any(part.empty for part in pieces):
        raise ValueError('One split is empty; choose boundaries for your actual data range')
    return tuple(part.sort_values(['pickup_datetime', 'row_id'], kind='stable').copy() for part in pieces)


def inputs_and_target(df, intended_destination_known=True):
    """B derives calendar features and fits any learned preprocessing on TRAIN only.

    Actual recorded dropoff zone stands in for intended destination; disclose this
    assumption. For a pickup-only variant, set intended_destination_known=False.
    Passenger count may be unknown; preserve missing values until train-only imputation.
    """
    columns = list(ALLOWED_RAW_INPUTS)
    if not intended_destination_known:
        columns.remove('dropoff_zone')
    return df[columns].copy(), df[TARGET].astype('float64').copy()

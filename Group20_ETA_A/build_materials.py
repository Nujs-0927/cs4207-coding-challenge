"""Build A's local report, honest poster drafts, speech and integration notebook."""
from pathlib import Path
import html
import json
import os
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib_config'))
import eta_data as eta
import pandas as pd
import numpy as np
import pyarrow as pa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSS = '''
:root{--ink:#142a36;--teal:#087f78;--muted:#536670;--pale:#eaf4f2;--line:#d5e1e5}
*{box-sizing:border-box}body{font:16px/1.75 'Segoe UI','Microsoft YaHei',sans-serif;color:var(--ink);background:#f2f6f7;margin:0}
main{max-width:1120px;margin:32px auto;background:white;padding:42px 50px;border-radius:16px}h1{font-size:34px;line-height:1.3}h2{margin-top:36px;font-size:23px;color:var(--teal)}
p,li{max-width:980px}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}th{background:var(--pale)}
code,pre{font-family:Consolas,monospace}code{background:#edf2f4;padding:2px 5px}pre{background:#142a36;color:#e6f8f5;padding:18px;overflow:auto;border-radius:8px;font-size:13px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
.note{background:#fff3dc;border-left:5px solid #dd9b22;padding:15px 20px}.good{background:var(--pale);padding:18px;border-radius:10px}.small{font-size:13px;color:var(--muted)}a{color:#006d75}img{max-width:100%}.stats{display:flex;gap:16px;flex-wrap:wrap}.stat{background:var(--pale);padding:16px 24px;flex:1}.stat strong{display:block;font-size:28px;color:var(--teal)}
@media(max-width:650px){main{margin:12px;padding:20px}h1{font-size:27px}table{display:block;overflow-x:auto}.stats{gap:10px}.stat{min-width:130px;padding:12px}.stat strong{font-size:25px}pre{font-size:12px}}
@media print{body{background:white}main{margin:0;padding:12mm;max-width:none}h2{break-after:avoid}table,figure,.note{break-inside:avoid}a{color:inherit}}
'''


def document(title, body):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style><main>{body}</main></html>'


def write_notebook():
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': text.splitlines(keepends=True), 'execution_count': None, 'outputs': []})
    md('''# Group 20 — A: data preparation for ETA prediction

This is A's data module, not the complete group submission. Local development uses the complete January 2026 Yellow Taxi file. Re-run the agreed scope on Rockhill before using results as assessment evidence.

Target: duration in minutes. Prediction setting: at pickup, assuming intended destination zone is known. Actual recorded dropoff zone is a proxy for this assumption. The cleaned population excludes fixed invalid/extreme-trip cases; report this selection limitation.
''')
    code('''from pathlib import Path
import sys, json
if 'DELIVERY_ROOT' in globals():
    ROOT = Path(DELIVERY_ROOT)
else:
    candidates = [Path.cwd(), Path.cwd() / 'CS4207/Group20_ETA_A', Path.cwd() / 'Group20_ETA_A']
    ROOT = next((p for p in candidates if (p / 'eta_data.py').exists()), None)
    if ROOT is None:
        raise FileNotFoundError('Open this notebook in the Group20_ETA_A folder.')
sys.path.insert(0, str(ROOT))
from eta_data import read_config, run_pipeline, RULES
from handoff import load_sample, chronological_split, inputs_and_target
import pandas as pd
config, settings = read_config(ROOT / 'config.json')
files = sorted((ROOT / 'data/raw').glob('yellow_tripdata_2026-01.parquet'))
if not files:
    raise FileNotFoundError('Place the raw January file in data/raw, or point files to the agreed Rockhill data files.')
print('Input files:', [p.name for p in files])
print('Config:', config)
''')
    md('''## 1. Re-run the actual cleaning code

The following cell executes the full cleaning pipeline again, rather than displaying invented or cached counts. Existing exports are only reused after their fingerprint is checked. For a fresh export, set `export_path` to a new path and run `run_pipeline(files, config, workers=1, output=export_path)`.
''')
    code('''export_path = ROOT / 'outputs/local_jan2026'
if not (export_path / 'manifest.json').exists():
    result = run_pipeline(files, config, workers=1, output=export_path)
else:
    result = run_pipeline(files, config, workers=1)
    saved = json.loads((export_path / 'manifest.json').read_text(encoding='utf-8'))
    assert result['ordered_result_sha256'] == saved['result']['ordered_result_sha256'], 'Raw data/config/code changed: make a fresh export.'
print(json.dumps(result, indent=2))
''')
    md('''## 2. Ordered cleaning audit

Rows are removed at their first failing rule, so removal counts do not overlap. Thresholds (1–180 minutes, at most 100 miles, at most 100 mph) are explicit project choices, not lecturer-prescribed constants. Do not drop rows simply because passenger count or rate code is missing.
''')
    code('''remaining = result['rows_in']
rows = [{'rule': 'Input rows', 'removed': 0, 'remaining': remaining}]
for key, label in RULES:
    remaining -= result['removed'][key]
    rows.append({'rule': label, 'removed': result['removed'][key], 'remaining': remaining})
audit = pd.DataFrame(rows)
print(audit.to_string(index=False))
assert remaining == result['rows_kept']
print('Kept Flex Fare rows:', result['flex_fare_rows_kept'])
''')
    md('''## 3. Deterministic development sample and schema

100,000 retained trips are selected across the whole month using a fixed row-key hash and seed, then sorted by pickup time. This is not the final test set. Stable row IDs preserve alignment across parallel feature tasks. Full exported parts are not globally chronological: downstream splitting must use timestamps.
''')
    code('''sample = load_sample(export_path / 'sample_100k.parquet')
print('Sample rows:', len(sample))
print(sample.head(5).to_string(index=False))
print(sample.dtypes.to_string())
print('Missing values:\\n', sample.isna().sum().to_string())
assert sample['row_id'].is_unique
assert sample['pickup_datetime'].is_monotonic_increasing
''')
    md('''## 4. B/C handoff and leakage exclusions

These January boundaries are an integration example, not a team-approved final split. Change them together when agreeing data scope. Fit imputation, category encoders, scaling and model parameters on training data only. `rate_code` is a final-at-end-of-trip audit category; never auto-include it. Raw distance, speed, fare, tip, payment and dropoff time are not exported as model inputs.
''')
    code('''train, validation, test = chronological_split(sample, '2026-01-22', '2026-01-27')
X_train_raw, y_train = inputs_and_target(train, intended_destination_known=True)
print('Illustrative split sizes:', len(train), len(validation), len(test))
print('Allowed raw X columns:', list(X_train_raw.columns))
print('Target:', y_train.name)
assert train['pickup_datetime'].max() < validation['pickup_datetime'].min()
assert validation['pickup_datetime'].max() < test['pickup_datetime'].min()
''')
    md('''## 5. Local data-parallel benchmark only

One full warm-up per worker configuration, then at least three repeats of identical work. This measures raw reading, cleaning, ordered aggregation and output hashing; it is not training speedup. Local Windows results must not be labelled Rockhill results. This module does not complete the group's two task-parallel patterns or two observed nondeterminism/fix experiments.
''')
    code('''benchmark_path = ROOT / 'outputs/local_benchmark/benchmark_summary.csv'
if benchmark_path.exists():
    print(pd.read_csv(benchmark_path).to_string(index=False))
else:
    print('Run: python eta_data.py --benchmark --output outputs/local_benchmark --repeats 3')
print('Server command: srun --partition=cs4207 python eta_data.py --input-glob "/home/cs4207/data/yellow_tripdata_2026-01.parquet" --output outputs/rockhill_run --workers 1')
''')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.13'}, 'verification': 'Pure-Python code cells executed in a fresh namespace by verify_delivery.py; also compatible with Jupyter.'}, 'nbformat': 4, 'nbformat_minor': 5}
    (ROOT / 'A_Data_Module.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')


def main():
    data = ROOT / 'outputs/local_jan2026'
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    r = manifest['result']
    audit = pd.read_csv(data / 'cleaning_audit.csv')
    bench = pd.read_csv(ROOT / 'outputs/local_benchmark/benchmark_summary.csv')
    raw, kept = r['rows_in'], r['rows_kept']
    rejected, retain_pct = raw - kept, 100 * kept / raw
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    nonzero = audit.loc[audit.removed > 0].copy()
    short_labels = {
        'pickup_outside_file_month': 'Pickup outside file month',
        'non_positive_duration': 'Duration <= 0 min',
        'duration_outside_scope': 'Duration outside 1–180 min',
        'missing_timestamps': 'Missing timestamps',
        'missing_or_nonfinite_distance': 'Missing / non-finite distance',
        'non_positive_distance': 'Distance <= 0 miles',
        'distance_above_limit': 'Distance > 100 miles',
        'missing_or_nonfinite_fare': 'Missing / non-finite fare',
        'non_positive_fare': 'Fare <= 0 dollars',
        'speed_above_limit': 'Speed > 100 mph',
    }
    labels = [short_labels.get(step, step) for step in nonzero['step']]
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    bars = ax.barh(labels, nonzero['removed'], color='#087f78')
    ax.set_xscale('log'); ax.invert_yaxis()
    ax.set_xlabel('Trips removed at first failing rule (log scale)')
    ax.set_title('Group 20 · January 2026 cleaning audit\nLOCAL DEVELOPMENT — rerun on Rockhill', fontsize=13)
    for bar, n in zip(bars, nonzero['removed']):
        ax.annotate(f'{n:,}', (bar.get_width(), bar.get_y() + bar.get_height()/2), xytext=(5, 0), textcoords='offset points', va='center', fontsize=10)
    ax.set_xlim(1, nonzero.removed.max() * 2.1)
    ax.grid(axis='x', alpha=.15)
    fig.savefig(figures / 'A_cleaning_removals.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.plot(bench.workers, bench.speedup, 'o-', color='#087f78', linewidth=2, label='Measured cleaning pipeline')
    ax.plot([1, 2, 4], [1, 2, 4], '--', color='#9aa8af', label='Ideal')
    ax.set(xticks=[1, 2, 4], xlabel='Workers', ylabel='Speedup', title='LOCAL WINDOWS DEVELOPMENT — not Rockhill\nData cleaning only · 3 repeats after warm-up')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(figures / 'A_local_cleaning_speedup.png', dpi=180); plt.close(fig)
    count_table = audit.to_html(index=False, border=0)
    performance_table = bench.to_html(index=False, border=0, float_format=lambda x: f'{x:.3f}')
    poster = f'''<p class="small">GROUP 20 · ETA PREDICTION · MEMBER A</p><h1>Problem, dataset &amp; cleaning</h1>
<p class="note"><strong>Local development results.</strong> January 2026 only. Re-run the same configuration on Rockhill and replace counts if necessary before the assessed poster is finalised.</p>
<p>Predict taxi journey duration in minutes from information available at pickup, assuming the intended destination zone is known.</p>
<div class="stats"><div class="stat"><strong>{raw:,}</strong>raw trips</div><div class="stat"><strong>{kept:,}</strong>retained trips</div><div class="stat"><strong>{retain_pct:.2f}%</strong>retained</div><div class="stat"><strong>100,000</strong>development sample</div></div>
<h2>Data preparation</h2><ul><li>Read selected Parquet columns in 100,000-row batches.</li><li>Derive the label from dropoff minus pickup time, in minutes.</li><li>Keep in-month pickups; remove non-positive or extreme duration/distance, invalid fares, and implausible observed speeds.</li><li>Project thresholds: 1–180 minutes, distance ≤100 miles, speed ≤100 mph. Fixed scope choices, with selection limitations.</li><li>Retain missing passenger count and rate code; mark missing categories explicitly. {r['flex_fare_rows_kept']:,} Flex Fare trips are retained.</li></ul>
<h2>Leakage &amp; reproducibility</h2><p>Dropoff time, realised distance, speed, fares, tips, payment and final rate code are excluded from the pickup-time feature whitelist. Use time-based train/validation/test splits and train-only learned preprocessing. Stable row IDs and fixed-order shard aggregation preserve alignment.</p>
<img src="figures/A_cleaning_removals.png" alt="Actual sequential cleaning removal counts"><p class="small">Source: NYC TLC January 2026 Yellow Taxi file. Counts are ordered and non-overlapping. Actual recorded dropoff zone is used only under an intended-destination assumption. This evaluation population excludes invalid/extreme observed trips.</p>'''
    (ROOT / 'A_Poster_Section_EN.html').write_text(document('Group 20 — A poster section', poster), encoding='utf-8')
    speech = f'''Group 20 — Member A — English speaking draft (local-development wording)

Hello, we are Group Twenty. Our project predicts taxi journey duration in minutes. We assume the intended destination is known at pickup.

For local development, we used the complete January twenty twenty-six Yellow Taxi file from NYC TLC, containing about {raw / 1e6:.2f} million trips. We calculated duration from the pickup and dropoff timestamps, and read the data in batches to control memory use.

Our cleaning rules remove out-of-month pickups, invalid durations and distances, non-positive fares, and implausible speeds. The fixed thresholds define our study population. We retained about {kept / 1e6:.2f} million trips, or {retain_pct:.1f} percent, while preserving common missing passenger and rate-code values.

We provided a reproducible hundred-thousand-trip sample. Dropoff time, actual distance and fares are not model inputs. These local results must be verified on Rockhill. Next, our team will explain the features and models.

Delivery notes: approximately 130–140 words. Rehearse with a timer and aim for 65–70 seconds to leave room for speaker transitions. Once the Rockhill run is complete, update the data scope and figures, then replace the local-development wording with an accurate account of the server run.
Do not claim that the group's models, task-parallel implementations or final experiments are complete before they have actually been completed.
'''
    (ROOT / 'A_Speaking_EN.txt').write_text(speech, encoding='utf-8')
    handoff = f'''GROUP 20 — A DATA HANDOFF
Status: real LOCAL development results; no Rockhill execution or website submission performed.
Data: NYC TLC yellow_tripdata_2026-01.parquet only, not the whole server collection.
Raw {raw:,}; retained {kept:,}; removed {rejected:,}; development sample 100,000.

FILES
outputs/local_jan2026/sample_100k.parquet — recommended sample, preserves types.
outputs/local_jan2026/sample_100k.csv — alternative interchange file.
outputs/local_jan2026/cleaned_parts/ — full clean month (local full folder only, excluded from the lightweight team ZIP).
outputs/local_jan2026/manifest.json — thresholds, source SHA256, data contract and provenance.
outputs/local_jan2026/cleaning_audit.csv — first-failure, non-overlapping counts.
A_Data_Module.ipynb — A's executed integration notebook, not the complete group notebook.
handoff.py — safe column selection and explicit date-based split helpers.

CONTRACT
row_id: unique stable source-file + original row offset; metadata only.
pickup_datetime: timestamp; use for chronology and calendar features.
duration_minutes: regression target, NEVER in X.
pickup_zone / dropoff_zone: category strings; dropoff assumes intended destination known.
passenger_count: numeric, can be missing; zero is retained as an uncertain recorded value.
passenger_count_missing: missing indicator. Fit any imputation on TRAIN ONLY.
rate_code: explicit category, final-at-trip-end audit metadata, EXCLUDE from X.
vendor_id: metadata, excluded by default.
Missing zone/rate values use __MISSING__; invalid/unknown values use __UNKNOWN__.
TLC codes 264/265 remain explicit special zone codes; do not assume they identify ordinary named zones.
Recorded timestamps are treated as TLC's local wall-clock values; no unverified UTC conversion is applied.

IMPORTANT
Never build X with df.drop('duration_minutes') alone. Use inputs_and_target().
Actual distance, speed, fare, tip, payment and dropoff time are deliberately absent from prediction data.
Distance/fare/dropoff are used retrospectively for quality checks and labels: report selection limitations.
Do not randomly split. Sort/split by pickup_datetime; full Parquet part order is NOT time order.
The 100k sample covers the retained month and is NOT an independent test set.
For a January-only integration example use validation >= Jan 22, test >= Jan 27; group C must approve final boundaries.
No learned imputer, scaler or one-hot encoder is fitted by A.
Do not remove duplicates merely because time/zone values match: there is no reliable trip ID.

GROUP GAPS TO ASSIGN
The proposed A/B/C/D plan has not explicitly assigned: two task-parallel patterns, two real nondeterminism/fix experiments, and fair speedup evidence for the chosen workloads.
Two people training different models simultaneously is not by itself evidence of coded task parallelism.
A's code covers data-parallel cleaning and deterministic aggregation only.
D using Rockhill does not exempt other official experiments from the server requirement.
Four talks of exactly 75 seconds leave zero transition margin; target 65–70 seconds each.

ROCKHILL
Use only your allocated Group 20 account. Do not share passwords in this package.
Run python in the group's activated environment via cs4207 partition; do not request GPU numbers or use student partition.
Run run_rockhill.sh after agreeing months and verifying four CPUs are available for the allocation.
Replace LOCAL counts/plots/speech only with actually generated Rockhill evidence.

DEADLINES (2026)
Poster upload: end of Nov 26. Presentation: Nov 27, 14:00–16:00, maximum five minutes.
Final notebook: Nov 29, 23:59. Actual upload destination should be checked; no upload has been made.
'''
    (ROOT / 'TEAM_HANDOFF_EN.txt').write_text(handoff, encoding='utf-8')
    rules = '''<table><tr><th>Rule</th><th>Current default / treatment</th><th>Rationale and limitations</th></tr>
<tr><td>File month</td><td>Keep pickups within the file month; dropoffs may cross into the next month</td><td>Assign trips by pickup time and remove out-of-month spillover records</td></tr>
<tr><td>Duration</td><td>1–180 minutes, inclusive</td><td>Exclude invalid or extreme trips; this fixed study scope needs team agreement</td></tr>
<tr><td>Distance</td><td>Greater than 0 and at most 100 miles</td><td>Exclude meter errors and atypical long trips; realised distance is used only for retrospective quality checks</td></tr>
<tr><td>Fare</td><td>Finite and greater than 0</td><td>Exclude refunds, free trips and invalid values; disclose the resulting population restriction</td></tr>
<tr><td>Average speed</td><td>At most 100 mph (approximately 161 km/h)</td><td>Computed from observed distance and duration, solely as a broad quality screen</td></tr>
<tr><td>Missing passenger count</td><td>Retain missing values; convert negative, above-nine or non-finite values to missing; retain zero as an uncertain recorded value</td><td>Do not discard all Flex Fare trips; B/C must fit any imputation on training data only</td></tr>
<tr><td>Zones and rate codes</td><td>Preserve string categories and explicit missing/unknown markers</td><td>Final rate_code is audit metadata only; B/C learn feature encodings on training data</td></tr>
<tr><td>Duplicate-looking records</td><td>Do not deduplicate on time or zone values alone</td><td>Distinct real journeys can share identical recorded attributes</td></tr></table>'''
    guide = f'''<p class="small">CS4207 · GROUP 20 · MEMBER A · 2026-11-25</p><h1>Member A: responsibilities and local delivery</h1>
<p class="good"><strong>A's responsibilities:</strong> prepare documented, audited clean data for the team; provide the matching poster section and a short speaking part; collect the other members' material, format the complete poster and upload it.</p>
<p class="note"><strong>Current status:</strong> the complete January 2026 file has been processed locally. These are real local results, not Rockhill assessment evidence or the group's model results. B/C/D's work, server execution and website submissions are still outstanding.</p>
<div class="stats"><div class="stat"><strong>{raw:,}</strong>raw records</div><div class="stat"><strong>{kept:,}</strong>clean records</div><div class="stat"><strong>{retain_pct:.2f}%</strong>retained</div><div class="stat"><strong>100,000</strong>team development sample</div></div>
<h2>1. Next steps for A</h2><ol><li>Read this guide and open A_Data_Module.ipynb; understand every cleaning rule and reported count.</li><li>Send Group20_A_TEAM_HANDOFF.zip to B/C/D and agree the target, field contract and prediction assumptions.</li><li>Confirm the thresholds and data months. The current scope is January only, not the entire server dataset.</li><li>Re-run on Rockhill using the allocated group account and save the actual environment, counts, timings and equality checks.</li><li>Add the material from A_Poster_Section_EN.html and the cleaning figure to the group poster.</li><li>Collect B's EDA/features, C's methodology/baselines and D's measured model results; replace every placeholder before submission.</li><li>Rehearse A's speaking part for approximately 65–70 seconds and upload the complete poster by the end of 26 November.</li></ol>
<h2>2. Target definition and leakage prevention</h2><p>The target <code>duration_minutes</code> is the elapsed seconds between pickup and dropoff divided by 60. The prediction setting assumes the intended destination zone is known at pickup. Recorded dropoff zone is a proxy for that assumption. For a pickup-only variant, use <code>inputs_and_target(..., intended_destination_known=False)</code>.</p>
<p>Allowed raw inputs are pickup time (B derives calendar features), pickup zone, destination zone under the stated assumption, passenger count and its missing indicator. <strong>Do not use every non-target column as X.</strong> row_id, source_file, final rate_code and vendor_id are metadata by default.</p>
<p>Raw dropoff time, realised distance, speed, fares, tips and payment type are absent from the prediction-data handoff. They are used only for labels or retrospective quality checks. Report this selection limitation; do not claim coverage of every real departure-time trip. Removing extreme durations also changes the target distribution.</p>
<h2>3. Cleaning rules: agree any changes as a team</h2>{rules}<p>These thresholds define the documented default study scope; they are not lecturer-prescribed constants. After changing <code>config.json</code>, run into a new output directory and update every affected poster figure.</p>
<h2>4. Measured cleaning audit</h2>{count_table}<p>Apply rules in the order shown. Each record is counted at its first failing rule, so removals do not overlap. Removed: {rejected:,}; retained: {kept:,}. Retained Flex Fare records with missing passenger count/rate code: {r['flex_fare_rows_kept']:,}.</p><img src="figures/A_cleaning_removals.png" alt="Measured cleaning removal counts">
<h2>5. Handoff to B/C/D</h2><pre>from handoff import load_sample, chronological_split, inputs_and_target
df = load_sample()
# January development example only; C and the team must agree final dates
train, val, test = chronological_split(df, '2026-01-22', '2026-01-27')
X_train_raw, y_train = inputs_and_target(train)
# B/C derive time features and fit imputation/encoding on training data only.</pre>
<p>Prefer the Parquet sample to avoid CSV type-inference problems. The 100,000-row sample uses a fixed seed and row-key hash across all retained trips in the month, followed by sorting by pickup time. It is neither the first 100,000 records nor an independent final test set. Full cleaned_parts preserve source-file/source-row order, <strong>not global chronological order</strong>; split on timestamps.</p>
<h2>6. Local and server execution</h2><p>In the original workspace, dependencies are installed in tmp/group20-runtime and detected automatically. On another computer or server, create a virtual environment and install requirements.txt. The lightweight ZIP excludes raw files and full cleaned parts; to rebuild, place the agreed raw file in data/raw or use the Rockhill path.</p><pre>python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python eta_data.py --output outputs/my_new_run --workers 1
python eta_data.py --benchmark --output outputs/my_benchmark --repeats 3
python build_materials.py
python verify_delivery.py</pre>
<p>build_materials.py and verify_delivery.py target the verified local_jan2026 delivery. For another run, explicitly use its corresponding outputs instead of relabelling this local report. Data export refuses to overwrite an existing output directory.</p>
<pre># Rockhill: use the allocated group account and activate the group environment
# January example only; change the months after agreeing the scope
srun --partition=cs4207 python eta_data.py \
  --input-glob '/home/cs4207/data/yellow_tripdata_2026-01.parquet' \
  --output outputs/rockhill_january --workers 1

# After confirming the allocation can provide four CPUs:
bash run_rockhill.sh</pre><p>Do not request a GPU number or use the student partition. Check the allocated CPU count before running four workers. This script is an execution entry point; no server login, password change or website upload has been performed for this delivery.</p>
<h2>7. Data-parallel evidence included</h2><p>The same function processes separate batches, followed by fixed-order shard aggregation. A bounded queue avoids loading the entire dataset. Local runs use 1/2/4 workers, one warm-up per configuration and three measured repeats; all output fingerprints match. Timing includes reading, pool startup, cleaning, aggregation and fingerprinting, but excludes sample selection and export writes.</p>{performance_table}<img src="figures/A_local_cleaning_speedup.png" alt="Local data-cleaning speedup"><p>This measures cleaning-pipeline performance, not model training. Process startup, serialisation, I/O and small-batch overhead limit speedup. Do not label these local Windows measurements as Rockhill results. The team still needs the formal performance interpretation, including Amdahl's law.</p>
<h2>8. Remaining group responsibilities</h2><p class="note">The original allocation covers EDA and different models but does not explicitly assign <strong>two task-parallel patterns, two observed nondeterminism sources with verified fixes, and fair sequential/parallel comparisons</strong>. A includes data-parallel cleaning. Assign parallel feature extraction and parallel hyperparameter search, plus owners for the nondeterminism experiments. C and D working on separate models at the same time does not by itself demonstrate coded task parallelism.</p>
<h2>9. Poster and speaking materials</h2><ul><li><a href="A_Poster_Section_EN.html">A's poster text and cleaning figure</a>: copy the relevant material into the final poster.</li><li><a href="Poster_Group20_DRAFT.html">Group poster layout draft</a>: B/C/D sections are explicitly marked as missing; this is not ready for submission.</li><li><a href="A_Speaking_EN.txt">A's English speaking draft and delivery notes</a>: update the local-development wording once actual server results are available.</li><li><a href="TEAM_HANDOFF_EN.txt">Team handoff notes</a>: field contract and outstanding rubric items.</li></ul>
<h2>10. Deadlines and final assembly</h2><p>2026 deadlines: poster upload by the end of 26 November; presentation on 27 November, 14:00–16:00, with a strict five-minute limit; final notebook by 29 November, 23:59. Four 75-second talks leave no transition margin, so target 65–70 seconds per speaker.</p><p>Before submission, check readable figures, clear units and labels, notebook evidence for every result, accurate data scope and environment, replacement of all placeholders, and the actual upload destination. This A-only package contains no final model scores or group conclusions.</p>
<h2>11. Delivery contents</h2><ul><li>eta_data.py / config.json: cleaning and data-parallel entry point.</li><li>A_Data_Module.ipynb: A's integration notebook; verify_delivery.py re-executes its code cells.</li><li>handoff.py: safe input selection and chronological split helpers.</li><li>outputs/local_jan2026/cleaned_parts: full cleaned January data, available only in the original local folder.</li><li>outputs/local_jan2026/sample_100k.csv / .parquet: 100,000-row development sample.</li><li>Manifest, cleaning audit, environment, benchmark and verification files: provenance and evidence.</li><li>Group20_A_TEAM_HANDOFF.zip: lightweight team package; raw files and full clean partitions are excluded.</li></ul>
<h2>12. Sources</h2><p><a href="{eta.SOURCE_PAGE}">NYC TLC official data page</a> · <a href="{eta.DATA_DICTIONARY}">Yellow Taxi data dictionary</a>. Course sources: CS4207 Coding Challenge Rubric, Week 10 Session 1, and the 19/23 November announcements. The manifest records the public raw file's SHA256; byte-for-byte equality with the server file has not yet been checked.</p>'''
    (ROOT / 'START_HERE_A.html').write_text(document('Group 20 — Member A delivery guide', guide), encoding='utf-8')
    draft = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>GROUP 20 — NOT FOR SUBMISSION</title><style>{CSS}
main{{max-width:1700px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:24px}}.panel{{padding:20px;border:1px solid var(--line);border-radius:10px}}.placeholder{{border:2px dashed #d89c3b;background:#fff9ec;min-height:220px;padding:20px}}h2{{font-size:21px}}@media(max-width:1000px){{.grid{{grid-template-columns:1fr 1fr}}}}@media print{{.grid{{grid-template-columns:repeat(4,1fr)}}body{{font-size:11px}}}}
</style><main><p class="note"><strong>DRAFT — NOT FOR SUBMISSION.</strong> A has real local data results. B/C/D content and Rockhill validation are missing. Final dimensions/format must follow the lecturer's instructions; this is an editable layout starter.</p><h1>Predicting Taxi Journey Duration</h1><p>CS4207 · Group 20 · NYC Yellow Taxi · Regression · Member names: [add all four names]</p><div class="grid">
<section class="panel"><h2>A · Problem &amp; data</h2><p>Estimate duration in minutes at pickup, assuming intended destination zone is known.</p><p><strong>{raw:,}</strong> January 2026 raw trips → <strong>{kept:,}</strong> retained locally.</p><p>Batch reading; in-month pickups; valid duration, distance and fare; speed quality check. Explicit missing-value handling.</p><img src="figures/A_cleaning_removals.png" alt="A actual local cleaning counts"><p>Dropoff time, fare and realised distance are excluded from X. Verify counts on Rockhill.</p></section>
<section class="panel"><h2>B · EDA &amp; features</h2><div class="placeholder">AWAITING B<br>Duration distribution; pickup time and zone insights; approved feature list and leakage exclusions.<br>No invented charts.</div></section>
<section class="panel"><h2>C · Method &amp; baseline</h2><div class="placeholder">AWAITING C<br>Time-based train/validation/test boundaries; model and shared MAE/RMSE/R² evaluation; actual baseline results.</div><h2>Parallel design</h2><div class="placeholder">AWAITING TEAM<br>Pipeline diagram: data parallelism + TWO implemented task-parallel patterns.</div></section>
<section class="panel"><h2>D · Results &amp; conclusion</h2><div class="placeholder">AWAITING D<br>Actual Rockhill model results; repeated sequential/parallel timings and speedup; two observed nondeterminism sources and verified fixes; limitations.</div></section>
</div><p class="small">Source: NYC TLC. Every final result must be traceable to the solution notebook. No final model conclusions have been generated in this A-only package.</p></main></html>'''
    (ROOT / 'Poster_Group20_DRAFT.html').write_text(draft, encoding='utf-8')
    versions = f'pandas=={pd.__version__}\nnumpy=={np.__version__}\npyarrow=={pa.__version__}\nmatplotlib=={matplotlib.__version__}\n'
    (ROOT / 'requirements-local-tested.txt').write_text(versions, encoding='utf-8')
    if not (ROOT / 'A_Data_Module.ipynb').exists():
        write_notebook()
    print(f'Built report, poster drafts, speech, handoff, figures and notebook: {ROOT}')


if __name__ == '__main__':
    main()

"""Builds D_Poster_Section_EN.html and D_Speaking_EN.txt from one run's d_summary.json,
so every number on the poster is traceable to the executed notebook."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSS = """:root{--ink:#142a36;--teal:#087f78;--muted:#536670;--pale:#eaf4f2;--line:#d5e1e5}
*{box-sizing:border-box}body{font:16px/1.75 'Segoe UI','Microsoft YaHei',sans-serif;color:var(--ink);background:#f2f6f7;margin:0}
main{max-width:1120px;margin:32px auto;background:white;padding:42px 50px;border-radius:16px}h1{font-size:34px;line-height:1.3}h2{margin-top:36px;font-size:23px;color:var(--teal)}
p,li{max-width:980px}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{background:var(--pale)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}tr.d td{font-weight:600}
code{background:#edf2f4;padding:2px 5px;font-family:Consolas,monospace}
.note{background:#fff3dc;border-left:5px solid #dd9b22;padding:15px 20px}.small{font-size:13px;color:var(--muted)}img{max-width:100%}.stats{display:flex;gap:16px;flex-wrap:wrap}.stat{background:var(--pale);padding:16px 24px;flex:1}.stat strong{display:block;font-size:28px;color:var(--teal)}
.figs{display:grid;grid-template-columns:1fr 1fr;gap:18px}.figs figure{margin:0}.figs .wide{grid-column:1/-1}figcaption{font-size:13px;color:var(--muted)}
@media(max-width:650px){main{margin:12px;padding:20px}h1{font-size:27px}.figs{grid-template-columns:1fr}.stat{min-width:130px;padding:12px}table{display:block;overflow-x:auto}}
@media print{body{background:white}main{margin:0;padding:12mm;max-width:none}figure,table,.note{break-inside:avoid}}"""

NUM = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}


def say(x):
    """3.89 -> 'three point nine' style for the script (one decimal)."""
    whole, dec = f'{x:.1f}'.split('.')
    w = NUM.get(int(whole), whole)
    return w if dec == '0' else f'{w} point {NUM.get(int(dec), dec) if dec != "0" else "zero"}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='local_sample')
    a = ap.parse_args()
    out = HERE / 'outputs' / a.run
    s = json.loads((out / 'd_summary.json').read_text())
    b, g = s['best_model'], s['improvement']
    best = b['name']
    board = {r['model']: r for r in s['scoreboard']}
    ref = s['reference_models']
    lin, zp = board[ref['best_linear']], board[ref['zone_pair']]
    rows_all = sum(s['rows'].values())
    local = not a.run.startswith('rockhill')
    o = s['nondeterminism_rf_prediction']
    test_days = {k: v for k, v in s['daily_MAE_best'].items() if k >= '2026-01-27'}
    val_days = {k: v for k, v in s['daily_MAE_best'].items() if k < '2026-01-27'}
    worst_day = max(test_days, key=test_days.get)
    buckets = s['test_error_by_duration']
    long = buckets[-1]
    repro_ok = all(r[0]['fingerprint'] == r[1]['fingerprint'] == r[2]['fingerprint'] for r in s['reproducibility'].values())
    params = ', '.join(f'{k} = {v}' for k, v in b['params'].items())

    # figures next to the page (same layout as B/C: figures/…)
    fig_dir = HERE / 'figures'
    fig_dir.mkdir(exist_ok=True)
    for f in (out / 'figures').glob('*.png'):
        shutil.copy2(f, fig_dir / f.name)

    show = ['Constant (train median)', ref['zone_pair'], ref['best_linear'], 'Random Forest', 'XGBoost', 'LightGBM']
    trs = ''.join(
        f'<tr class="{"d" if m in ("Random Forest", "XGBoost", "LightGBM") else ""}"><td>{html.escape(m)}{" ★" if m == best else ""}</td>'
        + ''.join(f'<td class="n">{board[m][c]:.2f}</td>' for c in ('validation_MAE', 'validation_R2', 'test_MAE', 'test_R2', 'test_bias'))
        + '</tr>' for m in show)
    table = ('<table><tr><th>Model</th><th class="n">Val MAE</th><th class="n">Val R²</th><th class="n">Test MAE</th>'
             f'<th class="n">Test R²</th><th class="n">Test bias</th></tr>{trs}</table>')
    scope = (f"Local development run on A's {rows_all:,}-trip sample (train {s['rows']['train']:,} · validation "
             f"{s['rows']['validation']:,} · test {s['rows']['test']:,}); {s['environment']['threads_per_model']} threads, "
             f"{s['environment']['platform'].split('-')[0]}." if local else
             f"Rockhill run on {rows_all:,} cleaned trips (train {s['rows']['train']:,} · validation {s['rows']['validation']:,} · "
             f"test {s['rows']['test']:,}); host {s['environment']['host']}, {s['environment']['threads_per_model']} threads.")
    note = ('<p class="note"><strong>Local development results.</strong> Re-run on Rockhill with the full-month features from B and '
            'C\'s full-month scoreboard, then regenerate this page with <code>make_d_materials.py</code>.</p>') if local else ''

    body = f'''<p class="small">GROUP 20 · ETA PREDICTION · MEMBER D</p><h1>Results &amp; conclusion</h1>
{note}<p class="small">{scope} All numbers from <code>D_Advanced_Models.ipynb</code>, scored with C's shared <code>evaluate()</code>.</p>
<div class="stats"><div class="stat"><strong>{b['validation_MAE']:.2f} min</strong>validation MAE, {best}</div><div class="stat"><strong>−{g['validation_vs_best_linear_pct']:.0f}%</strong>MAE vs best linear model</div><div class="stat"><strong>{b['test_MAE']:.2f} min</strong>test MAE (after the snowstorm)</div><div class="stat"><strong>{100 * s['importance_location_share']:.0f}%</strong>of split gain from location</div></div>
<h2>Best model and comparison</h2>
<p>Random Forest, LightGBM and XGBoost on B's {s['n_features']} pickup-time features, trained on <code>log1p(duration)</code>.
Each has a small grid tuned on <strong>validation</strong> (boosters early-stop on validation loss); test is scored once.
<strong>{best}</strong> ({params}) wins on validation.</p>
{table}
<figure><img src="figures/D_model_comparison.png" alt="Validation and test MAE for every model"><figcaption>MAE of all group models on the same split (C's baselines and linear models, D's tree models in bold).</figcaption></figure>
<ul><li>Trees beat the best linear model by <strong>{g['validation_vs_best_linear_pct']}%</strong> (validation) and <strong>{g['test_vs_best_linear_pct']}%</strong> (test): they learn zone × time interactions that the linear model needs hand-made products for.</li>
<li>Against the zone-pair median baseline: <strong>{g['validation_vs_zone_pair_pct']}%</strong> lower validation MAE.</li></ul>
<h2>What drives the prediction, and where it fails</h2>
<div class="figs"><figure><img src="figures/D_feature_importance.png" alt="Feature importance"><figcaption>{best}: {100 * s['importance_location_share']:.0f}% of the gain comes from zone encodings, zone categories and airport/same-zone flags.</figcaption></figure>
<figure><img src="figures/D_pred_vs_actual.png" alt="Predicted against actual duration"><figcaption>Test set: long trips are under-predicted (trips over {float(long['bucket'].split(',')[0].strip('(')):g} min: MAE {long['MAE']:.1f}, bias {long['bias']:.1f} min).</figcaption></figure></div>
<h2>Limitations</h2><ul>
<li><strong>Distribution shift:</strong> after the 25 Jan snowstorm, daily test MAE reaches {test_days[worst_day]:.1f} min ({worst_day}) versus {min(val_days.values()):.1f}–{max(val_days.values()):.1f} on validation days; test bias {b['test_bias']:.1f} min. Split dates were fixed before seeing this.</li>
<li><strong>No live traffic or weather</strong> input, and the recorded dropoff zone stands in for the intended destination.</li>
<li><strong>Restricted population:</strong> A's 1–180 min, ≤100 mile, ≤100 mph filters; January only.</li></ul>
<h2>Reproducibility</h2>
<p><strong>Non-determinism found in Part D:</strong> the Random Forest's parallel <code>predict()</code> adds the tree outputs in whatever order threads finish — {o[0]['distinct_outputs']} different outputs in {o[0]['calls']} calls (up to {o[0]['max_abs_diff_min']:.0e} min). Fix: one prediction thread (fixed summation order) → {o[1]['distinct_outputs']} of {o[1]['calls']}.
With fixed seeds, all three models give {'identical' if repro_ok else 'DIFFERENT'} test predictions across refits and thread counts.</p>
<h2>Conclusion &amp; future work</h2>
<p>Where a trip goes matters far more than when it starts. Gradient-boosted trees on leakage-free pickup-time features give the best ETA ({b['validation_MAE']:.2f} min validation MAE).
Next: add weather and incident data to handle events like the storm, train on more months, and retrain regularly with the same time-ordered evaluation.</p>'''
    page = (f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Group 20 — D poster section</title><style>{CSS}</style><main>\n{body}</main></html>')
    (HERE / 'D_Poster_Section_EN.html').write_text(page, encoding='utf-8')

    speech_main = f'''Our advanced models are Random Forest, LightGBM and XGBoost, trained on B's features and scored with C's shared evaluation.

{best} is our best model. On validation it is off by {say(b['validation_MAE'])} minutes on average, {g['validation_vs_best_linear_pct']:.0f} percent better than the best linear model, because trees learn how zones and time interact.

Location drives the prediction: about {100 * s['importance_location_share']:.0f} percent of the model's gain comes from zone features.

On the test days the error rises to {say(b['test_MAE'])} minutes. After the January snowstorm, trips took longer than anything in training, and long trips are under-predicted.

We also found that the Random Forest's parallel prediction changed in the last digits between runs; a fixed summation order made it identical.

Next, we would add weather data and more months. Thank you.'''
    speech = f'''Group 20 — Member D — Speaking script (Results + Conclusion), target 65–70 seconds
Numbers from run: {a.run} (D_Advanced_Models.ipynb){' — LOCAL SAMPLE: update after the Rockhill run' if local else ''}

{speech_main}

---
Likely questions
- Why log duration? Durations are right-skewed; on the log scale one 60-minute trip does not dominate the loss. Predictions are converted back with expm1 and clipped to A's 1–180 minute range.
- Why not distance? The recorded distance is only known after the trip: leakage. All inputs are known at pickup.
- How did you tune? Small grids per model, fitted on train and chosen by validation MAE; LightGBM/XGBoost stop early on validation loss. Test was scored once, for the chosen configuration.
- Why drop day_of_month? Training covers days 1–21, test 27–31; a tree would treat every test day as day 21.
- Why is test worse than validation? Snowstorm on 25 January: test bias {b['test_bias']:.1f} min, daily MAE up to {test_days[worst_day]:.1f} min on {worst_day}; it falls back to {test_days[max(test_days)]:.1f} on {max(test_days)}. Every model, including the baselines, shows the same drop.
- Why are long trips under-predicted? Few long trips per zone pair, no traffic or weather input, and a log-scale loss pulls predictions toward the typical trip.
- Random Forest non-determinism? scikit-learn adds the 300 tree predictions from several threads as each finishes. Floating-point addition is not associative, so the last digits change ({o[0]['max_abs_diff_min']:.0e} min). One prediction thread fixes the order; training stays parallel and seeded.
- Is the result reproducible? Fixed seed {s['config']['seed']}: identical fingerprints for refits and for 1 vs {s['environment']['threads_per_model']} threads, for all three models.
- Why LightGBM over XGBoost? Validation MAE {board['LightGBM']['validation_MAE']:.2f} vs {board['XGBoost']['validation_MAE']:.2f}; chosen on validation only.
- Feature importance metric? Share of total split gain; zone-pair encoding alone ≈ {100 * list(s['importance_top'].values())[0]:.0f}%.
'''
    (HERE / 'D_Speaking_EN.txt').write_text(speech, encoding='utf-8')
    print('Wrote D_Poster_Section_EN.html, D_Speaking_EN.txt and figures/ from', out)
    print('Speech words:', len(speech_main.split()))


if __name__ == '__main__':
    main()

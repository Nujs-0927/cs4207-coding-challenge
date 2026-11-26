"""Build Part B's poster section and speaking script from a notebook run's summary.json.

    python make_b_materials.py --run local_sample      # local development numbers
    python make_b_materials.py --run rockhill_<stamp>  # after the Rockhill run

Every number on the poster and in the script comes from summary.json, so after
the server run the materials update with one command.
"""
import argparse
import html
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM_WORDS = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven',
             8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
             15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',
             30: 'thirty', 40: 'forty'}


def words(n):
    n = int(round(n))
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    tens, ones = divmod(n, 10)
    return f'{NUM_WORDS[tens * 10]}-{NUM_WORDS[ones]}' if tens * 10 in NUM_WORDS else str(n)


# Same visual style as Member A's poster section, so A can merge them consistently
CSS = """:root{--ink:#142a36;--teal:#087f78;--muted:#536670;--pale:#eaf4f2;--line:#d5e1e5}
*{box-sizing:border-box}body{font:16px/1.75 'Segoe UI','Microsoft YaHei',sans-serif;color:var(--ink);background:#f2f6f7;margin:0}
main{max-width:1120px;margin:32px auto;background:white;padding:42px 50px;border-radius:16px}h1{font-size:34px;line-height:1.3}h2{margin-top:36px;font-size:23px;color:var(--teal)}
p,li{max-width:980px}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:10px;text-align:left;vertical-align:top}th{background:var(--pale)}
code{background:#edf2f4;padding:2px 5px;font-family:Consolas,monospace}
.note{background:#fff3dc;border-left:5px solid #dd9b22;padding:15px 20px}.small{font-size:13px;color:var(--muted)}img{max-width:100%}.stats{display:flex;gap:16px;flex-wrap:wrap}.stat{background:var(--pale);padding:16px 24px;flex:1}.stat strong{display:block;font-size:28px;color:var(--teal)}
.figs{display:grid;grid-template-columns:1fr 1fr;gap:18px}.figs figure{margin:0}.figs .wide{grid-column:1/-1}figcaption{font-size:13px;color:var(--muted)}
@media(max-width:650px){main{margin:12px;padding:20px}h1{font-size:27px}.figs{grid-template-columns:1fr}.stat{min-width:130px;padding:12px}}
@media print{body{background:white}main{margin:0;padding:12mm;max-width:none}figure,table,.note{break-inside:avoid}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='local_sample')
    a = ap.parse_args()
    run_dir = HERE / 'outputs' / a.run
    s = json.loads((run_dir / 'summary.json').read_text())
    figs = HERE / 'figures'; figs.mkdir(exist_ok=True)
    for f in (run_dir / 'figures').glob('*.png'):
        shutil.copy2(f, figs / f.name)

    local = not a.run.startswith('rockhill')
    t, v, r = s['target_train'], s['validation_sanity'], s['rows']
    grp = {g['group']: g['median'] for g in s['trip_groups_train']}
    bench = s['benchmark']; last = bench[-1]
    best = max(bench, key=lambda b: b['speedup'])
    equal = all(b['all_outputs_equal'] for b in bench)
    lo, hi = s['hour_weekday_median_range']
    airport = max(grp['To airport'], grp['From airport'])
    scope = (f"Local development run on A's {r['all']:,}-trip sample" if local
             else f"Rockhill run on {r['all']:,} cleaned trips ({html.escape(Path(s['data_source']).name)})")
    if best['workers'] > 1:
        par = (f"Speedup <strong>{best['speedup']:.2f}×</strong> at {best['workers']} workers "
               f"(efficiency {best['efficiency']:.2f}). Below ideal: loading and assembly stay serial, and each task pays "
               "process start-up and transfer costs (Amdahl's law).")
        par_say = (f"With {words(best['workers'])} workers, building the encodings is {best['speedup']:.1f} times faster, "
                   "and the output is identical to the serial run.")
    else:
        par = (f"No speedup at this input size ({last['speedup']:.2f}× at {last['workers']} workers): process start-up "
               "and data transfer outweigh the small amount of work. Measure speedup on the full Rockhill data.")
        par_say = ("Parallel and serial runs give identical output; on this small sample, "
                   "process overhead outweighed the gain.")

    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Group 20 — B poster section</title><style>{CSS}</style><main>
<p class="small">GROUP 20 · ETA PREDICTION · MEMBER B</p><h1>EDA &amp; feature engineering</h1>
{'<p class="note"><strong>Local development results.</strong> Re-run on Rockhill and regenerate this page with <code>make_b_materials.py</code> before the poster is finalised.</p>' if local else ''}
<p class="small">{scope}. EDA uses the training period only ({r['train']:,} trips, before {s['split']['validation_start']}).</p>
<div class="stats"><div class="stat"><strong>{t['50%']:.1f} min</strong>median trip (skew {t['skew']})</div><div class="stat"><strong>{airport:.0f} vs {grp['Same pickup & dropoff zone']:.0f} min</strong>airport vs same-zone</div><div class="stat"><strong>{s['n_features']}</strong>numeric features</div><div class="stat"><strong>−{v['reduction_pct']:.0f}%</strong>MAE, zone-pair lookup vs constant</div></div>
<h2>What the data shows</h2><ul>
<li><strong>Right-skewed target</strong>: median {t['50%']:.1f} min, mean {t['mean']:.1f} min, 95th percentile {t['95%']:.1f} min → we model <code>log1p(duration)</code>.</li>
<li><strong>Location dominates</strong>: median {grp['To airport']:.1f} min to an airport, {grp['From airport']:.1f} min from an airport, {grp['Other trips']:.1f} min for other trips and {grp['Same pickup & dropoff zone']:.1f} min within the same zone.</li>
<li><strong>Time adds a smaller shift</strong>: hour × weekday medians range from {lo:.0f} to {hi:.0f} min.</li></ul>
<div class="figs"><figure class="wide"><img src="figures/B_duration_distribution.png" alt="Duration distribution"><figcaption>Duration (left) and log1p(duration) (right), training period.</figcaption></figure>
<figure class="wide"><img src="figures/B_hour_weekday_heatmap.png" alt="Median duration by hour and weekday"><figcaption>Median duration by pickup hour and weekday.</figcaption></figure>
<figure><img src="figures/B_trip_groups.png" alt="Median duration by trip group"><figcaption>Airport and same-zone trips (medians).</figcaption></figure>
<figure><img src="figures/B_feature_signal.png" alt="Feature signal"><figcaption>Strongest features on validation (|Spearman|).</figcaption></figure></div>
<h2>Features</h2><table><tr><th>Group</th><th>Features</th></tr>
<tr><td>Calendar</td><td>hour, weekday, weekend, US holiday, day of month, cyclic sin/cos of time-of-day and weekday</td></tr>
<tr><td>Our peak definition</td><td>night 00–06 · <strong>morning peak 06–10</strong> · midday 10–16 · <strong>afternoon peak 16–20</strong> · evening 20–24</td></tr>
<tr><td>Zone flags</td><td>airport pickup/dropoff (EWR 1, JFK 132, LGA 138), same pickup and dropoff zone</td></tr>
<tr><td>Target encodings</td><td>smoothed mean log-duration per pickup zone, dropoff zone, zone pair, zone pair × time band, pickup zone × hour, plus log counts (reliability)</td></tr>
<tr><td>Passenger</td><td>count (train-median imputation) + missing indicator</td></tr></table>
<h2>Leakage control</h2><ul><li>Only pickup-time inputs: <strong>no distance, fares, tips, payment type or dropoff time</strong>.</li>
<li>All learned statistics are fitted on the <strong>training split only</strong>. Training rows get <strong>5-fold out-of-fold</strong> encodings, so no row sees its own label.</li>
<li>Chronological splits: validation from {s['split']['validation_start']}, test from {s['split']['test_start']}.</li></ul>
<h2>Task parallelism &amp; determinism</h2><p>The 30 independent encoding tasks (5 folds × 5 tables, plus 5 for evaluation) run in a process pool. A fixed fold seed and order-preserving <code>map</code> make the output <strong>{'bit-identical to serial in every run' if equal else 'NOT identical — investigate'}</strong>. {par}</p>
<img src="figures/B_feature_speedup.png" alt="Speedup chart" style="max-width:520px">
</main></html>"""
    (HERE / 'B_Poster_Section_EN.html').write_text(page, encoding='utf-8')

    speech = f"""Group 20 — Member B — Speaking script (EDA + Features), target 65–70 seconds
Numbers from run: {a.run}

Thanks. I'll cover what the data tells us and how we turned it into features.

First, trip duration is strongly right-skewed. The median is about {words(t['50%'])} minutes, but there is a long tail, so we model the log of duration.

Second, location matters far more than time. Airport trips take around {words(airport)} minutes, while trips within the same zone take about {words(grp['Same pickup & dropoff zone'])}. Time of day shifts the median by several minutes, as the heatmap shows.

From this, we built {words(s['n_features'])} features: calendar features, our own peak-hour bands, airport and same-zone flags, and smoothed target encodings for zones and zone pairs.

To avoid leakage, we fit the encodings on training data only, using out-of-fold values, and we never use distance, fares or dropoff time. {par_say}

Now C will explain our models.

---
Likely questions
- Why not trip distance? It is the realised distance, known only after the trip → leakage when predicting at pickup.
- What is target encoding? Each zone or zone pair is replaced by its average log-duration in the training data, smoothed toward the mean for rare keys.
- Why out-of-fold? Otherwise a training row's encoding contains its own label and the model overfits.
- Why log? Skew {t['skew']}; the log reduces the influence of very long trips.
- Why is dropoff zone allowed? We assume the destination is given at pickup, as in a ride-hailing app.
- Parallel correctness? Fixed fold seed + order-preserving map; hashes of serial and parallel outputs are identical.
- Signal check: zone-pair lookup alone gives MAE {v['MAE_zone_pair_lookup']} vs {v['MAE_constant_train_median']} min for a constant baseline (not a model).
"""
    (HERE / 'B_Speaking_EN.txt').write_text(speech, encoding='utf-8')
    print('Wrote B_Poster_Section_EN.html, B_Speaking_EN.txt and figures/ from', run_dir)
    print('Speech words:', len(speech.split('---')[0].split('\n', 2)[2].split()))


if __name__ == '__main__':
    main()

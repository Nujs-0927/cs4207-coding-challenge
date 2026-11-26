"""Build Part C's poster section and speaking script from a notebook run's c_summary.json.

    python make_c_materials.py --run local_full         # local development numbers
    python make_c_materials.py --run rockhill_<stamp>   # after the Rockhill run
    python make_c_materials.py --dir ../Group20_ETA_Final/outputs/<run>   # from the merged notebook (use this for the poster)

Every number on the poster and in the script comes from c_summary.json, so the
materials update with one command after the server run.
"""
import argparse
import html
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Same visual style as A's and B's poster sections, so A can merge them consistently
CSS = """:root{--ink:#142a36;--teal:#087f78;--muted:#536670;--pale:#eaf4f2;--line:#d5e1e5}
*{box-sizing:border-box}body{font:16px/1.75 'Segoe UI','Microsoft YaHei',sans-serif;color:var(--ink);background:#f2f6f7;margin:0}
main{max-width:1120px;margin:32px auto;background:white;padding:42px 50px;border-radius:16px}h1{font-size:34px;line-height:1.3}h2{margin-top:36px;font-size:23px;color:var(--teal)}
p,li{max-width:980px}table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}th{background:var(--pale)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}tr.best td{font-weight:600}
code{background:#edf2f4;padding:2px 5px;font-family:Consolas,monospace}
.note{background:#fff3dc;border-left:5px solid #dd9b22;padding:15px 20px}.small{font-size:13px;color:var(--muted)}img{max-width:100%}.stats{display:flex;gap:16px;flex-wrap:wrap}.stat{background:var(--pale);padding:16px 24px;flex:1}.stat strong{display:block;font-size:28px;color:var(--teal)}
.figs{display:grid;grid-template-columns:1fr 1fr;gap:18px}.figs figure{margin:0}.figs .wide{grid-column:1/-1}figcaption{font-size:13px;color:var(--muted)}
@media(max-width:650px){main{margin:12px;padding:20px}h1{font-size:27px}.figs{grid-template-columns:1fr}.stat{min-width:130px;padding:12px}table{display:block;overflow-x:auto}}
@media print{body{background:white}main{margin:0;padding:12mm;max-width:none}figure,table,.note{break-inside:avoid}}"""

WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 8: 'eight'}


def num(x, digits=2):
    """Fixed decimals without a negative zero."""
    return f'{x:.{digits}f}'.replace('-0.' + '0' * digits, '0.' + '0' * digits)


def sci(x):
    """5.2e-07 -> 5×10⁻⁷"""
    m, e = f'{x:.0e}'.split('e')
    return f"{m}×10{str(int(e)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def row_cells(values, tag='td'):
    first, *rest = values
    return f'<{tag}>{first}</{tag}>' + ''.join(f'<{tag} class="n">{v}</{tag}>' for v in rest)


def bench_line(rows, workers):
    return next(r for r in rows if r['workers'] == workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='local_full')
    ap.add_argument('--dir', help='run folder of the merged notebook, e.g. ../Group20_ETA_Final/outputs/<run>')
    a = ap.parse_args()
    run_dir = Path(a.dir).resolve() if a.dir else HERE / 'outputs' / a.run
    run_name = run_dir.name
    notebook = 'Group20_ETA_Final.ipynb (Part C)' if a.dir else 'C_Baseline_Models.ipynb'
    s = json.loads((run_dir / 'c_summary.json').read_text(encoding='utf-8'))
    figs = HERE / 'figures'
    figs.mkdir(exist_ok=True)
    for f in (run_dir / 'figures').glob('C_*.png'):
        shutil.copy2(f, figs / f.name)

    local = not run_name.startswith('rockhill')
    h, r, board = s['headline'], s['rows'], {m['model']: m for m in s['scoreboard']}
    zp, const = board['Zone-pair median'], board['Constant (train median)']
    best = board[h['best_model']]
    w = max(s['config']['workers'])
    search, grad = s['search_benchmark'], s['gd_benchmark']['full']['table']
    sw, gw = bench_line(search, w), bench_line(grad, w)
    small = s['gd_benchmark'].get('small')
    im, am, sh, ov = s['imbalance'], s['amdahl'], s['shift'], s['overhead'][-1]
    env = s['environment']
    daily = s['daily_mae']
    n_nd, shown = len(s['nondeterminism']), sum(x['distinct_before'] > 1 for x in s['nondeterminism'])
    scope = (f"Local development run on {r['all']:,} cleaned trips ({env['cores_usable']} cores used)" if local
             else f"Rockhill run on {r['all']:,} cleaned trips ({html.escape(str(env['cpu']))}, {env['cores_usable']} cores)")

    models = ''
    for m in s['scoreboard']:
        cells = [html.escape(m['model'])] + [num(m[f'{sp}_{k}']) for sp in ('validation', 'test') for k in ('MAE', 'RMSE', 'R2')]
        models += f"<tr{' class=best' if m['model'] == h['best_model'] else ''}>{row_cells(cells)}</tr>"
    speed = ''
    for name, rows in [('Hyperparameter search (18 trials)', search),
                       (f"Gradient, {s['gd_benchmark']['full']['rows']:,} rows", grad)] + (
                           [(f"Gradient, {small['rows']:,} rows", small['table'])] if small else []):
        for b in rows:
            speed += f"<tr>{row_cells([name, b['workers'], format(b['mean_s'], '.2f') + ' ± ' + format(b['std_s'], '.2f'), format(b['speedup'], '.2f') + '×', format(b['efficiency'], '.2f')])}</tr>"
    nd = ''.join(f"<tr><td>{html.escape(x['source'])}<br><span class='small'>{html.escape(x['what changed'])}</span></td>"
                 f"<td class='n'>{x['distinct_before']} of {x['runs_before']}</td><td>{html.escape(x['fix'])}</td>"
                 f"<td class='n'>{x['distinct_after']} of {x['runs_after']}</td></tr>" for x in s['nondeterminism'])
    if small and bench_line(small['table'], w)['speedup'] < 1:
        small_text = (f" On {small['rows']:,} rows the parallel version is slower "
                      f"({bench_line(small['table'], w)['speedup']:.2f}×): start-up and messages cost more than the maths.")
    else:
        small_text = ''
    limit = max(x['largest weight difference vs 8 shards'] for x in s['limits'])

    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Group 20 — C poster section</title><style>{CSS}</style><main>
<p class="small">GROUP 20 · ETA PREDICTION · MEMBER C</p><h1>Methodology &amp; baselines</h1>
{'<p class="note"><strong>Local development results.</strong> Re-run on Rockhill and regenerate this page with <code>make_c_materials.py</code> before the poster is finalised.</p>' if local else ''}
<p class="small">{scope}. All numbers come from <code>{notebook}</code>.</p>
<div class="stats"><div class="stat"><strong>{zp['validation_MAE']:.2f} min</strong>MAE of the zone-pair baseline (validation)</div><div class="stat"><strong>{best['validation_MAE']:.2f} min</strong>MAE of the best linear model (validation)</div><div class="stat"><strong>{sw['speedup']:.1f}× / {gw['speedup']:.1f}×</strong>search / gradient speedup, {w} workers</div><div class="stat"><strong>{shown} of {n_nd}</strong>sources of non-determinism observed and fixed</div></div>
<h2>Where parallelism is used</h2>
<img src="figures/C_pipeline.png" alt="Pipeline: data parallel cleaning, task parallel features, sequential split and baselines, task parallel search, data and task parallel gradient, tree models">
<h2>Time-based split</h2><ul>
<li><strong>Train</strong> 1–21 Jan ({r['train']:,} trips) → <strong>validation</strong> 22–26 Jan ({r['validation']:,}) → <strong>test</strong> 27–31 Jan ({r['test']:,}). Never random.</li>
<li>Every choice uses validation; test is scored once.</li>
<li><strong>Shift:</strong> after the 25 Jan snowstorm, 27–30 Jan trips average {sh['jan27_30_mean']:.1f} min; no training day exceeds {sh['train_daily_mean_max']:.1f}.</li></ul>
<img src="figures/C_split_timeline.png" alt="Trips per day and mean duration per day, coloured by split">
<h2>Evaluation and results</h2>
<p>One shared <code>evaluate()</code> for all models, always in minutes: <strong>MAE</strong> (main metric), <strong>RMSE</strong>, <strong>R²</strong>.</p>
<table><tr>{row_cells(['Model', 'Val MAE', 'Val RMSE', 'Val R²', 'Test MAE', 'Test RMSE', 'Test R²'], 'th')}</tr>{models}</table>
<div class="figs"><figure><img src="figures/C_model_comparison.png" alt="MAE by model"><figcaption>Where the trip goes matters far more than when it starts.</figcaption></figure>
<figure><img src="figures/C_daily_error.png" alt="Daily MAE"><figcaption>Daily MAE: up to {max(daily.values()):.1f} min after the storm, {daily[max(daily)]:.1f} min on 31 Jan.</figcaption></figure></div>
<ul><li>Knowing only weekday and hour barely helps ({const['validation_MAE']:.2f} → {board['Same weekday & hour']['validation_MAE']:.2f} min); the zone pair cuts MAE to <strong>{zp['validation_MAE']:.2f}</strong>.</li>
<li>Best linear model: zone encodings × time bands, <strong>{best['validation_MAE']:.2f} min</strong> ({h['val_vs_zone_pair_pct']:.1f}% better than the zone pair; search chose α = {s['best']['alpha']:g}).</li>
<li>Test MAE {best['test_MAE']:.2f} min, bias {h['test_bias']:+.1f} min: models trained on normal traffic predict too short after the storm.</li></ul>
<h2>Two parallel patterns, same answer as sequential</h2>
<ul><li><strong>Hyperparameter search (task parallel):</strong> {s['n_trials']} independent trials, {s['n_configs']} configurations, chosen on validation. The longest trial takes {im['ratio']:.0f}× the shortest; starting the longest first cuts {im['compute_listed_order_s']:.2f} s to {im['compute_longest_first_s']:.2f} s.</li>
<li><strong>Gradient (all-reduce):</strong> 8 equal row shards, gradient sums and counts added in shard order, one shared weight vector. Matches the closed-form ridge within {sci(s['gd_convergence']['max_prediction_gap_min'])} min.</li></ul>
<div class="figs"><figure><img src="figures/C_parallel_speedup.png" alt="Speedup against workers"><figcaption>Speedup including process start-up.</figcaption></figure>
<figure><img src="figures/C_trial_gantt.png" alt="Trials per worker"><figcaption>Load imbalance between search trials.</figcaption></figure></div>
<table><tr>{row_cells(['Workload (mean of 3 runs, outputs identical)', 'Workers', 'Seconds', 'Speedup', 'Efficiency'], 'th')}</tr>{speed}</table>
<p><strong>Amdahl's law:</strong> fitted parallel fraction p = {am['p_search']:.2f} (search) and {am['p_gradient']:.2f} (gradient). With {w} workers the gradient loses {ov['pool_startup_s']:.2f} s to process start-up, {ov['communication_s']:.2f} s to messages and {ov['other_s (memory bus, waiting)']:.2f} s to shared memory and waiting.{small_text}</p>
<h2>Non-determinism: shown, then fixed</h2>
<table><tr><th>Source</th><th class="n">Distinct outputs before</th><th>Fix</th><th class="n">After</th></tr>{nd}</table>
<p class="small">Limits: we claim identical output for one machine, one set of library versions and a fixed shard count. Changing the number of shards still moves weights by up to {sci(limit)}.</p>
</main></html>"""
    (HERE / 'C_Poster_Section_EN.html').write_text(page, encoding='utf-8')

    speech = f"""Group 20 — Member C — Speaking script (Methodology + Baselines), target 65–70 seconds
Numbers from run: {run_name}, {notebook}

Thanks. I'll explain how we test fairly, and what our models must beat.

We split by time, never at random: three weeks for training, five days to choose models, and the last five days for one final test. One shared function scores every model in minutes: MAE, RMSE and R squared.

A constant guess is wrong by {const['validation_MAE']:.1f} minutes. Looking up the median for the same zone pair cuts that to {zp['validation_MAE']:.1f}, so that is our baseline. Our best linear model reaches {best['validation_MAE']:.1f}. On the test days it rises to {best['test_MAE']:.1f}: trips took longer in the days after the January snowstorm.

Two parts run in parallel. With {WORDS.get(w, w)} workers, {s['n_trials']} model trials run {sw['speedup']:.1f} times faster, and the sharded gradient {gw['speedup']:.1f} times faster, with bit-identical results.

We also found {WORDS.get(shown, shown)} sources of non-determinism, and fixed them all.

Now D will show our advanced models.

---
Likely questions
- Why not a random split? The model would train on trips that happen after the ones it is tested on, so the score would be too good. The lecture says: train on earlier dates, test on later ones.
- Why these dates? 21 training days are exactly three weeks, so each weekday appears three times. Validation and test have five days each.
- Why is test worse than validation? Snowstorm on 25 January. On 27–30 January the mean trip is {sh['jan27_30_mean']:.1f} minutes, above every training day (max {sh['train_daily_mean_max']:.1f}). The bias is {h['test_bias']:.1f} minutes, and daily error returns to {daily[max(daily)]:.1f} on 31 January. We did not move the split after seeing this.
- Why MAE first? It reads directly as "how many minutes off is the ETA". RMSE punishes large misses; R squared compares with predicting the mean.
- Why log duration? Durations are right-skewed. On validation MAE the log target beats raw minutes for every feature set.
- Why does ridge barely help? {r['train']:,} rows against at most 115 inputs: little overfitting to remove. The search even chose alpha = {s['best']['alpha']:g} for the best model.
- Why drop day_of_month? Train only covers days 1–21, so days 22–31 would be extrapolation. band_night and is_peak are exact combinations of other columns.
- Why are the gradient shards independent? For fixed weights each row's gradient depends only on that row. We add sums and counts, then divide once.
- Why is the gradient only {gw['speedup']:.1f}× with {w} workers? Every iteration sends weights out and gradients back, start-up costs {ov['pool_startup_s']:.2f} s, and workers share one memory bus. Amdahl's law gives p = {am['p_gradient']:.2f}, so at most {am['max_speedup_gradient']:.1f}×. Shards are balanced (slowest / mean = {s['shard_balance']['slowest_over_mean']:.2f}).
- Why does the search scale better? Trials never talk to each other. Its limit is the longest trial: total ÷ longest = {im['bound_by_longest_trial']:.1f}×.
- How do you know parallel equals sequential? SHA-256 fingerprints of the raw bytes match for 1, 2 and 4 workers and every repeat. The gradient also matches the closed form, and that matches scikit-learn.
- Why do results differ in completion order? Floating-point addition is not associative, so a different order changes the last digits: {s['nondeterminism'][0]['what changed'].split(' (')[0]}.
- Why one BLAS thread? Otherwise "one worker" would already use several cores and the speedup would be meaningless.
- Limits of the fixes? Identical on the same machine, library versions and shard count. Other shard counts move weights by up to {sci(limit)}.
"""
    (HERE / 'C_Speaking_EN.txt').write_text(speech, encoding='utf-8')
    spoken = speech.split('---')[0].split('\n\n', 1)[1]
    print('Wrote C_Poster_Section_EN.html, C_Speaking_EN.txt and figures/ from', run_dir)
    print('Speech words:', len(spoken.split()))


if __name__ == '__main__':
    main()

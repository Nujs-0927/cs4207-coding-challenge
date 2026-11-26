"""Group 20 - Member C: figures. Same look as B's plots (blue marks, light grid)."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

INK, INK2, GRID = '#191919', '#666666', '#e7e7e7'
BLUE, ORANGE, GREEN, GREY = '#257de0', '#eb6834', '#1baf7a', '#b9bec3'
SPLIT_COLOURS = {'train': GREY, 'validation': BLUE, 'test': ORANGE}
SET_COLOURS = {'calendar': GREEN, 'encoded': ORANGE, 'interactions': BLUE}


def style(ax, title='', xlabel='', ylabel='', grid='y'):
    ax.set_title(title, loc='left', fontsize=13, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c2c2c2')
    ax.tick_params(colors=INK2)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)


def split_timeline(daily, storm_day=None):
    """daily: index = date, columns = trips, mean_min, split."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, height_ratios=[1, 1])
    x = np.arange(len(daily))
    colours = daily['split'].map(SPLIT_COLOURS)
    unit, div = ('Trips (thousands)', 1000) if daily['trips'].max() >= 20000 else ('Trips', 1)
    for ax, col, scale in ((axes[0], 'trips', div), (axes[1], 'mean_min', 1)):
        ax.bar(x, daily[col] / scale, width=0.72, color=colours)
    style(axes[0], 'Time-based split: train on the past, test on the future', ylabel=unit)
    style(axes[1], 'Mean trip duration shifts in the test window', 'January 2026 (day of month)', 'Minutes')
    axes[1].set_ylim(0, daily['mean_min'].max() * 1.18)
    axes[1].set_xticks(x, [d.day for d in daily.index], fontsize=9)
    handles = [Patch(color=c, label=f'{s} ({n} days)') for s, c in SPLIT_COLOURS.items()
               for n in [int((daily['split'] == s).sum())]]
    axes[0].legend(handles=handles, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.02),
                   labelcolor=INK2, fontsize=9)
    axes[0].set_ylim(0, daily['trips'].max() / div * 1.28)
    if storm_day is not None:
        i = list(daily.index).index(pd.Timestamp(storm_day))
        top = daily['trips'].iloc[i:i + 2].max() / div          # the two slow days sit between tall bars
        axes[0].annotate('snow-\nstorm', (i + 0.5, top), xytext=(0, 5), textcoords='offset points',
                         ha='center', va='bottom', color=INK, fontsize=9, linespacing=1.0)
    fig.tight_layout()
    return fig


def model_comparison(table, metric='MAE'):
    """table: Scoreboard.table(). Validation and test side by side, best at the bottom."""
    names = list(table.index)[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 0.58 * len(names) + 1.5))
    h = 0.34
    for col, colour, off in ((f'validation_{metric}', BLUE, h / 2 + 0.015), (f'test_{metric}', ORANGE, -h / 2 - 0.015)):
        v = table.loc[names, col].to_numpy()
        ax.barh(y + off, v, height=h, color=colour)
        for yi, vi in zip(y + off, v):
            ax.text(vi + 0.08, yi, f'{vi:.2f}', va='center', color=INK, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(0, table[[f'validation_{metric}', f'test_{metric}']].to_numpy().max() * 1.1)
    style(ax, f'{metric} in minutes (lower is better)', grid='x')
    ax.tick_params(axis='y', colors=INK, length=0)
    ax.legend(handles=[Patch(color=BLUE, label='validation, 22-26 Jan'), Patch(color=ORANGE, label='test, 27-31 Jan')],
              frameon=False, loc='lower right', labelcolor=INK2)
    fig.tight_layout()
    return fig


def speedup(series, title='Speedup against workers (total wall-clock, 3 repeats)'):
    """series: name -> DataFrame with columns workers, speedup. At most three lines."""
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    top = max(int(d['workers'].max()) for d in series.values())
    ax.plot([1, top], [1, top], color=INK2, linewidth=1, linestyle='--')
    ax.text(top, top, ' ideal', color=INK2, va='center')
    ends = []
    for (name, d), colour in zip(series.items(), (BLUE, ORANGE, GREEN)):
        ax.plot(d['workers'], d['speedup'], color=colour, linewidth=2, marker='o', markersize=8,
                markeredgecolor='white', markeredgewidth=1.5)
        ends.append([float(d['speedup'].iloc[-1]), float(d['workers'].iloc[-1]), name])
    gap, last = 0.07 * top, -np.inf
    for s, w, name in sorted(ends):                    # keep the end labels apart
        pos = max(s, last + gap)
        ax.text(w, pos, f'  {s:.2f}×', color=INK, va='center', fontsize=10)
        last = pos
    ax.axhline(1, color='#c2c2c2', linewidth=0.8)
    ax.set_xticks(sorted({int(w) for d in series.values() for w in d['workers']}))
    ax.set_xlim(0.8, top * 1.12)
    ax.set_ylim(0, top * 1.06)
    style(ax, title, 'Worker processes', 'Speedup vs sequential')
    ax.legend(handles=[Line2D([], [], color=c, marker='o', linewidth=2, label=n)
                       for n, c in zip(series, (BLUE, ORANGE, GREEN))],
              frameon=False, loc='upper left', labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def trial_gantt(results, workers):
    """Which worker ran which trial, and when: makes load imbalance visible."""
    t0 = min(r['began'] for r in results)
    pids = list(dict.fromkeys(r['pid'] for r in sorted(results, key=lambda r: r['began'])))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(pids) + 1.7))
    for r in results:
        y = pids.index(r['pid'])
        ax.barh(y, r['seconds'], left=r['began'] - t0, height=0.55, edgecolor='white', linewidth=1.5,
                color=SET_COLOURS[r['rows'][0]['features']])
    ax.set_yticks(range(len(pids)), [f'worker {i + 1}' for i in range(len(pids))])
    ax.invert_yaxis()
    style(ax, f'{len(results)} independent trials on {workers} workers, longest first', 'Seconds since the first trial started', grid='x')
    ax.tick_params(axis='y', length=0, colors=INK)
    ax.legend(handles=[Patch(color=c, label=n) for n, c in SET_COLOURS.items()], frameon=False, ncol=3,
              loc='upper center', bbox_to_anchor=(0.5, -0.32 if len(pids) < 4 else -0.24), labelcolor=INK2, fontsize=9)
    fig.tight_layout()
    return fig


def daily_error(daily, storm_day=None):
    """daily: index = date, one column of MAE per model, plus 'split'."""
    models = [c for c in daily.columns if c != 'split']
    x = np.arange(len(daily))
    fig, ax = plt.subplots(figsize=(9, 3.9))
    first_test = int((daily['split'] == 'validation').sum())
    ax.axvline(first_test - 0.5, color='#c2c2c2', linewidth=1)
    top = daily[models].to_numpy().max() * 1.3
    ax.set_ylim(0, top)
    ax.text(first_test - 0.58, top * 0.97, 'validation', ha='right', va='top', color=INK2)
    ax.text(first_test - 0.42, top * 0.97, 'test', ha='left', va='top', color=INK2)
    for m, colour, z in zip(models, (BLUE, GREY), (3, 2)):
        ax.plot(x, daily[m], color=colour, linewidth=2, marker='o', markersize=8, markeredgecolor='white',
                markeredgewidth=1.5, label=m, zorder=z)
    for i in (0, int(daily[models[0]].to_numpy().argmax())):
        ax.text(i, daily[models[0]].iloc[i] - 0.06 * top, f'{daily[models[0]].iloc[i]:.1f}', ha='center', va='top', color=INK)
    if storm_day is not None:
        i = list(daily.index).index(pd.Timestamp(storm_day))
        ax.annotate('snowstorm', (i, daily[models].iloc[i].max()), xytext=(0, 12), textcoords='offset points',
                    ha='center', color=INK, fontsize=9)
    ax.set_xticks(x, [f'{d:%a} {d.day}' for d in daily.index])
    style(ax, 'Daily MAE: errors jump after the storm, then recover', 'January 2026', 'MAE (minutes)')
    ax.legend(frameon=False, loc='lower right', labelcolor=INK2, ncol=2)
    fig.tight_layout()
    return fig


def pipeline():
    """Whole-project pipeline: where data parallelism and task parallelism are used."""
    steps = [
        ('A', 'Clean', 'Parquet row batches,\nsame rules on each batch', 'DATA PARALLEL', 'ordered merge of batches'),
        ('B', 'Features', '5 folds × 5 encoding\ntables, fixed columns', 'TASK PARALLEL', 'one job per fold and table'),
        ('C', 'Split, baselines', 'train → validation → test\nby pickup time', 'SEQUENTIAL', 'one shared evaluate()'),
        ('C', 'Model search', '18 trials, scored on\nvalidation only', 'TASK PARALLEL', 'one job per trial'),
        ('C', 'Gradient', '8 equal row shards,\nsums + counts → weights', 'DATA + TASK PARALLEL', 'all-reduce in shard order'),
        ('D', 'Tree models', 'Random Forest, boosting;\nsame split and metrics', 'SEE PART D', 'final comparison on test'),
    ]
    tint = {'DATA PARALLEL': '#dcebfb', 'TASK PARALLEL': '#fde4d8', 'DATA + TASK PARALLEL': '#dff3ea',
            'SEQUENTIAL': '#eef0f1', 'SEE PART D': '#eef0f1'}
    fig, ax = plt.subplots(figsize=(13.5, 2.35))
    w, gap = 2.0, 0.26
    for i, (who, name, text, kind, note) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1.7, boxstyle='round,pad=0,rounding_size=0.1',
                                    facecolor=tint[kind], edgecolor='none'))
        ax.text(x + 0.1, 1.5, f'{who} · {name}', fontsize=11.5, color=INK, weight='bold', va='center')
        ax.text(x + 0.1, 1.06, text, fontsize=9.6, color=INK, va='center', linespacing=1.3)
        ax.text(x + 0.1, 0.5, kind, fontsize=8.8, color=INK, weight='bold', va='center')
        ax.text(x + 0.1, 0.22, note, fontsize=9, color=INK2, va='center')
        if i:
            ax.annotate('', (x - 0.02, 0.85), (x - gap + 0.02, 0.85),
                        arrowprops={'arrowstyle': '-|>', 'color': INK2, 'lw': 1.2})
    ax.set_xlim(-0.05, len(steps) * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.05, 1.75)
    ax.axis('off')
    fig.tight_layout()
    return fig

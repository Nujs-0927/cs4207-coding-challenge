"""Group 20 - Member D: figures, same look as B's and C's (blue = validation, orange = test)."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

INK, INK2, GRID = '#191919', '#666666', '#e7e7e7'
BLUE, ORANGE, GREY = '#257de0', '#eb6834', '#b9bec3'
LABELS = {
    'te_pair': 'zone-pair encoding', 'te_pair_band': 'zone pair x time band', 'te_pair_blend': 'zone-pair blend',
    'te_pickup_hour': 'pickup zone x hour', 'te_pickup': 'pickup-zone encoding', 'te_dropoff': 'dropoff-zone encoding',
    'te_pair_count': 'zone-pair count', 'te_pair_band_count': 'pair x band count', 'te_pickup_count': 'pickup count',
    'te_dropoff_count': 'dropoff count', 'te_pickup_hour_count': 'pickup x hour count',
    'pickup_zone_code': 'pickup zone (category)', 'dropoff_zone_code': 'dropoff zone (category)',
    'tod_sin': 'time of day (sin)', 'tod_cos': 'time of day (cos)', 'dow_sin': 'weekday (sin)', 'dow_cos': 'weekday (cos)',
    'hour': 'hour', 'day_of_week': 'weekday', 'is_weekend': 'weekend', 'is_holiday': 'holiday',
    'same_zone': 'same pickup & dropoff zone', 'pickup_airport': 'pickup at airport', 'dropoff_airport': 'dropoff at airport',
    'passenger_count': 'passenger count', 'passenger_count_missing': 'passenger count missing',
}


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


def model_comparison(table, highlight=(), metric='MAE'):
    """table: Scoreboard.table(). Validation and test MAE side by side; D's models in bold."""
    names = list(table.index)[::-1]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.4))
    h = 0.36
    for col, colour, off in ((f'validation_{metric}', BLUE, h / 2 + 0.015), (f'test_{metric}', ORANGE, -h / 2 - 0.015)):
        v = table.loc[names, col].to_numpy()
        ax.barh(y + off, v, height=h, color=colour)
        for yi, vi in zip(y + off, v):
            ax.text(vi + 0.08, yi, f'{vi:.2f}', va='center', color=INK, fontsize=9)
    ax.set_yticks(y, names)
    for lab in ax.get_yticklabels():
        if lab.get_text() in highlight:
            lab.set_fontweight('bold')
    ax.set_xlim(0, table[[f'validation_{metric}', f'test_{metric}']].to_numpy().max() * 1.12)
    style(ax, f'{metric} in minutes, lower is better (bold = Part D)', grid='x')
    ax.tick_params(axis='y', colors=INK, length=0)
    ax.legend(handles=[Patch(color=BLUE, label='validation, 22-26 Jan'), Patch(color=ORANGE, label='test, 27-31 Jan')],
              frameon=False, loc='lower right', labelcolor=INK2)
    fig.tight_layout()
    return fig


def feature_importance(imp, title, top=12):
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.3))
    ax.barh([LABELS.get(c, c) for c in s.index], 100 * s.values, color=BLUE, height=0.6)
    for i, v in enumerate(100 * s.values):
        ax.text(v + 0.4, i, f'{v:.1f}%', va='center', color=INK, fontsize=9)
    ax.set_xlim(0, 100 * s.values.max() * 1.18)
    style(ax, title, 'Share of total split gain (%)', grid='x')
    ax.tick_params(axis='y', colors=INK, length=0)
    fig.tight_layout()
    return fig


def predicted_vs_actual(y, p, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    hb = ax.hexbin(y, p, gridsize=60, bins='log', mincnt=1, xscale='log', yscale='log',
                   extent=(0, np.log10(180), 0, np.log10(180)),
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', ['#cce3fd', '#257de0', '#0e3d7e']))
    ax.plot([1, 180], [1, 180], color=ORANGE, linewidth=1.5, linestyle='--', label='perfect prediction')
    ticks = [1, 2, 5, 10, 20, 50, 100, 180]
    ax.set_xticks(ticks, ticks)
    ax.set_yticks(ticks, ticks)
    ax.minorticks_off()
    style(ax, title, 'Actual duration (minutes, log scale)', 'Predicted duration (minutes, log scale)', grid=None)
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label('Trips per cell (log)', color=INK2)
    cb.outline.set_visible(False)
    ax.legend(frameon=False, loc='upper left', labelcolor=INK2)
    fig.tight_layout()
    return fig

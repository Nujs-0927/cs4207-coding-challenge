"""Group 20 pipeline diagram for the poster: where data and task parallelism are used.
    python make_pipeline_diagram.py   ->  figures/Group20_pipeline.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
INK, INK2, LINE = '#191919', '#5c5c5c', '#c2c2c2'
DATA, TASK, PLAIN = '#257de0', '#e67e22', '#f3f3f3'   # data-parallel blue, task-parallel orange

fig, ax = plt.subplots(figsize=(14, 5.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 5.2); ax.axis('off')


def box(x, y, w, h, title, lines, kind=None):
    face = {'data': '#e3effc', 'task': '#fdeedd', None: PLAIN}[kind]
    edge = {'data': DATA, 'task': TASK, None: LINE}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                facecolor=face, edgecolor=edge, linewidth=2 if kind else 1.2))
    ax.text(x + 0.15, y + h - 0.28, title, fontsize=12, weight='bold', color=INK, va='top')
    for i, t in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.72 - i * 0.36, t, fontsize=9.6, color=INK2, va='top')
    if kind:
        tag = 'DATA PARALLEL' if kind == 'data' else 'TASK PARALLEL'
        ax.text(x + w - 0.12, y + 0.14, tag, fontsize=8.5, weight='bold', color=edge, ha='right')


def arrow(x1, x2, y=2.75):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=16, color=INK2, linewidth=1.4))


ax.text(0.1, 5.0, 'Group 20 pipeline: predicting taxi trip duration at pickup', fontsize=14, weight='bold', color=INK, va='top')
box(0.1, 1.15, 2.2, 3.2, 'Raw data', ['NYC TLC Yellow Taxi', 'Parquet on Rockhill', 'read in row batches', '(iter_batches)'])
arrow(2.3, 2.65)
box(2.65, 1.15, 3.1, 3.2, 'A  Cleaning', ['batches → worker processes', 'same rules on every shard', 'merge in fixed batch order', 'audit: rows removed per rule', 'time-based split (no leakage)'], 'data')
arrow(5.75, 6.1)
box(6.1, 1.15, 3.3, 3.2, 'B  Feature extraction', ['30 independent encoding tasks', '(5 folds × 5 tables + 5 eval)', 'out-of-fold, train-only stats', 'recombined in fixed column order', '33 numeric features'], 'task')
arrow(9.4, 9.75, 3.7); arrow(9.4, 9.75, 1.95)
box(9.75, 3.0, 4.1, 1.35, 'C  Baseline models', ['baseline, linear / ridge', 'shared MAE / RMSE / R² evaluation'])
box(9.75, 1.15, 4.1, 1.65, 'D  Advanced models', ['RF / gradient boosting', 'parallel hyperparameter search'], 'task')
ax.text(0.1, 0.72, 'Determinism fixes:  fixed fold seed  ·  sorted zone-code vocabulary (no set/hash order)  ·  fixed batch order in the merge',
        fontsize=9.6, color=INK2)
ax.text(0.1, 0.34, 'Results collected in submission order (pool.map), never completion order  ·  every parallel stage hashed and compared with its serial run',
        fontsize=9.6, color=INK2)
(HERE / 'figures').mkdir(exist_ok=True)
fig.savefig(HERE / 'figures' / 'Group20_pipeline.png', dpi=200, bbox_inches='tight')
print('Wrote figures/Group20_pipeline.png')

"""
generate_charts.py
Regenerates all benchmark visualisation charts for the PGC Lab experiment.
Outputs three PNG files to the images/ directory:
  - images/performance_comparison_charts.png  (combined dual chart)
  - images/execution_time_chart.png           (standalone time chart)
  - images/speedup_chart.png                  (standalone speedup chart)

Usage:
    python generate_charts.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('images', exist_ok=True)

# ── Font & theme ──────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

# ── Benchmark data ────────────────────────────────────────────────────────────
paradigms = [
    'Sequential CPU\n(Single-Threaded)',
    'MPI Cluster\n(4 VM Nodes)',
    'OpenMP\n(8 CPU Threads)',
    'CUDA Acceleration\n(NVIDIA GPU)',
]
exec_times = [244.12, 92.98, 30.83, 0.165]
speedup_factors = [1.0, 2.63, 7.92, 1479.48]

# ── Visual style ──────────────────────────────────────────────────────────────
bar_colors = ['#6C5CE7', '#0984E3', '#FD79A8', '#00B894']
bar_hatches = ['///', '\\\\\\', '...', '***']

BAR_WIDTH = 0.52
EDGE_CLR = '#2D3436'
GRID_CLR = '#DFE6E9'
LABEL_CLR = '#2D3436'
AXIS_CLR = '#636E72'
LABEL_BOX = dict(boxstyle='round,pad=0.3', fc='#F1F2F6', ec='#B2BEC3', lw=1)


def _annotate_bars(ax, bars, values, fmt_fn):
    """Place a labelled box above each bar."""
    for bar, val in zip(bars, values):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y * 1.4,
            fmt_fn(val),
            ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color=LABEL_CLR,
            bbox=LABEL_BOX,
        )


def _style_axis(ax):
    """Apply common axis style."""
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle=':', color=GRID_CLR, alpha=0.8)
    ax.set_facecolor('#FFFFFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── 1. Combined dual-panel chart ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor='#FAFAFA')

bars1 = ax1.bar(paradigms, exec_times, color=bar_colors,
                width=BAR_WIDTH, edgecolor=EDGE_CLR, linewidth=1.5)
for bar, hatch in zip(bars1, bar_hatches):
    bar.set_hatch(hatch)
_style_axis(ax1)
ax1.set_title('Benchmark Execution Time (Log Scale)',
              fontsize=14, fontweight='bold', color=LABEL_CLR, pad=15)
ax1.set_ylabel('Execution Time in Seconds (Lower is Better)',
               fontsize=11, fontweight='bold', color=AXIS_CLR)
_annotate_bars(ax1, bars1, exec_times,
               lambda t: f'{t:.2f} s' if t >= 1 else f'{t:.3f} s')

bars2 = ax2.bar(paradigms, speedup_factors, color=bar_colors,
                width=BAR_WIDTH, edgecolor=EDGE_CLR, linewidth=1.5)
for bar, hatch in zip(bars2, bar_hatches):
    bar.set_hatch(hatch)
_style_axis(ax2)
ax2.set_title('Parallel Speedup Factor (Log Scale)',
              fontsize=14, fontweight='bold', color=LABEL_CLR, pad=15)
ax2.set_ylabel('Speedup Ratio vs. Sequential (Higher is Better)',
               fontsize=11, fontweight='bold', color=AXIS_CLR)
_annotate_bars(ax2, bars2, speedup_factors, lambda s: f'{s:.2f}x')

plt.suptitle(
    'Parallel & GPU Computing Lab — Experiment 1 Performance Analysis',
    fontsize=16, fontweight='bold', color=LABEL_CLR, y=1.02,
)
plt.tight_layout()
plt.savefig('images/performance_comparison_charts.png', dpi=300, bbox_inches='tight')
plt.close()

# ── 2. Standalone execution-time chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 6), facecolor='#FAFAFA')
bars = ax.bar(paradigms, exec_times, color=bar_colors,
              width=0.5, edgecolor=EDGE_CLR, linewidth=1.5)
for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)
_style_axis(ax)
ax.set_title('Matrix Multiplication (4000×4000) — Execution Time Comparison',
             fontsize=14, fontweight='bold', color=LABEL_CLR, pad=15)
ax.set_ylabel('Execution Time (Seconds, Log Scale)',
              fontsize=12, fontweight='bold', color=AXIS_CLR)
for bar, t in zip(bars, exec_times):
    y = bar.get_height()
    label = f'{t:.2f} s' if t >= 1 else f'{t:.3f} s'
    ax.text(bar.get_x() + bar.get_width() / 2.0, y * 1.35, label,
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=LABEL_CLR, bbox=LABEL_BOX)
plt.tight_layout()
plt.savefig('images/execution_time_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# ── 3. Standalone speedup chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 6), facecolor='#FAFAFA')
bars = ax.bar(paradigms, speedup_factors, color=bar_colors,
              width=0.5, edgecolor=EDGE_CLR, linewidth=1.5)
for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)
_style_axis(ax)
ax.set_title('Speedup Factor Relative to Sequential Baseline',
             fontsize=14, fontweight='bold', color=LABEL_CLR, pad=15)
ax.set_ylabel('Speedup Factor (×, Log Scale)',
              fontsize=12, fontweight='bold', color=AXIS_CLR)
for bar, s in zip(bars, speedup_factors):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, y * 1.35, f'{s:.2f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=LABEL_CLR, bbox=LABEL_BOX)
plt.tight_layout()
plt.savefig('images/speedup_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print('All benchmark charts regenerated successfully.')

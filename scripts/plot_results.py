"""Render the published GBM comparison from its exported evaluation scores."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/gbm_metrics.csv'))
    parser.add_argument('--output', type=Path, default=Path('docs/figures'))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = data.loc[data.reference.eq('denoised') & data.method.isin(['fixed', 'network'])]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    types = ['Myeloid', 'Oligodendrocytes', 'tumor']
    routes = [('patient_PW034', 'Same patient · PW034 → PW034'),
              ('patient_PW030', 'Different patient · PW030 → PW034')]
    colors = {'fixed': '#4176AD', 'network': '#15988A'}
    for metric, label in [('paper_sse', 'SSE'), ('pearson', 'Pearson'), ('r2', 'R²')]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
        for ax, (route, title) in zip(axes, routes):
            values = data.loc[data.scenario.eq(route)].pivot(index='cell_type', columns='method', values=metric).loc[types]
            positions = np.arange(len(types))
            for method, offset, name in [('fixed', -.19, 'Original mean shift'), ('network', .19, 'Cell-dependent shift')]:
                bars = ax.bar(positions + offset, values[method], width=.36, color=colors[method], label=name)
                ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
            ax.set_xticks(positions, types, rotation=25, ha='right')
            ax.set_title(title, fontsize=11, pad=13)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.18)
            ax.set_ylim(0, float(data[metric].max()) * 1.18)
        axes[0].set_ylabel(label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .9))
        fig.savefig(args.output / (metric + '.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    labels, improvements = [], []
    for route, _ in routes:
        values = data.loc[data.scenario.eq(route)].pivot(index='cell_type', columns='method', values='paper_sse').loc[types]
        for ct in types:
            labels.append(('Same patient' if route == 'patient_PW034' else 'Different patient') + ' · ' + ct)
            improvements.append((values.loc[ct, 'fixed'] - values.loc[ct, 'network']) / values.loc[ct, 'fixed'] * 100)
    bars = ax.barh(labels, improvements, color=[colors['network'] if x >= 0 else '#C97163' for x in improvements])
    ax.bar_label(bars, labels=[f'{x:+.1f}%' for x in improvements], padding=5)
    ax.axvline(0, color='#526070', linewidth=.8)
    ax.invert_yaxis()
    ax.set_xlim(min(improvements)-10, max(improvements)+14)
    ax.set_xlabel('SSE reduction relative to the original mean shift (%)')
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.18)
    fig.tight_layout()
    fig.savefig(args.output / 'sse_improvement.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

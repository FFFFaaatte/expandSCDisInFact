"""Aggregate comparable runs, plot raw-reference MSE and write a short report."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs')
    args = p.parse_args()
    root = Path(args.root)
    paths = [x for x in root.rglob('metrics.csv') if not {'smoke','demo'}.intersection(x.relative_to(root).parts)]
    if not paths:
        raise ValueError('No scientific runs found (smoke results are deliberately excluded)')
    data = pd.concat([pd.read_csv(x) for x in paths],ignore_index=True)
    keys = ['experiment','scenario','cell_type','method','reference']
    summary = data.groupby(keys)[['mse','pearson','r2','delta_pearson','mmd']].agg(['mean','std','count'])
    summary.to_csv(root/'summary.csv')
    raw = data.loc[data.reference.eq('raw')]
    # Pair seeds/scenarios/types before computing improvements; never compare different splits.
    pivot = raw.pivot(index=['experiment','seed','scenario','cell_type'],columns='method',values='mse')
    pivot['network_improvement_percent'] = (pivot.fixed-pivot.network)/pivot.fixed.replace(0,np.nan)*100
    pivot.to_csv(root/'paired_comparisons.csv')
    experiments = sorted(raw.experiment.unique())
    fig,axes = plt.subplots(1,len(experiments),figsize=(5*len(experiments),4),squeeze=False)
    for ax,experiment in zip(axes[0],experiments):
        table = raw.loc[raw.experiment.eq(experiment)].groupby(['seed','method']).mse.mean().unstack()
        table.mean().plot.bar(ax=ax,yerr=table.std().fillna(0),capsize=3)
        ax.set_title(experiment); ax.set_ylabel('Mean cell-type MSE (raw reference)')
    fig.tight_layout(); fig.savefig(root/'comparison.png',dpi=180); plt.close(fig)
    lines = ['# Cell-dependent shift: preliminary results','',
        'Primary evaluation uses the same observed target reference for every method. Denoised-reference scores are separate.',
        'The backbone is frozen; only the constant adapter or the cell-dependent adapter is trained.',
        'Cell-type centroids and distribution agreement do not identify individual-cell causal responses.','',
        '## Paired raw-reference comparisons','']
    for experiment,group in pivot.groupby(level='experiment'):
        values = group.network_improvement_percent
        lines.append('- %s: mean MSE improvement %.2f%%; %d/%d cell-type/scenario/seed comparisons improved; %d seed(s).' %
            (experiment,values.mean(),(values>0).sum(),values.notna().sum(),group.index.get_level_values('seed').nunique()))
    lines += ['','Positive percentages mean lower MSE than the original fixed shift. Inspect paired_comparisons.csv for regressions.',
        'Standard deviations across three seeds are descriptive, not strong statistical evidence.',
        'The published source settings and preprocessing are not guaranteed to reproduce every paper number exactly.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(root/'REPORT.md')

if __name__ == '__main__':
    main()

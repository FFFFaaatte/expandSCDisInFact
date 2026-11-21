"""Shared-reference selection and compact evaluation displays."""
METHODS = ('fixed', 'network')


def denoised_results(frame):
    """Select existing scores without rerunning inference or modifying the input."""
    selected = frame.loc[
        frame['reference'].eq('denoised') & frame['method'].isin(METHODS)
    ].copy()
    if selected.empty:
        raise ValueError('No denoised-reference scores are available in the supplied results.')
    return selected.sort_values(['scenario', 'cell_type', 'method']).reset_index(drop=True)


def comparison_table(frame):
    selected = denoised_results(frame)
    return selected[['scenario', 'cell_type', 'method', 'paper_sse', 'pearson', 'r2']]


def paired_improvements(frame):
    import numpy as np
    selected = denoised_results(frame)
    paired = selected.pivot(index=['scenario', 'cell_type'], columns='method', values='paper_sse')
    if {'fixed', 'network'}.issubset(paired.columns):
        paired['network_improvement_percent'] = (
            (paired['fixed'] - paired['network']) / paired['fixed'].replace(0, np.nan) * 100
        )
        for metric in ['pearson', 'r2']:
            values = selected.pivot(index=['scenario', 'cell_type'], columns='method', values=metric)
            paired[metric + '_change'] = values['network'] - values['fixed']
    return paired

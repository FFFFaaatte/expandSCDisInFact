"""Canonical inputs and pre-training split manifests."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

@dataclass
class Data:
    counts: object
    meta: pd.DataFrame
    conditions: list
    genes: np.ndarray

def dense(x):
    return x.toarray().astype(np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)

def normalized(x):
    if hasattr(x, 'sum'):
        total = np.asarray(x.sum(axis=1)).reshape(-1, 1)
    if np.any(total <= 0):
        raise ValueError('Zero-library cells must be removed before normalization')
    return dense(x) / total * 100.0

def find_one(root, pattern):
    found = sorted(Path(root).rglob(pattern))
    if len(found) != 1:
        raise ValueError('Expected one %s under %s, found %d' % (pattern, root, len(found)))
    return found[0]

def validate(data):
    if data.counts.shape != (len(data.meta), len(data.genes)):
        raise ValueError('Count/metadata/gene dimensions do not match')
    x = data.counts.data if sp.issparse(data.counts) else data.counts
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Counts must be finite and nonnegative')
    if not data.meta.index.is_unique:
        raise ValueError('Cell IDs must be unique')
    if data.meta[['batch', 'cell_type'] + data.conditions].isna().any().any():
        raise ValueError('Missing biological metadata')
    keep = np.asarray(data.counts.sum(axis=1)).ravel() > 0
    return Data(data.counts[keep], data.meta.loc[keep].copy(), data.conditions, data.genes)

def load_gbm(root):
    folder = find_one(root, 'counts_rna.npz').parent
    counts = sp.load_npz(folder / 'counts_rna.npz').tocsr()
    meta = pd.read_csv(folder / 'meta_cells.csv', sep='\t', index_col=0)
    anno = pd.read_csv(folder / 'meta_cells_seurat.csv', sep='\t', index_col=0)
    if set(meta.index) != set(anno.index):
        raise ValueError('GBM annotation cell IDs do not match')
    meta['cell_type'] = anno.loc[meta.index, 'mstatus']
    meta.loc[~meta.cell_type.isin(['Myeloid', 'Oligodendrocytes', 'tumor']), 'cell_type'] = 'Other'
    meta['batch'] = meta.patient_id.astype(str)
    meta['treatment'] = meta.treatment.astype(str)
    genes = np.atleast_1d(np.loadtxt(folder / 'genes.txt', dtype=str))
    return validate(Data(counts, meta, ['treatment'], genes))

def load_simulated(root, seed, setting='2conds_base_10000_500_0.4_100_8'):
    folders = [p for p in Path(root).rglob(setting) if p.is_dir()]
    if len(folders) != 1:
        raise ValueError('Expected exactly one simulation setting %s; found %s' % (setting, folders))
    folder = folders[0]
    rng = np.random.default_rng(seed)
    blocks, frames = [], []
    conditions = [('ctrl', 'healthy'), ('ctrl', 'severe'), ('stim', 'healthy'), ('stim', 'severe')]
    for batch in range(2):
        labels = pd.read_csv(folder / ('cell_label%d.txt' % (batch + 1)), sep='\t', index_col=0).values.squeeze()
        # One condition per underlying simulated cell, exactly as the prediction scripts.
        chunks = np.array_split(rng.permutation(len(labels)), 4)
        for (treatment, severity), indices in zip(conditions, chunks):
            matrix = pd.read_csv(folder / ('GxC%d_%s_%s.txt' % (batch + 1, treatment, severity)), sep='\t', header=None).values.T
            if len(matrix) != len(labels):
                raise ValueError('Simulation cell labels do not match counts')
            blocks.append(sp.csr_matrix(matrix[indices], dtype=np.float32))
            frames.append(pd.DataFrame({'batch': str(batch), 'treatment': treatment, 'severity': severity,
                'cell_type': labels[indices].astype(str), 'base_id': ['b%d-c%d' % (batch, i) for i in indices]},
                index=['b%d-c%d' % (batch, i) for i in indices]))
    return validate(Data(sp.vstack(blocks).tocsr(), pd.concat(frames), ['treatment', 'severity'], np.array(['gene_%d' % i for i in range(blocks[0].shape[1])])) )

def split(data, experiment, seed, min_cells=50):
    m = data.meta.copy()
    role = pd.Series('train', index=m.index)
    held_type = None
    if experiment == 'gbm':
        test = m.sample_id.eq('PW034-705')
        if not test.any():
            raise ValueError('Missing official GBM held-out sample PW034-705')
        role.loc[test] = 'test'
    elif experiment == 'gbm_celltype':
        table = pd.crosstab(m.cell_type, m.treatment)
        candidates = table.loc[table.min(axis=1) >= min_cells].index.tolist()
        candidates = [c for c in candidates if c != 'Other']
        if not candidates:
            raise ValueError('No cell type has sufficient control and treatment cells')
        held_type = 'Myeloid' if 'Myeloid' in candidates else table.loc[candidates].min(axis=1).idxmax()
        role.loc[m.cell_type.eq(held_type) & m.treatment.eq('0.2 uM panobinostat')] = 'test'
    elif experiment == 'sim_oos':
        role.loc[m.treatment.eq('ctrl') & m.severity.eq('severe')] = 'test'
    elif experiment != 'sim_is':
        raise ValueError(experiment)
    rng = np.random.default_rng(seed)
    keys = ['batch', 'cell_type'] + data.conditions
    for _, group in m.loc[role.eq('train')].groupby(keys, sort=True):
        ids = rng.permutation(group.index.to_numpy())
        if len(ids) < 5:
            continue
        n = max(1, int(len(ids) * .2))
        role.loc[ids[:n]] = 'val'
        if experiment == 'sim_is':
            role.loc[ids[n:2*n]] = 'test'
    if not role.eq('test').any() or not role.eq('val').any():
        raise ValueError('Empty validation or test split')
    if 'base_id' in m and m.base_id.duplicated().any():
        raise ValueError('Repeated underlying simulated cell')
    m['role'] = role
    return m, held_type


def load_demo(root):
    folder = Path(root) / 'demo'
    counts = sp.load_npz(folder / 'counts.npz').tocsr()
    meta = pd.read_csv(folder / 'meta_cells.csv', index_col=0).rename(columns={
        'condition 1':'treatment', 'condition 2':'severity', 'annos':'cell_type'})
    meta['batch'] = meta.batch.astype(str)
    meta.index = pd.Index(['b%s-c%s' % (b,i) for b,i in zip(meta.batch,meta.index)])
    return validate(Data(counts,meta,['treatment','severity'],np.array(['gene_%d'%i for i in range(counts.shape[1])])))

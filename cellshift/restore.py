"""Restore a frozen GBM experiment for additional adapter experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import load_gbm, normalized
from .model import Backbone, bandwidth_from_training
from .run import subset
from .upstream import create_scdisinfact_dataset, scdisinfact


def restore_gbm_run(root, directory=None):
    root = Path(root)
    if directory is None:
        candidates = [p.parent for p in (root / 'runs_notebook').glob('*/gbm/seed_0/backbone.pt')
                      if (p.parent / 'metrics.csv').exists()]
        if not candidates:
            raise FileNotFoundError('No saved GBM backbone with evaluation scores was found.')
        directory = max(candidates, key=lambda p: (p / 'metrics.csv').stat().st_mtime)
    out = Path(directory)
    config = json.loads((out / 'config.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(out / 'backbone.pt', map_location=device, weights_only=False)
    data = load_gbm(root / 'data')
    saved = pd.read_csv(out / 'split.csv', index_col=0)
    if not saved.index.equals(data.meta.index):
        raise ValueError('Saved split does not match the current GBM cells.')
    if not saved.role.isin(['train', 'val', 'test']).all():
        raise ValueError('Invalid saved split roles.')
    manifest = data.meta.copy()
    manifest['role'] = saved['role']
    training = subset(data, manifest.role.eq('train'))
    validation = subset(data, manifest.role.eq('val'))
    dictionary = create_scdisinfact_dataset(training.counts, training.meta.copy(),
        condition_key=data.conditions, batch_key='batch', batch_cond_key='sample_id')
    for key, saved_names in checkpoint['matching_dict'].items():
        current_names = dictionary['matching_dict'][key]
        if key == 'cond_names':
            matched = len(saved_names) == len(current_names) and all(np.array_equal(a,b) for a,b in zip(saved_names,current_names))
        else:
            matched = np.array_equal(saved_names,current_names)
        if not matched:
            raise ValueError('Checkpoint label mapping does not match the training partition: ' + key)
    dictionary['scaler'] = checkpoint['scaler']
    dims = checkpoint['dims']
    base = scdisinfact(dictionary, Ks=dims, batch_size=64, lr=5e-4,
                      seed=config['seed'], device=device)
    base.load_state_dict(checkpoint['state_dict'])
    base.eval()
    backbone = Backbone(base, data.conditions)
    train_encoded = backbone.encode(training.counts, training.meta)
    val_encoded = backbone.encode(validation.counts, validation.meta)
    backbone.fit_centroids(train_encoded)
    sample = np.random.default_rng(config['seed']).choice(len(training.meta), min(len(training.meta),512), replace=False)
    bandwidth = bandwidth_from_training(np.log1p(normalized(training.counts[sample])), config['seed'])
    return dict(data=data, manifest=manifest, training=training, validation=validation,
                backbone=backbone, base=base, dims=dims, train_encoded=train_encoded,
                val_encoded=val_encoded, bandwidth=bandwidth, SEED=config['seed'],
                ADAPTER_EPOCHS=config['adapter_epochs'], STEPS=config['steps'], OUT=out)

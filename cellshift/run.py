"""Train and evaluate a complete split. Run with python -m cellshift.run."""
import argparse
import hashlib
import json
import platform
import scipy
import sklearn
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from .data import Data, dense, normalized, load_gbm, load_simulated, load_demo, split
from .model import Backbone, bandwidth_from_training, mmd, train_adapter
from .upstream import REVISION, create_scdisinfact_dataset, scdisinfact


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')

def subset(data, mask):
    return Data(data.counts[np.asarray(mask)],data.meta.loc[mask].copy(),data.conditions,data.genes)

def scenarios(data, manifest, experiment, held_type, min_cells):
    m = manifest
    scenarios = []
    if experiment == 'gbm':
        target = m.role.eq('test')
        for patient in ['PW034','PW030']:
            source = m.batch.eq(patient) & m.treatment.eq('vehicle (DMSO)') & ~m.role.eq('test')
            scenarios.append(('patient_'+patient, source, target, ['0.2 uM panobinostat'],'PW034'))
    elif experiment == 'gbm_celltype':
        for patient in sorted(m.loc[m.role.eq('test'),'batch'].unique()):
            target = m.role.eq('test') & m.batch.eq(patient)
            source = m.cell_type.eq(held_type) & m.batch.eq(patient) & m.treatment.eq('vehicle (DMSO)')
            scenarios.append(('held_'+held_type+'_'+patient,source,target,['0.2 uM panobinostat'],patient))
    else:
        target = m.role.eq('test') & m.treatment.eq('ctrl') & m.severity.eq('severe') & m.batch.eq('0')
        for batch in ['0','1']:
            for treatment,severity in [('stim','severe'),('ctrl','healthy'),('stim','healthy')]:
                source = m.batch.eq(batch) & m.treatment.eq(treatment) & m.severity.eq(severity)
                if experiment == 'sim_is':
                    source &= m.role.eq('test')
                scenarios.append(('%s_%s_b%s' % (treatment,severity,batch),source,target,['ctrl','severe'],'0'))
    valid, skipped = [], []
    for name,source,target,conditions,batch in scenarios:
        source_types = m.loc[source].cell_type.value_counts()
        target_types = m.loc[target].cell_type.value_counts()
        types = [x for x in target_types.index if min(target_types[x],source_types.get(x,0)) >= min_cells and x!='Other']
        if not types:
            skipped.append({'scenario':name,'reason':'No shared cell type with >= %d cells' % min_cells})
        else:
            valid.append((name,source & m.cell_type.isin(types),target & m.cell_type.isin(types),conditions,batch))
    if not valid:
        raise ValueError('No evaluable scenarios: %s' % skipped)
    return valid, skipped


def metrics(pred, target, source_mean, bandwidth, seed):
    p, t = pred.mean(0), target.mean(0)
    mse = float(np.mean((p-t)**2))
    corr = float(np.corrcoef(p,t)[0,1]) if p.std()>0 and t.std()>0 else float('nan')
    rng = np.random.default_rng(seed)
    a = torch.tensor(np.log1p(pred[rng.choice(len(pred),min(256,len(pred)),replace=False)]))
    b = torch.tensor(np.log1p(target[rng.choice(len(target),min(256,len(target)),replace=False)]))
    return {'mse':mse,'paper_sse':mse*len(p),'pearson':corr,'r2':float(r2_score(t,p)),
            'delta_pearson':float(np.corrcoef(p-source_mean,t-source_mean)[0,1]) if (p-source_mean).std()>0 and (t-source_mean).std()>0 else float('nan'),
            'mmd':float(mmd(a.float(),b.float(),bandwidth))}


def synthetic(seed):
    """Synthetic integration-test fixture; excluded from scientific evaluation."""
    rng = np.random.default_rng(seed)
    frames, blocks = [], []
    for batch in ['0','1']:
        for treatment in ['ctrl','stim']:
            for severity in ['healthy','severe']:
                n = 60
                ct = np.array(['A']*30+['B']*30)
                rate = np.ones((n,24))*2
                rate[ct=='A',:6] += 4
                rate[:,6:12] += (treatment=='stim')*3
                rate[:,12:18] += (severity=='severe')*3
                rate += int(batch)
                x = rng.poisson(rate).astype(np.float32)
                blocks.append(x)
                frames.append(pd.DataFrame({'batch':batch,'treatment':treatment,'severity':severity,'cell_type':ct}))
    meta = pd.concat(frames,ignore_index=True)
    meta.index = ['fixture_%d'%i for i in range(len(meta))]
    return Data(np.concatenate(blocks),meta,['treatment','severity'],np.array(['g%d'%i for i in range(24)]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['gbm','gbm_celltype','sim_is','sim_oos','smoke','demo'],required=True)
    p.add_argument('--data-root',default='data')
    p.add_argument('--output',default='runs')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--base-epochs',type=int)
    p.add_argument('--adapter-epochs',type=int,default=100)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--min-cells',type=int,help='Default: 50 GBM, 10 simulation/demo')
    p.add_argument('--threads',type=int,default=4)
    args = p.parse_args()
    if (args.base_epochs is not None and args.base_epochs < 1) or args.adapter_epochs < 1 or args.steps < 1 or args.threads < 1 or (args.min_cells is not None and args.min_cells < 2):
        p.error('Epochs, steps and threads must be positive; min-cells must be at least 2')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    out = Path(args.output)/args.experiment/('seed_%d'%args.seed)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError('Use a new output directory; existing run will not be overwritten: '+str(out))
    started = time.monotonic()
    is_smoke = args.experiment=='smoke'
    experiment = 'sim_is' if args.experiment in ['smoke','demo'] else args.experiment
    if args.min_cells is None:
        args.min_cells = 50 if experiment.startswith('gbm') else 10
    data = synthetic(args.seed) if is_smoke else load_demo(args.data_root) if args.experiment=='demo' else (load_gbm(args.data_root) if experiment.startswith('gbm') else load_simulated(args.data_root,args.seed))
    manifest, held_type = split(data,experiment,args.seed,args.min_cells)
    manifest.to_csv(out/'split.csv')
    pd.Series(data.genes).to_csv(out/'genes.csv',index=False,header=False)
    config = vars(args).copy()
    config.update({'upstream_revision':REVISION,'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
        'gpu':torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,'held_cell_type':held_type,'smoke_only':is_smoke,'demonstration_only':args.experiment in ['smoke','demo'],'shape':data.counts.shape,
        'base_hyperparameters':{'dims':[8]+[2]*len(data.conditions),'batch_size':64,'lr':5e-4,'likelihood':'NB',
            'mmd_shared':1e-4,'mmd_unshared':1e-4,'kl_shared':1e-5,'kl_unshared':1e-2,'classification':1,'group_lasso':1},
        'adapter_hyperparameters':{'hidden':[64,64],'lr':1e-3,'patience':10,'max_group_cells':128,'kernel_scales':[.25,.5,1,2,4]},
        'split_sha256':hashlib.sha256((out/'split.csv').read_bytes()).hexdigest(),
        'base_epochs_effective':args.base_epochs if args.base_epochs is not None else (100 if experiment.startswith('gbm') else 50)})
    dump(out/'config.json',config)
    manifest.groupby(['role','batch','cell_type']+data.conditions).size().rename('cells').to_csv(out/'group_counts.csv')
    training = subset(data,manifest.role.eq('train'))
    validation = subset(data,manifest.role.eq('val'))
    dictionary = create_scdisinfact_dataset(training.counts,training.meta.copy(),condition_key=data.conditions,batch_key='batch',
        batch_cond_key='sample_id' if experiment.startswith('gbm') else None)
    dims = [8]+[2]*len(data.conditions)
    base = scdisinfact(dictionary,Ks=dims,batch_size=64,interval=10,lr=5e-4,seed=args.seed,device=torch.device(args.device))
    before = time.monotonic()
    losses = base.train_model(nepochs=max(0,config['base_epochs_effective']-1),recon_loss='NB')
    base.eval()
    torch.save({'state_dict':base.state_dict(),'scaler':dictionary['scaler'],'matching_dict':dictionary['matching_dict'],
        'conditions':data.conditions,'dims':dims,'config':config},out/'backbone.pt')
    dump(out/'base_losses.json',losses)
    backbone = Backbone(base,data.conditions)
    train_encoded = backbone.encode(training.counts,training.meta)
    val_encoded = backbone.encode(validation.counts,validation.meta)
    backbone.fit_centroids(train_encoded)
    # Official prediction parity on a small input, including NB output rescaling.
    tiny = subset(training,np.arange(len(training.meta))<8)
    labels = [names[-1] for names in backbone.labels]
    target_ids = [names.index(value) for names,value in zip(backbone.labels,labels)]
    official = base.predict_counts(dense(tiny.counts),tiny.meta.copy(),data.conditions,'batch',list(labels),backbone.batches[0])
    official = official/official.sum(1,keepdims=True)*100
    ours = backbone.predict(backbone.encode(tiny.counts,tiny.meta),target_ids,backbone.batches[0])
    np.testing.assert_allclose(ours,official,rtol=2e-4,atol=2e-4)
    base_seconds = time.monotonic()-before
    bandwidth = bandwidth_from_training(np.log1p(normalized(training.counts)),args.seed)
    adapters = {}
    for mode in ['constant','network']:
        adapter, history = train_adapter(backbone,train_encoded,val_encoded,training.meta,validation.meta,dims,mode,bandwidth,args.seed,
            epochs=args.adapter_epochs,steps=args.steps)
        adapters[mode] = adapter
        torch.save({'state_dict':adapter.state_dict(),'mode':mode,'dims':dims,'categories':[len(x) for x in backbone.labels],
            'bandwidth':bandwidth,'centroids':[x.cpu() for x in backbone.centroids]},out/(mode+'.pt'))
        pd.DataFrame(history).to_csv(out/(mode+'_history.csv'),index=False)
    jobs, skipped = scenarios(data,manifest,experiment,held_type,args.min_cells)
    dump(out/'skipped.json',skipped)
    rows = []
    for name,source_mask,target_mask,labels,batch in jobs:
        source,target = subset(data,source_mask),subset(data,target_mask)
        source_encoded = backbone.encode(source.counts,source.meta)
        target_encoded = backbone.encode(target.counts,target.meta)
        references = {'raw':normalized(target.counts),'denoised':backbone.predict(target_encoded,mode='identity')}
        ids = [names.index(value) for names,value in zip(backbone.labels,labels)]
        predictions = {'no_shift':backbone.predict(source_encoded,ids,batch,mode='identity'),
                       'fixed':backbone.predict(source_encoded,ids,batch)}
        predictions.update({mode:backbone.predict(source_encoded,ids,batch,adapter) for mode,adapter in adapters.items()})
        for method,prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise RuntimeError('Nonfinite predictions')
            for ct in sorted(target.meta.cell_type.unique()):
                si,ti = source.meta.cell_type.eq(ct).values,target.meta.cell_type.eq(ct).values
                source_mean = normalized(source.counts[si]).mean(0)
                for reference,truth in references.items():
                    rows.append(dict(experiment=args.experiment,seed=args.seed,scenario=name,cell_type=ct,method=method,
                        reference=reference,n_source=int(si.sum()),n_target=int(ti.sum()),**metrics(prediction[si],truth[ti],source_mean,bandwidth,args.seed)))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    dump(out/'timing.json',{'base_and_encoding_seconds':base_seconds,'total_seconds':time.monotonic()-started,'bandwidth':bandwidth,'official_prediction_parity':True})
    print('Completed:',out,flush=True)

if __name__ == '__main__':
    main()

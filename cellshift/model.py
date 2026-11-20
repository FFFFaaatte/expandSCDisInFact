"""Frozen-backbone adapters and distribution matching."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import dense, normalized

class Shift(nn.Module):
    def __init__(self, dims, categories, mode='network'):
        super().__init__()
        self.categories = categories
        self.mode = mode
        self.branches = nn.ModuleList()
        for d, n in zip(dims[1:], categories):
            if mode == 'network':
                self.branches.append(nn.Sequential(nn.Linear(dims[0]+d+2*n,64), nn.ReLU(),
                    nn.Linear(64,64), nn.ReLU(), nn.Linear(64,d)))
            elif mode == 'constant':
                self.branches.append(nn.Embedding(n*n,d))
                nn.init.zeros_(self.branches[-1].weight)
            else:
                raise ValueError(mode)

    def forward(self, shared, unshared, source, target):
        result = []
        for j, (z, n, branch) in enumerate(zip(unshared, self.categories, self.branches)):
            s, t = source[:,j], target[:,j]
            if self.mode == 'network':
                delta = branch(torch.cat([shared,z,F.one_hot(s,n).float(),F.one_hot(t,n).float()],dim=1))
            else:
                delta = branch(s*n+t)
            result.append(z + delta * (s != t).float()[:,None])
        return result

def mmd(x, y, bandwidth):
    # Biased MMD: stable for small groups and no assumption of paired cells.
    def kernel(a,b):
        distances = torch.cdist(a,b).square()
        return sum(torch.exp(-distances/(2*bandwidth*scale)) for scale in [.25, .5, 1, 2, 4])/5
    return kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean()

def expression(mu):
    return torch.log1p(mu / mu.sum(1,keepdim=True).clamp_min(1e-8)*100)

def bandwidth_from_training(x, seed):
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(len(x),512), replace=False)]
    x = torch.as_tensor(x, dtype=torch.float32)
    d = torch.pdist(x).square()
    positive = d[d > 0]
    return max(float(positive.median()) if len(positive) else 1.0, 1e-6)

class Backbone:
    def __init__(self, model, conditions):
        self.model = model.eval()
        self.conditions = conditions
        self.device = model.device
        matching = model.data_dict['matching_dict']
        self.labels = [list(x) for x in matching['cond_names']]
        self.batches = list(matching['batch_name'])
        self.centroids = None

    def ids(self, meta):
        batch = torch.tensor([self.batches.index(x) for x in meta.batch],device=self.device)[:,None]
        labels = torch.tensor(np.stack([[names.index(x) for x in meta[key]] for key,names in zip(self.conditions,self.labels)],axis=1),device=self.device)
        return batch, labels

    def encode(self, counts, meta, chunk=256):
        batch, labels = self.ids(meta)
        shared, unshared = [], [[] for _ in self.conditions]
        for start in range(0,len(meta),chunk):
            stop = start + chunk
            x = np.log1p(normalized(counts[start:stop]))
            x = self.model.data_dict['scaler'].transform(x).astype(np.float32)
            with torch.no_grad():
                inf = self.model.inference(torch.tensor(x,device=self.device),batch[start:stop])
            shared.append(inf['mu_c'].detach().cpu())
            for j,z in enumerate(inf['mu_d']):
                unshared[j].append(z.detach().cpu())
        return {'shared':torch.cat(shared), 'unshared':[torch.cat(x) for x in unshared],
                'batch':batch.cpu(), 'labels':labels.cpu()}

    def fit_centroids(self, encoded):
        self.centroids = [torch.stack([z[encoded['labels'][:,j]==i].mean(0) for i in range(len(names))]).to(self.device)
                          for j,(z,names) in enumerate(zip(encoded['unshared'],self.labels))]

    def decode(self, encoded, indices, target=None, target_batch=None, adapter=None, mode='fixed'):
        shared = encoded['shared'][indices].to(self.device)
        unshared = [z[indices].to(self.device) for z in encoded['unshared']]
        source = encoded['labels'][indices].to(self.device)
        batch = encoded['batch'][indices].to(self.device)
        target = source if target is None else torch.as_tensor(target,device=self.device).expand_as(source)
        if adapter is not None:
            unshared = adapter(shared,unshared,source,target)
        elif mode == 'fixed':
            unshared = [z+means[target[:,j]]-means[source[:,j]] for j,(z,means) in enumerate(zip(unshared,self.centroids))]
        elif mode != 'identity':
            raise ValueError(mode)
        if target_batch is not None:
            batch = torch.full_like(batch,self.batches.index(target_batch))
        return self.model.generative(shared,unshared,batch)['mu']

    def predict(self, encoded, target=None, target_batch=None, adapter=None, mode='fixed'):
        result = []
        with torch.no_grad():
            for start in range(0,len(encoded['shared']),256):
                mu = self.decode(encoded,slice(start,start+256),target,target_batch,adapter,mode)
                result.append((mu/mu.sum(1,keepdim=True).clamp_min(1e-8)*100).cpu().numpy())
        return np.concatenate(result)

def transitions(meta, conditions, min_cells=2):
    """Only single-factor, same-batch/same-cell-type training transitions."""
    groups = {}
    keys = ['batch','cell_type']+conditions
    for key, indices in meta.reset_index(drop=True).groupby(keys,sort=True).groups.items():
        if len(indices) >= min_cells:
            groups[key] = np.asarray(indices)
    result = []
    for source, source_ids in groups.items():
        for target, target_ids in groups.items():
            if source[:2] == target[:2] and sum(a!=b for a,b in zip(source[2:],target[2:])) == 1:
                result.append((source, target, source_ids, target_ids))
    return result

def train_adapter(backbone, train, val, train_meta, val_meta, dims, mode, bandwidth, seed, epochs=100, steps=50):
    torch.manual_seed(seed)
    adapter = Shift(dims,[len(x) for x in backbone.labels],mode).to(backbone.device)
    for p in backbone.model.parameters():
        p.requires_grad_(False)
    original = {k:v.detach().cpu().clone() for k,v in backbone.model.state_dict().items()}
    optimizer = torch.optim.Adam(adapter.parameters(),lr=1e-3)
    training_pairs = transitions(train_meta,backbone.conditions)
    validation_pairs = transitions(val_meta,backbone.conditions)
    if not training_pairs or not validation_pairs:
        raise ValueError('No valid transitions for adapter training/validation')
    rng = np.random.default_rng(seed)
    # Cache fixed denoised target expression; only permitted cells are encoded here.
    def targets(encoded):
        return torch.as_tensor(np.log1p(backbone.predict(encoded,mode='identity')),dtype=torch.float32)
    train_targets, val_targets = targets(train), targets(val)
    def loss(encoded, target_expr, pair, generator):
        _, target, src_ids, tgt_ids = pair
        src = generator.choice(src_ids,min(128,len(src_ids)),replace=False)
        tgt = generator.choice(tgt_ids,min(128,len(tgt_ids)),replace=False)
        labels = [names.index(value) for names,value in zip(backbone.labels,target[2:])]
        mu = backbone.decode(encoded,src,labels,adapter=adapter)
        return mmd(expression(mu),target_expr[tgt].to(backbone.device),bandwidth)
    best, bad, history, best_state = float('inf'), 0, [], None
    for epoch in range(epochs):
        adapter.train()
        losses = []
        for _ in range(steps):
            pair = training_pairs[int(rng.integers(len(training_pairs)))]
            optimizer.zero_grad()
            value = loss(train,train_targets,pair,rng)
            value.backward()
            optimizer.step()
            losses.append(float(value.detach()))
        adapter.eval()
        validation_rng = np.random.default_rng(seed+10000)
        with torch.no_grad():
            vl = np.mean([float(loss(val,val_targets,pair,validation_rng)) for pair in validation_pairs])
        history.append({'epoch':epoch+1,'train_mmd':float(np.mean(losses)),'val_mmd':float(vl)})
        print('%s epoch %d train %.5g val %.5g' % (mode,epoch+1,np.mean(losses),vl),flush=True)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
        else:
            bad += 1
        if bad >= 10:
            break
    if best_state is None:
        raise RuntimeError('Adapter produced no finite validation checkpoint')
    adapter.load_state_dict(best_state)
    assert all(torch.equal(original[k],v.detach().cpu()) for k,v in backbone.model.state_dict().items()), 'Backbone changed'
    return adapter.eval(), history

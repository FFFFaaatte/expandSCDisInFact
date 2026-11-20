"""Inspect data and split feasibility before launching GPU training."""
import argparse
from pathlib import Path
from .data import load_gbm, load_simulated, load_demo, split
from .run import scenarios

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['gbm','simulated','demo'],required=True)
    p.add_argument('--data-root',default='data')
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    data = {'gbm':load_gbm,'demo':load_demo}.get(args.dataset,lambda root:load_simulated(root,args.seed))(args.data_root)
    print('Cells x genes:',data.counts.shape)
    # Upstream materializes dense counts, normalized counts, size factors and tensors.
    print('One dense float32 matrix: %.2f GiB; budget several copies in host RAM.' % (data.counts.shape[0]*data.counts.shape[1]*4/1024**3))
    print(data.meta.groupby(['batch','cell_type']+data.conditions).size().to_string())
    experiments=['gbm','gbm_celltype'] if args.dataset=='gbm' else ['sim_is','sim_oos']
    for experiment in experiments:
        minimum=50 if args.dataset=='gbm' else 10
        manifest,held=split(data,experiment,args.seed,minimum)
        print('\n',experiment,'held type:',held,'roles:',manifest.role.value_counts().to_dict())
        jobs,skipped=scenarios(data,manifest,experiment,held,minimum)
        print('Evaluable scenarios:',[x[0] for x in jobs]);print('Skipped:',skipped)

if __name__=='__main__':
    main()

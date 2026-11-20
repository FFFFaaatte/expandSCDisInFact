import io
import tempfile
import tarfile
import unittest
from pathlib import Path
import numpy as np
import torch
from cellshift.data import split, normalized
from cellshift.download import extract
from cellshift.model import Shift, mmd, transitions
from cellshift.run import synthetic

class ScientificControls(unittest.TestCase):
    def test_oos_combination_and_validation_excluded(self):
        data = synthetic(4)
        m,_ = split(data,'sim_oos',4,2)
        forbidden = m.treatment.eq('ctrl') & m.severity.eq('severe')
        self.assertTrue(m.loc[forbidden,'role'].eq('test').all())
        self.assertFalse(m.loc[m.role.eq('train')].index.intersection(m.loc[m.role.eq('val')].index).size)
        train_m = m.loc[m.role.eq('train')]
        for source,target,*_ in transitions(train_m,data.conditions):
            self.assertEqual(source[:2],target[:2])
            self.assertEqual(sum(a!=b for a,b in zip(source[2:],target[2:])),1)
            self.assertNotEqual(target[2:],('ctrl','severe'))

    def test_gbm_celltype_holdout_removes_all_treated_cells(self):
        data = synthetic(0)
        data.meta['treatment'] = data.meta.treatment.map({'ctrl':'vehicle (DMSO)','stim':'0.2 uM panobinostat'})
        data.meta['cell_type'] = data.meta.cell_type.map({'A':'Myeloid','B':'tumor'})
        data.conditions = ['treatment']
        m,ct = split(data,'gbm_celltype',0,50)
        self.assertEqual(ct,'Myeloid')
        held = m.cell_type.eq(ct) & m.treatment.eq('0.2 uM panobinostat')
        self.assertTrue(m.loc[held,'role'].eq('test').all())
        self.assertTrue(m.loc[m.cell_type.eq(ct)&~held,'role'].isin(['train','val']).all())

    def test_identity_transition_and_cell_dependence(self):
        torch.manual_seed(0)
        adapter = Shift([8,2,2],[2,2])
        shared = torch.randn(8,8)
        zs = [torch.randn(8,2),torch.randn(8,2)]
        source = torch.zeros(8,2,dtype=torch.long)
        out = adapter(shared,zs,source,source)
        for a,b in zip(zs,out):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        target = source.clone(); target[:,0] = 1
        out = adapter(shared,zs,source,target)
        torch.testing.assert_close(out[1],zs[1],rtol=0,atol=0)
        self.assertGreater(float((out[0]-zs[0]).std(dim=0).sum()),0)
        out[0].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter.parameters()))

    def test_mmd_unpaired_permutation_and_normalization(self):
        x = torch.randn(20,4)
        torch.testing.assert_close(mmd(x,x[torch.randperm(20)],1),torch.tensor(0.),atol=1e-6,rtol=0)
        np.testing.assert_allclose(normalized(np.ones((3,5))).sum(1),100)
        with self.assertRaises(ValueError):
            normalized(np.zeros((1,5)))

    def test_archive_path_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name,kind in [('../escape',tarfile.REGTYPE),('link',tarfile.SYMTYPE)]:
                archive = Path(tmp)/'bad.tar'
                with tarfile.open(archive,'w') as tar:
                    info = tarfile.TarInfo(name); info.type=kind
                    info.linkname='/tmp/outside'; info.size=0
                    tar.addfile(info,io.BytesIO(b''))
                with self.assertRaises(ValueError):
                    extract(archive,Path(tmp)/'out')

class InputAndReporting(unittest.TestCase):
    def test_gbm_loader_aligns_annotations_by_cell_id(self):
        import pandas as pd
        import scipy.sparse as sp
        from cellshift.data import load_gbm
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'GBM_treatment/Fig4/processed';folder.mkdir(parents=True)
            sp.save_npz(folder/'counts_rna.npz',sp.csr_matrix(np.ones((3,4))))
            pd.DataFrame({'patient_id':['PW034']*3,'sample_id':['s']*3,'treatment':['vehicle (DMSO)']*3},index=['a','b','c']).to_csv(folder/'meta_cells.csv',sep='\t')
            pd.DataFrame({'mstatus':['tumor','Myeloid','Oligodendrocytes']},index=['c','a','b']).to_csv(folder/'meta_cells_seurat.csv',sep='\t')
            (folder/'genes.txt').write_text('g1\ng2\ng3\ng4\n')
            data=load_gbm(tmp)
            self.assertEqual(list(data.meta.cell_type),['Myeloid','Oligodendrocytes','tumor'])

    def test_simulation_assigns_each_underlying_cell_once(self):
        import pandas as pd
        from cellshift.data import load_simulated
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'2conds_base_10000_500_0.4_100_8';folder.mkdir()
            for batch in [1,2]:
                pd.DataFrame({'label':['A']*20+['B']*20}).to_csv(folder/('cell_label%d.txt'%batch),sep='\t')
                for a,b in [('ctrl','healthy'),('ctrl','severe'),('stim','healthy'),('stim','severe')]:
                    pd.DataFrame(np.ones((5,40))*(batch+1)).to_csv(folder/('GxC%d_%s_%s.txt'%(batch,a,b)),sep='\t',index=False,header=False)
            data=load_simulated(tmp,0)
            self.assertEqual(data.counts.shape,(80,5))
            self.assertTrue(data.meta.base_id.is_unique)
            m,_=split(data,'sim_oos',0,2)
            self.assertEqual(m.role.eq('test').sum(),20)

    def test_report_pairs_methods_and_excludes_demo(self):
        import pandas as pd
        from unittest.mock import patch
        from cellshift.report import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for seed in [0,1]:
                for method,value in [('no_shift',4),('fixed',2),('constant',1.8),('network',1)]:
                    rows.append(dict(experiment='gbm',seed=seed,scenario='test',cell_type='A',method=method,reference='raw',mse=value,pearson=.8,r2=.7,delta_pearson=.5,mmd=.1))
            folder=root/'gbm/seed_0';folder.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
            demo=root/'demo/seed_0';demo.mkdir(parents=True)
            pd.DataFrame(rows).assign(experiment='demo').to_csv(demo/'metrics.csv',index=False)
            with patch('sys.argv',['report','--root',tmp]):main()
            comparisons=pd.read_csv(root/'paired_comparisons.csv')
            self.assertEqual(set(comparisons.experiment),{'gbm'})
            np.testing.assert_allclose(comparisons.network_improvement_percent,50)
            self.assertTrue((root/'comparison.png').stat().st_size>0)


if __name__ == '__main__':
    unittest.main()

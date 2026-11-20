# Cell-dependent shifts for scDisInFact

An experimental extension of scDisInFact that replaces its fixed latent shift with a network predicting the **full shift for each cell**. The pretrained encoders and decoder remain frozen while the shift network is trained.

The upstream implementation is pinned to a specific revision. Full scientific experiments have not yet been completed; successful smoke tests and short demo runs do not establish an improvement over the original method.

## GBM experiment notebook

[notebooks/GBM_experiment.ipynb](notebooks/GBM_experiment.ipynb) implements the GBM sample-holdout experiment, including dataset composition, partitioning, backbone estimation, adapter optimization, and paired evaluation. Tables, optimization diagnostics, and prediction comparisons are rendered inline.

The notebook records each experiment under a separate directory in `runs_notebook`. The final section exposes the extended multi-seed benchmark through `RUN_EXTENDED`. CUDA is the default training target; CPU execution requires `ALLOW_CPU=True`.

## Environment

Use Python 3.10–3.12 in a dedicated environment. Install a CUDA-compatible PyTorch build for the GPU machine before installing the remaining requirements. The requirements accept an already installed compatible PyTorch version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m unittest discover -s tests -v
```

Run all commands from the repository root. Downloaded datasets and generated checkpoint directories are ignored by Git.

## Download and inspect data

For a short check using the authors' demo:

```bash
python -m cellshift.download demo
python -m cellshift.run --experiment demo --device cuda --base-epochs 2 --adapter-epochs 2 --steps 2 --min-cells 2 --output runs_demo
```

Download the full GBM and simulated datasets and inspect split feasibility:

```bash
python -m cellshift.download all
python -m cellshift.inspect --dataset gbm
python -m cellshift.inspect --dataset simulated
```

To download GBM alone, use `python -m cellshift.download gbm`.

The compressed GBM and simulated archives total approximately 1.35 GB. Archives are stored under `data/archives` and extracted under `data`. Interrupted downloads use `.part` files and support resumption; completed archives are checked against the MD5 checksums published on Zenodo. Allow enough disk space for both archives and extracted files.

Upstream preprocessing materializes several dense copies of the count matrix. Check the inspection output and available host RAM before training. Demo and synthetic smoke-test outputs are excluded from scientific reports.

## Run experiments

Start with GBM seed zero to assess runtime and memory requirements:

```bash
python -m cellshift.run --experiment gbm --seed 0 --device cuda
```

Run the full suite, skipping previously completed runs:

```bash
bash scripts/run_suite.sh
```

Or select individual experiments:

```bash
python -m cellshift.run --experiment gbm_celltype --seed 0 --device cuda
python -m cellshift.run --experiment sim_is --seed 0 --device cuda
python -m cellshift.run --experiment sim_oos --seed 0 --device cuda
python -m cellshift.report
```

The suite defaults to seeds 0, 1, and 2. Use `SEEDS="0" bash scripts/run_suite.sh` to evaluate seed zero only. Set `PYTHON`, `DATA_ROOT`, and `OUTPUT` to select the interpreter, input directory, and output directory.

Incomplete runs are not overwritten: use a new `--output` directory to retry. Resuming training within an epoch is not implemented.

For an integration smoke test without downloaded data or a GPU:

```bash
python -m cellshift.run --experiment smoke --device cpu --base-epochs 1 --adapter-epochs 2 --steps 2 --min-cells 2 --output runs_smoke
```

## Experimental protocol

| Experiment | Training and evaluation |
|---|---|
| `gbm` | Hold out sample `PW034-705`; predict panobinostat response from PW034 and PW030 control cells. |
| `gbm_celltype` | Hold out all treated Myeloid cells across patients; evaluate predictions from control cells in the same patient. |
| `sim_is` | Use setting `2conds_base_10000_500_0.4_100_8`; approximately 60% training, 20% validation, and 20% test within sufficiently large groups. |
| `sim_oos` | Hold out `ctrl + severe` in both batches; reserve 20% of remaining groups for validation. |

Each underlying simulated cell is assigned to exactly one condition combination, following the upstream prediction-script design. Corresponding versions of a cell cannot leak between splits. Both simulation experiments evaluate six prediction routes: changing the first factor, the second factor, or both, with or without a batch change. Only one of the paper's nine simulation settings is used.

Validation cells are excluded from backbone training. The scaler and latent centroids are fitted using training cells only. Splits and any replacement holdout cell type are recorded before training. If Myeloid lacks sufficient cells, select the eligible cell type with the largest minimum control/treated population, excluding `Other`.

Evaluation requires at least 50 source and target cells per cell type for GBM, and 10 for simulation because its test groups are smaller. Unsupported groups are recorded; an experiment with no valid evaluation scenarios fails explicitly.

### Backbone

Use latent dimensions `[8, 2]` for GBM and `[8, 2, 2]` for simulation, the upstream loss defaults, negative-binomial reconstruction, batch size 64, and learning rate `5e-4`. Train for 100 epochs on GBM and 50 on simulation. The upstream loop includes both zero and `nepochs`, so the runner subtracts one to obtain the intended number of epochs.

### Compared methods

All four methods share the same trained backbone and data split:

- `no_shift`: decode without changing biological factors; still apply the destination batch.
- `fixed`: apply the original difference between training-set latent centroids.
- `constant`: learn a cell-independent shift for each transition.
- `network`: predict the full shift using an MLP with two 64-unit ReLU hidden layers, taking the shared representation, the current factor-specific representation, and one-hot source and target labels. The original fixed shift is not added to its output.

Each condition factor has its own network. Unchanged factors remain exactly unchanged. When changing both factors, compute their shifts simultaneously from the original representations. Cell-type labels group the training loss but are not network inputs.

Training transitions change one factor within the same batch and cell type. Held-out condition combinations and held-out treated cell types never supply training targets.

### Adapter training

Train with multi-bandwidth MMD on library-size-normalized expression with a per-cell sum of 100, followed by `log1p`. Training targets are frozen-backbone reconstructions of permitted target training cells.

The base kernel bandwidth is the median squared distance among at most 512 training cells; bandwidth scales are 0.25, 0.5, 1, 2, and 4. Use Adam with learning rate `1e-3`, at most 100 epochs, and 50 steps per epoch. Sample transitions uniformly, with at most 128 cells per side. Select the checkpoint using validation MMD with patience 10.

Gradients pass through the decoder to the adapter, but the backbone weights remain frozen and are checked after training.

## Outputs and interpretation

Each command-line run records the split, group counts, configuration and upstream revision, scaler and backbone checkpoint, adapter checkpoints, loss histories, skipped scenarios, timing, and `metrics.csv`. Checkpoints use `torch.save`. Latent representations are cached during execution; full predicted count matrices are not saved.

The aggregate report includes `REPORT.md`, `summary.csv`, `paired_comparisons.csv`, and `comparison.png`. Comparisons are paired by seed, scenario, and cell type before calculating improvements.

The primary reference is the **same observed target expression for every method**. Scores against a shared frozen-backbone denoised reference are reported separately. Metrics include MSE, Pearson correlation, R², correlation of expression changes relative to control, and test MMD. Squared error in expression changes against a common control is mathematically identical to expression MSE and is not presented as independent evidence.

The `paper_sse` column stores the sum of squared errors, which the upstream GBM script labels MSE. Use `mse` for the actual mean squared error. Corrected splitting and software-version differences mean the results need not exactly match every published number. The goal is a controlled comparison of the original and proposed shifts, not a complete reproduction of the paper. scGen and scPreGAN are not retrained.

The simulated data do not necessarily require cell-dependent responses; a lack of improvement is also informative. Matching cell-type means or distributions does not establish accurate causal responses for individual cells. Three seeds provide preliminary descriptive evidence, not extensive statistical validation.

## Sources and license

- [Original paper, Nature Communications 2024](https://doi.org/10.1038/s41467-024-45227-w)
- [Official datasets and checksums](https://zenodo.org/records/10472715)
- [Pinned upstream source, revision 9466ecf](https://github.com/ZhangLabGT/scDisInFact/tree/9466ecf257d923f879cb39712c530d24826d88bc)

`vendor/scDisInFact` contains the required upstream files and two reference experiment scripts, without changes to the model implementation. The authors' GPL-3.0 license and README are retained. This project uses the same license.

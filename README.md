# SpatioTemporalObjectTracking
Graph translation network for modeling spatio-temporal dynamics of household objects for our paper ['Proactive Robot Assistance via Spatio-Temporal Object Modeling'](https://openreview.net/pdf?id=th7GW868Pok). The model reads in an input graph representing the environment and time, and translates it to a probabilistic output graph representing the environment at the next time step. It uses the HOMER dataset, details about which can be found [here](https://github.com/GT-RAIL/rail_tasksim/tree/homer/routines). The processed version of the dataset required to run this code is present in this repository, however the complete dataset, containing human readable scripts, object arrangement trees, visuals, etc., is available for download [here](https://www.dropbox.com/s/8qs1znw3fmqho44/HOMER.zip?dl=0)

<img src="GNNarchitecture.png"
     alt="GNN Architecture"
     style="float: center;" />

### Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). There is nothing to
install by hand -- `uv run` creates and syncs `.venv` from `uv.lock` on first use:

```bash
uv run ./run.py --help          # default torch from PyPI (CUDA-enabled, ~3GB)
uv sync --extra cpu             # or: CPU-only torch (~200MB), for laptops/CI/CPU nodes
uv sync --extra cu126           # or: an explicit CUDA 12.6 build
```

`--extra cpu` and `--extra cu126` are mutually exclusive; with neither, you get the
default PyPI torch, which also runs fine on a machine without a GPU.

Python and every dependency version are pinned (`.python-version`, `uv.lock`), so the
resolution is reproducible. The original conda environment this code shipped with
(Python 3.8 / torch 1.11 / Lightning 1.5) is kept for reference in
`requirements_conda.txt` but is no longer what the code targets -- see *Modernisation
notes* below.

### Running this model
To run the model on the existing dataset, you can use the `run.py` with the path to the dataset and config file. e.g. `uv run ./run.py --path=$dataset`.

Some common things to do are:
- Running the model as is on a dataset
     `uv run ./run.py --path=$dataset --name=ours --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt`
- Running the baselines on a dataset
     `uv run ./run.py --cfg=default --path=$dataset --baselines --train_days=$train_days --logs_dir=$logs_dir/$train_days`
- Pretrain a model on a combination of datasets (note: `run_pt.py` is referenced here but is not present in this repository)
     `uv run ./run_pt.py --path=$d_target --pretrain_dirs=$d0,$d1,$d2,$d3,$d4 --name=ourspt --train_days=$logs_dir/$train_days --logs_dir=$logs_dir/$train_days --write_ckpt`
- Finetune a pretrained model
     `uv run ./run.py --path=$d_target --name=ourspt --train_days=$train_days --logs_dir=$logs_dir/$train_days --write_ckpt --finetune --ckpt_dir=$ckpt_root/0/$target/ourspt`
- Run ablations or other configurations
     `uv run ./run.py --cfg=$myNewConfig --path=$dataset --name=$myNewConfig --train_days=30 --logs_dir=$logs_dir/$train_days --write_ckpt`



A processed version of the [HOMER dataset](https://github.com/GT-RAIL/rail_tasksim/tree/homer/routines) used for the results is present in the `data/` directory of this repository, and can be used directly with the model using the above commands. In order to use a different dataset generated using HOMER, first copy the dataset into `data/` directory, and then run `helpers/reader.py` with appropriate arguments to run the necessary pre-processing. This needs to be done only once.

If you're curious about the code itself:
- The model and it's helper functions can be found in `GraphTranslatorModule.py`
- The `reader.py` file contains code to process the (HOMER) dataset
- The evaluation functions for our model are in `evaluations.py`

### Modernisation notes

This fork runs on Python 3.12 / torch 2.x / Lightning 2.x. The model, losses and data
pipeline are unchanged; only APIs that were removed upstream were rewritten:

- `pytorch_lightning.core.lightning.LightningModule` -> `pytorch_lightning.LightningModule`
  (the old module path was deleted in Lightning 2.0).
- `Trainer(gpus=N)` -> `Trainer(accelerator=..., devices=...)`. `gpus=0` meant CPU, so
  with no GPU visible the run now asks for one CPU device.
- `self.log('Train loss', <dict>)` -> `self.log_dict({'Train loss/<component>': ...})`.
  Lightning 2.x refuses dict values; each loss component is now logged under its own key.
- `matplotlib.cm.get_cmap` -> `matplotlib.colormaps[...].resampled(...)` (removed in
  matplotlib 3.9).
- The cosine-embedding target in `GraphTranslatorModule` was hardcoded to `.to('cuda')`,
  which crashed any CPU run; it now follows `self.device`.
- `sys.path.append('helpers')` and the `config/`, `data/` defaults were relative to the
  working directory, so `run.py` only worked when invoked from the repo root. They are
  resolved against the repo root now, and `uv run /path/to/run.py` works from anywhere.
- Dropped an unused `import wandb` from `helpers/evaluations.py` (nothing in the repo
  calls wandb), which keeps it out of the dependency set.

### Citation
```
@inproceedings{patel2022proactive,
  title={Proactive Robot Assistance via Spatio-Temporal Object Modeling},
  author={Patel, Maithili and Chernova, Sonia},
  booktitle={6th Annual Conference on Robot Learning},
  year={2022}
}
```

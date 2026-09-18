

## Running the code

The code can be run in three steps: setting up the environment, configuring paths and services, and launching one of the experiment flows. A flow downloads and converts the dataset, preprocesses it at the required resolutions, trains the diffusion priors, tunes and runs the reconstruction methods, and produces the tables and figures of the paper. Every stage is cached, so re-running a flow only executes what changed.

1. [Setup](#setup): Installing the Python environment, the BART toolbox, external packages and compiling the local CUDA kernels.
2. [Configuration](#config): Configuring where datasets and artifacts are stored, which task runner to use, and experiment tracking.
3. [Running a flow](#flows): Launching one of the experiments in `hydra/flow`.

The repository is structured as follows:
```
 ├───setup                      # setup scripts (step 1)
 ├───hydra/config.yaml          # global configuration (step 2)
 ├───hydra/flow                 # experiment flows: method (variational method) and resorob (robustness experiments)
 ├───hydra/task                 # per-stage configuration: download, conversion, preprocessing, training, reconstruction, plotting
 ├───src                        # python code
 ├───submodules                 # CUDA code for Gaussian interpolation (compiled in step 1)
 ├───main.py                    # entry point (step 3)
```

<a name="setup"></a>
### 1. Setup
The code has been tested using the `pytorch/pytorch` docker image, with the docker service running on `Linux: Ubuntu 20.04 LTS`. Install the system packages, then the Python environment:
```bash
bash setup/root_setup.sh # as root (compilers, etc)
bash setup/user_setup.sh # as non-root (requirements, kernels etc)
```

<a name="datasets"></a>
### 2. Datasets
In our paper, we consider the following four datasets:
 - [Stanford (3T) 3D knee volumes](http://mridata.org/)
 - [Calgary-Campinas (3T) 3D brain volumes ](https://portal.conp.ca/dataset?id=projects/calgary-campinas)
 - [AHEAD ultra-high field (7T) 3D brain volumes](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/IHZGQM)
 - [Lüsebrink ultra-high field (7T) T1-weighted brain volumes](https://openneuro.org/datasets/ds003563)

Datasets are downloaded by the first stage of a flow into the configured cache directory (Calgary-Campinas and Lüsebrink via datalad/git-annex, Stanford and AHEAD from the URL lists in `hydra/task/download_dataset/urls`). Conversion to a common format and resolution changes are handled by the subsequent stages.

<a name="config"></a>
### 3. Configuration

Most aspects of the software can be configured using [Hydra](https://github.com/facebookresearch/hydra). Before the first run, set in `hydra/config.yaml` (or on the command line):
- **Paths**: `cluster_name` names the machine, `local_cache_path.<cluster_name>` is the directory where datasets, preprocessed data and models are cached, and `bart_path.<cluster_name>` points at the BART installation.
- **Task runner**: `task_runner: ray_local` (default) submits tasks to a running Ray cluster (`RAY_ADDRESS`, default `ray://localhost:10001`); `local` runs them in-process.
- **Wandb logging** (Optional): set the `entity` and `project` keys under `wandb` and set `log: True`. Wandb will ask for an API key on first run.

Flows are orchestrated with Prefect. A Prefect server (`PREFECT_API_URL`, default `http://127.0.0.1:4200/api`) and an S3-compatible object store used for caching (`RESROB_MINIO_ENDPOINT_URL`, default `http://127.0.0.1:9000`, with the buckets `resrob-caching`, `resrob-models` and `resrob-recon`) must be reachable.

<a name="flows"></a>
### 4. Running a flow

Each file in `hydra/flow` defines one experiment of the paper end to end. Launch it with
```bash
python main.py +flow=method/main_method_comparison_on_cc359_flow cluster_name=<name>
```
`hydra/flow/method` contains the method comparisons, ablations and sensitivity studies; `hydra/flow/resorob` contains the resolution-robustness experiments (training-resolution diversity, out-of-resolution generalisation, hyper-parameter transfer). The task configurations composed by a flow live in `hydra/task`, e.g. `train_recon/exps` for the trained priors and `train_recon/rec_method` for the reconstruction methods.


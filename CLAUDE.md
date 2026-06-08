# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Given an ensemble of inertial particle trajectories from a GHOST turbulence simulation, classify which drag model (and Stokes number) generated them. The intended use case is inferring particle physics from experimental or observational data where the underlying model is unknown.

The GHOST simulator lives at `/home/clark/repos/GHOST`. Reference `.lag` files for local testing are in this directory (`MR/St2/`, `MR/St8-89/`, and the root-level `*.lag` files).

**The simulation data is not available on this machine.** `/share/scratch*` paths do not and will not exist locally. Scripts are developed and tested here; data lives on the cluster.

## Running the pipeline

```bash
# Step 1: build and save the feature dataset (MPI-parallel, one rank per sim)
mpirun -n N python build_dataset.py --yaml simuls.yaml --output dataset.pkl

# Key options for dataset generation
mpirun -n N python build_dataset.py --batch-size 5000 --n-batches 100 --n-lags 15 \
    --max-steps 400 --output dataset.pkl

# Step 2: train a classifier on the saved dataset
python train_classifier.py --dataset dataset.pkl --output model.pkl

# Key options for training
python train_classifier.py --dataset dataset.pkl --n-trees 300 --n-folds 5 --output model.pkl

# Predict the model class for a new simulation directory
python predict.py --model model.pkl --path /path/to/sim/dir

# Inspect ensemble statistics vs physical lag time (MPI-parallel)
mpirun -n N python inspect_signals.py --yaml simuls.yaml --output inspect_signals.pdf

# Exploratory data analysis on a pre-built dataset.pkl
python eda.py --dataset dataset.pkl --output eda.pdf

# Inspect a single particle's trajectory
python test/plot_particle_timeseries.py --path /path/to/sim --quantity xlg --particle 42
python test/plot_particle_timeseries.py --path /path/to/sim --quantity vip --particle 0 --component mag

# Blind A/B visual comparison between two simulations
python test/blindtest_timeseries.py --path1 /share/scratch8/bespanol/MR/St2 --path2 /share/scratch8/bespanol/LAG/St0

# Inspect .lag files directly
python test/read_lag.py                  # reads all *.lag in current dir
python test/read_lag.py vip.00000148.lag
```

## .lag binary format

GHOST writes Fortran stream I/O binary files. Layout:

```
[float32: N_particles] [float32: time] [float32: x0 x1 ... xN-1 y0...yN-1 z0...zN-1]
```

Always read with `dtype=np.float32`. The data block reshapes to `(3, N)` (rows = x, y, z). Available quantities per timestep: `xlg` (position), `vlg` (fluid velocity at tracer), `vip` (inertial particle velocity), `wip` (vorticity at particle).

## Architecture

The pipeline has four layers:

**1. I/O** — `test/read_lag.py` (standalone) and the `_read_lag`/`_load_quantity`/`load_simulation` functions inside `dataset.py`. Both implement the same format; `dataset.py`'s version is used for bulk loading with particle index selection.

**2. Feature extraction** — `features.py`. A single feature vector summarises an *ensemble* (batch) of particles, not individual trajectories. Features: normalised VACF, MSD, 2nd- and 4th-order velocity structure functions (all at log-spaced lag indices), plus scalar velocity variance/kurtosis and acceleration variance/flatness. Feature length = `4 * n_lags + 4`.

**3. Dataset construction** — `dataset.py` + `build_dataset.py`. `build_dataset.py` is the MPI-parallel CLI: it distributes simulations round-robin across ranks, calls `make_feature_matrix()` (from `dataset.py`) per simulation, then rank 0 gathers, imputes NaNs, and saves a `.pkl` containing `X`, `y`, `feature_names`, `label_names`, `n_lags`, and `batch_size`. The serial `build_dataset()` function in `dataset.py` still exists as a programmatic entry point. Each training *sample* is one non-overlapping batch of `batch_size` particles. Labels are `"<MODEL>_St<st>"` strings mapped to integers.

**4. Model** — `train_classifier.py` / `predict.py`. Loads a pre-built dataset `.pkl` and trains a `sklearn` Pipeline of `StandardScaler → RandomForestClassifier`. Cross-validation is stratified k-fold. The saved model `.pkl` bundles the pipeline, label names, feature names, `n_lags`, and `batch_size` so `predict.py` can reconstruct compatible lag arrays at inference time.

## Simulation registry

`simuls.yaml` lists all simulations. Each entry has `parts.model` (LAG, MR, NLD, ONLD, BB, FAX) and `parts.st` (Stokes number). Data lives on remote scratch (`/share/scratch{8,12}/bespanol/`). Both `build_dataset.py` and `inspect_signals.py` silently skip directories that don't exist, so scripts run locally against whatever subset is mounted.

Models in the dataset:
- **LAG** — fluid tracers (St=0)
- **MR** — Maxey-Riley: Stokes drag + added-mass/pressure term
- **NLD** — nonlinear drag + MR acceleration term
- **ONLD** — nonlinear drag only (no acceleration term)
- **BB** — MR + Basset–Boussinesq history integral
- **FAX** — MR + Faxén correction

## Key design decisions

- Features are ensemble statistics, not per-particle time series, so the classifier is invariant to particle labelling and can handle variable particle counts.
- Lag indices are log-spaced (via `make_lags`) and span from 1 to `n_steps // 2`. The selection is evenly spread across the full unique integer range — not just the smallest values — to capture both short and long timescale behaviour.
- `--max-steps` in `build_dataset.py` lets you truncate trajectories to a known convergence point (use `inspect_signals.py` to find it). If omitted, the shortest trajectory across accessible sims is used.
- Positions need periodic-boundary unwrapping (`unwrap_positions`) before computing MSD; velocities do not.
- `batch_size` must match between training and inference; it is stored in the `.pkl` for this reason.
- MPI parallelism in `build_dataset.py` and `inspect_signals.py` assigns simulations round-robin across ranks so any number of ranks works. `label_map` is built deterministically from YAML order on every rank to ensure consistent integer label assignments.

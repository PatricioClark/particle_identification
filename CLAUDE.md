# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Given an ensemble of inertial particle trajectories from a GHOST turbulence simulation, classify which drag model (and Stokes number) generated them. The intended use case is inferring particle physics from experimental or observational data where the underlying model is unknown.

The GHOST simulator lives at `/home/clark/repos/GHOST`. Reference `.lag` files for local testing are in this directory (`MR/St2/`, `MR/St8-89/`, and the root-level `*.lag` files).

## Running the pipeline

```bash
# Train a classifier on all simulations in the YAML
python train_classifier.py --yaml simuls_bernardo.yaml

# Key options
python train_classifier.py --batch-size 500 --n-batches 10 --n-lags 15 --n-load 5000 --n-trees 300 --output model.pkl

# Predict the model class for a new simulation directory
python predict.py --model model.pkl --path /path/to/sim/dir

# Inspect a single particle's trajectory
python plot_particle_timeseries.py --path /path/to/sim --quantity xlg --particle 42
python plot_particle_timeseries.py --path /path/to/sim --quantity vip --particle 0 --component mag

# Blind A/B visual comparison between two simulations
python blindtest_timeseries.py --path1 /share/scratch8/bespanol/MR/St2 --path2 /share/scratch8/bespanol/LAG/St0

# Inspect .lag files directly
python read_lag.py                  # reads all *.lag in current dir
python read_lag.py vip.00000148.lag
```

## .lag binary format

GHOST writes Fortran stream I/O binary files. Layout:

```
[float32: N_particles] [float32: time] [float32: x0 x1 ... xN-1 y0...yN-1 z0...zN-1]
```

Always read with `dtype=np.float32`. The data block reshapes to `(3, N)` (rows = x, y, z). Available quantities per timestep: `xlg` (position), `vlg` (fluid velocity at tracer), `vip` (inertial particle velocity), `wip` (vorticity at particle).

## Architecture

The pipeline has four layers:

**1. I/O** — `read_lag.py` (standalone) and the `_read_lag`/`_load_quantity`/`load_simulation` functions inside `dataset.py`. Both implement the same format; `dataset.py`'s version is used for bulk loading with particle index selection.

**2. Feature extraction** — `features.py`. A single feature vector summarises an *ensemble* (batch) of particles, not individual trajectories. Features: normalised VACF, MSD, 2nd- and 4th-order velocity structure functions (all at log-spaced lag indices), plus scalar velocity variance/kurtosis and acceleration variance/flatness. Feature length = `4 * n_lags + 4`.

**3. Dataset construction** — `dataset.py`. `build_dataset()` is the top-level entry point. It probes all simulation directories, aligns them to the shortest common trajectory length, then calls `make_feature_matrix()` per simulation. Each training *sample* is one non-overlapping batch of `batch_size` particles — this mimics what you'd observe in an experiment. Labels are `"<MODEL>_St<st>"` strings mapped to integers.

**4. Model** — `train_classifier.py` / `predict.py`. A `sklearn` Pipeline of `StandardScaler → RandomForestClassifier`. Cross-validation is stratified k-fold. The saved `.pkl` bundles the pipeline, label names, feature names, `n_lags`, and `batch_size` so `predict.py` can reconstruct compatible lag arrays at inference time.

## Simulation registry

`simuls_bernardo.yaml` lists all simulations. Each entry has `parts.model` (LAG, MR, NLD, ONLD, BB, FAX) and `parts.st` (Stokes number). Data lives on remote scratch (`/share/scratch{8,12}/bespanol/`). `build_dataset()` silently skips directories that don't exist, so the script runs locally against whatever subset is mounted.

Models in the dataset:
- **LAG** — fluid tracers (St=0)
- **MR** — Maxey-Riley: Stokes drag + added-mass/pressure term
- **NLD** — nonlinear drag + MR acceleration term
- **ONLD** — nonlinear drag only (no acceleration term)
- **BB** — MR + Basset–Boussinesq history integral
- **FAX** — MR + Faxén correction

## Key design decisions

- Features are ensemble statistics, not per-particle time series, so the classifier is invariant to particle labelling and can handle variable particle counts.
- Lag indices are log-spaced (via `make_lags`) and capped at `n_steps // 2` to avoid biased estimators at long lags. All simulations are truncated to the shortest available trajectory length before lags are derived.
- Positions need periodic-boundary unwrapping (`unwrap_positions`) before computing MSD; velocities do not.
- `batch_size` must match between training and inference; it is stored in the `.pkl` for this reason.

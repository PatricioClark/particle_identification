# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Given an ensemble of inertial particle trajectories from a GHOST turbulence simulation, classify which drag model (and Stokes number) generated them. The intended use case is inferring particle physics from experimental or observational data where the underlying model is unknown.

The GHOST simulator lives at `/home/clark/repos/GHOST`. Reference `.lag` files for local testing are in this directory (`MR/St2/`, `MR/St8-89/`, and the root-level `*.lag` files).

**The simulation data is not available on this machine.** `/share/scratch*` paths do not and will not exist locally. Scripts are developed and tested here; data lives on the cluster.

## Running the pipeline

```bash
# Step 1: build and save the feature dataset (MPI-parallel, sharded + checkpointed)
mpirun -n N python build_dataset.py --yaml simuls.yaml \
    --checkpoint-dir ckpt --output dataset.pkl

# Key options for dataset generation. Total samples/sim = n-shards-per-sim * batches-per-shard.
mpirun -n N python build_dataset.py --batch-size 5000 --batches-per-shard 100 \
    --n-shards-per-sim 1 --n-load 50000 --n-lags 15 --max-steps 400 \
    --checkpoint-dir ckpt --output dataset.pkl

# Resume after a wall-time kill: re-run the SAME command — finished shards are skipped.
# Add more samples: re-run with a larger --n-shards-per-sim into the same --checkpoint-dir.
# Stitch existing checkpoints without recomputing:
python build_dataset.py --checkpoint-dir ckpt --output dataset.pkl --assemble-only

# Merge independently-built datasets into one (reconciles per-dataset label encodings)
python merge_datasets.py a.pkl b.pkl [c.pkl ...] --output merged.pkl

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

**3. Dataset construction** — `dataset.py` + `build_dataset.py`. The unit of work is a **shard**: one `(simulation, shard-index)` pair that calls `make_feature_matrix()` to load a particle pool *once* and emit `--batches-per-shard` feature rows. `build_dataset.py` flattens all shards across all sims, distributes them round-robin across MPI ranks (so ranks stay busy even when sims < ranks), and each shard **writes its own checkpoint** (`{label}__sim{i}__shard{j}.pkl`) atomically (temp + `os.replace`) as soon as it finishes. A shard whose checkpoint already exists is **skipped**, which is how resume-after-wall-time-kill works. When the shards are done, rank 0 `assemble()`s every checkpoint in `--checkpoint-dir` into the final `.pkl` (`X`, `y`, `feature_names`, `label_names`, `n_lags`, `batch_size`, `window_size`), imputing NaNs at that step; `--assemble-only` runs just the stitch. The serial `build_dataset()` function in `dataset.py` still exists as a programmatic entry point. Each training *sample* is one batch of `batch_size` particles drawn at random from the loaded pool — batches may overlap, which is what lets one pool load yield arbitrarily many samples. Labels are `"<MODEL>_St<st>_<forcing><resolution>"` strings (e.g. `"LAG_St0.0_RND512"`) mapped to integers — forcing/resolution are part of the label so sims that share `(model, st)` but come from a different flow realization (e.g. the TG768 and RND512 data sets) become distinct classes instead of silently merging.

`merge_datasets.py` stitches together datasets built in separate runs. Because each `.pkl`'s `y` is an integer index into *its own* `label_names`, it cannot raw-concatenate: it checks feature-space compatibility (`feature_names`, `n_lags`, `batch_size`, `window_size`), builds a union label list, and remaps every `y` into that union space before stacking. This is the robust way to add data — it operates on finished products and sidesteps the checkpoint/YAML-index coupling below.

**4. Model** — `train_classifier.py` / `predict.py`. Loads a pre-built dataset `.pkl` and trains a `sklearn` Pipeline of `StandardScaler → RandomForestClassifier`. Cross-validation is stratified k-fold. The saved model `.pkl` bundles the pipeline, label names, feature names, `n_lags`, and `batch_size` so `predict.py` can reconstruct compatible lag arrays at inference time.

## Simulation registry

`simuls.yaml` lists all simulations. Each entry has `parts.model` (LAG, MR, NLD, ONLD, HPP, BB, FAX) and `parts.st` (Stokes number). Data lives on remote scratch (`/share/scratch{8,12}/bespanol/`) and, for the second (HIT512, RND-forced) data set, under `/share/data4/bespanol/HIT512_multipart/bin/`. Both `build_dataset.py` and `inspect_signals.py` silently skip directories that don't exist, so scripts run locally against whatever subset is mounted. See `docs/resumen.tex` for the physics/equations behind each model label.

Models in the dataset:
- **LAG** — fluid tracers (St=0)
- **HPP** — naive heavy particle: linear Stokes drag only (no added-mass/pressure/buoyancy/acceleration term)
- **MR** — Maxey-Riley: Stokes drag + added-mass/pressure term
- **NLD** — nonlinear drag + MR acceleration term
- **ONLD** — nonlinear drag only (no acceleration term)
- **BB** — MR + Basset–Boussinesq history integral
- **FAX** — MR + Faxén correction

**Second data set (HIT512, RND forcing, multi-particle).** Simulations under `HIT512_multipart/bin/` run several drag models off one flow realization; each `outs_{MODEL}_{St}` directory holds `.lag` files directly (no nested `outs` subdir — `_glob_sorted` in `dataset.py` already falls back to the sim path itself when no `outs` subdir exists). On disk these dirs are named `outs_NLD_*`, but in this data set NLD and ONLD coincide (no acceleration term), so `simuls.yaml` labels them `ONLD` for consistency with the first data set — don't be misled by the directory name when adding more of these.

## Key design decisions

- Features are ensemble statistics, not per-particle time series, so the classifier is invariant to particle labelling and can handle variable particle counts.
- Lag indices are log-spaced (via `make_lags`) and span from 1 to `n_steps // 2`. The selection is evenly spread across the full unique integer range — not just the smallest values — to capture both short and long timescale behaviour.
- `--max-steps` in `build_dataset.py` lets you truncate trajectories to a known convergence point (use `inspect_signals.py` to find it). If omitted, the shortest trajectory across accessible sims is used.
- Positions need periodic-boundary unwrapping (`unwrap_positions`) before computing MSD; velocities do not.
- `batch_size` must match between training and inference; it is stored in the `.pkl` for this reason.
- MPI parallelism in `build_dataset.py` and `inspect_signals.py` assigns work round-robin across ranks so any number of ranks works. `label_map` is built deterministically from YAML order on every rank to ensure consistent integer label assignments.
- **Sharding/checkpointing exists to survive cluster wall-time limits.** The checkpoint is at the *shard* level (one pool load), not per batch, because the expensive, reusable unit of work is the trajectory read — resuming a half-done shard would have to reload the trajectory anyway, and per-batch files would flood the cluster FS with tiny inodes. `--batches-per-shard` is the granularity dial: smaller = finer checkpoints but more redundant reads; larger = fewer reads amortised over more rows. Per-shard seeds are `SeedSequence([seed, yaml_i, shard_j])`, reproducible and independent of rank count, so a new shard index always yields fresh (non-duplicate) samples.
- `--n-shards-per-sim > 1` splits one sim across ranks. Because each `.lag` file holds *all* particles and is read whole regardless of how many are selected, this **re-reads the trajectory per shard** — only worth it when feature compute, not I/O, is the wall-time bottleneck.
- **Checkpoint identity is `(label, yaml_i)` and `assemble()` globs the whole `--checkpoint-dir`.** It does not encode the sim's path or validate param consistency across shards. So reusing a checkpoint dir is only safe if you append sims at the *end* of the YAML (never reorder/insert/remove — that shifts `yaml_i` and leaves stale files that get double-counted) and pin `--max-steps` (so `min_steps`→lags/`window_size` don't drift between runs). When in doubt, use a fresh checkpoint dir, or merge finished `.pkl`s with `merge_datasets.py` instead.
- The label string is also embedded in each checkpoint filename and in the checkpoint's saved `label` field. **Changing the label format (e.g. what fields it's built from) invalidates checkpoints computed under the old format**: they won't be recognized as "already done" (different filename) and, worse, `assemble()` will `KeyError` on their old-format `label` when looking it up in the freshly-built `label_map`. Wipe or migrate old checkpoints before reusing a `--checkpoint-dir` across a label-format change.

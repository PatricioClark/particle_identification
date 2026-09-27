"""Build and save a labelled feature dataset from GHOST simulation directories.

Usage
-----
mpirun -n N python build_dataset.py [options]

Work is split into *shards*. A shard is one (simulation, shard-index) unit that
loads a particle pool once and emits ``--batches-per-shard`` feature rows. Each
shard writes its own checkpoint file as soon as it finishes, so a run that hits
the cluster wall-time can be resubmitted and will skip every shard already on
disk. When all shards exist, rank 0 assembles them into the final dataset .pkl.

    # fresh run (or resume — finished shards are skipped automatically)
    mpirun -n N python build_dataset.py --checkpoint-dir ckpt --output dataset.pkl

    # only stitch existing checkpoints into the dataset, no recomputation
    python build_dataset.py --checkpoint-dir ckpt --output dataset.pkl --assemble-only

Total samples per simulation = --n-shards-per-sim × --batches-per-shard.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import yaml
from mpi4py import MPI
from pathlib import Path

from dataset import probe_sim, make_feature_matrix
from features import make_lags

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def shard_label(sim):
    return f"{sim['parts']['model']}_St{sim['parts']['st']}"


def checkpoint_path(ckpt_dir, sim, yaml_i, shard_j):
    """Unique, deterministic checkpoint name. yaml_i disambiguates distinct
    sims that happen to share a (model, St) label."""
    return ckpt_dir / f"{shard_label(sim)}__sim{yaml_i:03d}__shard{shard_j:03d}.pkl"


def assemble(ckpt_dir, label_map, output):
    """Stitch every checkpoint in ckpt_dir into the final dataset .pkl."""
    ckpts = sorted(ckpt_dir.glob("*.pkl"))
    if not ckpts:
        print("No checkpoints to assemble.", file=sys.stderr)
        sys.exit(1)

    X_blocks, y_blocks = [], []
    feature_names = None
    n_lags = batch_size = window_size = None
    for c in ckpts:
        d = joblib.load(c)
        X_blocks.append(d["X"])
        y_blocks.append(np.full(len(d["X"]), label_map[d["label"]], dtype=int))
        feature_names = feature_names or d["feature_names"]
        n_lags      = d["n_lags"]
        batch_size  = d["batch_size"]
        window_size = d["window_size"]

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(label_names)} classes (from {len(ckpts)} shards)")
    for i, name in enumerate(label_names):
        print(f"  {i:2d}  {name}  ({(y == i).sum()} samples)")

    # Impute NaNs with column medians
    col_medians = np.nanmedian(X, axis=0)
    nan_locs    = np.isnan(X)
    X[nan_locs] = col_medians[np.where(nan_locs)[1]]

    joblib.dump(
        {
            "X":             X,
            "y":             y,
            "feature_names": feature_names,
            "label_names":   label_names,
            "n_lags":        n_lags,
            "batch_size":    batch_size,
            "window_size":   window_size,
        },
        output,
    )
    print(f"Dataset saved → {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml",             default="simuls.yaml")
    p.add_argument("--batch-size",       type=int, default=5000)
    p.add_argument("--batches-per-shard", type=int, default=100,
                   help="Feature rows emitted per shard (one pool load).")
    p.add_argument("--n-shards-per-sim", type=int, default=1,
                   help="Shards per simulation. >1 splits a sim across ranks "
                        "(extra trajectory reads — only worth it when feature "
                        "compute, not I/O, is the wall-time bottleneck).")
    p.add_argument("--n-load",           type=int, default=50000,
                   help="Particle pool size loaded once per shard. Keep it "
                        "much larger than --batch-size so batch overlap stays small.")
    p.add_argument("--n-lags",           type=int, default=15)
    p.add_argument("--max-steps",        type=int, default=None,
                   help="Truncate trajectories to this many steps. "
                        "Defaults to the shortest trajectory across all sims.")
    p.add_argument("--no-time-augment",  dest="time_augment",
                   action="store_false", default=True,
                   help="Disable per-particle random time offsets (on by default).")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--checkpoint-dir",   default="checkpoints",
                   help="Directory of per-shard checkpoint files (enables resume).")
    p.add_argument("--assemble-only",    action="store_true",
                   help="Skip computation; stitch existing checkpoints into --output.")
    p.add_argument("--output",           default="dataset.pkl")
    args = p.parse_args()

    with open(args.yaml) as fh:
        sims = yaml.safe_load(fh)

    # All ranks: filter to accessible sims and compute shared parameters
    accessible = [
        (i, s) for i, s in enumerate(sims)
        if Path(s["path"]).is_dir() and probe_sim(s["path"])[1] > 0
    ]

    if not accessible:
        if rank == 0:
            print("No accessible simulation directories found.", file=sys.stderr)
        sys.exit(1)

    # min_steps and lags — deterministic, same on all ranks
    min_steps = args.max_steps or min(probe_sim(s["path"])[1] for _, s in accessible)
    effective_steps = min_steps - 1
    lags = make_lags(effective_steps, args.n_lags)

    # label_map — built deterministically from accessible sims in YAML order
    label_map: dict[str, int] = {}
    for _, sim in accessible:
        label = shard_label(sim)
        if label not in label_map:
            label_map[label] = len(label_map)

    ckpt_dir = Path(args.checkpoint_dir)

    # ── Assemble-only: rank 0 stitches and exits ──────────────────────────────
    if args.assemble_only:
        if rank == 0:
            assemble(ckpt_dir, label_map, args.output)
        return

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Accessible simulations: {len(accessible)}, MPI ranks: {size}")
        print(f"window_size={min_steps}, effective_steps={effective_steps}")
        print(f"Time augmentation: {'on' if args.time_augment else 'off'}")
        print(f"Shards/sim: {args.n_shards_per_sim}, batches/shard: "
              f"{args.batches_per_shard}, pool: {args.n_load}")
        print(f"Checkpoints → {ckpt_dir}/")
        print(f"Lag indices ({args.n_lags}): {lags}\n")
    comm.Barrier()  # ensure ckpt_dir exists before any rank writes

    # Flat shard list, identical on every rank, distributed round-robin.
    shards = [
        (yaml_i, sim, j)
        for yaml_i, sim in accessible
        for j in range(args.n_shards_per_sim)
    ]
    my_shards = shards[rank::size]

    # ── Each rank computes its shards, checkpointing as it goes ───────────────
    for yaml_i, sim, j in my_shards:
        ckpt = checkpoint_path(ckpt_dir, sim, yaml_i, j)
        if ckpt.exists():
            print(f"  [rank {rank}] skip (done): {ckpt.name}")
            continue

        label = shard_label(sim)
        print(f"  [rank {rank}] {ckpt.name}  ({sim['path']})")

        # Unique, reproducible seed per shard — independent of rank count.
        seed = int(np.random.SeedSequence([args.seed, yaml_i, j]).generate_state(1)[0])

        X_block, feature_names = make_feature_matrix(
            sim["path"], lags,
            batch_size=args.batch_size,
            n_batches=args.batches_per_shard,
            n_load=args.n_load,
            max_steps=None if args.time_augment else min_steps,
            window_size=min_steps if args.time_augment else None,
            seed=seed,
        )
        if X_block.shape[0] == 0:
            continue

        # Atomic write: a wall-time kill mid-dump never leaves a partial .pkl.
        tmp = ckpt.with_name(ckpt.name + f".tmp.{rank}")
        joblib.dump(
            {
                "X":             X_block,
                "label":         label,
                "feature_names": feature_names,
                "n_lags":        args.n_lags,
                "batch_size":    args.batch_size,
                "window_size":   min_steps,
                "yaml_i":        yaml_i,
                "shard":         j,
                "seed":          seed,
            },
            tmp,
        )
        os.replace(tmp, ckpt)

    comm.Barrier()

    # ── Rank 0 assembles whatever checkpoints now exist ───────────────────────
    if rank == 0:
        assemble(ckpt_dir, label_map, args.output)


if __name__ == "__main__":
    main()

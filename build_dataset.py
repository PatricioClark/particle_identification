"""Build and save a labelled feature dataset from GHOST simulation directories.

Usage
-----
mpirun -n N python build_dataset.py [options]

Reads all simulations listed in the YAML, distributes them round-robin across
MPI ranks, builds feature matrices in parallel, and rank 0 assembles, imputes
NaNs, and saves the result to a .pkl file for use by train_classifier.py.
"""

import argparse
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml",       default="simuls.yaml")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--n-batches",  type=int, default=10)
    p.add_argument("--n-lags",     type=int, default=15)
    p.add_argument("--n-load",     type=int, default=5000)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     default="dataset.pkl")
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
    min_steps = min(probe_sim(s["path"])[1] for _, s in accessible)
    effective_steps = min_steps - 1
    lags = make_lags(effective_steps, args.n_lags)

    if rank == 0:
        print(f"Accessible simulations: {len(accessible)}, MPI ranks: {size}")
        print(f"min_steps={min_steps}, effective_steps={effective_steps}")
        print(f"Lag indices ({args.n_lags}): {lags}\n")

    # label_map — built deterministically from accessible sims in YAML order
    label_map: dict[str, int] = {}
    for _, sim in accessible:
        label = f"{sim['parts']['model']}_St{sim['parts']['st']}"
        if label not in label_map:
            label_map[label] = len(label_map)

    # Round-robin assignment
    my_sims = accessible[rank::size]

    # ── Each rank builds feature matrices for its simulations ─────────────────
    local_blocks = []  # list of (X_block, y_block, feature_names)
    for i, sim in my_sims:
        label  = f"{sim['parts']['model']}_St{sim['parts']['st']}"
        y_int  = label_map[label]
        print(f"  [rank {rank}] {label}  ({sim['path']})")

        X_block, feature_names = make_feature_matrix(
            sim["path"], lags,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            n_load=args.n_load,
            max_steps=min_steps,
            seed=args.seed + i,
        )
        if X_block.shape[0] == 0:
            continue
        y_block = np.full(len(X_block), y_int, dtype=int)
        local_blocks.append((X_block, y_block, feature_names))

    # ── Gather to rank 0 ──────────────────────────────────────────────────────
    all_blocks = comm.gather(local_blocks, root=0)

    if rank != 0:
        return

    blocks = [b for bucket in all_blocks for b in bucket]
    if not blocks:
        print("No feature blocks produced.", file=sys.stderr)
        sys.exit(1)

    X = np.vstack([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    feature_names = blocks[0][2]
    label_names   = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(label_names)} classes")
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
            "n_lags":        args.n_lags,
            "batch_size":    args.batch_size,
        },
        args.output,
    )
    print(f"Dataset saved → {args.output}")


if __name__ == "__main__":
    main()

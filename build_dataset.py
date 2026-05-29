"""Build and save a labelled feature dataset from GHOST simulation directories.

Usage
-----
python build_dataset.py [options]

Reads all simulations listed in the YAML, builds a labelled feature matrix,
imputes NaNs, and saves the result to a .pkl file for use by train_classifier.py.
"""

import argparse
import joblib
import numpy as np

from dataset import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml",       default="simuls.yaml")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Particles per training sample (default: 500)")
    p.add_argument("--n-batches",  type=int, default=10,
                   help="Training samples per simulation (default: 10)")
    p.add_argument("--n-lags",     type=int, default=15,
                   help="Lag points in feature vector (default: 15)")
    p.add_argument("--n-load",     type=int, default=5000,
                   help="Max particles loaded per simulation (default: 5000)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     default="dataset.pkl",
                   help="Path for saved dataset (default: dataset.pkl)")
    args = p.parse_args()

    print("Building dataset …")
    X, y, feat_names, label_names = build_dataset(
        args.yaml,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        n_lags=args.n_lags,
        n_load=args.n_load,
        seed=args.seed,
    )

    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(label_names)} classes")
    for i, name in enumerate(label_names):
        print(f"  {i:2d}  {name}  ({(y == i).sum()} samples)")

    col_medians = np.nanmedian(X, axis=0)
    nan_locs    = np.isnan(X)
    X[nan_locs] = col_medians[np.where(nan_locs)[1]]

    saved = {
        "X":             X,
        "y":             y,
        "feature_names": feat_names,
        "label_names":   label_names,
        "n_lags":        args.n_lags,
        "batch_size":    args.batch_size,
    }
    joblib.dump(saved, args.output)
    print(f"Dataset saved → {args.output}")


if __name__ == "__main__":
    main()

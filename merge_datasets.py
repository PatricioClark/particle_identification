"""Merge several feature datasets (built by build_dataset.py) into one .pkl.

Usage
-----
python merge_datasets.py a.pkl b.pkl [c.pkl ...] --output merged.pkl

Each input .pkl carries its own per-dataset label encoding: ``y`` is an integer
index into that dataset's ``label_names``. Two datasets can therefore use the
same integer for different classes, so the labels must be reconciled — not the
raw ``y`` arrays concatenated. This script:

  1. checks the inputs share a feature space (feature_names, n_lags,
     batch_size, window_size),
  2. builds a union label list (first dataset's order, new labels appended),
  3. remaps every dataset's ``y`` into that union space,
  4. stacks the feature matrices and saves the merged dataset.

NaN imputation already happened at build time, so it is not repeated here.
"""

import argparse
import sys

import joblib
import numpy as np


def _check_compatible(ref, ref_name, d, name):
    """Error out unless dataset d shares ref's feature space."""
    # feature_names is a list; the rest are scalars — compare both as lists so
    # numpy/list storage differences don't cause false mismatches.
    for key in ("feature_names", "n_lags", "batch_size", "window_size"):
        if list(np.atleast_1d(d[key])) != list(np.atleast_1d(ref[key])):
            print(
                f"Incompatible datasets: {key} differs between "
                f"'{ref_name}' and '{name}'.\n"
                f"  {ref_name}: {ref[key]}\n"
                f"  {name}: {d[key]}",
                file=sys.stderr,
            )
            sys.exit(1)


def merge(paths, output):
    datasets = [(p, joblib.load(p)) for p in paths]
    ref_name, ref = datasets[0]

    # ── Feature-space compatibility ───────────────────────────────────────────
    for name, d in datasets[1:]:
        _check_compatible(ref, ref_name, d, name)

    # ── Union label space (first dataset's order, new labels appended) ────────
    union: list[str] = []
    union_index: dict[str, int] = {}
    for _, d in datasets:
        for label in d["label_names"]:
            if label not in union_index:
                union_index[label] = len(union)
                union.append(label)

    # ── Remap each y into the union space, then stack ─────────────────────────
    X_parts, y_parts = [], []
    for name, d in datasets:
        remap = np.array([union_index[lbl] for lbl in d["label_names"]], dtype=int)
        X_parts.append(d["X"])
        y_parts.append(remap[d["y"]])
        print(f"  {name}: {d['X'].shape[0]} samples, "
              f"{len(d['label_names'])} classes")

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    print(f"\nMerged: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(union)} classes")
    for i, name in enumerate(union):
        print(f"  {i:2d}  {name}  ({(y == i).sum()} samples)")

    joblib.dump(
        {
            "X":             X,
            "y":             y,
            "feature_names": ref["feature_names"],
            "label_names":   union,
            "n_lags":        ref["n_lags"],
            "batch_size":    ref["batch_size"],
            "window_size":   ref["window_size"],
        },
        output,
    )
    print(f"\nMerged dataset saved → {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="Two or more dataset .pkl files.")
    p.add_argument("--output", default="merged.pkl")
    args = p.parse_args()

    if len(args.inputs) < 2:
        p.error("need at least two input datasets to merge")

    merge(args.inputs, args.output)


if __name__ == "__main__":
    main()

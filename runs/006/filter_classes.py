"""Remove classes from a dataset .pkl (no recomputation), relabelling y to 0..n-1.

Usage
-----
python filter_classes.py dataset.pkl dataset_sinBB.pkl --drop BB_
  (drops every class whose name starts with any of the --drop prefixes)
"""

import argparse

import joblib
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--drop", nargs="+", required=True, help="Class-name prefixes to remove")
    args = p.parse_args()

    d = joblib.load(args.dataset)
    names = d["label_names"]
    keep = [i for i, n in enumerate(names) if not n.startswith(tuple(args.drop))]
    dropped = [names[i] for i in range(len(names)) if i not in keep]
    remap = {old: new for new, old in enumerate(keep)}
    mask = np.isin(d["y"], keep)

    out = dict(d)
    out["X"] = d["X"][mask]
    out["y"] = np.array([remap[v] for v in d["y"][mask]], dtype=int)
    out["label_names"] = [names[i] for i in keep]
    joblib.dump(out, args.output)
    print(f"Removed {len(dropped)} classes: {dropped}")
    print(f"{args.output}: {out['X'].shape[0]} samples, {len(keep)} classes")


if __name__ == "__main__":
    main()

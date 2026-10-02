"""Compare two dataset.pkl files (and optionally their checkpoint dirs).

Usage
-----
python compare.py dataset.pkl ../002/dataset.pkl [--ckpt ckpt --ref-ckpt ../002/ckpt]
                  [--strip-label-suffix _TG768]

--strip-label-suffix removes that suffix from the first dataset's label names and
checkpoint file names before comparing (the current code appends the flow to the
labels, e.g. MR_St3.2_TG768, while run 002 used MR_St3.2). Keys present only in
the first dataset (e.g. flow_params) are reported but not counted as differences.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

# Tolerancia relativa para considerar dos valores iguales salvo redondeo
# (distinto CPU / versión de numpy cambia los últimos dígitos).
RTOL = 1e-12


def compare_arrays(name, a, b):
    """Return 'identical', 'rounding' or 'different'."""
    if a.shape != b.shape:
        print(f"  {name}: shape distinta {a.shape} vs {b.shape}")
        return "different"
    if np.array_equal(a, b):
        print(f"  {name}: idéntico")
        return "identical"
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a64 - b64)
    rel = (diff / np.maximum(np.abs(b64), np.finfo(float).tiny)).max()
    status = "rounding" if np.allclose(a64, b64, rtol=RTOL, atol=0) else "different"
    label = "idéntico salvo redondeo" if status == "rounding" else "DISTINTO"
    print(f"  {name}: {label}  max|diff|={diff.max():.3e}  max diff relativa={rel:.3e}  "
          f"elementos distintos={np.count_nonzero(diff)}/{diff.size}")
    return status


def compare_datasets(path, ref_path, suffix=""):
    d, r = joblib.load(path), joblib.load(ref_path)
    print(f"Dataset: {path}  vs  {ref_path}")
    statuses = []
    if suffix:
        d["label_names"] = [n.removesuffix(suffix) for n in d["label_names"]]
    if set(r) - set(d):
        print(f"  claves que faltan: {sorted(set(r) - set(d))}")
        statuses.append("different")
    if set(d) - set(r):
        print(f"  claves extra (no cuentan como diferencia): {sorted(set(d) - set(r))}")
    for k in ("feature_names", "label_names", "n_lags", "batch_size", "window_size"):
        same = list(d.get(k, [])) == list(r.get(k, [])) if isinstance(d.get(k), list) else d.get(k) == r.get(k)
        print(f"  {k}: {'igual' if same else f'DISTINTO {d.get(k)} vs {r.get(k)}'}")
        statuses.append("identical" if same else "different")
    statuses.append(compare_arrays("y", d["y"], r["y"]))
    statuses.append(compare_arrays("X", d["X"], r["X"]))
    return statuses


def compare_ckpts(ckpt_dir, ref_dir, suffix=""):
    mine = {p.name.replace(f"{suffix}__", "__") if suffix else p.name: p
            for p in Path(ckpt_dir).glob("*.pkl")}
    ref  = {p.name: p for p in Path(ref_dir).glob("*.pkl")}
    print(f"\nCheckpoints: {len(mine)} vs {len(ref)}")
    statuses = []
    for name in sorted(set(mine) ^ set(ref)):
        print(f"  solo en {'mío' if name in mine else 'ref'}: {name}")
        statuses.append("different")
    n_rounding = 0
    for name in sorted(set(mine) & set(ref)):
        a, b = joblib.load(mine[name]), joblib.load(ref[name])
        if a["seed"] != b["seed"]:
            print(f"  {name}: seed DISTINTA")
            statuses.append("different")
            continue
        if a["X"].shape == b["X"].shape and np.array_equal(a["X"], b["X"]):
            statuses.append("identical")
        elif a["X"].shape == b["X"].shape and np.allclose(a["X"], b["X"], rtol=RTOL, atol=0):
            statuses.append("rounding")
            n_rounding += 1
        else:
            print(f"  {name}: X DISTINTA")
            statuses.append("different")
    if "different" not in statuses:
        print(f"  seeds iguales; X idéntica en {len(statuses) - n_rounding}, "
              f"idéntica salvo redondeo en {n_rounding}")
    return statuses


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("ref")
    p.add_argument("--ckpt")
    p.add_argument("--ref-ckpt")
    p.add_argument("--strip-label-suffix", default="",
                   help="Suffix to remove from the first dataset's labels, e.g. _TG768")
    args = p.parse_args()

    statuses = compare_datasets(args.dataset, args.ref, args.strip_label_suffix)
    if args.ckpt and args.ref_ckpt:
        statuses += compare_ckpts(args.ckpt, args.ref_ckpt, args.strip_label_suffix)

    if "different" in statuses:
        result = "HAY DIFERENCIAS"
    elif "rounding" in statuses:
        result = f"IDÉNTICOS SALVO REDONDEO (rtol={RTOL:g})"
    else:
        result = "IDÉNTICOS"
    print("\nRESULTADO:", result)
    sys.exit(1 if "different" in statuses else 0)


if __name__ == "__main__":
    main()

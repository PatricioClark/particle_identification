"""Per-particle moments of the paired RND512 runs (HPP and ONLD with the same Stokes, same flow).

Same per-particle, per-window statistics as moments_dataset.py of 009 (local L/U scales of each window,
lags in t/T_window), but the paired runs are still short (~19 T), so the time blocks are the windows
themselves: consecutive blocks of 6 T and of 4 T of the flow, one window each, at the same times for
every species (A and B have the same flow bit for bit).

Two details of the paired runs:

  - Only outputs up to --last-file are used: later files belong to the job that is still running (and
    are being overwritten by it).
  - balance.txt is deduplicated. After each restart GHOST appends rows again from the restart time; the
    rows the previous job wrote past that time belong to a flow that was discarded (the restart is from
    single precision fields, so it does not reproduce it). For each time the last row written is kept.
    The spectra and the particle files have one file per output index, so the later job overwrote them.

With --disjoint, two disjoint pools of --n-pool particles are drawn: "M" (training) and "M_test" (test),
so that the test never sees particles the classifier was trained on, as with experimental data.

Usage
-----
mpirun -n 18 python pares_dataset.py --yaml simuls_pares.yaml --out-dir DIR [--last-file 3701] [--disjoint]
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import yaml

from dataset import _glob_sorted, load_simulation
from features import unwrap_positions
from lags_adim import find_balance, find_file, read_params, spectral_L
from moments_dataset import WINDOWS, comm, flow_key, label, local_scales, moment_names, per_particle, rank, size

BLOCKS = ["6", "4"]  # block (= window) lengths in T; lags from WINDOWS of 009


def dedup_balance(path):
    """balance.txt with, for each time, the last row written (the one of the job that is still valid)."""
    bal = np.loadtxt(path)
    _, last = np.unique(bal[::-1, 0], return_index=True)
    return bal[::-1][last]


def build_sim(sim, yaml_i, fp, args):
    path = sim["path"]
    skip = int(sim.get("skip", 0))
    files = [f for f in _glob_sorted(path, "xlg") if int(f.stem.split(".")[-1]) <= args.last_file]
    idx = [int(f.stem.split(".")[-1]) for f in files]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise RuntimeError(f"{sim['name']}: faltan salidas entre {idx[0]} y {idx[-1]}")
    n_use = len(files) - skip
    n_total = int(np.fromfile(files[0], dtype=np.float32, count=1)[0])
    nu = read_params(find_file(path, "parameter.inp"))["nu"]
    bal = dedup_balance(find_balance(path))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, yaml_i]))
    if args.disjoint:
        both = rng.choice(n_total, size=2 * args.n_pool, replace=False)
        pool = np.sort(np.concatenate([both[:args.n_pool], both[args.n_pool:]]))
        is_test = np.isin(pool, both[args.n_pool:])
    else:
        pool = np.sort(rng.choice(n_total, size=min(args.n_pool, n_total), replace=False))
        is_test = None

    times, pos, vel = load_simulation(path, pool, max_steps=n_use, skip=skip)
    dt = float(np.diff(times).mean())
    if np.abs(np.diff(times) - dt).max() > 0.01 * dt:
        raise RuntimeError(f"{sim['name']}: salidas no equiespaciadas en el tiempo")
    pos_u = unwrap_positions(pos.astype(np.float64))
    v64 = vel.astype(np.float64)
    del pos, vel
    print(f"  [rank {rank}] {sim['name']}: salidas {idx[skip]}-{idx[-1]}, t={times[0]:.2f}-{times[-1]:.2f}, "
          f"pool {len(pool)}", flush=True)

    out = {}
    for w in BLOCKS:
        ws = int(round(float(w) * fp["T"] / dt)) + 1
        res = {"M": [], "block": [], "scales": []}
        for k in range(n_use // ws):
            seg = slice(k * ws, (k + 1) * ws)
            sc = local_scales(path, bal, nu, times[seg][0], times[seg][-1])
            lags = np.asarray(WINDOWS[w]) * sc["T"] / dt
            if lags.min() < 0.9 or lags.max() > ws - 2:
                raise RuntimeError(f"{sim['name']} bloque {w}T/{k}: lags {lags} fuera de rango")
            sc["clamped"] = float(lags.min()) if lags.min() < 1 else None
            lags = np.maximum(lags, 1.0)
            sc["lags_steps"] = [float(x) for x in lags]
            M = per_particle(pos_u[:, :, seg], v64[:, :, seg], dt, lags, sc)
            if is_test is None:
                res["M"].append(M)
            else:
                res["M"].append(M[~is_test])
                res.setdefault("M_test", []).append(M[is_test])
            res["block"].append(k)
            res["scales"].append(sc)
        res["M"] = np.stack(res["M"])
        if "M_test" in res:
            res["M_test"] = np.stack(res["M_test"])
        res["block"] = np.array(res["block"])
        res["names"] = moment_names(len(WINDOWS[w]))
        out[w] = res
        t_loc = ", ".join("%.3f" % s["T"] for s in res["scales"])
        print(f"  [rank {rank}] {sim['name']}: {len(res['block'])} bloques de {w} T ({ws} salidas), "
              f"T local {t_loc}", flush=True)
    return {"name": sim["name"], "label": label(sim), "flow": flow_key(sim), "yaml_i": yaml_i,
            "n_pool": args.n_pool if args.disjoint else len(pool), "disjoint": args.disjoint,
            "files": [idx[skip], idx[-1]], "windows": out}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default="simuls_pares.yaml")
    p.add_argument("--flow-params", default="flow_params_007.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--last-file", type=int, default=3701, help="Last output index to use (state N: (N-1)*25+1)")
    p.add_argument("--n-pool", type=int, default=20000)
    p.add_argument("--disjoint", action="store_true", help="Separate pools of particles for training and test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    flows = json.loads(Path(args.flow_params).read_text())["flows"]
    sims = yaml.safe_load(open(args.yaml))
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    for i, sim in list(enumerate(sims))[rank::size]:
        ckpt = out_dir / f"{label(sim)}__sim{i:03d}.pkl"
        if ckpt.exists():
            continue
        res = build_sim(sim, i, flows[flow_key(sim)], args)
        tmp = ckpt.with_name(ckpt.name + f".tmp.{rank}")
        joblib.dump(res, tmp)
        os.replace(tmp, ckpt)
        print(f"  [rank {rank}] {sim['name']} → {ckpt}", flush=True)
    comm.Barrier()
    if rank == 0:
        print("Listo.", flush=True)


if __name__ == "__main__":
    main()

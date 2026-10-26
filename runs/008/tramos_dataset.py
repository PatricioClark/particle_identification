"""Dataset with time-coherent samples ("por tramo"), closer to an experiment.

In runs 006/007 every particle of a sample takes its window from a random
moment of the whole run, and the features are normalised with scales averaged
over the whole run: each sample mixes all the states of the flow. Here:

  - each simulation is cut into consecutive, non-overlapping tramos of one
    window (16.7 T of the flow: 251 outputs in TG768, 296 in RND512);
  - all the particles of a sample share the same tramo (the whole tramo is
    their window), like particles filmed at the same time in one measurement;
  - the flow scales are measured on that tramo (balance.txt and spectra over
    its time span).

Each sample is computed with three normalisations ("variants"):

  V0  flow scales, as in 007 (one U, L, T per flow; lags in t* = t/T_flow).
      Isolates the effect of the time-coherent sampling alone.
  V1  local L/U scales of the tramo: lags in t* = t/T_tramo, MSD/L^2, S2/U^2...
  V2  local Kolmogorov scales of the tramo: lags in units of tau_eta
      (--lags-eta, the 007 lags converted with T/tau_eta of TG768), and
      MSD/eta^2, S2/u_eta^2, S4/u_eta^4, v2/u_eta^2, a2/a_eta^2.

Lags are fractional in steps and interpolated (features.py of 007). Only one
tramo is loaded at a time, so memory stays low. One checkpoint per simulation
(round-robin over MPI ranks); finished checkpoints are skipped on resume.

Usage
-----
mpirun -n N python tramos_dataset.py --yaml simuls.yaml --out-dir ckpt
mpirun -n N python tramos_dataset.py --yaml simuls_nueva.yaml --out-dir ckpt_nueva
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import yaml
from mpi4py import MPI

from dataset import load_simulation, probe_sim
from features import extract_features
from lags_adim import find_balance, find_file, read_params, spectral_L

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

VARIANTS = ("V0", "V1", "V2")


def flow_key(sim):
    return f"{sim['flux']['forcing']}{sim['flux']['resolution'][0]}"


def label(sim):
    return f"{sim['parts']['model']}_St{sim['parts']['st']}_{flow_key(sim)}"


def tramo_scales(sim_path, nu, t0, t1):
    """U, L, T (L/U) and Kolmogorov scales over [t0, t1]; eps = nu <w^2> (balance.txt col 3)."""
    bal_path = find_balance(sim_path)
    bal = np.loadtxt(bal_path)
    m = (bal[:, 0] >= t0) & (bal[:, 0] <= t1)
    U = float(np.sqrt(bal[m, 1].mean()))
    eps = float(nu * bal[m, 2].mean())
    L, n_spec = spectral_L(sim_path, bal_path, t0, t1)
    return {"U": U, "L": L, "T": L / U, "eps": eps, "tau_eta": float(np.sqrt(nu / eps)),
            "eta": float((nu ** 3 / eps) ** 0.25), "u_eta": float((nu * eps) ** 0.25),
            "n_spec": n_spec, "t_span": [float(t0), float(t1)]}


def build_sim(sim, yaml_i, fp, args, lags_tstar, lags_eta):
    path = sim["path"]
    W, dt_flow = fp["window"], fp["dt"]
    skip = int(sim.get("skip", 0))
    n_total, n_steps = probe_sim(path)
    usable = n_steps - skip - 1          # the last output may be incomplete
    n_tr = usable // W
    if n_tr == 0:
        print(f"  [rank {rank}] {sim['name']}: trayectoria ({usable}) más corta que un tramo ({W})", flush=True)
        return None
    per_tramo = int(np.ceil(args.samples_per_sim / n_tr))
    nu = read_params(find_file(path, "parameter.inp"))["nu"]
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, yaml_i]))
    pool = np.sort(rng.choice(n_total, size=min(args.n_load, n_total), replace=False))
    print(f"  [rank {rank}] {sim['name']}: {n_tr} tramos de {W} salidas, {per_tramo} muestras por tramo, "
          f"pool {len(pool)}, nu={nu}", flush=True)

    X = {v: [] for v in VARIANTS}
    tramo_ids, scales, names = [], [], None
    for k in range(n_tr):
        times, pos, vel = load_simulation(path, pool, max_steps=W, skip=skip + k * W)
        dt = float(np.diff(times).mean())
        if abs(dt - dt_flow) > 1e-3 * dt_flow:
            raise RuntimeError(f"{sim['name']}: dt {dt} != dt del flujo {dt_flow}")
        sc = tramo_scales(path, nu, times[0], times[-1])
        variants = {
            "V0": (np.asarray(fp["lags"], float), {"U": fp["U"], "L": fp["L"], "T": fp["T"]}),
            "V1": (lags_tstar * sc["T"] / dt, {"U": sc["U"], "L": sc["L"], "T": sc["T"]}),
            "V2": (lags_eta * sc["tau_eta"] / dt, {"U": sc["u_eta"], "L": sc["eta"], "T": sc["tau_eta"]}),
        }
        # A local T (or tau_eta) slightly shorter than the flow's can put the first lag just
        # below one output (e.g. 0.98): it is set to one output and recorded in sc["clamped"].
        sc["clamped"] = {}
        for v, (lags, s) in list(variants.items()):
            if lags.min() < 0.9 or lags.max() > W - 2:
                raise RuntimeError(f"{sim['name']} tramo {k} {v}: lags {lags} fuera de [0.9, {W - 2}]")
            if lags.min() < 1:
                sc["clamped"][v] = float(lags.min())
                variants[v] = (np.maximum(lags, 1.0), s)
        sc["lags"] = {v: [float(x) for x in lags] for v, (lags, _) in variants.items()}
        scales.append(sc)
        print(f"  [rank {rank}] {sim['name']} tramo {k}: t={sc['t_span'][0]:.2f}-{sc['t_span'][1]:.2f} "
              f"U={sc['U']:.3f} T={sc['T']:.3f} eps={sc['eps']:.4f} tau_eta={sc['tau_eta']:.4f}"
              + (f"  [primer lag llevado a 1: {sc['clamped']}]" if sc["clamped"] else ""), flush=True)
        for _ in range(per_tramo):
            sel = rng.choice(len(pool), size=args.batch_size, replace=False)
            p, vv = pos[sel], vel[sel]
            for v, (lags, s) in variants.items():
                feat, names = extract_features(p, vv, dt, lags, scales=s)
                X[v].append(feat)
            tramo_ids.append(k)
        del pos, vel
    return {"name": sim["name"], "label": label(sim), "flow": flow_key(sim), "yaml_i": yaml_i,
            "expected": sim.get("expected", label(sim)), "skip": skip, "window": W,
            "X": {v: np.array(X[v]) for v in VARIANTS}, "tramo": np.array(tramo_ids),
            "scales": scales, "feature_names": names}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default="simuls.yaml")
    p.add_argument("--flow-params", default="flow_params_007.json",
                   help="Flow scales, window and lags of 007 (used for V0 and for the window length)")
    p.add_argument("--lags-eta", default="0.3731,1.8649,4.8479,16.7816,46.6155",
                   help="Lags in units of tau_eta for V2: the 007 t* lags times T/tau_eta of TG768 (0.2396/0.0429)")
    p.add_argument("--out-dir", default="ckpt")
    p.add_argument("--n-load", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--samples-per-sim", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fpar = json.loads(Path(args.flow_params).read_text())
    flows, lags_tstar = fpar["flows"], np.array(fpar["lags_tstar"])
    lags_eta = np.array([float(x) for x in args.lags_eta.split(",")])
    sims = yaml.safe_load(open(args.yaml))
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{len(sims)} simulaciones, {size} procesos; lags t* = {lags_tstar.tolist()}, "
              f"lags en tau_eta = {lags_eta.tolist()}", flush=True)
    comm.Barrier()

    for i, sim in list(enumerate(sims))[rank::size]:
        ckpt = out_dir / f"{label(sim)}__sim{i:03d}.pkl"
        if ckpt.exists():
            print(f"  [rank {rank}] {ckpt.name} ya existe, lo salteo", flush=True)
            continue
        if not Path(sim["path"]).is_dir():
            print(f"  [rank {rank}] {sim['name']}: no existe {sim['path']}", flush=True)
            continue
        res = build_sim(sim, i, flows[flow_key(sim)], args, lags_tstar, lags_eta)
        if res is None:
            continue
        tmp = ckpt.with_name(ckpt.name + f".tmp.{rank}")
        joblib.dump(res, tmp)
        os.replace(tmp, ckpt)
        print(f"  [rank {rank}] {sim['name']}: {len(res['tramo'])} muestras → {ckpt}", flush=True)
    comm.Barrier()
    if rank == 0:
        print("Listo.", flush=True)


if __name__ == "__main__":
    main()

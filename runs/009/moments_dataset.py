"""Per-particle, per-window statistics, to build any kind of sample afterwards.

All the features of a sample (features.py) are averages over its particles of
per-particle quantities (VACF numerator and denominator, MSD, S2, S4, <v^2>,
<v^4>, <a^2>, <a^4>), combined at the end into ratios (VACF, kurtosis,
flatness). So the per-particle averages are computed once, for every particle
of a pool and every time window, and a sample is just the average over the
(particle, window) pairs it is made of:

  - time-coherent sample: 5000 particles, all in the same window (as 008);
  - mixed sample: 5000 particles, each in a window of its own (as 006/007),
    restricted to the training blocks.

Windows. Each simulation is cut into consecutive blocks of 16.7 T of the flow
(the 008 tramos: 251 outputs in TG768, 296 in RND512). Inside each block,
consecutive sub-windows of 16.7 T (1), 6 T (2) and 4 T (4). The lags are the
007 t* lags up to half the window: 16.7 T: 0.0668, 0.334, 0.868, 3.00, 8.35;
6 T: 0.0668, 0.334, 0.868, 3.0; 4 T: 0.0668, 0.334, 0.868, 2.0.

Normalisation: local L/U scales of each window (V1 of 008): lags in t/T_window,
velocities by U_window, positions by L_window, accelerations by U/T. Every
particle is normalised with the scales of its own window, so mixed samples
combine windows consistently. Fractional lags are interpolated per particle
between the neighbouring integer lags (log-log for MSD, S2, S4; linear for
the VACF numerator), so values can differ slightly from features.py, which
interpolates after averaging.

Usage
-----
mpirun -n N python moments_dataset.py --yaml simuls.yaml --out-dir DIR [--n-pool 20000]
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
from features import unwrap_positions
from lags_adim import find_balance, find_file, read_params, spectral_L

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

WINDOWS = {  # window length in T -> lags in t*
    "16.7": [0.0668, 0.3339, 0.868, 3.0047, 8.3464],
    "6": [0.0668, 0.3339, 0.868, 3.0],
    "4": [0.0668, 0.3339, 0.868, 2.0],
}


def flow_key(sim):
    return f"{sim['flux']['forcing']}{sim['flux']['resolution'][0]}"


def label(sim):
    return f"{sim['parts']['model']}_St{sim['parts']['st']}_{flow_key(sim)}"


def moment_names(n_lags):
    return ([f"vnum_l{i:02d}" for i in range(n_lags)] + [f"msd_l{i:02d}" for i in range(n_lags)]
            + [f"s2_l{i:02d}" for i in range(n_lags)] + [f"s4_l{i:02d}" for i in range(n_lags)]
            + ["v2", "v4", "a2", "a4"])


def local_scales(sim_path, bal, nu, t0, t1):
    m = (bal[:, 0] >= t0) & (bal[:, 0] <= t1)
    U = float(np.sqrt(bal[m, 1].mean()))
    L, n_spec = spectral_L(sim_path, find_balance(sim_path), t0, t1)
    return {"U": U, "L": L, "T": L / U, "eps": float(nu * bal[m, 2].mean()), "n_spec": n_spec,
            "t_span": [float(t0), float(t1)]}


def per_particle(pos_u, vel, dt, lags, sc):
    """Per-particle normalised moments for one window. pos_u, vel: (n_p, 3, W) float64."""
    U, L, T = sc["U"], sc["L"], sc["T"]
    lags = np.asarray(lags, float)
    lo = np.floor(lags + 1e-9).astype(int)
    hi = np.where(lags - lo > 1e-9, lo + 1, lo)
    W = vel.shape[2]

    def at(tau):
        n = W - tau
        dv = vel[:, :, tau:tau + n] - vel[:, :, :n]
        dvm2 = (dv ** 2).sum(axis=1)                      # (n_p, n)
        disp = pos_u[:, :, tau:tau + n] - pos_u[:, :, :n]
        return ((vel[:, :, :n] * vel[:, :, tau:tau + n]).mean(axis=(1, 2)) / U ** 2,
                (disp ** 2).mean(axis=(1, 2)) / L ** 2,
                dvm2.mean(axis=1) / U ** 2,
                (dvm2 ** 2).mean(axis=1) / U ** 4)

    cache = {}
    vnum, msd, s2, s4 = [], [], [], []
    for l, a, b in zip(lags, lo, hi):
        for t in (a, b):
            if t not in cache:
                cache[t] = at(t)
        A, B = cache[a], cache[b]
        if a == b:
            w = 0.0
        else:
            w = (np.log(l) - np.log(a)) / (np.log(b) - np.log(a))
        wl = (l - a) / (b - a) if b > a else 0.0
        vnum.append((1 - wl) * A[0] + wl * B[0])
        for out, i in ((msd, 1), (s2, 2), (s4, 3)):
            xa, xb = np.maximum(A[i], 1e-300), np.maximum(B[i], 1e-300)
            out.append(np.exp((1 - w) * np.log(xa) + w * np.log(xb)))
    v2 = (vel ** 2).mean(axis=(1, 2)) / U ** 2
    v4 = (vel ** 4).mean(axis=(1, 2)) / U ** 4
    acc = np.diff(vel, axis=2) / dt
    aU = U / T
    a2 = (acc ** 2).mean(axis=(1, 2)) / aU ** 2
    a4 = (acc ** 4).mean(axis=(1, 2)) / aU ** 4
    return np.column_stack(vnum + msd + s2 + s4 + [v2, v4, a2, a4]).astype(np.float32)


def build_sim(sim, yaml_i, fp, args):
    path = sim["path"]
    Wb, T_flow = fp["window"], fp["T"]
    skip = int(sim.get("skip", 0))
    n_total, n_steps = probe_sim(path)
    n_blocks = (n_steps - skip - 1) // Wb
    if n_blocks == 0:
        return None
    nu = read_params(find_file(path, "parameter.inp"))["nu"]
    bal = np.loadtxt(find_balance(path))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, yaml_i]))
    pool = np.sort(rng.choice(n_total, size=min(args.n_pool, n_total), replace=False))
    out = {w: {"M": [], "block": [], "sub": [], "scales": []} for w in WINDOWS}
    for k in range(n_blocks):
        times, pos, vel = load_simulation(path, pool, max_steps=Wb, skip=skip + k * Wb)
        dt = float(np.diff(times).mean())
        pos_u = unwrap_positions(pos.astype(np.float64))
        v64 = vel.astype(np.float64)
        del pos, vel
        for w, lags_t in WINDOWS.items():
            ws = Wb if w == "16.7" else int(round(float(w) * T_flow / dt)) + 1
            for j in range(Wb // ws):
                seg = slice(j * ws, (j + 1) * ws)
                sc = local_scales(path, bal, nu, times[seg][0], times[seg][-1])
                lags = np.asarray(lags_t) * sc["T"] / dt
                if lags.min() < 0.9 or lags.max() > ws - 2:
                    raise RuntimeError(f"{sim['name']} bloque {k} ventana {w}T/{j}: lags {lags} fuera de rango")
                sc["clamped"] = float(lags.min()) if lags.min() < 1 else None
                lags = np.maximum(lags, 1.0)
                sc["lags_steps"] = [float(x) for x in lags]
                out[w]["M"].append(per_particle(pos_u[:, :, seg], v64[:, :, seg], dt, lags, sc))
                out[w]["block"].append(k)
                out[w]["sub"].append(j)
                out[w]["scales"].append(sc)
        print(f"  [rank {rank}] {sim['name']} bloque {k + 1}/{n_blocks} t={times[0]:.2f}-{times[-1]:.2f}", flush=True)
        del pos_u, v64
    for w in WINDOWS:
        out[w]["M"] = np.stack(out[w]["M"])               # (n_windows, n_pool, n_moments)
        out[w]["block"] = np.array(out[w]["block"])
        out[w]["sub"] = np.array(out[w]["sub"])
        out[w]["names"] = moment_names(len(WINDOWS[w]))
    return {"name": sim["name"], "label": label(sim), "flow": flow_key(sim), "yaml_i": yaml_i,
            "expected": sim.get("expected", label(sim)), "n_pool": len(pool), "windows": out}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default="simuls.yaml")
    p.add_argument("--flow-params", default="flow_params_007.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-pool", type=int, default=20000)
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
        if ckpt.exists() or not Path(sim["path"]).is_dir():
            continue
        res = build_sim(sim, i, flows[flow_key(sim)], args)
        if res is None:
            continue
        tmp = ckpt.with_name(ckpt.name + f".tmp.{rank}")
        joblib.dump(res, tmp)
        os.replace(tmp, ckpt)
        print(f"  [rank {rank}] {sim['name']} → {ckpt}", flush=True)
    comm.Barrier()
    if rank == 0:
        print("Listo.", flush=True)


if __name__ == "__main__":
    main()

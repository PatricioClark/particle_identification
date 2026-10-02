"""Inspect ensemble statistics vs lag in turnover-time units (t* = t / T, T = L/U).

Like inspect_signals.py, but every simulation is put in its own flow units so
that TG768 and RND512 can be compared on the same axes:

  x axis : t* = lag_time / T,   T = L / U (time-averaged, see lags_adim.py)
  VACF   : as is
  S2/U^2, S4/U^4, MSD/L^2

Each simulation loads as many steps as it needs to reach lags up to --t-max
(in units of T), so all curves cover the same t* range. The lags chosen for
run 006 (in t*) are marked as red dotted lines and the window as a dashed line.

Usage
-----
mpirun -n N python inspect_signals_adim.py [--yaml simuls.yaml] [--n-particles 2000]
       [--n-lags 50] [--t-max 20] [--mark-lags 0.0668,0.3339,0.868,3.0047,8.3464]
       [--window 16.69] [--output inspect_signals_adim.pdf]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml
from mpi4py import MPI

from dataset import probe_sim, load_simulation
from features import make_lags, unwrap_positions, _vacf, _msd, _sf
from lags_adim import sim_scales

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MODEL_CMAPS = {"LAG": "Greys", "MR": "Blues", "NLD": "Purples", "ONLD": "Reds",
               "BB": "Greens", "FAX": "Oranges", "HPP": "YlOrBr"}


def load_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default="simuls.yaml")
    p.add_argument("--n-particles", type=int, default=2000)
    p.add_argument("--n-lags", type=int, default=50)
    p.add_argument("--t-max", type=float, default=20.0, help="Largest lag, in units of T")
    p.add_argument("--mark-lags", default="0.0668,0.3339,0.868,3.0047,8.3464",
                   help="Lags to mark, in units of T (run-006 lags from lags_adim.py)")
    p.add_argument("--window", type=float, default=16.69, help="Window to mark, in units of T")
    p.add_argument("--output", default="inspect_signals_adim.pdf")
    return p.parse_args()


def compute_stats(sim, n_particles, n_lags, t_max, rng):
    path = sim["path"]
    n_total, n_steps = probe_sim(path)
    if n_total == 0 or n_steps == 0:
        return None
    sc = sim_scales(path, "L/U")
    if sc is None:
        return None
    T, U, L, dt = sc["T"], sc["U"], sc["L"], sc["dt"]

    # Steps needed so that make_lags (max lag = steps // 2) reaches t_max * T
    steps_needed = int(np.ceil(2 * t_max * T / dt)) + 2
    max_steps = min(steps_needed, n_steps)

    idxs = np.sort(rng.choice(n_total, size=min(n_particles, n_total), replace=False))
    times, pos, vel = load_simulation(path, idxs, max_steps=max_steps)
    dt_data = float(np.diff(times).mean())
    lags = make_lags(times.shape[0] - 1, n_lags)

    pos_u = unwrap_positions(pos.astype(np.float64))
    v64 = vel.astype(np.float64)
    return {
        "t_star": lags * dt_data / T,
        "vacf": _vacf(v64, lags),
        "s2": _sf(v64, lags, 2) / U**2,
        "s4": _sf(v64, lags, 4) / U**4,
        "msd": _msd(pos_u, lags) / L**2,
        "T": T, "U": U, "L": L, "dt": dt_data, "steps": times.shape[0],
    }


def style_for(sims_info):
    """Color by model (shade by Stokes), solid for 768, dashed for 512."""
    styles = {}
    by_model = {}
    for name, model, st, res in sims_info:
        by_model.setdefault(model, []).append((st, name))
    for model, items in by_model.items():
        cmap = plt.get_cmap(MODEL_CMAPS.get(model, "viridis"))
        sts = sorted(set(st for st, _ in items))
        for st, name in items:
            frac = 0.45 + 0.5 * (sts.index(st) / max(1, len(sts) - 1))
            styles[name] = {"color": cmap(frac)}
    for name, model, st, res in sims_info:
        styles[name]["linestyle"] = "-" if res == 768 else "--"
    return styles


def main():
    args = load_args()
    with open(args.yaml) as fh:
        sims = yaml.safe_load(fh)

    accessible = [s for s in sims if Path(s["path"]).is_dir() and probe_sim(s["path"])[1] > 0]
    if not accessible:
        if rank == 0:
            print("No accessible simulation directories found.", file=sys.stderr)
        sys.exit(1)
    if rank == 0:
        print(f"Accessible simulations: {len(accessible)}, MPI ranks: {size}, t_max = {args.t_max} T")

    local = []
    for i, sim in list(enumerate(accessible))[rank::size]:
        print(f"  [rank {rank}] computing {sim['name']} …", flush=True)
        stats = compute_stats(sim, args.n_particles, args.n_lags, args.t_max,
                              np.random.default_rng(42 + i))
        if stats is not None:
            print(f"  [rank {rank}] {sim['name']}: T={stats['T']:.3f}, {stats['steps']} pasos", flush=True)
            local.append((sim["name"], sim["parts"]["model"], sim["parts"]["st"],
                          sim["flux"]["resolution"][0], stats))

    gathered = comm.gather(local, root=0)
    if rank != 0:
        return

    results = [r for bucket in gathered for r in bucket]
    order = {s["name"]: i for i, s in enumerate(accessible)}
    results.sort(key=lambda r: order[r[0]])
    styles = style_for([(n, m, st, res) for n, m, st, res, _ in results])
    mark = [float(x) for x in args.mark_lags.split(",")] if args.mark_lags else []

    panels = [("vacf", "VACF", False), ("msd", "MSD / $L^2$", True),
              ("s2", "$S_2 / U^2$", True), ("s4", "$S_4 / U^4$", True)]

    with PdfPages(args.output) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ax, (key, title, logy) in zip(axes.flat, panels):
            for name, model, st, res, stats in results:
                ax.plot(stats["t_star"], stats[key], label=name, linewidth=1.3, **styles[name])
            for t in mark:
                ax.axvline(t, color="r", linestyle=":", linewidth=1.0, alpha=0.8)
            if args.window:
                ax.axvline(args.window, color="k", linestyle="--", linewidth=0.9, alpha=0.6)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("t / T   (T = L/U)")
            ax.set_title(title)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", fontsize=8, frameon=False)
        fig.suptitle(
            f"Estadísticas vs lag en tiempos de giro (n_particles={args.n_particles}, n_lags={args.n_lags})\n"
            f"Línea llena = TG768, discontinua = RND512.  Rojo punteado = lags de 006 "
            f"(t* = {', '.join(f'{t:.3g}' for t in mark)}).  Negro = ventana ({args.window:.3g} T)",
            fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.84, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

    print("\nEscalas usadas:")
    for name, model, st, res, stats in results:
        print(f"  {name:18s} T={stats['T']:.3f}  U={stats['U']:.3f}  L={stats['L']:.3f}  "
              f"dt={stats['dt']:.4f}  pasos={stats['steps']}")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

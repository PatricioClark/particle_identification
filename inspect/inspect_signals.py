"""Inspect ensemble statistics as a function of physical lag time.

Loads each accessible simulation in parallel (one MPI rank per simulation),
computes VACF, MSD, S2, and S4 over a dense lag grid, and plots them against
physical lag time on rank 0.  A vertical dashed line marks the max lag
currently used by the training pipeline (min_steps/2).

Usage
-----
mpirun -n N python inspect_signals.py [--yaml simuls.yaml] [--n-particles 1000]
                                       [--n-lags 50] [--output inspect_signals.pdf]

N should be <= number of accessible simulations; extra ranks are idle.
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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml",         default="simuls.yaml")
    p.add_argument("--n-particles",  type=int, default=1000)
    p.add_argument("--n-lags",       type=int, default=50)
    p.add_argument("--output",       default="inspect_signals.pdf")
    return p.parse_args()


def compute_stats(sim_path, n_particles, n_lags, max_steps, rng):
    n_total, n_steps = probe_sim(sim_path)
    if n_total == 0 or n_steps == 0:
        return None

    n_load = min(n_particles, n_total)
    idxs   = rng.choice(n_total, size=n_load, replace=False)

    times, pos, vel = load_simulation(sim_path, idxs, max_steps=max_steps)
    dt = float(np.diff(times).mean()) if len(times) > 1 else 1.0

    effective_steps = times.shape[0] - 1
    lags = make_lags(effective_steps, n_lags)
    lag_times = dt * lags

    pos_u = unwrap_positions(pos.astype(np.float64))
    v64   = vel.astype(np.float64)

    return {
        "lag_times": lag_times,
        "vacf":      _vacf(v64, lags),
        "msd":       _msd(pos_u, lags),
        "s2":        _sf(v64, lags, 2),
        "s4":        _sf(v64, lags, 4),
        "dt":        dt,
        "n_steps":   times.shape[0],
    }


def main():
    args = load_args()

    with open(args.yaml) as fh:
        sims = yaml.safe_load(fh)

    # All ranks: filter to accessible sims (fast filesystem check)
    accessible = [
        s for s in sims
        if Path(s["path"]).is_dir() and probe_sim(s["path"])[1] > 0
    ]

    if not accessible:
        if rank == 0:
            print("No accessible simulation directories found.", file=sys.stderr)
        sys.exit(1)

    # Round-robin: rank r handles accessible[r], accessible[r+size], …
    my_sims = list(enumerate(accessible))[rank::size]

    # All ranks compute min_steps independently — same filesystem view
    min_steps = min(probe_sim(s["path"])[1] for s in accessible)

    if rank == 0:
        print(f"Accessible simulations: {len(accessible)}, "
              f"MPI ranks: {size}, min_steps: {min_steps}")

    # ── Pass 2: each rank computes stats for its simulations ──────────────────
    local_results = []
    for i, sim in my_sims:
        model = sim["parts"]["model"]
        st    = sim["parts"]["st"]
        label = f"{model}_St{st}"
        rng   = np.random.default_rng(42 + i)
        print(f"  [rank {rank}] computing {label} …")
        stats = compute_stats(sim["path"], args.n_particles, args.n_lags,
                              max_steps=min_steps, rng=rng)
        if stats is not None:
            local_results.append((label, stats))

    # ── Gather to rank 0 ──────────────────────────────────────────────────────
    all_results = comm.gather(local_results, root=0)

    if rank != 0:
        return

    results = {
        label: stats
        for bucket in all_results
        for label, stats in bucket
    }

    if not results:
        print("No statistics computed.", file=sys.stderr)
        sys.exit(1)

    first = next(iter(results.values()))
    pipeline_max_lag_time = (min_steps // 2) * first["dt"]

    # ── Plot (rank 0 only) ────────────────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    panels = [
        ("vacf", "VACF",  False, False),
        ("msd",  "MSD",   True,  True),
        ("s2",   "$S_2$", True,  True),
        ("s4",   "$S_4$", True,  True),
    ]

    with PdfPages(args.output) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(
            f"Ensemble statistics vs lag time  "
            f"(n_particles={args.n_particles}, n_lags={args.n_lags})\n"
            f"Dashed line = max lag used by pipeline (min_steps/2 = {min_steps//2} steps)",
            fontsize=11,
        )

        for ax, (key, title, logx, logy) in zip(axes.flat, panels):
            for (label, stats), color in zip(results.items(), colors):
                ax.plot(stats["lag_times"], stats[key],
                        label=label, color=color, linewidth=1.2)

            ax.axvline(pipeline_max_lag_time, color="k", linestyle="--",
                       linewidth=0.9, alpha=0.6)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Lag time")
            ax.set_title(title)
            ax.legend(fontsize=7, loc="best")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

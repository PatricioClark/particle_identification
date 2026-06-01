"""Inspect ensemble statistics as a function of physical lag time.

Loads each accessible simulation, computes VACF, MSD, S2, and S4 over a
dense lag grid, and plots them against physical lag time.  A vertical dashed
line marks the max lag currently used by the training pipeline (min_steps/2).

Usage
-----
python inspect_signals.py [--yaml simuls.yaml] [--n-particles 1000]
                          [--n-lags 50] [--output inspect_signals.pdf]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml

from dataset import probe_sim, load_simulation
from features import make_lags, unwrap_positions, _vacf, _msd, _sf


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

    # Pass 1: find min_steps across all accessible sims
    n_steps_map = {}
    for sim in sims:
        path = sim["path"]
        if not Path(path).is_dir():
            continue
        _, n_steps = probe_sim(path)
        if n_steps > 0:
            n_steps_map[path] = n_steps

    if not n_steps_map:
        raise RuntimeError("No accessible simulation directories found.")

    min_steps = min(n_steps_map.values())
    print(f"Trajectory lengths: min={min_steps}, max={max(n_steps_map.values())} steps")

    # Pass 2: compute statistics per simulation
    rng     = np.random.default_rng(42)
    results = {}

    for sim in sims:
        path  = sim["path"]
        model = sim["parts"]["model"]
        st    = sim["parts"]["st"]
        label = f"{model}_St{st}"

        if path not in n_steps_map:
            print(f"  [skip] {label}")
            continue

        print(f"  computing {label} …")
        stats = compute_stats(path, args.n_particles, args.n_lags,
                              max_steps=min_steps, rng=rng)
        if stats is not None:
            results[label] = stats

    if not results:
        raise RuntimeError("No statistics computed.")

    # Vertical line: max lag used by the pipeline = min_steps/2 * dt
    # Use dt from the first result (all sims share the same dt in this dataset)
    first = next(iter(results.values()))
    pipeline_max_lag_time = (min_steps // 2) * first["dt"]

    # ── Plot ─────────────────────────────────────────────────────────────────
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

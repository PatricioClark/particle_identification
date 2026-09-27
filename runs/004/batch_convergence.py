"""Batch-size convergence check for ensemble feature estimates.

For a single simulation, draws K independent batches at each of several
batch sizes and measures the coefficient of variation (CV = std/|mean|) of
every feature across those draws.  If the CV falls well below 1 at the
working batch size, the estimate has converged.

Expected output: one PDF page per feature group (VACF, MSD, S2, S4, scalars)
with CV vs batch-size curves; each curve is one feature, coloured by lag
index.  A vertical dashed line marks the working batch size.

Usage
-----
python inspect/batch_convergence.py --path /path/to/sim [options]

python inspect/batch_convergence.py \\
    --path /share/scratch8/bespanol/MR/St2 \\
    --batch-sizes 50,100,250,500,1000,2500,5000 \\
    --n-repeats 50 --n-lags 15 --max-steps 400 \\
    --working-batch-size 5000 \\
    --output batch_convergence.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Allow running from the repo root or from inspect/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import probe_sim, load_simulation
from features import make_lags, extract_features


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--path",               required=True,
                   help="simulation directory to analyse")
    p.add_argument("--batch-sizes",        default="50,100,250,500,1000,2500,5000",
                   help="comma-separated list of batch sizes to sweep")
    p.add_argument("--n-repeats",          type=int, default=50,
                   help="independent batches per batch-size point")
    p.add_argument("--n-lags",             type=int, default=15)
    p.add_argument("--max-steps",          type=int, default=None)
    p.add_argument("--n-load",             type=int, default=None,
                   help="particle pool size (default: load all)")
    p.add_argument("--working-batch-size", type=int, default=5000,
                   help="current training batch size (shown as dashed line)")
    p.add_argument("--seed",               type=int, default=0)
    p.add_argument("--output",             default="batch_convergence.pdf")
    return p.parse_args()


def compute_cv_matrix(pos, vel, dt, lags, batch_sizes, n_repeats, seed):
    """Return CV matrix: (len(batch_sizes), n_features).

    For each batch size draw n_repeats independent batches and compute
    std / |mean| across those draws for each feature.
    """
    rng = np.random.default_rng(seed)
    pool_size = pos.shape[0]

    cv_rows = []
    for bs in batch_sizes:
        if bs > pool_size:
            print(f"  [skip] batch_size={bs} > pool_size={pool_size}")
            cv_rows.append(np.full(4 * len(lags) + 4, np.nan))
            continue

        samples = []
        for _ in range(n_repeats):
            sel = rng.choice(pool_size, size=bs, replace=False)
            feat, names = extract_features(pos[sel], vel[sel], dt, lags)
            samples.append(feat)

        arr = np.array(samples)          # (n_repeats, n_features)
        mean = np.abs(arr.mean(axis=0))
        std  = arr.std(axis=0)
        cv   = np.where(mean > 1e-12, std / mean, np.nan)
        cv_rows.append(cv)
        print(f"  batch_size={bs:6d}  median CV={np.nanmedian(cv):.4f}")

    return np.array(cv_rows), names


def _lag_colours(n_lags):
    return plt.cm.viridis(np.linspace(0, 1, n_lags))


def plot_group(ax, batch_sizes, cv_matrix, col_indices, title, working_bs, n_lags):
    """Plot CV vs batch-size for a group of lag-indexed features."""
    colours = _lag_colours(n_lags)
    for k, ci in enumerate(col_indices):
        ax.plot(batch_sizes, cv_matrix[:, ci], color=colours[k % len(colours)],
                marker="o", ms=3, lw=1.2)
    ax.axvline(working_bs, color="red", ls="--", lw=1, label=f"working N={working_bs}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size (particles)")
    ax.set_ylabel("CV  (std / |mean|)")
    ax.set_title(title)
    ax.legend(fontsize=7)


def plot_scalars(ax, batch_sizes, cv_matrix, col_indices, names, working_bs):
    """Plot CV vs batch-size for the four scalar features."""
    colours = plt.cm.tab10(np.linspace(0, 1, len(col_indices)))
    for k, ci in enumerate(col_indices):
        ax.plot(batch_sizes, cv_matrix[:, ci], color=colours[k],
                marker="o", ms=4, lw=1.5, label=names[ci])
    ax.axvline(working_bs, color="red", ls="--", lw=1, label=f"working N={working_bs}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size (particles)")
    ax.set_ylabel("CV  (std / |mean|)")
    ax.set_title("Scalar features")
    ax.legend(fontsize=8)


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    sim_path = args.path
    n_total, n_steps = probe_sim(sim_path)
    if n_total == 0:
        print(f"No particles found in {sim_path}")
        sys.exit(1)

    max_steps = args.max_steps
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    effective_steps = n_steps - 1
    lags = make_lags(effective_steps, args.n_lags)
    print(f"Simulation: {sim_path}")
    print(f"  n_particles={n_total}, n_steps={n_steps}")
    print(f"  lag indices ({args.n_lags}): {lags}")

    pool_size = args.n_load if args.n_load is not None else n_total
    pool_size = min(pool_size, n_total)
    max_needed = max(bs for bs in batch_sizes if bs <= pool_size)
    pool_size  = max(pool_size, max_needed)

    rng = np.random.default_rng(args.seed)
    pool_idxs = rng.choice(n_total, size=pool_size, replace=False)
    print(f"  loading pool of {pool_size} particles …")

    times, pos, vel = load_simulation(sim_path, pool_idxs, max_steps=max_steps)
    dt = float(np.diff(times).mean()) if len(times) > 1 else 1.0
    print(f"  loaded: pos={pos.shape}, vel={vel.shape}, dt={dt:.4g}\n")

    print("Computing CV at each batch size …")
    cv_matrix, feat_names = compute_cv_matrix(
        pos, vel, dt, lags, batch_sizes, args.n_repeats, args.seed + 1,
    )

    n_lags = len(lags)
    groups = {
        "VACF":    list(range(0,            n_lags)),
        "MSD":     list(range(n_lags,       2 * n_lags)),
        "S2":      list(range(2 * n_lags,   3 * n_lags)),
        "S4":      list(range(3 * n_lags,   4 * n_lags)),
    }
    scalar_cols = list(range(4 * n_lags, 4 * n_lags + 4))
    group_titles = {
        "VACF": "VACF — coefficient of variation vs batch size",
        "MSD":  "MSD  — coefficient of variation vs batch size",
        "S2":   "S2   — coefficient of variation vs batch size",
        "S4":   "S4   — coefficient of variation vs batch size",
    }

    print(f"\nWriting {args.output} …")
    with PdfPages(args.output) as pdf:
        # Summary page: all groups in one figure
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(
            f"Batch-size convergence  |  {Path(sim_path).name}  |  "
            f"{args.n_repeats} repeats per point",
            fontsize=12,
        )

        for ax, (key, cols) in zip(axes.flat[:4], groups.items()):
            plot_group(ax, batch_sizes, cv_matrix, cols,
                       group_titles[key], args.working_batch_size, n_lags)

        plot_scalars(axes.flat[4], batch_sizes, cv_matrix,
                     scalar_cols, feat_names, args.working_batch_size)

        # Add a colourbar legend for lag index on last free axis
        ax_cb = axes.flat[5]
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(vmin=0, vmax=n_lags - 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax_cb, orientation="vertical", fraction=0.8, pad=0.05)
        ax_cb.set_title("Lag index\n(colour scale)", fontsize=9)
        ax_cb.axis("off")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Detailed pages: one page per group
        for key, cols in groups.items():
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_group(ax, batch_sizes, cv_matrix, cols,
                       group_titles[key], args.working_batch_size, n_lags)
            sm = plt.cm.ScalarMappable(cmap="viridis",
                                       norm=plt.Normalize(vmin=0, vmax=n_lags - 1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Lag index")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Detailed scalars page
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_scalars(ax, batch_sizes, cv_matrix,
                     scalar_cols, feat_names, args.working_batch_size)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()

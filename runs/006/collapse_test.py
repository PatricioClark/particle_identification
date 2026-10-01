"""Collapse test: which turnover-time definition makes TG768 and RND512 comparable?

Tracers only follow the fluid. If a time scale T correctly maps one flow onto
the other, the tracer statistics plotted against t/T (with amplitudes
normalised by the rms velocity U) should overlap. RND512 LAG cannot be used
(its stored positions advance ~81x slower than its velocity), so the TG768
tracers are compared with the lightest RND512 particles (St = 0.1), which
behave almost like tracers except at the shortest lags.

Candidate time scales, computed per simulation:
  2pi/U    box size over rms velocity, U = sqrt(<v^2>) from balance.txt col 2
  L/U      spectral integral scale L = int(E/k) / int(E) (last spectra), over U
  tau_L    first zero crossing of the tracers' velocity autocorrelation
  tau_eta  Kolmogorov time sqrt(nu/eps), eps = nu * <omega^2> (balance.txt col 3)

Amplitude normalisation (same for every candidate):
  VACF as is, S2 / U^2, S4 / U^4, MSD / (U T)^2

Collapse metric on the common t/T range: mean |log10 a - log10 b| for S2, S4,
MSD and mean |a - b| for the VACF. Smaller = better overlap.

Usage
-----
python collapse_test.py [--yaml simuls.yaml] [--sims "TG768 LAG,RND512 HPP 0.1,RND512 ONLD 0.1"]
                        [--n-particles 5000] [--max-steps 900] [--n-lags 60]
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml

from dataset import probe_sim, load_simulation
from features import make_lags, unwrap_positions, _vacf, _msd, _sf

CANDIDATES = ["2pi/U", "L/U", "tau_L", "tau_eta"]
QUANTITIES = [("vacf", "VACF"), ("s2", "$S_2/U^2$"), ("s4", "$S_4/U^4$"), ("msd", "MSD$/(UT)^2$")]


def find_file(sim_path, name):
    """File next to the sim, or in the run dir / its todir (RND512 layout)."""
    p = Path(sim_path)
    for cand in (p / name, p.parent / "todir" / name, p.parent / name):
        if cand.exists():
            return cand
    return None


def read_nu(sim_path):
    par = find_file(sim_path, "parameter.inp")
    for line in par.read_text().splitlines():
        m = re.match(r"\s*nu\s*=\s*([0-9.eE+-]+)", line)
        if m:
            return float(m.group(1))
    raise ValueError(f"nu not found in {par}")


def spectral_L(sim_path, n_spectra=5):
    bal = find_file(sim_path, "balance.txt")
    files = sorted((f for f in bal.parent.glob("kspectrum.*.txt") if f.stat().st_size > 0),
                   key=lambda f: int(f.stem.split(".")[1]))
    Ls = []
    for f in files[-n_spectra:]:
        k, E = np.loadtxt(f, unpack=True)
        m = k > 0
        Ls.append(np.trapezoid(E[m] / k[m], k[m]) / np.trapezoid(E[m], k[m]))
    return float(np.mean(Ls))


def zero_crossing(lag_times, vacf):
    idx = np.where(vacf <= 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    t0, t1, c0, c1 = lag_times[i - 1], lag_times[i], vacf[i - 1], vacf[i]
    return float(t0 + (t1 - t0) * c0 / (c0 - c1))


def sim_stats(sim, n_particles, max_steps, n_lags, rng):
    path = sim["path"]
    n_total, n_steps = probe_sim(path)
    idxs = np.sort(rng.choice(n_total, size=min(n_particles, n_total), replace=False))
    times, pos, vel = load_simulation(path, idxs, max_steps=max_steps)
    dt = float(np.diff(times).mean())

    lags = make_lags(times.shape[0] - 1, n_lags)
    lag_times = lags * dt
    pos_u = unwrap_positions(pos.astype(np.float64))
    v64 = vel.astype(np.float64)
    stats = {"lag_times": lag_times, "vacf": _vacf(v64, lags), "msd": _msd(pos_u, lags),
             "s2": _sf(v64, lags, 2), "s4": _sf(v64, lags, 4)}

    bal = np.loadtxt(find_file(path, "balance.txt"))
    span = (bal[:, 0] >= times[0]) & (bal[:, 0] <= times[-1])
    U = float(np.sqrt(bal[span, 1].mean()))
    nu = read_nu(path)
    eps = nu * float(bal[span, 2].mean())
    L = spectral_L(path)

    scales = {"2pi/U": 2 * np.pi / U, "L/U": L / U,
              "tau_L": zero_crossing(lag_times, stats["vacf"]),
              "tau_eta": float(np.sqrt(nu / eps))}
    return {"name": sim["name"], "dt": dt, "U": U, "L": L, "nu": nu, "eps": eps,
            "n_steps_used": times.shape[0], "scales": scales, "stats": stats}


def normalised(res, cand):
    T, U, st = res["scales"][cand], res["U"], res["stats"]
    x = st["lag_times"] / T
    return x, {"vacf": st["vacf"], "s2": st["s2"] / U**2, "s4": st["s4"] / U**4,
               "msd": st["msd"] / (U * T) ** 2}


def collapse_metric(a, b, cand):
    xa, ya = normalised(a, cand)
    xb, yb = normalised(b, cand)
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    if not np.isfinite(lo + hi) or hi <= lo:
        return {q: np.nan for q, _ in QUANTITIES}, (lo, hi)
    grid = np.logspace(np.log10(lo), np.log10(hi), 50)
    out = {}
    for q, _ in QUANTITIES:
        if q == "vacf":
            out[q] = float(np.mean(np.abs(np.interp(grid, xa, ya[q]) - np.interp(grid, xb, yb[q]))))
        else:
            la = np.interp(np.log10(grid), np.log10(xa), np.log10(ya[q]))
            lb = np.interp(np.log10(grid), np.log10(xb), np.log10(yb[q]))
            out[q] = float(np.mean(np.abs(la - lb)))
    return out, (lo, hi)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default="simuls.yaml")
    p.add_argument("--sims", default="TG768 LAG,RND512 HPP 0.1,RND512 ONLD 0.1",
                   help="First one is the reference; the rest are compared against it")
    p.add_argument("--n-particles", type=int, default=5000)
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--n-lags", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="collapse_test.pdf")
    args = p.parse_args()

    names = [s.strip() for s in args.sims.split(",")]
    with open(args.yaml) as fh:
        sims = {s["name"]: s for s in yaml.safe_load(fh)}
    rng = np.random.default_rng(args.seed)

    results = []
    for name in names:
        print(f"Cargando {name} …", flush=True)
        results.append(sim_stats(sims[name], args.n_particles, args.max_steps, args.n_lags, rng))

    print("\nEscalas por simulación:")
    print(f"{'simulación':14s} {'dt':>7s} {'U':>6s} {'L_esp':>6s} {'eps':>7s} " +
          " ".join(f"{c:>8s}" for c in CANDIDATES))
    for r in results:
        print(f"{r['name']:14s} {r['dt']:7.4f} {r['U']:6.3f} {r['L']:6.3f} {r['eps']:7.4f} " +
              " ".join(f"{r['scales'][c]:8.3f}" for c in CANDIDATES))

    ref = results[0]
    metrics = {}   # metrics[other_name][candidate][quantity]
    for other in results[1:]:
        print(f"\nCociente {other['name']} / {ref['name']}:  " +
              "  ".join(f"{c}={other['scales'][c] / ref['scales'][c]:.2f}" for c in CANDIDATES))
        print(f"Colapso {ref['name']} vs {other['name']} (menor = mejor superposición):")
        print(f"{'candidata':10s} " + " ".join(f"{q:>8s}" for q, _ in QUANTITIES) + "   rango t/T común")
        metrics[other["name"]] = {}
        for c in CANDIDATES:
            m, (lo, hi) = collapse_metric(ref, other, c)
            metrics[other["name"]][c] = m
            print(f"{c:10s} " + " ".join(f"{m[q]:8.3f}" for q, _ in QUANTITIES) + f"   [{lo:.3g}, {hi:.3g}]")

    with PdfPages(args.output) as pdf:
        fig, axes = plt.subplots(len(CANDIDATES), len(QUANTITIES), figsize=(16, 14))
        for i, c in enumerate(CANDIDATES):
            for j, (q, title) in enumerate(QUANTITIES):
                ax = axes[i, j]
                for r, color in zip(results, ["C0", "C3", "C2", "C1", "C4", "C5"]):
                    if not np.isfinite(r["scales"][c]):
                        continue
                    x, y = normalised(r, c)
                    ax.plot(x, y[q], color=color, label=r["name"])
                ax.set_xscale("log")
                if q != "vacf":
                    ax.set_yscale("log")
                vals = ", ".join(f"{metrics[o['name']][c][q]:.3f}" for o in results[1:])
                ax.set_title(f"{title}   (T = {c};  métrica: {vals})", fontsize=8)
                ax.set_xlabel(f"t / ({c})")
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"Prueba de colapso: {ref['name']} vs " +
                     ", ".join(o["name"] for o in results[1:]) + " con cada tiempo característico")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()

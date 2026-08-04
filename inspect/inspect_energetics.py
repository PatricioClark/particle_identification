"""Plot energetics diagnostics for a GHOST run.

Reads parameter.inp, balance.txt, and the last few kspectrum*.txt snapshots
and plots E(t), E(k), and derived turbulence scales L, U, T, eta, k_eta,
t_eta.

Usage
-----
python inspect/inspect_energetics.py --path /path/to/run/dir \
    [--n-spectra 5] [--output inspect_energetics.png]

--path should point to the run's base directory (the one containing
parameter.inp, next to the GHOST executable). Text diagnostics
(balance.txt, kspectrum.*.txt) are looked up in the directory named by
parameter.inp's &HD todir, falling back to idir/odir or --path itself.

Definitions
-----------
E(t)      = 0.5 * <v^2>                          (balance.txt col 2)
eps(t)    = 2 * nu * <omega^2>                   (balance.txt col 3)
U(t)      = sqrt(<v^2>)                          rms velocity magnitude
L(t)      = int(E(k)/k dk) / int(E(k) dk)        spectral moment ratio,
                                                  evaluated per kspectrum snapshot
T(t)      = L(t) / U(t)                          eddy turnover time
eta(t)    = (nu^3 / eps)^(1/4)                   Kolmogorov length
k_eta(t)  = 1 / eta(t)                           Kolmogorov wavenumber
t_eta(t)  = (nu / eps)^(1/2)                     Kolmogorov time

E(t), U(t), eta(t), k_eta(t), t_eta(t) are dense (every balance.txt row).
L(t) and T(t) are sparse: one value per kspectrum snapshot among the last
--n-spectra, since they require the spectrum.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_parameter_inp(path):
    """Return dict of scalar values (float where possible) from parameter.inp."""
    vals = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("!")[0].strip()
        m = re.match(r"(\w+)\s*=\s*(.+?)\s*,?\s*$", line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip().rstrip(",").strip('"').strip("'")
        try:
            vals[key] = float(raw)
        except ValueError:
            vals[key] = raw
    return vals


def resolve_text_dir(base_path, params):
    """Locate the directory holding balance.txt / kspectrum*.txt."""
    base = Path(base_path)
    for key in ("todir", "idir", "odir"):
        if key in params:
            cand = base / str(params[key])
            if (cand / "balance.txt").exists():
                return cand
    if (base / "balance.txt").exists():
        return base
    raise FileNotFoundError(
        f"Could not find balance.txt under {base} (checked todir/idir/odir and base path)"
    )


def sorted_kspectrum_files(text_dir):
    return sorted(Path(text_dir).glob("kspectrum.*.txt"),
                  key=lambda p: int(p.stem.split(".")[1]))


def snapshot_time(idx, sstep, dt):
    return (idx - 1) * sstep * dt


def load_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True, help="Run base directory containing parameter.inp")
    p.add_argument("--n-spectra", type=int, default=5,
                   help="Number of most recent kspectrum snapshots to use")
    p.add_argument("--output", default="inspect_energetics.png")
    return p.parse_args()


def main():
    args = load_args()
    base = Path(args.path)

    params = read_parameter_inp(base / "parameter.inp")
    nu, dt, sstep = params["nu"], params["dt"], params["sstep"]

    text_dir = resolve_text_dir(base, params)

    bal = np.loadtxt(text_dir / "balance.txt")
    t, v2, w2 = bal[:, 0], bal[:, 1], bal[:, 2]
    E = 0.5 * v2
    eps = 2.0 * nu * w2
    U = np.sqrt(v2)
    eta = (nu ** 3 / eps) ** 0.25
    k_eta = 1.0 / eta
    t_eta = (nu / eps) ** 0.5

    spec_files = sorted_kspectrum_files(text_dir)[-args.n_spectra:]
    if not spec_files:
        raise FileNotFoundError(f"No kspectrum.*.txt files found in {text_dir}")

    spectra = []
    spec_times, L_list, T_list = [], [], []
    for f in spec_files:
        idx = int(f.stem.split(".")[1])
        ts = snapshot_time(idx, sstep, dt)
        k, Ek = np.loadtxt(f, unpack=True)
        spectra.append((ts, k, Ek))

        L = np.trapezoid(Ek / k, k) / np.trapezoid(Ek, k)
        U_ts = np.interp(ts, t, U)
        spec_times.append(ts)
        L_list.append(L)
        T_list.append(L / U_ts)

    spec_times = np.array(spec_times)
    L_arr = np.array(L_list)
    T_arr = np.array(T_list)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    ax = axes[0, 0]
    ax.plot(t, E)
    ax.set_xlabel("t"); ax.set_ylabel("E(t)"); ax.set_title("Kinetic energy")

    ax = axes[0, 1]
    k_ref = spectra[0][1]
    Ek_avg = np.mean([Ek for _, _, Ek in spectra], axis=0)
    dk = np.mean(np.diff(k_ref))
    k_cutoff = (2.0 / 3.0) * (params["nx"] / 2.0) * dk
    mask = k_ref <= k_cutoff
    ax.loglog(k_ref[mask], Ek_avg[mask])
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.set_title(f"Spectrum (avg of last {len(spectra)}, dealiased modes omitted)")

    ax = axes[0, 2]
    ax.plot(t, U)
    ax.set_xlabel("t"); ax.set_ylabel("U(t)"); ax.set_title("rms velocity")

    ax = axes[0, 3]
    ax.plot(t, eta)
    ax.set_xlabel("t"); ax.set_ylabel(r"$\eta(t)$"); ax.set_title("Kolmogorov length")

    ax = axes[1, 0]
    ax.plot(spec_times, L_arr, "o-")
    ax.set_xlabel("t"); ax.set_ylabel("L(t)")
    ax.set_title(f"Integral scale (last {len(spectra)} snaps)")

    ax = axes[1, 1]
    ax.plot(spec_times, T_arr, "o-")
    ax.set_xlabel("t"); ax.set_ylabel("T(t)"); ax.set_title("Eddy turnover time (L/U)")

    ax = axes[1, 2]
    ax.plot(t, k_eta)
    ax.set_xlabel("t"); ax.set_ylabel(r"$k_\eta(t)$"); ax.set_title("Kolmogorov wavenumber")

    ax = axes[1, 3]
    ax.plot(t, t_eta)
    ax.set_xlabel("t"); ax.set_ylabel(r"$\tau_\eta(t)$"); ax.set_title("Kolmogorov time")

    fig.suptitle(f"Energetics: {base}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")

    print(f"\nLatest values (t={t[-1]:.4f}):")
    print(f"  E      = {E[-1]:.4e}")
    print(f"  eps    = {eps[-1]:.4e}")
    print(f"  U      = {U[-1]:.4e}")
    print(f"  eta    = {eta[-1]:.4e}")
    print(f"  k_eta  = {k_eta[-1]:.4e}")
    print(f"  t_eta  = {t_eta[-1]:.4e}")
    print(f"  L (last spectrum, t={spec_times[-1]:.4f}) = {L_arr[-1]:.4e}")
    print(f"  T (last spectrum) = {T_arr[-1]:.4e}")


if __name__ == "__main__":
    main()

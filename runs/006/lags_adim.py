"""Convert the run-002 lags to turnover-time units and back to steps per simulation.

Check proposed by Patricio: once lags are expressed in units of the large-eddy
turnover time T, converting them back to steps for each TG768 simulation must
recover the original lags [1, 5, 13, 45, 125] (up to 1-2 steps, since T varies
slightly between runs). The same t* lags are then converted for the RND512
simulations.

For each simulation:
  dt  = time between consecutive .lag files (read from the files themselves)
  U   = sqrt(<v^2>), with <v^2> = balance.txt column 2, averaged over the time
        span covered by the particle files
  L   = int(E/k) / int(E), averaged over the kspectrum files in that span
  T   = L / U     (--t-def L/U, default; best in collapse_test.py)
        2*pi / U  (--t-def 2pi/U)

Reference: T_ref = mean T over the accessible TG768 sims, dt_ref = their dt.
  t*_lag = steps_002 * dt_ref / T_ref
  steps_i = round(t*_lag * T_i / dt_i)

Usage
-----
python lags_adim.py [--yaml simuls.yaml] [--t-def L/U] [--lags 1,5,13,45,125] [--window 251]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import yaml


def lag_files(sim_path, quantity="xlg"):
    p = Path(sim_path)
    d = p / "outs" if (p / "outs").is_dir() else p
    return sorted(d.glob(f"{quantity}.*.lag"), key=lambda f: int(f.stem.split(".")[-1]))


def last_nonempty(files):
    """Last .lag file with data (the final one is sometimes empty, as dataset.py notes)."""
    for f in reversed(files):
        if f.stat().st_size > 0:
            return f
    return None


def lag_time(path):
    return float(np.fromfile(path, dtype=np.float32, count=2)[1])


def find_balance(sim_path):
    """balance.txt next to the sim, or in the run's todir/ (RND512 layout)."""
    p = Path(sim_path)
    for cand in (p / "balance.txt", p.parent / "todir" / "balance.txt"):
        if cand.exists():
            return cand
    return None


def find_file(sim_path, name):
    """File next to the sim, or in the run dir (RND512 layout)."""
    p = Path(sim_path)
    for cand in (p / name, p.parent / name):
        if cand.exists():
            return cand
    return None


def read_params(par_path):
    vals = {}
    for line in Path(par_path).read_text().splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*([0-9.eE+-]+)", line.split("!")[0])
        if m:
            try:
                vals[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return vals


def spectral_L(sim_path, bal_path, t0, t_end, max_spectra=20):
    """Mean integral scale over up to max_spectra spectra (evenly spaced) whose time is in [t0, t_end].

    Spectrum times come from the file index, (idx - 1) * sstep * dt, so no file
    has to be stat'ed (slow on the shared disk). Empty files are skipped on read.
    """
    par = read_params(find_file(sim_path, "parameter.inp"))
    snaps = []
    for f in bal_path.parent.glob("kspectrum.*.txt"):
        t_snap = (int(f.stem.split(".")[1]) - 1) * par["sstep"] * par["dt"]
        if t0 <= t_snap <= t_end:
            snaps.append((t_snap, f))
    snaps.sort()
    if len(snaps) > max_spectra:
        idx = np.round(np.linspace(0, len(snaps) - 1, max_spectra)).astype(int)
        snaps = [snaps[i] for i in idx]
    Ls = []
    for _, f in snaps:
        data = np.loadtxt(f, ndmin=2)
        if data.size == 0:
            continue
        k, E = data[:, 0], data[:, 1]
        m = k > 0
        Ls.append(np.trapezoid(E[m] / k[m], k[m]) / np.trapezoid(E[m], k[m]))
    return float(np.mean(Ls)), len(Ls)


def sim_scales(sim_path, t_def):
    files = lag_files(sim_path)
    if len(files) < 2:
        return None
    t0, t1, t_end = lag_time(files[0]), lag_time(files[1]), lag_time(last_nonempty(files))
    dt = t1 - t0

    bal_path = find_balance(sim_path)
    if bal_path is None:
        return None
    bal = np.loadtxt(bal_path)
    in_span = (bal[:, 0] >= t0) & (bal[:, 0] <= t_end)
    U = float(np.sqrt(bal[in_span, 1].mean()))
    L, n_spec = spectral_L(sim_path, bal_path, t0, t_end) if t_def == "L/U" else (2 * np.pi, 0)
    return {"dt": dt, "U": U, "L": L, "n_spec": n_spec, "T": L / U, "n_steps": len(files),
            "t_span": (t0, t_end), "balance": bal_path}


def to_steps(t_star, T, dt):
    return np.maximum(1, np.round(np.asarray(t_star) * T / dt).astype(int))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default="simuls.yaml")
    p.add_argument("--t-def", choices=["L/U", "2pi/U"], default="L/U",
                   help="Turnover-time definition (default L/U, best in collapse_test.py)")
    p.add_argument("--lags", default="1,5,13,45,125", help="Run-002 lags in steps")
    p.add_argument("--window", type=int, default=251, help="Run-002 window in steps")
    args = p.parse_args()

    lags_002 = np.array([int(x) for x in args.lags.split(",")])

    with open(args.yaml) as fh:
        sims = yaml.safe_load(fh)

    rows = []
    for sim in sims:
        if not Path(sim["path"]).is_dir():
            print(f"  [skip] {sim['name']}: no existe {sim['path']}")
            continue
        print(f"  procesando {sim['name']} …", flush=True)
        sc = sim_scales(sim["path"], args.t_def)
        if sc is None:
            print(f"  [skip] {sim['name']}: faltan .lag o balance.txt")
            continue
        rows.append((sim, sc))

    tg = [sc for sim, sc in rows if sim["flux"]["resolution"][0] == 768]
    T_ref = float(np.mean([sc["T"] for sc in tg]))
    dt_ref = float(np.mean([sc["dt"] for sc in tg]))
    t_star = lags_002 * dt_ref / T_ref
    w_star = (args.window - 1) * dt_ref / T_ref

    print(f"\nTiempo de giro: T = {args.t_def}")
    print(f"Referencia TG768: T_ref = {T_ref:.3f}, dt_ref = {dt_ref:.4f}")
    print(f"Lags 002 en pasos: {lags_002.tolist()}")
    print(f"Lags 002 en t* = t/T: {np.round(t_star, 4).tolist()}")
    print(f"Ventana 002: {args.window} pasos = {w_star:.3f} T\n")

    header = (f"{'simulación':18s} {'dt':>7s} {'U':>6s} {'L':>6s} {'T':>6s} {'pasos':>6s}  "
              f"{'lags en pasos':28s} {'ventana':>7s}")
    print(header)
    print("-" * len(header))
    current_res = None
    all_ok = True
    for sim, sc in rows:
        res = sim["flux"]["resolution"][0]
        if res != current_res and current_res is not None:
            print()
        current_res = res
        steps = to_steps(t_star, sc["T"], sc["dt"])
        window = int(to_steps(w_star, sc["T"], sc["dt"])) + 1
        flag = ""
        if res == 768:
            ok = np.all(np.abs(steps - lags_002) <= np.maximum(2, 0.03 * lags_002))
            all_ok &= ok
            flag = "ok" if ok else "REVISAR"
        if window > sc["n_steps"]:
            flag += " VENTANA > TRAYECTORIA"
        print(f"{sim['name']:18s} {sc['dt']:7.4f} {sc['U']:6.3f} {sc['L']:6.3f} {sc['T']:6.3f} "
              f"{sc['n_steps']:6d}  {str(steps.tolist()):28s} {window:7d}  {flag}")

    print(f"\nVerificación TG768: {'todas recuperan los lags de 002' if all_ok else 'HAY SIMULACIONES A REVISAR'}")


if __name__ == "__main__":
    main()

"""Check the lag interpolation of features.py with real trajectories.

(a) Integer lags: features must be identical to run 006's features.py.
(b) Subsampling test: keep one output out of s (dt' = s*dt), so a lag of L
    original steps becomes L/s, fractional, in the subsampled series. Compare,
    against the exact value from the full series at lag L:
      - interp: interpolated from the subsampled series (what 007 does)
      - round : value at the rounded lag round(L/s) of the subsampled series
                (what 006 does when a t* lag falls between outputs)
    Done for a tracer, an intermediate and a heavy TG768 particle.

Usage
-----
python verify_interp.py [--yaml simuls.yaml] [--n-particles 2000] [--n-steps 400]
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import yaml

from dataset import load_simulation, probe_sim
import features as f7

SIMS = ["TG768 LAG", "TG768 MR 3.2", "TG768 ONLD 312"]
TESTS = {2: [3, 5, 13, 45, 125], 3: [4, 5, 7, 14, 46, 124]}   # stride -> target lags (orig. steps)


def load_006_features():
    spec = importlib.util.spec_from_file_location("features006", Path(__file__).parent / "../006/features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stats(pos_u, v, lags):
    """VACF, MSD, S2, S4 at (possibly fractional) lags, through features.py's interpolation."""
    il, lo, hi, w = f7._interp_lags(lags)
    return {
        "vacf": f7._combine(f7._vacf(v, il), lags, il, lo, hi, w, log=False),
        "msd": f7._combine(f7._msd(pos_u, il), lags, il, lo, hi, w, log=True),
        "s2": f7._combine(f7._sf(v, il, 2), lags, il, lo, hi, w, log=True),
        "s4": f7._combine(f7._sf(v, il, 4), lags, il, lo, hi, w, log=True),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml", default="simuls.yaml")
    p.add_argument("--n-particles", type=int, default=2000)
    p.add_argument("--n-steps", type=int, default=400)
    args = p.parse_args()

    sims = {s["name"]: s for s in yaml.safe_load(open(args.yaml))}
    f6 = load_006_features()
    rng = np.random.default_rng(0)

    for name in SIMS:
        path = sims[name]["path"]
        n_total, _ = probe_sim(path)
        idxs = np.sort(rng.choice(n_total, size=args.n_particles, replace=False))
        times, pos, vel = load_simulation(path, idxs, max_steps=args.n_steps)
        dt = float(np.diff(times).mean())
        print(f"\n=== {name}: {len(idxs)} partículas, {pos.shape[2]} pasos, dt = {dt:.4f}")

        # (a) integer lags: identical to 006
        lags = np.array([1, 5, 13, 45, 125])
        x6, _ = f6.extract_features(pos, vel, dt, lags, scales={"U": 1.3, "L": 0.33, "T": 0.24})
        x7, _ = f7.extract_features(pos, vel, dt, lags, scales={"U": 1.3, "L": 0.33, "T": 0.24})
        print(f"(a) lags enteros, 007 vs 006: idénticos = {np.array_equal(x6, x7)}")

        # (b) subsampling
        pos_u = f7.unwrap_positions(pos)
        v = vel.astype(np.float64)
        for s, targets in TESTS.items():
            targets = np.array(targets)
            exact = stats(pos_u, v, targets.astype(float))
            sub_p, sub_v = pos_u[:, :, ::s], v[:, :, ::s]
            interp = stats(sub_p, sub_v, targets / s)
            rnd = stats(sub_p, sub_v, np.maximum(1, np.round(targets / s)).astype(float))
            print(f"(b) submuestreo cada {s}: lag (pasos originales → submuestreados), "
                  f"error relativo % [interpolado | redondeado]")
            for q in ("vacf", "msd", "s2", "s4"):
                cells = "  ".join(
                    f"{L}→{L / s:.2f}: {100 * (interp[q][k] / exact[q][k] - 1):+6.2f} | "
                    f"{100 * (rnd[q][k] / exact[q][k] - 1):+6.1f}"
                    for k, L in enumerate(targets))
                print(f"    {q:4s}  {cells}")


if __name__ == "__main__":
    main()

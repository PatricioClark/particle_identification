"""How nonlinear is the drag of each ONLD species in each 512 run? (test of the ONLD 1 hypothesis)

GHOST's nonlinear drag (Wang & Maxey 1993, inerpart_solver.f90) multiplies the Stokes drag by

    c = 1 + 0.15 Re_p^0.687 = 1 + rep |u - v_p|^0.687,   rep = 0.15 (18 tau gamma / nu)^0.3435,

with u the fluid velocity at the particle (vlg) and v_p the particle velocity (vip). So the particle
responds as with an effective response time tau / c, i.e. an effective Stokes St_real / <c>, where
St_real = tau / tau_eta and tau_eta = 1 / sqrt(<w^2>) (eps = nu <w^2>, balance.txt column 3).

Hypothesis: the classifier trained on the paired runs calls the ONLD 1 of the other runs "HPP 0.35-0.6"
because there the flow is less intense, the slip is smaller, <c> is closer to 1 and the particle is less
"nonlinear". For each run (bespanol originals, the new run with seed 2207, the paired runs) and ONLD
species this prints the flow energy <v^2>, the slip |u - v_p| (also over U), <c>, St_real and the
effective St_real / <c>; for comparison, St_real of the HPP of the paired runs.

Only a contiguous slice of --n-parts particles of each file is read (three seeks per file).

Usage
-----
python slip_onld.py [--n-files 40] [--n-parts 20000]
"""

import argparse
from pathlib import Path

import numpy as np

NU, GAMMA, TAU_ETA_LABEL = 7e-4, 1.0, 0.09416
BES = "/share/data4/bespanol/HIT512_multipart/bin"
NUE = "/share/data4/msvidal/RND512_nuevas"
PAR = "/share/data4/msvidal/RND512_pares"

# run name -> (balance.txt, first file, last file, {St label: species dir})
RUNS = {
    "bespanol": (f"{BES}/todir/balance.txt", 300, 3035,
                 {"0.1": f"{BES}/outs_NLD_0-1", "1": f"{BES}/outs_NLD_1", "20": f"{BES}/outs_NLD_20",
                  "100": f"{BES}/outs_NLD_100"}),
    "nueva": (f"{NUE}/todir/balance.txt", 3077, 3390,
              {"0.1": f"{NUE}/outs_NLD_0-1", "1": f"{NUE}/outs_NLD_1", "20": f"{NUE}/outs_NLD_20",
               "100": f"{NUE}/outs_NLD_100"}),
    "pareadas": (f"{PAR}/A/todir/balance.txt", 3427, 3926,
                 {st: f"{PAR}/{'A' if float(st) < 1 else 'B'}/outs_ONLD_{st.replace('.', '-')}"
                  for st in ["0.1", "0.2", "0.35", "0.6", "1", "1.4", "2", "4", "20"]}),
}


def dedup_balance(path):
    """balance.txt with, for each time, the last row written (rows of restarted jobs repeat times)."""
    bal = np.loadtxt(path)
    _, last = np.unique(bal[::-1, 0], return_index=True)
    return bal[::-1][last]


def read_slice(path, n):
    """(time, (3, n) array) of the first n particles of a .lag file."""
    with open(path, "rb") as f:
        head = np.fromfile(f, dtype=np.float32, count=2)
        N = int(head[0])
        out = np.empty((3, n), dtype=np.float64)
        for c in range(3):
            f.seek((2 + c * N) * 4)
            out[c] = np.fromfile(f, dtype=np.float32, count=n)
    return float(head[1]), out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-files", type=int, default=40, help="Outputs per run, evenly spread")
    p.add_argument("--n-parts", type=int, default=20000)
    args = p.parse_args()

    print(f"{'corrida':9s} {'St etiq':>7s} {'<v2> flujo':>10s} {'|slip|':>7s} {'|slip|/U':>8s} "
          f"{'<c>':>6s} {'c p90':>6s} {'St real':>8s} {'St real/<c>':>11s}")
    for run, (bal_path, f0, f1, species) in RUNS.items():
        bal = dedup_balance(bal_path)
        idx = np.unique(np.linspace(f0, f1, args.n_files).astype(int))
        for st, d in species.items():
            tau = float(st) * TAU_ETA_LABEL
            rep = 0.15 * (18 * tau * GAMMA / NU) ** 0.3435
            slip, cs, v2, teta = [], [], [], []
            for i in idx:
                fv, fu = Path(d) / f"vip.{i:08d}.lag", Path(d) / f"vlg.{i:08d}.lag"
                if not (fv.exists() and fu.exists()):
                    continue
                t, vp = read_slice(fv, args.n_parts)
                _, u = read_slice(fu, args.n_parts)
                s = np.sqrt(((u - vp) ** 2).sum(axis=0))
                slip.append(s.mean())
                cs.append(1 + rep * s ** 0.687)
                j = np.argmin(np.abs(bal[:, 0] - t))
                v2.append(bal[j, 1])
                teta.append(1 / np.sqrt(bal[j, 2]))
            c_all = np.concatenate(cs)
            U = np.sqrt(np.mean(v2))
            st_real = tau / np.mean(teta)
            print(f"{run:9s} {st:>7s} {np.mean(v2):10.2f} {np.mean(slip):7.3f} {np.mean(slip) / U:8.3f} "
                  f"{c_all.mean():6.2f} {np.percentile(c_all, 90):6.2f} {st_real:8.2f} {st_real / c_all.mean():11.2f}"
                  f"   ({len(cs)} salidas)", flush=True)
        if run == "pareadas":
            teta_p = np.mean([1 / np.sqrt(bal[np.argmin(np.abs(bal[:, 0] - t)), 2])
                              for t in np.linspace(102.8, 117.75, 50)])
            print("   HPP de las pareadas (c = 1): St real " + ", ".join(
                f"{st} → {float(st) * TAU_ETA_LABEL / teta_p:.2f}"
                for st in ["0.1", "0.2", "0.35", "0.6", "1", "1.4", "2", "4", "20"]))


if __name__ == "__main__":
    main()

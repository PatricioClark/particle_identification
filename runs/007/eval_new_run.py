"""Evaluate a trained classifier on the new RND512 run (other forcing seed + ONLD St~3.2).

Also used as a control on stretches of the ORIGINAL run (--run-dir of bespanol's
HIT512_multipart/bin, --skip/--n-out selecting the stretch): same procedure, same
length, so a failure there points to the flow's variability in time and not to
the new seed.

The new run (/share/data4/msvidal/RND512_nuevas) restarts bespanol's RND512
flow with another forcing seed. Its 8 original species should be recognised
as their RND512 classes if the classifier looks at the particle physics and
not at the particular simulation; the ONLD St~3.2 species has no class of
its own in RND512 and is compared with ONLD_St3.2_TG768.

Samples are built exactly as in the training dataset (pool of --n-load
particles, batches of --batch-size, per-particle random time offsets,
lags and window in turnover times, normalised features), except that:
  - the first --skip outputs are discarded (the new flow starts identical to
    the old one and drifts away; ONLD 3.2 starts from the HPP 4 state);
  - the flow scales (U, L, T) are, with --scales own, measured on the new run
    itself (balance.txt and spectra over the analysed span), as one would do
    with data of unknown origin; with --scales train, the training values of
    RND512 are used. If the window in steps does not fit in the remaining
    trajectory, the whole remaining trajectory is used.

Step 1, features (MPI, one species per rank, round-robin):
    mpirun -n 9 python eval_new_run.py features --scales own --out-dir new_run/own
Step 2, predictions (light):
    python eval_new_run.py predict --model model.pkl --out-dir new_run/own
"""

import argparse
import json
from pathlib import Path

import numpy as np

RUN_DIR = "/share/data4/msvidal/RND512_nuevas"
# species dir -> class it should be recognised as (None: no class of its own in RND512)
SPECIES = {
    "outs_HPP_0-1": "HPP_St0.1_RND512",
    "outs_HPP_0-6": "HPP_St0.6_RND512",
    "outs_HPP_1": "HPP_St1_RND512",
    "outs_HPP_4": "HPP_St4_RND512",
    "outs_NLD_0-1": "ONLD_St0.1_RND512",
    "outs_NLD_1": "ONLD_St1_RND512",
    "outs_NLD_20": "ONLD_St20_RND512",
    "outs_NLD_100": "ONLD_St100_RND512",
    "outs_ONLD_3-2": None,
}


def n_use(run_dir, skip, n_out):
    """Outputs analysed: n_out after `skip`, or all but the last one (GHOST may have been
    killed by the time limit while writing it)."""
    from dataset import probe_sim
    available = probe_sim(Path(run_dir) / "outs_HPP_0-1")[1] - skip - 1
    return available if n_out is None else min(n_out, available)


def run_scales(run_dir, skip, n):
    """U, L, T and dt of a run over the analysed span (n outputs after `skip`)."""
    from dataset import _glob_sorted, _read_lag
    from lags_adim import find_balance, spectral_L

    sim_path = Path(run_dir) / "outs_HPP_0-1"
    files = _glob_sorted(sim_path, "xlg")[skip:skip + n]
    t0, t1 = _read_lag(files[0])[0], _read_lag(files[1])[0]
    t_end = _read_lag(files[-1])[0]
    bal_path = find_balance(sim_path)
    bal = np.loadtxt(bal_path)
    span = (bal[:, 0] >= t0) & (bal[:, 0] <= t_end)
    U = float(np.sqrt(bal[span, 1].mean()))
    L, n_spec = spectral_L(sim_path, bal_path, t0, t_end)
    return {"U": U, "L": L, "T": L / U, "dt": t1 - t0, "t_span": [t0, t_end],
            "n_steps": len(files), "n_spec": n_spec}


def cmd_features(args):
    from mpi4py import MPI
    from dataset import make_feature_matrix, probe_sim

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    fpar = json.loads(Path(args.flow_params).read_text())
    lags_tstar, window_tstar = np.array(fpar["lags_tstar"]), fpar["window_tstar"]
    interp = fpar.get("interp_lags", False)

    if rank == 0:
        n = n_use(args.run_dir, args.skip, args.n_out)
        if args.scales == "own":
            sc = run_scales(args.run_dir, args.skip, n)
        else:
            tr = fpar["flows"]["RND512"]
            sc = {k: tr[k] for k in ("U", "L", "T", "dt")}
            sc["n_steps"] = n
        lags = lags_tstar * sc["T"] / sc["dt"]
        if not interp:
            lags = np.maximum(1, np.round(lags).astype(int))
        window = int(np.round(window_tstar * sc["T"] / sc["dt"])) + 1
        sc.update({"lags": [float(x) for x in lags], "window": window, "skip": args.skip,
                   "run_dir": args.run_dir,
                   "scales": args.scales, "interp_lags": interp})
        print(f"Escalas ({args.scales}): U={sc['U']:.4f} L={sc['L']:.4f} T={sc['T']:.4f} "
              f"dt={sc['dt']:.4f}; lags={np.round(lags, 3).tolist()}; ventana={window} pasos; "
              f"disponibles tras descartar {args.skip}: {sc['n_steps']}", flush=True)
        if window >= sc["n_steps"]:
            print(f"[aviso] la ventana ({window}) no entra en la trayectoria ({sc['n_steps']}): "
                  "se usa la trayectoria entera, sin desplazamientos al azar", flush=True)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        Path(args.out_dir, "scales.json").write_text(json.dumps(sc, indent=2))
    else:
        sc = None
    sc = comm.bcast(sc, root=0)
    lags = np.array(sc["lags"]) if interp else np.array(sc["lags"]).astype(int)
    scales = {"U": sc["U"], "L": sc["L"], "T": sc["T"]}

    for k, sp in list(enumerate(SPECIES))[rank::size]:
        out = Path(args.out_dir) / f"{sp}.npz"
        if not (Path(args.run_dir) / sp).is_dir():
            continue  # outs_ONLD_3-2 only exists in the new run
        if out.exists():
            print(f"  [rank {rank}] {sp}: ya existe, lo salteo", flush=True)
            continue
        print(f"  [rank {rank}] {sp} …", flush=True)
        seed = int(np.random.SeedSequence([args.seed, k]).generate_state(1)[0])
        X, names = make_feature_matrix(
            str(Path(args.run_dir) / sp), lags, batch_size=args.batch_size,
            n_batches=args.n_batches, n_load=args.n_load, window_size=sc["window"],
            seed=seed, scales=scales, skip=args.skip, max_steps=sc["n_steps"])
        np.savez(out, X=X, names=np.array(names))
        print(f"  [rank {rank}] {sp}: {X.shape[0]} muestras → {out}", flush=True)


def cmd_predict(args):
    import joblib

    model = joblib.load(args.model)
    pipe, labels = model["pipeline"], list(model["label_names"])
    sc = json.loads(Path(args.out_dir, "scales.json").read_text())
    span = f", t = {sc['t_span'][0]:.2f}–{sc['t_span'][1]:.2f}" if "t_span" in sc else ""
    print(f"Modelo: {args.model} ({len(labels)} clases). Muestras: {args.out_dir} "
          f"(escalas {sc['scales']}, U={sc['U']:.3f}, T={sc['T']:.4f}, descartadas {sc['skip']} "
          f"salidas, {sc['n_steps']} usadas{span})\n")

    hits = []
    for sp, expected in SPECIES.items():
        f = Path(args.out_dir) / f"{sp}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        if list(d["names"]) != list(model["feature_names"]):
            raise RuntimeError(f"{sp}: features distintas de las del modelo")
        pred = pipe.predict(d["X"])
        names, counts = np.unique([labels[i] for i in pred], return_counts=True)
        order = np.argsort(-counts)
        top = ", ".join(f"{names[i]} {counts[i]}" for i in order[:4])
        if expected is None:
            print(f"{sp:15s} (sin clase propia)  → {top}")
        else:
            ok = float(np.mean([labels[i] == expected for i in pred]))
            hits.append(ok)
            print(f"{sp:15s} esperado {expected:18s} acierto {ok:5.2f}  → {top}")
    if hits:
        print(f"\nAcierto medio en las 8 especies con clase: {np.mean(hits):.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--run-dir", default=RUN_DIR)
    f.add_argument("--flow-params", default="flow_params_007.json",
                   help="Training flow parameters: lags/window in t*, interp_lags, RND512 scales")
    f.add_argument("--scales", choices=["own", "train"], default="own")
    f.add_argument("--skip", type=int, default=50, help="Outputs discarded at the start")
    f.add_argument("--n-out", type=int, default=None,
                   help="Outputs analysed after --skip (default: all but the last one)")
    f.add_argument("--n-load", type=int, default=50000)
    f.add_argument("--batch-size", type=int, default=5000)
    f.add_argument("--n-batches", type=int, default=100)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--out-dir", required=True)
    q = sub.add_parser("predict")
    q.add_argument("--model", required=True)
    q.add_argument("--out-dir", required=True)
    args = p.parse_args()
    cmd_features(args) if args.cmd == "features" else cmd_predict(args)


if __name__ == "__main__":
    main()

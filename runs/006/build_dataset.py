"""Build and save a labelled feature dataset from GHOST simulation directories.

Usage
-----
mpirun -n N python build_dataset.py [options]

Work is split into *shards*. A shard is one (simulation, shard-index) unit that
loads a particle pool once and emits ``--batches-per-shard`` feature rows. Each
shard writes its own checkpoint file as soon as it finishes, so a run that hits
the cluster wall-time can be resubmitted and will skip every shard already on
disk. When all shards exist, rank 0 assembles them into the final dataset .pkl.

    # fresh run (or resume — finished shards are skipped automatically)
    mpirun -n N python build_dataset.py --checkpoint-dir ckpt --output dataset.pkl

    # only stitch existing checkpoints into the dataset, no recomputation
    python build_dataset.py --checkpoint-dir ckpt --output dataset.pkl --assemble-only

Total samples per simulation = --n-shards-per-sim × --batches-per-shard.

Turnover-time mode (run 006)
----------------------------
With --lags-tstar, lags and window are given in units of the large-eddy
turnover time T = L/U instead of steps. Simulations are grouped by flow
(forcing + resolution); each flow gets one T (mean over its simulations, see
lags_adim.py), and its own lags and window in steps:

    lags_flow   = round(lags_tstar * T_flow / dt_flow)
    window_flow = round(window_tstar * T_flow / dt_flow) + 1

Unless --no-normalize, dimensional features are also divided by the flow
scales (see features.extract_features). Flow parameters are stored in the
dataset under "flow_params".

The flow scales are slow to read (spectra on the shared disk), so they are
computed once, by a single process, and saved to --flow-params; the MPI run
then only reads that file:

    # 1) once, single process: compute and save the flow parameters
    python build_dataset.py --lags-tstar 0.0668,0.3339,0.868,3.0047,8.3464 \
        --window-tstar 16.693 --flow-params flow_params.json --flow-params-only
    # 2) the MPI run reads them
    mpirun -n N python build_dataset.py --lags-tstar 0.0668,0.3339,0.868,3.0047,8.3464 \
        --window-tstar 16.693 --flow-params flow_params.json \
        --checkpoint-dir ckpt --output dataset.pkl
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import yaml
from mpi4py import MPI
from pathlib import Path

from dataset import probe_sim, make_feature_matrix
from features import make_lags
from lags_adim import sim_scales

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def shard_label(sim):
    forcing = sim["flux"]["forcing"]
    res     = sim["flux"]["resolution"][0]
    return f"{sim['parts']['model']}_St{sim['parts']['st']}_{forcing}{res}"


def flow_key(sim):
    return f"{sim['flux']['forcing']}{sim['flux']['resolution'][0]}"


def flow_parameters(accessible, lags_tstar, window_tstar, t_def):
    """One T per flow (mean over its sims) and the flow's lags/window in steps."""
    per_flow = {}
    for _, sim in accessible:
        sc = sim_scales(sim["path"], t_def)
        if sc is None:
            raise RuntimeError(f"No scales for {sim['name']} (missing .lag or balance.txt)")
        per_flow.setdefault(flow_key(sim), []).append((sim["name"], sc))

    params = {}
    for flow, items in per_flow.items():
        dts = np.array([sc["dt"] for _, sc in items])
        if np.ptp(dts) > 1e-6 * dts.mean():
            raise RuntimeError(f"Flow {flow}: simulations with different dt {dts}")
        dt = float(dts.mean())
        T = float(np.mean([sc["T"] for _, sc in items]))
        U = float(np.mean([sc["U"] for _, sc in items]))
        L = float(np.mean([sc["L"] for _, sc in items]))
        lags = np.maximum(1, np.round(np.asarray(lags_tstar) * T / dt).astype(int))
        if np.any(np.diff(lags) <= 0):
            raise RuntimeError(f"Flow {flow}: lags {lags} are not strictly increasing")
        window = int(np.round(window_tstar * T / dt)) + 1
        min_steps = min(sc["n_steps"] for _, sc in items)
        if window >= min_steps:
            raise RuntimeError(f"Flow {flow}: window {window} >= shortest trajectory {min_steps}")
        params[flow] = {"T": T, "U": U, "L": L, "dt": dt, "lags": lags, "window": window,
                        "n_sims": len(items), "sims": [n for n, _ in items]}
    return params


def save_flow_params(path, flows, lags_tstar, window_tstar, t_def):
    out = {"lags_tstar": lags_tstar, "window_tstar": window_tstar, "t_def": t_def,
           "flows": {f: {**fp, "lags": [int(x) for x in fp["lags"]]} for f, fp in flows.items()}}
    Path(path).write_text(json.dumps(out, indent=2))


def load_flow_params(path, lags_tstar, window_tstar, t_def, accessible):
    """Read precomputed flow parameters and check they match this run."""
    d = json.loads(Path(path).read_text())
    if (d["lags_tstar"] != lags_tstar or d["window_tstar"] != window_tstar
            or d["t_def"] != t_def):
        raise RuntimeError(f"{path} was computed with lags_tstar={d['lags_tstar']}, "
                           f"window_tstar={d['window_tstar']}, t_def={d['t_def']}; "
                           "recompute it with --flow-params-only")
    names = sorted(sim["name"] for _, sim in accessible)
    saved = sorted(n for fp in d["flows"].values() for n in fp["sims"])
    if names != saved:
        raise RuntimeError(f"{path} was computed for other simulations; "
                           "recompute it with --flow-params-only")
    return {f: {**fp, "lags": np.array(fp["lags"])} for f, fp in d["flows"].items()}


def checkpoint_path(ckpt_dir, sim, yaml_i, shard_j):
    """Unique, deterministic checkpoint name. yaml_i disambiguates distinct
    sims that happen to share a (model, St) label."""
    return ckpt_dir / f"{shard_label(sim)}__sim{yaml_i:03d}__shard{shard_j:03d}.pkl"


def assemble(ckpt_dir, label_map, output):
    """Stitch every checkpoint in ckpt_dir into the final dataset .pkl."""
    ckpts = sorted(ckpt_dir.glob("*.pkl"))
    if not ckpts:
        print("No checkpoints to assemble.", file=sys.stderr)
        sys.exit(1)

    X_blocks, y_blocks = [], []
    feature_names = None
    n_lags = batch_size = None
    windows, flow_params, normalized = set(), {}, set()
    for c in ckpts:
        d = joblib.load(c)
        X_blocks.append(d["X"])
        y_blocks.append(np.full(len(d["X"]), label_map[d["label"]], dtype=int))
        feature_names = feature_names or d["feature_names"]
        n_lags      = d["n_lags"]
        batch_size  = d["batch_size"]
        windows.add(d["window_size"])
        normalized.add(d.get("normalized", False))
        if "flow" in d:
            flow_params[d["flow"]] = d["flow_params"]
    if len(normalized) > 1:
        print("WARNING: checkpoints mix normalized and raw features", file=sys.stderr)
    # A single window means the dataset is compatible with the old pipeline
    window_size = windows.pop() if len(windows) == 1 else None

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(label_names)} classes (from {len(ckpts)} shards)")
    for i, name in enumerate(label_names):
        print(f"  {i:2d}  {name}  ({(y == i).sum()} samples)")

    # Impute NaNs with column medians
    col_medians = np.nanmedian(X, axis=0)
    nan_locs    = np.isnan(X)
    X[nan_locs] = col_medians[np.where(nan_locs)[1]]

    joblib.dump(
        {
            "X":             X,
            "y":             y,
            "feature_names": feature_names,
            "label_names":   label_names,
            "n_lags":        n_lags,
            "batch_size":    batch_size,
            "window_size":   window_size,
            "flow_params":   flow_params,
            "normalized":    normalized.pop() if len(normalized) == 1 else None,
        },
        output,
    )
    print(f"Dataset saved → {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml",             default="simuls.yaml")
    p.add_argument("--batch-size",       type=int, default=5000)
    p.add_argument("--batches-per-shard", type=int, default=100,
                   help="Feature rows emitted per shard (one pool load).")
    p.add_argument("--n-shards-per-sim", type=int, default=1,
                   help="Shards per simulation. >1 splits a sim across ranks "
                        "(extra trajectory reads — only worth it when feature "
                        "compute, not I/O, is the wall-time bottleneck).")
    p.add_argument("--n-load",           type=int, default=50000,
                   help="Particle pool size loaded once per shard. Keep it "
                        "much larger than --batch-size so batch overlap stays small.")
    p.add_argument("--n-lags",           type=int, default=15)
    p.add_argument("--max-steps",        type=int, default=None,
                   help="Truncate trajectories to this many steps. "
                        "Defaults to the shortest trajectory across all sims.")
    p.add_argument("--no-time-augment",  dest="time_augment",
                   action="store_false", default=True,
                   help="Disable per-particle random time offsets (on by default).")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--checkpoint-dir",   default="checkpoints",
                   help="Directory of per-shard checkpoint files (enables resume).")
    p.add_argument("--assemble-only",    action="store_true",
                   help="Skip computation; stitch existing checkpoints into --output.")
    p.add_argument("--output",           default="dataset.pkl")
    p.add_argument("--lags-tstar",       default=None,
                   help="Comma-separated lags in units of T = L/U (turnover-time mode). "
                        "Overrides --n-lags/--max-steps.")
    p.add_argument("--window-tstar",     type=float, default=None,
                   help="Window length in units of T (required with --lags-tstar).")
    p.add_argument("--t-def",            choices=["L/U", "2pi/U"], default="L/U")
    p.add_argument("--no-normalize",     dest="normalize", action="store_false", default=True,
                   help="Turnover-time mode: keep dimensional features as they are.")
    p.add_argument("--flow-params",      default="flow_params.json",
                   help="Turnover-time mode: file with the precomputed flow parameters.")
    p.add_argument("--flow-params-only", action="store_true",
                   help="Compute --flow-params (single process) and exit.")
    args = p.parse_args()
    if args.lags_tstar and args.window_tstar is None:
        p.error("--lags-tstar requires --window-tstar")
    if args.flow_params_only and (not args.lags_tstar or size > 1):
        p.error("--flow-params-only needs --lags-tstar and a single process")

    with open(args.yaml) as fh:
        sims = yaml.safe_load(fh)

    # All ranks: filter to accessible sims and compute shared parameters
    accessible = [
        (i, s) for i, s in enumerate(sims)
        if Path(s["path"]).is_dir() and probe_sim(s["path"])[1] > 0
    ]

    if not accessible:
        if rank == 0:
            print("No accessible simulation directories found.", file=sys.stderr)
        sys.exit(1)

    if args.lags_tstar:
        # Turnover-time mode: rank 0 reads the flow scales (slow shared disk), then broadcasts
        lags_tstar = [float(x) for x in args.lags_tstar.split(",")]
        if args.flow_params_only:
            flows = flow_parameters(accessible, lags_tstar, args.window_tstar, args.t_def)
            save_flow_params(args.flow_params, flows, lags_tstar, args.window_tstar, args.t_def)
            for flow, fp in flows.items():
                print(f"  {flow}: {fp['n_sims']} sims, T={fp['T']:.4f}, lags={fp['lags'].tolist()}, "
                      f"window={fp['window']}")
            print(f"Flow parameters saved → {args.flow_params}")
            return
        if not Path(args.flow_params).exists():
            if rank == 0:
                print(f"Missing {args.flow_params}: compute it first with a single process and "
                      "--flow-params-only (see the docstring).", file=sys.stderr)
            sys.exit(1)
        flows = load_flow_params(args.flow_params, lags_tstar, args.window_tstar,
                                 args.t_def, accessible)
        n_lags = len(lags_tstar)
    else:
        # min_steps and lags — deterministic, same on all ranks
        min_steps = args.max_steps or min(probe_sim(s["path"])[1] for _, s in accessible)
        effective_steps = min_steps - 1
        lags = make_lags(effective_steps, args.n_lags)
        flows = None
        n_lags = args.n_lags

    # label_map — built deterministically from accessible sims in YAML order
    label_map: dict[str, int] = {}
    for _, sim in accessible:
        label = shard_label(sim)
        if label not in label_map:
            label_map[label] = len(label_map)

    ckpt_dir = Path(args.checkpoint_dir)

    # ── Assemble-only: rank 0 stitches and exits ──────────────────────────────
    if args.assemble_only:
        if rank == 0:
            assemble(ckpt_dir, label_map, args.output)
        return

    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Accessible simulations: {len(accessible)}, MPI ranks: {size}")
        if flows is None:
            print(f"window_size={min_steps}, effective_steps={effective_steps}")
        print(f"Time augmentation: {'on' if args.time_augment else 'off'}")
        print(f"Shards/sim: {args.n_shards_per_sim}, batches/shard: "
              f"{args.batches_per_shard}, pool: {args.n_load}")
        print(f"Checkpoints → {ckpt_dir}/")
        if flows is None:
            print(f"Lag indices ({args.n_lags}): {lags}\n")
        else:
            print(f"Turnover-time mode: T = {args.t_def}, lags t* = {lags_tstar}, "
                  f"window t* = {args.window_tstar}, normalize = {args.normalize}")
            for flow, fp in flows.items():
                print(f"  {flow}: {fp['n_sims']} sims, T={fp['T']:.4f}, U={fp['U']:.4f}, "
                      f"L={fp['L']:.4f}, dt={fp['dt']:.4f}, lags={fp['lags'].tolist()}, "
                      f"window={fp['window']}")
            print()
    comm.Barrier()  # ensure ckpt_dir exists before any rank writes

    # Flat shard list, identical on every rank, distributed round-robin.
    shards = [
        (yaml_i, sim, j)
        for yaml_i, sim in accessible
        for j in range(args.n_shards_per_sim)
    ]
    my_shards = shards[rank::size]

    # ── Each rank computes its shards, checkpointing as it goes ───────────────
    for yaml_i, sim, j in my_shards:
        ckpt = checkpoint_path(ckpt_dir, sim, yaml_i, j)
        if ckpt.exists():
            print(f"  [rank {rank}] skip (done): {ckpt.name}")
            continue

        label = shard_label(sim)
        print(f"  [rank {rank}] {ckpt.name}  ({sim['path']})")

        # Unique, reproducible seed per shard — independent of rank count.
        seed = int(np.random.SeedSequence([args.seed, yaml_i, j]).generate_state(1)[0])

        if flows is None:
            sim_lags, sim_window, scales = lags, min_steps, None
        else:
            fp = flows[flow_key(sim)]
            sim_lags, sim_window = fp["lags"], fp["window"]
            scales = {"U": fp["U"], "L": fp["L"], "T": fp["T"]} if args.normalize else None

        X_block, feature_names = make_feature_matrix(
            sim["path"], sim_lags,
            batch_size=args.batch_size,
            n_batches=args.batches_per_shard,
            n_load=args.n_load,
            max_steps=None if args.time_augment else sim_window,
            window_size=sim_window if args.time_augment else None,
            seed=seed,
            scales=scales,
        )
        if X_block.shape[0] == 0:
            continue

        # Atomic write: a wall-time kill mid-dump never leaves a partial .pkl.
        tmp = ckpt.with_name(ckpt.name + f".tmp.{rank}")
        joblib.dump(
            {
                "X":             X_block,
                "label":         label,
                "feature_names": feature_names,
                "n_lags":        n_lags,
                "batch_size":    args.batch_size,
                "window_size":   sim_window,
                "lags":          np.asarray(sim_lags),
                "normalized":    scales is not None,
                **({"flow": flow_key(sim), "flow_params": flows[flow_key(sim)]} if flows else {}),
                "yaml_i":        yaml_i,
                "shard":         j,
                "seed":          seed,
            },
            tmp,
        )
        os.replace(tmp, ckpt)

    comm.Barrier()

    # ── Rank 0 assembles whatever checkpoints now exist ───────────────────────
    if rank == 0:
        assemble(ckpt_dir, label_map, args.output)


if __name__ == "__main__":
    main()

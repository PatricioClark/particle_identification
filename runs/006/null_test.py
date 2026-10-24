"""Control test: does the classifier separate particle pools rather than particle models?

In 006 every class is built from ONE pool of 50 000 particles, and its 100
samples (batches of 5000) are all drawn from that pool, so they share the
pool's fluctuations. Two different pools could then be told apart even if they
were the same species. This test builds, for two species, two DISJOINT pools
each (A and B), with exactly the 006 procedure (RND512 lags/window in turnover
times, normalized features, per-particle time offsets), and checks:

  (i)   pool A vs pool B of the same species   -> should be ~50% (chance)
  (ii)  species 1 vs species 2, pools A only   -> what 006 measures
  (iii) train on pools A, test on pools B      -> accuracy on unseen particles

Step 1, one pool per call (heavy: reads the whole trajectory):
    python null_test.py features --sim outs_HPP_0-1 --pool A --out null/HPP_0-1_A.npz
Step 2, classify (light):
    python null_test.py analyze --dir null
"""

import argparse
import json
from pathlib import Path

import numpy as np

BASE = "/share/data4/bespanol/HIT512_multipart/bin"


def pool_indices(n_total, sim, pool, n_load):
    """Disjoint pools A/B: one permutation per species (seeded by its name), A = first n_load, B = next."""
    seed = int(np.frombuffer(sim.encode(), dtype=np.uint8).sum()) + 1000
    perm = np.random.default_rng(seed).permutation(n_total)
    k = {"A": 0, "B": 1}[pool]
    return np.sort(perm[k * n_load:(k + 1) * n_load])


def cmd_features(args):
    from dataset import probe_sim, load_simulation
    from features import extract_features

    fp = json.load(open(args.flow_params))["flows"]["RND512"]
    lags, window = np.array(fp["lags"]), fp["window"]
    scales = {"U": fp["U"], "L": fp["L"], "T": fp["T"]}
    path = f"{BASE}/{args.sim}/"

    n_total, n_steps = probe_sim(path)
    idxs = pool_indices(n_total, args.sim, args.pool, args.n_load)
    print(f"{args.sim} pool {args.pool}: {len(idxs)} partículas de {n_total}, {n_steps} pasos, "
          f"lags {lags.tolist()}, ventana {window}", flush=True)
    times, pos, vel = load_simulation(path, idxs)
    dt = float(np.diff(times).mean())
    T = pos.shape[2]

    rng = np.random.default_rng([args.seed, args.pool == "B"])
    rows = []
    for _ in range(args.n_batches):
        sel = rng.choice(len(idxs), size=args.batch_size, replace=False)
        offsets = rng.integers(0, T - window + 1, size=args.batch_size)
        t_idx = offsets[:, None] + np.arange(window)[None, :]
        t_idx3d = np.broadcast_to(t_idx[:, np.newaxis, :], (args.batch_size, 3, window))
        p = np.take_along_axis(pos[sel], t_idx3d, axis=2)
        v = np.take_along_axis(vel[sel], t_idx3d, axis=2)
        feat, names = extract_features(p, v, dt, lags, scales=scales)
        rows.append(feat)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=np.array(rows), names=np.array(names), sim=args.sim, pool=args.pool)
    print(f"Saved → {args.out}", flush=True)


def cmd_analyze(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = {}
    for f in sorted(Path(args.dir).glob("*.npz")):
        d = np.load(f)
        data[(str(d["sim"]), str(d["pool"]))] = d["X"]
    sims = sorted({s for s, _ in data})
    print(f"Pools: {sorted(data)}")

    def pipe():
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                        class_weight="balanced", random_state=42,
                                                        n_jobs=-1))])

    def cv(X0, X1):
        X = np.vstack([X0, X1]); y = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
        cvs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(pipe(), X, y, cv=cvs).mean()

    print("\n(i) mismo modelo, grupos de partículas distintos (azar = 0.50):")
    for s in sims:
        print(f"    {s}: A vs B  acierto CV = {cv(data[(s, 'A')], data[(s, 'B')]):.3f}")

    s1, s2 = sims
    print(f"\n(ii) {s1} vs {s2}, solo grupos A (como en 006):  acierto CV = "
          f"{cv(data[(s1, 'A')], data[(s2, 'A')]):.3f}")

    Xtr = np.vstack([data[(s1, "A")], data[(s2, "A")]])
    ytr = np.r_[np.zeros(len(data[(s1, "A")])), np.ones(len(data[(s2, "A")]))]
    Xte = np.vstack([data[(s1, "B")], data[(s2, "B")]])
    yte = np.r_[np.zeros(len(data[(s1, "B")])), np.ones(len(data[(s2, "B")]))]
    m = pipe().fit(Xtr, ytr)
    pred = m.predict(Xte)
    print(f"\n(iii) entrenado con grupos A, evaluado con grupos B (partículas nunca vistas):")
    print(f"    acierto = {np.mean(pred == yte):.3f}   "
          f"({s1}: {np.mean(pred[yte == 0] == 0):.2f}, {s2}: {np.mean(pred[yte == 1] == 1):.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--sim", required=True, help="outs_* directory under HIT512_multipart/bin")
    f.add_argument("--pool", choices=["A", "B"], required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--flow-params", default="flow_params_006.json")
    f.add_argument("--n-load", type=int, default=50000)
    f.add_argument("--batch-size", type=int, default=5000)
    f.add_argument("--n-batches", type=int, default=100)
    f.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("analyze")
    a.add_argument("--dir", default="null")
    args = p.parse_args()
    cmd_features(args) if args.cmd == "features" else cmd_analyze(args)


if __name__ == "__main__":
    main()

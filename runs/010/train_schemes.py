"""Compare window lengths and training schemes on the same held-out time blocks.

Uses the per-particle, per-window statistics of moments_dataset.py. Time
blocks of 16.7 T (independent: the flow energy decorrelates in ~10 T) are
split into 3 folds by block index (block % 3), the same for every simulation
of a flow, so the test blocks are moments of the flow the classifier never
saw, for every class at once.

For each window length (16.7, 6, 4 T) and each training scheme:

  coherent : each training sample = 5000 particles in one window (as 008);
  mixed    : each training sample = 5000 particles, each in a random window
             of the training blocks (Patricio's proposal; as 006/007 but
             restricted to the training blocks).

Test samples are always time-coherent (one window), like a measurement. The
new RND512 run is evaluated with a classifier trained on all the blocks.

Usage
-----
python train_schemes.py --moments DIR --moments-nueva DIR
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WINDOWS = ["16.7", "6", "4"]
SCHEMES = ["coherent", "mixed"]
PAIRS = [("HPP_St0.1_RND512", "ONLD_St0.1_RND512"), ("HPP_St0.6_RND512", "ONLD_St1_RND512"),
         ("HPP_St4_RND512", "ONLD_St20_RND512")]
LIGHT768 = ["LAG_St0.0_TG768", "BB_St8.89_TG768", "MR_St0.76_TG768", "MR_St3.2_TG768", "NLD_St8.89_TG768"]


def load(d):
    return sorted((joblib.load(f) for f in Path(d).glob("*.pkl")), key=lambda r: r["yaml_i"])


def features(m, n_lags):
    """Sample features from averaged per-particle moments (same layout as features.py)."""
    k = n_lags
    v2, v4, a2, a4 = m[..., 4 * k], m[..., 4 * k + 1], m[..., 4 * k + 2], m[..., 4 * k + 3]
    return np.concatenate([m[..., :k] / v2[..., None], m[..., k:2 * k], m[..., 2 * k:3 * k],
                           m[..., 3 * k:4 * k], np.stack([v2, v4 / v2 ** 2, a2, a4 / a2 ** 2], -1)], -1)


def coherent_samples(M, windows, n, batch, rng, n_lags):
    """n samples spread over the given windows; each sample = batch particles of one window."""
    out = []
    per = int(np.ceil(n / len(windows)))
    for w in windows:
        for _ in range(per):
            sel = rng.choice(M.shape[1], size=batch, replace=False)
            out.append(M[w, sel].astype(np.float64).mean(0))
    return features(np.array(out), n_lags)


def mixed_samples(M, windows, n, batch, rng, n_lags):
    """n samples; each particle of a sample in a random window among the given ones."""
    out = []
    for _ in range(n):
        sel = rng.choice(M.shape[1], size=batch, replace=False)
        ws = rng.choice(windows, size=batch)
        out.append(M[ws, sel].astype(np.float64).mean(0))
    return features(np.array(out), n_lags)


def pipe(seed):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                    class_weight="balanced", random_state=seed, n_jobs=-1))])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--moments", required=True)
    p.add_argument("--moments-nueva", required=True)
    p.add_argument("--n-train", type=int, default=100, help="Training samples per simulation and fold")
    p.add_argument("--n-test", type=int, default=10, help="Test samples per test window")
    p.add_argument("--batch", type=int, default=5000)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    runs, nueva = load(args.moments), load(args.moments_nueva)
    print(f"{len(runs)} simulaciones de entrenamiento, {len(nueva)} especies de la corrida nueva; "
          f"pool {runs[0]['n_pool']} partículas, muestras de {args.batch}\n")
    summary = []
    for wl in WINDOWS:
        n_lags = len(runs[0]["windows"][wl]["names"]) // 4 - 1
        n_win = {r["label"]: len(r["windows"][wl]["block"]) for r in runs}
        for scheme in SCHEMES:
            rng = np.random.default_rng(args.seed)
            make = coherent_samples if scheme == "coherent" else mixed_samples
            y_true, y_pred = [], []
            for f in range(args.n_folds):
                Xtr, ytr, Xte, yte = [], [], [], []
                for r in runs:
                    W = r["windows"][wl]
                    tr = np.where(W["block"] % args.n_folds != f)[0]
                    te = np.where(W["block"] % args.n_folds == f)[0]
                    if len(tr) == 0 or len(te) == 0:
                        continue
                    Xtr.append(make(W["M"], tr, args.n_train, args.batch, rng, n_lags))
                    ytr += [r["label"]] * len(Xtr[-1])
                    Xte.append(coherent_samples(W["M"], te, args.n_test * len(te), args.batch, rng, n_lags))
                    yte += [r["label"]] * len(Xte[-1])
                model = pipe(args.seed).fit(np.vstack(Xtr), ytr)
                y_pred += list(model.predict(np.vstack(Xte)))
                y_true += yte
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            ok = y_true == y_pred
            flow = np.array([l.rsplit("_", 1)[1] for l in y_true])
            pairs = np.mean(ok[np.isin(y_true, [c for pr in PAIRS for c in pr])])
            light = np.mean(ok[np.isin(y_true, LIGHT768)])

            # New run: train on all blocks
            Xall, yall = [], []
            for r in runs:
                W = r["windows"][wl]
                allw = np.arange(len(W["block"]))
                Xall.append(make(W["M"], allw, args.n_train, args.batch, rng, n_lags))
                yall += [r["label"]] * len(Xall[-1])
            model = pipe(args.seed).fit(np.vstack(Xall), yall)
            per_sp, rows = [], []
            for r in nueva:
                W = r["windows"][wl]
                pn = model.predict(coherent_samples(W["M"], np.arange(len(W["block"])), 100, args.batch, rng, n_lags))
                vals, cnt = np.unique(pn, return_counts=True)
                top = ", ".join(f"{vals[i]} {cnt[i]}" for i in np.argsort(-cnt)[:2])
                if r["expected"]:
                    per_sp.append(float(np.mean(pn == r["expected"])))
                    rows.append(f"      {r['name']:15s} {per_sp[-1]:5.2f} → {top}")
                else:
                    rows.append(f"      {r['name']:15s}  —    → {top}")
            n_ok = sum(a >= 0.5 for a in per_sp)
            print(f"── ventana {wl} T, entrenamiento {scheme} (ventanas por simulación: "
                  f"{min(n_win.values())}–{max(n_win.values())}) ──")
            print(f"   prueba por bloques: total {ok.mean():.3f} | TG768 {ok[flow == 'TG768'].mean():.3f} | "
                  f"RND512 {ok[flow == 'RND512'].mean():.3f} | HPP/ONLD 512 {pairs:.3f} | livianas 768 {light:.3f}")
            print(f"   corrida nueva: {n_ok} de {len(per_sp)} especies bien")
            print("\n".join(rows) + "\n")
            summary.append((wl, scheme, ok.mean(), ok[flow == "TG768"].mean(), ok[flow == "RND512"].mean(),
                            pairs, light, n_ok, len(per_sp)))

    print("── Resumen ──")
    print(f"{'ventana':>7s} {'entrenamiento':13s} {'total':>6s} {'TG768':>6s} {'RND512':>6s} "
          f"{'HPP/ONLD':>8s} {'livianas':>8s} {'nueva':>6s}")
    for wl, s, a, a7, a5, pr, li, nb, nt in summary:
        print(f"{wl:>7s} {s:13s} {a:6.3f} {a7:6.3f} {a5:6.3f} {pr:8.3f} {li:8.3f} {nb:3d}/{nt}")


if __name__ == "__main__":
    main()

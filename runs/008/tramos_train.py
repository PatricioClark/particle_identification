"""Train and evaluate on the time-coherent dataset (tramos_dataset.py).

For each normalisation (V0 flow scales, V1 local L/U, V2 local Kolmogorov) and
each feature set (all 24 features, or without acc_variance/acc_flatness):

  - Cross-validation grouped by tramo (StratifiedGroupKFold): all the samples
    of a tramo go together to training or to test, so every test tramo is a
    moment of the flow the classifier has not seen. TG768 simulations have
    3-4 tramos, so 3 folds.
  - The new RND512 run (another forcing seed): train on all the training
    tramos, predict its samples, accuracy per species.

Classifier as in 006/007: StandardScaler + RandomForest (300 trees, balanced).

Usage
-----
python tramos_train.py --ckpt ckpt --ckpt-nueva ckpt_nueva [--n-folds 3]
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VARIANTS = {"V0": "escalas del flujo (como 007)", "V1": "L/U del tramo", "V2": "Kolmogorov del tramo"}
ACC = ["acc_variance", "acc_flatness"]
PAIRS = [("HPP_St0.1_RND512", "ONLD_St0.1_RND512"), ("HPP_St0.6_RND512", "ONLD_St1_RND512"),
         ("HPP_St4_RND512", "ONLD_St20_RND512")]


def load(ckpt_dir):
    return sorted((joblib.load(f) for f in Path(ckpt_dir).glob("*.pkl")), key=lambda r: r["yaml_i"])


def pipe(seed):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                    class_weight="balanced", random_state=seed, n_jobs=-1))])


def stack(runs, variant, keep):
    X = np.vstack([r["X"][variant][:, keep] for r in runs])
    lab = np.concatenate([[r["label"]] * len(r["tramo"]) for r in runs])
    groups = np.concatenate([r["yaml_i"] * 1000 + r["tramo"] for r in runs])
    return X, lab, groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="ckpt")
    p.add_argument("--ckpt-nueva", default="ckpt_nueva")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    runs, nueva = load(args.ckpt), load(args.ckpt_nueva)
    names = list(runs[0]["feature_names"])
    labels = [r["label"] for r in runs]
    n_tr = {r["label"]: len(set(r["tramo"])) for r in runs}
    print(f"Entrenamiento: {len(runs)} simulaciones, {sum(len(r['tramo']) for r in runs)} muestras; "
          f"tramos por simulación: min {min(n_tr.values())}, max {max(n_tr.values())}")
    print(f"Corrida nueva: {len(nueva)} especies\n")

    feature_sets = {"todas": list(range(len(names))),
                    "sin aceleración": [i for i, n in enumerate(names) if n not in ACC]}
    summary = []
    for v, vdesc in VARIANTS.items():
        for fs_name, keep in feature_sets.items():
            X, y, groups = stack(runs, v, keep)
            cv = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            pred = cross_val_predict(pipe(args.seed), X, y, cv=cv, groups=groups)
            ok = pred == y
            flow = np.array([l.rsplit("_", 1)[1] for l in y])
            pair_acc = [np.mean(ok[np.isin(y, pr)]) for pr in PAIRS]

            model = pipe(args.seed).fit(X, y)
            per_sp, rows = [], []
            for r in nueva:
                pn = model.predict(r["X"][v][:, keep])
                vals, cnt = np.unique(pn, return_counts=True)
                top = ", ".join(f"{vals[i]} {cnt[i]}" for i in np.argsort(-cnt)[:3])
                if r["expected"]:
                    acc = float(np.mean(pn == r["expected"]))
                    per_sp.append(acc)
                    rows.append(f"      {r['name']:15s} acierto {acc:5.2f} → {top}")
                else:
                    rows.append(f"      {r['name']:15s} (sin clase)   → {top}")

            print(f"── {v} ({vdesc}), features: {fs_name} ──")
            print(f"   CV por tramos: total {ok.mean():.3f} | TG768 {ok[flow == 'TG768'].mean():.3f} | "
                  f"RND512 {ok[flow == 'RND512'].mean():.3f}")
            print("   pares HPP/ONLD de 512: " + "  ".join(f"{a.split('_')[0]}{a.split('_')[1][2:]}/"
                  f"{b.split('_')[0]}{b.split('_')[1][2:]} {acc:.2f}" for (a, b), acc in zip(PAIRS, pair_acc)))
            low = sorted(((np.mean(ok[y == l]), l) for l in labels))[:5]
            print("   clases con menos acierto: " + ", ".join(f"{l} {a:.2f}" for a, l in low))
            print(f"   corrida nueva: {sum(a >= 0.5 for a in per_sp)} de {len(per_sp)} especies bien "
                  f"(acierto medio {np.mean(per_sp):.3f})")
            print("\n".join(rows) + "\n")
            summary.append((v, fs_name, ok.mean(), ok[flow == "TG768"].mean(), ok[flow == "RND512"].mean(),
                            sum(a >= 0.5 for a in per_sp), len(per_sp)))

    print("── Resumen ──")
    print(f"{'variante':4s} {'features':16s} {'CV total':>8s} {'TG768':>6s} {'RND512':>6s} {'nueva':>6s}")
    for v, fs, a, a7, a5, nb, nt in summary:
        print(f"{v:4s} {fs:16s} {a:8.3f} {a7:6.3f} {a5:6.3f} {nb:3d}/{nt}")


if __name__ == "__main__":
    main()

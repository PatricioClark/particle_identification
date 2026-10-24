"""Does the classifier recognise the particle model or the simulation?

Patricio's concern: the 006 classifier works too well. Every class is one
simulation, so cross-validation only ever tests on simulations it has already
seen. The 8 RND512 classes are the exception: they share a single flow
realisation, so among them there is no "simulation" to recognise, only physics.

Checks, all with the train_classifier.py pipeline (StandardScaler + RF):
  1. 21-class CV: accuracy within each flow, and errors that cross flows.
  2. RND512-only CV (8 classes, same flow): confusion matrix.
  3. Same as 2 without the level features (vel_variance, acc_variance), which
     are the ones most sensitive to the flow's energy.
  4. Spread of each class: distance between class centroids in units of the
     within-class scatter, to see how tightly the samples cluster.

Usage
-----
python check_flow_vs_model.py [--dataset dataset.pkl] [--n-trees 300] [--n-folds 5]
"""

import argparse
import joblib

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEVEL_FEATURES = ["vel_variance", "acc_variance"]


def make_pipe(n_trees, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                       class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])


def cv_predict(X, y, args):
    cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    return cross_val_predict(make_pipe(args.n_trees, args.seed), X, y, cv=cv)


def print_confusion(y, y_pred, names):
    labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    short = [names[i].replace("_RND512", "").replace("_TG768", "") for i in labels]
    w = max(len(s) for s in short)
    print(" " * (w + 2) + " ".join(f"{i:>4d}" for i in range(len(labels))))
    for i, row in enumerate(cm):
        print(f"{i:>2d} {short[i]:{w}s}" + " ".join(f"{v:>4d}" for v in row))
    print(f"acierto: {np.trace(cm) / cm.sum():.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset.pkl")
    p.add_argument("--n-trees", type=int, default=300)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = joblib.load(args.dataset)
    X, y = d["X"], np.asarray(d["y"])
    names, feats = list(d["label_names"]), list(d["feature_names"])
    flow = np.array([names[i].rsplit("_", 1)[1] for i in y])

    # 1. 21-class CV, split by flow
    print("── 1. Clasificador de 21 clases (CV), por flujo ──")
    y_pred = cv_predict(X, y, args)
    pred_flow = np.array([names[i].rsplit("_", 1)[1] for i in y_pred])
    for f in sorted(set(flow)):
        m = flow == f
        print(f"  {f}: acierto {np.mean(y_pred[m] == y[m]):.3f} ({m.sum()} muestras), "
              f"asignadas al otro flujo: {np.sum(pred_flow[m] != f)}")
    m512 = flow == "RND512"
    print("  Confusión de las clases RND512 dentro del clasificador de 21 clases:")
    print_confusion(y[m512], y_pred[m512], names)

    # 2. RND512 only
    print("\n── 2. Solo RND512 (8 clases, mismo flujo) ──")
    X5, y5 = X[m512], y[m512]
    print_confusion(y5, cv_predict(X5, y5, args), names)

    # 3. RND512 only, without level features
    keep = [i for i, n in enumerate(feats) if n not in LEVEL_FEATURES]
    print(f"\n── 3. Solo RND512, sin {', '.join(LEVEL_FEATURES)} ──")
    print_confusion(y5, cv_predict(X5[:, keep], y5, args), names)

    # 4. Class spread: centroid distance / within-class scatter (standardised features)
    print("\n── 4. Separación entre clases (todas): distancia entre centroides / dispersión interna ──")
    Z = StandardScaler().fit_transform(X)
    classes = sorted(set(y))
    cent = {c: Z[y == c].mean(0) for c in classes}
    scat = {c: np.sqrt(((Z[y == c] - cent[c]) ** 2).sum(1).mean()) for c in classes}
    print(f"  dispersión interna media: {np.mean(list(scat.values())):.3f} (en unidades de desvío global)")
    rows = []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            rows.append((np.linalg.norm(cent[a] - cent[b]) / (0.5 * (scat[a] + scat[b])), a, b))
    rows.sort()
    print("  pares más cercanos (separación < ~2 => se pisan):")
    for r, a, b in rows[:8]:
        print(f"    {r:6.2f}  {names[a]:20s} vs {names[b]}")


if __name__ == "__main__":
    main()

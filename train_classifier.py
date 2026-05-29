"""Train a Random Forest classifier on ensemble trajectory features.

Usage
-----
python train_classifier.py [options]

The script reads all simulations listed in simuls_bernardo.yaml, builds a
labelled feature dataset, runs stratified k-fold cross-validation, prints
a classification report, saves a confusion-matrix figure and a feature-
importance figure, then fits a final model on all data and saves it.
"""

import argparse
import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from dataset import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yaml",        default="simuls_bernardo.yaml")
    p.add_argument("--batch-size",  type=int, default=500,
                   help="Particles per training sample (default: 500)")
    p.add_argument("--n-batches",   type=int, default=10,
                   help="Training samples per simulation (default: 10)")
    p.add_argument("--n-lags",      type=int, default=15,
                   help="Lag points in feature vector (default: 15)")
    p.add_argument("--n-load",      type=int, default=5000,
                   help="Max particles loaded per simulation (default: 5000)")
    p.add_argument("--n-trees",     type=int, default=300)
    p.add_argument("--n-folds",     type=int, default=5)
    p.add_argument("--output",      default="model.pkl",
                   help="Path for saved model (default: model.pkl)")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("Building dataset …")
    X, y, feat_names, label_names = build_dataset(
        args.yaml,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        n_lags=args.n_lags,
        n_load=args.n_load,
        seed=args.seed,
    )

    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(label_names)} classes")
    for i, name in enumerate(label_names):
        print(f"  {i:2d}  {name}  ({(y == i).sum()} samples)")

    # Replace any NaNs with column medians (can arise at very large lags)
    col_medians = np.nanmedian(X, axis=0)
    nan_locs    = np.isnan(X)
    X[nan_locs] = col_medians[np.where(nan_locs)[1]]

    # ── Cross-validate ────────────────────────────────────────────────────────
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=args.n_trees,
            max_features="sqrt",
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])

    print(f"\nCross-validating ({args.n_folds}-fold stratified) …")
    cv     = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    print("\n── Classification report " + "─" * 50)
    print(classification_report(y, y_pred, target_names=label_names, zero_division=0))

    # Confusion matrix
    n_cls  = len(label_names)
    fig_cm, ax_cm = plt.subplots(figsize=(max(6, n_cls), max(5, n_cls - 1)))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, display_labels=label_names,
        ax=ax_cm, xticks_rotation=45, colorbar=False,
    )
    ax_cm.set_title("Confusion matrix (cross-validated)")
    fig_cm.tight_layout()
    cm_path = Path(args.output).with_suffix(".confusion.png")
    fig_cm.savefig(cm_path, dpi=150)
    print(f"Confusion matrix → {cm_path}")

    # ── Train on full data ────────────────────────────────────────────────────
    print("\nFitting final model on full dataset …")
    pipe.fit(X, y)

    # Feature importances
    importances = pipe.named_steps["clf"].feature_importances_
    top_idx     = np.argsort(importances)[::-1][:20]
    print("\n── Top 20 features " + "─" * 55)
    for rank, i in enumerate(top_idx):
        print(f"  {rank+1:2d}. {feat_names[i]:30s}  {importances[i]:.4f}")

    fig_fi, ax_fi = plt.subplots(figsize=(10, 4))
    ax_fi.bar(range(len(top_idx)), importances[top_idx])
    ax_fi.set_xticks(range(len(top_idx)))
    ax_fi.set_xticklabels([feat_names[i] for i in top_idx],
                           rotation=45, ha="right", fontsize=8)
    ax_fi.set_ylabel("Importance")
    ax_fi.set_title("Top 20 feature importances")
    fig_fi.tight_layout()
    fi_path = Path(args.output).with_suffix(".importances.png")
    fig_fi.savefig(fi_path, dpi=150)
    print(f"Feature importances → {fi_path}")

    # ── Save model ────────────────────────────────────────────────────────────
    saved = {
        "pipeline":      pipe,
        "label_names":   label_names,
        "feature_names": feat_names,
        "n_lags":        args.n_lags,
        "batch_size":    args.batch_size,
    }
    joblib.dump(saved, args.output)
    print(f"Model saved → {args.output}")


if __name__ == "__main__":
    main()

"""Exploratory data analysis of the particle drag-model feature dataset.

Produces eda.pdf with the following pages:
  1. Lag-curve plots  — mean ± std of VACF, MSD, S2, S4 per class
  2. Scalar features  — box plots of vel/acc variance and kurtosis/flatness
  3. Correlation      — Pearson heatmap of all 64 features
  4. PCA              — PC1 vs PC2 scatter coloured by class
  5. t-SNE            — 2-D t-SNE scatter coloured by class
  6. Mutual info      — ranked bar chart of feature importance

Usage
-----
python eda.py --dataset /path/to/dataset.pkl --output eda.pdf
"""

import argparse
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif


def load(path):
    d = joblib.load(path)
    return (
        d["X"], d["y"], d["feature_names"], d["label_names"],
        d["n_lags"], d["batch_size"],
    )


def class_palette(n):
    return plt.cm.tab10(np.linspace(0, 1, n))


# ── Page 1: lag curves ────────────────────────────────────────────────────────

def plot_lag_curves(pdf, X, y, feature_names, label_names, n_lags):
    groups = ["vacf", "msd", "s2", "s4"]
    titles = {
        "vacf": "Velocity Autocorrelation (VACF)",
        "msd":  "Mean Squared Displacement (MSD)",
        "s2":   "2nd-order Structure Function $S_2$",
        "s4":   "4th-order Structure Function $S_4$",
    }
    colors = class_palette(len(label_names))
    lag_idx = np.arange(n_lags)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lag-curve features: mean ± std per class", fontsize=13)

    for ax, grp in zip(axes.flat, groups):
        cols = [i for i, n in enumerate(feature_names) if n.startswith(grp + "_l")]
        for ci, label in enumerate(label_names):
            mask = y == ci
            vals = X[mask][:, cols]
            mu  = vals.mean(axis=0)
            sig = vals.std(axis=0)
            ax.plot(lag_idx, mu, color=colors[ci], label=label)
            ax.fill_between(lag_idx, mu - sig, mu + sig, alpha=0.15, color=colors[ci])
        ax.set_title(titles[grp])
        ax.set_xlabel("Lag index")
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 2: scalar feature box plots ─────────────────────────────────────────

def plot_scalar_boxes(pdf, X, y, feature_names, label_names):
    scalars = ["vel_variance", "vel_kurtosis", "acc_variance", "acc_flatness"]
    colors  = class_palette(len(label_names))

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("Scalar features by class", fontsize=13)

    for ax, name in zip(axes, scalars):
        col = feature_names.index(name)
        data = [X[y == ci, col] for ci in range(len(label_names))]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(label_names) + 1))
        ax.set_xticklabels(label_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(name.replace("_", " "))

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 3: PCA ───────────────────────────────────────────────────────────────

def plot_pca(pdf, X, y, label_names):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    colors = class_palette(len(label_names))

    fig, ax = plt.subplots(figsize=(7, 6))
    for ci, label in enumerate(label_names):
        mask = y == ci
        ax.scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.7,
                   color=colors[ci], label=label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — PC1 vs PC2")
    ax.legend(fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 4: t-SNE ─────────────────────────────────────────────────────────────

def plot_tsne(pdf, X, y, label_names):
    Xs = StandardScaler().fit_transform(X)
    Z = TSNE(n_components=2, perplexity=30, random_state=0,
             max_iter=1000).fit_transform(Xs)
    colors = class_palette(len(label_names))

    fig, ax = plt.subplots(figsize=(7, 6))
    for ci, label in enumerate(label_names):
        mask = y == ci
        ax.scatter(Z[mask, 0], Z[mask, 1], s=18, alpha=0.7,
                   color=colors[ci], label=label)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE (perplexity=30)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 5: mutual information ────────────────────────────────────────────────

def plot_mutual_info(pdf, X, y, feature_names):
    mi = mutual_info_classif(X, y, random_state=0)
    order = np.argsort(mi)[::-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(mi)), mi[order], color="steelblue", edgecolor="none")
    ax.set_xticks(range(len(mi)))
    ax.set_xticklabels([feature_names[i] for i in order],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mutual information (nats)")
    ax.set_title("Feature importance — mutual information with class label")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset.pkl")
    p.add_argument("--output",  default="eda.pdf")
    args = p.parse_args()

    X, y, feature_names, label_names, n_lags, _ = load(args.dataset)
    print(f"Loaded: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(label_names)} classes")

    with PdfPages(args.output) as pdf:
        print("Page 1: lag curves …")
        plot_lag_curves(pdf, X, y, feature_names, label_names, n_lags)

        print("Page 2: scalar box plots …")
        plot_scalar_boxes(pdf, X, y, feature_names, label_names)

        print("Page 3: PCA …")
        plot_pca(pdf, X, y, label_names)

        print("Page 4: t-SNE …")
        plot_tsne(pdf, X, y, label_names)

        print("Page 5: mutual information …")
        plot_mutual_info(pdf, X, y, feature_names)

    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

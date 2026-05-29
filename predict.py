"""Apply a trained classifier to a simulation directory or experimental data.

Usage
-----
python predict.py --model model.pkl --path /path/to/sim/dir [options]

The script loads a saved model, draws random batches of particles from the
given directory, computes ensemble features per batch, and reports the
predicted model probabilities aggregated over all batches.
"""

import argparse
import joblib
import numpy as np

from dataset import load_simulation, probe_sim
from features import extract_features


def predict_directory(model_path, sim_path, batch_size=500, n_batches=5,
                      n_load=2500, seed=0, verbose=True):
    """Predict model class for a simulation/experimental directory.

    Parameters
    ----------
    model_path : path to .pkl saved by train_classifier.py
    sim_path   : directory containing xlg.*.lag (and optionally vip.*.lag)
    batch_size : particles per batch (should match training)
    n_batches  : number of independent batches to average over
    n_load     : total particles to load
    seed       : RNG seed for particle selection

    Returns
    -------
    dict with keys:
        winner      — predicted class label (str)
        probs       — {label: probability} for all classes
        batch_preds — per-batch predicted labels
    """
    saved       = joblib.load(model_path)
    pipe        = saved["pipeline"]
    label_names = saved["label_names"]
    n_lags      = saved.get("n_lags", 15)

    rng     = np.random.default_rng(seed)
    n_total, _ = probe_sim(sim_path)
    if n_total == 0:
        raise FileNotFoundError(f"No xlg.*.lag files in {sim_path}")

    n_avail          = min(n_load, n_total)
    n_batches_actual = min(n_batches, n_avail // batch_size)
    n_need           = n_batches_actual * batch_size

    if n_batches_actual == 0:
        raise ValueError(
            f"Not enough particles: need at least {batch_size}, found {n_avail}."
        )

    idxs = rng.choice(n_total, size=n_need, replace=False)

    if verbose:
        print(f"Loading {n_need} particles ({n_batches_actual} batches) "
              f"from {sim_path} …")

    times, pos, vel = load_simulation(sim_path, idxs)
    dt = float(np.diff(times).mean()) if len(times) > 1 else 1.0

    # Build lag array that matches what the model was trained with
    from features import make_lags
    lags = make_lags(pos.shape[2], n_lags)

    feats = []
    for b in range(n_batches_actual):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        f, _ = extract_features(pos[sl], vel[sl], dt, lags)
        feats.append(f)

    X = np.array(feats, dtype=np.float64)
    col_medians = np.nanmedian(X, axis=0)
    nan_locs    = np.isnan(X)
    X[nan_locs] = col_medians[np.where(nan_locs)[1]]

    probs      = pipe.predict_proba(X)            # (n_batches, n_classes)
    preds      = pipe.predict(X)
    mean_probs = probs.mean(axis=0)
    winner_idx = int(mean_probs.argmax())

    if verbose:
        print(f"\n── Predictions (averaged over {n_batches_actual} batches) " + "─" * 20)
        for i, (name, prob) in enumerate(zip(label_names, mean_probs)):
            marker = "  ← winner" if i == winner_idx else ""
            print(f"  {name:25s}  {prob:.3f}{marker}")
        print(f"\nWinner: {label_names[winner_idx]}")

    return {
        "winner":      label_names[winner_idx],
        "probs":       dict(zip(label_names, mean_probs.tolist())),
        "batch_preds": [label_names[int(p)] for p in preds],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",       default="model.pkl")
    p.add_argument("--path",        required=True,
                   help="Simulation or experimental data directory")
    p.add_argument("--batch-size",  type=int, default=500)
    p.add_argument("--n-batches",   type=int, default=5)
    p.add_argument("--n-load",      type=int, default=2500)
    p.add_argument("--seed",        type=int, default=0)
    args = p.parse_args()

    predict_directory(
        args.model, args.path,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        n_load=args.n_load,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

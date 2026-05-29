"""Smoke test for build_dataset with timing.

Assumes a simuls_bernardo.yaml (or --yaml override) with the standard
convention. Uses small parameters so only a handful of particles are loaded.

Usage:
    python test_build_dataset.py
    python test_build_dataset.py --yaml simuls_bernardo.yaml
"""

import argparse
import time

import numpy as np

from dataset import build_dataset

BATCH_SIZE = 50
N_BATCHES  = 2
N_LAGS     = 5
N_LOAD     = 200   # max particles to load per sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="simuls_bernardo.yaml")
    args = parser.parse_args()

    t0 = time.time()
    X, y, feature_names, label_names = build_dataset(
        args.yaml,
        batch_size=BATCH_SIZE,
        n_batches=N_BATCHES,
        n_lags=N_LAGS,
        n_load=N_LOAD,
    )
    elapsed = time.time() - t0

    expected_features = 4 * N_LAGS + 4
    assert X.ndim == 2,                     f"X should be 2-D, got shape {X.shape}"
    assert X.shape[1] == expected_features, f"expected {expected_features} features, got {X.shape[1]}"
    assert y.shape == (X.shape[0],),        f"y shape {y.shape} doesn't match X rows {X.shape[0]}"
    assert len(feature_names) == expected_features
    assert len(label_names) > 0
    assert not np.any(np.isnan(X)),         "NaN in feature matrix"
    assert not np.any(np.isinf(X)),         "Inf in feature matrix"

    print(f"\nTest passed in {elapsed:.2f}s")
    print(f"  X: {X.shape}  y: {y.shape}")
    print(f"  Labels ({len(label_names)}): {label_names}")
    print(f"  Features: {feature_names[:4]} ...")

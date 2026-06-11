"""Build (X, y) training datasets from GHOST simulation directories.

Each *sample* is an ensemble statistic computed over a batch of
batch_size uncorrelated particles — the same observable you'd get
from an experiment with many tracked particles.

Label format:  "<MODEL>_St<st>"   e.g. "MR_St8.89", "LAG_St0.0"
"""

import yaml
import numpy as np
from pathlib import Path

from features import make_lags, extract_features

# ── Low-level I/O ────────────────────────────────────────────────────────────

def _read_lag(path):
    """Return (time: float, data: ndarray (3, N) float32)."""
    raw = np.fromfile(path, dtype=np.float32)
    n   = int(raw[0])
    t   = float(raw[1])
    return t, raw[2:].reshape(3, n)


def _glob_sorted(sim_path, quantity):
    p = Path(sim_path)
    search_dir = p / "outs" if (p / "outs").is_dir() else p
    return sorted(
        search_dir.glob(f"{quantity}.*.lag"),
        key=lambda p: int(p.stem.split(".")[-1]),
    )


def probe_sim(sim_path, quantity="xlg"):
    """Return (n_particles, n_steps) without loading the full time series."""
    files = _glob_sorted(sim_path, quantity)
    if not files:
        return 0, 0
    n = int(np.fromfile(files[0], dtype=np.float32)[0])
    return n, len(files)


def _load_quantity(sim_path, quantity, idxs, max_steps=None):
    """Load time series for selected particle indices.

    Returns (times: (T,), series: (n_p, 3, T)) or (None, None).
    """
    files = _glob_sorted(sim_path, quantity)
    if max_steps is not None:
        files = files[:max_steps]
    if not files:
        return None, None

    times, frames = [], []
    for f in files:
        t, data = _read_lag(f)
        times.append(t)
        frames.append(data[:, idxs].T.astype(np.float32))  # (n_p, 3)

    return np.array(times), np.stack(frames, axis=2)   # (n_p, 3, T)


# ── Simulation loader ─────────────────────────────────────────────────────────

def load_simulation(sim_path, idxs, max_steps=None):
    """Load xlg and vip (or vlg for LAG tracers) for selected particles.

    Returns
    -------
    times : (T,)
    pos   : (n_p, 3, T) float32
    vel   : (n_p, 3, T) float32
    """
    times, pos = _load_quantity(sim_path, "xlg", idxs, max_steps)
    if times is None or pos is None:
        raise FileNotFoundError(f"No xlg.*.lag files found in {sim_path}")

    _, vel = _load_quantity(sim_path, "vip", idxs, max_steps)
    if vel is None:
        _, vel = _load_quantity(sim_path, "vlg", idxs, max_steps)
        if vel is None:
            raise FileNotFoundError(f"No vip.*.lag or vlg.*.lag files found in {sim_path}")
        print(f"    velocity source: vlg (fluid tracer)")
    else:
        print(f"    velocity source: vip (inertial particle)")

    return times, pos, vel


# ── Feature-matrix builder for one simulation ─────────────────────────────────

def make_feature_matrix(sim_path, lags, batch_size=500, n_batches=10,
                        n_load=5000, max_steps=None, window_size=None, seed=0):
    """Compute (n_batches × n_features) feature matrix for one sim.

    A pool of up to n_load particles is loaded once, then each batch is an
    independent random subsample of batch_size particles drawn from that pool.
    Batches may overlap; with a pool much larger than batch_size the overlap is
    small. Decoupling the batch draw from the pool size lets a single load yield
    arbitrarily many ensemble samples, which is what makes sharding cheap.

    When window_size is provided and the loaded trajectory is longer, each
    particle in a batch gets an independent random time offset, so different
    temporal windows contribute to the ensemble statistics.

    Returns
    -------
    X     : (n_batches, n_features) float64
    names : list[str]
    """
    rng     = np.random.default_rng(seed)
    n_total, _ = probe_sim(sim_path)
    if n_total == 0:
        return np.empty((0, 0)), []

    pool_size = min(n_load, n_total)
    if pool_size < batch_size:
        print(f"    [warn] not enough particles for even one batch in {sim_path}")
        return np.empty((0, 0)), []

    pool_idxs = rng.choice(n_total, size=pool_size, replace=False)
    print(f"    loading pool of {pool_size} particles → {n_batches} batches")

    times, pos, vel = load_simulation(sim_path, pool_idxs, max_steps=max_steps)
    dt = float(np.diff(times).mean()) if len(times) > 1 else 1.0
    T  = pos.shape[2]

    do_offsets = window_size is not None and T > window_size

    rows, names = [], []
    for _ in range(n_batches):
        sel = rng.choice(pool_size, size=batch_size, replace=False)
        if do_offsets:
            offsets = rng.integers(0, T - window_size + 1, size=batch_size)
            t_idx   = offsets[:, None] + np.arange(window_size)[None, :]  # (B, W)
            t_idx3d = np.broadcast_to(t_idx[:, np.newaxis, :], (batch_size, 3, window_size))
            p = np.take_along_axis(pos[sel], t_idx3d, axis=2)
            v = np.take_along_axis(vel[sel], t_idx3d, axis=2)
        else:
            p, v = pos[sel], vel[sel]
        feat, names = extract_features(p, v, dt, lags)
        rows.append(feat)

    return np.array(rows, dtype=np.float64), names


# ── Full dataset builder ──────────────────────────────────────────────────────

def build_dataset(yaml_path, batch_size=500, n_batches=10, n_lags=15,
                  n_load=5000, seed=42, time_augment=True):
    """Build (X, y, feature_names, label_names) from all sims in the YAML.

    Parameters
    ----------
    yaml_path    : path to simuls_bernardo.yaml
    batch_size   : particles per training sample
    n_batches    : max training samples per simulation
    n_lags       : number of lag points in the feature vector
    n_load       : max particles to load from each simulation
    seed         : random seed
    time_augment : if True, apply per-particle random time offsets

    Returns
    -------
    X            : (n_samples, n_features) float64
    y            : (n_samples,) int
    feature_names: list[str]
    label_names  : list[str]  — label_names[i] is the class for y==i
    window_size  : int — trajectory window used for feature computation
    """
    with open(yaml_path) as fh:
        sims = yaml.safe_load(fh)

    # ── Pass 1: probe trajectory lengths to pick a common window_size ────────
    n_steps_per_sim = {}
    for sim in sims:
        path = sim["path"]
        if not Path(path).is_dir():
            continue
        _, n_steps = probe_sim(path)
        if n_steps > 0:
            n_steps_per_sim[path] = n_steps

    if not n_steps_per_sim:
        raise RuntimeError("No accessible simulation directories found.")

    min_steps = min(n_steps_per_sim.values())
    print(f"Trajectory lengths: min={min_steps}, "
          f"max={max(n_steps_per_sim.values())} steps across {len(n_steps_per_sim)} sims")
    print(f"Using window_size={min_steps} for consistency.\n")

    # Derive velocity from xlg costs 1 step; be conservative
    effective_steps = min_steps - 1
    lags = make_lags(effective_steps, n_lags)
    print(f"Lag indices ({n_lags}): {lags}\n")

    # ── Pass 2: build feature matrices ───────────────────────────────────────
    X_blocks, y_blocks = [], []
    label_map:  dict[str, int] = {}
    feature_names = None

    for i, sim in enumerate(sims):
        model = sim["parts"]["model"]
        st    = sim["parts"]["st"]
        label = f"{model}_St{st}"
        path  = sim["path"]

        if not Path(path).is_dir():
            print(f"  [skip] {label}: {path}")
            continue

        if label not in label_map:
            label_map[label] = len(label_map)
        y_int = label_map[label]

        print(f"  {label}  ({path})")
        X_block, names = make_feature_matrix(
            path, lags,
            batch_size=batch_size,
            n_batches=n_batches,
            n_load=n_load,
            max_steps=None if time_augment else min_steps,
            window_size=min_steps if time_augment else None,
            seed=seed + i,
        )
        if X_block.shape[0] == 0:
            continue
        if feature_names is None:
            feature_names = names

        X_blocks.append(X_block)
        y_blocks.append(np.full(len(X_block), y_int, dtype=int))

    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    return X, y, feature_names, label_names, min_steps

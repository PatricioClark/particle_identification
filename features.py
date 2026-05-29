"""Ensemble feature extraction from batches of particle trajectories.

All statistics are computed over the full batch of particles and then
averaged, so a single feature vector summarises an ensemble.

Feature layout (len = 4*len(lags) + 4):
    vacf_l00..vacf_lNN   — normalised Lagrangian velocity autocorrelation
    msd_l00..msd_lNN     — mean-squared displacement (box-units²)
    s2_l00..s2_lNN       — 2nd-order velocity structure function
    s4_l00..s4_lNN       — 4th-order velocity structure function
    vel_variance         — <|v|²>
    vel_kurtosis         — <|v|⁴>/<|v|²>²
    acc_variance         — <|a|²>  (a = Δv/Δt)
    acc_flatness         — <|a|⁴>/<|a|²>²
"""

import numpy as np

_BOX = 2.0 * np.pi


def make_lags(n_steps, n_lags):
    """Return exactly n_lags unique log-spaced lag indices in [1, n_steps//2].

    Raises ValueError if n_steps is too short.
    """
    max_lag = n_steps // 2
    if max_lag < n_lags:
        raise ValueError(
            f"Trajectory too short: n_steps={n_steps} gives max_lag={max_lag} "
            f"but n_lags={n_lags}. Use fewer lags or longer trajectories."
        )
    # Log-spaced candidates; 4× oversampling ensures enough unique integers
    raw = np.round(np.logspace(0, np.log10(max_lag), n_lags * 4)).astype(int)
    raw = np.clip(raw, 1, max_lag)
    unique = list(np.unique(raw))
    if len(unique) < n_lags:
        # Fill gaps with smallest missing integers
        missing = [i for i in range(1, max_lag + 1) if i not in set(unique)]
        unique = sorted(unique + missing[: n_lags - len(unique)])
    return np.array(unique[:n_lags], dtype=int)


def unwrap_positions(pos, box=_BOX):
    """Remove periodic-box jumps. pos: (..., T) → unwrapped same shape."""
    diffs = np.diff(pos.astype(np.float64), axis=-1)
    diffs -= box * np.round(diffs / box)
    out = np.empty(pos.shape, dtype=np.float64)
    out[..., 0] = pos[..., 0]
    out[..., 1:] = pos[..., 0:1] + np.cumsum(diffs, axis=-1)
    return out


def _vacf(vel, lags):
    """Normalised VACF. vel: (n_p, 3, T) → (len(lags),)."""
    v = vel.astype(np.float64)
    v0sq = float((v ** 2).mean()) or 1.0
    out = np.empty(len(lags))
    for i, tau in enumerate(lags):
        n = v.shape[2] - tau
        out[i] = float((v[:, :, :n] * v[:, :, tau: tau + n]).mean()) / v0sq
    return out


def _msd(pos_u, lags):
    """MSD (box-units²). pos_u: (n_p, 3, T) unwrapped → (len(lags),)."""
    out = np.empty(len(lags))
    for i, tau in enumerate(lags):
        n = pos_u.shape[2] - tau
        disp = pos_u[:, :, tau: tau + n] - pos_u[:, :, :n]
        out[i] = float((disp ** 2).mean())
    return out


def _sf(vel, lags, order):
    """Structure function S_order(τ). vel: (n_p, 3, T) → (len(lags),)."""
    v = vel.astype(np.float64)
    out = np.empty(len(lags))
    for i, tau in enumerate(lags):
        n = v.shape[2] - tau
        dv = v[:, :, tau: tau + n] - v[:, :, :n]
        dv_mag = np.sqrt((dv ** 2).sum(axis=1))   # (n_p, n)
        out[i] = float((dv_mag ** order).mean())
    return out


def extract_features(pos, vel, dt, lags):
    """Compute feature vector for one ensemble batch.

    Parameters
    ----------
    pos  : (n_p, 3, T) — positions in box units [0, 2π]
    vel  : (n_p, 3, T) — particle velocities
    dt   : float        — time interval between consecutive snapshots
    lags : 1-D int array — lag step indices (produced by make_lags)

    Returns
    -------
    features : float64 ndarray, shape (4*len(lags) + 4,)
    names    : list[str] of the same length
    """
    pos_u = unwrap_positions(pos)
    v64   = vel.astype(np.float64)

    c  = _vacf(v64, lags)
    d  = _msd(pos_u, lags)
    s2 = _sf(v64, lags, 2)
    s4 = _sf(v64, lags, 4)

    v2 = float((v64 ** 2).mean())
    v4 = float((v64 ** 4).mean())
    vel_kurtosis = v4 / v2 ** 2 if v2 > 0 else 0.0

    acc   = np.diff(v64, axis=2) / dt
    a2    = float((acc ** 2).mean())
    a4    = float((acc ** 4).mean())
    acc_flatness = a4 / a2 ** 2 if a2 > 0 else 0.0

    features = np.concatenate([c, d, s2, s4, [v2, vel_kurtosis, a2, acc_flatness]])

    n = len(lags)
    names = (
        [f"vacf_l{i:02d}" for i in range(n)]
        + [f"msd_l{i:02d}" for i in range(n)]
        + [f"s2_l{i:02d}" for i in range(n)]
        + [f"s4_l{i:02d}" for i in range(n)]
        + ["vel_variance", "vel_kurtosis", "acc_variance", "acc_flatness"]
    )
    return features, names

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

Lags may be fractional (in steps), e.g. a lag fixed in turnover times that
falls between two outputs. The statistics are then computed at the two
neighbouring integer lags and interpolated (see _interp_lags): log-log for
MSD, S2 and S4 (power laws at short lags, e.g. S2 ∝ τ²), linear for VACF.
Integer lags give exactly the same values as before.
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
    # Generate a dense log-spaced candidate set, deduplicate, then pick
    # n_lags evenly-spread indices so the selection spans 1…max_lag.
    raw = np.unique(
        np.clip(
            np.round(np.logspace(0, np.log10(max_lag), n_lags * 4)).astype(int),
            1, max_lag,
        )
    )
    if len(raw) < n_lags:
        missing = [i for i in range(1, max_lag + 1) if i not in set(raw)]
        raw = np.array(sorted(raw.tolist() + missing[: n_lags - len(raw)]))
    idx = np.round(np.linspace(0, len(raw) - 1, n_lags)).astype(int)
    return raw[idx]


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


def _interp_lags(lags):
    """Integer lags needed to evaluate the (possibly fractional) lags, and how to combine them.

    Returns (int_lags, lo, hi, w): int_lags sorted unique; for lag k, its value is
    interpolated between int_lags[lo[k]] and int_lags[hi[k]] with weight w[k] on hi
    (w = 0 and lo = hi when the lag is an integer). w is the position in log(τ),
    which is the right one for the log-log interpolation; for the linear one
    (VACF) _combine recomputes it in τ.
    """
    lags = np.asarray(lags, dtype=np.float64)
    if np.any(lags < 1):
        raise ValueError(f"Lags must be >= 1 step, got {lags}")
    floor = np.floor(lags + 1e-9).astype(int)
    ceil = np.where(lags - floor > 1e-9, floor + 1, floor)
    int_lags = np.unique(np.concatenate([floor, ceil]))
    lo = np.searchsorted(int_lags, floor)
    hi = np.searchsorted(int_lags, ceil)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(ceil > floor, (np.log(lags) - np.log(floor)) / (np.log(ceil) - np.log(floor)), 0.0)
    return int_lags, lo, hi, w


def _combine(vals, lags, int_lags, lo, hi, w, log):
    """Interpolate statistics computed at int_lags to the requested lags."""
    lags = np.asarray(lags, dtype=np.float64)
    a, b = vals[lo], vals[hi]
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.exp((1 - w) * np.log(a) + w * np.log(b))
        return np.where(lo == hi, a, out)
    span = int_lags[hi] - int_lags[lo]
    wl = np.where(span > 0, (lags - int_lags[lo]) / np.where(span > 0, span, 1), 0.0)
    return (1 - wl) * a + wl * b


def extract_features(pos, vel, dt, lags, scales=None):
    """Compute feature vector for one ensemble batch.

    Parameters
    ----------
    pos    : (n_p, 3, T) — positions in box units [0, 2π]
    vel    : (n_p, 3, T) — particle velocities
    dt     : float        — time interval between consecutive snapshots
    lags   : 1-D array — lag step indices (make_lags), or fractional lags in
             steps, which are interpolated from the neighbouring integer lags
    scales : dict with the flow's "U", "L", "T" (T = L/U), or None.
             If given, dimensional features are made dimensionless:
             MSD / L², S2 / U², S4 / U⁴, vel_variance / U², acc_variance / (U/T)².
             VACF, vel_kurtosis and acc_flatness are already dimensionless.

    Returns
    -------
    features : float64 ndarray, shape (4*len(lags) + 4,)
    names    : list[str] of the same length
    """
    pos_u = unwrap_positions(pos)
    v64   = vel.astype(np.float64)

    int_lags, lo, hi, w = _interp_lags(lags)
    c  = _combine(_vacf(v64, int_lags), lags, int_lags, lo, hi, w, log=False)
    d  = _combine(_msd(pos_u, int_lags), lags, int_lags, lo, hi, w, log=True)
    s2 = _combine(_sf(v64, int_lags, 2), lags, int_lags, lo, hi, w, log=True)
    s4 = _combine(_sf(v64, int_lags, 4), lags, int_lags, lo, hi, w, log=True)

    v2 = float((v64 ** 2).mean())
    v4 = float((v64 ** 4).mean())
    vel_kurtosis = v4 / v2 ** 2 if v2 > 0 else 0.0

    acc   = np.diff(v64, axis=2) / dt
    a2    = float((acc ** 2).mean())
    a4    = float((acc ** 4).mean())
    acc_flatness = a4 / a2 ** 2 if a2 > 0 else 0.0

    if scales is not None:
        U, L, T = scales["U"], scales["L"], scales["T"]
        d  = d / L**2
        s2 = s2 / U**2
        s4 = s4 / U**4
        v2 = v2 / U**2
        a2 = a2 / (U / T) ** 2

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

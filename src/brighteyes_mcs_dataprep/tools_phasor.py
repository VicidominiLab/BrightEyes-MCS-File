"""Minimal lifetime estimators used by :mod:`brighteyes_mcs_dataprep.alignment`."""

import numpy as np
from scipy.signal import savgol_filter

__all__ = [
    "estimate_lifetime_from_birfi",
    "estimate_lifetime_from_log",
    "estimate_lifetime_from_circmean",
]


def estimate_lifetime_from_birfi(
    x,
    y,
    window_length=11,
    polyorder=3,
    persistence=5,
    threshold=0.05,
    axis=0,
    return_bounds=False,
):
    """Estimate fluorescence lifetime from the centroid of a detected decay window."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D time axis")
    if y.ndim == 0:
        raise ValueError("y must contain at least one sample")
    if y.shape[axis] != x.shape[0]:
        raise ValueError("x and y must have matching lengths along the time axis")

    y_moved = np.moveaxis(y, axis, 0)
    original_shape = y_moved.shape[1:]
    y_2d = y_moved.reshape(x.shape[0], -1)
    n_time = y_2d.shape[0]

    if n_time < 3:
        raise ValueError("at least three time samples are required")
    if persistence < 1:
        raise ValueError("persistence must be at least 1")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if window_length > n_time:
        raise ValueError("window_length cannot exceed the number of time samples")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length")

    delta = float(np.mean(np.diff(x))) if n_time > 1 else 1.0
    dy = savgol_filter(
        y_2d,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta,
        axis=0,
        mode="interp",
    )

    t0 = np.argmin(dy, axis=0)
    y_min = np.min(y_2d, axis=0)
    y_range = np.max(y_2d, axis=0) - y_min
    t1 = np.full(y_2d.shape[1], n_time - 1, dtype=int)
    tau = np.full(y_2d.shape[1], np.nan, dtype=float)

    for ch in range(y_2d.shape[1]):
        stop = n_time - persistence
        for idx in range(t0[ch] + 1, stop):
            avg_diff = np.mean(dy[idx : idx + persistence, ch])
            amplitude = max(y_2d[idx + persistence, ch] - y_min[ch], 0.0)
            if avg_diff > 0.0 and amplitude > threshold * y_range[ch]:
                t1[ch] = idx
                break

        x_window = x[t0[ch] : t1[ch] + 1]
        y_window = y_2d[t0[ch] : t1[ch] + 1, ch]
        x_local = x_window - np.min(x_window)
        y_clamped = np.clip(y_window - np.min(y_window), a_min=0.0, a_max=None)
        weight_sum = np.sum(y_clamped)
        if weight_sum > 0.0:
            tau[ch] = np.sum(x_local * y_clamped) / weight_sum

    tau = tau.reshape(original_shape) if original_shape else tau[0]
    t0 = t0.reshape(original_shape) if original_shape else int(t0[0])
    t1 = t1.reshape(original_shape) if original_shape else int(t1[0])

    if return_bounds:
        return tau, t0, t1
    return tau


def estimate_lifetime_from_log(
    data_hist,
    t_ns,
    dt_ns,
    nbin,
    period_ns,
    start_level=0.95,
    end_level=0.25,
):
    """Estimate lifetime by fitting a log-linear decay section."""
    data_hist = np.asarray(data_hist, dtype=float)
    t_ns = np.asarray(t_ns, dtype=float)
    if (
        data_hist.size < 4
        or t_ns.size != data_hist.size
        or not np.isfinite(dt_ns)
        or dt_ns <= 0
        or nbin <= 0
        or not np.isfinite(period_ns)
        or period_ns <= 0
        or not np.isfinite(start_level)
        or not np.isfinite(end_level)
        or not (0.0 < end_level < start_level < 1.0)
    ):
        return None, None, None, None, None

    peak_idx = int(np.argmax(data_hist))
    trace_sum_peak0 = np.roll(data_hist, -peak_idx)
    trace_x_peak0_ns = np.roll(
        np.mod(t_ns - t_ns[peak_idx], period_ns),
        -peak_idx,
    )

    peak_value = float(trace_sum_peak0[0])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    y_start = start_level * peak_value
    idx_start_candidates = np.flatnonzero(trace_sum_peak0 <= y_start)
    fallback_levels = [end_level, 0.30, 0.40]
    fallback_levels = [level for level in fallback_levels if 0.0 < level < start_level]
    fallback_levels = list(dict.fromkeys(fallback_levels))
    idx_end_candidates = np.asarray([], dtype=int)
    selected_end_level = None
    for candidate_end_level in fallback_levels:
        y_end_candidate = candidate_end_level * peak_value
        idx_end_candidate = np.flatnonzero(trace_sum_peak0 <= y_end_candidate)
        if idx_start_candidates.size != 0 and idx_end_candidate.size != 0:
            selected_end_level = candidate_end_level
            idx_end_candidates = idx_end_candidate
            break

    if (
        selected_end_level is None
        or idx_start_candidates.size == 0
        or idx_end_candidates.size == 0
    ):
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    start_idx = int(idx_start_candidates[0])
    end_idx = int(idx_end_candidates[0])
    if end_idx <= start_idx:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    y_section = trace_sum_peak0[start_idx : end_idx + 1]
    x_section_ns = trace_x_peak0_ns[start_idx : end_idx + 1]
    positive = y_section > 0
    if np.count_nonzero(positive) < 4:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    x_fit_bins = np.arange(start_idx, end_idx + 1, dtype=float)[positive]
    x_fit_ns = x_section_ns[positive]
    y_fit_input = y_section[positive]
    slope, intercept = np.polyfit(x_fit_bins, np.log(y_fit_input), 1)
    if not np.isfinite(slope) or slope >= 0:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    tau_ns = -float(dt_ns) / slope
    if not np.isfinite(tau_ns) or tau_ns <= 0:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    y_fit = np.exp(intercept + slope * x_fit_bins)
    x_fit_bins_plot = np.mod(x_fit_bins + int(peak_idx), int(nbin))
    x_fit_ns_plot = np.mod(x_fit_ns + int(peak_idx) * dt_ns, period_ns)

    fit_curve = {
        "fit_len": int(end_idx - start_idx + 1),
        "x_fit_bins": x_fit_bins_plot,
        "x_fit_ns": x_fit_ns_plot,
        "y_fit": y_fit,
    }
    return float(tau_ns), fit_curve, peak_idx, trace_sum_peak0, trace_x_peak0_ns


def estimate_lifetime_from_circmean(
    counts,
    t0_ns,
    repetition_rate_MHz=40.0,
    background_per_bin=0.0,
    truncate_ns=None,
):
    """Estimate lifetime from the circular mean delay after ``t0_ns``."""
    y = np.asarray(counts, dtype=float)
    n = y.size

    Tcycle_ns = 1e3 / repetition_rate_MHz
    dt_ns = Tcycle_ns / n
    t_ns = (np.arange(n) + 0.5) * dt_ns
    delay_ns = (t_ns - t0_ns) % Tcycle_ns
    w = np.clip(y - background_per_bin, 0.0, None)

    if truncate_ns is not None:
        keep = delay_ns < truncate_ns
        w = w[keep]
        delay_ns = delay_ns[keep]

    s0 = np.sum(w)
    if s0 <= 0:
        return np.nan

    return np.sum(w * delay_ns) / s0

"""Calibration plotting helpers for BrightEyes MCS files."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

__all__ = [
    "normalize_histogram",
    "plot_calibration_fit_traces",
    "plot_calibration_lifetime_summary",
    "plot_calibration_shift_summary",
    "plot_lifetime_histogram",
    "weighted_lifetime_stats",
]


def normalize_histogram(histogram):
    """Return a unit-sum copy of a histogram, or zeros when it cannot be normalized."""
    histogram = np.asarray(histogram, dtype=float)
    total = np.sum(histogram)
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(histogram, dtype=float)
    return histogram / total


def weighted_lifetime_stats(lifetime, weights=None):
    """Return weighted mean and RMS for finite lifetime values."""
    lifetime = np.asarray(lifetime, dtype=float).ravel()
    mask = np.isfinite(lifetime)
    if weights is None:
        values = lifetime[mask]
        if values.size == 0:
            return np.nan, np.nan
        return float(np.mean(values)), float(np.std(values))

    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size != lifetime.size:
        raise ValueError("weights must match lifetime size")
    mask &= np.isfinite(weights) & (weights > 0)
    values = lifetime[mask]
    weights = weights[mask]
    if values.size == 0 or np.sum(weights) <= 0:
        return np.nan, np.nan
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(max(variance, 0.0)))


def _format_histogram_axis(ax, lifetime_axis="x", count_label="Pixel counts"):
    if lifetime_axis == "x":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlabel("Lifetime (ns)")
        ax.set_ylabel(count_label)
    elif lifetime_axis == "y":
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(count_label)
        ax.set_ylabel("Lifetime (ns)")
    else:
        raise ValueError("lifetime_axis must be 'x' or 'y'")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((3, 3))
    if lifetime_axis == "x":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def _column(table, name):
    if hasattr(table, "__getitem__"):
        values = table[name]
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        return np.asarray(values)
    raise TypeError("table must provide column access by name")


def _plot_vertical_value_histogram(
    values,
    ax,
    bins,
    color=None,
    label=None,
    alpha=0.22,
    linewidth=1.4,
    gaussian=False,
    show_stats=False,
):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    mean = float(np.mean(values))
    std = float(np.std(values))
    stats_label = None
    if show_stats:
        stats_prefix = label if label is not None else "data"
        stats_label = f"{stats_prefix}: mu={mean:.3g}, std={std:.3g}"

    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    can_plot_gaussian = gaussian and values.size > 1 and np.isfinite(std) and std > 0
    hist_label = label
    if show_stats:
        hist_label = stats_label if not can_plot_gaussian else None

    hist_line = ax.plot(
        hist,
        centers,
        drawstyle="steps-mid",
        color=color,
        linewidth=linewidth,
        label=hist_label,
    )[0]
    color = hist_line.get_color()
    ax.fill_betweenx(centers, hist, step="mid", color=color, alpha=alpha)

    if can_plot_gaussian:
        curve = np.exp(-0.5 * ((centers - mean) / std) ** 2)
        if np.sum(curve) > 0:
            curve = curve * (np.sum(hist) / np.sum(curve))
        ax.plot(
            curve,
            centers,
            color=color,
            linewidth=2,
            label=stats_label,
        )


def plot_lifetime_histogram(
    lifetime,
    weights=None,
    lifetime_bounds=None,
    bins=500,
    ax=None,
    color="yellowgreen",
    edgecolor="darkgreen",
    gaussian=True,
    label=None,
    lifetime_axis="x",
    count_label="Pixel counts",
):
    """Plot a lifetime histogram with optional intensity weights and Gaussian overlay."""
    if ax is None:
        _, ax = plt.subplots()
    if lifetime_axis not in {"x", "y"}:
        raise ValueError("lifetime_axis must be 'x' or 'y'")

    values = np.asarray(lifetime, dtype=float).ravel()
    mask = np.isfinite(values)
    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != values.size:
            raise ValueError("weights must match lifetime size")
        mask &= np.isfinite(weights) & (weights > 0)
        weights = weights[mask]
    values = values[mask]

    hist, edges = np.histogram(values, bins=bins, range=lifetime_bounds, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if lifetime_axis == "x":
        ax.plot(centers, hist, drawstyle="steps-mid", color=edgecolor, linewidth=1.5, label=label)
        ax.fill_between(centers, hist, step="mid", facecolor=color, alpha=0.6)
    else:
        ax.plot(
            hist,
            centers,
            drawstyle="steps-mid",
            color=edgecolor,
            linewidth=1.5,
            label=label,
        )
        ax.fill_betweenx(centers, hist, step="mid", facecolor=color, alpha=0.6)

    if gaussian and values.size > 1:
        mean, rms = weighted_lifetime_stats(values, weights=weights)
        if np.isfinite(mean) and np.isfinite(rms) and rms > 0:
            curve = np.exp(-0.5 * ((centers - mean) / rms) ** 2)
            if np.sum(curve) > 0:
                curve = curve * (np.sum(hist) / np.sum(curve))
            if lifetime_axis == "x":
                ax.plot(
                    centers,
                    curve,
                    color="red",
                    linewidth=2,
                    label=f"mu={mean:.2f} ns, sigma={rms:.2f} ns",
                )
                ax.axvline(mean, color="red", linestyle="--", linewidth=1)
            else:
                ax.plot(
                    curve,
                    centers,
                    color="red",
                    linewidth=2,
                    label=f"mu={mean:.2f} ns, sigma={rms:.2f} ns",
                )
                ax.axhline(mean, color="red", linestyle="--", linewidth=1)
            ax.legend(loc="upper right")

    _format_histogram_axis(ax, lifetime_axis=lifetime_axis, count_label=count_label)
    return ax


def plot_calibration_lifetime_summary(summary_table, fig=None, histogram_lifetime_axis="y"):
    """Plot fitted calibration lifetime and reference lifetime by channel."""
    channels = _column(summary_table, "channel")
    tau = _column(summary_table, "tau_ns").astype(float)
    tau_err = _column(summary_table, "tau_err_ns").astype(float)
    tau_ref = _column(summary_table, "tau_ref_ns").astype(float)
    fit_error = _column(summary_table, "fit_error").astype(float)

    if fig is None:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1, 0.55], wspace=0.16)
    ax_tau = fig.add_subplot(gs[0, 0])
    share_lifetime_axis = ax_tau if histogram_lifetime_axis == "y" else None
    ax_hist = fig.add_subplot(gs[0, 1], sharey=share_lifetime_axis)
    ax_error = fig.add_subplot(gs[1, 0], sharex=ax_tau)
    ax_error_hist = fig.add_subplot(gs[1, 1], sharey=ax_error)

    ax_tau.errorbar(
        channels,
        tau,
        yerr=tau_err,
        fmt="o-",
        color="tab:blue",
        linewidth=2,
        markersize=5,
        capsize=3,
        label="Fitted lifetime",
    )
    if np.any(np.isfinite(tau_ref)):
        ax_tau.plot(channels, tau_ref, "o-", color="tab:orange", label="Reference lifetime")
    mean_tau = np.nanmean(tau)
    std_tau = np.nanstd(tau)
    if np.isfinite(mean_tau):
        ax_tau.axhline(mean_tau, color="tab:blue", alpha=0.7)
        if np.isfinite(std_tau) and std_tau > 0:
            ax_tau.axhspan(mean_tau - std_tau, mean_tau + std_tau, color="tab:blue", alpha=0.08)
    ax_tau.set_ylabel("Lifetime (ns)")
    ax_tau.set_title("Calibration lifetime by channel")
    ax_tau.grid(True, alpha=0.3)
    ax_tau.legend(loc="best")

    plot_lifetime_histogram(
        tau,
        lifetime_bounds=None,
        bins=min(20, max(5, len(tau))),
        ax=ax_hist,
        gaussian=True,
        lifetime_axis=histogram_lifetime_axis,
        count_label="Channel count",
    )
    ax_hist.set_title("Lifetime distribution")
    if histogram_lifetime_axis == "y":
        ax_hist.tick_params(axis="y", labelleft=False, labelright=False)
        ax_hist.set_ylabel("")

    ax_error.plot(channels, fit_error, "o-", color="tab:red", linewidth=1.8, markersize=5)
    ax_error.set_xlabel("Channel")
    ax_error.set_ylabel("Fit RMSE")
    ax_error.grid(True, alpha=0.3)

    fit_error_bins = min(20, max(5, np.count_nonzero(np.isfinite(fit_error))))
    _plot_vertical_value_histogram(
        fit_error,
        ax_error_hist,
        bins=fit_error_bins,
        color="tab:red",
    )
    ax_error_hist.set_xlabel("Channel count")
    ax_error_hist.set_title("RMSE distribution")
    ax_error_hist.grid(True, alpha=0.25)
    ax_error_hist.tick_params(axis="y", labelleft=False, labelright=False)
    ax_error_hist.set_ylabel("")

    return fig, (ax_tau, ax_hist, ax_error)


def plot_calibration_shift_summary(summary_tables, labels=None, reference_channel=None, fig=None):
    """Plot stored channel skew and common delay for one or more calibration groups."""
    if not isinstance(summary_tables, (list, tuple)):
        summary_tables = [summary_tables]
    if labels is None:
        labels = [f"group {idx + 1}" for idx in range(len(summary_tables))]

    if fig is None:
        fig = plt.figure(figsize=(16, 6.4), constrained_layout=True)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[6, 1.15],
            height_ratios=[1.2, 1],
            wspace=0.08,
            hspace=0.18,
        )
        ax_shift = fig.add_subplot(gs[0, 0])
        ax_shift_hist = fig.add_subplot(gs[0, 1], sharey=ax_shift)
        ax_delay = fig.add_subplot(gs[1, 0])
        ax_delay_hist = fig.add_subplot(gs[1, 1], sharey=ax_delay)
    else:
        if len(fig.axes) >= 4:
            ax_shift, ax_shift_hist, ax_delay, ax_delay_hist = fig.axes[:4]
        else:
            fig.clear()
            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=[6, 1.15],
                height_ratios=[1.2, 1],
                wspace=0.08,
                hspace=0.18,
            )
            ax_shift = fig.add_subplot(gs[0, 0])
            ax_shift_hist = fig.add_subplot(gs[0, 1], sharey=ax_shift)
            ax_delay = fig.add_subplot(gs[1, 0])
            ax_delay_hist = fig.add_subplot(gs[1, 1], sharey=ax_delay)

    for table, label in zip(summary_tables, labels):
        channels = _column(table, "channel")
        channel_skew = _column(table, "channel_skew").astype(float)
        channel_skew_err = _column(table, "channel_skew_est_err").astype(float)
        common_delay = _column(table, "common_delay_in_ns").astype(float)
        fit_common_delay = _column(table, "fit_common_delay_in_ns").astype(float)
        fit_common_delay_err = _column(table, "fit_common_delay_err_in_ns").astype(float)

        shift_container = ax_shift.errorbar(
            channels,
            channel_skew,
            yerr=channel_skew_err,
            fmt="o-",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=label,
        )
        delay_container = ax_delay.errorbar(
            channels,
            common_delay,
            yerr=fit_common_delay_err,
            fmt="o--",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=label,
        )
        delay_container = ax_delay.errorbar(
            channels,
            fit_common_delay,
            yerr=fit_common_delay_err,
            fmt="o--",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=label,
        )
        shift_color = shift_container.lines[0].get_color()
        delay_color = delay_container.lines[0].get_color()
        shift_bins = min(20, max(5, np.count_nonzero(np.isfinite(channel_skew))))
        delay_bins = min(20, max(5, np.count_nonzero(np.isfinite(fit_common_delay))))
        _plot_vertical_value_histogram(
            channel_skew,
            ax_shift_hist,
            bins=shift_bins,
            color=shift_color,
            label=label,
            gaussian=True,
            show_stats=True,
        )
        _plot_vertical_value_histogram(
            fit_common_delay,
            ax_delay_hist,
            bins=delay_bins,
            color=delay_color,
            label=label,
            gaussian=True,
            show_stats=True,
        )

    if reference_channel is not None:
        ax_shift.axvline(reference_channel, color="0.75", linestyle=":", linewidth=1.2)

    ax_shift.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_shift.set_ylabel("Channel skew (bins)")
    ax_shift.set_title("Channel skew")
    ax_shift.grid(True, alpha=0.3)
    ax_shift.legend(loc="best")
    ax_shift_hist.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_shift_hist.set_xlabel("Channel count")
    ax_shift_hist.set_title("Distribution")
    ax_shift_hist.grid(True, alpha=0.25)
    ax_shift_hist.tick_params(axis="y", labelleft=False, labelright=False)
    ax_shift_hist.set_ylabel("")
    handles, _ = ax_shift_hist.get_legend_handles_labels()
    if handles:
        ax_shift_hist.legend(loc="best", fontsize="small")

    ax_delay.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_delay.set_xlabel("Channel")
    ax_delay.set_ylabel("Common delay (ns)")
    ax_delay.set_title("Fitted common delay")
    ax_delay.grid(True, alpha=0.3)
    ax_delay.legend(loc="best")
    ax_delay_hist.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_delay_hist.set_xlabel("Channel count")
    ax_delay_hist.set_title("Distribution")
    ax_delay_hist.grid(True, alpha=0.25)
    ax_delay_hist.tick_params(axis="y", labelleft=False, labelright=False)
    ax_delay_hist.set_ylabel("")
    handles, _ = ax_delay_hist.get_legend_handles_labels()
    if handles:
        ax_delay_hist.legend(loc="best", fontsize="small")

    return fig, (ax_shift, ax_delay)


def plot_calibration_fit_traces(t, traces, title=None, ax=None, log_scale=True):
    """Plot normalized calibration histograms and fitted trace for one channel."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    t = np.asarray(t, dtype=float)

    default_styles = {
        "data_for_fit": ("tab:blue", 1.0, 2.0),
        "ref_for_fit": ("tab:purple", 0.35, 1.5),
        "ref_common_delay_realigned": ("tab:purple", 0.9, 1.8),
        "irf_for_fit": ("tab:green", 0.35, 1.5),
        "irf_common_delay_realigned": ("tab:green", 0.9, 1.8),
        "data_fitted": ("tab:red", 1.0, 2.0),
    }

    for name, values in traces.items():
        if values is None:
            continue
        color, alpha, linewidth = default_styles.get(name, (None, 0.9, 1.5))
        ax.plot(
            t,
            normalize_histogram(values),
            label=name,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized counts")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax

"""Alignment, fitting, and IRF estimation utilities."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import shift
from tqdm.auto import tqdm
try:
    from scipy import optimize as scipy_optimize
except ImportError:  # pragma: no cover - optional dependency
    scipy_optimize = None
try:
    import torch
    from torch.fft import fftn, ifftn, ifftshift
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    fftn = ifftn = ifftshift = None

from .tools_phasor import (
    estimate_lifetime_from_birfi,
    estimate_lifetime_from_circmean,
    estimate_lifetime_from_log,
)

__all__ = ["Alignment"]


class Alignment:
    """Static helpers for fitting, shifts, and IRF estimation."""

    @staticmethod
    def _require_torch():
        if torch is None or fftn is None or ifftn is None or ifftshift is None:
            raise ImportError("torch is required for Alignment methods")

    @staticmethod
    def _require_scipy_optimize():
        if scipy_optimize is None:
            raise ImportError("scipy is required for Alignment fitting methods")

    @staticmethod
    def to_numpy_1d(x, dtype=None):
        if torch is not None and torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    @staticmethod
    def to_torch_1d(x, dtype=None, device=None):
        Alignment._require_torch()
        if torch.is_tensor(x):
            tensor = x.detach().clone()
            if device is not None:
                tensor = tensor.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            return tensor
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    @staticmethod
    def _normalize_histogram_1d(x, name="histogram"):
        hist = Alignment.to_numpy_1d(x, dtype=float)
        total = hist.sum()
        if not np.isfinite(total) or total <= 0:
            warnings.warn(
                f"{name} has a non-positive or non-finite sum; returning zeros without normalization",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros_like(hist, dtype=float)
        return hist / total

    @staticmethod
    def centroid(data):
        """Return the intensity-weighted centroid index of a 1D histogram."""
        hist = Alignment.to_numpy_1d(data, dtype=float)
        total = np.sum(hist)
        if not np.isfinite(total) or total <= 0:
            return np.nan
        return float(np.sum(np.arange(hist.size, dtype=float) * hist) / total)

    @staticmethod
    def clean_irf(irf, threshold=0.3, window=6):
        """
        Keep only the IRF samples around the thresholded centroid.

        ``threshold`` is interpreted in the same units as ``irf``. Normalize the
        input first when using fractional thresholds such as ``0.3``.
        ``window`` is expressed in histogram bins.
        """
        hist = Alignment.to_numpy_1d(irf, dtype=float)
        time = np.arange(hist.size, dtype=float)
        thresholded = np.where(hist > float(threshold), hist, 0.0)
        center = Alignment.centroid(thresholded)
        cleaned = np.zeros_like(hist, dtype=float)
        if not np.isfinite(center):
            return cleaned

        keep = (time > center - float(window)) & (time < center + float(window))
        cleaned[keep] = hist[keep]
        return cleaned

    @staticmethod
    def clean_irf_stack(irf, threshold=0.3, window=6, time_axis=0, normalize=False):
        """
        Apply :meth:`clean_irf` along the time axis of an IRF stack.

        Parameters are the same as :meth:`clean_irf`. If ``normalize=True``, each
        trace is divided by its finite positive maximum before thresholding and
        the normalized trace is returned, matching the historical notebook use.
        """
        irf_array = np.asarray(irf, dtype=float)
        moved = np.moveaxis(irf_array, time_axis, 0)
        flat = moved.reshape(moved.shape[0], -1)
        cleaned = np.empty_like(flat, dtype=float)

        for idx in range(flat.shape[1]):
            trace = flat[:, idx]
            if normalize:
                trace_max = np.nanmax(trace)
                if np.isfinite(trace_max) and trace_max > 0:
                    trace = trace / trace_max
                else:
                    trace = np.zeros_like(trace, dtype=float)
            cleaned[:, idx] = Alignment.clean_irf(trace, threshold=threshold, window=window)

        cleaned = cleaned.reshape(moved.shape)
        return np.moveaxis(cleaned, 0, time_axis)

    @staticmethod
    def _wrap_to_period(x, period, center=0.0):
        period = float(period)
        center = float(center)
        return center + np.mod(np.asarray(x) - center + 0.5 * period, period) - 0.5 * period

    @staticmethod
    def _normalize_circular_params(circular_params, n_params, lb, ub, p0):
        if circular_params is None:
            return {}

        if isinstance(circular_params, dict):
            circular_items = circular_params.items()
        else:
            circular_items = circular_params

        normalized = {}
        for item in circular_items:
            if isinstance(item, tuple) and len(item) == 2:
                idx, period = item
            else:
                idx, period = item, None

            idx = int(idx)
            if idx < 0 or idx >= n_params:
                raise IndexError(f"circular parameter index out of range: {idx}")

            if period is None:
                if np.isfinite(lb[idx]) and np.isfinite(ub[idx]):
                    period = ub[idx] - lb[idx]
                else:
                    raise ValueError(
                        f"missing circular period for parameter {idx}; provide it explicitly "
                        "or use finite bounds so it can be inferred"
                    )

            if float(period) <= 0:
                raise ValueError(f"circular period must be positive for parameter {idx}")

            if np.isfinite(lb[idx]) and np.isfinite(ub[idx]):
                center = 0.5 * (lb[idx] + ub[idx])
            else:
                center = float(p0[idx])

            normalized[idx] = {"period": float(period), "center": float(center)}

        return normalized

    @staticmethod
    def curve_fit_circular(
        f,
        xdata,
        ydata,
        p0=None,
        sigma=None,
        absolute_sigma=False,
        bounds=(-np.inf, np.inf),
        circular_curve_period=None,
        circular_curve_center=0.0,
        circular_params=None,
        maxfev=None,
        method="trf",
        **kwargs,
    ):
        """
        ``curve_fit``-like helper aware of circular data and circular parameters.

        Parameters
        ----------
        f : callable
            Model function ``f(xdata, *params)``.
        xdata, ydata : array-like
            Input coordinates and observations.
        p0, sigma, absolute_sigma, bounds, maxfev :
            Same meaning as in ``scipy.optimize.curve_fit``.
        circular_curve_period : float, optional
            If provided, residuals are wrapped onto a circle with this period.
        circular_curve_center : float, default 0.0
            Center of the wrapped residual interval.
        circular_params : dict or iterable, optional
            Circular parameters. A dict maps ``param_index -> period``. If an
            iterable is used, each item can be either ``index`` or
            ``(index, period)``. When only the index is given, the period is
            inferred from finite bounds.

        Returns
        -------
        popt, pcov : ndarray
            Best-fit parameters and covariance, like ``curve_fit``.
        """
        Alignment._require_scipy_optimize()

        xdata = np.asarray(xdata)
        ydata = Alignment.to_numpy_1d(ydata, dtype=float)

        if p0 is None:
            raise ValueError("curve_fit_circular requires an explicit p0")
        p0 = Alignment.to_numpy_1d(p0, dtype=float)

        n_params = len(p0)

        lb, ub = bounds
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (n_params,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (n_params,)).copy()

        circular_params = Alignment._normalize_circular_params(circular_params, n_params, lb, ub, p0)

        if sigma is None:
            sigma_array = None
        else:
            sigma_array = np.asarray(sigma, dtype=float)

        def wrap_params(params):
            params_wrapped = np.asarray(params, dtype=float).copy()
            for idx, spec in circular_params.items():
                params_wrapped[idx] = Alignment._wrap_to_period(
                    params_wrapped[idx],
                    period=spec["period"],
                    center=spec["center"],
                )
            return params_wrapped

        def residuals(params):
            params_eval = wrap_params(params)
            y_model = Alignment.to_numpy_1d(f(xdata, *params_eval), dtype=float)
            resid = y_model - ydata
            if circular_curve_period is not None:
                resid = Alignment._wrap_to_period(
                    resid,
                    period=circular_curve_period,
                    center=circular_curve_center,
                )
            if sigma_array is not None:
                resid = resid / sigma_array
            return resid

        p0_internal = p0.copy()
        for idx, spec in circular_params.items():
            p0_internal[idx] = Alignment._wrap_to_period(
                p0_internal[idx],
                period=spec["period"],
                center=spec["center"],
            )

        for idx in range(n_params):
            if np.isfinite(lb[idx]) and p0_internal[idx] < lb[idx]:
                p0_internal[idx] = lb[idx]
            if np.isfinite(ub[idx]) and p0_internal[idx] > ub[idx]:
                p0_internal[idx] = ub[idx]

        max_nfev = maxfev if maxfev is not None else kwargs.pop("max_nfev", None)
        result = scipy_optimize.least_squares(
            residuals,
            p0_internal,
            bounds=(lb, ub),
            method=method,
            max_nfev=max_nfev,
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"circular curve fit failed: {result.message}")

        popt = wrap_params(result.x)

        _, s, vt = np.linalg.svd(result.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result.jac.shape) * s[0] if s.size else 0.0
        s = s[s > threshold]
        vt = vt[: s.size]
        pcov = np.dot(vt.T / (s ** 2), vt) if s.size else np.full((n_params, n_params), np.inf)

        if not absolute_sigma and sigma_array is not None:
            dof = max(0, len(ydata) - n_params)
            if dof > 0:
                cost = 2.0 * result.cost
                pcov *= cost / dof

        return popt, pcov

    @staticmethod
    def model_data(
        t: np.ndarray,
        C: float,
        tau: float,
        period: float,
        shift_bins: float = 0.,
        mode: str = "binned",
    ) -> np.ndarray:
        """
        Periodic mono-exponential decay model.

        Units:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau`` is in nanoseconds.
        - ``C`` is the model amplitude.
        - ``shift_bins`` is applied in histogram-bin units.
        - ``mode`` selects either the center-sampled or bin-integrated model.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        C_norm = float(C)
        tau_ns = float(tau)
        period_ns = float(period)
        mode = str(mode).lower()
        shift_ns = float(shift_bins * (period_ns / len(t_ns)))

        # Center the decay on the middle bin before applying the sub-bin shift.
        t_local_ns = t_ns - shift_ns - period_ns - (period_ns / 2)

        if mode == "binned":
            if len(t_ns) < 2:
                raise ValueError("mode='binned' requires at least two time samples")

            dt_ns = float(t_ns[1] - t_ns[0])
            if not np.allclose(np.diff(t_ns), dt_ns):
                raise ValueError("mode='binned' requires uniformly spaced t values")

            t_start_ns = t_local_ns - 0.5 * dt_ns
            t_end_ns = t_start_ns + dt_ns
            u0_ns = np.mod(t_start_ns, period_ns)
            u1_ns = u0_ns + dt_ns
            denom = 1 - np.exp(-period_ns / tau_ns)

            model_hist = np.empty_like(t_ns, dtype=float)
            same_period = u1_ns <= period_ns

            model_hist[same_period] = (
                tau_ns
                * (
                    np.exp(-u0_ns[same_period] / tau_ns)
                    - np.exp(-u1_ns[same_period] / tau_ns)
                )
                / denom
            )

            if np.any(~same_period):
                wrapped_u1_ns = u1_ns[~same_period] - period_ns
                first_leg = (
                    tau_ns
                    * (
                        np.exp(-u0_ns[~same_period] / tau_ns)
                        - np.exp(-period_ns / tau_ns)
                    )
                    / denom
                )
                second_leg = (
                    tau_ns
                    * (
                        1.0 - np.exp(-wrapped_u1_ns / tau_ns)
                    )
                    / denom
                )
                model_hist[~same_period] = first_leg + second_leg
        elif mode == "sampled":
            model_hist = (
                np.exp(-(np.mod(t_local_ns, period_ns)) / tau_ns)
                / (1 - np.exp(-period_ns / tau_ns))
            )
        else:
            raise ValueError(f"Unsupported model_data mode: {mode}. Supported sampled, binned")

        model_hist = C_norm * model_hist / model_hist.sum()

        return model_hist

    @staticmethod
    def rectangular_IRF(t, dt):
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        dt_ns = float(dt)
        offset_ns = (t_ns.max() - t_ns.min()) / 2
        return np.where((t_ns >= offset_ns - dt_ns) & (t_ns <= offset_ns + dt_ns), 1.0, 0.0)

    @staticmethod
    def pad_tensor(x: torch.Tensor, pad_left: int, pad_right: int, dim: int, mode: str = "reflect"):
        Alignment._require_torch()
        if pad_left == 0 and pad_right == 0:
            return x

        length = x.shape[dim]

        if mode == "reflect":
            left_idx = torch.arange(pad_left, 0, -1, device=x.device)
            right_idx = torch.arange(length - 2, length - pad_right - 2, -1, device=x.device)
        elif mode == "replicate":
            left_idx = torch.zeros(pad_left, dtype=torch.long, device=x.device)
            right_idx = torch.full((pad_right,), length - 1, dtype=torch.long, device=x.device)
        elif mode == "constant":
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_left + pad_right
            constant_pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            return torch.cat(
                [
                    constant_pad.narrow(dim, 0, pad_left),
                    x,
                    constant_pad.narrow(dim, pad_left, pad_right),
                ],
                dim=dim,
            )
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")

        pad_left_tensor = x.index_select(dim, left_idx)
        pad_right_tensor = x.index_select(dim, right_idx)
        return torch.cat([pad_left_tensor, x, pad_right_tensor], dim=dim)

    @staticmethod
    def median_filter(x: torch.Tensor, window_size=3, dims=None, mode="reflect"):
        Alignment._require_torch()
        if dims is None:
            dims = list(range(x.ndim))

        if isinstance(window_size, int):
            window_size = [window_size] * len(dims)
        elif len(window_size) != len(dims):
            raise ValueError("window_size must be scalar or match len(dims)")

        for w in window_size:
            if w % 2 == 0:
                raise ValueError(f"All window sizes must be odd, got {w}")

        out = x
        for d, w in zip(dims, window_size):
            pad_left = (w - 1) // 2
            pad_right = w // 2
            out = Alignment.pad_tensor(out, pad_left, pad_right, d, mode=mode)
            out = out.unfold(d, w, 1).median(dim=-1).values

        return out

    @staticmethod
    def partial_convolution_fft(volume: torch.Tensor, kernel: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                                axis: str = 'jk', fourier: tuple = (False, False)):
        Alignment._require_torch()
        dim3 = dim1 + dim2
        dim3 = ''.join(sorted(set(dim3), key=dim3.index))

        dims = [dim1, dim2, dim3]
        axis_list = [[d.find(c) for c in axis] for d in dims]

        volume_fft = fftn(volume, dim=axis_list[0]) if not fourier[0] else volume
        kernel_fft = fftn(kernel, dim=axis_list[1]) if not fourier[1] else kernel

        conv = torch.einsum(f'{dim1},{dim2}->{dim3}', volume_fft, kernel_fft)
        conv = ifftn(conv, dim=axis_list[2])
        conv = ifftshift(conv, dim=axis_list[2])
        return torch.real(conv)

    @staticmethod
    def IRF_from_data_deconvolution(ref_data, t, C_R, tau_R, period, iterations=30, eps=1e-8, regularization=3):
        """
        Estimate the IRF from a reference histogram.

        Units:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau_R`` is in nanoseconds.
        - ``C_R`` is the reference-model amplitude.
        - ``period`` is the excitation period in the same time units as ``t``.
        """
        Alignment._require_torch()
        ref_hist = Alignment.to_torch_1d(ref_data, dtype=torch.float64)
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        C_ref = float(C_R)
        tau_ref_ns = float(tau_R)
        period_ns = float(period)

        irf_est = torch.ones_like(ref_hist)

        kernel = Alignment.to_torch_1d(Alignment.model_data(t=t_ns, C=C_ref, tau=tau_ref_ns, period=period_ns))

        kernel_t = kernel.clone().flip(0)

        kernel = fftn(kernel, dim=0)
        kernel_t = fftn(kernel_t, dim=0)

        y = torch.clamp(ref_hist, min=0)

        for _ in range(iterations):
            conv = Alignment.partial_convolution_fft(irf_est, kernel, dim1="t", dim2="t", axis="t", fourier=(False, True))
            conv = torch.clamp(conv, min=eps)
            relative_blur = y / conv
            correction = Alignment.partial_convolution_fft(
                relative_blur, kernel_t, dim1='t', dim2='t', axis='t', fourier=(0, 1)
            )
            irf_est = irf_est * correction
            irf_est = torch.clamp(irf_est, min=0)
            if regularization > 1:
                irf_est = Alignment.median_filter(irf_est, window_size=regularization, dims=[0], mode='replicate')

        return irf_est

    @staticmethod
    def linear_shift(data, shift_value, cyclic=True):
        xp = np.arange(0, data.shape[0])
        fp = data.copy()
        x = np.arange(0, data.shape[0]) - shift_value
        if cyclic:
            x = np.mod(x, data.shape[0])
        return np.interp(x, xp, fp)

    @staticmethod
    def fit_data_with_ref_or_irf(
        t,
        data,
        period,
        ref=None,
        tau_ref=None,
        irf=None,
        C_ref=1.0,
        irf_output="original",
        shift_output=None,
        fit_type="likelihood",
        mode="irf_shift",
        initial_tau=None,
        initial_dT=None,
        initial_C=None,
        force_C_normalized=False,
        irf_iterations=30,
        eps=1e-8,
        regularization=3,
    ):
        """
        Fit ``data`` using either ``ref`` + ``tau_ref`` or a directly provided ``irf``.

        Parameters
        ----------
        t, data, period :
            Same units and meaning as in ``perform_fit_data``.
        ref : array-like, optional
            Reference decay used to estimate the IRF through
            ``IRF_from_data_deconvolution``. When ``irf`` is not given,
            ``tau_ref`` can be provided explicitly or estimated from ``ref``.
        tau_ref : float or str, optional
            Lifetime of ``ref`` in nanoseconds. If omitted, it is estimated
            from ``ref`` with ``estimate_lifetime_from_log``. String selectors
            are also accepted:
            - ``"circmean"`` uses ``estimate_lifetime_from_circmean``
            - ``"birfi"`` uses ``estimate_lifetime_from_birfi``
            - ``"log"`` uses ``estimate_lifetime_from_log``
        irf : array-like, optional
            Directly provided IRF. When this is given, ``tau_ref`` is ignored.
        C_ref : float, default 1.0
            Reference amplitude used during IRF estimation from ``ref``.
        fit_type : {"likelihood", "curve_fit_circular", "curve_fit"}, default "likelihood"
            Fitting backend used by ``perform_fit_data``. ``"likelihood"``
            uses Poisson likelihood/deviance residuals and is recommended
            for low-count per-pixel histograms. ``"curve_fit_circular"`` keeps
            the circular-parameter sigma-weighted least-squares behavior, while
            ``"curve_fit"`` uses SciPy's standard ``curve_fit``.
        irf_output : {"original", "shifted"}, default "original"
            Controls the returned ``irf`` in the result dictionary.
            - ``"original"`` returns the estimated IRF (if ``ref`` was used) or
              the provided IRF (if ``irf`` was used).
            - ``"shifted"`` returns that IRF after ``linear_shift(..., dT)``.
        shift_output : {None, "ref", "reference", "data"}, default None
            Optionally returns an additional shifted histogram:
            - ``"ref"`` / ``"reference"`` returns ``ref_shifted`` using ``+dT``.
            - ``"data"`` returns ``data_shifted`` using ``-dT``.

        Returns
        -------
        dict
            Result dictionary with at least:
            - ``C``: fitted normalized amplitude
            - ``tau_ref``: reference lifetime actually used for IRF estimation,
              in nanoseconds, or ``None`` when no reference lifetime was used
            - ``dT``: fitted shift in bins
            - ``dT_ns``: fitted shift in nanoseconds
            - ``tau``: fitted lifetime in nanoseconds
            - ``irf``: returned IRF according to ``irf_output``
            - ``fit``: fitted histogram
            - ``cov``: covariance matrix from ``perform_fit_data``
            - ``irf_source``: ``"estimated_from_ref"`` or ``"provided"``
            When requested, the dictionary also includes ``ref_shifted`` or
            ``data_shifted``.

            All histogram outputs returned by this helper are normalized to
            unit sum. This function does not return unnormalized ``data``,
            ``ref``, or ``irf`` histograms.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        data_hist = Alignment.to_numpy_1d(data, dtype=float)

        if len(t_ns) == 0:
            raise ValueError("t must contain at least one sample")
        if len(t_ns) != len(data_hist):
            raise ValueError("t and data must have the same 1D length")

        period_ns = float(period)
        dt_ns = float(period_ns / len(t_ns))

        data_hist_norm = Alignment._normalize_histogram_1d(data_hist, name="data")

        if (ref is None) == (irf is None):
            raise ValueError("provide exactly one of ref or irf")

        irf_output = str(irf_output).lower()
        if irf_output in {"estimated", "input", "provided", "unshifted"}:
            irf_output = "original"
        if irf_output not in {"original", "shifted"}:
            raise ValueError("irf_output must be 'original' or 'shifted'")

        if shift_output is not None:
            shift_output = str(shift_output).lower()
        if shift_output == "reference":
            shift_output = "ref"
        if shift_output not in {None, "ref", "data"}:
            raise ValueError("shift_output must be None, 'ref', 'reference', or 'data'")

        used_tau_ref_ns = None

        if irf is None:
            ref_hist = Alignment.to_numpy_1d(ref, dtype=float)
            if len(ref_hist) != len(t_ns):
                raise ValueError("t and ref must have the same 1D length")

            if tau_ref is None:
                tau_ref_mode = "log"
            elif isinstance(tau_ref, str):
                tau_ref_mode = tau_ref.strip().lower()
            else:
                tau_ref_mode = None

            if tau_ref_mode is None:
                tau_ref_ns = float(tau_ref)
            elif tau_ref_mode in {"log", "estimate_lifetime_from_log"}:
                tau_ref_result = estimate_lifetime_from_log(
                    data_hist=ref_hist,
                    t_ns=t_ns,
                    dt_ns=dt_ns,
                    nbin=len(t_ns),
                    period_ns=period_ns,
                )
                tau_ref_ns = tau_ref_result[0]
            elif tau_ref_mode in {"circmean", "estimate_lifetime_from_circmean"}:
                repetition_rate_MHz = 1e3 / period_ns
                t0_ns = float((int(np.argmax(ref_hist)) + 0.5) * dt_ns)
                tau_ref_ns = estimate_lifetime_from_circmean(
                    ref_hist,
                    t0_ns=t0_ns,
                    repetition_rate_MHz=repetition_rate_MHz,
                )
            elif tau_ref_mode in {"birfi", "estimate_lifetime_from_birfi"}:
                tau_ref_ns = estimate_lifetime_from_birfi(t_ns, ref_hist)
            else:
                try:
                    tau_ref_ns = float(tau_ref)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "tau_ref must be a float, None, 'log', 'circmean', or 'birfi'"
                    ) from exc

            if tau_ref_ns is None:
                raise ValueError("unable to estimate tau_ref from ref")
            tau_ref_ns_array = np.asarray(tau_ref_ns, dtype=float)
            if tau_ref_ns_array.size != 1:
                raise ValueError("tau_ref estimator must return a scalar lifetime")
            tau_ref_ns = float(tau_ref_ns_array.reshape(-1)[0])
            if not np.isfinite(tau_ref_ns) or tau_ref_ns <= 0:
                raise ValueError("unable to estimate a valid positive tau_ref from ref")
            used_tau_ref_ns = tau_ref_ns

            ref_hist_norm = Alignment._normalize_histogram_1d(ref_hist, name="ref")
            irf_hist = np.asarray(
                Alignment.IRF_from_data_deconvolution(
                    ref_hist_norm,
                    t_ns,
                    C_ref,
                    tau_ref_ns,
                    period_ns,
                    iterations=irf_iterations,
                    eps=eps,
                    regularization=regularization,
                ),
                dtype=float,
            )
            irf_source = "estimated_from_ref"
        else:
            ref_hist_norm = None
            irf_hist = Alignment.to_numpy_1d(irf, dtype=float)
            if len(irf_hist) != len(t_ns):
                raise ValueError("t and irf must have the same 1D length")
            irf_source = "provided"

        irf_hist_norm = Alignment._normalize_histogram_1d(irf_hist, name="irf")

        fit_result, fit_cov = Alignment.perform_fit_data(
            t_ns,
            data_hist,
            irf_hist_norm,
            period_ns,
            initial_tau=initial_tau,
            initial_dT=initial_dT,
            initial_C=initial_C,
            mode=mode,
            fit_type=fit_type,
            force_C_normalized=force_C_normalized,
        )

        dT_bins = float(fit_result["dT"])
        tau_ns = float(fit_result["tau"])
        C_value = float(fit_result["C"])
        fit_is_valid = np.isfinite(C_value) and np.isfinite(dT_bins) and np.isfinite(tau_ns)

        returned_irf = irf_hist_norm.copy()
        if irf_output == "shifted" and fit_is_valid:
            returned_irf = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(returned_irf, dT_bins, cyclic=True),
                name="shifted irf",
            )
        elif irf_output == "shifted":
            returned_irf = np.zeros_like(irf_hist_norm, dtype=float)

        if fit_is_valid:
            fitted_hist = Alignment._normalize_histogram_1d(
                Alignment.fit_model_data(
                    t_ns,
                    C_value,
                    dT_bins,
                    tau_ns,
                    irf=irf_hist_norm,
                    period=period_ns,
                    mode=mode,
                ),
                name="fit",
            )
        else:
            fitted_hist = np.zeros_like(data_hist_norm, dtype=float)

        result = {
            "C": C_value,
            "tau_ref": used_tau_ref_ns,
            "dT": dT_bins,
            "dT_ns": dT_bins * dt_ns,
            "tau": tau_ns,
            "irf": returned_irf,
            "fit": fitted_hist,
            "cov": fit_cov,
            "irf_source": irf_source,
        }

        if shift_output == "ref":
            if ref_hist_norm is None:
                raise ValueError("shift_output='ref' requires ref to be provided")
            if fit_is_valid:
                result["ref_shifted"] = Alignment._normalize_histogram_1d(
                    Alignment.linear_shift(ref_hist_norm, dT_bins, cyclic=True),
                    name="shifted ref",
                )
            else:
                result["ref_shifted"] = np.zeros_like(ref_hist_norm, dtype=float)
        elif shift_output == "data":
            if fit_is_valid:
                result["data_shifted"] = Alignment._normalize_histogram_1d(
                    Alignment.linear_shift(data_hist_norm, -dT_bins, cyclic=True),
                    name="shifted data",
                )
            else:
                result["data_shifted"] = np.zeros_like(data_hist_norm, dtype=float)

        return result

    @staticmethod
    def fit_model_data(t, C, dT, tau, irf, period, mode="irf_shift"):
        """
        Convolve the mono-exponential model with a shifted IRF.

        Units:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau`` is in nanoseconds.
        - ``dT`` is in histogram bins. In ``"irf_shift"`` mode it shifts the
          IRF; in ``"model_shift"`` mode it shifts the mono-exponential model.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)
        C_norm = float(C)
        dT_bins = float(dT)
        tau_ns = float(tau)
        period_ns = float(period)

        if mode == "model_shift":
            pure_model_hist = Alignment.model_data(
                t=t_ns,
                C=C_norm,
                tau=tau_ns,
                period=period_ns,
                shift_bins=dT_bins,
            )
            fit_irf_hist = irf_hist
        elif mode == "irf_shift":
            pure_model_hist = Alignment.model_data(
                t=t_ns,
                C=C_norm,
                tau=tau_ns,
                period=period_ns,
            )
            fit_irf_hist = shift(irf_hist, dT_bins, order=1, mode="grid-wrap")
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported model_shift, irf_shift")

        pure_model_hist = pure_model_hist / pure_model_hist.sum()
        fit_irf_hist = fit_irf_hist / fit_irf_hist.sum()

        pure_model_hist = Alignment.to_torch_1d(pure_model_hist)
        fit_irf_hist = Alignment.to_torch_1d(fit_irf_hist)
        return Alignment.partial_convolution_fft(
            pure_model_hist,
            fit_irf_hist,
            dim1="t",
            dim2="t",
            axis="t",
            fourier=(0, 0),
        )

    @staticmethod
    def estimate_peak_dT_bins(data, irf):
        """
        Estimate a direct circular shift seed from the data/IRF peak locations.

        The returned shift is wrapped to the public ``[-nbin/2, nbin/2)`` bin
        convention used by the fitting helpers.
        """
        data_hist = Alignment.to_numpy_1d(data, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)

        if len(data_hist) != len(irf_hist):
            raise ValueError("data and irf must have the same 1D length")

        nbin = len(data_hist)
        data_peak_bin = int(np.argmax(data_hist))
        irf_peak_bin = int(np.argmax(irf_hist))

        return float(
            Alignment._wrap_to_period(
                data_peak_bin - irf_peak_bin,
                period=float(nbin),
                center=0.0,
            )
        )

    @staticmethod
    def _nan_fit_result():
        return {
            "C": np.nan,
            "dT": np.nan,
            "tau": np.nan,
        }, np.full((3, 3), np.nan, dtype=float)

    @staticmethod
    def _skip_fit_with_warning(histogram_name):
        warnings.warn(
            f"{histogram_name} histogram has a non-positive or non-finite sum; "
            "skipping fit and returning NaNs",
            RuntimeWarning,
            stacklevel=3,
        )
        return Alignment._nan_fit_result()

    @staticmethod
    def _canonical_fit_type(fit_type):
        fit_type_aliases = {
            "likelihood": "likelihood",
            "poisson": "likelihood",
            "poisson_likelihood": "likelihood",
            "poisson-likelihood": "likelihood",
            "poisson_deviance": "likelihood",
            "mle": "likelihood",
            "curve_fit_circular": "curve_fit_circular",
            "curve-fit-circular": "curve_fit_circular",
            "circular_curve_fit": "curve_fit_circular",
            "circular": "curve_fit_circular",
            "curve_fit": "curve_fit",
            "curve-fit": "curve_fit",
            "weighted_ls": "curve_fit_circular",
            "weighted-least-squares": "curve_fit_circular",
            "weighted_least_squares": "curve_fit_circular",
            "neyman": "curve_fit_circular",
            "chi_square": "curve_fit_circular",
            "chisquare": "curve_fit_circular",
        }
        fit_type = str(fit_type).strip().lower()
        if fit_type not in fit_type_aliases:
            raise ValueError(
                "Unsupported fit_type: "
                f"{fit_type}. Supported likelihood, curve_fit_circular, curve_fit"
            )
        return fit_type_aliases[fit_type]

    @staticmethod
    def _fit_initial_guess(initial_C, initial_dT, initial_tau):
        initial_guess = [1.0, 0.0, 1.0]
        if initial_C is not None:
            initial_guess[0] = initial_C
        if initial_dT is not None:
            initial_guess[1] = initial_dT
        if initial_tau is not None:
            initial_guess[2] = initial_tau
        return initial_guess

    @staticmethod
    def _prepare_fit_irf(data_hist_norm, irf_hist_norm, initial_dT):
        """
        Optionally pre-shift the IRF so the optimizer fits a small residual dT.
        """
        if initial_dT is not None:
            return irf_hist_norm, initial_dT, None

        dT_seed_bins = Alignment.estimate_peak_dT_bins(data_hist_norm, irf_hist_norm)
        fit_irf_hist_norm = np.asarray(
            shift(irf_hist_norm, dT_seed_bins, order=1, mode="grid-wrap"),
            dtype=float,
        )
        fit_irf_hist_norm = fit_irf_hist_norm / fit_irf_hist_norm.sum()
        return fit_irf_hist_norm, 0.0, dT_seed_bins

    @staticmethod
    def _full_fit_bounds(nbin, tau_lower_bound, tau_upper_bound):
        return (
            [0.0, -nbin / 2, tau_lower_bound],
            [np.inf, nbin / 2, tau_upper_bound],
        )

    @staticmethod
    def _fixed_c_fit_bounds(nbin, tau_lower_bound, tau_upper_bound):
        return (
            [-nbin / 2, tau_lower_bound],
            [nbin / 2, tau_upper_bound],
        )

    @staticmethod
    def _expand_fixed_c_fit(popt_fixed_c, cov_fixed_c):
        popt = np.array([1.0, popt_fixed_c[0], popt_fixed_c[1]], dtype=float)
        cov = np.full((3, 3), np.nan, dtype=float)
        cov[1:, 1:] = cov_fixed_c
        return popt, cov

    @staticmethod
    def _normalized_model_probability(fit_model_numpy, t_ns, dT_bins, tau_ns):
        model_hist = fit_model_numpy(t_ns, 1.0, dT_bins, tau_ns)
        model_hist = np.asarray(model_hist, dtype=float)
        model_hist = np.clip(model_hist, 1e-15, None)
        model_sum = model_hist.sum()
        if not np.isfinite(model_sum) or model_sum <= 0:
            return None
        return model_hist / model_sum

    @staticmethod
    def _poisson_deviance_residual(observed_counts, model_counts):
        observed_counts = np.asarray(observed_counts, dtype=float)
        model_counts = np.clip(np.asarray(model_counts, dtype=float), 1e-12, None)

        deviance = model_counts.copy()
        positive_counts = observed_counts > 0
        deviance[positive_counts] = (
            observed_counts[positive_counts]
            * np.log(observed_counts[positive_counts] / model_counts[positive_counts])
            - (observed_counts[positive_counts] - model_counts[positive_counts])
        )
        deviance = np.maximum(deviance, 0.0)
        return np.sign(observed_counts - model_counts) * np.sqrt(2.0 * deviance)

    @staticmethod
    def _covariance_from_least_squares(result, n_observations, n_params, scale_by_cost=False):
        _, s, vt = np.linalg.svd(result.jac, full_matrices=False)
        threshold = (
            np.finfo(float).eps * max(result.jac.shape) * s[0]
            if s.size
            else 0.0
        )
        s = s[s > threshold]
        vt = vt[: s.size]
        pcov = (
            np.dot(vt.T / (s ** 2), vt)
            if s.size
            else np.full((n_params, n_params), np.inf)
        )
        if scale_by_cost:
            dof = max(0, n_observations - n_params)
            if dof > 0:
                pcov *= 2.0 * result.cost / dof
        return pcov

    @staticmethod
    def _run_poisson_fit(
        fit_model_numpy,
        t_ns,
        data_hist,
        data_sum,
        initial_guess,
        nbin,
        tau_lower_bound,
        force_C_normalized,
    ):
        tau_upper_bound = t_ns.max()

        if force_C_normalized:
            def residual(params):
                dT_bins, tau_ns = params
                model_probability = Alignment._normalized_model_probability(
                    fit_model_numpy,
                    t_ns,
                    dT_bins,
                    tau_ns,
                )
                if model_probability is None:
                    return np.full_like(data_hist, 1e12, dtype=float)
                return Alignment._poisson_deviance_residual(
                    observed_counts = data_hist,
                    model_counts = data_sum * model_probability,
                )

            p0 = initial_guess[1:]
            bounds = Alignment._fixed_c_fit_bounds(nbin, tau_lower_bound, tau_upper_bound)
            result = scipy_optimize.least_squares(
                residual,
                p0,
                bounds=bounds,
                max_nfev=600000,
            )
            if not result.success:
                raise RuntimeError(f"poisson fit failed: {result.message}")

            cov_fixed_c = Alignment._covariance_from_least_squares(
                result,
                n_observations=len(data_hist),
                n_params=len(p0),
                scale_by_cost=False,
            )
            return Alignment._expand_fixed_c_fit(result.x, cov_fixed_c)

        def residual(params):
            C_norm, dT_bins, tau_ns = params
            model_probability = Alignment._normalized_model_probability(
                fit_model_numpy,
                t_ns,
                dT_bins,
                tau_ns,
            )
            if model_probability is None:
                return np.full_like(data_hist, 1e12, dtype=float)
            model_counts = data_sum * np.clip(C_norm, 1e-12, None) * model_probability
            return Alignment._poisson_deviance_residual(
                observed_counts = data_hist,
                model_counts = model_counts)

        bounds = Alignment._full_fit_bounds(nbin, tau_lower_bound, tau_upper_bound)
        result = scipy_optimize.least_squares(
            residual,
            initial_guess,
            bounds=bounds,
            max_nfev=600000,
        )
        if not result.success:
            raise RuntimeError(f"poisson fit failed: {result.message}")

        cov = Alignment._covariance_from_least_squares(
            result,
            n_observations=len(data_hist),
            n_params=len(initial_guess),
            scale_by_cost=False,
        )
        return result.x, cov

    @staticmethod
    def _run_curve_fit(
        fit_type,
        model_function,
        t_ns,
        data_hist_norm,
        sigma,
        p0,
        bounds,
        circular_params,
        circular_curve_period=None,
    ):
        if fit_type == "curve_fit_circular":
            return Alignment.curve_fit_circular(
                model_function,
                t_ns,
                data_hist_norm,
                sigma=sigma,
                p0=p0,
                bounds=bounds,
                circular_params=circular_params,
                circular_curve_period=circular_curve_period,
                maxfev=600000,
            )
        if fit_type == "curve_fit":
            return scipy_optimize.curve_fit(
                model_function,
                t_ns,
                data_hist_norm,
                sigma=sigma,
                p0=p0,
                bounds=bounds,
                maxfev=600000,
            )
        raise ValueError(
            f"Unsupported fit_type: {fit_type}. Supported curve_fit_circular, curve_fit"
        )

    @staticmethod
    def _run_weighted_least_squares_fit(
        fit_model_numpy,
        t_ns,
        data_hist,
        data_hist_norm,
        data_sum,
        initial_guess,
        fit_type,
        nbin,
        tau_lower_bound,
        force_C_normalized,
    ):
        sigma = np.sqrt(np.clip(data_hist, 1.0, None)) / data_sum
        tau_upper_bound = t_ns.max()

        if force_C_normalized:
            def fit_model_fixed_c(t_ns_fit, dT_bins, tau_ns):
                return fit_model_numpy(t_ns_fit, 1.0, dT_bins, tau_ns)

            p0 = initial_guess[1:]
            bounds = Alignment._fixed_c_fit_bounds(nbin, tau_lower_bound, tau_upper_bound)
            popt_fixed_c, cov_fixed_c = Alignment._run_curve_fit(
                fit_type,
                fit_model_fixed_c,
                t_ns,
                data_hist_norm,
                sigma,
                p0,
                bounds,
                circular_params={0: float(nbin)},
                circular_curve_period=float(nbin),
            )
            return Alignment._expand_fixed_c_fit(popt_fixed_c, cov_fixed_c)

        bounds = Alignment._full_fit_bounds(nbin, tau_lower_bound, tau_upper_bound)
        return Alignment._run_curve_fit(
            fit_type,
            fit_model_numpy,
            t_ns,
            data_hist_norm,
            sigma,
            initial_guess,
            bounds,
            circular_params={1: float(nbin)},
        )

    @staticmethod
    def _restore_seeded_dT(popt, dT_seed_bins, nbin):
        if dT_seed_bins is None:
            return popt

        popt = np.asarray(popt, dtype=float).copy()
        popt[1] = float(
            Alignment._wrap_to_period(
                dT_seed_bins + popt[1],
                period=float(nbin),
                center=0.0,
            )
        )
        return popt

    @staticmethod
    def perform_fit_data(
        t,
        data,
        irf,
        period,
        initial_tau=None,
        initial_dT=None,
        initial_C=None,
        irf_min=1e-5,
        mode="irf_shift",
        fit_type="likelihood",
        force_C_normalized=False,
    ):
        """
        Fit ``data`` with ``fit_model_data``.

        Unit contract:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau`` / ``initial_tau`` are in nanoseconds.
        - ``dT`` / ``initial_dT`` are in bins.
        - ``fit_type`` can be ``"likelihood"``, ``"curve_fit_circular"``, or
          ``"curve_fit"``. ``"likelihood"`` uses a Poisson likelihood/deviance
          fit and is recommended for per-pixel low-count histograms because it
          avoids the downward lifetime bias of observed-count chi-square
          weights. The curve-fit modes keep the historical sigma-weighted
          least-squares behavior.
        - ``force_C_normalized=True`` keeps ``C`` fixed to ``1.0`` and only
          fits ``dT`` and ``tau``.
        - when ``initial_dT is None``, the fitter first pre-shifts the IRF by a
          direct peak-based seed and then fits only the residual shift around
          zero. This avoids the circular-boundary trap near ``+-nbin/2`` while
          keeping the public returned ``dT`` in the same convention.
        - when ``initial_dT`` is provided, it is used directly as the initial
          guess for the public ``dT`` parameter.
        - returned ``C`` is a normalized 0..1 amplitude because the data is
          normalized by its sum and the IRF by its sum.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        data_hist = Alignment.to_numpy_1d(data, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)

        if len(t_ns) != len(data_hist) or len(t_ns) != len(irf_hist):
            raise ValueError("t, data, and irf must have the same 1D length")

        data_sum = data_hist.sum()
        irf_sum = irf_hist.sum()

        if not np.isfinite(data_sum) or data_sum <= 0:
            return Alignment._skip_fit_with_warning("data")
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            return Alignment._skip_fit_with_warning("irf")

        Alignment._require_scipy_optimize()

        nbin = len(t_ns)
        tau_lower_bound = float(irf_min)
        if tau_lower_bound <= 0:
            raise ValueError("irf_min must be positive")

        fit_type = Alignment._canonical_fit_type(fit_type)

        data_hist_norm = data_hist / data_sum
        irf_hist_norm = irf_hist / irf_sum
        fit_irf_hist_norm, fit_initial_dT, dT_seed_bins = Alignment._prepare_fit_irf(
            data_hist_norm,
            irf_hist_norm,
            initial_dT,
        )

        def fit_model_numpy(t_ns_fit, C_norm, dT_bins, tau_ns):
            model_hist = Alignment.fit_model_data(
                t_ns_fit,
                C_norm,
                dT_bins,
                tau_ns,
                irf=fit_irf_hist_norm,
                period=period,
                mode=mode,
            )
            return Alignment.to_numpy_1d(model_hist, dtype=float)

        initial_guess = Alignment._fit_initial_guess(
            initial_C,
            fit_initial_dT,
            initial_tau,
        )

        if fit_type == "likelihood":
            popt, cov = Alignment._run_poisson_fit(
                fit_model_numpy,
                t_ns,
                data_hist,
                data_sum,
                initial_guess,
                nbin,
                tau_lower_bound,
                force_C_normalized,
            )
        else:
            popt, cov = Alignment._run_weighted_least_squares_fit(
                fit_model_numpy,
                t_ns,
                data_hist,
                data_hist_norm,
                data_sum,
                initial_guess,
                fit_type,
                nbin,
                tau_lower_bound,
                force_C_normalized,
            )

        popt = Alignment._restore_seeded_dT(popt, dT_seed_bins, nbin)
        return {"C": popt[0], "dT": popt[1], "tau": popt[2]}, cov

    @staticmethod
    def _fit_map_covariance_errors(covariance):
        covariance = np.asarray(covariance, dtype=float)
        errors = np.full(3, np.nan, dtype=float)
        if covariance.shape != (3, 3):
            return errors

        diag = np.diag(covariance)
        valid = np.isfinite(diag) & (diag >= 0)
        errors[valid] = np.sqrt(diag[valid])
        return errors

    @staticmethod
    def _fit_map_one_pixel(
        flat_idx,
        hist,
        nx,
        t,
        irf,
        period,
        initial_tau,
        initial_dT,
        initial_C,
        mode,
        fit_type,
        force_C_normalized,
        catch_exceptions,
    ):
        y = int(flat_idx // nx)
        x = int(flat_idx % nx)
        hist = np.asarray(hist, dtype=float)

        try:
            fit_res, cov = Alignment.perform_fit_data(
                t=t,
                data=hist,
                irf=irf,
                period=period,
                initial_tau=initial_tau,
                initial_dT=initial_dT,
                initial_C=initial_C,
                mode=mode,
                fit_type=fit_type,
                force_C_normalized=force_C_normalized,
            )
            errors = Alignment._fit_map_covariance_errors(cov)
            return (
                y,
                x,
                float(fit_res["C"]),
                float(fit_res["dT"]),
                float(fit_res["tau"]),
                float(errors[0]),
                float(errors[1]),
                float(errors[2]),
            )
        except Exception:
            if not catch_exceptions:
                raise
            return (y, x, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    @staticmethod
    def generate_fit_maps(
        data,
        irf,
        t,
        period,
        initial_tau=None,
        initial_dT=None,
        initial_C=None,
        mode="irf_shift",
        fit_type="likelihood",
        force_C_normalized=True,
        min_counts=0.0,
        valid_mask=None,
        n_jobs=1,
        backend="loky",
        show_progress=True,
        catch_exceptions=True,
    ):
        """
        Fit every pixel histogram in a ``(y, x, t)`` image and return fit maps.

        Returns a dictionary with ``C``, ``dT``, ``tau``, ``C_err``,
        ``dT_err``, and ``tau_err`` maps. Invalid or failed pixels are filled
        with NaNs.
        """
        data_array = np.asarray(data, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)
        t_ns = Alignment.to_numpy_1d(t, dtype=float)

        if data_array.ndim != 3:
            raise ValueError(f"data must have shape (ny, nx, nbins), got {data_array.shape}")

        ny, nx, nbins = data_array.shape
        if irf_hist.shape != (nbins,):
            raise ValueError(f"irf must have shape ({nbins},), got {irf_hist.shape}")
        if t_ns.shape != (nbins,):
            raise ValueError(f"t must have shape ({nbins},), got {t_ns.shape}")
        if not np.all(np.isfinite(t_ns)):
            raise ValueError("t contains non-finite values")
        if not np.all(np.isfinite(irf_hist)) or np.sum(irf_hist) <= 0:
            raise ValueError("irf contains non-finite values or has non-positive sum")

        data_2d = data_array.reshape(-1, nbins)
        pixel_is_valid = (
            np.all(np.isfinite(data_2d), axis=1)
            & (np.sum(data_2d, axis=1) > float(min_counts))
        )

        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape == (ny, nx):
                valid_mask = valid_mask.ravel()
            elif valid_mask.shape != (ny * nx,):
                raise ValueError(
                    f"valid_mask must have shape {(ny, nx)} or {(ny * nx,)}, got {valid_mask.shape}"
                )
            pixel_is_valid &= valid_mask

        valid_indices = np.flatnonzero(pixel_is_valid)

        fit_maps = {
            "C": np.full((ny, nx), np.nan, dtype=float),
            "dT": np.full((ny, nx), np.nan, dtype=float),
            "tau": np.full((ny, nx), np.nan, dtype=float),
            "C_err": np.full((ny, nx), np.nan, dtype=float),
            "dT_err": np.full((ny, nx), np.nan, dtype=float),
            "tau_err": np.full((ny, nx), np.nan, dtype=float),
        }

        if valid_indices.size == 0:
            return fit_maps

        progress = tqdm if show_progress else (lambda x, **kwargs: x)
        worker_kwargs = dict(
            nx=nx,
            t=t_ns,
            irf=irf_hist,
            period=float(period),
            initial_tau=initial_tau,
            initial_dT=initial_dT,
            initial_C=initial_C,
            mode=mode,
            fit_type=fit_type,
            force_C_normalized=force_C_normalized,
            catch_exceptions=catch_exceptions,
        )

        if int(n_jobs) == 1:
            results = [
                Alignment._fit_map_one_pixel(int(idx), data_2d[int(idx)], **worker_kwargs)
                for idx in progress(valid_indices, desc="Fitting pixels")
            ]
        else:
            try:
                from joblib import Parallel, delayed
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("joblib is required when n_jobs is not 1") from exc

            results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
                delayed(Alignment._fit_map_one_pixel)(int(idx), data_2d[int(idx)], **worker_kwargs)
                for idx in progress(valid_indices, desc="Fitting pixels")
            )

        for y, x, C, dT, tau, C_err, dT_err, tau_err in results:
            fit_maps["C"][y, x] = C
            fit_maps["dT"][y, x] = dT
            fit_maps["tau"][y, x] = tau
            fit_maps["C_err"][y, x] = C_err
            fit_maps["dT_err"][y, x] = dT_err
            fit_maps["tau_err"][y, x] = tau_err

        return fit_maps

    @staticmethod
    def fit_maps_to_stack(fit_maps, names=("C", "dT", "tau", "C_err", "dT_err", "tau_err")):
        """Return a stack and name list from a fit-map dictionary."""
        names = list(names)
        return np.stack([np.asarray(fit_maps[name], dtype=float) for name in names], axis=0), names

    @staticmethod
    def phasor_delay_from_hist(hist, period_ns, harmonic=1):
        hist = Alignment.to_numpy_1d(hist, dtype=float)
        phasor_value = np.fft.fft(hist / hist.sum())[harmonic].conj()
        phase_rad = np.mod(np.angle(phasor_value), 2 * np.pi)
        delay_ns = phase_rad / (2 * np.pi) * period_ns
        return phasor_value, phase_rad, delay_ns

    @staticmethod
    def hist_for_plot(hist):
        return Alignment.to_numpy_1d(hist)


    @staticmethod
    def sum_channel_applying_shifts(data, shifts_array, axis=(0, 1, 2, 3), reverse_shifts=True):
        """
        Apply fractional cyclic shifts to the channel dimension of histogram data
        and sum all channels, conserving total counts.

        Parameters
        ----------
        data : ndarray
            Input array with shape:
                (rep, z, y, x, bin, ch)
            Example:
                (1, 1, 501, 501, 91, 25)

            - rep: repetitions
            - z, y, x: spatial dimensions
            - bin: histogram bins (e.g. 91)
            - ch: channels to be shifted and summed (e.g. 25)

        shifts_array : ndarray
            Shape: (ch,)
            Fractional shifts (in bin units) applied to each channel.

        axis : int, tuple of int, or None, default (0, 1, 2, 3)
            Axes to sum after applying the shifts and summing the channel axis.
            Axes refer to the output array before this final sum, i.e. the
            input shape without the last channel axis: (rep, z, y, x, bin).
            Use ``axis=()`` or ``axis=None`` to keep all non-channel axes.

        Method
        ------
        - Each histogram along the 'bin' axis is shifted by a fractional amount.
        - Shifts are applied using **conservative redistribution**:
            each bin contributes to two neighboring bins with weights:
                (1 - alpha) and alpha
        - Indices are wrapped modulo n_bins → **cyclic behavior**
        - All channels are then summed.
        - The selected output axes are summed if ``axis`` is not empty.

        Properties
        ----------
        - ✔ Exact conservation of total counts (photons)
        - ✔ No negative values introduced
        - ✔ Sub-bin precision (fractional shifts)
        - ✔ Cyclic boundary conditions

        Implementation details
        ----------------------
        - Leading dimensions (rep, z, y, x) are flattened for batch processing.
        - All shifts are computed vectorially across channels.
        - Accumulation uses np.add.at (scatter-add).
        - A loop over flattened batches remains (NumPy limitation).

        Returns
        -------
        out : ndarray
            Output array with the channel axis removed and with ``axis`` summed.
            With the default ``axis=(0, 1, 2, 3)``, the output shape is
            ``(bin,)``. With ``axis=()`` or ``axis=None``, the output shape is
            ``(rep, z, y, x, bin)``.

        Notes
        -----
        - Positive shifts move counts toward lower bin indices.
        To invert direction, change:
            dst = i - s   -->   dst = i + s
        """

        data = np.asarray(data, dtype=float)
        shifts = np.asarray(shifts_array, dtype=float)
        if reverse_shifts:
            shifts = -shifts

        *prefix, n_bins, n_hist = data.shape
        if shifts.shape != (n_hist,):
            raise ValueError(f"shifts_array must have shape ({n_hist},), got {shifts.shape}")

        flat = data.reshape(-1, n_bins, n_hist)  # (B, bin, ch)
        B = flat.shape[0]

        i = np.arange(n_bins)[:, None]           # (bin,1)
        s = shifts[None, :]                      # (1,ch)

        dst = i - s
        j0 = np.floor(dst).astype(int)
        alpha = dst - j0
        j1 = j0 + 1

        j0 %= n_bins
        j1 %= n_bins

        # Flatten everything
        flat_data = flat.reshape(B, -1)          # (B, bin*ch)
        j0 = j0.reshape(-1)
        j1 = j1.reshape(-1)
        alpha = alpha.reshape(-1)

        w0 = (1 - alpha)
        w1 = alpha

        out = np.zeros((B, n_bins), dtype=float)

        for b in range(B):
            np.add.at(out[b], j0, w0 * flat_data[b])
            np.add.at(out[b], j1, w1 * flat_data[b])

        out = out.reshape(*prefix, n_bins)

        if axis is None:
            return out

        if np.isscalar(axis):
            axis = (int(axis),)
        else:
            axis = tuple(int(ax) for ax in axis)

        if len(axis) == 0:
            return out

        ndim = out.ndim
        normalized_axis = tuple(ax + ndim if ax < 0 else ax for ax in axis)
        invalid_axis = [ax for ax in normalized_axis if ax < 0 or ax >= ndim]
        if invalid_axis:
            raise np.exceptions.AxisError(invalid_axis[0], ndim=ndim)

        return out.sum(axis=normalized_axis)

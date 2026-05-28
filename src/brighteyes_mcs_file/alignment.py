"""Alignment, fitting, and IRF estimation utilities."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
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
        tau_reference_ns = float(tau_R)
        period_ns = float(period)

        irf_est = torch.ones_like(ref_hist)

        kernel = Alignment.to_torch_1d(Alignment.model_data(t=t_ns, C=C_ref, tau=tau_reference_ns, period=period_ns))

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
        hist = Alignment.to_numpy_1d(data, dtype=float)
        x = np.arange(hist.shape[0], dtype=float) - float(shift_value)
        if not cyclic:
            return np.interp(x, np.arange(hist.shape[0]), hist, left=0.0, right=0.0)

        j0 = np.floor(x).astype(np.intp)
        alpha = x - j0
        j1 = j0 + 1
        return (
            (1.0 - alpha) * hist[j0 % hist.shape[0]]
            + alpha * hist[j1 % hist.shape[0]]
        )

    @staticmethod
    def _model_data_binned_fast(
        t_base_ns,
        dt_ns,
        nbin,
        period_ns,
        C,
        tau,
        shift_bins=0.0,
    ):
        C_norm = float(C)
        tau_ns = float(tau)
        shift_ns = float(shift_bins) * (period_ns / nbin)
        t_local_ns = t_base_ns - shift_ns

        t_start_ns = t_local_ns - 0.5 * dt_ns
        u0_ns = np.mod(t_start_ns, period_ns)
        u1_ns = u0_ns + dt_ns
        denom = 1.0 - np.exp(-period_ns / tau_ns)

        model_hist = np.empty(nbin, dtype=float)
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
                * (1.0 - np.exp(-wrapped_u1_ns / tau_ns))
                / denom
            )
            model_hist[~same_period] = first_leg + second_leg

        return C_norm * model_hist / model_hist.sum()

    @staticmethod
    def _cyclic_fft_convolve_centered(volume, kernel, kernel_fft=None):
        volume_fft = np.fft.fft(np.asarray(volume, dtype=float))
        if kernel_fft is None:
            kernel_fft = np.fft.fft(np.asarray(kernel, dtype=float))
        conv = np.fft.ifft(volume_fft * kernel_fft)
        return np.fft.ifftshift(np.real(conv))

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
        model_fn=None,
        p0=None,
        bounds=None,
        parameter_names=None,
        param_names=None,
        model_kwargs=None,
        amplitude_param="C",
        delay_param="dT",
        lifetime_param="tau",
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
        model_fn, p0, bounds, parameter_names, model_kwargs : optional
            Optional custom full-model fit configuration forwarded to
            ``perform_fit_data``. ``model_fn`` receives
            ``(t, irf, period, *params)`` and returns the fitted histogram.

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
            - ``parameter_names``, ``param_values``, ``param_errors``, and
              ``params``: generic parameter outputs for default and custom fits
            When requested, the dictionary also includes ``ref_shifted`` or
            ``data_shifted``.

            All histogram outputs returned by this helper are normalized to
            unit sum. This function does not return unnormalized ``data``,
            ``ref``, or ``irf`` histograms.
        """
        parameter_names = Alignment._resolve_parameter_names_alias(parameter_names, param_names)
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

        used_tau_reference_ns = None

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
                tau_reference_ns = float(tau_ref)
            elif tau_ref_mode in {"log", "estimate_lifetime_from_log"}:
                tau_ref_result = estimate_lifetime_from_log(
                    data_hist=ref_hist,
                    t_ns=t_ns,
                    dt_ns=dt_ns,
                    nbin=len(t_ns),
                    period_ns=period_ns,
                )
                tau_reference_ns = tau_ref_result[0]
            elif tau_ref_mode in {"circmean", "estimate_lifetime_from_circmean"}:
                repetition_rate_MHz = 1e3 / period_ns
                t0_ns = float((int(np.argmax(ref_hist)) + 0.5) * dt_ns)
                tau_reference_ns = estimate_lifetime_from_circmean(
                    ref_hist,
                    t0_ns=t0_ns,
                    repetition_rate_MHz=repetition_rate_MHz,
                )
            elif tau_ref_mode in {"birfi", "estimate_lifetime_from_birfi"}:
                tau_reference_ns = estimate_lifetime_from_birfi(t_ns, ref_hist)
            else:
                try:
                    tau_reference_ns = float(tau_ref)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "tau_ref must be a float, None, 'log', 'circmean', or 'birfi'"
                    ) from exc

            if tau_reference_ns is None:
                raise ValueError("unable to estimate tau_ref from ref")
            tau_reference_ns_array = np.asarray(tau_reference_ns, dtype=float)
            if tau_reference_ns_array.size != 1:
                raise ValueError("tau_ref estimator must return a scalar lifetime")
            tau_reference_ns = float(tau_reference_ns_array.reshape(-1)[0])
            if not np.isfinite(tau_reference_ns) or tau_reference_ns <= 0:
                raise ValueError("unable to estimate a valid positive tau_ref from ref")
            used_tau_reference_ns = tau_reference_ns

            ref_hist_norm = Alignment._normalize_histogram_1d(ref_hist, name="ref")
            irf_hist = np.asarray(
                Alignment.IRF_from_data_deconvolution(
                    ref_hist_norm,
                    t_ns,
                    C_ref,
                    tau_reference_ns,
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
            model_fn=model_fn,
            p0=p0,
            bounds=bounds,
            parameter_names=parameter_names,
            model_kwargs=model_kwargs,
            amplitude_param=amplitude_param,
            delay_param=delay_param,
            lifetime_param=lifetime_param,
        )

        dT_bins = float(fit_result["dT"])
        C_value = float(fit_result.get("C", np.nan))
        tau_ns = float(fit_result.get("tau", np.nan))
        param_values = np.asarray(fit_result.get("param_values", []), dtype=float)
        fit_is_valid = param_values.size > 0 and np.all(np.isfinite(param_values))

        returned_irf = irf_hist_norm.copy()
        if irf_output == "shifted" and fit_is_valid and np.isfinite(dT_bins):
            returned_irf = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(returned_irf, dT_bins, cyclic=True),
                name="shifted irf",
            )
        elif irf_output == "shifted":
            returned_irf = np.zeros_like(irf_hist_norm, dtype=float)

        if fit_is_valid and fit_result.get("fit") is not None:
            fitted_hist = Alignment._normalize_histogram_1d(
                fit_result["fit"],
                name="fit",
            )
        else:
            fitted_hist = np.zeros_like(data_hist_norm, dtype=float)

        result = {
            "C": C_value,
            "tau_ref": used_tau_reference_ns,
            "dT": dT_bins,
            "dT_ns": dT_bins * dt_ns,
            "tau": tau_ns,
            "irf": returned_irf,
            "fit": fitted_hist,
            "cov": fit_cov,
            "irf_source": irf_source,
            "params": dict(fit_result.get("params", {})),
            "parameter_names": list(fit_result.get("parameter_names", [])),
            "param_names": list(fit_result.get("parameter_names", [])),
            "param_values": np.asarray(fit_result.get("param_values", []), dtype=float),
            "param_errors": np.asarray(fit_result.get("param_errors", []), dtype=float),
            "model_name": fit_result.get("model_name", Alignment._callable_name(model_fn)),
        }

        if shift_output == "ref":
            if ref_hist_norm is None:
                raise ValueError("shift_output='ref' requires ref to be provided")
            if fit_is_valid and np.isfinite(dT_bins):
                result["ref_shifted"] = Alignment._normalize_histogram_1d(
                    Alignment.linear_shift(ref_hist_norm, dT_bins, cyclic=True),
                    name="shifted ref",
                )
            else:
                result["ref_shifted"] = np.zeros_like(ref_hist_norm, dtype=float)
        elif shift_output == "data":
            if fit_is_valid and np.isfinite(dT_bins):
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
        period_ns = float(period)
        mode = str(mode)

        if len(t_ns) != len(irf_hist):
            raise ValueError("t and irf must have the same 1D length")
        if len(t_ns) < 2:
            raise ValueError("fit_model_data requires at least two time samples")
        if mode not in {"model_shift", "irf_shift"}:
            raise ValueError(
                f"Unsupported mode: {mode}. Supported model_shift, irf_shift"
            )

        dt_ns = float(t_ns[1] - t_ns[0])
        if not np.allclose(np.diff(t_ns), dt_ns):
            raise ValueError("fit_model_data requires uniformly spaced t values")

        irf_sum = irf_hist.sum()
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            raise ValueError("irf contains non-finite values or has non-positive sum")

        C_norm = float(C)
        dT_bins = float(dT)
        tau_ns = float(tau)
        nbin = len(t_ns)
        t_base_ns = t_ns - period_ns - (period_ns / 2.0)
        irf_hist = irf_hist / irf_sum

        if mode == "model_shift":
            pure_model_hist = Alignment._model_data_binned_fast(
                t_base_ns,
                dt_ns,
                nbin,
                period_ns,
                C_norm,
                tau_ns,
                shift_bins=dT_bins,
            )
            fit_irf_hist = irf_hist
            fit_irf_fft = np.fft.fft(fit_irf_hist)
        else:
            pure_model_hist = Alignment._model_data_binned_fast(
                t_base_ns,
                dt_ns,
                nbin,
                period_ns,
                C_norm,
                tau_ns,
            )
            fit_irf_hist = Alignment.linear_shift(
                irf_hist,
                dT_bins,
                cyclic=True,
            )
            fit_irf_fft = None

        pure_model_hist = pure_model_hist / pure_model_hist.sum()
        fit_irf_hist = fit_irf_hist / fit_irf_hist.sum()

        return Alignment._cyclic_fft_convolve_centered(
            pure_model_hist,
            fit_irf_hist,
            kernel_fft=fit_irf_fft,
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
    def _nan_fit_result(parameter_names=None, dt_ns=np.nan):
        if parameter_names is None:
            parameter_names = ["C", "dT", "tau"]
        parameter_names = list(parameter_names)
        values = np.full(len(parameter_names), np.nan, dtype=float)
        errors = np.full(len(parameter_names), np.nan, dtype=float)
        params = {name: value for name, value in zip(parameter_names, values)}
        result = {
            "C": params.get("C", np.nan),
            "dT": params.get("dT", np.nan),
            "dT_ns": params.get("dT", np.nan) * float(dt_ns),
            "tau": params.get("tau", np.nan),
            "params": params,
            "parameter_names": parameter_names,
            "param_names": parameter_names,
            "param_values": values,
            "param_errors": errors,
            "fit": None,
        }
        return result, np.full((len(parameter_names), len(parameter_names)), np.nan, dtype=float)

    @staticmethod
    def _skip_fit_with_warning(histogram_name, parameter_names=None, dt_ns=np.nan):
        warnings.warn(
            f"{histogram_name} histogram has a non-positive or non-finite sum; "
            "skipping fit and returning NaNs",
            RuntimeWarning,
            stacklevel=3,
        )
        return Alignment._nan_fit_result(parameter_names=parameter_names, dt_ns=dt_ns)

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
    def _resolve_parameter_names_alias(parameter_names, param_names):
        if param_names is None:
            return parameter_names
        if parameter_names is not None and list(parameter_names) != list(param_names):
            raise ValueError("parameter_names and param_names cannot disagree")
        return param_names

    @staticmethod
    def _fit_initial_guess(initial_C, initial_dT, initial_tau):
        initial_guess = [1.0, 0.0, 1.0]
        if initial_C is not None:
            initial_guess[0] = initial_C
        if initial_dT is not None:
            initial_guess[1] = initial_dT
        if initial_tau is not None:
            initial_guess[2] = initial_tau
        return np.asarray(initial_guess, dtype=float)

    @staticmethod
    def _callable_name(callable_obj):
        if callable_obj is None:
            return "single_exponential"
        return getattr(
            callable_obj,
            "__qualname__",
            getattr(callable_obj, "__name__", callable_obj.__class__.__name__),
        )

    @staticmethod
    def _infer_model_parameter_names(model_fn, n_params):
        try:
            signature = inspect.signature(model_fn)
        except (TypeError, ValueError):
            return None

        positional = [
            param
            for param in signature.parameters.values()
            if param.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        inferred = [param.name for param in positional[3:]]
        if len(inferred) == int(n_params):
            return inferred
        return None

    @staticmethod
    def _normalize_parameter_names(parameter_names, n_params, model_fn=None):
        n_params = int(n_params)
        if parameter_names is None:
            if model_fn is None and n_params == 3:
                parameter_names = ["C", "dT", "tau"]
            else:
                parameter_names = Alignment._infer_model_parameter_names(model_fn, n_params)
                if parameter_names is None:
                    parameter_names = [f"p{idx}" for idx in range(n_params)]

        parameter_names = [str(name) for name in parameter_names]
        if len(parameter_names) != n_params:
            raise ValueError(
                f"parameter_names must contain {n_params} names, got {len(parameter_names)}"
            )
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must not contain duplicates")
        return parameter_names

    @staticmethod
    def _resolve_fit_setup(
        model_fn,
        p0,
        parameter_names,
        initial_C,
        initial_dT,
        initial_tau,
    ):
        if p0 is None:
            if model_fn is not None:
                raise ValueError("p0 is required when model_fn is provided")
            p0_array = Alignment._fit_initial_guess(initial_C, initial_dT, initial_tau)
        else:
            p0_array = Alignment.to_numpy_1d(p0, dtype=float)

        if p0_array.ndim != 1 or p0_array.size == 0:
            raise ValueError("p0 must be a non-empty 1D sequence")
        if not np.all(np.isfinite(p0_array)):
            raise ValueError("p0 must contain only finite values")

        names = Alignment._normalize_parameter_names(
            parameter_names,
            len(p0_array),
            model_fn=model_fn,
        )
        return p0_array.astype(float, copy=True), names

    @staticmethod
    def _normalize_fit_bounds(
        bounds,
        n_params,
        parameter_names,
        nbin,
        tau_lower_bound,
        tau_upper_bound,
        amplitude_param="C",
        delay_param="dT",
        lifetime_param="tau",
    ):
        n_params = int(n_params)
        if bounds is None:
            lb = np.full(n_params, -np.inf, dtype=float)
            ub = np.full(n_params, np.inf, dtype=float)
        else:
            lb, ub = bounds
            lb = np.broadcast_to(np.asarray(lb, dtype=float), (n_params,)).copy()
            ub = np.broadcast_to(np.asarray(ub, dtype=float), (n_params,)).copy()

        if np.any(lb > ub):
            raise ValueError("each lower bound must be <= the corresponding upper bound")

        for idx, name in enumerate(parameter_names):
            if name == amplitude_param and not np.isfinite(lb[idx]):
                lb[idx] = 0.0
            if name == delay_param:
                if not np.isfinite(lb[idx]):
                    lb[idx] = -float(nbin) / 2.0
                if not np.isfinite(ub[idx]):
                    ub[idx] = float(nbin) / 2.0
            if name == lifetime_param:
                if not np.isfinite(lb[idx]):
                    lb[idx] = float(tau_lower_bound)
                if not np.isfinite(ub[idx]):
                    ub[idx] = float(tau_upper_bound)

        return lb, ub

    @staticmethod
    def _fit_active_state(p0, bounds, parameter_names, force_C_normalized, amplitude_param):
        p0 = np.asarray(p0, dtype=float).copy()
        lb, ub = bounds
        lb = np.asarray(lb, dtype=float).copy()
        ub = np.asarray(ub, dtype=float).copy()
        active_mask = np.ones(p0.shape, dtype=bool)

        if force_C_normalized and amplitude_param in parameter_names:
            amp_idx = parameter_names.index(amplitude_param)
            p0[amp_idx] = 1.0
            active_mask[amp_idx] = False

        for idx in range(p0.size):
            if np.isfinite(lb[idx]) and p0[idx] < lb[idx]:
                p0[idx] = lb[idx]
            if np.isfinite(ub[idx]) and p0[idx] > ub[idx]:
                p0[idx] = ub[idx]

        if not np.any(active_mask):
            raise ValueError("at least one parameter must remain free during fitting")

        return p0, lb, ub, active_mask

    @staticmethod
    def _expand_active_params(active_params, p0_full, active_mask):
        params = np.asarray(p0_full, dtype=float).copy()
        params[np.asarray(active_mask, dtype=bool)] = np.asarray(active_params, dtype=float)
        return params

    @staticmethod
    def _expand_active_covariance(cov_active, active_mask):
        active_mask = np.asarray(active_mask, dtype=bool)
        cov = np.full((active_mask.size, active_mask.size), np.nan, dtype=float)
        cov_active = np.asarray(cov_active, dtype=float)
        active_indices = np.flatnonzero(active_mask)
        if cov_active.shape == (active_indices.size, active_indices.size):
            cov[np.ix_(active_indices, active_indices)] = cov_active
        return cov

    @staticmethod
    def _fit_param_errors(covariance, n_params):
        covariance = np.asarray(covariance, dtype=float)
        errors = np.full(int(n_params), np.nan, dtype=float)
        if covariance.shape != (int(n_params), int(n_params)):
            return errors
        diag = np.diag(covariance)
        valid = np.isfinite(diag) & (diag >= 0)
        errors[valid] = np.sqrt(diag[valid])
        return errors

    @staticmethod
    def _evaluate_fit_model(model_function, t_ns, params, expected_len, invalid_fill=None):
        try:
            model = Alignment.to_numpy_1d(model_function(t_ns, *params), dtype=float)
        except Exception:
            if invalid_fill is None:
                return None
            return np.full(int(expected_len), float(invalid_fill), dtype=float)

        if model.shape != (int(expected_len),):
            if invalid_fill is None:
                return None
            return np.full(int(expected_len), float(invalid_fill), dtype=float)
        if not np.all(np.isfinite(model)):
            if invalid_fill is None:
                return None
            return np.full(int(expected_len), float(invalid_fill), dtype=float)
        if np.any(model < 0):
            model = np.clip(model, 0.0, None)
        if float(np.sum(model)) <= 0:
            if invalid_fill is None:
                return None
            return np.full(int(expected_len), float(invalid_fill), dtype=float)
        return model

    @staticmethod
    def _default_fit_model_function(
        t_base_ns,
        dt_ns,
        nbin,
        period_ns,
        fit_irf_hist_norm,
        fit_irf_fft,
        mode,
    ):
        def fit_model_numpy(_t_ns_fit, C_norm, dT_bins, tau_ns):
            shift_bins = dT_bins if mode == "model_shift" else 0.0
            pure_model_hist = Alignment._model_data_binned_fast(
                t_base_ns,
                dt_ns,
                nbin,
                period_ns,
                1.0,
                tau_ns,
                shift_bins=shift_bins,
            )
            if mode == "irf_shift":
                fit_irf_hist = Alignment.linear_shift(
                    fit_irf_hist_norm,
                    dT_bins,
                    cyclic=True,
                )
                kernel_fft = None
            else:
                fit_irf_hist = fit_irf_hist_norm
                kernel_fft = fit_irf_fft

            model = Alignment._cyclic_fft_convolve_centered(
                pure_model_hist / pure_model_hist.sum(),
                fit_irf_hist / fit_irf_hist.sum(),
                kernel_fft=kernel_fft,
            )
            return float(C_norm) * model

        return fit_model_numpy

    @staticmethod
    def _custom_fit_model_function(model_fn, irf_hist_norm, period_ns, model_kwargs):
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

        def fit_model_numpy(t_ns_fit, *params):
            return model_fn(
                t_ns_fit,
                irf_hist_norm,
                period_ns,
                *params,
                **model_kwargs,
            )

        return fit_model_numpy

    @staticmethod
    def _prepare_fit_irf(data_hist_norm, irf_hist_norm, initial_dT):
        """
        Optionally pre-shift the IRF so the optimizer fits a small residual dT.
        """
        if initial_dT is not None:
            return irf_hist_norm, initial_dT, None

        dT_seed_bins = Alignment.estimate_peak_dT_bins(data_hist_norm, irf_hist_norm)
        fit_irf_hist_norm = Alignment.linear_shift(irf_hist_norm, dT_seed_bins, cyclic=True)
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
        p0_full,
        lb,
        ub,
        active_mask,
    ):
        p0_active = np.asarray(p0_full, dtype=float)[active_mask]
        lb_active = np.asarray(lb, dtype=float)[active_mask]
        ub_active = np.asarray(ub, dtype=float)[active_mask]

        def residual(active_params):
            params = Alignment._expand_active_params(active_params, p0_full, active_mask)
            model = Alignment._evaluate_fit_model(
                fit_model_numpy,
                t_ns,
                params,
                expected_len=len(data_hist),
            )
            if model is None:
                return np.full_like(data_hist, 1e12, dtype=float)
            model_counts = data_sum * np.clip(model, 1e-12, None)
            return Alignment._poisson_deviance_residual(
                observed_counts=data_hist,
                model_counts=model_counts,
            )

        result = scipy_optimize.least_squares(
            residual,
            p0_active,
            bounds=(lb_active, ub_active),
            max_nfev=600000,
        )
        if not result.success:
            raise RuntimeError(f"poisson fit failed: {result.message}")

        cov_active = Alignment._covariance_from_least_squares(
            result,
            n_observations=len(data_hist),
            n_params=len(p0_active),
            scale_by_cost=False,
        )
        return (
            Alignment._expand_active_params(result.x, p0_full, active_mask),
            Alignment._expand_active_covariance(cov_active, active_mask),
        )

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
        p0_full,
        lb,
        ub,
        active_mask,
        fit_type,
        circular_params,
    ):
        sigma = np.sqrt(np.clip(data_hist, 1.0, None)) / data_sum
        p0_active = np.asarray(p0_full, dtype=float)[active_mask]
        lb_active = np.asarray(lb, dtype=float)[active_mask]
        ub_active = np.asarray(ub, dtype=float)[active_mask]
        active_indices = np.flatnonzero(active_mask)
        active_index_lookup = {
            int(full_idx): int(active_idx)
            for active_idx, full_idx in enumerate(active_indices)
        }
        circular_active = {
            active_index_lookup[full_idx]: period
            for full_idx, period in circular_params.items()
            if full_idx in active_index_lookup
        }

        def fit_model_active(t_ns_fit, *active_params):
            params = Alignment._expand_active_params(active_params, p0_full, active_mask)
            return Alignment._evaluate_fit_model(
                fit_model_numpy,
                t_ns_fit,
                params,
                expected_len=len(data_hist_norm),
                invalid_fill=1e12,
            )

        popt_active, cov_active = Alignment._run_curve_fit(
            fit_type,
            fit_model_active,
            t_ns,
            data_hist_norm,
            sigma,
            p0_active,
            (lb_active, ub_active),
            circular_params=circular_active,
        )
        return (
            Alignment._expand_active_params(popt_active, p0_full, active_mask),
            Alignment._expand_active_covariance(cov_active, active_mask),
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
        model_fn=None,
        p0=None,
        bounds=None,
        parameter_names=None,
        param_names=None,
        model_kwargs=None,
        amplitude_param="C",
        delay_param="dT",
        lifetime_param="tau",
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
        - custom models can be supplied with ``model_fn``. The callable must
          follow ``model_fn(t, irf, period, *params, **model_kwargs)`` and
          return a 1D fitted histogram in the same normalized units as
          ``data / data.sum()``. ``p0`` is required for custom models.
        """
        parameter_names = Alignment._resolve_parameter_names_alias(parameter_names, param_names)
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        data_hist = Alignment.to_numpy_1d(data, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)
        custom_model = model_fn is not None

        if len(t_ns) != len(data_hist) or len(t_ns) != len(irf_hist):
            raise ValueError("t, data, and irf must have the same 1D length")
        if len(t_ns) < 2:
            raise ValueError("perform_fit_data requires at least two time samples")
        if not custom_model and mode not in {"model_shift", "irf_shift"}:
            raise ValueError(
                f"Unsupported mode: {mode}. Supported model_shift, irf_shift"
            )
        dt_ns = float(t_ns[1] - t_ns[0])
        if not np.allclose(np.diff(t_ns), dt_ns):
            raise ValueError("perform_fit_data requires uniformly spaced t values")

        data_sum = data_hist.sum()
        irf_sum = irf_hist.sum()

        if not np.isfinite(data_sum) or data_sum <= 0:
            return Alignment._skip_fit_with_warning("data")
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            return Alignment._skip_fit_with_warning("irf")

        Alignment._require_scipy_optimize()

        nbin = len(t_ns)
        period_ns = float(period)
        t_base_ns = t_ns - period_ns - (period_ns / 2.0)
        tau_lower_bound = float(irf_min)
        if tau_lower_bound <= 0:
            raise ValueError("irf_min must be positive")

        fit_type = Alignment._canonical_fit_type(fit_type)

        data_hist_norm = data_hist / data_sum
        irf_hist_norm = irf_hist / irf_sum
        tau_upper_bound = t_ns.max()

        if custom_model:
            p0_full, resolved_parameter_names = Alignment._resolve_fit_setup(
                model_fn,
                p0,
                parameter_names,
                initial_C,
                initial_dT,
                initial_tau,
            )
            dT_seed_bins = None
            fit_model_numpy = Alignment._custom_fit_model_function(
                model_fn,
                irf_hist_norm,
                period_ns,
                model_kwargs,
            )
        else:
            seed_initial_dT = initial_dT
            if p0 is not None:
                p0_array = Alignment.to_numpy_1d(p0, dtype=float)
                if p0_array.size > 1:
                    seed_initial_dT = float(p0_array[1])
            fit_irf_hist_norm, fit_initial_dT, dT_seed_bins = Alignment._prepare_fit_irf(
                data_hist_norm,
                irf_hist_norm,
                seed_initial_dT,
            )
            p0_full, resolved_parameter_names = Alignment._resolve_fit_setup(
                None,
                p0,
                parameter_names,
                initial_C,
                fit_initial_dT,
                initial_tau,
            )
            fit_irf_fft = np.fft.fft(fit_irf_hist_norm) if mode == "model_shift" else None
            fit_model_numpy = Alignment._default_fit_model_function(
                t_base_ns,
                dt_ns,
                nbin,
                period_ns,
                fit_irf_hist_norm,
                fit_irf_fft,
                mode,
            )

        lb, ub = Alignment._normalize_fit_bounds(
            bounds,
            len(p0_full),
            resolved_parameter_names,
            nbin,
            tau_lower_bound,
            tau_upper_bound,
            amplitude_param=amplitude_param,
            delay_param=delay_param,
            lifetime_param=lifetime_param,
        )
        p0_full, lb, ub, active_mask = Alignment._fit_active_state(
            p0_full,
            (lb, ub),
            resolved_parameter_names,
            force_C_normalized,
            amplitude_param,
        )

        circular_params = {}
        if delay_param in resolved_parameter_names:
            delay_idx = resolved_parameter_names.index(delay_param)
            if np.isfinite(lb[delay_idx]) and np.isfinite(ub[delay_idx]):
                circular_params[delay_idx] = float(ub[delay_idx] - lb[delay_idx])

        if fit_type == "likelihood":
            popt, cov = Alignment._run_poisson_fit(
                fit_model_numpy,
                t_ns,
                data_hist,
                data_sum,
                p0_full,
                lb,
                ub,
                active_mask,
            )
        else:
            popt, cov = Alignment._run_weighted_least_squares_fit(
                fit_model_numpy,
                t_ns,
                data_hist,
                data_hist_norm,
                data_sum,
                p0_full,
                lb,
                ub,
                active_mask,
                fit_type,
                circular_params,
            )

        if not custom_model:
            popt = Alignment._restore_seeded_dT(popt, dT_seed_bins, nbin)

        param_errors = Alignment._fit_param_errors(cov, len(popt))
        params = {
            name: float(value)
            for name, value in zip(resolved_parameter_names, np.asarray(popt, dtype=float))
        }

        if custom_model:
            fitted_hist = Alignment._evaluate_fit_model(
                fit_model_numpy,
                t_ns,
                popt,
                expected_len=len(t_ns),
            )
        else:
            fitted_hist = Alignment.fit_model_data(
                t_ns,
                params.get(amplitude_param, np.nan),
                params.get(delay_param, np.nan),
                params.get(lifetime_param, np.nan),
                irf=irf_hist_norm,
                period=period_ns,
                mode=mode,
            )
        if fitted_hist is not None:
            fitted_hist = np.asarray(fitted_hist, dtype=float)

        delay_value = params.get(delay_param, np.nan)
        result = {
            "C": params.get(amplitude_param, np.nan),
            "dT": delay_value,
            "dT_ns": delay_value * dt_ns,
            "tau": params.get(lifetime_param, np.nan),
            "params": params,
            "parameter_names": list(resolved_parameter_names),
            "param_names": list(resolved_parameter_names),
            "param_values": np.asarray(popt, dtype=float),
            "param_errors": param_errors,
            "fit": fitted_hist,
            "model_name": Alignment._callable_name(model_fn),
        }
        return result, cov

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
        model_fn,
        p0,
        bounds,
        parameter_names,
        param_names,
        model_kwargs,
        amplitude_param,
        delay_param,
        lifetime_param,
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
                model_fn=model_fn,
                p0=p0,
                bounds=bounds,
                parameter_names=parameter_names,
                param_names=param_names,
                model_kwargs=model_kwargs,
                amplitude_param=amplitude_param,
                delay_param=delay_param,
                lifetime_param=lifetime_param,
            )
            errors = np.asarray(fit_res.get("param_errors", []), dtype=float)
            return (
                y,
                x,
                np.asarray(fit_res["param_values"], dtype=float),
                errors,
            )
        except Exception:
            if not catch_exceptions:
                raise
            resolved_names = Alignment._resolve_parameter_names_alias(parameter_names, param_names)
            param_count = len(resolved_names) if resolved_names is not None else 3
            return (
                y,
                x,
                np.full(param_count, np.nan, dtype=float),
                np.full(param_count, np.nan, dtype=float),
            )

    @staticmethod
    def _fit_map_pixel_chunk(indices, histograms, **worker_kwargs):
        return [
            Alignment._fit_map_one_pixel(int(idx), hist, **worker_kwargs)
            for idx, hist in zip(indices, histograms)
        ]

    @staticmethod
    def _fit_map_job_chunk_size(valid_count, n_jobs, job_chunk_size):
        if job_chunk_size is not None:
            return max(1, int(job_chunk_size))

        if int(n_jobs) == 1:
            return 1

        if int(n_jobs) > 0:
            worker_count = int(n_jobs)
        else:
            try:
                from os import cpu_count
                worker_count = cpu_count() or 1
            except Exception:  # pragma: no cover - extremely defensive fallback
                worker_count = 1

        target_chunks = max(worker_count * 8, 1)
        return max(1, int(np.ceil(valid_count / target_chunks)))

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
        force_C_normalized=None,
        min_counts=0.0,
        min_peak_counts=0.0,
        min_nonzero_bins=1,
        valid_mask=None,
        n_jobs=1,
        backend="loky",
        job_chunk_size=None,
        show_progress=True,
        catch_exceptions=True,
        model_fn=None,
        p0=None,
        bounds=None,
        parameter_names=None,
        param_names=None,
        model_kwargs=None,
        amplitude_param="C",
        delay_param="dT",
        lifetime_param="tau",
    ):
        """
        Fit every pixel histogram in a ``(y, x, t)`` image and return fit maps.

        Returns a dictionary with one map per fitted parameter and matching
        ``*_err`` maps. The default model keeps the historical ``C``, ``dT``,
        ``tau``, ``C_err``, ``dT_err``, and ``tau_err`` maps. Custom models use
        ``model_fn(t, irf, period, *params, **model_kwargs)`` and return maps
        named by ``parameter_names`` or by the custom function signature. Invalid
        or failed pixels are filled with NaNs.

        ``min_counts``, ``min_peak_counts``, ``min_nonzero_bins``, and
        ``valid_mask`` are applied before fitting so low-information pixels can
        be skipped cheaply. With ``n_jobs != 1``, pixels are submitted to joblib
        in chunks instead of one job per pixel; override ``job_chunk_size`` when
        a specific chunk size is needed.

        When ``force_C_normalized`` is left as ``None``, the historical default
        is kept for the built-in single-exponential model (``C`` fixed to 1)
        while custom models fit their amplitude parameter unless explicitly
        forced with ``force_C_normalized=True``.
        """
        parameter_names = Alignment._resolve_parameter_names_alias(parameter_names, param_names)
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

        resolved_force_C_normalized = (
            model_fn is None if force_C_normalized is None else bool(force_C_normalized)
        )

        if model_fn is not None:
            p0_full, resolved_parameter_names = Alignment._resolve_fit_setup(
                model_fn,
                p0,
                parameter_names,
                initial_C,
                initial_dT,
                initial_tau,
            )
        else:
            p0_full, resolved_parameter_names = Alignment._resolve_fit_setup(
                None,
                p0,
                parameter_names,
                initial_C,
                initial_dT,
                initial_tau,
            )

        data_2d = data_array.reshape(-1, nbins)
        data_sums = np.sum(data_2d, axis=1)
        pixel_is_valid = (
            np.all(np.isfinite(data_2d), axis=1)
            & (data_sums > float(min_counts))
            & (np.max(data_2d, axis=1) >= float(min_peak_counts))
            & (np.count_nonzero(data_2d > 0, axis=1) >= int(min_nonzero_bins))
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
            "parameter_names": list(resolved_parameter_names),
            "param_names": list(resolved_parameter_names),
        }
        for name in resolved_parameter_names:
            fit_maps[name] = np.full((ny, nx), np.nan, dtype=float)
            fit_maps[f"{name}_err"] = np.full((ny, nx), np.nan, dtype=float)

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
            force_C_normalized=resolved_force_C_normalized,
            model_fn=model_fn,
            p0=p0_full if model_fn is not None else p0,
            bounds=bounds,
            parameter_names=resolved_parameter_names,
            param_names=None,
            model_kwargs=model_kwargs,
            amplitude_param=amplitude_param,
            delay_param=delay_param,
            lifetime_param=lifetime_param,
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

            chunk_size = Alignment._fit_map_job_chunk_size(
                valid_indices.size,
                n_jobs=n_jobs,
                job_chunk_size=job_chunk_size,
            )
            index_chunks = [
                valid_indices[start:start + chunk_size]
                for start in range(0, valid_indices.size, chunk_size)
            ]
            results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
                delayed(Alignment._fit_map_pixel_chunk)(
                    chunk,
                    data_2d[chunk],
                    **worker_kwargs,
                )
                for chunk in progress(index_chunks, desc="Fitting pixel chunks")
            )
            results = [row for chunk_result in results for row in chunk_result]

        for y, x, values, errors in results:
            values = np.asarray(values, dtype=float)
            errors = np.asarray(errors, dtype=float)
            for param_idx, name in enumerate(resolved_parameter_names):
                fit_maps[name][y, x] = values[param_idx] if param_idx < values.size else np.nan
                fit_maps[f"{name}_err"][y, x] = (
                    errors[param_idx] if param_idx < errors.size else np.nan
                )

        return fit_maps

    @staticmethod
    def fit_maps_to_stack(fit_maps, names=None):
        """Return a stack and name list from a fit-map dictionary."""
        if names is None:
            if "parameter_names" in fit_maps:
                parameter_names = list(fit_maps["parameter_names"])
                names = parameter_names + [
                    f"{name}_err"
                    for name in parameter_names
                    if f"{name}_err" in fit_maps
                ]
            elif "param_names" in fit_maps:
                param_names = list(fit_maps["param_names"])
                names = param_names + [
                    f"{name}_err"
                    for name in param_names
                    if f"{name}_err" in fit_maps
                ]
            else:
                names = ("C", "dT", "tau", "C_err", "dT_err", "tau_err")
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
    def _resolve_shift_sum_backend(backend):
        backend = str(backend).strip().lower()
        aliases = {
            "auto": "auto",
            "cpu": "cpu",
            "numpy": "cpu",
            "np": "cpu",
            "gpu": "gpu",
            "cuda": "gpu",
        }
        if backend not in aliases:
            raise ValueError("backend must be one of 'auto', 'cpu', or 'gpu'")

        backend = aliases[backend]
        gpu_available = (
            torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
        )

        if backend == "auto":
            if gpu_available:
                return "gpu"
            warnings.warn(
                "sum_channel_applying_shifts backend='auto' requested GPU, "
                "but CUDA is not available; falling back to CPU.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "cpu"

        if backend == "gpu" and not gpu_available:
            if torch is None:
                raise ImportError("backend='gpu' requires torch with CUDA support")
            raise RuntimeError(
                "backend='gpu' requested, but torch.cuda.is_available() is False"
            )

        return backend

    @staticmethod
    def _shift_sum_indices_and_weights(n_bins, shifts):
        i = np.arange(n_bins, dtype=float)[:, None]
        s = np.asarray(shifts, dtype=float)[None, :]

        dst = i - s
        j0 = np.floor(dst).astype(np.intp)
        alpha = dst - j0
        j1 = j0 + 1

        return (
            (j0 % n_bins).reshape(-1),
            (j1 % n_bins).reshape(-1),
            (1.0 - alpha).reshape(-1),
            alpha.reshape(-1),
        )

    @staticmethod
    def _shift_sum_weight_matrix_numpy(n_bins, shifts):
        j0, j1, w0, w1 = Alignment._shift_sum_indices_and_weights(n_bins, shifts)
        rows = np.arange(j0.size)
        weights = np.zeros((j0.size, n_bins), dtype=float)
        np.add.at(weights, (rows, j0), w0)
        np.add.at(weights, (rows, j1), w1)
        return weights

    @staticmethod
    def _sum_channel_applying_shifts_cpu(flat, n_bins, shifts):
        weights = Alignment._shift_sum_weight_matrix_numpy(n_bins, shifts)
        return flat.reshape(flat.shape[0], -1) @ weights

    @staticmethod
    def _default_gpu_chunk_size(batch_size, flat_width, dtype):
        bytes_per_value = torch.empty((), dtype=dtype).element_size()
        target_bytes = 256 * 1024 * 1024
        return max(
            1,
            min(batch_size, target_bytes // max(flat_width * bytes_per_value, 1)),
        )

    @staticmethod
    def _sum_channel_applying_shifts_gpu(flat, n_bins, shifts, show_progress, chunk_size=None):
        Alignment._require_torch()
        device = torch.device("cuda")
        dtype = torch.float64
        flat_width = n_bins * len(shifts)
        batch_size = flat.shape[0]
        if chunk_size is None:
            chunk_size = Alignment._default_gpu_chunk_size(batch_size, flat_width, dtype)
        else:
            chunk_size = max(1, int(chunk_size))

        weights_np = Alignment._shift_sum_weight_matrix_numpy(n_bins, shifts)
        weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
        out = np.empty((batch_size, n_bins), dtype=float)

        progress = tqdm if show_progress else (lambda x, **kwargs: x)
        for start in progress(
            range(0, batch_size, chunk_size),
            desc="Summing shifted histogram chunks",
        ):
            stop = min(start + chunk_size, batch_size)
            chunk = torch.as_tensor(
                flat[start:stop].reshape(stop - start, flat_width),
                dtype=dtype,
                device=device,
            )
            out[start:stop] = (chunk @ weights).cpu().numpy()

        return out

    @staticmethod
    def sum_channel_applying_shifts(
        data,
        shifts_array,
        axis=(0, 1, 2, 3),
        reverse_shifts=True,
        backend="auto",
        chunk_size=None,
        show_progress=True,
    ):
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

        backend : {"auto", "cpu", "gpu"}, default "auto"
            Execution backend. ``"auto"`` uses CUDA when PyTorch reports an
            available GPU and warns before falling back to CPU otherwise.
            ``"cpu"`` uses a NumPy/BLAS matrix multiplication. ``"gpu"``
            requires a CUDA-capable PyTorch installation.

        chunk_size : int, optional
            Number of flattened histograms processed per GPU chunk. Ignored by
            the CPU backend.

        show_progress : bool, default True
            If ``True``, show a ``tqdm`` progress bar for GPU chunks.

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
        - CPU accumulation is a single weighted matrix multiplication.
        - GPU accumulation is chunked to limit CUDA memory use.

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
        resolved_backend = Alignment._resolve_shift_sum_backend(backend)
        if resolved_backend == "gpu":
            out = Alignment._sum_channel_applying_shifts_gpu(
                flat,
                n_bins,
                shifts,
                show_progress=show_progress,
                chunk_size=chunk_size,
            )
        else:
            out = Alignment._sum_channel_applying_shifts_cpu(flat, n_bins, shifts)

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

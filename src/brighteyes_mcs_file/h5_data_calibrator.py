"""HDF5 calibration and structure inspection helpers."""

from html import escape
import json
from pathlib import Path
import shutil
import warnings

import h5py
import numpy as np
from tqdm.auto import tqdm

from .alignment import Alignment
from . import mcs
from .channel_skew_estimator import estimate_channel_skew

__all__ = [
    "H5DataCalibrator",
    "calibrate_h5_file",
    "show_h5_structure",
    "show_h5_structure_html",
]

DEFAULT_DATA_KEY = ("data", "data_channels_extra")
DEFAULT_REFERENCE_KEY = None
DEFAULT_TAU_REF = None
DEFAULT_REFERENCE_TYPE = "ref"
DEFAULT_FIT_MODE = "model_shift"
DEFAULT_FIT_TYPE = "likelihood"
DEFAULT_C_REF = 1.0
DEFAULT_IRF_ITERATIONS = 300
DEFAULT_REGULARIZATION = 0
DEFAULT_CLEAN_IRF = False
DEFAULT_IRF_CORRECTIONS_TYPE = "median"
DEFAULT_CHANNEL_SKEW_TYPE = "phase_cross_correlation"
DEFAULT_CHANNEL_SKEW_SOURCE = "ref"
DEFAULT_CHANNEL_SKEW_FIT_REFERENCE_CHANNEL = 12
DEFAULT_CHANNEL_SKEW_FIT_UPSAMPLING = 10
DEFAULT_CHANNEL_SKEW_FIT_APODIZE = False
DEFAULT_OVERWRITE = True


class H5DataCalibrator:
    """
    Calibrate per-channel FLIM histograms stored in HDF5 files.

    Parameters
    ----------
    data_path : str or path-like
        HDF5 file containing the data to calibrate.
    reference_path : str or path-like
        HDF5 file containing the reference histogram or IRF source.
    data_key : str or iterable of str, default ``("data", "data_channels_extra")``
        Dataset key or keys to calibrate in ``data_path``. When ``None``, the
        class falls back to ``DEFAULT_DATA_KEYS``.
    reference_key : None, str, iterable of str, or dict, default ``None``
        Dataset key selection for the reference file. If ``None``, each data
        key is mapped to itself. A single string is reused for all data keys, an
        iterable can provide one key per data key, and a dict can map each data
        key explicitly.
    reference_type : {"ref", "irf"}, default ``"ref"``
        Type of reference data. Aliases such as ``"reference"`` are normalized
        internally to ``"ref"``.
    tau_ref : float or None, default ``None``
        Reference lifetime in ns. When ``None``, the reference lifetime is
        estimated from the reference data when needed by the chosen fit mode.
    fit_mode : str, default ``"model_shift"``
        Fitting mode forwarded to the alignment routines.
    fit_type : {"likelihood", "curve_fit_circular", "curve_fit"}, default ``"likelihood"``
        Fitting backend forwarded to the alignment routines.
    C_ref : float, default ``1.0``
        Reference amplitude scaling factor.
    output_path : str or path-like or None, default ``None``
        Output HDF5 path. When ``None``, a default calibrated output path is
        derived from ``data_path``.
    overwrite : bool, default ``True``
        If ``True``, overwrite an existing output file.
    channels : iterable of int or None, default ``None``
        Optional subset of channel indices to calibrate. When ``None``, all
        available channels are processed.
    calibration_key : str, default ``"calibration"``
        Group name used to store calibration outputs in the destination file.
    period_ns : float or None, default ``None``
        Laser period in ns. When ``None``, the value is inferred from metadata
        when possible.
    initial_tau : float or None, default ``None``
        Optional initial guess for the lifetime fit.
    initial_dT : float or None, default ``None``
        Optional initial guess for the temporal shift fit.
    initial_C : float or None, default ``None``
        Optional initial guess for the amplitude fit.
    force_C_normalized : bool, default ``False``
        If ``True``, force the fitted amplitude term to remain normalized.
    model_fn : callable or None, default ``None``
        Optional full fit model callable
        ``model_fn(t, irf, period, *params, **model_kwargs)``. When omitted,
        the default single-exponential ``C, dT, tau`` model is used.
    p0, bounds, param_names, model_kwargs : optional
        Initial values, bounds, names, and extra keyword arguments for
        ``model_fn``. ``p0`` is required when ``model_fn`` is provided.
    amplitude_param, delay_param, lifetime_param : str
        Parameter names used to populate legacy output datasets. Custom HDF5
        calibration requires ``delay_param`` to be present so common-delay
        correction can still be computed.
    irf_iterations : int, default ``300``
        Number of iterations used when estimating the IRF from the reference
        data.
    eps : float, default ``1e-8``
        Numerical stability constant passed to the fitting routines.
    regularization : float, default ``0``
        Regularization strength used during IRF estimation.
    clean_irf : bool, default ``False``
        If ``True`` and ``reference_type="irf"``, apply
        :meth:`Alignment.clean_irf_stack` to the realigned IRF stack using the
        historical notebook settings before it is rescaled for output.
    irf_corrections_type : {"median", "single_ch"}, default ``"median"``
        Strategy used to choose the delay applied when building the realigned
        IRF/reference stacks. ``"median"`` uses the median fitted delay across
        finite fitted channels. ``"single_ch"`` uses each channel's own fitted
        delay, preserving the historical behavior.
    channel_skew_type : {"phase_cross_correlation"}, default ``"phase_cross_correlation"``
        Strategy used to populate ``channel_skew`` outputs. Only
        ``"phase_cross_correlation"`` is currently supported.
    channel_skew_source : {"ref", "irf", "data", "metadata"} or numpy.ndarray, default ``"ref"``
        Source used for channel-skew generation. String values select one of
        the stored calibration histograms, while a 1D NumPy array forces the
        final ``channel_skew`` values directly. When the ``data`` key is
        present, non-``data`` groups are anchored by default to the selected
        reference channel from the ``data`` group. ``"metadata"`` is reserved
        and currently raises ``NotImplementedError``.
    channel_skew_fit_reference_channel : int, default ``12``
        Reference channel index used by the local channel-skew estimator. The value is matched
        against the calibrated ``channel_index`` entries for each dataset key.
        When the default value ``12`` is not present, the middle calibrated
        channel is used automatically.
    channel_skew_fit_upsampling : int, default ``10``
        Upsampling factor forwarded to the local channel-skew estimator.
    channel_skew_fit_apodize : bool, default ``False``
        Apodization flag forwarded to the local channel-skew estimator.

    Notes
    -----
    The input datasets are expected to have shape
    ``[repetition, z, y, x, t, ch]``. Only one channel histogram at a time is
    materialized in memory, so the whole 6D dataset is never converted to a
    NumPy array up front. Calibration results for each data key are always
    written under ``<calibration_key>/<data_key>/``, even in single-key mode.
    """

    DEFAULT_DATA_KEYS = DEFAULT_DATA_KEY

    def __init__(
        self,
        data_path,
        reference_path,
        data_key=DEFAULT_DATA_KEY,
        reference_key=None,
        reference_type=DEFAULT_REFERENCE_TYPE,
        tau_ref=DEFAULT_TAU_REF,
        fit_mode=DEFAULT_FIT_MODE,
        fit_type=DEFAULT_FIT_TYPE,
        C_ref=DEFAULT_C_REF,
        output_path=None,
        overwrite=DEFAULT_OVERWRITE,
        channels=None,
        calibration_key="calibration",
        period_ns=None,
        initial_tau=None,
        initial_dT=None,
        initial_C=None,
        force_C_normalized=False,
        model_fn=None,
        p0=None,
        bounds=None,
        param_names=None,
        model_kwargs=None,
        amplitude_param="C",
        delay_param="dT",
        lifetime_param="tau",
        irf_iterations=DEFAULT_IRF_ITERATIONS,
        eps=1e-8,
        regularization=DEFAULT_REGULARIZATION,
        clean_irf=DEFAULT_CLEAN_IRF,
        irf_corrections_type=DEFAULT_IRF_CORRECTIONS_TYPE,
        channel_skew_type=DEFAULT_CHANNEL_SKEW_TYPE,
        channel_skew_source=DEFAULT_CHANNEL_SKEW_SOURCE,
        channel_skew_fit_reference_channel=DEFAULT_CHANNEL_SKEW_FIT_REFERENCE_CHANNEL,
        channel_skew_fit_upsampling=DEFAULT_CHANNEL_SKEW_FIT_UPSAMPLING,
        channel_skew_fit_apodize=DEFAULT_CHANNEL_SKEW_FIT_APODIZE,
    ):
        self.data_path = Path(data_path)
        self.reference_path = Path(reference_path)
        self.data_keys = self._normalize_key_sequence(data_key, "data_key")
        self.reference_key_map = self._normalize_reference_keys(reference_key, self.data_keys)
        self.reference_type = self._normalize_reference_type(reference_type)
        self.tau_ref = tau_ref
        self.fit_mode = fit_mode
        self.fit_type = Alignment._canonical_fit_type(fit_type)
        self.C_ref = C_ref
        self.output_path = Path(output_path) if output_path is not None else self._default_output_path()
        self.overwrite = overwrite
        self.channels = channels
        self.calibration_key = calibration_key
        self.period_ns = period_ns
        self.initial_tau = initial_tau
        self.initial_dT = initial_dT
        self.initial_C = initial_C
        self.force_C_normalized = force_C_normalized
        self.model_fn = model_fn
        self.p0 = p0
        self.bounds = bounds
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.amplitude_param = str(amplitude_param)
        self.delay_param = str(delay_param)
        self.lifetime_param = str(lifetime_param)
        _, self.param_names = Alignment._resolve_fit_setup(
            self.model_fn,
            self.p0,
            param_names,
            self.initial_C,
            self.initial_dT,
            self.initial_tau,
        )
        if self.model_fn is not None and self.delay_param not in self.param_names:
            raise ValueError(
                f"param_names must include delay_param={self.delay_param!r} "
                "when model_fn is provided so calibration delays can be stored"
            )
        self.irf_iterations = irf_iterations
        self.eps = eps
        self.regularization = regularization
        self.clean_irf = bool(clean_irf)
        self.irf_corrections_type = self._normalize_irf_corrections_type(
            irf_corrections_type
        )
        self.channel_skew_type = self._normalize_channel_skew_type(channel_skew_type)
        self.channel_skew_source = self._normalize_channel_skew_source(channel_skew_source)
        self.channel_skew_fit_reference_channel = int(channel_skew_fit_reference_channel)
        self.channel_skew_fit_upsampling = int(channel_skew_fit_upsampling)
        self.channel_skew_fit_apodize = bool(channel_skew_fit_apodize)

        if self.channel_skew_fit_upsampling <= 0:
            raise ValueError("channel_skew_fit_upsampling must be a positive integer")

    @staticmethod
    def _normalize_irf_corrections_type(irf_corrections_type):
        normalized = str(irf_corrections_type).strip().lower()
        aliases = {
            "median": "median",
            "med": "median",
            "single_ch": "single_ch",
            "single_channel": "single_ch",
            "channel": "single_ch",
            "ch": "single_ch",
            "single": "single_ch",
            "each": "single_ch",
            "per_channel": "single_ch",
        }
        if normalized not in aliases:
            raise ValueError(
                "irf_corrections_type must be 'median' or one of "
                "'single_ch', 'single_channel', 'channel', 'ch', 'single', "
                "'each', 'per_channel'"
            )
        return aliases[normalized]

    @staticmethod
    def _normalize_channel_skew_type(channel_skew_type):
        normalized = str(channel_skew_type).strip().lower()
        if normalized != "phase_cross_correlation":
            raise NotImplementedError(
                "channel_skew_type values other than 'phase_cross_correlation' are not supported yet"
            )
        return "phase_cross_correlation"

    @staticmethod
    def _normalize_channel_skew_source(channel_skew_source):
        if isinstance(channel_skew_source, np.ndarray):
            source_array = np.asarray(channel_skew_source, dtype=float)
            if source_array.ndim != 1:
                raise ValueError(
                    "channel_skew_source array must be 1D with one value per calibrated channel"
                )
            return source_array

        normalized = str(channel_skew_source).strip().lower()
        if normalized not in {"ref", "irf", "data", "metadata"}:
            raise ValueError(
                "channel_skew_source must be 'ref', 'irf', 'data', 'metadata', or a 1D numpy.ndarray"
            )
        return normalized

    @classmethod
    def _normalize_key_sequence(cls, keys, param_name):
        if keys is None:
            keys = cls.DEFAULT_DATA_KEYS

        if isinstance(keys, (str, Path)):
            normalized = [str(keys)]
        else:
            try:
                normalized = [str(key) for key in keys]
            except TypeError as exc:
                raise TypeError(f"{param_name} must be a string or an iterable of strings") from exc

        normalized = [key for key in normalized if key]
        if not normalized:
            raise ValueError(f"{param_name} must contain at least one dataset key")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{param_name} must not contain duplicates")
        return normalized

    @staticmethod
    def _normalize_reference_keys(reference_key, data_keys):
        if reference_key is None:
            return {data_key: data_key for data_key in data_keys}

        if isinstance(reference_key, dict):
            mapping = {}
            for data_key in data_keys:
                if data_key not in reference_key:
                    raise KeyError(f"missing reference_key entry for data key {data_key!r}")
                mapping[data_key] = str(reference_key[data_key])
            return mapping

        if isinstance(reference_key, (str, Path)):
            reference_key_str = str(reference_key)
            return {data_key: reference_key_str for data_key in data_keys}

        try:
            reference_keys = [str(key) for key in reference_key]
        except TypeError as exc:
            raise TypeError(
                "reference_key must be None, a string, a mapping, or an iterable of strings"
            ) from exc

        if len(reference_keys) == 1:
            return {data_key: reference_keys[0] for data_key in data_keys}
        if len(reference_keys) != len(data_keys):
            raise ValueError(
                "reference_key iterable must have length 1 or match the number of data keys"
            )
        return {data_key: ref_key for data_key, ref_key in zip(data_keys, reference_keys)}

    @staticmethod
    def _normalize_reference_type(reference_type):
        normalized = str(reference_type).strip().lower()
        aliases = {
            "reference": "ref",
            "reference_histogram": "ref",
            "ref_histogram": "ref",
            "fluorescence_reference": "ref",
            "irf_histogram": "irf",
            "input_irf": "irf",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"ref", "irf"}:
            raise ValueError("reference_type must be 'ref' or 'irf'")
        return normalized

    def _default_output_path(self):
        suffix = self.data_path.suffix or ".h5"
        return self.data_path.with_name(f"{self.data_path.stem}_calib{suffix}")

    @staticmethod
    def _metadata_get(metadata, key, default=None):
        if metadata is None:
            return default
        if isinstance(metadata, dict):
            return metadata.get(key, default)
        if hasattr(metadata, "get"):
            try:
                return metadata.get(key, default)
            except TypeError:
                pass
        return getattr(metadata, key, default)

    @staticmethod
    def _metadata_items(metadata):
        if metadata is None:
            return []
        if isinstance(metadata, dict):
            return list(metadata.items())
        if hasattr(metadata, "items"):
            try:
                return list(metadata.items())
            except TypeError:
                pass
        if hasattr(metadata, "__dict__") and vars(metadata):
            return [(key, value) for key, value in vars(metadata).items() if not key.startswith("_")]

        items = []
        for key in dir(metadata):
            if key.startswith("_"):
                continue
            try:
                value = getattr(metadata, key)
            except Exception:
                continue
            if callable(value):
                continue
            items.append((key, value))
        return items

    @staticmethod
    def build_time_axis(metadata, nbin=None, period_ns=None):
        if nbin is None:
            metadata_nbin = H5DataCalibrator._metadata_get(metadata, "dfd_nbins")
            if metadata_nbin is None:
                raise ValueError("metadata must provide dfd_nbins or nbin must be passed explicitly as it was not possible to obtain automatically")
            nbin = int(metadata_nbin)
        else:
            nbin = int(nbin)

        if nbin <= 0:
            raise ValueError("nbin must be positive")

        if period_ns is None:
            dfd_freq_MHz = H5DataCalibrator._metadata_get(metadata, "dfd_freq")
            if dfd_freq_MHz is None:
                raise ValueError("metadata must provide dfd_freq or period_ns must be passed explicitly as it was not possible to obtain automatically")
            dfd_freq_MHz = float(dfd_freq_MHz)
            if not np.isfinite(dfd_freq_MHz) or dfd_freq_MHz <= 0:
                raise ValueError("metadata.dfd_freq must be a positive finite value")
            period_ns = 1e3 / dfd_freq_MHz
        else:
            period_ns = float(period_ns)

        if not np.isfinite(period_ns) or period_ns <= 0:
            raise ValueError("period_ns must be a positive finite value")

        dt_ns = period_ns / nbin
        t_ns = np.arange(nbin, dtype=float) * dt_ns
        return nbin, dt_ns, period_ns, t_ns

    @staticmethod
    def _open_dataset(handle, key=None):
        if key is not None:
            if key not in handle:
                raise KeyError(f"dataset key {key!r} not found in {handle.filename!r}")
            dataset = handle[key]
        else:
            dataset = None
            for candidate in H5DataCalibrator.DEFAULT_DATA_KEYS:
                if candidate in handle:
                    dataset = handle[candidate]
                    break
            if dataset is None:
                raise KeyError(
                    f"no default dataset found in {handle.filename!r}; "
                    f"tried {H5DataCalibrator.DEFAULT_DATA_KEYS}"
                )

        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"{key!r} in {handle.filename!r} is not an HDF5 dataset")
        return dataset

    @staticmethod
    def _validate_dataset_layout(dataset, name):
        if dataset.ndim != 6:
            raise ValueError(
                f"{name} dataset must have shape [repetition, z, y, x, t, ch], got {dataset.shape}"
            )
        if dataset.shape[-2] <= 0 or dataset.shape[-1] <= 0:
            raise ValueError(f"{name} dataset must contain positive time and channel dimensions")

    @staticmethod
    def _resolve_channels(channels, channel_count):
        if channels is None:
            return list(range(int(channel_count)))

        resolved = []
        for channel in channels:
            channel_index = int(channel)
            if channel_index < 0 or channel_index >= channel_count:
                raise IndexError(f"channel {channel_index} out of range for {channel_count} channels")
            resolved.append(channel_index)

        if len(set(resolved)) != len(resolved):
            raise ValueError("channels must not contain duplicates")
        return resolved

    @staticmethod
    def _resolve_reference_channel_map(data_dataset, reference_dataset):
        data_channel_count = int(data_dataset.shape[-1])
        reference_channel_count = int(reference_dataset.shape[-1])

        if reference_channel_count == data_channel_count:
            return {channel: channel for channel in range(data_channel_count)}
        if reference_channel_count == 1:
            return {channel: 0 for channel in range(data_channel_count)}

        raise ValueError(
            "reference dataset must have either the same number of channels as data "
            "or exactly one channel"
        )

    @staticmethod
    def _sum_histogram_for_channel(dataset, channel_index):
        nbin = int(dataset.shape[-2])
        histogram = np.zeros(nbin, dtype=np.float64)

        for repetition_index in range(int(dataset.shape[0])):
            for z_index in range(int(dataset.shape[1])):
                histogram += np.asarray(
                    dataset[repetition_index, z_index, :, :, :, channel_index],
                    dtype=np.float64,
                ).sum(axis=(0, 1))

        return histogram

    @staticmethod
    def _sum_dataset_over_non_channel_axes(dataset):
        channel_count = int(dataset.shape[-1])
        fingerprint = np.zeros(channel_count, dtype=np.float64)

        for repetition_index in range(int(dataset.shape[0])):
            for z_index in range(int(dataset.shape[1])):
                fingerprint += np.asarray(
                    dataset[repetition_index, z_index, :, :, :, :],
                    dtype=np.float64,
                ).sum(axis=(0, 1, 2))

        return fingerprint

    @staticmethod
    def _normalize_stack_to_fingerprint(stack, fingerprint):
        stack = np.asarray(stack, dtype=np.float64)
        fingerprint = np.asarray(fingerprint, dtype=np.float64)

        if stack.ndim != 2:
            raise ValueError(f"stack must have shape (t, ch), got {stack.shape}")
        if fingerprint.ndim != 1 or fingerprint.shape[0] != stack.shape[1]:
            raise ValueError(
                "fingerprint must be 1D with one entry per stack channel "
                f"(got {fingerprint.shape} for stack shape {stack.shape})"
            )

        column_sums = np.sum(stack, axis=0, keepdims=True)
        weighted = np.divide(
            stack,
            column_sums,
            out=np.zeros_like(stack, dtype=np.float64),
            where=np.isfinite(column_sums) & (column_sums > 0),
        )
        weighted *= fingerprint[np.newaxis, :]
        total = float(np.sum(weighted))
        if np.isfinite(total) and total > 0:
            weighted /= total
        else:
            weighted.fill(0.0)

        return weighted

    @staticmethod
    def _prepare_attr_value(value):
        if value is None:
            return "None"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return H5DataCalibrator._prepare_attr_value(value.item())
            if value.dtype.kind in {"i", "u", "f", "b"}:
                return value
            return json.dumps(value.tolist(), default=str)
        if isinstance(value, (list, tuple, dict, set)):
            return json.dumps(value, default=str)
        return str(value)

    @classmethod
    def _set_group_attrs(cls, group, attrs):
        for key, value in attrs.items():
            group.attrs[str(key)] = cls._prepare_attr_value(value)

    @classmethod
    def _write_metadata_group(cls, parent_group, group_name, metadata):
        metadata_group = parent_group.create_group(group_name)
        for key, value in cls._metadata_items(metadata):
            metadata_group.attrs[str(key)] = cls._prepare_attr_value(value)
        return metadata_group

    @staticmethod
    def _replace_dataset(group, name, data):
        if name in group:
            del group[name]
        array = np.asarray(data)
        kwargs = {}
        if array.ndim > 0 and array.size > 0:
            kwargs["compression"] = "gzip"
        return group.create_dataset(name, data=array, **kwargs)

    @staticmethod
    def _format_tau_ref_input(tau_ref):
        if tau_ref is None:
            return "None"
        if isinstance(tau_ref, str):
            return tau_ref
        try:
            return float(tau_ref)
        except (TypeError, ValueError):
            return str(tau_ref)

    @staticmethod
    def _std_from_variance(variance):
        try:
            variance = float(variance)
        except (TypeError, ValueError):
            return np.nan
        if not np.isfinite(variance):
            return np.nan
        return float(np.sqrt(max(variance, 0.0)))

    @classmethod
    def _parameter_error_payload(
        cls,
        covariance,
        dt_ns,
        param_names=None,
        amplitude_param="C",
        delay_param="dT",
        lifetime_param="tau",
    ):
        errors = {
            "fit_param_C_err": np.nan,
            "tau_err_ns": np.nan,
            "fit_common_delay_err_in_bins": np.nan,
            "fit_common_delay_err_in_ns": np.nan,
        }
        covariance = np.asarray(covariance, dtype=float)
        if param_names is None:
            param_names = ["C", "dT", "tau"]
        param_names = list(param_names)
        if covariance.shape != (len(param_names), len(param_names)):
            return errors

        diag = np.diag(covariance)
        if amplitude_param in param_names:
            errors["fit_param_C_err"] = cls._std_from_variance(
                diag[param_names.index(amplitude_param)]
            )
        if delay_param in param_names:
            errors["fit_common_delay_err_in_bins"] = cls._std_from_variance(
                diag[param_names.index(delay_param)]
            )
        if lifetime_param in param_names:
            errors["tau_err_ns"] = cls._std_from_variance(
                diag[param_names.index(lifetime_param)]
            )
        if np.isfinite(errors["fit_common_delay_err_in_bins"]) and np.isfinite(dt_ns):
            errors["fit_common_delay_err_in_ns"] = float(
                errors["fit_common_delay_err_in_bins"] * float(dt_ns)
            )
        return errors

    @staticmethod
    def _fit_error(data_for_fit, data_fitted):
        try:
            data_norm = Alignment._normalize_histogram_1d(
                data_for_fit,
                name="data_for_fit",
            )
        except ValueError:
            return np.nan

        data_fitted = np.asarray(data_fitted, dtype=float)
        if data_fitted.shape != data_norm.shape:
            return np.nan
        fit_sum = float(np.sum(data_fitted))
        if not np.isfinite(fit_sum) or fit_sum <= 0:
            return np.nan

        fit_norm = data_fitted / fit_sum
        residual = data_norm - fit_norm
        return float(np.sqrt(np.mean(np.square(residual))))

    @staticmethod
    def _compute_irf_correction_delays(fit_delay_in_bins, irf_corrections_type, data_key):
        fit_delay_in_bins = np.asarray(fit_delay_in_bins, dtype=float)
        correction_delay = np.full_like(fit_delay_in_bins, np.nan, dtype=float)
        finite_fit = np.isfinite(fit_delay_in_bins)

        if irf_corrections_type == "single_ch":
            correction_delay[finite_fit] = fit_delay_in_bins[finite_fit]
            return correction_delay

        if not np.any(finite_fit):
            warnings.warn(
                (
                    "Unable to compute median IRF correction delay for data key "
                    f"{data_key!r}: no finite fitted delays were found"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return correction_delay

        correction_delay[finite_fit] = float(np.nanmedian(fit_delay_in_bins[finite_fit]))
        return correction_delay

    @staticmethod
    def _realign_histogram_stack(stack, correction_delay_in_bins, output_name):
        stack = np.asarray(stack, dtype=float)
        correction_delay_in_bins = np.asarray(correction_delay_in_bins, dtype=float)
        if stack.ndim != 2:
            raise ValueError(
                f"{output_name} source stack must have shape (t, ch), got {stack.shape}"
            )
        if correction_delay_in_bins.shape != (stack.shape[1],):
            raise ValueError(
                f"{output_name} correction delay must have shape ({stack.shape[1]},), "
                f"got {correction_delay_in_bins.shape}"
            )

        realigned = np.zeros_like(stack, dtype=float)
        for channel_position, correction_delay in enumerate(correction_delay_in_bins):
            if not np.isfinite(correction_delay):
                continue
            hist = stack[:, channel_position]
            if not np.isfinite(hist).all() or np.sum(hist) <= 0:
                continue
            realigned[:, channel_position] = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(hist, correction_delay, cyclic=True),
                name=output_name,
            )
        return realigned

    @staticmethod
    def _empty_fit_payload(
        nbin,
        reference_type,
        data_for_fit_histogram,
        ref_for_fit_histogram,
        irf_type,
        param_names=None,
    ):
        if param_names is None:
            param_names = ["C", "dT", "tau"]
        param_count = len(param_names)
        zero_hist = np.zeros(int(nbin), dtype=float)
        payload = {
            "fit_param_C": np.nan,
            "fit_param_C_err": np.nan,
            "tau_ns": np.nan,
            "tau_err_ns": np.nan,
            "tau_ref_ns": np.nan,
            "fit_common_delay_in_bins": np.nan,
            "fit_common_delay_in_ns": np.nan,
            "fit_common_delay_err_in_bins": np.nan,
            "fit_common_delay_err_in_ns": np.nan,
            "fit_error": np.nan,
            "data_for_fit": np.asarray(data_for_fit_histogram, dtype=float),
            "irf_for_fit": zero_hist.copy(),
            "data_fitted": zero_hist.copy(),
            "fit_params": np.full(param_count, np.nan, dtype=float),
            "fit_param_errs": np.full(param_count, np.nan, dtype=float),
            "fit_covariance": np.full((param_count, param_count), np.nan, dtype=float),
            "irf_type": str(irf_type),
        }
        if reference_type == "ref":
            payload["ref_for_fit"] = np.asarray(ref_for_fit_histogram, dtype=float)
        return payload

    def _prepare_output_file(self):
        if self.output_path.resolve() == self.data_path.resolve():
            raise ValueError("output_path must be different from data_path")
        if self.output_path.exists():
            if not self.overwrite:
                raise FileExistsError(f"output file already exists: {self.output_path}")
            self.output_path.unlink()
        shutil.copy2(self.data_path, self.output_path)
        return self.output_path

    def _get_target_group(self, calibration_group, data_key):
        if data_key in calibration_group:
            del calibration_group[data_key]
        return calibration_group.create_group(data_key)

    @staticmethod
    def _channel_skew_source_attr_value(channel_skew_source):
        if isinstance(channel_skew_source, np.ndarray):
            return "ext"
        return str(channel_skew_source)

    def _resolve_channel_skew_reference_position(self, channel_index, data_key):
        channel_index = np.asarray(channel_index, dtype=int)
        matches = np.flatnonzero(channel_index == self.channel_skew_fit_reference_channel)
        if matches.size > 0:
            resolved_position = int(matches[0])
            return resolved_position, int(channel_index[resolved_position])

        if self.channel_skew_fit_reference_channel == DEFAULT_CHANNEL_SKEW_FIT_REFERENCE_CHANNEL:
            resolved_position = int(len(channel_index) // 2)
            return resolved_position, int(channel_index[resolved_position])

        raise ValueError(
            "channel_skew_fit_reference_channel="
            f"{self.channel_skew_fit_reference_channel} is not present in the calibrated "
            f"channel_index for data key {data_key!r}"
        )

    def _validate_channel_skew_configuration(self, channel_index, data_key):
        channel_index = np.asarray(channel_index, dtype=int)
        source = self.channel_skew_source

        if isinstance(source, np.ndarray):
            if source.shape != (len(channel_index),):
                raise ValueError(
                    "channel_skew_source array must have shape "
                    f"({len(channel_index)},) for data key {data_key!r}, got {source.shape}"
                )
            return

        if source == "metadata":
            raise NotImplementedError(
                "channel_skew_source='metadata' must still be implemented"
            )

        if source == "ref" and self.reference_type != "ref":
            raise ValueError(
                f"channel_skew_source='ref' is not available for data key {data_key!r} "
                f"when reference_type={self.reference_type!r}"
            )

    @staticmethod
    def _validate_channel_skew_source_histogram(calib, channel_count, data_key, source_name):
        calib = np.asarray(calib, dtype=float)
        if calib.ndim != 2 or calib.shape[1] != channel_count:
            raise ValueError(
                f"channel skew source {source_name!r} for data key {data_key!r} must have shape "
                f"(t, {channel_count}), got {calib.shape}"
            )
        return calib

    def _run_shift_vectors(self, calib, reference_position, data_key):
        reference_hist = calib[:, reference_position : reference_position + 1]
        shifts_1d, errors_1d, _ = estimate_channel_skew(
            calib,
            reference_hist=reference_hist,
            reference_channel=0,
            upsampling=self.channel_skew_fit_upsampling,
            apodize=self.channel_skew_fit_apodize,
        )

        shifts_1d = np.asarray(shifts_1d, dtype=float)
        errors_1d = np.asarray(errors_1d, dtype=float)
        if shifts_1d.shape != (calib.shape[1],) or errors_1d.shape != (calib.shape[1],):
            raise ValueError(
                f"estimate_channel_skew returned unexpected shapes for data key {data_key!r}: "
                f"shifts={shifts_1d.shape}, errors={errors_1d.shape}"
            )

        channel_numbers = np.arange(calib.shape[1], dtype=float)
        shifts = np.column_stack((channel_numbers, shifts_1d))
        errors = np.column_stack((np.full(calib.shape[1], np.nan), errors_1d))
        return shifts, errors

    def _compute_channel_skew(self, channel_index, source_histograms, data_key, channel_skew_cache):
        channel_index = np.asarray(channel_index, dtype=int)
        channel_count = int(len(channel_index))
        source = self.channel_skew_source

        if isinstance(source, np.ndarray):
            if source.shape != (channel_count,):
                raise ValueError(
                    "channel_skew_source array must have shape "
                    f"({channel_count},) for data key {data_key!r}, got {source.shape}"
                )
            return (
                np.asarray(source, dtype=float),
                np.full(channel_count, np.nan, dtype=float),
                {
                    "reference_data_key": data_key,
                    "reference_channel_resolved": np.nan,
                    "reference_position": np.nan,
                    "ext_data": np.asarray(source, dtype=float),
                },
            )

        if source == "metadata":
            raise NotImplementedError(
                "channel_skew_source='metadata' must still be implemented"
            )

        if source not in source_histograms:
            raise ValueError(
                f"channel_skew_source={source!r} is not available for data key {data_key!r}"
            )

        calib = self._validate_channel_skew_source_histogram(
            source_histograms[source],
            channel_count,
            data_key,
            source,
        )

        reference_data_key = data_key
        reference_channel_index = channel_index
        reference_sources = source_histograms

        if data_key != "data" and "data" in channel_skew_cache:
            reference_data_key = "data"
            reference_channel_index = np.asarray(channel_skew_cache["data"]["channel_index"], dtype=int)
            reference_sources = channel_skew_cache["data"]["sources"]

        if source not in reference_sources:
            raise ValueError(
                f"channel_skew_source={source!r} is not available in reference data key "
                f"{reference_data_key!r} for data key {data_key!r}"
            )

        reference_source_hist = self._validate_channel_skew_source_histogram(
            reference_sources[source],
            len(reference_channel_index),
            reference_data_key,
            source,
        )
        reference_position, reference_channel_resolved = self._resolve_channel_skew_reference_position(
            reference_channel_index,
            reference_data_key,
        )

        if reference_data_key == data_key:
            shift_input = calib
            shift_reference_position = reference_position
        else:
            shift_input = np.concatenate(
                [calib, reference_source_hist[:, reference_position : reference_position + 1]],
                axis=1,
            )
            shift_reference_position = channel_count

        shifts, errors = self._run_shift_vectors(
            shift_input,
            shift_reference_position,
            data_key,
        )

        return (
            shifts[:channel_count, 1],
            errors[:channel_count, 1],
            {
                "reference_data_key": reference_data_key,
                "reference_channel_resolved": reference_channel_resolved,
                "reference_position": reference_position,
                "ext_data": None,
            },
        )

    def calibrate(self):
        data_metadata = mcs.metadata_load(str(self.data_path))
        reference_metadata = mcs.metadata_load(str(self.reference_path))
        output_path = self._prepare_output_file()

        with h5py.File(self.data_path, "r") as data_handle, \
             h5py.File(self.reference_path, "r") as reference_handle, \
             h5py.File(output_path, "a") as output_handle:

            if self.calibration_key in output_handle:
                del output_handle[self.calibration_key]
            calibration_group = output_handle.create_group(self.calibration_key)

            self._set_group_attrs(
                calibration_group,
                {
                    "source_data_file": str(self.data_path),
                    "source_reference_file": str(self.reference_path),
                    "output_file": str(output_path),
                    "data_keys": json.dumps(self.data_keys),
                    "reference_keys": json.dumps(self.reference_key_map),
                    "reference_type": self.reference_type,
                    "tau_ref_input": self._format_tau_ref_input(self.tau_ref),
                    "fit_mode": self.fit_mode,
                    "fit_type": self.fit_type,
                    "C_ref": self.C_ref,
                    "irf_iterations": self.irf_iterations,
                    "eps": self.eps,
                    "regularization": self.regularization,
                    "clean_irf": self.clean_irf,
                    "irf_corrections_type": self.irf_corrections_type,
                    "initial_tau": self.initial_tau,
                    "initial_dT": self.initial_dT,
                    "initial_C": self.initial_C,
                    "force_C_normalized": self.force_C_normalized,
                    "fit_model_name": Alignment._callable_name(self.model_fn),
                    "fit_param_names": json.dumps(self.param_names),
                    "fit_p0": self.p0,
                    "fit_bounds": self.bounds,
                    "fit_model_kwargs": json.dumps(self.model_kwargs, default=str),
                    "fit_amplitude_param": self.amplitude_param,
                    "fit_delay_param": self.delay_param,
                    "fit_lifetime_param": self.lifetime_param,
                    "channel_skew_type": self.channel_skew_type,
                    "channel_skew_source": self._channel_skew_source_attr_value(
                        self.channel_skew_source
                    ),
                    "channel_skew_fit_reference_channel": self.channel_skew_fit_reference_channel,
                    "channel_skew_fit_upsampling": self.channel_skew_fit_upsampling,
                    "channel_skew_fit_apodize": self.channel_skew_fit_apodize,
                    "fit_error_metric": "rmse_normalized_histograms",
                    "data_key_count": len(self.data_keys),
                },
            )
            self._write_metadata_group(calibration_group, "input_data_metadata", data_metadata)
            self._write_metadata_group(calibration_group, "input_reference_metadata", reference_metadata)

            ordered_data_keys = list(self.data_keys)
            if "data" in ordered_data_keys:
                ordered_data_keys.insert(0, ordered_data_keys.pop(ordered_data_keys.index("data")))
            channel_skew_cache = {}

            data_key_iterator = tqdm(
                ordered_data_keys,
                desc="Calibrating data keys",
                unit="key",
                disable=len(ordered_data_keys) <= 1,
            )
            for data_key in data_key_iterator:
                reference_key = self.reference_key_map[data_key]
                data_dataset = self._open_dataset(data_handle, data_key)
                reference_dataset = self._open_dataset(reference_handle, reference_key)

                self._validate_dataset_layout(data_dataset, f"data[{data_key}]")
                self._validate_dataset_layout(reference_dataset, f"reference[{reference_key}]")

                if data_dataset.shape[-2] != reference_dataset.shape[-2]:
                    raise ValueError(
                        "data and reference datasets must have the same number of time bins "
                        f"(got {data_dataset.shape[-2]} and {reference_dataset.shape[-2]}) "
                        f"for data key {data_key!r}"
                    )

                channel_indices = self._resolve_channels(self.channels, int(data_dataset.shape[-1]))
                reference_channel_map = self._resolve_reference_channel_map(data_dataset, reference_dataset)
                if len(channel_indices) == 0:
                    raise ValueError("channels must contain at least one channel index")
                self._validate_channel_skew_configuration(channel_indices, data_key)

                nbin, dt_ns, period_ns, t_ns = self.build_time_axis(
                    data_metadata,
                    nbin=int(data_dataset.shape[-2]),
                    period_ns=self.period_ns,
                )
                laser_freq_in_mhz = 1e3 / period_ns

                target_group = self._get_target_group(calibration_group, data_key)
                self._set_group_attrs(
                    target_group,
                    {
                        "data_key": data_key,
                        "reference_key": reference_key,
                        "reference_type": self.reference_type,
                        "tau_ref_input": self._format_tau_ref_input(self.tau_ref),
                        "fit_mode": self.fit_mode,
                        "fit_type": self.fit_type,
                        "C_ref": self.C_ref,
                        "irf_iterations": self.irf_iterations,
                        "eps": self.eps,
                        "regularization": self.regularization,
                        "clean_irf": self.clean_irf,
                        "irf_corrections_type": self.irf_corrections_type,
                        "initial_tau": self.initial_tau,
                        "initial_dT": self.initial_dT,
                        "initial_C": self.initial_C,
                        "force_C_normalized": self.force_C_normalized,
                        "fit_model_name": Alignment._callable_name(self.model_fn),
                        "fit_param_names": json.dumps(self.param_names),
                        "fit_p0": self.p0,
                        "fit_bounds": self.bounds,
                        "fit_model_kwargs": json.dumps(self.model_kwargs, default=str),
                        "fit_amplitude_param": self.amplitude_param,
                        "fit_delay_param": self.delay_param,
                        "fit_lifetime_param": self.lifetime_param,
                        "channel_skew_type": self.channel_skew_type,
                        "channel_skew_source": self._channel_skew_source_attr_value(
                            self.channel_skew_source
                        ),
                        "channel_skew_fit_reference_channel": self.channel_skew_fit_reference_channel,
                        "channel_skew_fit_upsampling": self.channel_skew_fit_upsampling,
                        "channel_skew_fit_apodize": self.channel_skew_fit_apodize,
                        "number_of_bins": nbin,
                        "bin_width_in_ns": dt_ns,
                        "laser_period_in_ns": period_ns,
                        "laser_freq_in_mhz": laser_freq_in_mhz,
                        "channel_count": int(data_dataset.shape[-1]),
                        "channel_count_calibrated": len(channel_indices),
                        "data_shape": list(data_dataset.shape),
                        "reference_shape": list(reference_dataset.shape),
                        "channel_axis": -1,
                        "stacked_histogram_layout": "(t, ch)",
                        "fit_error_metric": "rmse_normalized_histograms",
                    },
                )
                reference_fingerprint = self._sum_dataset_over_non_channel_axes(reference_dataset)
                self._replace_dataset(target_group, "irf_fingerprint", reference_fingerprint)
                if self.reference_type == "ref":
                    self._replace_dataset(target_group, "ref_fingerprint", reference_fingerprint)

                stacked_channel_index = []
                stacked_channel_used_for_reference_in_time_skew = []
                stacked_fit_param_C = []
                stacked_fit_param_C_err = []
                stacked_tau_ns = []
                stacked_tau_err_ns = []
                stacked_tau_ref_ns = []
                stacked_fit_common_delay_in_bins = []
                stacked_fit_common_delay_in_ns = []
                stacked_fit_common_delay_err_in_bins = []
                stacked_fit_common_delay_err_in_ns = []
                stacked_fit_error = []
                stacked_irf_type = []
                stacked_data_for_fit = []
                stacked_irf_for_fit = []
                stacked_data_fitted = []
                stacked_ref_for_fit = []
                stacked_fit_params = []
                stacked_fit_param_errs = []
                stacked_fit_covariances = []

                channel_iterator = tqdm(
                    channel_indices,
                    desc=f"Calibrating {data_key}",
                    unit="ch",
                    leave=False,
                )
                for channel_index in channel_iterator:
                    reference_channel_for_fit = reference_channel_map[channel_index]
                    data_for_fit_histogram = self._sum_histogram_for_channel(data_dataset, channel_index)
                    ref_for_fit_histogram = self._sum_histogram_for_channel(
                        reference_dataset,
                        reference_channel_for_fit,
                    )
                    data_sum = float(np.sum(data_for_fit_histogram))
                    reference_sum = float(np.sum(ref_for_fit_histogram))

                    if not np.isfinite(data_sum) or data_sum <= 0:
                        warnings.warn(
                            (
                                f"Skipping calibration for data key {data_key!r}, channel {channel_index}: "
                                "data histogram has a non-positive or non-finite sum"
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        fit_payload = self._empty_fit_payload(
                            nbin=nbin,
                            reference_type=self.reference_type,
                            data_for_fit_histogram=data_for_fit_histogram,
                            ref_for_fit_histogram=ref_for_fit_histogram,
                            irf_type="skipped_zero_sum_data",
                            param_names=self.param_names,
                        )
                    elif not np.isfinite(reference_sum) or reference_sum <= 0:
                        warnings.warn(
                            (
                                f"Skipping calibration for data key {data_key!r}, channel {channel_index}: "
                                f"reference channel {reference_channel_for_fit} histogram has a non-positive "
                                "or non-finite sum"
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        fit_payload = self._empty_fit_payload(
                            nbin=nbin,
                            reference_type=self.reference_type,
                            data_for_fit_histogram=data_for_fit_histogram,
                            ref_for_fit_histogram=ref_for_fit_histogram,
                            irf_type="skipped_zero_sum_reference",
                            param_names=self.param_names,
                        )
                    else:
                        fit_kwargs = {
                            "t": t_ns,
                            "data": data_for_fit_histogram,
                            "period": period_ns,
                            "C_ref": self.C_ref,
                            "irf_output": "original",
                            "shift_output": "ref" if self.reference_type == "ref" else None,
                            "fit_type": self.fit_type,
                            "mode": self.fit_mode,
                            "initial_tau": self.initial_tau,
                            "initial_dT": self.initial_dT,
                            "initial_C": self.initial_C,
                            "force_C_normalized": self.force_C_normalized,
                            "model_fn": self.model_fn,
                            "p0": self.p0,
                            "bounds": self.bounds,
                            "param_names": self.param_names,
                            "model_kwargs": self.model_kwargs,
                            "amplitude_param": self.amplitude_param,
                            "delay_param": self.delay_param,
                            "lifetime_param": self.lifetime_param,
                            "irf_iterations": self.irf_iterations,
                            "eps": self.eps,
                            "regularization": self.regularization,
                        }
                        if self.reference_type == "ref":
                            fit_kwargs["ref"] = ref_for_fit_histogram
                            fit_kwargs["tau_ref"] = self.tau_ref
                        else:
                            fit_kwargs["irf"] = ref_for_fit_histogram

                        try:
                            fit_result = Alignment.fit_data_with_ref_or_irf(**fit_kwargs)
                            irf_for_fit = np.asarray(fit_result["irf"], dtype=float)
                            data_fitted = np.asarray(fit_result["fit"], dtype=float)
                            parameter_errors = self._parameter_error_payload(
                                fit_result["cov"],
                                dt_ns,
                                param_names=fit_result["param_names"],
                                amplitude_param=self.amplitude_param,
                                delay_param=self.delay_param,
                                lifetime_param=self.lifetime_param,
                            )
                            fit_payload = {
                                "fit_param_C": float(fit_result["C"]),
                                "fit_param_C_err": float(parameter_errors["fit_param_C_err"]),
                                "tau_ns": float(fit_result["tau"]),
                                "tau_err_ns": float(parameter_errors["tau_err_ns"]),
                                "tau_ref_ns": (
                                    np.nan
                                    if fit_result["tau_ref"] is None
                                    else float(fit_result["tau_ref"])
                                ),
                                "fit_common_delay_in_bins": float(fit_result["dT"]),
                                "fit_common_delay_in_ns": float(fit_result["dT_ns"]),
                                "fit_common_delay_err_in_bins": float(
                                    parameter_errors["fit_common_delay_err_in_bins"]
                                ),
                                "fit_common_delay_err_in_ns": float(
                                    parameter_errors["fit_common_delay_err_in_ns"]
                                ),
                                "fit_error": float(
                                    self._fit_error(data_for_fit_histogram, data_fitted)
                                ),
                                "data_for_fit": np.asarray(data_for_fit_histogram, dtype=float),
                                "irf_for_fit": np.asarray(irf_for_fit, dtype=float),
                                "data_fitted": data_fitted,
                                "fit_params": np.asarray(
                                    fit_result["param_values"],
                                    dtype=float,
                                ),
                                "fit_param_errs": np.asarray(
                                    fit_result["param_errors"],
                                    dtype=float,
                                ),
                                "fit_covariance": np.asarray(
                                    fit_result["cov"],
                                    dtype=float,
                                ),
                                "irf_type": str(fit_result["irf_source"]),
                            }
                            if self.reference_type == "ref":
                                fit_payload["ref_for_fit"] = np.asarray(
                                    ref_for_fit_histogram,
                                    dtype=float,
                                )
                        except Exception as exc:
                            warnings.warn(
                                (
                                    f"Calibration fit failed for data key {data_key!r}, channel "
                                    f"{channel_index}: {exc}"
                                ),
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            fit_payload = self._empty_fit_payload(
                                nbin=nbin,
                                reference_type=self.reference_type,
                                data_for_fit_histogram=data_for_fit_histogram,
                                ref_for_fit_histogram=ref_for_fit_histogram,
                                irf_type="fit_failed",
                                param_names=self.param_names,
                            )

                    stacked_channel_index.append(channel_index)
                    stacked_channel_used_for_reference_in_time_skew.append(reference_channel_for_fit)
                    stacked_fit_param_C.append(float(fit_payload["fit_param_C"]))
                    stacked_fit_param_C_err.append(float(fit_payload["fit_param_C_err"]))
                    stacked_tau_ns.append(float(fit_payload["tau_ns"]))
                    stacked_tau_err_ns.append(float(fit_payload["tau_err_ns"]))
                    stacked_tau_ref_ns.append(float(fit_payload["tau_ref_ns"]))
                    stacked_fit_common_delay_in_bins.append(
                        float(fit_payload["fit_common_delay_in_bins"])
                    )
                    stacked_fit_common_delay_in_ns.append(
                        float(fit_payload["fit_common_delay_in_ns"])
                    )
                    stacked_fit_common_delay_err_in_bins.append(
                        float(fit_payload["fit_common_delay_err_in_bins"])
                    )
                    stacked_fit_common_delay_err_in_ns.append(
                        float(fit_payload["fit_common_delay_err_in_ns"])
                    )
                    stacked_fit_error.append(float(fit_payload["fit_error"]))
                    stacked_irf_type.append(str(fit_payload["irf_type"]))
                    stacked_data_for_fit.append(np.asarray(fit_payload["data_for_fit"], dtype=float))
                    stacked_irf_for_fit.append(np.asarray(fit_payload["irf_for_fit"], dtype=float))
                    stacked_data_fitted.append(np.asarray(fit_payload["data_fitted"], dtype=float))
                    stacked_fit_params.append(np.asarray(fit_payload["fit_params"], dtype=float))
                    stacked_fit_param_errs.append(
                        np.asarray(fit_payload["fit_param_errs"], dtype=float)
                    )
                    stacked_fit_covariances.append(
                        np.asarray(fit_payload["fit_covariance"], dtype=float)
                    )
                    if self.reference_type == "ref":
                        stacked_ref_for_fit.append(np.asarray(fit_payload["ref_for_fit"], dtype=float))

                channel_index_array = np.asarray(stacked_channel_index, dtype=int)
                channel_used_for_reference_in_time_skew_array = np.asarray(
                    stacked_channel_used_for_reference_in_time_skew,
                    dtype=int,
                )
                reference_fingerprint_for_output_channels = reference_fingerprint[
                    channel_used_for_reference_in_time_skew_array
                ]
                fit_param_C_array = np.asarray(stacked_fit_param_C, dtype=float)
                fit_param_C_err_array = np.asarray(stacked_fit_param_C_err, dtype=float)
                tau_ns_array = np.asarray(stacked_tau_ns, dtype=float)
                tau_err_ns_array = np.asarray(stacked_tau_err_ns, dtype=float)
                tau_ref_ns_array = np.asarray(stacked_tau_ref_ns, dtype=float)
                fit_common_delay_in_bins_array = np.asarray(
                    stacked_fit_common_delay_in_bins,
                    dtype=float,
                )
                fit_common_delay_in_ns_array = np.asarray(
                    stacked_fit_common_delay_in_ns,
                    dtype=float,
                )
                common_delay_in_bins_array = self._compute_irf_correction_delays(
                    fit_common_delay_in_bins_array,
                    self.irf_corrections_type,
                    data_key,
                )
                common_delay_in_ns_array = common_delay_in_bins_array * float(dt_ns)
                fit_common_delay_err_in_bins_array = np.asarray(
                    stacked_fit_common_delay_err_in_bins,
                    dtype=float,
                )
                fit_common_delay_err_in_ns_array = np.asarray(
                    stacked_fit_common_delay_err_in_ns,
                    dtype=float,
                )
                fit_error_array = np.asarray(stacked_fit_error, dtype=float)
                data_for_fit_stack = np.stack(stacked_data_for_fit, axis=-1)
                irf_for_fit_stack = np.stack(stacked_irf_for_fit, axis=-1)
                irf_common_delay_realigned_stack = self._realign_histogram_stack(
                    irf_for_fit_stack,
                    common_delay_in_bins_array,
                    "irf_common_delay_realigned",
                )
                if self.clean_irf and self.reference_type == "irf":
                    irf_common_delay_realigned_stack = Alignment.clean_irf_stack(
                        irf_common_delay_realigned_stack,
                        threshold=0.3,
                        window=2.0 / dt_ns,
                        time_axis=0,
                        normalize=True,
                    )
                irf_common_delay_realigned_stack = self._normalize_stack_to_fingerprint(
                    irf_common_delay_realigned_stack,
                    reference_fingerprint_for_output_channels,
                )
                data_fitted_stack = np.stack(stacked_data_fitted, axis=-1)
                fit_params_array = np.stack(stacked_fit_params, axis=0)
                fit_param_errs_array = np.stack(stacked_fit_param_errs, axis=0)
                fit_covariances_array = np.stack(stacked_fit_covariances, axis=0)

                self._replace_dataset(target_group, "channel_index", channel_index_array)
                self._replace_dataset(
                    target_group,
                    "channel_used_for_reference_in_time_skew",
                    channel_used_for_reference_in_time_skew_array,
                )
                self._replace_dataset(target_group, "fit_param_C", fit_param_C_array)
                self._replace_dataset(target_group, "fit_param_C_err", fit_param_C_err_array)
                self._replace_dataset(target_group, "tau_ns", tau_ns_array)
                self._replace_dataset(target_group, "tau_err_ns", tau_err_ns_array)
                self._replace_dataset(target_group, "tau_ref_ns", tau_ref_ns_array)
                self._replace_dataset(
                    target_group,
                    "fit_common_delay_in_bins",
                    fit_common_delay_in_bins_array,
                )
                self._replace_dataset(
                    target_group,
                    "fit_common_delay_in_ns",
                    fit_common_delay_in_ns_array,
                )
                self._replace_dataset(
                    target_group,
                    "common_delay_in_bins",
                    common_delay_in_bins_array,
                )
                self._replace_dataset(
                    target_group,
                    "common_delay_in_ns",
                    common_delay_in_ns_array,
                )
                self._replace_dataset(
                    target_group,
                    "fit_common_delay_err_in_bins",
                    fit_common_delay_err_in_bins_array,
                )
                self._replace_dataset(
                    target_group,
                    "fit_common_delay_err_in_ns",
                    fit_common_delay_err_in_ns_array,
                )
                self._replace_dataset(
                    target_group,
                    "fit_error",
                    fit_error_array,
                )
                self._replace_dataset(
                    target_group,
                    "data_for_fit",
                    data_for_fit_stack,
                )
                self._replace_dataset(
                    target_group,
                    "irf_for_fit",
                    irf_for_fit_stack,
                )
                self._replace_dataset(
                    target_group,
                    "irf_common_delay_realigned",
                    irf_common_delay_realigned_stack,
                )
                self._replace_dataset(
                    target_group,
                    "data_fitted",
                    data_fitted_stack,
                )
                string_dtype = h5py.string_dtype(encoding="utf-8")
                if "fit_param_names" in target_group:
                    del target_group["fit_param_names"]
                target_group.create_dataset(
                    "fit_param_names",
                    data=np.asarray(self.param_names, dtype=object),
                    dtype=string_dtype,
                )
                self._replace_dataset(target_group, "fit_params", fit_params_array)
                self._replace_dataset(target_group, "fit_param_errs", fit_param_errs_array)
                self._replace_dataset(target_group, "fit_covariances", fit_covariances_array)

                channel_skew_sources = {
                    "data": data_for_fit_stack,
                    "irf": irf_common_delay_realigned_stack,
                }
                if self.reference_type == "ref":
                    ref_for_fit_stack = np.stack(stacked_ref_for_fit, axis=-1)
                    ref_common_delay_realigned_stack = self._realign_histogram_stack(
                        ref_for_fit_stack,
                        common_delay_in_bins_array,
                        "ref_common_delay_realigned",
                    )
                    ref_common_delay_realigned_stack = self._normalize_stack_to_fingerprint(
                        ref_common_delay_realigned_stack,
                        reference_fingerprint_for_output_channels,
                    )
                    self._replace_dataset(
                        target_group,
                        "ref_for_fit",
                        ref_for_fit_stack,
                    )
                    self._replace_dataset(
                        target_group,
                        "ref_common_delay_realigned",
                        ref_common_delay_realigned_stack,
                    )
                    channel_skew_sources["ref"] = ref_common_delay_realigned_stack

                (
                    channel_skew,
                    channel_skew_est_err,
                    channel_skew_reference_info,
                ) = self._compute_channel_skew(
                    channel_index_array,
                    channel_skew_sources,
                    data_key,
                    channel_skew_cache,
                )
                target_group.attrs["channel_skew_fit_reference_data_key"] = str(
                    channel_skew_reference_info["reference_data_key"]
                )
                target_group.attrs[
                    "channel_skew_fit_reference_channel_resolved"
                ] = channel_skew_reference_info["reference_channel_resolved"]
                target_group.attrs["channel_skew_fit_reference_position"] = (
                    channel_skew_reference_info["reference_position"]
                )
                channel_skew = np.asarray(channel_skew, dtype=float)
                channel_skew_est_err = np.asarray(channel_skew_est_err, dtype=float)
                self._replace_dataset(
                    target_group,
                    "channel_skew",
                    channel_skew,
                )
                self._replace_dataset(
                    target_group,
                    "channel_skew_est_err",
                    channel_skew_est_err,
                )
                ext_data = channel_skew_reference_info.get("ext_data")
                if ext_data is None:
                    if "channel_skew_ext_data" in target_group:
                        del target_group["channel_skew_ext_data"]
                else:
                    self._replace_dataset(
                        target_group,
                        "channel_skew_ext_data",
                        np.asarray(ext_data, dtype=float),
                    )
                string_dtype = h5py.string_dtype(encoding="utf-8")
                if "irf_type" in target_group:
                    del target_group["irf_type"]
                target_group.create_dataset(
                    "irf_type",
                    data=np.asarray(stacked_irf_type, dtype=object),
                    dtype=string_dtype,
                )

                target_group.attrs["data_key_group_mode"] = "nested_under_calibration"
                channel_skew_cache[data_key] = {
                    "channel_index": channel_index_array.copy(),
                    "sources": {
                        key: np.asarray(value, dtype=float).copy()
                        for key, value in channel_skew_sources.items()
                    },
                }

        return str(output_path)


def calibrate_h5_file(
    data_path,
    reference_path,
    data_key=DEFAULT_DATA_KEY,
    reference_key=DEFAULT_REFERENCE_KEY,
    reference_type=DEFAULT_REFERENCE_TYPE,
    tau_ref=DEFAULT_TAU_REF,
    fit_mode=DEFAULT_FIT_MODE,
    fit_type=DEFAULT_FIT_TYPE,
    C_ref=DEFAULT_C_REF,
    irf_iterations=DEFAULT_IRF_ITERATIONS,
    regularization=DEFAULT_REGULARIZATION,
    clean_irf=DEFAULT_CLEAN_IRF,
    irf_corrections_type=DEFAULT_IRF_CORRECTIONS_TYPE,
    channel_skew_type=DEFAULT_CHANNEL_SKEW_TYPE,
    channel_skew_source=DEFAULT_CHANNEL_SKEW_SOURCE,
    channel_skew_fit_reference_channel=DEFAULT_CHANNEL_SKEW_FIT_REFERENCE_CHANNEL,
    channel_skew_fit_upsampling=DEFAULT_CHANNEL_SKEW_FIT_UPSAMPLING,
    channel_skew_fit_apodize=DEFAULT_CHANNEL_SKEW_FIT_APODIZE,
    overwrite=DEFAULT_OVERWRITE,
    model_fn=None,
    p0=None,
    bounds=None,
    param_names=None,
    model_kwargs=None,
    amplitude_param="C",
    delay_param="dT",
    lifetime_param="tau",
    **kwargs,
):
    """
    Calibrate an HDF5 FLIM file against a reference file.

    This is a convenience wrapper around :class:`H5DataCalibrator` that exposes
    the most commonly used calibration parameters directly and forwards any
    extra keyword arguments to the class constructor.

    Parameters
    ----------
    data_path : str or path-like
        HDF5 file containing the data to calibrate.
    reference_path : str or path-like
        HDF5 file containing the reference histogram or IRF source.
    data_key : str or iterable of str, default ``("data", "data_channels_extra")``
        Dataset key or keys to calibrate from ``data_path``.
    reference_key : None, str or iterable of str or dict, default ``None``
        Dataset key or keys to read from ``reference_path``. When ``None``,
        each entry in ``data_key`` is matched to the same key name in the
        reference file. A dict can also be used to map each data key to a
        different reference key.
    reference_type : {"ref", "irf"}, default ``"ref"``
        Type of reference input used during calibration.
    tau_ref : float or None, default ``None``
        Reference lifetime in ns. If ``None``, it is estimated from the
        reference data when needed.
    fit_mode : str, default ``"model_shift"``
        Fitting mode forwarded to the alignment routines.
    fit_type : {"likelihood", "curve_fit_circular", "curve_fit"}, default ``"likelihood"``
        Fitting backend forwarded to the alignment routines.
    C_ref : float, default ``1.0``
        Reference amplitude scaling factor.
    irf_iterations : int, default ``300``
        Number of iterations used when estimating the IRF.
    regularization : float, default ``0``
        Regularization strength used during IRF estimation.
    clean_irf : bool, default ``False``
        If ``True`` and ``reference_type="irf"``, apply
        :meth:`Alignment.clean_irf_stack` to the realigned IRF stack using the
        historical notebook settings before it is rescaled for output.
    irf_corrections_type : {"median", "single_ch"}, default ``"median"``
        Strategy used to choose the delay applied to the realigned IRF/reference
        stacks. Median mode stores the raw fitted per-channel delay separately
        from the correction delay actually used.
    channel_skew_type : {"phase_cross_correlation"}, default ``"phase_cross_correlation"``
        Strategy used to generate ``channel_skew`` outputs. Only
        ``"phase_cross_correlation"`` is currently supported.
    channel_skew_source : {"ref", "irf", "data", "metadata"} or numpy.ndarray, default ``"ref"``
        Input used for channel-skew generation. String values select a
        calibration histogram, while a 1D NumPy array forces the final
        ``channel_skew`` values directly. When the ``data`` key is
        present, non-``data`` groups are anchored by default to the selected
        reference channel from the ``data`` group. ``"metadata"`` is reserved
        and currently raises ``NotImplementedError``. If a NumPy array is
        supplied, the stored HDF5 attribute value becomes ``"ext"`` and the
        input vector is also written to ``channel_skew_ext_data``.
    channel_skew_fit_reference_channel : int, default ``12``
        Reference channel index used when computing channel skew with the local
        estimator. When the default value ``12`` is not present, the
        middle calibrated channel is used automatically.
    channel_skew_fit_upsampling : int, default ``10``
        Upsampling factor forwarded to the local channel-skew estimator.
    channel_skew_fit_apodize : bool, default ``False``
        Apodization flag forwarded to the local channel-skew estimator.
    overwrite : bool, default ``True``
        If ``True``, overwrite an existing output file.
    model_fn, p0, bounds, param_names, model_kwargs : optional
        Optional custom full-model fit configuration. ``model_fn`` receives
        ``(t, irf, period, *params)`` and returns the fitted histogram.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`H5DataCalibrator`, including ``output_path=None``,
        ``channels=None``, ``calibration_key="calibration"``,
        ``period_ns=None``, ``initial_tau=None``, ``initial_dT=None``,
        ``initial_C=None``, ``force_C_normalized=False``, and ``eps=1e-8``.

    Returns
    -------
    str
        Path to the calibrated output HDF5 file. Each calibrated dataset is
        written under ``<calibration_key>/<data_key>/``.
    """
    return H5DataCalibrator(
        data_path,
        reference_path,
        data_key=data_key,
        reference_key=reference_key,
        reference_type=reference_type,
        tau_ref=tau_ref,
        fit_mode=fit_mode,
        fit_type=fit_type,
        C_ref=C_ref,
        irf_iterations=irf_iterations,
        regularization=regularization,
        clean_irf=clean_irf,
        irf_corrections_type=irf_corrections_type,
        channel_skew_type=channel_skew_type,
        channel_skew_source=channel_skew_source,
        channel_skew_fit_reference_channel=channel_skew_fit_reference_channel,
        channel_skew_fit_upsampling=channel_skew_fit_upsampling,
        channel_skew_fit_apodize=channel_skew_fit_apodize,
        overwrite=overwrite,
        model_fn=model_fn,
        p0=p0,
        bounds=bounds,
        param_names=param_names,
        model_kwargs=model_kwargs,
        amplitude_param=amplitude_param,
        delay_param=delay_param,
        lifetime_param=lifetime_param,
        **kwargs,
    ).calibrate()


def show_h5_structure(file_path, include_attrs=True, attrs_inline=False):
    """
    Return and print a readable tree view of an HDF5 file structure.

    Parameters
    ----------
    file_path : str or path-like
        HDF5 file to inspect.
    include_attrs : bool, default True
        If ``True``, include group and dataset attributes in the output.
    attrs_inline : bool, default False
        If ``True``, print attributes as indented ``.attrs.<name> = value``
        lines directly below each group or dataset.
    """
    lines = []

    def append_attrs(node, level, node_name=None):
        if not include_attrs:
            return
        attrs_items = list(node.attrs.items())
        if not attrs_items:
            return
        indent = "  " * level
        if attrs_inline:
            prefix = f"{node_name}.attrs" if node_name else ".attrs"
            joined = ", ".join(f"{key}={value!r}" for key, value in attrs_items)
            lines.append(f"{indent}{prefix}: {joined}")
            return
        for key, value in attrs_items:
            lines.append(f"{indent}@{key} = {value!r}")

    def visit(name, obj):
        if name == "":
            lines.append("/")
            append_attrs(obj, 1)
            return

        level = name.count("/")
        prefix = "  " * level
        node_name = name.split("/")[-1]
        if isinstance(obj, h5py.Group):
            lines.append(f"{prefix}{node_name}/")
            append_attrs(obj, level + 1, node_name=node_name)
        elif isinstance(obj, h5py.Dataset):
            lines.append(
                f"{prefix}{node_name} shape={obj.shape} dtype={obj.dtype}"
            )
            append_attrs(obj, level + 1, node_name=node_name)

    with h5py.File(file_path, "r") as handle:
        visit("", handle)
        handle.visititems(visit)

    structure = "\n".join(lines)
    print(structure)
    return structure


def show_h5_structure_html(file_path, include_attrs=True, attrs_inline=True, display_output=True):
    """
    Return an HTML tree view of an HDF5 file structure.

    Parameters
    ----------
    file_path : str or path-like
        HDF5 file to inspect.
    include_attrs : bool, default True
        If ``True``, include group and dataset attributes in the output.
    attrs_inline : bool, default True
        If ``True``, render all attributes for a node on one line.
    display_output : bool, default True
        If ``True`` and IPython is available, display the HTML immediately.
    """

    def format_value(value):
        value_repr = escape(repr(value))
        if isinstance(value, np.ndarray):
            return (
                f"{value_repr} "
                f"<span class='h5-array-shape'>shape={escape(str(value.shape))}</span>"
            )
        return value_repr

    def render_attrs(node, node_name):
        if not include_attrs or len(node.attrs) == 0:
            return ""

        items = list(node.attrs.items())
        if attrs_inline:
            joined = ", ".join(
                f"<span class='h5-attr-key'>{escape(str(key))}</span>="
                f"<span class='h5-attr-value'>{format_value(value)}</span>"
                for key, value in items
            )
            return (
                f"<div class='h5-attrs-inline'>"
                f"<span class='h5-attrs-prefix'>"
                f"<span class='h5-node-ref'>{escape(node_name)}.attrs</span>:"
                f"</span>"
                f"<span class='h5-attrs-content'>{joined}</span>"
                f"</div>"
            )

        parts = ["<ul class='h5-attrs-list'>"]
        for key, value in items:
            parts.append(
                "<li>"
                f"<span class='h5-node-ref'>{escape(node_name)}.attrs.</span>"
                f"<span class='h5-attr-key'>{escape(str(key))}</span>"
                f" = <span class='h5-attr-value'>{format_value(value)}</span>"
                "</li>"
            )
        parts.append("</ul>")
        return "".join(parts)

    def render_node(name, obj):
        node_name = name.split("/")[-1] if name else "/"
        if isinstance(obj, h5py.Dataset):
            label = (
                f"<span class='h5-dataset'>{escape(node_name)}</span> "
                f"<span class='h5-meta'>shape={escape(str(obj.shape))} "
                f"dtype={escape(str(obj.dtype))}</span>"
            )
            attrs_html = render_attrs(obj, node_name)
            return f"<li>{label}{attrs_html}</li>"

        label = f"<span class='h5-group'>{escape(node_name)}</span>/"
        attrs_html = render_attrs(obj, node_name)
        children = []
        for child_name in obj.keys():
            children.append(render_node(child_name, obj[child_name]))
        children_html = ""
        if children:
            children_html = f"<ul>{''.join(children)}</ul>"
        return (
            "<li>"
            "<details class='h5-branch' open>"
            f"<summary>{label}</summary>"
            f"{attrs_html}"
            f"{children_html}"
            "</details>"
            "</li>"
        )

    with h5py.File(file_path, "r") as handle:
        children = [render_node(name, handle[name]) for name in handle.keys()]
        root_attrs_html = render_attrs(handle, "/")

    html_output = f"""
<div class="h5-tree">
  <style>
    .h5-tree {{
      color-scheme: light dark;
      font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
      font-size: 13px;
      line-height: 1.5;
      color: var(--h5-fg);
      --h5-fg: #1f2937;
      --h5-muted: #6b7280;
      --h5-border: #d1d5db;
      --h5-group: #0f766e;
      --h5-dataset: #1d4ed8;
      --h5-attrs: #7c2d12;
      --h5-node-ref: #7c3aed;
      --h5-attr-key: #b45309;
      --h5-attr-value: #374151;
      --h5-root: #111827;
    }}
    @media (prefers-color-scheme: dark) {{
      .h5-tree {{
        --h5-fg: #e5e7eb;
        --h5-muted: #9ca3af;
        --h5-border: #4b5563;
        --h5-group: #5eead4;
        --h5-dataset: #93c5fd;
        --h5-attrs: #fdba74;
        --h5-node-ref: #c4b5fd;
        --h5-attr-key: #fbbf24;
        --h5-attr-value: #f3f4f6;
        --h5-root: #f9fafb;
      }}
    }}
    .h5-tree ul {{
      list-style: none;
      margin: 0.2rem 0 0.2rem 1.1rem;
      padding-left: 1rem;
      border-left: 1px solid var(--h5-border);
    }}
    .h5-tree li {{
      margin: 0.2rem 0;
    }}
    .h5-tree summary {{
      cursor: pointer;
      list-style: none;
    }}
    .h5-tree summary::-webkit-details-marker {{
      display: none;
    }}
    .h5-branch > summary::before {{
      content: "▾";
      color: var(--h5-muted);
      display: inline-block;
      width: 1rem;
    }}
    .h5-branch:not([open]) > summary::before {{
      content: "▸";
    }}
    .h5-group {{
      color: var(--h5-group);
      font-weight: 700;
    }}
    .h5-dataset {{
      color: var(--h5-dataset);
      font-weight: 700;
    }}
    .h5-meta {{
      color: var(--h5-muted);
      font-weight: 500;
    }}
    .h5-attrs-inline, .h5-attrs-list {{
      margin-top: 0.15rem;
      color: var(--h5-attrs);
    }}
    .h5-attrs-inline {{
      display: grid;
      grid-template-columns: max-content 1fr;
      column-gap: 0.45rem;
      align-items: start;
    }}
    .h5-attrs-prefix {{
      white-space: nowrap;
    }}
    .h5-attrs-content {{
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .h5-node-ref {{
      color: var(--h5-node-ref);
      font-weight: 700;
    }}
    .h5-attr-key {{
      color: var(--h5-attr-key);
      font-weight: 700;
    }}
    .h5-attr-value {{
      color: var(--h5-attr-value);
    }}
    .h5-array-shape {{
      color: var(--h5-muted);
      font-weight: 500;
    }}
    .h5-root {{
      color: var(--h5-root);
      font-weight: 800;
    }}
  </style>
  <div class="h5-root">/</div>
  {root_attrs_html}
  <ul>
    {''.join(children)}
  </ul>
</div>
""".strip()

    if display_output:
        try:
            from IPython.display import HTML, display
            display(HTML(html_output))
        except ImportError:
            pass

    return html_output

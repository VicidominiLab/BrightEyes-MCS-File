"""HDF5 calibration, output-building, and structure inspection helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import textwrap
from html import escape
from typing import Optional
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
from .h5_file_hash import channel_fingerprint_file_hash_attrs
from .h5_output_writers import ensure_output_group

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    PackageNotFoundError = Exception
    version = None

__all__ = [
    "H5DataCalibrator",
    "H5OutputBuilder",
    "add_output_to_h5_file",
    "build_h5_output",
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
BRIGHTEYES_H5_DATA_FORMAT_VERSION = "0.0.6"
BRIGHTEYES_H5_SCHEMA_NAME = "brighteyes_mcs_file"
BRIGHTEYES_H5_SCHEMA_VARIANT = "unified_metadata_axes"
BRIGHTEYES_H5_DATA_PATH = "/raw"
BRIGHTEYES_H5_METADATA_PATH = "/raw/metadata"
BRIGHTEYES_H5_AXES_PATH = "/raw/axes"
BRIGHTEYES_H5_LEGACY_PATH = "/raw/legacy"
BRIGHTEYES_H5_CALIBRATION_PATH = "/calibration"
BRIGHTEYES_H5_OUTPUT_PATH = "/output"

DATASET_KEY_ALIASES = {
    "data": ("data",),
    "spad": ("raw/spad",),
    "raw/spad": ("raw/spad",),
    "data_channels_extra": ("data_channels_extra",),
    "aux": ("raw/aux",),
    "raw/aux": ("raw/aux",),
    "data_analog": ("data_analog",),
    "analog": ("raw/analog",),
    "raw/analog": ("raw/analog",),
    "thumbnail": ("thumbnail",),
}


def _dataset_key_candidates(key):
    normalized = str(key).strip("/")
    aliases = DATASET_KEY_ALIASES.get(normalized, (normalized,))
    candidates = []
    for candidate in (normalized, *aliases):
        candidate = str(candidate).strip("/")
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


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
    p0, bounds, parameter_names, model_kwargs : optional
        Initial values, bounds, names, and extra keyword arguments for
        ``model_fn``. ``p0`` is required when ``model_fn`` is provided.
    amplitude_param, delay_param, lifetime_param : str
        Parameter names used to populate result datasets. Custom HDF5
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
        :meth:`Alignment.clean_irf_stack` to the aligned IRF stack using the
        historical notebook settings before it is rescaled for output.
    irf_corrections_type : {"median", "single_ch"}, default ``"median"``
        Strategy used to choose the delay applied when building the aligned
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
    create_output : bool, default ``True``
        If ``True``, build the ``/output`` analysis group in the calibrated
        file after writing ``/calibration``.
    output_options : dict or None, default ``None``
        Optional keyword arguments forwarded to :class:`H5OutputBuilder` when
        ``create_output=True``. The output builder always writes in place to
        the calibrated file; any nested ``output_path`` option is ignored.

    Notes
    -----
    The input datasets are expected to have channel-last 6D or 8D acquisition
    layout. Only one channel histogram at a time is materialized in memory, so
    the whole dataset is never converted to a NumPy array up front. Calibration
    results are written under ``<calibration_key>/results/<product>/``.
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
        parameter_names=None,
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
        create_output=True,
        output_options=None,
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
        if param_names is not None:
            if parameter_names is not None and list(parameter_names) != list(param_names):
                raise ValueError("parameter_names and param_names cannot disagree")
            parameter_names = param_names
        self.model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self.amplitude_param = str(amplitude_param)
        self.delay_param = str(delay_param)
        self.lifetime_param = str(lifetime_param)
        _, self.parameter_names = Alignment._resolve_fit_setup(
            self.model_fn,
            self.p0,
            parameter_names,
            self.initial_C,
            self.initial_dT,
            self.initial_tau,
        )
        if self.model_fn is not None and self.delay_param not in self.parameter_names:
            raise ValueError(
                f"parameter_names must include delay_param={self.delay_param!r} "
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
        self.create_output = bool(create_output)
        self.output_options = {} if output_options is None else dict(output_options)

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
            metadata_nbin = H5DataCalibrator._metadata_first(
                metadata,
                ("time_bins", "digital_time_bins", "dfd_nbins", "nbin"),
            )
            if metadata_nbin is None:
                raise ValueError(
                    "metadata must provide time_bins, digital_time_bins, dfd_nbins, "
                    "or nbin, or nbin must be passed explicitly"
                )
            nbin = int(metadata_nbin)
        else:
            nbin = int(nbin)

        if nbin <= 0:
            raise ValueError("nbin must be positive")

        if period_ns is None:
            laser_frequency_mhz = H5DataCalibrator._metadata_float_first(
                metadata,
                ("laser_frequency_mhz", "laser_freq_mhz", "dfd_freq"),
            )
            if np.isfinite(laser_frequency_mhz) and laser_frequency_mhz > 0:
                period_ns = 1e3 / laser_frequency_mhz
                dt_ns = period_ns / nbin
                t_ns = np.arange(nbin, dtype=float) * dt_ns
                return nbin, dt_ns, period_ns, t_ns

            pixel_dwell_time_us = H5DataCalibrator._metadata_float_first(
                metadata,
                ("pixel_dwell_time_us", "pixel_dwell_time_in_us", "pxdwelltime"),
            )
            if np.isfinite(pixel_dwell_time_us) and pixel_dwell_time_us > 0:
                dt_ns = pixel_dwell_time_us * 1000.0 / nbin
                period_ns = dt_ns * nbin
                t_ns = np.arange(nbin, dtype=float) * dt_ns
                return nbin, dt_ns, period_ns, t_ns

            time_resolution_us = H5DataCalibrator._metadata_float_first(
                metadata,
                ("time_resolution_us", "time_resolution", "dt"),
            )
            if np.isfinite(time_resolution_us) and time_resolution_us > 0:
                dt_ns = time_resolution_us * 1000.0
                period_ns = dt_ns * nbin
                t_ns = np.arange(nbin, dtype=float) * dt_ns
                return nbin, dt_ns, period_ns, t_ns

            raise ValueError(
                "metadata must provide laser_frequency_mhz/dfd_freq, "
                "pixel_dwell_time_us, time_resolution_us, or period_ns"
            )
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
            candidates = _dataset_key_candidates(key)
        else:
            candidates = tuple(
                candidate
                for default_key in H5DataCalibrator.DEFAULT_DATA_KEYS
                for candidate in _dataset_key_candidates(default_key)
            )

        seen = []
        non_dataset_matches = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.append(candidate)
            if candidate not in handle:
                continue
            dataset = handle[candidate]
            if isinstance(dataset, h5py.Dataset):
                return dataset
            non_dataset_matches.append(candidate)

        tried = ", ".join(seen)
        if non_dataset_matches:
            raise TypeError(
                f"dataset key {key!r} in {handle.filename!r} resolved only to "
                f"non-dataset HDF5 nodes: {', '.join(non_dataset_matches)}"
            )
        if key is None:
            raise KeyError(f"no default dataset found in {handle.filename!r}; tried {tried}")
        raise KeyError(f"dataset key {key!r} not found in {handle.filename!r}; tried {tried}")

    @staticmethod
    def _validate_dataset_layout(dataset, name):
        if dataset.ndim not in {6, 8}:
            raise ValueError(
                f"{name} dataset must be 6D or 8D with time and channel as the final axes, "
                f"got {dataset.shape}"
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
        prefix_shape = tuple(int(size) for size in dataset.shape[:2])
        prefix_slice = (slice(None),) * max(dataset.ndim - 4, 0)

        for prefix in np.ndindex(prefix_shape):
            block = np.asarray(
                dataset[prefix + prefix_slice + (slice(None), channel_index)],
                dtype=np.float64,
            )
            histogram += block.sum(axis=tuple(range(block.ndim - 1)))

        return histogram

    @staticmethod
    def _sum_dataset_over_non_channel_axes(dataset):
        channel_count = int(dataset.shape[-1])
        fingerprint = np.zeros(channel_count, dtype=np.float64)
        prefix_shape = tuple(int(size) for size in dataset.shape[:2])
        prefix_slice = (slice(None),) * max(dataset.ndim - 4, 0)

        for prefix in np.ndindex(prefix_shape):
            block = np.asarray(
                dataset[prefix + prefix_slice + (slice(None), slice(None))],
                dtype=np.float64,
            )
            fingerprint += block.sum(axis=tuple(range(block.ndim - 1)))

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

    @classmethod
    def _replace_dataset_with_attrs(cls, group, name, data, attrs=None):
        dataset = cls._replace_dataset(group, name, data)
        if attrs:
            cls._set_group_attrs(dataset, attrs)
        return dataset

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
    def _format_tau_ref_input_ns(tau_ref):
        try:
            value = float(tau_ref)
        except (TypeError, ValueError):
            return np.nan
        return value if np.isfinite(value) else np.nan

    @staticmethod
    def _package_version():
        if version is None:
            return "unknown"
        try:
            return version("brighteyes-mcs-file")
        except PackageNotFoundError:
            return "unknown"

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

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
        parameter_names=None,
        amplitude_param="C",
        delay_param="dT",
        lifetime_param="tau",
    ):
        errors = {
            "amplitude_err": np.nan,
            "tau_err_ns": np.nan,
            "fitted_delay_err_bins": np.nan,
            "fitted_delay_err_ns": np.nan,
        }
        covariance = np.asarray(covariance, dtype=float)
        if parameter_names is None:
            parameter_names = ["C", "dT", "tau"]
        parameter_names = list(parameter_names)
        if covariance.shape != (len(parameter_names), len(parameter_names)):
            return errors

        diag = np.diag(covariance)
        if amplitude_param in parameter_names:
            errors["amplitude_err"] = cls._std_from_variance(
                diag[parameter_names.index(amplitude_param)]
            )
        if delay_param in parameter_names:
            errors["fitted_delay_err_bins"] = cls._std_from_variance(
                diag[parameter_names.index(delay_param)]
            )
        if lifetime_param in parameter_names:
            errors["tau_err_ns"] = cls._std_from_variance(
                diag[parameter_names.index(lifetime_param)]
            )
        if np.isfinite(errors["fitted_delay_err_bins"]) and np.isfinite(dt_ns):
            errors["fitted_delay_err_ns"] = float(
                errors["fitted_delay_err_bins"] * float(dt_ns)
            )
        return errors

    @staticmethod
    def _residual_error(measured_trace, fitted_trace):
        try:
            data_norm = Alignment._normalize_histogram_1d(
                measured_trace,
                name="measured_trace",
            )
        except ValueError:
            return np.nan

        fitted_trace = np.asarray(fitted_trace, dtype=float)
        if fitted_trace.shape != data_norm.shape:
            return np.nan
        fit_sum = float(np.sum(fitted_trace))
        if not np.isfinite(fit_sum) or fit_sum <= 0:
            return np.nan

        fit_norm = fitted_trace / fit_sum
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

        aligned = np.zeros_like(stack, dtype=float)
        for channel_position, correction_delay in enumerate(correction_delay_in_bins):
            if not np.isfinite(correction_delay):
                continue
            hist = stack[:, channel_position]
            if not np.isfinite(hist).all() or np.sum(hist) <= 0:
                continue
            aligned[:, channel_position] = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(hist, correction_delay, cyclic=True),
                name=output_name,
            )
        return aligned

    @staticmethod
    def _empty_fit_payload(
        nbin,
        reference_type,
        measured_trace_histogram,
        reference_trace_histogram,
        irf_type,
        parameter_names=None,
    ):
        if parameter_names is None:
            parameter_names = ["C", "dT", "tau"]
        param_count = len(parameter_names)
        zero_hist = np.zeros(int(nbin), dtype=float)
        payload = {
            "amplitude": np.nan,
            "amplitude_err": np.nan,
            "tau_ns": np.nan,
            "tau_err_ns": np.nan,
            "tau_reference_ns": np.nan,
            "fitted_delay_bins": np.nan,
            "fitted_delay_ns": np.nan,
            "fitted_delay_err_bins": np.nan,
            "fitted_delay_err_ns": np.nan,
            "residual_error": np.nan,
            "measured_trace": np.asarray(measured_trace_histogram, dtype=float),
            "reference_trace": np.asarray(reference_trace_histogram, dtype=float),
            "irf_trace": zero_hist.copy(),
            "fitted_trace": zero_hist.copy(),
            "parameters": np.full(param_count, np.nan, dtype=float),
            "parameter_errors": np.full(param_count, np.nan, dtype=float),
            "covariance": np.full((param_count, param_count), np.nan, dtype=float),
            "irf_type": str(irf_type),
        }
        return payload

    def _prepare_output_file(self):
        if self.output_path.resolve() == self.data_path.resolve():
            raise ValueError("output_path must be different from data_path")
        if self.output_path.exists():
            if not self.overwrite:
                raise FileExistsError(f"output file already exists: {self.output_path}")
            self.output_path.unlink()
        return self.output_path

    @classmethod
    def _copy_attrs(cls, target, attrs):
        for key, value in attrs.items():
            target.attrs[str(key)] = cls._prepare_attr_value(value)

    @staticmethod
    def _safe_float(value, default=np.nan):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value):
            return default
        return value

    @staticmethod
    def _safe_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _metadata_float(cls, metadata, key, default=np.nan):
        return cls._safe_float(cls._metadata_get(metadata, key, default), default)

    @classmethod
    def _metadata_int(cls, metadata, key, default=0):
        return cls._safe_int(cls._metadata_get(metadata, key, default), default)

    @classmethod
    def _metadata_bool(cls, metadata, key, default=False):
        value = cls._metadata_get(metadata, key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def _metadata_first(cls, metadata, keys, default=None):
        for key in keys:
            value = cls._metadata_get(metadata, key, None)
            if value is not None:
                return value
        return default

    @classmethod
    def _metadata_float_first(cls, metadata, keys, default=np.nan):
        for key in keys:
            value = cls._safe_float(cls._metadata_get(metadata, key, np.nan), np.nan)
            if np.isfinite(value):
                return value
        return default

    @classmethod
    def _metadata_int_first(cls, metadata, keys, default=0):
        for key in keys:
            value = cls._metadata_get(metadata, key, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return default

    @classmethod
    def _metadata_bool_first(cls, metadata, keys, default=False):
        for key in keys:
            value = cls._metadata_get(metadata, key, None)
            if value is None:
                continue
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        return bool(default)

    @staticmethod
    def _pixel_spacing(range_um, count):
        range_um = H5DataCalibrator._safe_float(range_um)
        count = H5DataCalibrator._safe_int(count, 0)
        if not np.isfinite(range_um) or count <= 0:
            return np.nan
        if count == 1:
            return 0.0
        return float(range_um) / float(count - 1)

    @classmethod
    def _index_axis_values(cls, count, spacing, units):
        count = int(count)
        spacing = cls._safe_float(spacing)
        if count <= 0:
            return np.asarray([], dtype=np.float64), "index"
        if np.isfinite(spacing):
            return np.arange(count, dtype=np.float64) * spacing, units
        return np.arange(count, dtype=np.float64), "index"

    @classmethod
    def _spatial_axis_values(cls, count, range_um, offset_um):
        count = int(count)
        range_um = cls._safe_float(range_um)
        offset_um = cls._safe_float(offset_um, 0.0)
        if count <= 0:
            return np.asarray([], dtype=np.float64), "index"
        if count == 1:
            return np.asarray([offset_um], dtype=np.float64), "um"
        if np.isfinite(range_um):
            half_range = float(range_um) / 2.0
            return (
                np.linspace(
                    offset_um - half_range,
                    offset_um + half_range,
                    count,
                    dtype=np.float64,
                ),
                "um",
            )
        return np.arange(count, dtype=np.float64), "index"

    @classmethod
    def _calibrated_offset_um(cls, metadata, axis):
        offset_um = cls._metadata_float_first(
            metadata,
            (f"offset_{axis}_um", f"{axis}_offset_um"),
        )
        if np.isfinite(offset_um):
            return offset_um

        offset_v = cls._metadata_float_first(metadata, (f"offset_{axis}",))
        calibration_um_per_v = cls._metadata_float_first(metadata, (f"calib_{axis}",))
        if np.isfinite(offset_v) and np.isfinite(calibration_um_per_v):
            return float(offset_v * calibration_um_per_v)
        return np.nan

    @classmethod
    def _create_dataset_with_attrs(cls, group, name, data, attrs):
        if name in group:
            del group[name]
        dataset = group.create_dataset(name, data=np.asarray(data))
        cls._set_group_attrs(dataset, attrs)
        return dataset

    @staticmethod
    def _canonical_data_path_for_product(product_name):
        if product_name == "spad":
            return "/raw/spad"
        if product_name == "aux":
            return "/raw/aux"
        if product_name == "analog":
            return "/raw/analog"
        return f"/raw/{str(product_name).strip('/')}"

    @staticmethod
    def _analog_adc_calibration_attrs(channel_count):
        channel_count = max(0, int(channel_count))
        channel_nan_values = [float("nan")] * channel_count
        return {
            "adc_calibration_formula": (
                "voltage_v = adc_offset_v + adc_slope_v_per_adc_unit * adc_counts"
            ),
            "adc_offset_v": np.nan,
            "adc_slope_v_per_adc_unit": np.nan,
            "adc_channel_offset_v_json": json.dumps(channel_nan_values),
            "adc_channel_slope_v_per_adc_unit_json": json.dumps(channel_nan_values),
        }

    @classmethod
    def _copy_payload_dataset(
        cls,
        source_handle,
        data_group,
        source_key,
        target_name,
        attrs,
        *,
        required=False,
    ):
        try:
            source_dataset = cls._open_dataset(source_handle, source_key)
        except KeyError:
            if required:
                raise
            return None

        if target_name in data_group:
            del data_group[target_name]
        source_handle.copy(source_dataset.name, data_group, name=target_name)
        target_dataset = data_group[target_name]
        cls._set_group_attrs(target_dataset, attrs)
        if target_dataset.ndim == 8 and "axis_order" in target_dataset.attrs:
            channel_axis_name = str(target_dataset.attrs["axis_order"]).split(",")[-1]
            target_dataset.attrs["axis_order"] = (
                "repetition,z,y,x,circular_repetition,circular_point,"
                f"time_bin,{channel_axis_name}"
            )
            target_dataset.attrs["axis_4"] = "circular_repetition"
            target_dataset.attrs["axis_5"] = "circular_point"
            target_dataset.attrs["axis_6"] = "time_bin"
            target_dataset.attrs["axis_7"] = channel_axis_name
        target_dataset.attrs["actual_source_path"] = source_dataset.name
        target_dataset.attrs["shape_json"] = json.dumps(list(target_dataset.shape))
        target_dataset.attrs["dtype_preserved_from_source"] = True
        return target_dataset

    @classmethod
    def _copy_legacy_groups(cls, source_handle, data_group):
        legacy_group = data_group.require_group("legacy")
        cls._set_group_attrs(
            legacy_group,
            {
                "description": (
                    "Verbatim legacy BrightEyes-MCS root attributes and configuration groups."
                ),
                "source_file": source_handle.filename,
                "use_for_analysis": "Prefer /raw/metadata for normalized analysis metadata.",
            },
        )

        root_attrs_group = legacy_group.require_group("root_attrs")
        cls._copy_attrs(root_attrs_group, source_handle.attrs)

        copied_paths = []
        legacy_sources = {
            "configurationGUI": ("configurationGUI", "raw/legacy/configurationGUI"),
            "configurationGUI_beforeStart": (
                "configurationGUI_beforeStart",
                "raw/legacy/configurationGUI_beforeStart",
            ),
            "configurationFPGA": ("configurationFPGA", "raw/legacy/configurationFPGA"),
            "configurationSpadFCSmanager": (
                "configurationSpadFCSmanager",
                "raw/legacy/configurationSpadFCSmanager",
            ),
            "rawStreamAcquisition": (
                "rawStreamAcquisition",
                "raw/legacy/rawStreamAcquisition",
            ),
        }
        for target_name, candidates in legacy_sources.items():
            for candidate in candidates:
                if candidate not in source_handle:
                    continue
                if target_name in legacy_group:
                    del legacy_group[target_name]
                source_handle.copy(source_handle[candidate], legacy_group, name=target_name)
                copied_paths.append(f"/{candidate}")
                break

        legacy_group.attrs["source_group_paths_json"] = json.dumps(copied_paths)
        return legacy_group

    def _safe_time_axis(self, metadata, timebins):
        try:
            nbin, dt_ns, period_ns, t_ns = self.build_time_axis(
                metadata,
                nbin=timebins,
                period_ns=self.period_ns,
            )
            return nbin, dt_ns, period_ns, t_ns, "ns", []
        except Exception as exc:
            return (
                int(timebins),
                np.nan,
                np.nan,
                np.arange(int(timebins), dtype=np.float64),
                "index",
                [f"unable to infer digital time axis in ns: {exc}"],
            )

    def _write_data_metadata_and_axes(
        self,
        output_handle,
        data_group,
        data_metadata,
        primary_dataset,
        extra_dataset=None,
        analog_dataset=None,
    ):
        metadata_group = data_group.require_group("metadata")
        axes_group = data_group.require_group("axes")

        shape = tuple(int(size) for size in primary_dataset.shape)
        nrep, nz, ny, nx = shape[:4]
        digital_time_bins = int(shape[-2])
        primary_channel_count = int(shape[-1])
        extra_channel_count = int(extra_dataset.shape[-1]) if extra_dataset is not None else 0
        analog_channel_count = int(analog_dataset.shape[-1]) if analog_dataset is not None else 0
        analog_time_bins = int(analog_dataset.shape[-2]) if analog_dataset is not None else 0

        range_x_um = self._metadata_float_first(data_metadata, ("range_x_um", "rangex", "range_x"))
        range_y_um = self._metadata_float_first(data_metadata, ("range_y_um", "rangey", "range_y"))
        range_z_um = self._metadata_float_first(data_metadata, ("range_z_um", "rangez", "range_z"))
        offset_x_um = self._calibrated_offset_um(data_metadata, "x")
        offset_y_um = self._calibrated_offset_um(data_metadata, "y")
        offset_z_um = self._calibrated_offset_um(data_metadata, "z")
        pixel_size_x_um = self._metadata_float_first(
            data_metadata,
            ("pixel_size_x_um", "dx"),
            self._pixel_spacing(range_x_um, nx),
        )
        pixel_size_y_um = self._metadata_float_first(
            data_metadata,
            ("pixel_size_y_um", "dy"),
            self._pixel_spacing(range_y_um, ny),
        )
        pixel_size_z_um = self._metadata_float_first(
            data_metadata,
            ("pixel_size_z_um", "dz"),
            self._pixel_spacing(range_z_um, nz),
        )
        time_resolution_us = self._metadata_float_first(
            data_metadata,
            ("time_resolution_us", "time_resolution", "dt"),
        )
        base_time_bins = self._metadata_int_first(
            data_metadata,
            ("base_time_bins_per_pixel", "base_timebins_per_pixel", "timebin_per_pixel", "nbin"),
            digital_time_bins,
        )
        dfd_active = self._metadata_bool_first(
            data_metadata,
            ("dfd_active", "dfd_activate"),
            False,
        )
        acquisition_mode = "dfd" if dfd_active else "normal"
        timing_reference = "laser_period" if dfd_active else "pixel_dwell"
        pixel_dwell_time_us = self._metadata_float_first(
            data_metadata,
            ("pixel_dwell_time_us", "pixel_dwell_time_in_us", "pxdwelltime"),
            time_resolution_us * base_time_bins
            if np.isfinite(time_resolution_us) and base_time_bins > 0
            else np.nan,
        )
        _, digital_time_bin_ns, time_axis_period_ns, digital_time_axis, time_units, warnings_list = (
            self._safe_time_axis(data_metadata, digital_time_bins)
        )

        explicit_period_ns = self._safe_float(getattr(self, "period_ns", np.nan))
        if np.isfinite(explicit_period_ns) and explicit_period_ns > 0:
            laser_frequency_mhz = 1000.0 / explicit_period_ns
        elif dfd_active and np.isfinite(time_axis_period_ns) and time_axis_period_ns > 0:
            laser_frequency_mhz = 1000.0 / time_axis_period_ns
        else:
            laser_frequency_mhz = self._metadata_float_first(
                data_metadata,
                ("laser_frequency_mhz", "laser_freq_mhz", "dfd_freq"),
            )

        laser_period_ns = (
            1000.0 / laser_frequency_mhz
            if np.isfinite(laser_frequency_mhz) and laser_frequency_mhz > 0
            else np.nan
        )
        analog_time_bin_ns = time_resolution_us * 1000.0 if np.isfinite(time_resolution_us) else np.nan
        if dfd_active:
            dfd_time_bins = self._metadata_int_first(
                data_metadata,
                ("dfd_time_bins", "DFDnbins"),
                digital_time_bins,
            )
        else:
            dfd_time_bins = 0
        if dfd_active and dfd_time_bins > 0 and dfd_time_bins != digital_time_bins:
            warnings_list.append(
                "DFD time-bin metadata differs from /raw/spad shape; dataset shape is authoritative"
            )

        frame_time_s = self._metadata_float_first(data_metadata, ("frame_time_s", "frametime"))
        if not np.isfinite(frame_time_s) and np.isfinite(pixel_dwell_time_us):
            frame_time_s = pixel_dwell_time_us * nx * ny / 1e6
        volume_time_s = self._metadata_float_first(data_metadata, ("volume_time_s",))
        if not np.isfinite(volume_time_s) and np.isfinite(frame_time_s):
            volume_time_s = frame_time_s * nz
        acquisition_duration_s = self._metadata_float_first(
            data_metadata,
            ("acquisition_duration_s", "duration"),
        )
        if not np.isfinite(acquisition_duration_s) and np.isfinite(volume_time_s):
            acquisition_duration_s = volume_time_s * nrep
        time_bins_meaning = (
            "DFD histogram bins per pixel" if dfd_active else "digital FIFO time bins per pixel"
        )

        metadata_attrs = {
            "description": (
                "Normalized acquisition metadata derived from BrightEyes-MCS legacy configuration."
            ),
            "source_file": str(self.data_path),
            "source_data_format_version": output_handle.attrs.get(
                "source_data_format_version",
                "unknown",
            ),
            "source_comment": self._metadata_get(data_metadata, "comment", ""),
            "legacy_path": BRIGHTEYES_H5_LEGACY_PATH,
            "axes_path": BRIGHTEYES_H5_AXES_PATH,
            "data_spad_path": "/raw/spad",
            "data_aux_path": "/raw/aux" if extra_dataset is not None else "",
            "data_analog_path": "/raw/analog" if analog_dataset is not None else "",
            "acquisition_mode": acquisition_mode,
            "dfd_active": dfd_active,
            "timing_reference": timing_reference,
            "time_bins": digital_time_bins,
            "time_bins_meaning": time_bins_meaning,
            "nrep": nrep,
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "digital_time_bins": digital_time_bins,
            "analog_time_bins": analog_time_bins,
            "spad_channel_count": primary_channel_count,
            "aux_channel_count": extra_channel_count,
            "analog_channel_count": analog_channel_count,
            "range_x_um": range_x_um,
            "range_y_um": range_y_um,
            "range_z_um": range_z_um,
            "offset_x_um": offset_x_um,
            "offset_y_um": offset_y_um,
            "offset_z_um": offset_z_um,
            "pixel_size_x_um": pixel_size_x_um,
            "pixel_size_y_um": pixel_size_y_um,
            "pixel_size_z_um": pixel_size_z_um,
            "time_resolution_us": time_resolution_us,
            "pixel_dwell_time_us": pixel_dwell_time_us,
            "pixel_dwell_time_ns": (
                pixel_dwell_time_us * 1000.0 if np.isfinite(pixel_dwell_time_us) else np.nan
            ),
            "laser_frequency_mhz": laser_frequency_mhz,
            "laser_period_ns": laser_period_ns,
            "time_bin_ns": digital_time_bin_ns,
            "digital_time_bin_ns": digital_time_bin_ns,
            "analog_time_bin_ns": analog_time_bin_ns,
            "frame_time_s": frame_time_s,
            "volume_time_s": volume_time_s,
            "acquisition_duration_s": acquisition_duration_s,
            "snake_walk_xy": self._metadata_bool_first(data_metadata, ("snake_walk_xy", "snake"), False),
            "snake_walk_z": self._metadata_bool_first(data_metadata, ("snake_walk_z", "snake_z"), False),
            "subpixel_scan_mode": "circular" if primary_dataset.ndim == 8 else "none",
            "circular_point_count": int(shape[5]) if primary_dataset.ndim == 8 else 1,
            "circular_repetition_count": int(shape[4]) if primary_dataset.ndim == 8 else 1,
            "metadata_warnings_json": json.dumps(warnings_list),
        }
        self._set_group_attrs(metadata_group, metadata_attrs)

        acquisition_group = metadata_group.require_group("acquisition")
        scan_group = acquisition_group.require_group("scan")
        timing_group = acquisition_group.require_group("timing")
        fifo_group = acquisition_group.require_group("fifo")
        channels_group = acquisition_group.require_group("channels")
        scan_attrs = {
            key: metadata_attrs[key]
            for key in (
                "subpixel_scan_mode",
                "nx",
                "ny",
                "nz",
                "nrep",
                "range_x_um",
                "range_y_um",
                "range_z_um",
                "offset_x_um",
                "offset_y_um",
                "offset_z_um",
                "pixel_size_x_um",
                "pixel_size_y_um",
                "pixel_size_z_um",
                "snake_walk_xy",
                "snake_walk_z",
                "circular_point_count",
                "circular_repetition_count",
            )
        }
        scan_attrs.update(
            {"spatial_scan_mode": "raster", "circular_active": primary_dataset.ndim == 8}
        )
        self._set_group_attrs(scan_group, scan_attrs)
        self._set_group_attrs(
            timing_group,
            {
                "acquisition_mode": acquisition_mode,
                "timing_reference": timing_reference,
                "time_resolution_us": time_resolution_us,
                "base_time_bins_per_pixel": base_time_bins,
                "time_bins": digital_time_bins,
                "time_bins_meaning": time_bins_meaning,
                "digital_time_bins": digital_time_bins,
                "analog_time_bins": analog_time_bins,
                "digital_time_bin_ns": digital_time_bin_ns,
                "analog_time_bin_ns": analog_time_bin_ns,
                "pixel_dwell_time_us": pixel_dwell_time_us,
                "pixel_dwell_time_ns": metadata_attrs["pixel_dwell_time_ns"],
                "laser_frequency_mhz": laser_frequency_mhz,
                "laser_period_ns": laser_period_ns,
                "dfd_active": dfd_active,
                "dfd_time_bins": dfd_time_bins,
                "dfd_trigger_selector": self._metadata_int_first(data_metadata, ("dfd_trigger_selector",), -1),
                "dfd_laser_sync_debug": self._metadata_bool_first(
                    data_metadata,
                    ("dfd_laser_sync_debug",),
                    False,
                ),
                "metadata_warnings_json": json.dumps(warnings_list),
            },
        )
        self._set_group_attrs(
            fifo_group,
            {
                "digital_fifo_active": True,
                "analog_fifo_active": analog_dataset is not None,
                "spad_source_path": "/raw/spad",
                "aux_source_path": "/raw/aux" if extra_dataset is not None else "",
                "analog_source_path": "/raw/analog" if analog_dataset is not None else "",
                "spad_dtype": str(primary_dataset.dtype),
                "aux_dtype": str(extra_dataset.dtype) if extra_dataset is not None else "",
                "analog_dtype": str(analog_dataset.dtype) if analog_dataset is not None else "",
                "spad_shape_json": json.dumps(list(primary_dataset.shape)),
                "aux_shape_json": (
                    json.dumps(list(extra_dataset.shape)) if extra_dataset is not None else "[]"
                ),
                "analog_shape_json": (
                    json.dumps(list(analog_dataset.shape)) if analog_dataset is not None else "[]"
                ),
            },
        )
        self._set_group_attrs(
            channels_group,
            {
                "spad_channel_count": primary_channel_count,
                "aux_channel_count": extra_channel_count,
                "analog_channel_count": analog_channel_count,
                "detector_channel_axis_path": "/raw/axes/detector_channel_index",
                "aux_channel_axis_path": "/raw/axes/aux_channel_index",
                "analog_channel_axis_path": "/raw/axes/analog_channel_index",
                "detector_channel_labels_json": "[]",
                "aux_channel_labels_json": "[]",
                "analog_channel_labels_json": json.dumps(
                    [f"analog_{index}" for index in range(analog_channel_count)]
                ),
            },
        )

        self._set_group_attrs(
            axes_group,
            {
                "description": "Physical and logical axes for measured payload and derived products.",
            },
        )
        self._create_dataset_with_attrs(
            axes_group,
            "repetition_index",
            np.arange(nrep, dtype=np.int64),
            {"axis": "repetition", "units": "index", "long_name": "repetition index"},
        )
        for name, count, range_um, offset_um, pixel_size_um, axis in (
            ("z_um", nz, range_z_um, offset_z_um, pixel_size_z_um, "z"),
            ("y_um", ny, range_y_um, offset_y_um, pixel_size_y_um, "y"),
            ("x_um", nx, range_x_um, offset_x_um, pixel_size_x_um, "x"),
        ):
            values, units = self._spatial_axis_values(count, range_um, offset_um)
            self._create_dataset_with_attrs(
                axes_group,
                name,
                values,
                {
                    "axis": axis,
                    "units": units,
                    "long_name": f"{axis} position",
                    "range_um": range_um,
                    "offset_um": offset_um,
                    "pixel_size_um": pixel_size_um,
                    "coordinate_rule": "linspace(offset_um - range_um/2, offset_um + range_um/2, n)",
                },
            )
        self._create_dataset_with_attrs(
            axes_group,
            "digital_time_ns",
            digital_time_axis,
            {
                "axis": "time_bin",
                "units": time_units,
                "long_name": "digital FIFO time axis",
                "acquisition_mode": acquisition_mode,
                "timing_reference": timing_reference,
                "time_bins": digital_time_bins,
                "time_bins_meaning": time_bins_meaning,
                "bin_width_ns": digital_time_bin_ns,
            },
        )
        if analog_dataset is not None:
            analog_axis, analog_units = self._index_axis_values(
                analog_time_bins,
                analog_time_bin_ns,
                "ns",
            )
            self._create_dataset_with_attrs(
                axes_group,
                "analog_time_ns",
                analog_axis,
                {
                    "axis": "time_bin",
                    "units": analog_units,
                    "long_name": "analog FIFO time axis",
                    "time_bins": analog_time_bins,
                    "bin_width_ns": analog_time_bin_ns,
                },
            )
        self._create_dataset_with_attrs(
            axes_group,
            "detector_channel_index",
            np.arange(primary_channel_count, dtype=np.int64),
            {"axis": "detector_channel", "units": "index", "long_name": "detector channel index"},
        )
        if extra_dataset is not None:
            self._create_dataset_with_attrs(
                axes_group,
                "aux_channel_index",
                np.arange(extra_channel_count, dtype=np.int64),
                {
                    "axis": "aux_channel",
                    "units": "index",
                    "long_name": "auxiliary digital channel index",
                },
            )
        if analog_dataset is not None:
            self._create_dataset_with_attrs(
                axes_group,
                "analog_channel_index",
                np.arange(analog_channel_count, dtype=np.int64),
                {"axis": "analog_channel", "units": "index", "long_name": "analog channel index"},
            )

    def _initialize_schema_005(self, output_handle, data_handle, data_metadata):
        self._set_group_attrs(
            output_handle,
            {
                "data_format_version": BRIGHTEYES_H5_DATA_FORMAT_VERSION,
                "schema_name": BRIGHTEYES_H5_SCHEMA_NAME,
                "schema_variant": BRIGHTEYES_H5_SCHEMA_VARIANT,
                "file_kind": "calibrated_measurement",
                "default": "/raw/spad",
                "data_path": BRIGHTEYES_H5_DATA_PATH,
                "metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                "axes_path": BRIGHTEYES_H5_AXES_PATH,
                "legacy_path": BRIGHTEYES_H5_LEGACY_PATH,
                "calibration_path": f"/{self.calibration_key.strip('/')}",
                "output_path": BRIGHTEYES_H5_OUTPUT_PATH,
                "contains_measured_payload": True,
                "contains_legacy_configuration": True,
                "contains_calibration": True,
                "contains_output": self.create_output,
                "compatibility_root_configuration_groups": False,
                "source_data_format_version": data_handle.attrs.get("data_format_version", "unknown"),
                "comment": data_handle.attrs.get("comment", ""),
            },
        )

        data_group = output_handle.create_group("raw")
        self._set_group_attrs(
            data_group,
            {
                "description": (
                    "Measured BrightEyes-MCS payload with normalized metadata, axes, "
                    "and legacy configuration."
                ),
                "source_file": str(self.data_path),
                "source_data_format_version": data_handle.attrs.get("data_format_version", "unknown"),
                "source_root_default": data_handle.attrs.get("default", ""),
                "source_payload_paths_json": json.dumps(
                    [
                        path
                        for path in ("/data", "/data_channels_extra", "/data_analog", "/thumbnail")
                        if path.strip("/") in data_handle
                    ]
                ),
                "values_preserved": True,
                "shape_preserved": True,
                "dtype_preserved": True,
                "spad": "/raw/spad",
                "aux": "/raw/aux",
                "analog": "/raw/analog",
                "thumbnail": "/thumbnail",
                "metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                "axes_path": BRIGHTEYES_H5_AXES_PATH,
                "legacy_path": BRIGHTEYES_H5_LEGACY_PATH,
                "source_axis_order": "repetition,z,y,x,time_bin,channel",
                "note": (
                    "Dataset attrs are added for readability; numeric payload values are unchanged."
                ),
            },
        )

        primary_dataset = self._copy_payload_dataset(
            data_handle,
            data_group,
            "data",
            "spad",
            {
                "data_role": "spad_detector_counts",
                "long_name": "SPAD digital detector photon counts",
                "description": "Photon counts decoded from the BrightEyes-MCS digital FIFO.",
                "source_input_path": "/data",
                "source_fifo": "FIFO",
                "source_key": "data",
                "units": "counts",
                "axis_order": "repetition,z,y,x,time_bin,detector_channel",
                "axis_0": "repetition",
                "axis_1": "z",
                "axis_2": "y",
                "axis_3": "x",
                "axis_4": "time_bin",
                "axis_5": "detector_channel",
                "time_axis_index": -2,
                "channel_axis_index": -1,
                "time_axis_path": "/raw/axes/digital_time_ns",
                "channel_axis_path": "/raw/axes/detector_channel_index",
                "metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                "legacy_metadata_path": BRIGHTEYES_H5_LEGACY_PATH,
                "calibration_result_path": "/calibration/results/spad",
                "expected_shape_source": (
                    "configurationFPGA/configurationSpadFCSmanager plus dataset shape"
                ),
            },
            required=True,
        )
        extra_dataset = self._copy_payload_dataset(
            data_handle,
            data_group,
            "data_channels_extra",
            "aux",
            {
                "data_role": "auxiliary_digital_counts",
                "long_name": "auxiliary digital FIFO counts",
                "description": "Auxiliary digital channels decoded from the BrightEyes-MCS digital FIFO.",
                "source_input_path": "/data_channels_extra",
                "source_fifo": "FIFO",
                "source_key": "data_channels_extra",
                "units": "counts",
                "axis_order": "repetition,z,y,x,time_bin,aux_channel",
                "axis_0": "repetition",
                "axis_1": "z",
                "axis_2": "y",
                "axis_3": "x",
                "axis_4": "time_bin",
                "axis_5": "aux_channel",
                "time_axis_index": -2,
                "channel_axis_index": -1,
                "time_axis_path": "/raw/axes/digital_time_ns",
                "channel_axis_path": "/raw/axes/aux_channel_index",
                "metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                "legacy_metadata_path": BRIGHTEYES_H5_LEGACY_PATH,
                "calibration_result_path": "/calibration/results/aux",
            },
            required=False,
        )
        analog_dataset = self._copy_payload_dataset(
            data_handle,
            data_group,
            "data_analog",
            "analog",
            {
                "data_role": "analog_fifo",
                "long_name": "analog FIFO samples",
                "description": "Analog FIFO samples copied from the BrightEyes-MCS input file.",
                "source_input_path": "/data_analog",
                "source_fifo": "FIFOAnalog",
                "source_key": "data_analog",
                "units": "adc_counts",
                "axis_order": "repetition,z,y,x,time_bin,analog_channel",
                "time_axis_index": -2,
                "channel_axis_index": -1,
                "time_axis_path": "/raw/axes/analog_time_ns",
                "channel_axis_path": "/raw/axes/analog_channel_index",
                "metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                "legacy_metadata_path": BRIGHTEYES_H5_LEGACY_PATH,
                "calibration_result_path": "",
            },
            required=False,
        )
        if analog_dataset is not None:
            self._set_group_attrs(
                analog_dataset,
                self._analog_adc_calibration_attrs(analog_dataset.shape[-1]),
            )
        self._copy_payload_dataset(
            data_handle,
            output_handle,
            "thumbnail",
            "thumbnail",
            {
                "data_role": "preview_thumbnail",
                "long_name": "preview thumbnail",
                "description": "Preview JPEG bytes copied from the BrightEyes-MCS input file.",
                "source_input_path": "/thumbnail",
                "units": "encoded_bytes",
            },
            required=False,
        )
        self._copy_legacy_groups(data_handle, data_group)
        self._write_data_metadata_and_axes(
            output_handle,
            data_group,
            data_metadata,
            primary_dataset,
            extra_dataset=extra_dataset,
            analog_dataset=analog_dataset,
        )

        output_group = output_handle.create_group("output")
        self._set_group_attrs(
            output_group,
            {
                "description": "Derived analysis outputs. Measured payload is stored under /raw.",
                "source_data_path": "/raw/spad",
                "source_aux_data_path": "/raw/aux" if extra_dataset is not None else "",
                "metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                "axes_path": BRIGHTEYES_H5_AXES_PATH,
                "legacy_path": BRIGHTEYES_H5_LEGACY_PATH,
                "default": "",
                "run_count": 0,
            },
        )

    @staticmethod
    def _calibration_product_name(data_key):
        normalized = str(data_key).strip("/")
        product_map = {
            "data": "spad",
            "spad": "spad",
            "raw/spad": "spad",
            "data_channels_extra": "aux",
            "aux": "aux",
            "raw/aux": "aux",
        }
        if normalized in product_map:
            return product_map[normalized]
        return normalized.replace("/", "_")

    @staticmethod
    def _source_axes_for_dataset(dataset):
        if dataset.ndim == 6:
            return "repetition,z,y,x,time_bin,channel"
        if dataset.ndim == 8:
            return "repetition,z,y,x,circular_repetition,circular_point,time_bin,channel"
        return ",".join(f"axis_{index}" for index in range(dataset.ndim))

    @staticmethod
    def _source_reduction_axes_for_dataset(dataset):
        if dataset.ndim == 6:
            return "repetition,z,y,x"
        if dataset.ndim == 8:
            return "repetition,z,y,x,circular_repetition,circular_point"
        return ",".join(f"axis_{index}" for index in range(max(dataset.ndim - 2, 0)))

    @staticmethod
    def _subpixel_scan_mode_for_dataset(dataset):
        return "circular" if dataset.ndim == 8 else "none"

    @staticmethod
    def _source_fifo_for_product(product_name):
        if product_name == "spad":
            return "FIFO"
        if product_name == "aux":
            return "FIFO_extra"
        return ""

    def _get_target_group(self, calibration_group, product_name):
        results_group = calibration_group.require_group("results")
        if product_name in results_group:
            del results_group[product_name]
        return results_group.create_group(product_name)

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

        if self._calibration_product_name(data_key) != "spad":
            primary_cache_key = "spad"
            if primary_cache_key in channel_skew_cache:
                reference_data_key = primary_cache_key
                reference_channel_index = np.asarray(
                    channel_skew_cache[primary_cache_key]["channel_index"],
                    dtype=int,
                )
                reference_sources = channel_skew_cache[primary_cache_key]["sources"]

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
             h5py.File(output_path, "w") as output_handle:

            self._initialize_schema_005(output_handle, data_handle, data_metadata)

            if self.calibration_key in output_handle:
                del output_handle[self.calibration_key]
            calibration_group = output_handle.create_group(self.calibration_key)

            self._set_group_attrs(
                calibration_group,
                {
                    "calibrated_products_json": "[]",
                    "reference_type": self.reference_type,
                    "reference_product_map_json": "{}",
                    "fit_mode": self.fit_mode,
                    "fit_type": self.fit_type,
                    "residual_error_metric": "rmse_normalized_histograms",
                    "C_ref": self.C_ref,
                    "tau_ref_input_ns": self._format_tau_ref_input_ns(self.tau_ref),
                    "irf_iterations": self.irf_iterations,
                    "eps": self.eps,
                    "regularization": self.regularization,
                    "clean_irf": self.clean_irf,
                    "irf_corrections_type": self.irf_corrections_type,
                    "channel_skew_type": self.channel_skew_type,
                    "channel_skew_source": self._channel_skew_source_attr_value(
                        self.channel_skew_source
                    ),
                    "source_reduction_axes": "",
                    "metadata_path": f"/{self.calibration_key.strip('/')}/metadata",
                    "source_metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                    "axes_path": f"/{self.calibration_key.strip('/')}/axes",
                    "source_axes_path": BRIGHTEYES_H5_AXES_PATH,
                    "data_path": BRIGHTEYES_H5_DATA_PATH,
                    "legacy_path": BRIGHTEYES_H5_LEGACY_PATH,
                    "results_path": f"/{self.calibration_key.strip('/')}/results",
                },
            )
            calibration_metadata_group = calibration_group.create_group("metadata")
            self._set_group_attrs(
                calibration_metadata_group,
                {
                    "description": "Normalized metadata for this calibration run.",
                    "source_data_file": str(self.data_path),
                    "source_reference_file": str(self.reference_path),
                    "source_metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                    "source_axes_path": BRIGHTEYES_H5_AXES_PATH,
                    "results_path": f"/{self.calibration_key.strip('/')}/results",
                    "reference_type": self.reference_type,
                    "fit_mode": self.fit_mode,
                    "fit_type": self.fit_type,
                    "calibrated_products_json": "[]",
                    "reference_product_map_json": "{}",
                },
            )
            calibration_axes_group = calibration_group.create_group("axes")
            self._set_group_attrs(
                calibration_axes_group,
                {
                    "description": "Axes used by calibration histograms and per-channel results.",
                    "source_axes_path": BRIGHTEYES_H5_AXES_PATH,
                },
            )
            provenance_group = calibration_group.create_group("provenance")
            data_file_hash_attrs = channel_fingerprint_file_hash_attrs(data_handle, prefix="source_data_file")
            reference_file_hash_attrs = channel_fingerprint_file_hash_attrs(
                reference_handle,
                prefix="source_reference_file",
            )
            self._set_group_attrs(
                provenance_group,
                {
                    "created_by_package": "brighteyes_mcs_file",
                    "created_by_version": self._package_version(),
                    "created_at_iso8601": self._utc_now(),
                    "source_data_file": str(self.data_path),
                    "source_reference_file": str(self.reference_path),
                    **data_file_hash_attrs,
                    **reference_file_hash_attrs,
                    "output_file": str(output_path),
                    "source_data_format_version": data_handle.attrs.get("data_format_version", "unknown"),
                    "source_reference_format_version": reference_handle.attrs.get("data_format_version", "unknown"),
                    "payload_group": BRIGHTEYES_H5_DATA_PATH,
                    "metadata_path": f"/{self.calibration_key.strip('/')}/metadata",
                    "source_metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                    "source_axes_path": BRIGHTEYES_H5_AXES_PATH,
                    "legacy_configuration_path": BRIGHTEYES_H5_LEGACY_PATH,
                    "raw_data_values_preserved": True,
                    "data_keys_json": json.dumps(self.data_keys),
                    "reference_keys_json": json.dumps(self.reference_key_map),
                    "tau_ref_input": self._format_tau_ref_input(self.tau_ref),
                    "initial_tau": self.initial_tau,
                    "initial_dT": self.initial_dT,
                    "initial_C": self.initial_C,
                    "force_C_normalized": self.force_C_normalized,
                    "fit_model_name": Alignment._callable_name(self.model_fn),
                    "fit_parameter_names_json": json.dumps(self.parameter_names),
                    "fit_p0": self.p0,
                    "fit_bounds": self.bounds,
                    "fit_model_kwargs_json": json.dumps(self.model_kwargs, default=str),
                    "fit_amplitude_param": self.amplitude_param,
                    "fit_delay_param": self.delay_param,
                    "fit_lifetime_param": self.lifetime_param,
                    "channel_skew_fit_reference_channel": self.channel_skew_fit_reference_channel,
                    "channel_skew_fit_upsampling": self.channel_skew_fit_upsampling,
                    "channel_skew_fit_apodize": self.channel_skew_fit_apodize,
                },
            )
            inputs_group = calibration_group.create_group("inputs")
            calibration_group.create_group("results")
            self._write_metadata_group(provenance_group, "input_data_metadata", data_metadata)
            self._write_metadata_group(provenance_group, "input_reference_metadata", reference_metadata)

            ordered_data_keys = list(self.data_keys)
            primary_positions = [
                index
                for index, key in enumerate(ordered_data_keys)
                if self._calibration_product_name(key) == "spad"
            ]
            if primary_positions:
                ordered_data_keys.insert(0, ordered_data_keys.pop(primary_positions[0]))
            channel_skew_cache = {}
            calibrated_products = []
            reference_product_map = {}
            source_reduction_axes_seen = []

            data_key_iterator = tqdm(
                ordered_data_keys,
                desc="Calibrating data keys",
                unit="key",
                disable=len(ordered_data_keys) <= 1,
            )
            for data_key in data_key_iterator:
                reference_key = self.reference_key_map[data_key]
                product_name = self._calibration_product_name(data_key)
                reference_product = self._calibration_product_name(reference_key)
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
                laser_frequency_mhz = 1e3 / period_ns
                if "time_ns" not in calibration_axes_group:
                    self._replace_dataset_with_attrs(
                        calibration_axes_group,
                        "time_ns",
                        t_ns,
                        {
                            "axis": "time_bin",
                            "units": "ns",
                            "long_name": "calibration histogram time axis",
                            "time_bins": nbin,
                            "time_bin_ns": dt_ns,
                            "source_time_axis_path": "/raw/axes/digital_time_ns",
                        },
                    )
                self._replace_dataset_with_attrs(
                    calibration_axes_group,
                    f"{product_name}_channel_index",
                    np.asarray(channel_indices, dtype=np.int64),
                    {
                        "axis": "channel",
                        "units": "index",
                        "long_name": f"{product_name} calibrated channel index",
                    },
                )

                source_axes = self._source_axes_for_dataset(data_dataset)
                source_reduction_axes = self._source_reduction_axes_for_dataset(data_dataset)
                subpixel_scan_mode = self._subpixel_scan_mode_for_dataset(data_dataset)
                acquisition_mode = (
                    "dfd" if bool(self._metadata_get(data_metadata, "dfd_activate", False)) else "normal"
                )
                source_reduction_axes_seen.append(source_reduction_axes)
                if product_name not in calibrated_products:
                    calibrated_products.append(product_name)
                reference_product_map[product_name] = reference_product
                canonical_source_data_path = self._canonical_data_path_for_product(product_name)

                target_group = self._get_target_group(calibration_group, product_name)
                self._set_group_attrs(
                    target_group,
                    {
                        "product_name": product_name,
                        "source_data_path": canonical_source_data_path,
                        "source_input_path": data_dataset.name,
                        "source_fifo": self._source_fifo_for_product(product_name),
                        "source_data_key": data_key,
                        "source_axes": source_axes,
                        "source_reduction_axes": source_reduction_axes,
                        "histogram_axes": "time_bin,channel",
                        "time_axis_path": f"/{self.calibration_key.strip('/')}/axes/time_ns",
                        "source_time_axis_path": "/raw/axes/digital_time_ns",
                        "metadata_path": f"/{self.calibration_key.strip('/')}/metadata",
                        "source_metadata_path": BRIGHTEYES_H5_METADATA_PATH,
                        "axes_path": f"/{self.calibration_key.strip('/')}/axes",
                        "source_axes_path": BRIGHTEYES_H5_AXES_PATH,
                        "legacy_path": BRIGHTEYES_H5_LEGACY_PATH,
                        "data_key": data_key,
                        "reference_key": reference_key,
                        "reference_product": reference_product,
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
                        "fit_parameter_names": json.dumps(self.parameter_names),
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
                        "time_bins": nbin,
                        "time_bin_ns": dt_ns,
                        "laser_period_ns": period_ns,
                        "laser_frequency_mhz": laser_frequency_mhz,
                        "channel_count": int(data_dataset.shape[-1]),
                        "channel_count_calibrated": len(channel_indices),
                        "data_shape": list(data_dataset.shape),
                        "reference_shape": list(reference_dataset.shape),
                        "channel_axis": -1,
                        "stacked_histogram_layout": "(t, ch)",
                        "residual_error_metric": "rmse_normalized_histograms",
                        "acquisition_mode": acquisition_mode,
                        "subpixel_scan_mode": subpixel_scan_mode,
                    },
                )
                reference_fingerprint = self._sum_dataset_over_non_channel_axes(reference_dataset)

                stacked_channel_index = []
                stacked_reference_mask = []
                stacked_amplitude = []
                stacked_amplitude_err = []
                stacked_tau_ns = []
                stacked_tau_err_ns = []
                stacked_tau_reference_ns = []
                stacked_fitted_delay_bins = []
                stacked_fitted_delay_ns = []
                stacked_fitted_delay_err_bins = []
                stacked_fitted_delay_err_ns = []
                stacked_residual_error = []
                stacked_irf_type = []
                stacked_measured_trace = []
                stacked_irf_trace = []
                stacked_fitted_trace = []
                stacked_reference_trace = []
                stacked_parameters = []
                stacked_parameter_errors = []
                stacked_covariance = []

                channel_iterator = tqdm(
                    channel_indices,
                    desc=f"Calibrating {data_key}",
                    unit="ch",
                    leave=False,
                )
                for channel_index in channel_iterator:
                    reference_channel_for_fit = reference_channel_map[channel_index]
                    measured_trace_histogram = self._sum_histogram_for_channel(data_dataset, channel_index)
                    reference_trace_histogram = self._sum_histogram_for_channel(
                        reference_dataset,
                        reference_channel_for_fit,
                    )
                    data_sum = float(np.sum(measured_trace_histogram))
                    reference_sum = float(np.sum(reference_trace_histogram))

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
                            measured_trace_histogram=measured_trace_histogram,
                            reference_trace_histogram=reference_trace_histogram,
                            irf_type="skipped_zero_sum_data",
                            parameter_names=self.parameter_names,
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
                            measured_trace_histogram=measured_trace_histogram,
                            reference_trace_histogram=reference_trace_histogram,
                            irf_type="skipped_zero_sum_reference",
                            parameter_names=self.parameter_names,
                        )
                    else:
                        fit_kwargs = {
                            "t": t_ns,
                            "data": measured_trace_histogram,
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
                            "parameter_names": self.parameter_names,
                            "model_kwargs": self.model_kwargs,
                            "amplitude_param": self.amplitude_param,
                            "delay_param": self.delay_param,
                            "lifetime_param": self.lifetime_param,
                            "irf_iterations": self.irf_iterations,
                            "eps": self.eps,
                            "regularization": self.regularization,
                        }
                        if self.reference_type == "ref":
                            fit_kwargs["ref"] = reference_trace_histogram
                            fit_kwargs["tau_ref"] = self.tau_ref
                        else:
                            fit_kwargs["irf"] = reference_trace_histogram

                        try:
                            fit_result = Alignment.fit_data_with_ref_or_irf(**fit_kwargs)
                            irf_trace = np.asarray(fit_result["irf"], dtype=float)
                            fitted_trace = np.asarray(fit_result["fit"], dtype=float)
                            parameter_errors = self._parameter_error_payload(
                                fit_result["cov"],
                                dt_ns,
                                parameter_names=fit_result["parameter_names"],
                                amplitude_param=self.amplitude_param,
                                delay_param=self.delay_param,
                                lifetime_param=self.lifetime_param,
                            )
                            fit_payload = {
                                "amplitude": float(fit_result["C"]),
                                "amplitude_err": float(parameter_errors["amplitude_err"]),
                                "tau_ns": float(fit_result["tau"]),
                                "tau_err_ns": float(parameter_errors["tau_err_ns"]),
                                "tau_reference_ns": (
                                    np.nan
                                    if fit_result["tau_ref"] is None
                                    else float(fit_result["tau_ref"])
                                ),
                                "fitted_delay_bins": float(fit_result["dT"]),
                                "fitted_delay_ns": float(fit_result["dT_ns"]),
                                "fitted_delay_err_bins": float(
                                    parameter_errors["fitted_delay_err_bins"]
                                ),
                                "fitted_delay_err_ns": float(
                                    parameter_errors["fitted_delay_err_ns"]
                                ),
                                "residual_error": float(
                                    self._residual_error(measured_trace_histogram, fitted_trace)
                                ),
                                "measured_trace": np.asarray(measured_trace_histogram, dtype=float),
                                "reference_trace": np.asarray(reference_trace_histogram, dtype=float),
                                "irf_trace": np.asarray(irf_trace, dtype=float),
                                "fitted_trace": fitted_trace,
                                "parameters": np.asarray(
                                    fit_result["param_values"],
                                    dtype=float,
                                ),
                                "parameter_errors": np.asarray(
                                    fit_result["param_errors"],
                                    dtype=float,
                                ),
                                "covariance": np.asarray(
                                    fit_result["cov"],
                                    dtype=float,
                                ),
                                "irf_type": str(fit_result["irf_source"]),
                            }
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
                                measured_trace_histogram=measured_trace_histogram,
                                reference_trace_histogram=reference_trace_histogram,
                                irf_type="fit_failed",
                                parameter_names=self.parameter_names,
                            )

                    stacked_channel_index.append(channel_index)
                    stacked_reference_mask.append(reference_channel_for_fit)
                    stacked_amplitude.append(float(fit_payload["amplitude"]))
                    stacked_amplitude_err.append(float(fit_payload["amplitude_err"]))
                    stacked_tau_ns.append(float(fit_payload["tau_ns"]))
                    stacked_tau_err_ns.append(float(fit_payload["tau_err_ns"]))
                    stacked_tau_reference_ns.append(float(fit_payload["tau_reference_ns"]))
                    stacked_fitted_delay_bins.append(
                        float(fit_payload["fitted_delay_bins"])
                    )
                    stacked_fitted_delay_ns.append(
                        float(fit_payload["fitted_delay_ns"])
                    )
                    stacked_fitted_delay_err_bins.append(
                        float(fit_payload["fitted_delay_err_bins"])
                    )
                    stacked_fitted_delay_err_ns.append(
                        float(fit_payload["fitted_delay_err_ns"])
                    )
                    stacked_residual_error.append(float(fit_payload["residual_error"]))
                    stacked_irf_type.append(str(fit_payload["irf_type"]))
                    stacked_measured_trace.append(np.asarray(fit_payload["measured_trace"], dtype=float))
                    stacked_reference_trace.append(
                        np.asarray(fit_payload["reference_trace"], dtype=float)
                    )
                    stacked_irf_trace.append(np.asarray(fit_payload["irf_trace"], dtype=float))
                    stacked_fitted_trace.append(np.asarray(fit_payload["fitted_trace"], dtype=float))
                    stacked_parameters.append(np.asarray(fit_payload["parameters"], dtype=float))
                    stacked_parameter_errors.append(
                        np.asarray(fit_payload["parameter_errors"], dtype=float)
                    )
                    stacked_covariance.append(
                        np.asarray(fit_payload["covariance"], dtype=float)
                    )
                channel_index_array = np.asarray(stacked_channel_index, dtype=int)
                reference_mask_array = np.asarray(
                    stacked_reference_mask,
                    dtype=int,
                )
                reference_fingerprint_for_output_channels = reference_fingerprint[
                    reference_mask_array
                ]
                amplitude_array = np.asarray(stacked_amplitude, dtype=float)
                amplitude_err_array = np.asarray(stacked_amplitude_err, dtype=float)
                tau_ns_array = np.asarray(stacked_tau_ns, dtype=float)
                tau_err_ns_array = np.asarray(stacked_tau_err_ns, dtype=float)
                tau_reference_ns_array = np.asarray(stacked_tau_reference_ns, dtype=float)
                fitted_delay_bins_array = np.asarray(
                    stacked_fitted_delay_bins,
                    dtype=float,
                )
                fitted_delay_ns_array = np.asarray(
                    stacked_fitted_delay_ns,
                    dtype=float,
                )
                delay_correction_bins_array = self._compute_irf_correction_delays(
                    fitted_delay_bins_array,
                    self.irf_corrections_type,
                    data_key,
                )
                delay_correction_ns_array = delay_correction_bins_array * float(dt_ns)
                fitted_delay_err_bins_array = np.asarray(
                    stacked_fitted_delay_err_bins,
                    dtype=float,
                )
                fitted_delay_err_ns_array = np.asarray(
                    stacked_fitted_delay_err_ns,
                    dtype=float,
                )
                residual_error_array = np.asarray(stacked_residual_error, dtype=float)
                measured_trace_stack = np.stack(stacked_measured_trace, axis=-1)
                irf_trace_stack = np.stack(stacked_irf_trace, axis=-1)
                aligned_irf_trace_stack = self._realign_histogram_stack(
                    irf_trace_stack,
                    delay_correction_bins_array,
                    "aligned/irf_trace",
                )
                if self.clean_irf and self.reference_type == "irf":
                    aligned_irf_trace_stack = Alignment.clean_irf_stack(
                        aligned_irf_trace_stack,
                        threshold=0.3,
                        window=2.0 / dt_ns,
                        time_axis=0,
                        normalize=True,
                    )
                aligned_irf_trace_stack = self._normalize_stack_to_fingerprint(
                    aligned_irf_trace_stack,
                    reference_fingerprint_for_output_channels,
                )
                fitted_trace_stack = np.stack(stacked_fitted_trace, axis=-1)
                parameters_array = np.stack(stacked_parameters, axis=0)
                parameter_errors_array = np.stack(stacked_parameter_errors, axis=0)
                covariance_array = np.stack(stacked_covariance, axis=0)
                reference_trace_stack = np.stack(stacked_reference_trace, axis=-1)

                if product_name in inputs_group:
                    del inputs_group[product_name]
                input_product_group = inputs_group.create_group(product_name)
                self._replace_dataset_with_attrs(
                    input_product_group,
                    "data_histogram",
                    measured_trace_stack,
                    {
                        "source_data_path": canonical_source_data_path,
                        "source_input_path": data_dataset.name,
                        "reduction_axes": source_reduction_axes,
                        "axes": "time_bin,channel",
                    },
                )
                self._replace_dataset_with_attrs(
                    input_product_group,
                    "reference_histogram",
                    reference_trace_stack,
                    {
                        "source_reference_path": reference_dataset.name,
                        "axes": "time_bin,channel",
                    },
                )
                self._replace_dataset(input_product_group, "reference_fingerprint", reference_fingerprint)

                channels_group = target_group.create_group("channels")
                timing_group = target_group.create_group("timing")
                fit_group = target_group.create_group("fit")
                aligned_group = target_group.create_group("aligned")

                self._replace_dataset(channels_group, "index", channel_index_array)
                self._replace_dataset(
                    channels_group,
                    "reference_mask",
                    reference_mask_array,
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "measured_trace",
                    measured_trace_stack,
                    {
                        "axes": "time_bin,channel",
                        "reduction_axes": source_reduction_axes,
                    },
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "reference_trace",
                    reference_trace_stack,
                    {"axes": "time_bin,channel"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "irf_trace",
                    irf_trace_stack,
                    {"axes": "time_bin,channel"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "fitted_trace",
                    fitted_trace_stack,
                    {"axes": "time_bin,channel"},
                )
                self._replace_dataset(fit_group, "amplitude", amplitude_array)
                self._replace_dataset(fit_group, "amplitude_err", amplitude_err_array)
                self._replace_dataset_with_attrs(fit_group, "tau_ns", tau_ns_array, {"units": "ns"})
                self._replace_dataset_with_attrs(
                    fit_group,
                    "tau_err_ns",
                    tau_err_ns_array,
                    {"units": "ns"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "tau_reference_ns",
                    tau_reference_ns_array,
                    {"units": "ns"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "fitted_delay_bins",
                    fitted_delay_bins_array,
                    {"units": "bins"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "fitted_delay_ns",
                    fitted_delay_ns_array,
                    {"units": "ns"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "fitted_delay_err_bins",
                    fitted_delay_err_bins_array,
                    {"units": "bins"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "fitted_delay_err_ns",
                    fitted_delay_err_ns_array,
                    {"units": "ns"},
                )
                self._replace_dataset_with_attrs(
                    timing_group,
                    "delay_correction_bins",
                    delay_correction_bins_array,
                    {"units": "bins"},
                )
                self._replace_dataset_with_attrs(
                    timing_group,
                    "delay_correction_ns",
                    delay_correction_ns_array,
                    {"units": "ns"},
                )
                self._replace_dataset_with_attrs(
                    timing_group,
                    "delay_correction_err_bins",
                    fitted_delay_err_bins_array,
                    {"units": "bins"},
                )
                self._replace_dataset_with_attrs(
                    timing_group,
                    "delay_correction_err_ns",
                    fitted_delay_err_ns_array,
                    {"units": "ns"},
                )
                self._replace_dataset_with_attrs(
                    fit_group,
                    "residual_error",
                    residual_error_array,
                    {"metric": "rmse_normalized_histograms"},
                )
                string_dtype = h5py.string_dtype(encoding="utf-8")
                fit_group.create_dataset(
                    "parameter_names",
                    data=np.asarray(self.parameter_names, dtype=object),
                    dtype=string_dtype,
                )
                self._replace_dataset(fit_group, "parameters", parameters_array)
                self._replace_dataset(fit_group, "parameter_errors", parameter_errors_array)
                self._replace_dataset(fit_group, "covariance", covariance_array)
                self._replace_dataset(
                    aligned_group,
                    "irf_trace",
                    aligned_irf_trace_stack,
                )

                channel_skew_sources = {
                    "data": measured_trace_stack,
                    "irf": aligned_irf_trace_stack,
                }
                if self.reference_type == "ref":
                    aligned_reference_trace_stack = self._realign_histogram_stack(
                        reference_trace_stack,
                        delay_correction_bins_array,
                        "aligned/reference_trace",
                    )
                    aligned_reference_trace_stack = self._normalize_stack_to_fingerprint(
                        aligned_reference_trace_stack,
                        reference_fingerprint_for_output_channels,
                    )
                    self._replace_dataset(
                        aligned_group,
                        "reference_trace",
                        aligned_reference_trace_stack,
                    )
                    channel_skew_sources["ref"] = aligned_reference_trace_stack

                (
                    channel_skew,
                    channel_skew_err,
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
                channel_skew_err = np.asarray(channel_skew_err, dtype=float)
                self._replace_dataset_with_attrs(
                    timing_group,
                    "channel_skew_bins",
                    channel_skew,
                    {"units": "bins"},
                )
                self._replace_dataset_with_attrs(
                    timing_group,
                    "channel_skew_err_bins",
                    channel_skew_err,
                    {"units": "bins"},
                )
                ext_data = channel_skew_reference_info.get("ext_data")
                if ext_data is not None:
                    self._replace_dataset(
                        timing_group,
                        "channel_skew_external",
                        np.asarray(ext_data, dtype=float),
                    )
                fit_group.create_dataset(
                    "irf_type",
                    data=np.asarray(stacked_irf_type, dtype=object),
                    dtype=string_dtype,
                )

                target_group.attrs["calibration_group_layout"] = "results_by_product"
                channel_skew_cache_entry = {
                    "channel_index": channel_index_array.copy(),
                    "sources": {
                        key: np.asarray(value, dtype=float).copy()
                        for key, value in channel_skew_sources.items()
                    },
                }
                channel_skew_cache[data_key] = channel_skew_cache_entry
                channel_skew_cache[product_name] = channel_skew_cache_entry

            calibration_group.attrs["calibrated_products_json"] = json.dumps(calibrated_products)
            calibration_group.attrs["reference_product_map_json"] = json.dumps(reference_product_map)
            calibration_metadata_group.attrs["calibrated_products_json"] = json.dumps(calibrated_products)
            calibration_metadata_group.attrs["reference_product_map_json"] = json.dumps(reference_product_map)
            unique_reduction_axes = list(dict.fromkeys(source_reduction_axes_seen))
            calibration_group.attrs["source_reduction_axes"] = (
                unique_reduction_axes[0] if len(unique_reduction_axes) == 1 else json.dumps(unique_reduction_axes)
            )
            calibration_metadata_group.attrs["source_reduction_axes"] = calibration_group.attrs[
                "source_reduction_axes"
            ]

        if self.create_output:
            output_options = dict(self.output_options)
            output_options.pop("output_path", None)
            output_options["in_place"] = True
            output_options.setdefault("overwrite", self.overwrite)
            H5OutputBuilder(output_path, **output_options).build()

        return str(output_path)


DEFAULT_OUTPUT_KEY = "output"
DEFAULT_SUM_CHANNELS_RUN_ID = "sum_channels_without_corrections_001"
DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID = "sum_channels_001"
SUM_IRF_TRACE_OUTPUT_KIND = "sum_irf_trace"
SUM_REFERENCE_TRACE_OUTPUT_KIND = "sum_reference_trace"
DEFAULT_SUM_IRF_TRACE_ID = "sum_irf_001"
DEFAULT_SUM_REFERENCE_TRACE_ID = "sum_ref_001"
SUM_CHANNELS_AGGREGATION = "sum_channels_without_corrections"
SUM_CHANNELS_WITH_SKEW_CORRECTION_AGGREGATION = "sum_channels_with_skew_correction"
SUM_CHANNELS_TOOL_NAME = "Sum channels"
SUM_CHANNELS_WITH_SKEW_CORRECTION_TOOL_NAME = "Sum channels with skew correction"
DEFAULT_SPAD_DATA_KEYS = ("raw/spad",)
DEFAULT_AUX_DATA_KEYS = ("raw/aux",)
DEFAULT_ANALOG_DATA_KEYS = ("raw/analog",)
DEFAULT_PRIMARY_DATA_KEYS = DEFAULT_SPAD_DATA_KEYS
DEFAULT_EXTRA_DATA_KEYS = DEFAULT_AUX_DATA_KEYS
DEFAULT_MAX_BLOCK_BYTES = 128 * 1024 * 1024

AXIS_ORDERS = {
    6: {
        "source": "repetition,z,y,x,time_bin,channel",
        "virtual": "repetition,z,y,x,time_bin",
        "layout": "raster_6d",
        "subpixel_scan_mode": "none",
    },
    8: {
        "source": (
            "repetition,z,y,x,circular_repetition,circular_point,"
            "time_bin,channel"
        ),
        "virtual": (
            "repetition,z,y,x,circular_repetition,circular_point,time_bin"
        ),
        "layout": "circular_8d",
        "subpixel_scan_mode": "circular",
    },
}


class H5OutputBuilder:
    """
    Add the BrightEyes ``/output`` group to an HDF5 file.

    The builder creates two kinds of output:

    - ``/output/virtual_channels/<kind>/channel_<i>``: HDF5 virtual datasets
      grouped by ``spad``, ``aux``, and ``analog`` source kind that point to a
      single source channel without copying raw data.
    - ``/output/<sum_channels_run_id>/products/spad``: the
      "sum_channels_without_corrections" output, computed by summing the SPAD
      data along the final channel axis. When auxiliary digital data are present
      and ``include_aux_sum=True``, the auxiliary channels are also summed into
      ``products/aux``.
    - ``/output/<sum_channels_with_skew_correction_run_id>/products/spad``: the
      "sum_channels_with_skew_correction" output, stored by default as
      ``/output/sum_channels_001/products/spad`` and computed with
      :meth:`Alignment.sum_channel_applying_shifts` and the stored calibration
      ``channel_skew`` vector.

    Source datasets are read from the schema 0.0.6 paths ``/raw/spad`` and
    ``/raw/aux`` by default. Legacy root-level ``/data`` and
    ``/data_channels_extra`` input files can be read by explicitly passing
    ``spad_data_key="data"`` or ``aux_data_key="data_channels_extra"``. The
    source datasets must use the BrightEyes
    channel-last layout:
    ``[repetition, z, y, x, time_bin, channel]`` or the circular 8D variant
    ``[repetition, z, y, x, circular_repetition, circular_point, time_bin,
    channel]``.

    When ``output_path`` is ``None``, the ``/output`` group is appended directly
    to ``data_path``. Pass an explicit ``output_path`` to copy the input file
    first and write outputs to the copy.
    """

    def __init__(
        self,
        data_path,
        output_path=None,
        *,
        in_place=False,
        overwrite=True,
        output_key=DEFAULT_OUTPUT_KEY,
        spad_data_key=None,
        aux_data_key=None,
        primary_data_key=None,
        extra_data_key=None,
        create_virtual_channels=True,
        create_sum_channels=True,
        create_sum=None,
        create_sum_channels_with_skew_correction=True,
        create_sum_using_shift=None,
        create_sum_shifted=None,
        sum_channels_run_id=DEFAULT_SUM_CHANNELS_RUN_ID,
        sum_run_id=None,
        sum_channels_with_skew_correction_run_id=DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID,
        sum_using_shift_run_id=None,
        channels=None,
        aux_channels=None,
        extra_channels=None,
        include_aux_sum=None,
        include_extra_sum=True,
        include_aux_shifted_sum=None,
        include_extra_shifted_sum=True,
        require_shifted_sum=False,
        shifted_sum_backend="gpu",
        shifted_sum_chunk_size=None,
        shifted_sum_reverse_shifts=True,
        shifted_sum_show_progress=False,
        compression="gzip",
        max_block_bytes=DEFAULT_MAX_BLOCK_BYTES,
    ):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path) if output_path is not None else None
        self.in_place = bool(in_place)
        self.overwrite = bool(overwrite)
        self.output_key = str(output_key).strip("/") or DEFAULT_OUTPUT_KEY
        self.primary_data_key = self._override_if_not_none(primary_data_key, spad_data_key)
        self.extra_data_key = self._override_if_not_none(extra_data_key, aux_data_key)
        self.create_virtual_channels = bool(create_virtual_channels)
        if create_sum is not None:
            self._warn_deprecated_alias("create_sum", "create_sum_channels")
            create_sum_channels = create_sum
        self.create_sum_channels = bool(create_sum_channels)
        for alias, value in (
            ("create_sum_using_shift", create_sum_using_shift),
            ("create_sum_shifted", create_sum_shifted),
        ):
            if value is not None:
                self._warn_deprecated_alias(
                    alias,
                    "create_sum_channels_with_skew_correction",
                )
                create_sum_channels_with_skew_correction = value
        self.create_sum_channels_with_skew_correction = bool(
            create_sum_channels_with_skew_correction
        )
        if sum_run_id is not None:
            self._warn_deprecated_alias("sum_run_id", "sum_channels_run_id")
            sum_channels_run_id = sum_run_id
        if sum_using_shift_run_id is not None:
            self._warn_deprecated_alias(
                "sum_using_shift_run_id",
                "sum_channels_with_skew_correction_run_id",
            )
            sum_channels_with_skew_correction_run_id = sum_using_shift_run_id
        self.sum_channels_run_id = str(sum_channels_run_id)
        self.sum_channels_with_skew_correction_run_id = str(
            sum_channels_with_skew_correction_run_id
        )
        self.channels = channels
        self.extra_channels = self._override_if_not_none(extra_channels, aux_channels)
        include_extra_sum = self._override_if_not_none(include_extra_sum, include_aux_sum)
        include_extra_shifted_sum = self._override_if_not_none(
            include_extra_shifted_sum,
            include_aux_shifted_sum,
        )
        self.include_extra_sum = bool(include_extra_sum)
        self.include_extra_shifted_sum = bool(include_extra_shifted_sum)
        self.require_shifted_sum = bool(require_shifted_sum)
        self.shifted_sum_backend = str(shifted_sum_backend)
        self.shifted_sum_chunk_size = shifted_sum_chunk_size
        self.shifted_sum_reverse_shifts = bool(shifted_sum_reverse_shifts)
        self.shifted_sum_show_progress = bool(shifted_sum_show_progress)
        self.compression = compression
        self.max_block_bytes = int(max_block_bytes)

        if self.max_block_bytes <= 0:
            raise ValueError("max_block_bytes must be positive")

    @staticmethod
    def _override_if_not_none(value, *overrides):
        for override in overrides:
            if override is not None:
                value = override
        return value

    @staticmethod
    def _warn_deprecated_alias(alias, canonical):
        warnings.warn(
            (
                f"H5OutputBuilder parameter '{alias}' is deprecated; "
                f"use '{canonical}' instead."
            ),
            DeprecationWarning,
            stacklevel=3,
        )

    @property
    def create_sum(self):
        self._warn_deprecated_alias("create_sum", "create_sum_channels")
        return self.create_sum_channels

    @create_sum.setter
    def create_sum(self, value):
        self._warn_deprecated_alias("create_sum", "create_sum_channels")
        self.create_sum_channels = bool(value)

    @property
    def create_sum_using_shift(self):
        self._warn_deprecated_alias(
            "create_sum_using_shift",
            "create_sum_channels_with_skew_correction",
        )
        return self.create_sum_channels_with_skew_correction

    @create_sum_using_shift.setter
    def create_sum_using_shift(self, value):
        self._warn_deprecated_alias(
            "create_sum_using_shift",
            "create_sum_channels_with_skew_correction",
        )
        self.create_sum_channels_with_skew_correction = bool(value)

    @property
    def create_sum_shifted(self):
        self._warn_deprecated_alias(
            "create_sum_shifted",
            "create_sum_channels_with_skew_correction",
        )
        return self.create_sum_channels_with_skew_correction

    @create_sum_shifted.setter
    def create_sum_shifted(self, value):
        self._warn_deprecated_alias(
            "create_sum_shifted",
            "create_sum_channels_with_skew_correction",
        )
        self.create_sum_channels_with_skew_correction = bool(value)

    @property
    def sum_run_id(self):
        self._warn_deprecated_alias("sum_run_id", "sum_channels_run_id")
        return self.sum_channels_run_id

    @sum_run_id.setter
    def sum_run_id(self, value):
        self._warn_deprecated_alias("sum_run_id", "sum_channels_run_id")
        self.sum_channels_run_id = str(value)

    @property
    def sum_using_shift_run_id(self):
        self._warn_deprecated_alias(
            "sum_using_shift_run_id",
            "sum_channels_with_skew_correction_run_id",
        )
        return self.sum_channels_with_skew_correction_run_id

    @sum_using_shift_run_id.setter
    def sum_using_shift_run_id(self, value):
        self._warn_deprecated_alias(
            "sum_using_shift_run_id",
            "sum_channels_with_skew_correction_run_id",
        )
        self.sum_channels_with_skew_correction_run_id = str(value)

    @staticmethod
    def _pixel_spacing_from_range(range_um, point_count):
        if not np.isfinite(range_um):
            return np.nan
        point_count = int(point_count)
        if point_count <= 0:
            return np.nan
        if point_count == 1:
            return 0.0
        return float(range_um) / float(point_count - 1)

    def _run_path(self, run_id, *parts):
        path_parts = [self.output_key, str(run_id).strip("/")]
        path_parts.extend(str(part).strip("/") for part in parts if part)
        return "/" + "/".join(path_parts)

    def _metadata_path(self, run_id):
        return self._run_path(run_id, "metadata")

    def _time_axis_path(self, run_id):
        return self._run_path(run_id, "axes/time_ns")

    def _product_path(self, run_id, product_name):
        return self._run_path(run_id, "products", product_name)

    @staticmethod
    def _trace_collection_defaults():
        return {
            "default_irf_trace_id": "",
            "default_ref_trace_id": "",
        }

    @staticmethod
    def _register_trace_output(output_group, trace_kind, output_id, output_path):
        default_attrs = {
            SUM_IRF_TRACE_OUTPUT_KIND: "default_irf_trace_id",
            SUM_REFERENCE_TRACE_OUTPUT_KIND: "default_ref_trace_id",
        }
        default_id_attr = default_attrs.get(trace_kind)
        if default_id_attr is None:
            return
        default_id = str(output_group.attrs.get(default_id_attr, ""))
        if not default_id or default_id == output_id:
            output_group.attrs[default_id_attr] = output_id

    def _create_common_run_groups(self, run_group):
        run_group.create_group("intermediates")
        run_group.create_group("tables")
        provenance_group = run_group.create_group("provenance")
        self._set_attrs(
            provenance_group,
            {
                "command": "H5OutputBuilder.build",
                **self._source_file_hash_attrs(run_group.file),
            },
        )
        views_group = run_group.create_group("views")
        self._set_attrs(views_group, {"description": "Optional display-friendly products."})

    def _default_output_path(self):
        suffix = self.data_path.suffix or ".h5"
        return self.data_path.with_name(f"{self.data_path.stem}_output{suffix}")

    def _source_file_hash_attrs(self, handle=None):
        if handle is not None:
            return channel_fingerprint_file_hash_attrs(handle, prefix="source_file")
        with h5py.File(self.data_path, "r") as source_handle:
            return channel_fingerprint_file_hash_attrs(source_handle, prefix="source_file")

    def _prepare_output_file(self):
        if self.in_place or self.output_path is None:
            if self.output_path is not None and self.output_path.resolve() != self.data_path.resolve():
                raise ValueError("output_path must be omitted or equal to data_path when in_place=True")
            return self.data_path

        output_path = self.output_path
        if output_path.resolve() == self.data_path.resolve():
            raise ValueError("output_path must be different from data_path unless in_place=True")
        if output_path.exists():
            if not self.overwrite:
                raise FileExistsError(f"output file already exists: {output_path}")
            output_path.unlink()
        shutil.copy2(self.data_path, output_path)
        return output_path

    _prepare_attr_value = staticmethod(H5DataCalibrator._prepare_attr_value)
    _metadata_get = staticmethod(H5DataCalibrator._metadata_get)
    _spatial_axis_values = staticmethod(H5DataCalibrator._spatial_axis_values)

    @classmethod
    def _set_attrs(cls, node, attrs):
        H5DataCalibrator._set_group_attrs(node, attrs)

    @staticmethod
    def _safe_float(value, default=np.nan):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value):
            return default
        return value

    @staticmethod
    def _safe_int(value, default=None):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _package_version():
        if version is None:
            return "unknown"
        try:
            return version("brighteyes-mcs-file")
        except PackageNotFoundError:
            return "unknown"

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _read_mcs_metadata(path):
        try:
            return mcs.metadata_load(str(path))
        except Exception:
            return None

    @staticmethod
    def _get_existing_path(handle, candidates):
        for candidate in candidates:
            key = str(candidate).strip("/")
            if key in handle:
                return f"/{key}"
        return ""

    @staticmethod
    def _find_dataset(handle, explicit_key, default_candidates, label, required):
        candidates = [explicit_key] if explicit_key is not None else list(default_candidates)
        for candidate in candidates:
            for key in _dataset_key_candidates(candidate):
                if key not in handle:
                    continue
                obj = handle[key]
                if isinstance(obj, h5py.Dataset):
                    return obj
                if explicit_key is not None and key == str(explicit_key).strip("/"):
                    continue
        if required:
            tried = ", ".join(
                key
                for candidate in candidates
                for key in _dataset_key_candidates(candidate)
            )
            raise KeyError(f"could not find {label} dataset; tried {tried}")
        return None

    @staticmethod
    def _validate_channel_last_dataset(dataset, label):
        if dataset.ndim not in AXIS_ORDERS:
            raise ValueError(
                f"{label} dataset must be 6D or 8D with channel as the final axis, "
                f"got shape {dataset.shape}"
            )
        if dataset.shape[-2] <= 0 or dataset.shape[-1] <= 0:
            raise ValueError(f"{label} dataset must contain positive time and channel dimensions")

    @staticmethod
    def _resolve_channels(channels, channel_count, label):
        if channels is None:
            return list(range(int(channel_count)))
        resolved = sorted({int(channel) for channel in channels})
        if len(resolved) == 0:
            raise ValueError(f"{label} must contain at least one channel")
        for channel in resolved:
            if channel < 0 or channel >= channel_count:
                raise IndexError(f"{label} channel {channel} out of range for {channel_count} channels")
        return resolved

    @staticmethod
    def _source_selection_string(ndim, channel_index):
        return "[" + ", ".join([":"] * (ndim - 1) + [str(channel_index)]) + "]"

    @staticmethod
    def _channel_indexer(channels, channel_count):
        if channels == list(range(channel_count)):
            return slice(None)
        return channels

    @staticmethod
    def _default_sum_dtype(source_dtype, channel_count):
        dtype = np.dtype(source_dtype)
        if dtype.kind == "u":
            max_sum = int(np.iinfo(dtype).max) * int(channel_count)
            if max_sum <= np.iinfo(np.uint32).max:
                return np.dtype("uint32")
            return np.dtype("uint64")
        if dtype.kind == "i":
            return np.dtype("int64")
        if dtype.kind == "f":
            return np.dtype("float64")
        return np.dtype("float64")

    @staticmethod
    def _time_axis(output_time_bins, time_bin_ns):
        if np.isfinite(time_bin_ns):
            return np.arange(output_time_bins, dtype=np.float64) * float(time_bin_ns), "ns"
        return np.arange(output_time_bins, dtype=np.float64), "bin"

    def _source_info(self, dataset, kind):
        self._validate_channel_last_dataset(dataset, kind)
        axis_info = AXIS_ORDERS[dataset.ndim]
        channel_axis_paths = {
            "spad": "/raw/axes/detector_channel_index",
            "aux": "/raw/axes/aux_channel_index",
            "analog": "/raw/axes/analog_channel_index",
        }
        return {
            "dataset": dataset,
            "kind": kind,
            "path": dataset.name,
            "shape": tuple(dataset.shape),
            "channel_count": int(dataset.shape[-1]),
            "time_bins": int(dataset.shape[-2]),
            "channel_axis_path": channel_axis_paths.get(kind, ""),
            "source_axis_order": axis_info["source"],
            "output_axis_order": axis_info["virtual"],
            "data_layout": axis_info["layout"],
            "subpixel_scan_mode": axis_info["subpixel_scan_mode"],
            "units": "adc_counts" if kind == "analog" else "counts",
        }

    @staticmethod
    def _calibration_path_for_source(handle, source_info):
        if source_info is None:
            return ""
        candidates = [
            f"/calibration/results/{source_info['kind']}",
        ]
        if source_info["kind"] == "spad":
            candidates.append("/calibration/results/primary")
        elif source_info["kind"] == "aux":
            candidates.append("/calibration/results/extra")
        for candidate in candidates:
            if candidate.strip("/") in handle:
                return candidate
        return ""

    @classmethod
    def _find_child_dataset(cls, handle, parent_path, candidates):
        if not parent_path:
            return None, ""
        parent_key = str(parent_path).strip("/")
        if parent_key not in handle:
            return None, ""
        parent = handle[parent_key]
        if not isinstance(parent, h5py.Group):
            return None, ""
        for candidate in candidates:
            key = str(candidate).strip("/")
            if key in parent and isinstance(parent[key], h5py.Dataset):
                dataset = parent[key]
                return dataset, dataset.name
        return None, ""

    def _resolve_channel_skew(self, handle, source_info, selected_channels):
        calibration_path = self._calibration_path_for_source(handle, source_info)
        if not calibration_path:
            return None, "", f"no calibration group found for {source_info['path']}"

        channel_skew_dataset, channel_skew_path = self._find_child_dataset(
            handle,
            calibration_path,
            (
                "timing/channel_skew_bins",
            ),
        )
        if channel_skew_dataset is None:
            return None, "", f"no channel_skew dataset found under {calibration_path}"

        channel_skew = np.asarray(channel_skew_dataset[...], dtype=float)
        if channel_skew.ndim != 1:
            return None, channel_skew_path, (
                f"channel_skew at {channel_skew_path} must be 1D, got {channel_skew.shape}"
            )

        channel_index_dataset, _ = self._find_child_dataset(
            handle,
            calibration_path,
            (
                "channels/index",
            ),
        )
        if channel_index_dataset is not None:
            channel_index = np.asarray(channel_index_dataset[...], dtype=int)
        elif channel_skew.shape[0] == source_info["channel_count"]:
            channel_index = np.arange(source_info["channel_count"], dtype=int)
        else:
            return None, channel_skew_path, (
                f"channel_skew length {channel_skew.shape[0]} does not match source "
                f"channel count {source_info['channel_count']} and no channel_index was found"
            )

        if channel_index.ndim != 1 or channel_index.shape != channel_skew.shape:
            return None, channel_skew_path, (
                "channel_index must be 1D with the same length as channel_skew "
                f"(got {channel_index.shape} and {channel_skew.shape})"
            )

        skew_by_channel = {int(channel): float(skew) for channel, skew in zip(channel_index, channel_skew)}
        missing_channels = [channel for channel in selected_channels if channel not in skew_by_channel]
        if missing_channels:
            return None, channel_skew_path, (
                f"channel_skew at {channel_skew_path} is missing source channels {missing_channels}"
            )

        selected_skew = np.asarray(
            [skew_by_channel[int(channel)] for channel in selected_channels],
            dtype=float,
        )
        return selected_skew, channel_skew_path, ""

    def _selected_aligned_trace(
        self,
        handle,
        calibration_path,
        trace_name,
        selected_channels,
    ):
        trace_dataset, trace_path = self._find_child_dataset(
            handle,
            calibration_path,
            (f"aligned/{trace_name}",),
        )
        if trace_dataset is None:
            return None, "", "", f"no aligned/{trace_name} dataset found under {calibration_path}"

        trace = np.asarray(trace_dataset[...], dtype=float)
        if trace.ndim != 2:
            return None, trace_path, "", (
                f"aligned trace at {trace_path} must be 2D, got {trace.shape}"
            )

        channel_index_dataset, channel_index_path = self._find_child_dataset(
            handle,
            calibration_path,
            ("channels/index",),
        )
        if channel_index_dataset is None:
            return None, trace_path, "", f"no channels/index dataset found under {calibration_path}"

        channel_index = np.asarray(channel_index_dataset[...], dtype=int)
        if channel_index.ndim != 1 or channel_index.shape != (trace.shape[1],):
            return None, trace_path, channel_index_path, (
                "channels/index must be 1D with the same length as the "
                f"aligned trace channel axis (got {channel_index.shape} and {trace.shape})"
            )

        position_by_channel = {
            int(channel): position
            for position, channel in enumerate(channel_index)
        }
        missing_channels = [
            channel for channel in selected_channels if channel not in position_by_channel
        ]
        if missing_channels:
            return None, trace_path, channel_index_path, (
                f"aligned trace at {trace_path} is missing source channels {missing_channels}"
            )

        positions = [position_by_channel[int(channel)] for channel in selected_channels]
        return trace[:, positions], trace_path, channel_index_path, ""

    def _resolve_source_infos(self, handle):
        primary_dataset = self._find_dataset(
            handle,
            self.primary_data_key,
            DEFAULT_SPAD_DATA_KEYS,
            "SPAD data",
            required=self.create_sum_channels or self.create_sum_channels_with_skew_correction,
        )
        extra_dataset = self._find_dataset(
            handle,
            self.extra_data_key,
            DEFAULT_AUX_DATA_KEYS,
            "aux data",
            required=False,
        )

        primary_info = self._source_info(primary_dataset, "spad") if primary_dataset is not None else None
        extra_info = self._source_info(extra_dataset, "aux") if extra_dataset is not None else None
        if primary_info is not None and extra_info is not None:
            if primary_info["shape"][:-1] != extra_info["shape"][:-1]:
                raise ValueError(
                    "SPAD and aux data must share all non-channel dimensions "
                    f"(got {primary_info['shape']} and {extra_info['shape']})"
                )
        return primary_info, extra_info

    def _resolve_analog_source_info(self, handle):
        analog_dataset = self._find_dataset(
            handle,
            None,
            DEFAULT_ANALOG_DATA_KEYS,
            "analog data",
            required=False,
        )
        return self._source_info(analog_dataset, "analog") if analog_dataset is not None else None

    def _timing_attr_source(self, handle, primary_info):
        for metadata_timing_path in (
            "/raw/metadata/acquisition/timing",
            "/metadata/acquisition/timing",
        ):
            if metadata_timing_path.strip("/") in handle:
                return metadata_timing_path, handle[metadata_timing_path].attrs

        calibration_path = self._calibration_path_for_source(handle, primary_info)
        if calibration_path:
            return calibration_path, handle[calibration_path].attrs

        return "", {}

    def _build_metadata_attrs(self, handle, primary_info, channels, metadata, fallback_time_axis_run_id=None):
        source_shape = primary_info["shape"]
        nrep, nz, ny, nx = (int(source_shape[i]) for i in range(4))
        circular_repetition_count = int(source_shape[4]) if len(source_shape) == 8 else 1
        circular_point_count = int(source_shape[5]) if len(source_shape) == 8 else 1
        output_time_bins = int(source_shape[-2])
        source_metadata_path = self._get_existing_path(
            handle,
            (
                "/raw/metadata",
                "/metadata",
                "/calibration/provenance/input_data_metadata",
            ),
        )
        source_axes_path = self._get_existing_path(handle, ("/raw/axes", "/axes"))
        source_time_axis_path = self._get_existing_path(
            handle,
            ("/raw/axes/digital_time_ns", "/axes/digital_time_ns"),
        )
        source_timing_metadata_path, timing_attrs = self._timing_attr_source(handle, primary_info)

        laser_period_ns = self._safe_float(timing_attrs.get("laser_period_ns", np.nan))
        if not np.isfinite(laser_period_ns):
            laser_period_ns = self._safe_float(getattr(self, "period_ns", np.nan))

        laser_frequency_mhz = self._safe_float(timing_attrs.get("laser_frequency_mhz", np.nan))
        if not np.isfinite(laser_frequency_mhz):
            if np.isfinite(laser_period_ns) and laser_period_ns > 0:
                laser_frequency_mhz = 1000.0 / laser_period_ns
            else:
                laser_frequency_mhz = self._safe_float(self._metadata_get(metadata, "dfd_freq", np.nan))

        if not np.isfinite(laser_period_ns) and np.isfinite(laser_frequency_mhz) and laser_frequency_mhz > 0:
            laser_period_ns = 1000.0 / laser_frequency_mhz

        time_bin_ns = self._safe_float(
            timing_attrs.get(
                "time_bin_ns",
                timing_attrs.get(
                    "digital_time_bin_ns",
                    timing_attrs.get("timebin_in_ns", timing_attrs.get("digital_timebin_ns", np.nan)),
                ),
            )
        )

        dfd_active = bool(
            timing_attrs.get(
                "dfd_active",
                self._metadata_get(metadata, "dfd_active", self._metadata_get(metadata, "dfd_activate", False)),
            )
        )
        acquisition_mode = str(timing_attrs.get("acquisition_mode", "dfd" if dfd_active else "normal"))
        timing_reference = str(
            timing_attrs.get(
                "timing_reference",
                "laser_period" if acquisition_mode == "dfd" else "pixel_dwell",
            )
        )

        range_x_um = self._safe_float(self._metadata_get(metadata, "rangex", np.nan))
        range_y_um = self._safe_float(self._metadata_get(metadata, "rangey", np.nan))
        range_z_um = self._safe_float(self._metadata_get(metadata, "rangez", np.nan))
        offset_x_um = H5DataCalibrator._calibrated_offset_um(metadata, "x")
        offset_y_um = H5DataCalibrator._calibrated_offset_um(metadata, "y")
        offset_z_um = H5DataCalibrator._calibrated_offset_um(metadata, "z")

        pixel_size_x_um = self._safe_float(
            self._metadata_get(metadata, "pixel_size_x_um", np.nan),
            self._pixel_spacing_from_range(range_x_um, nx),
        )
        pixel_size_y_um = self._safe_float(
            self._metadata_get(metadata, "pixel_size_y_um", np.nan),
            self._pixel_spacing_from_range(range_y_um, ny),
        )
        pixel_size_z_um = self._safe_float(
            self._metadata_get(metadata, "pixel_size_z_um", np.nan),
            self._pixel_spacing_from_range(range_z_um, nz),
        )

        pixel_dwell_time_us = self._safe_float(timing_attrs.get("pixel_dwell_time_us", np.nan))
        if not np.isfinite(pixel_dwell_time_us):
            dt_us = self._safe_float(self._metadata_get(metadata, "dt", np.nan))
            nbin = self._safe_float(self._metadata_get(metadata, "nbin", np.nan))
            if np.isfinite(dt_us) and np.isfinite(nbin):
                pixel_dwell_time_us = dt_us * nbin
        if not np.isfinite(time_bin_ns) and np.isfinite(laser_period_ns):
            time_bin_ns = laser_period_ns / output_time_bins
        if not np.isfinite(time_bin_ns) and np.isfinite(pixel_dwell_time_us):
            time_bin_ns = pixel_dwell_time_us * 1000.0 / output_time_bins
        pixel_dwell_time_ns = pixel_dwell_time_us * 1000.0 if np.isfinite(pixel_dwell_time_us) else np.nan
        frame_time_s = pixel_dwell_time_us * nx * ny / 1e6 if np.isfinite(pixel_dwell_time_us) else np.nan
        volume_time_s = frame_time_s * nz if np.isfinite(frame_time_s) else np.nan
        acquisition_duration_s = volume_time_s * nrep if np.isfinite(volume_time_s) else np.nan

        if not source_time_axis_path:
            if fallback_time_axis_run_id is None:
                fallback_time_axis_run_id = self.sum_channels_run_id
            source_time_axis_path = self._time_axis_path(fallback_time_axis_run_id)

        return {
            "source_metadata_path": source_metadata_path,
            "source_timing_metadata_path": source_timing_metadata_path,
            "source_axes_path": source_axes_path,
            "source_time_axis_path": source_time_axis_path,
            "acquisition_mode": acquisition_mode,
            "timing_reference": timing_reference,
            "data_layout": primary_info["data_layout"],
            "subpixel_scan_mode": primary_info["subpixel_scan_mode"],
            "nrep": nrep,
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "circular_repetition_count": circular_repetition_count,
            "circular_point_count": circular_point_count,
            "output_nrep": nrep,
            "output_nz": nz,
            "output_ny": ny,
            "output_nx": nx,
            "output_circular_repetition_count": circular_repetition_count,
            "output_circular_point_count": circular_point_count,
            "range_x_um": range_x_um,
            "range_y_um": range_y_um,
            "range_z_um": range_z_um,
            "offset_x_um": offset_x_um,
            "offset_y_um": offset_y_um,
            "offset_z_um": offset_z_um,
            "range_to_pixel_spacing_rule": "range_um / (n - 1) for n > 1; 0 for n == 1",
            "pixel_size_x_um": pixel_size_x_um,
            "pixel_size_y_um": pixel_size_y_um,
            "pixel_size_z_um": pixel_size_z_um,
            "output_pixel_size_x_um": pixel_size_x_um,
            "output_pixel_size_y_um": pixel_size_y_um,
            "output_pixel_size_z_um": pixel_size_z_um,
            "time_bins": output_time_bins,
            "output_time_bins": output_time_bins,
            "selected_time_bins_json": json.dumps(list(range(output_time_bins))),
            "time_bin_ns": time_bin_ns,
            "time_axis_start_ns": 0.0,
            "time_axis_last_ns": time_bin_ns * max(output_time_bins - 1, 0)
            if np.isfinite(time_bin_ns)
            else np.nan,
            "time_axis_span_ns": time_bin_ns * output_time_bins
            if np.isfinite(time_bin_ns)
            else np.nan,
            "laser_frequency_mhz": laser_frequency_mhz,
            "laser_period_ns": laser_period_ns,
            "dwell_time_per_circular_point_us": self._safe_float(
                timing_attrs.get("dwell_time_per_circular_point_us", np.nan)
            ),
            "pixel_dwell_time_us": pixel_dwell_time_us,
            "pixel_dwell_time_ns": pixel_dwell_time_ns,
            "frame_time_s": frame_time_s,
            "volume_time_s": volume_time_s,
            "acquisition_duration_s": acquisition_duration_s,
            "spad_channel_count": primary_info["channel_count"],
            "selected_channel_count": len(channels),
            "selected_channels_json": json.dumps(channels),
            "channel_aggregation": SUM_CHANNELS_AGGREGATION,
            "use_calibration": bool(source_timing_metadata_path.startswith("/calibration")),
            "calibration_result_path": source_timing_metadata_path
            if source_timing_metadata_path.startswith("/calibration")
            else "",
            "metadata_warnings_json": "[]",
        }

    def _create_axis_dataset(self, axes_group, name, data, units, long_name, axis):
        dataset = axes_group.create_dataset(name, data=np.asarray(data, dtype=np.float64))
        self._set_attrs(
            dataset,
            {
                "units": units,
                "long_name": long_name,
                "axis": axis,
            },
        )
        return dataset

    def _create_axes(self, run_group, metadata_attrs):
        axes_group = run_group.create_group("axes")
        nrep = int(metadata_attrs["output_nrep"])
        nz = int(metadata_attrs["output_nz"])
        ny = int(metadata_attrs["output_ny"])
        nx = int(metadata_attrs["output_nx"])
        circular_repetition_count = int(
            metadata_attrs.get("output_circular_repetition_count", 1)
        )
        circular_point_count = int(metadata_attrs.get("output_circular_point_count", 1))
        output_time_bins = int(metadata_attrs["output_time_bins"])

        self._create_axis_dataset(
            axes_group,
            "repetition_index",
            np.arange(nrep, dtype=np.float64),
            "index",
            "repetition index",
            "repetition",
        )

        for name, count, range_key, offset_key, pixel_key, axis in (
            ("z_um", nz, "range_z_um", "offset_z_um", "output_pixel_size_z_um", "z"),
            ("y_um", ny, "range_y_um", "offset_y_um", "output_pixel_size_y_um", "y"),
            ("x_um", nx, "range_x_um", "offset_x_um", "output_pixel_size_x_um", "x"),
        ):
            values, units = self._spatial_axis_values(
                count,
                metadata_attrs[range_key],
                metadata_attrs[offset_key],
            )
            dataset = self._create_axis_dataset(
                axes_group,
                name,
                values,
                units,
                f"{axis} position",
                axis,
            )
            self._set_attrs(
                dataset,
                {
                    "range_um": metadata_attrs[range_key],
                    "offset_um": metadata_attrs[offset_key],
                    "pixel_size_um": metadata_attrs[pixel_key],
                    "coordinate_rule": "linspace(offset_um - range_um/2, offset_um + range_um/2, n)",
                },
            )

        if metadata_attrs.get("data_layout") == "circular_8d":
            self._create_axis_dataset(
                axes_group,
                "circular_repetition_index",
                np.arange(circular_repetition_count, dtype=np.float64),
                "index",
                "circular repetition index",
                "circular_repetition",
            )
            self._create_axis_dataset(
                axes_group,
                "circular_point_index",
                np.arange(circular_point_count, dtype=np.float64),
                "index",
                "circular point index",
                "circular_point",
            )

        time_values, time_units = self._time_axis(output_time_bins, metadata_attrs["time_bin_ns"])
        self._create_axis_dataset(axes_group, "time_ns", time_values, time_units, "time", "time_bin")
        return axes_group

    def _create_virtual_channel_dataset(self, group, source_info, channel_index):
        source_dataset = source_info["dataset"]
        name = f"channel_{int(channel_index)}"
        layout = h5py.VirtualLayout(shape=source_dataset.shape[:-1], dtype=source_dataset.dtype)
        source = h5py.VirtualSource(
            ".",
            source_dataset.name,
            shape=source_dataset.shape,
            dtype=source_dataset.dtype,
        )
        source_selection = (slice(None),) * (source_dataset.ndim - 1) + (int(channel_index),)
        layout[(slice(None),) * (source_dataset.ndim - 1)] = source[source_selection]
        dataset = group.create_virtual_dataset(name, layout, fillvalue=0)
        self._set_attrs(
            dataset,
            {
                "virtual_channel_name": name,
                "virtual_channel_type": source_info["kind"],
                "virtual_source_file": ".",
                "source_data_path": source_info["path"],
                "source_channel_index": int(channel_index),
                "source_channel_axis": -1,
                "source_channel_axis_path": source_info["channel_axis_path"],
                "source_selection": self._source_selection_string(source_dataset.ndim, channel_index),
                "axis_order": source_info["output_axis_order"],
                "units": source_info["units"],
                "is_virtual_dataset": True,
            },
        )

    def _create_virtual_channel_group(self, parent_group, source_info):
        group = parent_group.create_group(source_info["kind"])
        self._set_attrs(
            group,
            {
                "description": (
                    f"Virtual datasets mapping individual {source_info['kind']} "
                    "source channels."
                ),
                "virtual_channel_type": source_info["kind"],
                "source_data_path": source_info["path"],
                "source_channel_axis": -1,
                "source_channel_axis_path": source_info["channel_axis_path"],
                "channel_count": source_info["channel_count"],
                "source_axis_order": source_info["source_axis_order"],
                "virtual_axis_order": source_info["output_axis_order"],
                "data_layout": source_info["data_layout"],
                "units": source_info["units"],
                "naming_rule": "channel_<channel_index>",
            },
        )
        for channel_index in range(source_info["channel_count"]):
            self._create_virtual_channel_dataset(group, source_info, channel_index)

    def _create_virtual_channels(self, output_group, primary_info, extra_info, analog_info=None):
        group = output_group.create_group("virtual_channels")
        source_infos = [
            source_info
            for source_info in (primary_info, extra_info, analog_info)
            if source_info is not None
        ]
        axis_source_info = source_infos[0] if source_infos else None
        self._set_attrs(
            group,
            {
                "description": (
                    "Typed virtual datasets mapping individual source channels."
                ),
                "spad_source_data_path": primary_info["path"] if primary_info is not None else "",
                "aux_source_data_path": extra_info["path"] if extra_info is not None else "",
                "analog_source_data_path": analog_info["path"] if analog_info is not None else "",
                "spad_channel_count": primary_info["channel_count"] if primary_info is not None else 0,
                "aux_channel_count": extra_info["channel_count"] if extra_info is not None else 0,
                "analog_channel_count": analog_info["channel_count"] if analog_info is not None else 0,
                "source_axis_order": axis_source_info["source_axis_order"] if axis_source_info is not None else "",
                "virtual_axis_order": axis_source_info["output_axis_order"] if axis_source_info is not None else "",
                "naming_rule": "<kind>/channel_<channel_index>",
                "kind_groups_json": json.dumps([source_info["kind"] for source_info in source_infos]),
            },
        )
        for source_info in source_infos:
            self._create_virtual_channel_group(group, source_info)

    def _sum_channels_to_dataset(self, source_dataset, target_dataset, channels):
        channel_count = int(source_dataset.shape[-1])
        channel_indexer = self._channel_indexer(channels, channel_count)
        output_shape = source_dataset.shape[:-1]
        y_size = int(source_dataset.shape[2])
        x_size = int(source_dataset.shape[3])
        suffix_shape = tuple(int(size) for size in source_dataset.shape[4:-1])
        suffix_size = int(np.prod(suffix_shape, dtype=np.int64)) if suffix_shape else 1
        bytes_per_y = max(
            1,
            x_size
            * suffix_size
            * len(channels)
            * max(np.dtype(source_dataset.dtype).itemsize, np.dtype(target_dataset.dtype).itemsize),
        )
        y_chunk = max(1, min(y_size, self.max_block_bytes // bytes_per_y))
        prefix_shape = tuple(int(size) for size in source_dataset.shape[:2])
        suffix_slices = (slice(None),) * len(suffix_shape)

        for prefix in np.ndindex(prefix_shape):
            for y0 in range(0, y_size, y_chunk):
                y1 = min(y0 + y_chunk, y_size)
                source_selection = (
                    prefix
                    + (slice(y0, y1), slice(None))
                    + suffix_slices
                    + (channel_indexer,)
                )
                target_selection = (
                    prefix
                    + (slice(y0, y1), slice(None))
                    + suffix_slices
                )
                block = source_dataset[source_selection]
                summed = np.sum(block, axis=-1, dtype=target_dataset.dtype)
                expected_shape = output_shape[:2] + (y1 - y0, output_shape[3]) + output_shape[4:]
                if summed.shape != expected_shape[2:]:
                    summed = np.reshape(summed, expected_shape[2:])
                target_dataset[target_selection] = summed

    def _sum_channels_with_shifts_to_dataset(self, source_dataset, target_dataset, channels, shifts):
        channel_count = int(source_dataset.shape[-1])
        channel_indexer = self._channel_indexer(channels, channel_count)
        shifts = np.asarray(shifts, dtype=float)
        if shifts.shape != (len(channels),):
            raise ValueError(f"shifts must have shape ({len(channels)},), got {shifts.shape}")

        output_shape = source_dataset.shape[:-1]
        y_size = int(source_dataset.shape[2])
        x_size = int(source_dataset.shape[3])
        suffix_shape = tuple(int(size) for size in source_dataset.shape[4:-1])
        suffix_size = int(np.prod(suffix_shape, dtype=np.int64)) if suffix_shape else 1
        bytes_per_y = max(
            1,
            x_size
            * suffix_size
            * len(channels)
            * max(np.dtype(source_dataset.dtype).itemsize, np.dtype(target_dataset.dtype).itemsize),
        )
        y_chunk = max(1, min(y_size, self.max_block_bytes // bytes_per_y))
        prefix_shape = tuple(int(size) for size in source_dataset.shape[:2])
        suffix_slices = (slice(None),) * len(suffix_shape)

        for prefix in np.ndindex(prefix_shape):
            for y0 in range(0, y_size, y_chunk):
                y1 = min(y0 + y_chunk, y_size)
                source_selection = (
                    prefix
                    + (slice(y0, y1), slice(None))
                    + suffix_slices
                    + (channel_indexer,)
                )
                target_selection = (
                    prefix
                    + (slice(y0, y1), slice(None))
                    + suffix_slices
                )
                block = source_dataset[source_selection]
                shifted_sum = Alignment.sum_channel_applying_shifts(
                    block,
                    shifts,
                    axis=(),
                    reverse_shifts=self.shifted_sum_reverse_shifts,
                    backend=self.shifted_sum_backend,
                    chunk_size=self.shifted_sum_chunk_size,
                    show_progress=self.shifted_sum_show_progress,
                )
                expected_shape = output_shape[:2] + (y1 - y0, output_shape[3]) + output_shape[4:]
                if shifted_sum.shape != expected_shape[2:]:
                    shifted_sum = np.reshape(shifted_sum, expected_shape[2:])
                target_dataset[target_selection] = shifted_sum

    def _create_sum_channels_product(
        self,
        products_group,
        name,
        source_info,
        channels,
        metadata_attrs,
        source_calibration_path,
        long_name,
        description,
    ):
        source_dataset = source_info["dataset"]
        dtype = self._default_sum_dtype(source_dataset.dtype, len(channels))
        kwargs = {
            "shape": source_dataset.shape[:-1],
            "dtype": dtype,
            "chunks": True,
        }
        if self.compression:
            kwargs["compression"] = self.compression
        dataset = products_group.create_dataset(name, **kwargs)
        self._set_attrs(
            dataset,
            {
                "units": "counts",
                "long_name": long_name,
                "axis_order": source_info["output_axis_order"],
                "source_data_path": source_info["path"],
                "source_calibration_path": source_calibration_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "metadata_path": self._metadata_path(self.sum_channels_run_id),
                "time_axis_path": self._time_axis_path(self.sum_channels_run_id),
                "time_bin_ns": metadata_attrs["time_bin_ns"],
                "laser_frequency_mhz": metadata_attrs["laser_frequency_mhz"],
                "laser_period_ns": metadata_attrs["laser_period_ns"],
                "source_channel_axis": -1,
                "selected_channels_json": json.dumps(channels),
                "channel_aggregation": SUM_CHANNELS_AGGREGATION,
                "description": description,
            },
        )
        self._sum_channels_to_dataset(source_dataset, dataset, channels)
        return dataset

    def _create_sum_channels_with_skew_correction_product(
        self,
        products_group,
        name,
        source_info,
        channels,
        shifts,
        channel_skew_path,
        metadata_attrs,
        source_calibration_path,
        long_name,
        description,
    ):
        source_dataset = source_info["dataset"]
        kwargs = {
            "shape": source_dataset.shape[:-1],
            "dtype": np.dtype("float64"),
            "chunks": True,
        }
        if self.compression:
            kwargs["compression"] = self.compression
        dataset = products_group.create_dataset(name, **kwargs)
        self._set_attrs(
            dataset,
            {
                "units": "counts",
                "long_name": long_name,
                "axis_order": source_info["output_axis_order"],
                "source_data_path": source_info["path"],
                "source_calibration_path": source_calibration_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "metadata_path": self._metadata_path(
                    self.sum_channels_with_skew_correction_run_id
                ),
                "time_axis_path": self._time_axis_path(
                    self.sum_channels_with_skew_correction_run_id
                ),
                "time_bin_ns": metadata_attrs["time_bin_ns"],
                "laser_frequency_mhz": metadata_attrs["laser_frequency_mhz"],
                "laser_period_ns": metadata_attrs["laser_period_ns"],
                "source_channel_axis": -1,
                "selected_channels_json": json.dumps(channels),
                "channel_aggregation": SUM_CHANNELS_WITH_SKEW_CORRECTION_AGGREGATION,
                "channel_skew_path": channel_skew_path,
                "channel_skew_json": json.dumps(np.asarray(shifts, dtype=float).tolist()),
                "reverse_shifts": self.shifted_sum_reverse_shifts,
                "shift_backend": self.shifted_sum_backend,
                "description": description,
            },
        )
        self._sum_channels_with_shifts_to_dataset(source_dataset, dataset, channels, shifts)
        return dataset

    def _create_common_shifted_calibration_trace(
        self,
        output_group,
        handle,
        output_id,
        trace_kind,
        calibration_path,
        trace_name,
        channels,
        shifts,
        channel_skew_path,
        metadata_attrs,
        long_name,
        description,
    ):
        trace, trace_path, channel_index_path, error = self._selected_aligned_trace(
            handle,
            calibration_path,
            trace_name,
            channels,
        )
        if error:
            output_group.attrs[f"{output_id}_skipped"] = True
            output_group.attrs[f"{output_id}_skip_reason"] = error
            return None

        shifted_sum = Alignment.sum_channel_applying_shifts(
            trace,
            shifts,
            axis=(),
            reverse_shifts=self.shifted_sum_reverse_shifts,
            backend=self.shifted_sum_backend,
            chunk_size=self.shifted_sum_chunk_size,
            show_progress=self.shifted_sum_show_progress,
        )
        if output_id in output_group:
            del output_group[output_id]
        run_group = output_group.create_group(output_id)
        trace_product_path = self._product_path(output_id, "trace")
        trace_metadata_path = self._metadata_path(output_id)
        trace_time_axis_path = self._time_axis_path(output_id)
        self._set_attrs(
            run_group,
            {
                "output_id": output_id,
                "output_type": "trace_tool",
                "trace_kind": trace_kind,
                "tool_name": long_name,
                "created_utc": self._utc_now(),
                "software_name": "brighteyes_mcs_file",
                "software_version": self._package_version(),
                "algorithm_name": "sum_channel_applying_shifts",
                "algorithm_version": "0.1.0",
                "source_output_run_id": self.sum_channels_with_skew_correction_run_id,
                "source_output_run_path": self._run_path(
                    self.sum_channels_with_skew_correction_run_id
                ),
                "source_trace_path": trace_path,
                "source_calibration_path": calibration_path,
                "source_channel_index_path": channel_index_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "source_time_axis_path": metadata_attrs["source_time_axis_path"],
                "output_data_path": trace_product_path,
                "metadata_path": trace_metadata_path,
                "time_axis_path": trace_time_axis_path,
                "channel_skew_path": channel_skew_path,
                "parameter_encoding": "attrs_and_json",
            },
        )

        inputs_group = run_group.create_group("inputs")
        self._set_attrs(
            inputs_group,
            {
                "source_trace_path": trace_path,
                "source_calibration_path": calibration_path,
                "source_channel_index_path": channel_index_path,
                "source_output_run_path": self._run_path(
                    self.sum_channels_with_skew_correction_run_id
                ),
                "selected_channels_json": json.dumps(channels),
                "channel_skew_path": channel_skew_path,
                "channel_skew_json": json.dumps(np.asarray(shifts, dtype=float).tolist()),
            },
        )

        metadata_group = run_group.create_group("metadata")
        self._set_attrs(metadata_group, metadata_attrs)

        parameters_group = run_group.create_group("parameters")
        self._set_attrs(
            parameters_group,
            {
                "parameters_json": json.dumps(
                    {
                        "channels": channels,
                        "reverse_shifts": self.shifted_sum_reverse_shifts,
                        "backend": self.shifted_sum_backend,
                        "chunk_size": self.shifted_sum_chunk_size,
                    }
                ),
                "tool_name": long_name,
                "tool_mode": trace_kind,
                "channel_selection_json": json.dumps(channels),
                "time_bin_selection_json": metadata_attrs["selected_time_bins_json"],
                "normalize_output": False,
                "use_calibration": True,
                "channel_skew_path": channel_skew_path,
                "reverse_shifts": self.shifted_sum_reverse_shifts,
                "backend": self.shifted_sum_backend,
            },
        )

        self._create_axes(run_group, metadata_attrs)
        products_group = run_group.create_group("products")
        dataset = products_group.create_dataset(
            "trace",
            data=np.asarray(shifted_sum, dtype=np.float64),
        )
        self._set_attrs(
            dataset,
            {
                "output_id": output_id,
                "output_run_path": run_group.name,
                "output_type": "trace",
                "trace_kind": trace_kind,
                "units": "normalized_counts",
                "long_name": long_name,
                "axis_order": "time_bin",
                "source_output_run_id": self.sum_channels_with_skew_correction_run_id,
                "source_output_run_path": self._run_path(
                    self.sum_channels_with_skew_correction_run_id
                ),
                "source_trace_path": trace_path,
                "source_calibration_path": calibration_path,
                "source_channel_index_path": channel_index_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "metadata_path": trace_metadata_path,
                "source_time_axis_path": metadata_attrs["source_time_axis_path"],
                "time_axis_path": trace_time_axis_path,
                "time_bin_ns": metadata_attrs["time_bin_ns"],
                "laser_frequency_mhz": metadata_attrs["laser_frequency_mhz"],
                "laser_period_ns": metadata_attrs["laser_period_ns"],
                "source_channel_axis": -1,
                "selected_channels_json": json.dumps(channels),
                "channel_aggregation": SUM_CHANNELS_WITH_SKEW_CORRECTION_AGGREGATION,
                "channel_skew_path": channel_skew_path,
                "channel_skew_json": json.dumps(np.asarray(shifts, dtype=float).tolist()),
                "algorithm_name": "sum_channel_applying_shifts",
                "algorithm_version": "0.1.0",
                "reverse_shifts": self.shifted_sum_reverse_shifts,
                "shift_backend": self.shifted_sum_backend,
                "description": description,
            },
        )
        self._create_common_run_groups(run_group)
        self._register_trace_output(output_group, trace_kind, output_id, dataset.name)
        return dataset

    def _create_sum_channels_run(self, handle, output_group, primary_info, extra_info):
        metadata = self._read_mcs_metadata(self.data_path)
        channels = self._resolve_channels(self.channels, primary_info["channel_count"], "channels")
        metadata_attrs = self._build_metadata_attrs(
            handle,
            primary_info,
            channels,
            metadata,
            fallback_time_axis_run_id=self.sum_channels_run_id,
        )
        extra_channels = None
        if extra_info is not None:
            extra_channels = self._resolve_channels(
                self.extra_channels,
                extra_info["channel_count"],
                "aux_channels",
            )
        primary_calibration_path = self._calibration_path_for_source(handle, primary_info)
        extra_calibration_path = self._calibration_path_for_source(handle, extra_info)

        run_group = output_group.create_group(self.sum_channels_run_id)
        source_aux_data_path = extra_info["path"] if extra_info is not None else ""
        self._set_attrs(
            run_group,
            {
                "output_id": self.sum_channels_run_id,
                "output_type": "image_tool",
                "tool_name": SUM_CHANNELS_TOOL_NAME,
                "created_utc": self._utc_now(),
                "software_name": "brighteyes_mcs_file",
                "software_version": self._package_version(),
                "algorithm_name": "sum_along_channel_axis",
                "algorithm_version": "0.1.0",
                "source_data_path": primary_info["path"],
                "source_aux_data_path": source_aux_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_aux_calibration_path": extra_calibration_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "source_timing_metadata_path": metadata_attrs["source_timing_metadata_path"],
                "source_axes_path": metadata_attrs["source_axes_path"],
                "input_axis_order": primary_info["source_axis_order"],
                "output_axis_order": primary_info["output_axis_order"],
                "output_data_path": self._product_path(self.sum_channels_run_id, "spad"),
                "time_axis_source": metadata_attrs["source_time_axis_path"],
                "time_axis_path": self._time_axis_path(self.sum_channels_run_id),
                "channel_axis_source": f"{primary_info['path']} final axis",
                "parameter_encoding": "attrs_and_json",
            },
        )

        inputs_group = run_group.create_group("inputs")
        self._set_attrs(
            inputs_group,
            {
                "source_data_path": primary_info["path"],
                "source_aux_data_path": source_aux_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_aux_calibration_path": extra_calibration_path,
                "source_paths_json": json.dumps(
                    [path for path in (primary_info["path"], source_aux_data_path) if path]
                ),
                "input_axis_order": primary_info["source_axis_order"],
                "selected_channels_json": json.dumps(channels),
                "selected_aux_channels_json": json.dumps(extra_channels or []),
                "selected_time_bins_json": metadata_attrs["selected_time_bins_json"],
                "mask_source_path": "",
            },
        )

        metadata_group = run_group.create_group("metadata")
        self._set_attrs(metadata_group, metadata_attrs)

        parameters_group = run_group.create_group("parameters")
        self._set_attrs(
            parameters_group,
            {
                "parameters_json": json.dumps(
                    {
                        "channels": channels,
                        "aux_channels": extra_channels or [],
                        "include_aux_sum": self.include_extra_sum,
                    }
                ),
                "tool_name": SUM_CHANNELS_TOOL_NAME,
                "tool_mode": "channel_sum",
                "channel_selection_json": json.dumps(channels),
                "time_bin_selection_json": metadata_attrs["selected_time_bins_json"],
                "spatial_binning": "none",
                "normalize_output": False,
                "use_calibration": metadata_attrs["use_calibration"],
            },
        )

        self._create_axes(run_group, metadata_attrs)
        products_group = run_group.create_group("products")
        self._create_sum_channels_product(
            products_group,
            "spad",
            primary_info,
            channels,
            metadata_attrs,
            primary_calibration_path,
            "channel-summed SPAD digital detector counts",
            (
                "SPAD sum_channels_without_corrections output produced by summing "
                "selected detector channels."
            ),
        )

        if self.include_extra_sum and extra_info is not None:
            self._create_sum_channels_product(
                products_group,
                "aux",
                extra_info,
                extra_channels,
                metadata_attrs,
                extra_calibration_path,
                "channel-summed auxiliary digital FIFO counts",
                (
                    "Auxiliary sum_channels_without_corrections output produced by "
                    "summing selected aux digital channels."
                ),
            )

        self._create_common_run_groups(run_group)

    def _create_sum_channels_with_skew_correction_run(
        self,
        handle,
        output_group,
        primary_info,
        extra_info,
    ):
        metadata = self._read_mcs_metadata(self.data_path)
        channels = self._resolve_channels(self.channels, primary_info["channel_count"], "channels")
        primary_shifts, primary_skew_path, primary_skew_error = self._resolve_channel_skew(
            handle,
            primary_info,
            channels,
        )
        if primary_shifts is None:
            message = (
                f"{SUM_CHANNELS_WITH_SKEW_CORRECTION_AGGREGATION} skipped: "
                f"{primary_skew_error}"
            )
            if self.require_shifted_sum:
                raise ValueError(message)
            output_group.attrs["sum_channels_with_skew_correction_skipped"] = True
            output_group.attrs["sum_channels_with_skew_correction_skip_reason"] = message
            return False

        metadata_attrs = self._build_metadata_attrs(
            handle,
            primary_info,
            channels,
            metadata,
            fallback_time_axis_run_id=self.sum_channels_with_skew_correction_run_id,
        )
        metadata_attrs["channel_aggregation"] = SUM_CHANNELS_WITH_SKEW_CORRECTION_AGGREGATION

        extra_channels = None
        extra_shifts = None
        extra_skew_path = ""
        extra_skew_error = ""
        if extra_info is not None:
            extra_channels = self._resolve_channels(
                self.extra_channels,
                extra_info["channel_count"],
                "aux_channels",
            )
            extra_shifts, extra_skew_path, extra_skew_error = self._resolve_channel_skew(
                handle,
                extra_info,
                extra_channels,
            )

        primary_calibration_path = self._calibration_path_for_source(handle, primary_info)
        extra_calibration_path = self._calibration_path_for_source(handle, extra_info)
        source_aux_data_path = extra_info["path"] if extra_info is not None else ""

        run_group = output_group.create_group(self.sum_channels_with_skew_correction_run_id)
        self._set_attrs(
            run_group,
            {
                "output_id": self.sum_channels_with_skew_correction_run_id,
                "output_type": "image_tool",
                "tool_name": SUM_CHANNELS_WITH_SKEW_CORRECTION_TOOL_NAME,
                "created_utc": self._utc_now(),
                "software_name": "brighteyes_mcs_file",
                "software_version": self._package_version(),
                "algorithm_name": "sum_channel_applying_shifts",
                "algorithm_version": "0.1.0",
                "source_data_path": primary_info["path"],
                "source_aux_data_path": source_aux_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_aux_calibration_path": extra_calibration_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "source_timing_metadata_path": metadata_attrs["source_timing_metadata_path"],
                "source_axes_path": metadata_attrs["source_axes_path"],
                "input_axis_order": primary_info["source_axis_order"],
                "output_axis_order": primary_info["output_axis_order"],
                "output_data_path": self._product_path(
                    self.sum_channels_with_skew_correction_run_id,
                    "spad",
                ),
                "time_axis_source": metadata_attrs["source_time_axis_path"],
                "time_axis_path": self._time_axis_path(
                    self.sum_channels_with_skew_correction_run_id
                ),
                "channel_axis_source": f"{primary_info['path']} final axis",
                "channel_skew_path": primary_skew_path,
                "parameter_encoding": "attrs_and_json",
            },
        )

        inputs_group = run_group.create_group("inputs")
        self._set_attrs(
            inputs_group,
            {
                "source_data_path": primary_info["path"],
                "source_aux_data_path": source_aux_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_aux_calibration_path": extra_calibration_path,
                "source_paths_json": json.dumps(
                    [path for path in (primary_info["path"], source_aux_data_path) if path]
                ),
                "input_axis_order": primary_info["source_axis_order"],
                "selected_channels_json": json.dumps(channels),
                "selected_aux_channels_json": json.dumps(extra_channels or []),
                "selected_time_bins_json": metadata_attrs["selected_time_bins_json"],
                "channel_skew_path": primary_skew_path,
                "aux_channel_skew_path": extra_skew_path,
                "mask_source_path": "",
            },
        )

        metadata_group = run_group.create_group("metadata")
        self._set_attrs(metadata_group, metadata_attrs)

        parameters_group = run_group.create_group("parameters")
        self._set_attrs(
            parameters_group,
            {
                "parameters_json": json.dumps(
                    {
                        "channels": channels,
                        "aux_channels": extra_channels or [],
                        "include_aux_shifted_sum": self.include_extra_shifted_sum,
                        "reverse_shifts": self.shifted_sum_reverse_shifts,
                        "backend": self.shifted_sum_backend,
                        "chunk_size": self.shifted_sum_chunk_size,
                    }
                ),
                "tool_name": SUM_CHANNELS_WITH_SKEW_CORRECTION_TOOL_NAME,
                "tool_mode": "channel_sum_with_skew_correction",
                "channel_selection_json": json.dumps(channels),
                "time_bin_selection_json": metadata_attrs["selected_time_bins_json"],
                "spatial_binning": "none",
                "normalize_output": False,
                "use_calibration": True,
                "channel_skew_path": primary_skew_path,
                "reverse_shifts": self.shifted_sum_reverse_shifts,
                "backend": self.shifted_sum_backend,
            },
        )

        self._create_axes(run_group, metadata_attrs)
        products_group = run_group.create_group("products")
        self._create_sum_channels_with_skew_correction_product(
            products_group,
            "spad",
            primary_info,
            channels,
            primary_shifts,
            primary_skew_path,
            metadata_attrs,
            primary_calibration_path,
            "channel-skew-corrected sum of SPAD digital detector counts",
            (
                "SPAD sum_channels_with_skew_correction output produced with "
                "Alignment.sum_channel_applying_shifts(data, channel_skew)."
            ),
        )
        self._create_common_shifted_calibration_trace(
            output_group,
            handle,
            DEFAULT_SUM_IRF_TRACE_ID,
            SUM_IRF_TRACE_OUTPUT_KIND,
            primary_calibration_path,
            "irf_trace",
            channels,
            primary_shifts,
            primary_skew_path,
            metadata_attrs,
            "channel-skew-corrected summed IRF trace",
            (
                "Common IRF trace produced from aligned/irf_trace using "
                "Alignment.sum_channel_applying_shifts(trace, channel_skew)."
            ),
        )
        self._create_common_shifted_calibration_trace(
            output_group,
            handle,
            DEFAULT_SUM_REFERENCE_TRACE_ID,
            SUM_REFERENCE_TRACE_OUTPUT_KIND,
            primary_calibration_path,
            "reference_trace",
            channels,
            primary_shifts,
            primary_skew_path,
            metadata_attrs,
            "channel-skew-corrected summed reference trace",
            (
                "Common reference trace produced from aligned/reference_trace "
                "using Alignment.sum_channel_applying_shifts(trace, channel_skew)."
            ),
        )

        if self.include_extra_shifted_sum and extra_info is not None:
            if extra_shifts is not None:
                self._create_sum_channels_with_skew_correction_product(
                    products_group,
                    "aux",
                    extra_info,
                    extra_channels,
                    extra_shifts,
                    extra_skew_path,
                    metadata_attrs,
                    extra_calibration_path,
                    "channel-skew-corrected sum of auxiliary digital FIFO counts",
                    (
                        "Auxiliary sum_channels_with_skew_correction output produced with "
                        "Alignment.sum_channel_applying_shifts(data, channel_skew)."
                    ),
                )
            else:
                run_group.attrs["aux_sum_channels_with_skew_correction_skipped"] = True
                run_group.attrs["aux_sum_channels_with_skew_correction_skip_reason"] = extra_skew_error

        self._create_common_run_groups(run_group)
        skew_group = run_group.create_group("intermediates/channel_skew")
        skew_group.create_dataset("spad", data=np.asarray(primary_shifts, dtype=np.float64))
        if extra_shifts is not None:
            skew_group.create_dataset("aux", data=np.asarray(extra_shifts, dtype=np.float64))
        return True

    def build(self):
        """Create or replace the ``/output`` group and return the output path."""
        output_path = self._prepare_output_file()
        with h5py.File(output_path, "a") as handle:
            root_attrs = {
                "output_path": f"/{self.output_key}",
                "contains_output": True,
            }
            if (
                "raw" in handle
                and isinstance(handle["raw"], h5py.Group)
                and "spad" in handle["raw"]
            ):
                root_attrs.update(
                    {
                        "data_format_version": BRIGHTEYES_H5_DATA_FORMAT_VERSION,
                        "schema_name": BRIGHTEYES_H5_SCHEMA_NAME,
                        "schema_variant": BRIGHTEYES_H5_SCHEMA_VARIANT,
                    }
                )
            self._set_attrs(handle, root_attrs)
            primary_info, extra_info = self._resolve_source_infos(handle)

            if self.output_key in handle:
                if not self.overwrite:
                    raise FileExistsError(f"/{self.output_key} already exists in {output_path}")
                del handle[self.output_key]
            output_group = ensure_output_group(handle, self.output_key)

            if self.create_virtual_channels:
                analog_info = self._resolve_analog_source_info(handle)
                self._create_virtual_channels(output_group, primary_info, extra_info, analog_info)
            default_run = ""
            if self.create_sum_channels:
                self._create_sum_channels_run(handle, output_group, primary_info, extra_info)
                default_run = self.sum_channels_run_id
            if self.create_sum_channels_with_skew_correction:
                shifted_created = self._create_sum_channels_with_skew_correction_run(
                    handle,
                    output_group,
                    primary_info,
                    extra_info,
                )
                if shifted_created:
                    default_run = self.sum_channels_with_skew_correction_run_id

            output_group.attrs["default"] = default_run

        return str(output_path)


def build_h5_output(data_path, output_path=None, **kwargs):
    """
    Build the BrightEyes ``/output`` group for an HDF5 file.

    When ``output_path`` is ``None``, the input file is updated in place. Pass
    an explicit ``output_path`` to preserve the input file and write a copied
    output file.

    Parameters are forwarded to :class:`H5OutputBuilder`.
    """

    return H5OutputBuilder(data_path, output_path=output_path, **kwargs).build()


def add_output_to_h5_file(data_path, output_path=None, **kwargs):
    """Compatibility alias for :func:`build_h5_output`."""

    return build_h5_output(data_path, output_path=output_path, **kwargs)


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
    parameter_names=None,
    param_names=None,
    model_kwargs=None,
    amplitude_param="C",
    delay_param="dT",
    lifetime_param="tau",
    create_output=True,
    output_options=None,
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
        :meth:`Alignment.clean_irf_stack` to the aligned IRF stack using the
        historical notebook settings before it is rescaled for output.
    irf_corrections_type : {"median", "single_ch"}, default ``"median"``
        Strategy used to choose the delay applied to the aligned IRF/reference
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
        input vector is also written to ``timing/channel_skew_external``.
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
    model_fn, p0, bounds, parameter_names, model_kwargs : optional
        Optional custom full-model fit configuration. ``model_fn`` receives
        ``(t, irf, period, *params)`` and returns the fitted histogram.
    create_output : bool, default ``True``
        If ``True``, build the ``/output`` analysis group in the calibrated
        file after calibration is complete.
    output_options : dict or None, default ``None``
        Optional keyword arguments forwarded to :class:`H5OutputBuilder`. The
        output builder always writes in place to the calibrated file; any
        nested ``output_path`` option is ignored.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`H5DataCalibrator`, including ``output_path=None``,
        ``channels=None``, ``calibration_key="calibration"``,
        ``period_ns=None``, ``initial_tau=None``, ``initial_dT=None``,
        ``initial_C=None``, ``force_C_normalized=False``, and ``eps=1e-8``.

    Returns
    -------
    str
        Path to the calibrated output HDF5 file. Each calibrated product is
        written under ``<calibration_key>/results/<product>/``.
    """
    if param_names is not None:
        if parameter_names is not None and list(parameter_names) != list(param_names):
            raise ValueError("parameter_names and param_names cannot disagree")
        parameter_names = param_names

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
        parameter_names=parameter_names,
        model_kwargs=model_kwargs,
        amplitude_param=amplitude_param,
        delay_param=delay_param,
        lifetime_param=lifetime_param,
        create_output=create_output,
        output_options=output_options,
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


def show_h5_structure(
    file_path,
    *,
    include_attrs: bool = True,
    attrs_inline: bool = False,
    tree_chars: bool = True,
    max_depth: Optional[int] = None,
    max_attr_len: int = 120,
) -> str:
    """
    Return and print a readable tree view of an HDF5 file structure.

    Parameters
    ----------
    file_path : str or path-like
        HDF5 file to inspect.
    include_attrs : bool, default True
        Include group and dataset attributes in the output.
    attrs_inline : bool, default False
        Print attributes on one line below each node (``node.attrs: k=v, …``).
        When *False*, each attribute is printed on its own line as ``@key = value``.
    tree_chars : bool, default True
        Use ``├──`` / ``└──`` connectors for a classic tree look.
        When *False*, use plain indentation (easier to copy-paste).
    max_depth : int or None, default None
        Maximum nesting depth to visit (root = 0).  ``None`` means unlimited.
    max_attr_len : int, default 120
        Truncate attribute value reprs that exceed this many characters.
    """
    lines: list[str] = []

    def _truncate(value) -> str:
        r = repr(value)
        if len(r) > max_attr_len:
            r = r[: max_attr_len - 3] + "..."
        return r

    def _append_attrs(node, indent: str, node_name: str = "") -> None:
        if not include_attrs:
            return
        items = list(node.attrs.items())
        if not items:
            return
        if attrs_inline:
            prefix = f"{node_name}.attrs" if node_name else ".attrs"
            joined = ", ".join(f"{k}={_truncate(v)}" for k, v in items)
            lines.append(f"{indent}{prefix}: {joined}")
        else:
            for k, v in items:
                lines.append(f"{indent}@{k} = {_truncate(v)}")

    def _visit(name: str, obj, siblings_remaining: bool = False) -> None:
        depth = name.count("/") if name else 0
        if max_depth is not None and depth > max_depth:
            return

        if name == "":
            lines.append("/")
            _append_attrs(obj, "  ")
            return

        node_name = name.split("/")[-1]
        base_indent = "  " * depth

        if tree_chars:
            connector = "├── " if siblings_remaining else "└── "
            label_indent = base_indent + connector
            child_indent = base_indent + ("│   " if siblings_remaining else "    ")
        else:
            label_indent = base_indent
            child_indent = base_indent + "  "

        if isinstance(obj, h5py.Dataset):
            lines.append(
                f"{label_indent}{node_name}  "
                f"shape={obj.shape}  dtype={obj.dtype}"
            )
            _append_attrs(obj, child_indent, node_name=node_name)

        elif isinstance(obj, h5py.Group):
            lines.append(f"{label_indent}{node_name}/")
            _append_attrs(obj, child_indent, node_name=node_name)
            # recurse into children
            if max_depth is None or depth < max_depth:
                keys = list(obj.keys())
                for i, child_key in enumerate(keys):
                    _visit(
                        f"{name}/{child_key}" if name else child_key,
                        obj[child_key],
                        siblings_remaining=(i < len(keys) - 1),
                    )

    with h5py.File(file_path, "r") as fh:
        _visit("", fh)
        # top-level children
        keys = list(fh.keys())
        for i, key in enumerate(keys):
            _visit(key, fh[key], siblings_remaining=(i < len(keys) - 1))

    structure = "\n".join(lines)
    print(structure)
    return structure


# ──────────────────────────────────────────────────────────────────────────────
# HTML viewer (Jupyter / JupyterLab)
# ──────────────────────────────────────────────────────────────────────────────

# A unique prefix keeps multiple widgets on the same notebook page independent.
_WIDGET_COUNTER = 0


def show_h5_structure_html(
    file_path,
    *,
    include_attrs: bool = True,
    attrs_inline: bool = True,
    show_full_path: bool = True,
    max_attr_len: int = 200,
    display_output: bool = True,
) -> str:
    """
    Return an interactive HTML tree view of an HDF5 file structure.

    All groups are **collapsed by default**; click a group label to expand it.
    Level-buttons on each group let you open/close *all siblings at the same
    depth* in one click.  Global **Expand all** / **Collapse all** buttons sit
    at the top of the widget.

    Parameters
    ----------
    file_path : str or path-like
        HDF5 file to inspect.
    include_attrs : bool, default True
        Include group and dataset attributes.
    attrs_inline : bool, default True
        Render all attributes for a node on one line.
    show_full_path : bool, default False
        If *True*, render the full HDF5 path for each group and dataset.
    max_attr_len : int, default 200
        Truncate attribute value reprs longer than this.
    display_output : bool, default True
        If *True* and IPython is available, display the HTML immediately.
    """
    global _WIDGET_COUNTER
    _WIDGET_COUNTER += 1
    uid = f"h5tree_{_WIDGET_COUNTER}"

    # ── helpers ────────────────────────────────────────────────────────────────

    def _fmt_value(value) -> str:
        r = repr(value)
        if len(r) > max_attr_len:
            r = r[: max_attr_len - 3] + "..."
        r_esc = escape(r)
        if isinstance(value, np.ndarray):
            return (
                f"{r_esc} "
                f"<span class='h5-array-shape'>shape={escape(str(value.shape))}</span>"
            )
        return r_esc

    def _render_attrs(node, node_name: str) -> str:
        if not include_attrs or len(node.attrs) == 0:
            return ""
        items = list(node.attrs.items())
        if attrs_inline:
            pairs = ", ".join(
                f"<span class='h5-attr-key'>{escape(str(k))}</span>"
                f"=<span class='h5-attr-value'>{_fmt_value(v)}</span>"
                for k, v in items
            )
            return (
                f"<div class='h5-attrs-inline'>"
                f"<span class='h5-attrs-prefix'>"
                f"<span class='h5-node-ref'>{escape(node_name)}.attrs</span>: "
                f"</span>"
                f"<span class='h5-attrs-content'>{pairs}</span>"
                f"</div>"
            )
        rows = "".join(
            f"<li>"
            f"<span class='h5-node-ref'>{escape(node_name)}.attrs.</span>"
            f"<span class='h5-attr-key'>{escape(str(k))}</span>"
            f" = <span class='h5-attr-value'>{_fmt_value(v)}</span>"
            f"</li>"
            for k, v in items
        )
        return f"<ul class='h5-attrs-list'>{rows}</ul>"

    def _render_full_path(path: str) -> str:
        if not show_full_path:
            return ""
        full_path = "/" if not path else path if path.startswith("/") else f"/{path}"
        return f"<span class='h5-full-path'>{escape(full_path)}</span>"

    def _render_node(name: str, obj, depth: int) -> str:
        node_name = name.split("/")[-1] if name else "/"
        full_path_html = _render_full_path(name)

        if isinstance(obj, h5py.Dataset):
            attrs_html = _render_attrs(obj, node_name)
            return (
                f"<li data-depth='{depth}' class='h5-li-dataset'>"
                f"<div class='h5-node-line'>"
                f"<span class='h5-node-main'>"
                f"<span class='h5-dataset'>{escape(node_name)}</span> "
                f"<span class='h5-meta'>shape={escape(str(obj.shape))} "
                f"dtype={escape(str(obj.dtype))}</span>"
                f"</span>"
                f"{full_path_html}"
                f"</div>"
                f"{attrs_html}"
                f"</li>"
            )

        # Group
        attrs_html = _render_attrs(obj, node_name)
        children_html = "".join(
            _render_node(
                f"{name}/{child_key}" if name else child_key,
                obj[child_key],
                depth + 1,
            )
            for child_key in obj.keys()
        )
        children_block = f"<ul data-depth='{depth + 1}'>{children_html}</ul>" if children_html else ""

        # Level-control buttons (expand / collapse siblings at same depth)
        level_btns = (
            f"<span class='h5-level-btns' data-depth='{depth}' data-widget='{uid}'>"
            f"<button class='h5-lvl-btn' "
            f"onclick=\"h5LevelToggle('{uid}', {depth}, true, this)\""
            f" title='Espandi tutti i gruppi di questo livello'>▸</button>"
            f"<button class='h5-lvl-btn' "
            f"onclick=\"h5LevelToggle('{uid}', {depth}, false, this)\""
            f" title='Comprimi tutti i gruppi di questo livello'>▾</button>"
            f"</span>"
        )

        return (
            f"<li data-depth='{depth}' class='h5-li-group'>"
            f"<details class='h5-branch'>"  # NOT open → collapsed by default
            f"<summary>"
            f"<span class='h5-node-line'>"
            f"<span class='h5-node-main'>"
            f"<span class='h5-group'>{escape(node_name)}</span>/"
            f"{level_btns}"
            f"</span>"
            f"{full_path_html}"
            f"</span>"
            f"</summary>"
            f"{attrs_html}"
            f"{children_block}"
            f"</details>"
            f"</li>"
        )

    # ── build tree ─────────────────────────────────────────────────────────────
    with h5py.File(file_path, "r") as fh:
        root_attrs_html = _render_attrs(fh, "/")
        children_html = "".join(
            _render_node(key, fh[key], depth=1) for key in fh.keys()
        )

    js = textwrap.dedent(f"""
    <script>
    (function () {{
      // Expand / collapse ALL groups in the widget
      function h5All(widgetId, open) {{
        var root = document.getElementById(widgetId);
        if (!root) return;
        root.querySelectorAll('details.h5-branch').forEach(function(d) {{
          d.open = open;
        }});
      }}

      // Expand / collapse all groups at a specific depth inside a widget
      function h5LevelToggle(widgetId, depth, open, btn) {{
        // Stop the click from toggling the parent <details>
        if (btn) {{ btn.closest('details') && (event || window.event) && (event || window.event).stopPropagation(); }}
        var root = document.getElementById(widgetId);
        if (!root) return;
        root.querySelectorAll('li.h5-li-group[data-depth="' + depth + '"]').forEach(function(li) {{
          var det = li.querySelector(':scope > details.h5-branch');
          if (det) det.open = open;
        }});
      }}

      // Expose globally so onclick= attributes work
      window.h5All = h5All;
      window.h5LevelToggle = h5LevelToggle;
    }})();
    </script>
    """).strip()

    css = textwrap.dedent("""
    <style>
    .h5-tree {
      color-scheme: light dark;
      font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
      font-size: 13px;
      line-height: 1.6;
      --h5-fg:         #1f2937;
      --h5-muted:      #6b7280;
      --h5-border:     #d1d5db;
      --h5-group:      #0f766e;
      --h5-dataset:    #1d4ed8;
      --h5-attrs:      #7c2d12;
      --h5-node-ref:   #7c3aed;
      --h5-attr-key:   #b45309;
      --h5-attr-value: #374151;
      --h5-root:       #111827;
      --h5-btn-bg:     #f3f4f6;
      --h5-btn-border: #d1d5db;
      --h5-btn-fg:     #374151;
    }
    @media (prefers-color-scheme: dark) {
      .h5-tree {
        --h5-fg:         #e5e7eb;
        --h5-muted:      #9ca3af;
        --h5-border:     #4b5563;
        --h5-group:      #5eead4;
        --h5-dataset:    #93c5fd;
        --h5-attrs:      #fdba74;
        --h5-node-ref:   #c4b5fd;
        --h5-attr-key:   #fbbf24;
        --h5-attr-value: #f3f4f6;
        --h5-root:       #f9fafb;
        --h5-btn-bg:     #374151;
        --h5-btn-border: #6b7280;
        --h5-btn-fg:     #e5e7eb;
      }
    }
    .h5-tree { color: var(--h5-fg); }
    .h5-tree ul {
      list-style: none;
      margin: 0.15rem 0 0.15rem 1rem;
      padding-left: 1rem;
      border-left: 1px solid var(--h5-border);
    }
    /* root-level list: no left border */
    .h5-tree > ul {
      margin-left: 0;
      border-left: none;
    }
    .h5-tree li { margin: 0.2rem 0; }

    /* summary spans the available width; inner row handles alignment */
    .h5-tree summary {
      cursor: pointer;
      list-style: none;
      display: block;
    }
    .h5-tree summary::-webkit-details-marker { display: none; }

    /* collapse/expand triangle on the <details> element */
    .h5-branch > summary::before {
      content: "▸";
      color: var(--h5-muted);
      width: 0.9rem;
      display: inline-block;
    }
    .h5-branch[open] > summary::before { content: "▾"; }

    .h5-group   { color: var(--h5-group);   font-weight: 700; }
    .h5-dataset { color: var(--h5-dataset); font-weight: 700; }
    .h5-meta    { color: var(--h5-muted);   font-weight: 400; }
    .h5-node-line {
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      width: 100%;
      min-width: 0;
    }
    .h5-node-main {
      display: inline-flex;
      align-items: baseline;
      gap: 0.35rem;
      min-width: 0;
      flex: 1 1 auto;
    }
    .h5-full-path {
      color: var(--h5-muted);
      font-weight: 400;
      margin-left: auto;
      font-size: 0.95em;
      text-align: right;
      white-space: nowrap;
      flex: 0 0 auto;
    }

    /* attributes */
    .h5-attrs-inline, .h5-attrs-list {
      color: var(--h5-attrs);
      margin-top: 0.1rem;
      margin-left: 1.1rem;
    }
    .h5-attrs-inline {
      display: grid;
      grid-template-columns: max-content 1fr;
      column-gap: 0.4rem;
    }
    .h5-attrs-prefix  { white-space: nowrap; }
    .h5-attrs-content { min-width: 0; overflow-wrap: anywhere; }
    .h5-node-ref      { color: var(--h5-node-ref); font-weight: 700; }
    .h5-attr-key      { color: var(--h5-attr-key);  font-weight: 700; }
    .h5-attr-value    { color: var(--h5-attr-value); }
    .h5-array-shape   { color: var(--h5-muted); }

    /* root label */
    .h5-root { color: var(--h5-root); font-weight: 800; }

    /* global toolbar */
    .h5-toolbar {
      margin-bottom: 0.5rem;
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
    }

    /* shared button style */
    .h5-btn, .h5-lvl-btn {
      font-family: inherit;
      font-size: 11px;
      padding: 1px 7px;
      border-radius: 4px;
      border: 1px solid var(--h5-btn-border);
      background: var(--h5-btn-bg);
      color: var(--h5-btn-fg);
      cursor: pointer;
      line-height: 1.6;
    }
    .h5-btn:hover, .h5-lvl-btn:hover { opacity: 0.75; }

    /* level buttons sit inline next to group label; don't inherit summary colour */
    .h5-level-btns {
      display: inline-flex;
      gap: 3px;
      vertical-align: middle;
      flex: 0 0 auto;
    }
    /* prevent level-btn click from toggling the parent details */
    .h5-level-btns button { pointer-events: all; }
    </style>
    """).strip()

    toolbar = (
        f"<div class='h5-toolbar'>"
        f"<button class='h5-btn' onclick=\"h5All('{uid}', true)\">⊞ Expand all</button>"
        f"<button class='h5-btn' onclick=\"h5All('{uid}', false)\">⊟ Collapse all</button>"
        f"</div>"
    )

    html_output = (
        f"{js}\n"
        f"<div class='h5-tree' id='{uid}'>\n"
        f"{css}\n"
        f"{toolbar}"
        f"<div class='h5-root'>/</div>\n"
        f"{root_attrs_html}\n"
        f"<ul data-depth='1'>\n"
        f"{children_html}\n"
        f"</ul>\n"
        f"</div>"
    )

    if display_output:
        try:
            from IPython.display import HTML, display as ipy_display
            ipy_display(HTML(html_output))
        except ImportError:
            pass

    return html_output

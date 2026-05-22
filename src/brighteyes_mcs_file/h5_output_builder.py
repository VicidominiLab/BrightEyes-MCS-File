"""HDF5 /output group builders."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from .alignment import Alignment
from . import mcs

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    PackageNotFoundError = Exception
    version = None

__all__ = [
    "H5OutputBuilder",
    "add_output_to_h5_file",
    "build_h5_output",
]

DEFAULT_OUTPUT_KEY = "output"
DEFAULT_SUM_RUN_ID = "sum_001"
DEFAULT_SUM_USING_SHIFT_RUN_ID = "sum_using_shift_001"
DEFAULT_PRIMARY_DATA_KEYS = ("data/primary", "data")
DEFAULT_EXTRA_DATA_KEYS = ("data/extra", "data_channels_extra")
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

    - ``/output/virtual_channels``: HDF5 virtual datasets named
      ``data_channel_<i>`` and ``data_extra_channel_<i>`` that point to a
      single source channel without copying raw data.
    - ``/output/<sum_run_id>/products/image``: the "Sum" output, computed by
      summing the primary data along the final channel axis. When extra digital
      data are present and ``include_extra_sum=True``, the extra channels are
      also summed into ``products/image_extra``.
    - ``/output/<sum_using_shift_run_id>/products/image``: the
      "Sum_using_shift" output, computed with
      :meth:`Alignment.sum_channel_applying_shifts` and the stored calibration
      ``channel_skew`` vector.

    Source datasets must use the BrightEyes channel-last layout:
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
        primary_data_key=None,
        extra_data_key=None,
        create_virtual_channels=True,
        create_sum=True,
        create_sum_using_shift=True,
        create_sum_shifted=None,
        sum_run_id=DEFAULT_SUM_RUN_ID,
        sum_using_shift_run_id=DEFAULT_SUM_USING_SHIFT_RUN_ID,
        channels=None,
        extra_channels=None,
        include_extra_sum=True,
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
        self.primary_data_key = primary_data_key
        self.extra_data_key = extra_data_key
        self.create_virtual_channels = bool(create_virtual_channels)
        self.create_sum = bool(create_sum)
        if create_sum_shifted is not None:
            create_sum_using_shift = create_sum_shifted
        self.create_sum_using_shift = bool(create_sum_using_shift)
        self.sum_run_id = str(sum_run_id)
        self.sum_using_shift_run_id = str(sum_using_shift_run_id)
        self.channels = channels
        self.extra_channels = extra_channels
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

    def _default_output_path(self):
        suffix = self.data_path.suffix or ".h5"
        return self.data_path.with_name(f"{self.data_path.stem}_output{suffix}")

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
                return H5OutputBuilder._prepare_attr_value(value.item())
            if value.dtype.kind in {"i", "u", "f", "b"}:
                return value
            return json.dumps(value.tolist(), default=str)
        if isinstance(value, (list, tuple, dict, set)):
            return json.dumps(value, default=str)
        return str(value)

    @classmethod
    def _set_attrs(cls, node, attrs):
        for key, value in attrs.items():
            node.attrs[str(key)] = cls._prepare_attr_value(value)

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
            key = str(candidate).strip("/")
            if key not in handle:
                continue
            obj = handle[key]
            if isinstance(obj, h5py.Dataset):
                return obj
            if explicit_key is not None:
                raise TypeError(f"{label} key {candidate!r} exists but is not a dataset")
        if required:
            tried = ", ".join(str(candidate) for candidate in candidates)
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
    def _time_axis(output_timebins, timebin_in_ns):
        if np.isfinite(timebin_in_ns):
            return np.arange(output_timebins, dtype=np.float64) * float(timebin_in_ns), "ns"
        return np.arange(output_timebins, dtype=np.float64), "bin"

    @staticmethod
    def _axis_values(count, spacing, units):
        if np.isfinite(spacing):
            return np.arange(count, dtype=np.float64) * float(spacing), units
        return np.arange(count, dtype=np.float64), "index"

    def _source_info(self, dataset, kind):
        self._validate_channel_last_dataset(dataset, kind)
        axis_info = AXIS_ORDERS[dataset.ndim]
        return {
            "dataset": dataset,
            "kind": kind,
            "path": dataset.name,
            "shape": tuple(dataset.shape),
            "channel_count": int(dataset.shape[-1]),
            "timebins": int(dataset.shape[-2]),
            "source_axis_order": axis_info["source"],
            "output_axis_order": axis_info["virtual"],
            "data_layout": axis_info["layout"],
            "subpixel_scan_mode": axis_info["subpixel_scan_mode"],
        }

    @staticmethod
    def _calibration_path_for_source(handle, source_info):
        if source_info is None:
            return ""
        source_path = source_info["path"].strip("/")
        candidates = [
            f"/calibration/{source_path}",
            f"/calibration/results/{source_info['kind']}",
        ]
        if source_info["kind"] == "primary":
            candidates.extend(("/calibration/data", "/calibration/results/primary"))
        elif source_info["kind"] == "extra":
            candidates.extend(("/calibration/data_channels_extra", "/calibration/results/extra"))

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
                "channel_skew",
                "delay/channel_skew_bins",
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
                "channel_index",
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

    def _resolve_source_infos(self, handle):
        primary_dataset = self._find_dataset(
            handle,
            self.primary_data_key,
            DEFAULT_PRIMARY_DATA_KEYS,
            "primary data",
            required=self.create_sum or self.create_sum_using_shift,
        )
        extra_dataset = self._find_dataset(
            handle,
            self.extra_data_key,
            DEFAULT_EXTRA_DATA_KEYS,
            "extra data",
            required=False,
        )

        primary_info = self._source_info(primary_dataset, "primary") if primary_dataset is not None else None
        extra_info = self._source_info(extra_dataset, "extra") if extra_dataset is not None else None
        if primary_info is not None and extra_info is not None:
            if primary_info["shape"][:-1] != extra_info["shape"][:-1]:
                raise ValueError(
                    "primary and extra data must share all non-channel dimensions "
                    f"(got {primary_info['shape']} and {extra_info['shape']})"
                )
        return primary_info, extra_info

    def _timing_attr_source(self, handle, primary_info):
        metadata_timing_path = "/metadata/acquisition/timing"
        if metadata_timing_path.strip("/") in handle:
            return metadata_timing_path, handle[metadata_timing_path].attrs

        calibration_path = self._calibration_path_for_source(handle, primary_info)
        if calibration_path:
            return calibration_path, handle[calibration_path].attrs

        return "", {}

    def _build_metadata_attrs(self, handle, primary_info, channels, metadata, fallback_time_axis_run_id=None):
        source_shape = primary_info["shape"]
        nrep, nz, ny, nx = (int(source_shape[i]) for i in range(4))
        output_timebins = int(source_shape[-2])
        source_metadata_path = self._get_existing_path(
            handle,
            ("/metadata", "/calibration/input_data_metadata"),
        )
        source_axes_path = self._get_existing_path(handle, ("/axes",))
        source_time_axis_path = self._get_existing_path(handle, ("/axes/digital_time_ns",))
        source_timing_metadata_path, timing_attrs = self._timing_attr_source(handle, primary_info)

        laser_frequency_mhz = self._safe_float(
            timing_attrs.get("laser_freq_mhz", timing_attrs.get("laser_freq_in_mhz", np.nan))
        )
        if not np.isfinite(laser_frequency_mhz):
            laser_frequency_mhz = self._safe_float(self._metadata_get(metadata, "dfd_freq", np.nan))

        laser_period_ns = self._safe_float(
            timing_attrs.get("laser_period_ns", timing_attrs.get("laser_period_in_ns", np.nan))
        )
        if not np.isfinite(laser_period_ns) and np.isfinite(laser_frequency_mhz) and laser_frequency_mhz > 0:
            laser_period_ns = 1000.0 / laser_frequency_mhz

        timebin_in_ns = self._safe_float(
            timing_attrs.get("digital_time_bin_in_ns", timing_attrs.get("bin_width_in_ns", np.nan))
        )
        if not np.isfinite(timebin_in_ns) and np.isfinite(laser_period_ns):
            timebin_in_ns = laser_period_ns / output_timebins

        dfd_active = bool(self._metadata_get(metadata, "dfd_activate", False))
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

        pixel_size_x_um = range_x_um / nx if np.isfinite(range_x_um) and nx > 0 else np.nan
        pixel_size_y_um = range_y_um / ny if np.isfinite(range_y_um) and ny > 0 else np.nan
        pixel_size_z_um = range_z_um / nz if np.isfinite(range_z_um) and nz > 0 else np.nan

        pixel_dwell_time_us = self._safe_float(timing_attrs.get("pixel_dwell_time_us", np.nan))
        if not np.isfinite(pixel_dwell_time_us):
            dt_us = self._safe_float(self._metadata_get(metadata, "dt", np.nan))
            nbin = self._safe_float(self._metadata_get(metadata, "nbin", np.nan))
            if np.isfinite(dt_us) and np.isfinite(nbin):
                pixel_dwell_time_us = dt_us * nbin
        pixel_dwell_time_in_ns = pixel_dwell_time_us * 1000.0 if np.isfinite(pixel_dwell_time_us) else np.nan
        line_time_s = pixel_dwell_time_us * nx / 1e6 if np.isfinite(pixel_dwell_time_us) else np.nan
        frame_time_s = pixel_dwell_time_us * nx * ny / 1e6 if np.isfinite(pixel_dwell_time_us) else np.nan
        volume_time_s = frame_time_s * nz if np.isfinite(frame_time_s) else np.nan
        acquisition_duration_s = volume_time_s * nrep if np.isfinite(volume_time_s) else np.nan

        if not source_time_axis_path:
            if fallback_time_axis_run_id is None:
                fallback_time_axis_run_id = self.sum_run_id
            source_time_axis_path = f"/{self.output_key}/{fallback_time_axis_run_id}/axes/time_ns"

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
            "output_nrep": nrep,
            "output_nz": nz,
            "output_ny": ny,
            "output_nx": nx,
            "range_x_um": range_x_um,
            "range_y_um": range_y_um,
            "range_z_um": range_z_um,
            "pixel_size_x_um": pixel_size_x_um,
            "pixel_size_y_um": pixel_size_y_um,
            "pixel_size_z_um": pixel_size_z_um,
            "output_pixel_size_x_um": pixel_size_x_um,
            "output_pixel_size_y_um": pixel_size_y_um,
            "output_pixel_size_z_um": pixel_size_z_um,
            "timebins": output_timebins,
            "output_timebins": output_timebins,
            "selected_time_bins_json": json.dumps(list(range(output_timebins))),
            "timebin_in_ns": timebin_in_ns,
            "digital_time_bin_in_ns": timebin_in_ns,
            "bin_width_in_ns": timebin_in_ns,
            "time_axis_start_ns": 0.0,
            "time_axis_last_ns": timebin_in_ns * max(output_timebins - 1, 0)
            if np.isfinite(timebin_in_ns)
            else np.nan,
            "time_axis_span_ns": timebin_in_ns * output_timebins
            if np.isfinite(timebin_in_ns)
            else np.nan,
            "laser_frequency_mhz": laser_frequency_mhz,
            "laser_freq_mhz": laser_frequency_mhz,
            "laser_freq_in_mhz": laser_frequency_mhz,
            "laser_period_ns": laser_period_ns,
            "laser_period_in_ns": laser_period_ns,
            "dwell_time_per_circular_point_us": self._safe_float(
                timing_attrs.get("dwell_time_per_circular_point_us", np.nan)
            ),
            "pixel_dwell_time_us": pixel_dwell_time_us,
            "pixel_dwell_time_in_ns": pixel_dwell_time_in_ns,
            "line_time_s": line_time_s,
            "frame_time_s": frame_time_s,
            "volume_time_s": volume_time_s,
            "acquisition_duration_s": acquisition_duration_s,
            "primary_channel_count": primary_info["channel_count"],
            "selected_channel_count": len(channels),
            "selected_channels_json": json.dumps(channels),
            "channel_aggregation": "sum",
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
        output_timebins = int(metadata_attrs["output_timebins"])

        self._create_axis_dataset(
            axes_group,
            "repetition_index",
            np.arange(nrep, dtype=np.float64),
            "index",
            "repetition index",
            "repetition",
        )

        z_values, z_units = self._axis_values(nz, metadata_attrs["output_pixel_size_z_um"], "um")
        y_values, y_units = self._axis_values(ny, metadata_attrs["output_pixel_size_y_um"], "um")
        x_values, x_units = self._axis_values(nx, metadata_attrs["output_pixel_size_x_um"], "um")
        self._create_axis_dataset(axes_group, "z_um", z_values, z_units, "z position", "z")
        self._create_axis_dataset(axes_group, "y_um", y_values, y_units, "y position", "y")
        self._create_axis_dataset(axes_group, "x_um", x_values, x_units, "x position", "x")

        time_values, time_units = self._time_axis(output_timebins, metadata_attrs["timebin_in_ns"])
        self._create_axis_dataset(axes_group, "time_ns", time_values, time_units, "time", "time_bin")
        return axes_group

    def _create_virtual_channel_dataset(self, group, source_info, channel_index, name, channel_type):
        source_dataset = source_info["dataset"]
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
                "virtual_channel_type": channel_type,
                "virtual_source_file": ".",
                "source_data_path": source_info["path"],
                "source_channel_index": int(channel_index),
                "source_channel_axis": -1,
                "source_selection": self._source_selection_string(source_dataset.ndim, channel_index),
                "axis_order": source_info["output_axis_order"],
                "units": "counts",
                "is_virtual_dataset": True,
            },
        )

    def _create_virtual_channels(self, output_group, primary_info, extra_info):
        group = output_group.create_group("virtual_channels")
        axis_source_info = primary_info if primary_info is not None else extra_info
        self._set_attrs(
            group,
            {
                "virtual_channel_schema_version": "0.1.0",
                "description": "Named virtual datasets mapping individual source channels.",
                "primary_source_data_path": primary_info["path"] if primary_info is not None else "",
                "extra_source_data_path": extra_info["path"] if extra_info is not None else "",
                "primary_channel_count": primary_info["channel_count"] if primary_info is not None else 0,
                "extra_channel_count": extra_info["channel_count"] if extra_info is not None else 0,
                "source_axis_order": axis_source_info["source_axis_order"] if axis_source_info is not None else "",
                "virtual_axis_order": axis_source_info["output_axis_order"] if axis_source_info is not None else "",
                "naming_rule_primary": "data_channel_<channel_index>",
                "naming_rule_extra": "data_extra_channel_<extra_channel_index>",
            },
        )

        if primary_info is not None:
            for channel_index in range(primary_info["channel_count"]):
                self._create_virtual_channel_dataset(
                    group,
                    primary_info,
                    channel_index,
                    f"data_channel_{channel_index}",
                    "primary",
                )

        if extra_info is not None:
            for channel_index in range(extra_info["channel_count"]):
                self._create_virtual_channel_dataset(
                    group,
                    extra_info,
                    channel_index,
                    f"data_extra_channel_{channel_index}",
                    "extra",
                )

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

    def _create_sum_product(
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
                "metadata_path": f"/{self.output_key}/{self.sum_run_id}/metadata",
                "time_axis_path": f"/{self.output_key}/{self.sum_run_id}/axes/time_ns",
                "timebin_in_ns": metadata_attrs["timebin_in_ns"],
                "laser_frequency_mhz": metadata_attrs["laser_frequency_mhz"],
                "laser_period_ns": metadata_attrs["laser_period_ns"],
                "source_channel_axis": -1,
                "selected_channels_json": json.dumps(channels),
                "channel_aggregation": "sum",
                "description": description,
            },
        )
        self._sum_channels_to_dataset(source_dataset, dataset, channels)
        return dataset

    def _create_sum_using_shift_product(
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
                "metadata_path": f"/{self.output_key}/{self.sum_using_shift_run_id}/metadata",
                "time_axis_path": f"/{self.output_key}/{self.sum_using_shift_run_id}/axes/time_ns",
                "timebin_in_ns": metadata_attrs["timebin_in_ns"],
                "laser_frequency_mhz": metadata_attrs["laser_frequency_mhz"],
                "laser_period_ns": metadata_attrs["laser_period_ns"],
                "source_channel_axis": -1,
                "selected_channels_json": json.dumps(channels),
                "channel_aggregation": "sum_using_shift",
                "channel_skew_path": channel_skew_path,
                "channel_skew_json": json.dumps(np.asarray(shifts, dtype=float).tolist()),
                "reverse_shifts": self.shifted_sum_reverse_shifts,
                "shift_backend": self.shifted_sum_backend,
                "description": description,
            },
        )
        self._sum_channels_with_shifts_to_dataset(source_dataset, dataset, channels, shifts)
        return dataset

    def _create_sum_run(self, handle, output_group, primary_info, extra_info):
        metadata = self._read_mcs_metadata(self.data_path)
        channels = self._resolve_channels(self.channels, primary_info["channel_count"], "channels")
        metadata_attrs = self._build_metadata_attrs(
            handle,
            primary_info,
            channels,
            metadata,
            fallback_time_axis_run_id=self.sum_run_id,
        )
        extra_channels = None
        if extra_info is not None:
            extra_channels = self._resolve_channels(
                self.extra_channels,
                extra_info["channel_count"],
                "extra_channels",
            )
        primary_calibration_path = self._calibration_path_for_source(handle, primary_info)
        extra_calibration_path = self._calibration_path_for_source(handle, extra_info)

        run_group = output_group.create_group(self.sum_run_id)
        source_extra_data_path = extra_info["path"] if extra_info is not None else ""
        self._set_attrs(
            run_group,
            {
                "output_id": self.sum_run_id,
                "output_type": "image_tool",
                "tool_name": "Sum",
                "created_utc": self._utc_now(),
                "software_name": "brighteyes_mcs_file",
                "software_version": self._package_version(),
                "algorithm_name": "sum_along_channel_axis",
                "algorithm_version": "0.1.0",
                "source_data_path": primary_info["path"],
                "source_extra_data_path": source_extra_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_extra_calibration_path": extra_calibration_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "source_timing_metadata_path": metadata_attrs["source_timing_metadata_path"],
                "source_axes_path": metadata_attrs["source_axes_path"],
                "input_axis_order": primary_info["source_axis_order"],
                "output_axis_order": primary_info["output_axis_order"],
                "output_data_path": f"/{self.output_key}/{self.sum_run_id}/products/image",
                "time_axis_source": metadata_attrs["source_time_axis_path"],
                "time_axis_path": f"/{self.output_key}/{self.sum_run_id}/axes/time_ns",
                "channel_axis_source": f"{primary_info['path']} final axis",
                "parameter_encoding": "attrs_and_json",
            },
        )

        inputs_group = run_group.create_group("inputs")
        self._set_attrs(
            inputs_group,
            {
                "source_data_path": primary_info["path"],
                "source_extra_data_path": source_extra_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_extra_calibration_path": extra_calibration_path,
                "source_paths_json": json.dumps(
                    [path for path in (primary_info["path"], source_extra_data_path) if path]
                ),
                "input_axis_order": primary_info["source_axis_order"],
                "selected_channels_json": json.dumps(channels),
                "selected_extra_channels_json": json.dumps(extra_channels or []),
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
                        "extra_channels": extra_channels or [],
                        "include_extra_sum": self.include_extra_sum,
                    }
                ),
                "tool_name": "Sum",
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
        self._create_sum_product(
            products_group,
            "image",
            primary_info,
            channels,
            metadata_attrs,
            primary_calibration_path,
            "channel-summed primary digital detector counts",
            "Primary Sum output produced by summing selected detector channels.",
        )

        if self.include_extra_sum and extra_info is not None:
            self._create_sum_product(
                products_group,
                "image_extra",
                extra_info,
                extra_channels,
                metadata_attrs,
                extra_calibration_path,
                "channel-summed auxiliary digital FIFO counts",
                "Auxiliary Sum output produced by summing selected extra digital channels.",
            )

        run_group.create_group("intermediates")
        run_group.create_group("tables")
        provenance_group = run_group.create_group("provenance")
        self._set_attrs(
            provenance_group,
            {
                "command": "H5OutputBuilder.build",
                "input_file_fingerprint": "",
                "calibration_fingerprint": "",
            },
        )
        views_group = run_group.create_group("views")
        self._set_attrs(views_group, {"description": "Optional display-friendly products."})

    def _create_sum_using_shift_run(self, handle, output_group, primary_info, extra_info):
        metadata = self._read_mcs_metadata(self.data_path)
        channels = self._resolve_channels(self.channels, primary_info["channel_count"], "channels")
        primary_shifts, primary_skew_path, primary_skew_error = self._resolve_channel_skew(
            handle,
            primary_info,
            channels,
        )
        if primary_shifts is None:
            message = f"Sum_using_shift skipped: {primary_skew_error}"
            if self.require_shifted_sum:
                raise ValueError(message)
            output_group.attrs["sum_using_shift_skipped"] = True
            output_group.attrs["sum_using_shift_skip_reason"] = message
            return False

        metadata_attrs = self._build_metadata_attrs(
            handle,
            primary_info,
            channels,
            metadata,
            fallback_time_axis_run_id=self.sum_using_shift_run_id,
        )
        metadata_attrs["channel_aggregation"] = "sum_using_shift"

        extra_channels = None
        extra_shifts = None
        extra_skew_path = ""
        extra_skew_error = ""
        if extra_info is not None:
            extra_channels = self._resolve_channels(
                self.extra_channels,
                extra_info["channel_count"],
                "extra_channels",
            )
            extra_shifts, extra_skew_path, extra_skew_error = self._resolve_channel_skew(
                handle,
                extra_info,
                extra_channels,
            )

        primary_calibration_path = self._calibration_path_for_source(handle, primary_info)
        extra_calibration_path = self._calibration_path_for_source(handle, extra_info)
        source_extra_data_path = extra_info["path"] if extra_info is not None else ""

        run_group = output_group.create_group(self.sum_using_shift_run_id)
        self._set_attrs(
            run_group,
            {
                "output_id": self.sum_using_shift_run_id,
                "output_type": "image_tool",
                "tool_name": "Sum_using_shift",
                "created_utc": self._utc_now(),
                "software_name": "brighteyes_mcs_file",
                "software_version": self._package_version(),
                "algorithm_name": "sum_channel_applying_shifts",
                "algorithm_version": "0.1.0",
                "source_data_path": primary_info["path"],
                "source_extra_data_path": source_extra_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_extra_calibration_path": extra_calibration_path,
                "source_metadata_path": metadata_attrs["source_metadata_path"],
                "source_timing_metadata_path": metadata_attrs["source_timing_metadata_path"],
                "source_axes_path": metadata_attrs["source_axes_path"],
                "input_axis_order": primary_info["source_axis_order"],
                "output_axis_order": primary_info["output_axis_order"],
                "output_data_path": f"/{self.output_key}/{self.sum_using_shift_run_id}/products/image",
                "time_axis_source": metadata_attrs["source_time_axis_path"],
                "time_axis_path": f"/{self.output_key}/{self.sum_using_shift_run_id}/axes/time_ns",
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
                "source_extra_data_path": source_extra_data_path,
                "source_calibration_path": primary_calibration_path,
                "source_extra_calibration_path": extra_calibration_path,
                "source_paths_json": json.dumps(
                    [path for path in (primary_info["path"], source_extra_data_path) if path]
                ),
                "input_axis_order": primary_info["source_axis_order"],
                "selected_channels_json": json.dumps(channels),
                "selected_extra_channels_json": json.dumps(extra_channels or []),
                "selected_time_bins_json": metadata_attrs["selected_time_bins_json"],
                "channel_skew_path": primary_skew_path,
                "extra_channel_skew_path": extra_skew_path,
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
                        "extra_channels": extra_channels or [],
                        "include_extra_shifted_sum": self.include_extra_shifted_sum,
                        "reverse_shifts": self.shifted_sum_reverse_shifts,
                        "backend": self.shifted_sum_backend,
                        "chunk_size": self.shifted_sum_chunk_size,
                    }
                ),
                "tool_name": "Sum_using_shift",
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
        self._create_sum_using_shift_product(
            products_group,
            "image",
            primary_info,
            channels,
            primary_shifts,
            primary_skew_path,
            metadata_attrs,
            primary_calibration_path,
            "channel-skew-corrected sum of primary digital detector counts",
            (
                "Primary Sum_using_shift output produced with "
                "Alignment.sum_channel_applying_shifts(data, channel_skew)."
            ),
        )

        if self.include_extra_shifted_sum and extra_info is not None:
            if extra_shifts is not None:
                self._create_sum_using_shift_product(
                    products_group,
                    "image_extra",
                    extra_info,
                    extra_channels,
                    extra_shifts,
                    extra_skew_path,
                    metadata_attrs,
                    extra_calibration_path,
                    "channel-skew-corrected sum of auxiliary digital FIFO counts",
                    (
                        "Auxiliary Sum_using_shift output produced with "
                        "Alignment.sum_channel_applying_shifts(data, channel_skew)."
                    ),
                )
            else:
                run_group.attrs["extra_sum_using_shift_skipped"] = True
                run_group.attrs["extra_sum_using_shift_skip_reason"] = extra_skew_error

        run_group.create_group("intermediates")
        run_group.create_group("tables")
        skew_group = run_group.create_group("intermediates/channel_skew")
        skew_group.create_dataset("primary", data=np.asarray(primary_shifts, dtype=np.float64))
        if extra_shifts is not None:
            skew_group.create_dataset("extra", data=np.asarray(extra_shifts, dtype=np.float64))
        provenance_group = run_group.create_group("provenance")
        self._set_attrs(
            provenance_group,
            {
                "command": "H5OutputBuilder.build",
                "input_file_fingerprint": "",
                "calibration_fingerprint": "",
            },
        )
        views_group = run_group.create_group("views")
        self._set_attrs(views_group, {"description": "Optional display-friendly products."})
        return True

    def build(self):
        """Create or replace the ``/output`` group and return the output path."""
        output_path = self._prepare_output_file()
        with h5py.File(output_path, "a") as handle:
            primary_info, extra_info = self._resolve_source_infos(handle)

            if self.output_key in handle:
                if not self.overwrite:
                    raise FileExistsError(f"/{self.output_key} already exists in {output_path}")
                del handle[self.output_key]
            output_group = handle.create_group(self.output_key)
            self._set_attrs(
                output_group,
                {
                    "output_schema_version": "0.1.0",
                    "default": "",
                    "run_count": 0,
                    "description": "Outputs of image-analysis tools. Raw data are not copied here.",
                },
            )

            if self.create_virtual_channels:
                self._create_virtual_channels(output_group, primary_info, extra_info)
            default_run = ""
            run_count = 0
            if self.create_sum:
                self._create_sum_run(handle, output_group, primary_info, extra_info)
                default_run = self.sum_run_id
                run_count += 1
            if self.create_sum_using_shift:
                shifted_created = self._create_sum_using_shift_run(
                    handle,
                    output_group,
                    primary_info,
                    extra_info,
                )
                if shifted_created:
                    if not default_run:
                        default_run = self.sum_using_shift_run_id
                    run_count += 1

            output_group.attrs["default"] = default_run
            output_group.attrs["run_count"] = run_count

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

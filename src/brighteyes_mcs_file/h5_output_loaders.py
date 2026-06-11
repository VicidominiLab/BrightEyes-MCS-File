"""Strict readers for BrightEyes HDF5 output defaults."""

from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .h5_data_calibrator import (
    BRIGHTEYES_H5_DATA_PATH,
    BRIGHTEYES_H5_OUTPUT_PATH,
    SUM_IRF_TRACE_OUTPUT_KIND,
    SUM_REFERENCE_TRACE_OUTPUT_KIND,
)

__all__ = [
    "load_default_output_spad",
    "load_default_irf",
    "load_default_ref",
    "load_raw",
    "load_virtual_channel",
]

VALID_RAW_KINDS = {"spad", "aux", "analog"}
VALID_VIRTUAL_CHANNEL_KINDS = VALID_RAW_KINDS


@contextmanager
def _open_source(source):
    if isinstance(source, h5py.File):
        yield source
        return
    if isinstance(source, (str, bytes, Path)):
        with h5py.File(source, "r") as handle:
            yield handle
        return
    raise TypeError("source must be a file path or an open h5py.File")


def _attr_to_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _attr_to_string(value.item())
    return str(value)


def _required_attr(attrs, name, owner_path):
    if name not in attrs:
        raise KeyError(f"{owner_path}.attrs[{name!r}] is required by the current scheme")
    value = _attr_to_string(attrs[name]).strip()
    if not value:
        raise KeyError(f"{owner_path}.attrs[{name!r}] is empty")
    return value


def _normalize_h5_path(path):
    path = _attr_to_string(path).strip()
    if not path:
        raise KeyError("HDF5 path is empty")
    return "/" + path.strip("/")


def _output_group(handle):
    key = BRIGHTEYES_H5_OUTPUT_PATH.strip("/")
    if key not in handle or not isinstance(handle[key], h5py.Group):
        raise KeyError(f"{BRIGHTEYES_H5_OUTPUT_PATH} group is required by the current scheme")
    return handle[key]


def _dataset_at(handle, path):
    path = _normalize_h5_path(path)
    key = path.strip("/")
    if key not in handle or not isinstance(handle[key], h5py.Dataset):
        raise KeyError(f"{path} dataset is required by the current scheme")
    return handle[key]


def _read_dataset(dataset, selection):
    if selection is None:
        return np.asarray(dataset[...])
    return np.asarray(dataset[selection])


def _validate_raw_kind(kind, label):
    kind = _attr_to_string(kind).strip()
    if kind not in VALID_RAW_KINDS:
        allowed = ", ".join(sorted(VALID_RAW_KINDS))
        raise ValueError(f"{label} must be one of {allowed}; got {kind!r}")
    return kind


def load_raw(source, kind="spad", selection=None):
    """
    Load ``/raw/<kind>`` from a current-schema file.

    ``kind`` must be one of ``"spad"``, ``"aux"``, or ``"analog"``.
    ``selection`` is forwarded to the HDF5 dataset before loading.
    """
    kind = _validate_raw_kind(kind, "raw kind")
    with _open_source(source) as handle:
        dataset = _dataset_at(handle, f"{BRIGHTEYES_H5_DATA_PATH}/{kind}")
        return _read_dataset(dataset, selection)


def load_default_output_spad(source, selection=None):
    """
    Load ``/output/<default>/products/spad`` from a current-scheme file.

    ``source`` may be a file path or an open ``h5py.File``. ``selection`` is
    forwarded to the HDF5 dataset before loading, so callers can avoid reading a
    full image stack when only one slice is needed.
    """
    with _open_source(source) as handle:
        output = _output_group(handle)
        run_id = _required_attr(output.attrs, "default", output.name)
        run_path = _normalize_h5_path(f"{output.name}/{run_id}")
        run_key = run_path.strip("/")
        if run_key not in handle or not isinstance(handle[run_key], h5py.Group):
            raise KeyError(f"{run_path} group selected by /output default is missing")
        run_group = handle[run_key]
        output_type = _attr_to_string(run_group.attrs.get("output_type", "")).strip()
        if output_type != "image_tool":
            raise ValueError(f"{run_path}.attrs['output_type'] must be 'image_tool'")
        dataset = _dataset_at(handle, f"{run_path}/products/spad")
        return _read_dataset(dataset, selection)


def load_virtual_channel(source, kind, channel_index, selection=None):
    """
    Load ``/output/virtual_channels/<kind>/channel_<channel_index>``.

    ``kind`` must be one of ``"spad"``, ``"aux"``, or ``"analog"``. The
    loader is strict on the grouped current-schema layout and does not fall
    back to old flat virtual channel names.
    """
    kind = _validate_raw_kind(kind, "virtual channel kind")
    try:
        channel_index = int(channel_index)
    except (TypeError, ValueError) as exc:
        raise TypeError("channel_index must be an integer") from exc
    if channel_index < 0:
        raise ValueError("channel_index must be non-negative")

    path = f"{BRIGHTEYES_H5_OUTPUT_PATH}/virtual_channels/{kind}/channel_{channel_index}"
    with _open_source(source) as handle:
        dataset = _dataset_at(handle, path)
        virtual_type = _attr_to_string(
            dataset.attrs.get("virtual_channel_type", "")
        ).strip()
        if virtual_type != kind:
            raise ValueError(
                f"{dataset.name}.attrs['virtual_channel_type'] must be {kind!r}, "
                f"got {virtual_type!r}"
            )
        return _read_dataset(dataset, selection)


def _load_default_trace(source, trace_kind, selection=None):
    id_attrs = {
        SUM_IRF_TRACE_OUTPUT_KIND: "default_irf_trace_id",
        SUM_REFERENCE_TRACE_OUTPUT_KIND: "default_ref_trace_id",
    }
    id_attr = id_attrs[trace_kind]
    with _open_source(source) as handle:
        output = _output_group(handle)
        trace_id = _required_attr(output.attrs, id_attr, output.name)
        trace_path = f"{output.name}/{trace_id}/products/trace"
        dataset = _dataset_at(handle, trace_path)
        output_type = _attr_to_string(dataset.attrs.get("output_type", "")).strip()
        actual_trace_kind = _attr_to_string(dataset.attrs.get("trace_kind", "")).strip()
        if output_type != "trace":
            raise ValueError(f"{dataset.name}.attrs['output_type'] must be 'trace'")
        if actual_trace_kind != trace_kind:
            raise ValueError(
                f"{dataset.name}.attrs['trace_kind'] must be {trace_kind!r}, "
                f"got {actual_trace_kind!r}"
            )
        return _read_dataset(dataset, selection)


def load_default_irf(source, selection=None):
    """Load the default summed IRF trace selected by ``/output`` attrs."""
    return _load_default_trace(source, SUM_IRF_TRACE_OUTPUT_KIND, selection=selection)


def load_default_ref(source, selection=None):
    """Load the default summed reference trace selected by ``/output`` attrs."""
    return _load_default_trace(
        source,
        SUM_REFERENCE_TRACE_OUTPUT_KIND,
        selection=selection,
    )

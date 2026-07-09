"""Helpers for adding BrightEyes analysis outputs to current-schema HDF5 files.

The current schema keeps measured data under ``/raw`` and calibration artifacts
under ``/calibration``. Derived notebook and tool results belong under
``/output/<run_id>``. This module provides the small writer used by examples and
analysis scripts so those results can be copied or appended without recreating
the full calibrated file layout by hand.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import h5py
import numpy as np

from .h5_file_hash import channel_fingerprint_file_hash_attrs

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    PackageNotFoundError = Exception
    version = None


BRIGHTEYES_H5_DATA_FORMAT_VERSION = "0.0.6"
BRIGHTEYES_H5_SCHEMA_NAME = "brighteyes_mcs_file"
BRIGHTEYES_H5_SCHEMA_VARIANT = "unified_metadata_axes"
BRIGHTEYES_H5_OUTPUT_PATH = "/output"

OUTPUT_DEFAULT_ATTRS = {
    "default": "",
    "default_run": "",
    "default_irf_trace_id": "",
    "default_ref_trace_id": "",
}

VALID_WRITE_MODES = {"append", "copy", "outputs_only"}

__all__ = [
    "H5OutputProduct",
    "ensure_output_group",
    "resolve_output_run_id",
    "write_attrs",
    "write_h5_output_run",
]


@dataclass
class H5OutputProduct:
    """Dataset payload and metadata for one output product.

    ``name`` is relative to ``/output/<run_id>/products`` or to an intermediate
    collection. It may include slashes, for example ``"fit_maps/tau"``.
    ``attrs`` should carry the schema-facing context readers need, such as
    source paths, units, data role, and axis order.
    """

    name: str
    data: object
    attrs: dict | None = None
    compression: str | None = "gzip"
    chunks: object = True


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _package_version():
    if version is None:
        return "unknown"
    try:
        return version("brighteyes-mcs-dataprep")
    except PackageNotFoundError:
        return "unknown"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _attr_value(value):
    """Return an HDF5-compatible attribute value.

    h5py attributes cannot store arbitrary Python containers directly. Lists,
    tuples, dicts, NumPy arrays, and NumPy scalars are converted to plain JSON
    strings so downstream readers can decode them without relying on pickle or
    notebook-specific Python objects.
    """

    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_json_default)
    return value


def _render_templates(value, run_id, run_path):
    """Replace run placeholders inside attrs, metadata, and product names."""

    if isinstance(value, str):
        return value.replace("{run_id}", run_id).replace("{run_path}", run_path)
    if isinstance(value, Path):
        return Path(_render_templates(str(value), run_id, run_path))
    if isinstance(value, dict):
        return {
            key: _render_templates(item, run_id, run_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_render_templates(item, run_id, run_path) for item in value]
    if isinstance(value, tuple):
        return tuple(_render_templates(item, run_id, run_path) for item in value)
    return value


def _render_product(product, run_id, run_path):
    return H5OutputProduct(
        name=_render_templates(product.name, run_id, run_path),
        data=product.data,
        attrs=_render_templates(product.attrs or {}, run_id, run_path),
        compression=product.compression,
        chunks=product.chunks,
    )


def write_attrs(node, attrs):
    """Write HDF5 attrs, encoding structured Python values as JSON strings."""

    if not attrs:
        return
    for key, value in attrs.items():
        node.attrs[str(key)] = _attr_value(value)


def _normalize_output_key(output_key=BRIGHTEYES_H5_OUTPUT_PATH):
    key = str(output_key).strip("/")
    if not key:
        raise ValueError("output_key must not be empty")
    return key


def _normalize_run_id(run_id):
    run_id = str(run_id).strip("/")
    if not run_id:
        raise ValueError("run_id must not be empty")
    if "/" in run_id:
        raise ValueError("run_id must name one direct child of /output")
    return run_id


def _default_output_path(source_path, suffix):
    source_path = Path(source_path)
    return source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix or '.h5'}")


def ensure_output_group(handle, output_key=BRIGHTEYES_H5_OUTPUT_PATH):
    """Return ``/output`` and ensure its small default-index attrs exist."""

    group = handle.require_group(_normalize_output_key(output_key))
    for key, value in OUTPUT_DEFAULT_ATTRS.items():
        if key not in group.attrs:
            group.attrs[key] = value
    return group


def resolve_output_run_id(output_group, run_id, output_key_overwrite=False):
    """
    Resolve a run id under ``output_group``.

    Existing ids are auto-versioned unless ``output_key_overwrite`` is true, in
    which case only the matching child is deleted.
    """

    run_id = _normalize_run_id(run_id)
    if run_id not in output_group:
        return run_id

    if output_key_overwrite:
        del output_group[run_id]
        return run_id

    match = re.match(r"^(?P<base>.*?)(?:_(?P<index>\d+))?$", run_id)
    base = match.group("base") if match else run_id
    index = match.group("index") if match else None
    if index is None:
        base = run_id
        width = 3
        next_index = 1
    else:
        width = len(index)
        next_index = int(index) + 1

    while True:
        candidate = f"{base}_{next_index:0{width}d}"
        if candidate not in output_group:
            return candidate
        next_index += 1


def _normalize_products(products):
    """Normalize the required product collection for ``/products``.

    The public writer accepts either explicit ``H5OutputProduct`` objects or a
    simple ``{name: data}`` mapping. Explicit products are preferred for
    schema-compliant exports because they can carry per-dataset attrs.
    """

    if isinstance(products, dict):
        products = [
            value if isinstance(value, H5OutputProduct) else H5OutputProduct(name, value)
            for name, value in products.items()
        ]
    else:
        products = list(products)

    normalized = []
    for product in products:
        if not isinstance(product, H5OutputProduct):
            raise TypeError("products must contain H5OutputProduct instances or be a name->data mapping")
        name = str(product.name).strip("/")
        if not name:
            raise ValueError("product names must not be empty")
        normalized.append(
            H5OutputProduct(
                name=name,
                data=product.data,
                attrs=product.attrs or {},
                compression=product.compression,
                chunks=product.chunks,
            )
        )
    if not normalized:
        raise ValueError("at least one output product is required")
    return normalized


def _normalize_collection(collection):
    """Normalize optional collections such as axes and intermediates."""

    if collection is None:
        return []
    if isinstance(collection, dict):
        return [
            value if isinstance(value, H5OutputProduct) else H5OutputProduct(name, value)
            for name, value in collection.items()
        ]
    return list(collection)


def _dataset_kwargs(product, array):
    """Build creation kwargs while avoiding compression for scalar datasets."""

    kwargs = {"data": array}
    if np.shape(array) != ():
        if product.chunks is not None:
            kwargs["chunks"] = product.chunks
        if product.compression:
            kwargs["compression"] = product.compression
    return kwargs


def _write_dataset(group, product, attrs=None):
    """Create one dataset and attach common attrs before product-specific attrs."""

    name = str(product.name).strip("/")
    if "/" in name:
        parent_path, dataset_name = name.rsplit("/", 1)
        group = group.require_group(parent_path)
        name = dataset_name
    array = np.asarray(product.data)
    dataset = group.create_dataset(name, **_dataset_kwargs(product, array))
    write_attrs(dataset, attrs)
    write_attrs(dataset, product.attrs)
    return dataset


def _write_dataset_collection(group, collection, common_attrs=None):
    for product in _normalize_collection(collection):
        _write_dataset(group, product, attrs=common_attrs)


def _select_default_product(products, default_product):
    """Return the product selected by ``default_product`` or the first product."""

    if default_product is None:
        return products[0]
    default_product = str(default_product).strip("/")
    if "/products/" in default_product:
        default_product = default_product.rsplit("/products/", 1)[1].strip("/")
    for product in products:
        if product.name == default_product:
            return product
    available = ", ".join(product.name for product in products)
    raise ValueError(
        f"default_product must match one product name; got {default_product!r}. "
        f"Available products: {available}"
    )


def _prepare_target_file(source_path, output_path, mode):
    """Return the HDF5 file path to open for the requested write mode."""

    source_path = Path(source_path)
    if mode == "append":
        # Append writes into the source file itself. Passing a different
        # output_path would be ambiguous, so require callers to use mode="copy"
        # for the first step of an exported pipeline.
        if output_path is not None and Path(output_path) != source_path:
            raise ValueError("output_path is not used with mode='append'")
        return source_path
    if mode == "copy":
        # Copy keeps /raw and /calibration intact, then writes the first
        # /output run into the copied file. Later pipeline steps append to that
        # copied file.
        target = Path(output_path) if output_path is not None else _default_output_path(
            source_path,
            "_with_output",
        )
        shutil.copy2(source_path, target)
        return target
    if mode == "outputs_only":
        # Fragment mode is useful for derived products that intentionally do
        # not carry the measured payload. The root attrs still make the file
        # identifiable as a BrightEyes output fragment.
        return Path(output_path) if output_path is not None else _default_output_path(
            source_path,
            "_outputs",
        )
    allowed = ", ".join(sorted(VALID_WRITE_MODES))
    raise ValueError(f"mode must be one of {allowed}; got {mode!r}")


def _root_attrs(source_path, mode):
    """Root attrs that mark a file as containing current-schema outputs."""

    attrs = {
        "contains_output": True,
        "output_path": BRIGHTEYES_H5_OUTPUT_PATH,
    }
    if mode == "outputs_only":
        attrs.update(
            {
                "data_format_version": BRIGHTEYES_H5_DATA_FORMAT_VERSION,
                "schema_name": BRIGHTEYES_H5_SCHEMA_NAME,
                "schema_variant": BRIGHTEYES_H5_SCHEMA_VARIANT,
                "source_file": str(Path(source_path)),
            }
        )
    return attrs


def _source_file_hash_attrs(source_path):
    with h5py.File(source_path, "r") as handle:
        return channel_fingerprint_file_hash_attrs(handle, prefix="source_file")


def write_h5_output_run(
    source_path,
    run_id,
    products,
    *,
    mode="append",
    output_path=None,
    output_key_overwrite=False,
    output_type="image_tool",
    tool_name="",
    algorithm_name="",
    parameters=None,
    metadata=None,
    axes=None,
    inputs=None,
    attrs=None,
    intermediates=None,
    set_default=False,
    default_product=None,
):
    """Write one current-schema BrightEyes analysis run under ``/output``.

    Parameters
    ----------
    source_path:
        Existing HDF5 file to append to, copy from, or reference for an
        ``outputs_only`` fragment.
    run_id:
        Requested child name under ``/output``. Existing names are versioned
        unless ``output_key_overwrite`` is true.
    products:
        Datasets written under ``/output/<run_id>/products``. Use
        ``H5OutputProduct`` when product-level attrs such as ``data_role``,
        ``axis_order``, ``source_data_path``, and units are known.
    mode:
        ``"copy"`` for the first exported pipeline step, ``"append"`` for later
        steps in the same file, or ``"outputs_only"`` for a standalone output
        fragment.
    axes, intermediates:
        Optional dataset collections written under ``axes`` and
        ``intermediates``. They accept the same product objects as
        ``products``.
    set_default:
        If true, set ``/output.attrs["default"]`` to a product dataset path and
        ``/output.attrs["default_run"]`` to the enclosing run path.
    default_product:
        Optional product name to use for ``/output.attrs["default"]`` when
        ``set_default`` is true. If omitted, the first product is used.

    Returns
    -------
    tuple[str, str]
        The path actually written and the resolved run id.
    """

    mode = str(mode)
    if mode not in VALID_WRITE_MODES:
        allowed = ", ".join(sorted(VALID_WRITE_MODES))
        raise ValueError(f"mode must be one of {allowed}; got {mode!r}")

    products = _normalize_products(products)
    source_hash_attrs = _source_file_hash_attrs(source_path)
    target_path = _prepare_target_file(source_path, output_path, mode)
    h5_mode = "w" if mode == "outputs_only" else "a"

    with h5py.File(target_path, h5_mode) as handle:
        write_attrs(handle, _root_attrs(source_path, mode))
        output_group = ensure_output_group(handle)
        actual_run_id = resolve_output_run_id(
            output_group,
            run_id,
            output_key_overwrite=output_key_overwrite,
        )
        run_group = output_group.create_group(actual_run_id)
        actual_run_path = run_group.name
        rendered_attrs = _render_templates(attrs or {}, actual_run_id, actual_run_path)
        rendered_inputs = _render_templates(inputs or {}, actual_run_id, actual_run_path)
        rendered_metadata = _render_templates(metadata or {}, actual_run_id, actual_run_path)
        rendered_parameters = _render_templates(parameters, actual_run_id, actual_run_path)
        rendered_products = [
            _render_product(product, actual_run_id, actual_run_path)
            for product in products
        ]
        rendered_intermediates = None
        if intermediates is not None:
            rendered_intermediates = [
                _render_product(product, actual_run_id, actual_run_path)
                for product in _normalize_collection(intermediates)
            ]

        metadata_path = f"{run_group.name}/metadata"
        time_axis_path = f"{run_group.name}/axes/time_ns"
        default_product_obj = _select_default_product(rendered_products, default_product)
        default_product_path = f"{run_group.name}/products/{default_product_obj.name}"
        run_attrs = {
            "output_id": actual_run_id,
            "output_type": output_type,
            "tool_name": tool_name,
            "created_utc": _utc_now(),
            "software_name": "brighteyes_mcs_dataprep",
            "software_version": _package_version(),
            "algorithm_name": algorithm_name,
            "source_file": str(Path(source_path)),
            "output_data_path": default_product_path,
            "metadata_path": metadata_path,
            "time_axis_path": time_axis_path,
            "parameter_encoding": "attrs_and_json",
        }
        if rendered_attrs:
            run_attrs.update(rendered_attrs)
        if default_product is not None:
            run_attrs["output_data_path"] = default_product_path
        write_attrs(run_group, run_attrs)

        inputs_group = run_group.create_group("inputs")
        write_attrs(inputs_group, rendered_inputs)

        provenance_group = run_group.create_group("provenance")
        write_attrs(
            provenance_group,
            {
                "source_file": str(Path(source_path)),
                **source_hash_attrs,
            },
        )

        metadata_group = run_group.create_group("metadata")
        write_attrs(metadata_group, rendered_metadata)

        parameters_group = run_group.create_group("parameters")
        if rendered_parameters is not None:
            write_attrs(parameters_group, {"parameters_json": rendered_parameters})
            write_attrs(parameters_group, rendered_parameters)

        axes_group = run_group.create_group("axes")
        _write_dataset_collection(axes_group, axes)

        products_group = run_group.create_group("products")
        for product in rendered_products:
            product_common_attrs = {
                "output_id": actual_run_id,
                "output_run_path": run_group.name,
                "metadata_path": metadata_path,
                "time_axis_path": time_axis_path,
            }
            _write_dataset(products_group, product, attrs=product_common_attrs)

        if rendered_intermediates is not None:
            intermediates_group = run_group.create_group("intermediates")
            _write_dataset_collection(intermediates_group, rendered_intermediates)

        if set_default:
            output_group.attrs["default"] = default_product_path
            output_group.attrs["default_run"] = actual_run_path

    return str(target_path), actual_run_id

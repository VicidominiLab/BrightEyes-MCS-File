import h5py
import numpy as np
import pytest

from brighteyes_mcs_file import (
    load_default_irf,
    load_default_output_spad,
    load_default_ref,
    load_raw,
    load_virtual_channel,
)
from brighteyes_mcs_file.h5_data_calibrator import (
    SUM_IRF_TRACE_OUTPUT_KIND,
    SUM_REFERENCE_TRACE_OUTPUT_KIND,
)


def _create_trace(output_group, name, trace_kind, data):
    products_group = output_group.create_group(f"{name}/products")
    dataset = products_group.create_dataset("trace", data=np.asarray(data, dtype=float))
    dataset.attrs["output_type"] = "trace"
    dataset.attrs["trace_kind"] = trace_kind
    return dataset


def test_default_output_loaders_follow_current_schema_attrs(tmp_path):
    file_path = tmp_path / "current_output.h5"
    image = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

    with h5py.File(file_path, "w") as handle:
        output_group = handle.create_group("output")
        output_group.attrs["default"] = "sum_channels_002"
        output_group.attrs["default_irf_trace_id"] = "sum_irf_002"
        output_group.attrs["default_ref_trace_id"] = "sum_ref_001"

        run_group = output_group.create_group("sum_channels_002")
        run_group.attrs["output_type"] = "image_tool"
        products_group = run_group.create_group("products")
        products_group.create_dataset("spad", data=image)

        _create_trace(
            output_group,
            "sum_irf_001",
            SUM_IRF_TRACE_OUTPUT_KIND,
            [1.0, 2.0, 3.0],
        )
        _create_trace(
            output_group,
            "sum_irf_002",
            SUM_IRF_TRACE_OUTPUT_KIND,
            [10.0, 20.0, 30.0],
        )
        _create_trace(
            output_group,
            "sum_ref_001",
            SUM_REFERENCE_TRACE_OUTPUT_KIND,
            [100.0, 200.0, 300.0],
        )

    np.testing.assert_array_equal(load_default_output_spad(file_path), image)
    np.testing.assert_array_equal(
        load_default_output_spad(file_path, selection=np.s_[1, ...]),
        image[1, ...],
    )
    np.testing.assert_allclose(load_default_irf(file_path), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(load_default_ref(file_path), [100.0, 200.0, 300.0])

    with h5py.File(file_path, "r") as handle:
        np.testing.assert_allclose(
            load_default_irf(handle, selection=np.s_[1:]),
            [20.0, 30.0],
        )


def test_default_output_loaders_do_not_fall_back_to_old_paths(tmp_path):
    file_path = tmp_path / "old_output.h5"
    with h5py.File(file_path, "w") as handle:
        output_group = handle.create_group("output")
        output_group.create_dataset("sum_irf_trace", data=[1.0, 2.0, 3.0])

    with pytest.raises(KeyError, match="default"):
        load_default_output_spad(file_path)
    with pytest.raises(KeyError, match="default_irf_trace_id"):
        load_default_irf(file_path)
    with pytest.raises(KeyError, match="default_ref_trace_id"):
        load_default_ref(file_path)


def test_default_output_spad_does_not_fall_back_to_image_product(tmp_path):
    file_path = tmp_path / "old_image_product.h5"
    with h5py.File(file_path, "w") as handle:
        output_group = handle.create_group("output")
        output_group.attrs["default"] = "sum_channels_001"
        run_group = output_group.create_group("sum_channels_001")
        run_group.attrs["output_type"] = "image_tool"
        run_group.create_group("products").create_dataset("image", data=[1, 2, 3])

    with pytest.raises(KeyError, match="/output/sum_channels_001/products/spad"):
        load_default_output_spad(file_path)


def test_load_raw_selects_current_schema_payloads(tmp_path):
    file_path = tmp_path / "raw_payloads.h5"
    spad = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    aux = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    analog = np.arange(8, dtype=np.int16).reshape(2, 4)

    with h5py.File(file_path, "w") as handle:
        raw = handle.create_group("raw")
        raw.create_dataset("spad", data=spad)
        raw.create_dataset("aux", data=aux)
        raw.create_dataset("analog", data=analog)

    np.testing.assert_array_equal(load_raw(file_path), spad)
    np.testing.assert_array_equal(load_raw(file_path, "spad"), spad)
    np.testing.assert_array_equal(
        load_raw(file_path, "aux", selection=np.s_[1, ...]),
        aux[1, ...],
    )
    np.testing.assert_array_equal(load_raw(file_path, "analog"), analog)

    with h5py.File(file_path, "r") as handle:
        np.testing.assert_array_equal(load_raw(handle, "spad"), spad)


def test_load_raw_is_strict(tmp_path):
    file_path = tmp_path / "raw_payloads_strict.h5"
    with h5py.File(file_path, "w") as handle:
        raw = handle.create_group("raw")
        raw.create_dataset("spad", data=[1, 2, 3])

    with pytest.raises(ValueError, match="raw kind"):
        load_raw(file_path, "data")
    with pytest.raises(KeyError, match="/raw/aux"):
        load_raw(file_path, "aux")


def test_load_virtual_channel_follows_grouped_current_schema(tmp_path):
    file_path = tmp_path / "virtual_channels.h5"
    spad = np.arange(12, dtype=np.uint16).reshape(3, 4)
    aux = np.arange(6, dtype=np.uint8).reshape(2, 3)
    analog = np.arange(8, dtype=np.int16).reshape(2, 4)

    with h5py.File(file_path, "w") as handle:
        output = handle.create_group("output")
        virtual_channels = output.create_group("virtual_channels")
        for kind, data in (("spad", spad), ("aux", aux), ("analog", analog)):
            group = virtual_channels.create_group(kind)
            dataset = group.create_dataset("channel_0", data=data)
            dataset.attrs["virtual_channel_type"] = kind

    np.testing.assert_array_equal(load_virtual_channel(file_path, "spad", 0), spad)
    np.testing.assert_array_equal(
        load_virtual_channel(file_path, "aux", 0, selection=np.s_[1, ...]),
        aux[1, ...],
    )
    np.testing.assert_array_equal(load_virtual_channel(file_path, "analog", 0), analog)

    with h5py.File(file_path, "r") as handle:
        np.testing.assert_array_equal(load_virtual_channel(handle, "spad", 0), spad)


def test_load_virtual_channel_is_strict(tmp_path):
    file_path = tmp_path / "virtual_channels_strict.h5"
    with h5py.File(file_path, "w") as handle:
        output = handle.create_group("output")
        virtual_channels = output.create_group("virtual_channels")
        virtual_channels.create_dataset("data_channel_0", data=[1, 2, 3])
        group = virtual_channels.create_group("spad")
        dataset = group.create_dataset("channel_0", data=[1, 2, 3])
        dataset.attrs["virtual_channel_type"] = "aux"

    with pytest.raises(ValueError, match="virtual channel kind"):
        load_virtual_channel(file_path, "detector", 0)
    with pytest.raises(ValueError, match="non-negative"):
        load_virtual_channel(file_path, "spad", -1)
    with pytest.raises(KeyError, match="/output/virtual_channels/aux/channel_0"):
        load_virtual_channel(file_path, "aux", 0)
    with pytest.raises(ValueError, match="virtual_channel_type"):
        load_virtual_channel(file_path, "spad", 0)

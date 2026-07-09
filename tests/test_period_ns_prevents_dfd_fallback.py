import json

import h5py
import numpy as np
import pytest

from brighteyes_mcs_reader import load_default_output_spad, load_virtual_channel
from brighteyes_mcs_dataprep.h5_data_calibrator import (
    DEFAULT_SUM_IRF_TRACE_ID,
    DEFAULT_SUM_CHANNELS_RUN_ID,
    DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID,
    DEFAULT_SUM_REFERENCE_TRACE_ID,
    H5DataCalibrator,
    H5OutputBuilder,
    SUM_IRF_TRACE_OUTPUT_KIND,
    SUM_REFERENCE_TRACE_OUTPUT_KIND,
)


class BadMetadata:
    @property
    def dfd_freq(self):
        raise AssertionError("dfd_freq should not be accessed when period_ns is explicit")

    @property
    def dfd_nbins(self):
        raise AssertionError("dfd_nbins should not be accessed when period_ns is explicit")


def test_build_metadata_attrs_uses_explicit_period_ns_without_dfd_fallback(monkeypatch):
    calibrator = H5OutputBuilder.__new__(H5OutputBuilder)
    calibrator.period_ns = 25.0
    calibrator.sum_channels_run_id = "sum"
    calibrator._time_axis_path = lambda run_id: "/axes/time_ns"

    monkeypatch.setattr(
        H5OutputBuilder,
        "_timing_attr_source",
        lambda self, handle, primary_info: ("", {}),
    )
    monkeypatch.setattr(
        H5OutputBuilder,
        "_get_existing_path",
        lambda self, handle, candidates: None,
    )

    attrs = calibrator._build_metadata_attrs(
        handle=object(),
        primary_info={
            "shape": (1, 1, 1, 1, 64),
            "data_layout": "",
            "subpixel_scan_mode": "",
            "channel_count": 1,
        },
        channels=[0],
        metadata=BadMetadata(),
    )

    assert np.isfinite(attrs["laser_period_ns"])
    assert attrs["laser_period_ns"] == 25.0
    assert np.isfinite(attrs["laser_frequency_mhz"])
    assert np.isclose(attrs["laser_frequency_mhz"], 1000.0 / 25.0)


def test_payload_copy_adds_symmetric_source_shape_dtype_attrs(tmp_path):
    path = tmp_path / "copy_payload.h5"
    data = np.arange(6, dtype=np.uint8).reshape(1, 1, 1, 1, 3, 2)

    with h5py.File(path, "w") as h5:
        source = h5.create_dataset("data_channels_extra", data=data)
        raw = h5.create_group("raw")
        copied = H5DataCalibrator._copy_payload_dataset(
            h5,
            raw,
            "data_channels_extra",
            "aux",
            {"source_input_path": "/data_channels_extra"},
            required=True,
        )

        assert copied.name == "/raw/aux"
        assert copied.attrs["actual_source_path"] == source.name
        assert json.loads(copied.attrs["shape_json"]) == list(data.shape)
        assert copied.attrs["dtype_preserved_from_source"] == np.bool_(True)


def test_analog_adc_calibration_attrs_default_to_nan_per_channel():
    attrs = H5DataCalibrator._analog_adc_calibration_attrs(2)

    assert attrs["adc_calibration_formula"] == (
        "voltage_v = adc_offset_v + adc_slope_v_per_adc_unit * adc_counts"
    )
    assert np.isnan(attrs["adc_offset_v"])
    assert np.isnan(attrs["adc_slope_v_per_adc_unit"])

    channel_offsets = json.loads(attrs["adc_channel_offset_v_json"])
    channel_slopes = json.loads(attrs["adc_channel_slope_v_per_adc_unit_json"])
    assert len(channel_offsets) == 2
    assert len(channel_slopes) == 2
    assert all(np.isnan(value) for value in channel_offsets)
    assert all(np.isnan(value) for value in channel_slopes)


def test_timing_attr_source_falls_back_to_calibration_result(tmp_path):
    path = tmp_path / "timing_fallback.h5"
    builder = H5OutputBuilder.__new__(H5OutputBuilder)

    with h5py.File(path, "w") as h5:
        dataset = h5.create_dataset("raw/spad", data=np.zeros((1, 1, 1, 1, 3, 2)))
        calibration = h5.create_group("calibration/results/spad")
        calibration.attrs["time_bin_ns"] = 1.25
        source_info = builder._source_info(dataset, "spad")

        source_path, attrs = builder._timing_attr_source(h5, source_info)

        assert source_path == "/calibration/results/spad"
        assert attrs["time_bin_ns"] == 1.25


def test_create_common_shifted_calibration_trace(tmp_path):
    output_path = tmp_path / "trace_output.h5"
    calibrator = H5OutputBuilder.__new__(H5OutputBuilder)
    calibrator.output_key = "output"
    calibrator.sum_channels_with_skew_correction_run_id = "sum_channels_001"
    calibrator.shifted_sum_reverse_shifts = False
    calibrator.shifted_sum_backend = "cpu"
    calibrator.shifted_sum_chunk_size = None
    calibrator.shifted_sum_show_progress = False

    metadata_attrs = {
        "source_metadata_path": "/raw/metadata",
        "source_time_axis_path": "/calibration/axes/time_ns",
        "time_bin_ns": 1.0,
        "laser_frequency_mhz": 80.0,
        "laser_period_ns": 12.5,
        "selected_time_bins_json": json.dumps([0, 1, 2, 3]),
        "output_nrep": 1,
        "output_nz": 1,
        "output_ny": 1,
        "output_nx": 1,
        "output_circular_repetition_count": 1,
        "output_circular_point_count": 1,
        "output_time_bins": 4,
        "data_layout": "raster_6d",
        "range_z_um": 0.0,
        "range_y_um": 0.0,
        "range_x_um": 0.0,
        "offset_z_um": 0.0,
        "offset_y_um": 0.0,
        "offset_x_um": 0.0,
        "output_pixel_size_z_um": 0.0,
        "output_pixel_size_y_um": 0.0,
        "output_pixel_size_x_um": 0.0,
    }

    with h5py.File(output_path, "w") as h5:
        output_group = h5.create_group("output")
        calibration_group = h5.create_group("calibration/results/spad")
        calibration_group.create_dataset("channels/index", data=[0, 1])
        calibration_group.create_dataset(
            "aligned/irf_trace",
            data=np.asarray(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                ]
            ),
        )

        dataset = calibrator._create_common_shifted_calibration_trace(
            output_group,
            h5,
            DEFAULT_SUM_IRF_TRACE_ID,
            SUM_IRF_TRACE_OUTPUT_KIND,
            "/calibration/results/spad",
            "irf_trace",
            [0, 1],
            np.asarray([0.0, 0.0]),
            "/calibration/results/spad/timing/channel_skew_bins",
            metadata_attrs,
            "summed IRF trace",
            "test trace",
        )

        assert dataset.name == f"/output/{DEFAULT_SUM_IRF_TRACE_ID}/products/trace"
        np.testing.assert_allclose(dataset[...], [11.0, 22.0, 33.0, 44.0])
        assert output_group.attrs["default_irf_trace_id"] == dataset.name
        assert f"{DEFAULT_SUM_IRF_TRACE_ID}_path" not in output_group.attrs
        assert "default_sum_irf_trace_path" not in output_group.attrs
        assert "sum_irf_trace_ids_json" not in output_group.attrs
        assert dataset.attrs["source_trace_path"] == (
            "/calibration/results/spad/aligned/irf_trace"
        )
        assert dataset.attrs["metadata_path"] == f"/output/{DEFAULT_SUM_IRF_TRACE_ID}/metadata"
        assert dataset.attrs["time_axis_path"] == f"/output/{DEFAULT_SUM_IRF_TRACE_ID}/axes/time_ns"


def test_output_builder_deprecated_aliases_map_to_canonical_names():
    with pytest.warns(DeprecationWarning) as warnings:
        builder = H5OutputBuilder(
            "input.h5",
            create_sum=False,
            create_sum_using_shift=False,
            create_sum_shifted=True,
            sum_run_id="legacy_sum",
            sum_using_shift_run_id="legacy_shift",
        )

    messages = [str(record.message) for record in warnings]
    assert any("create_sum" in message for message in messages)
    assert any("create_sum_using_shift" in message for message in messages)
    assert any("create_sum_shifted" in message for message in messages)
    assert any("sum_run_id" in message for message in messages)
    assert any("sum_using_shift_run_id" in message for message in messages)

    assert builder.create_sum_channels is False
    assert builder.create_sum_channels_with_skew_correction is True
    assert builder.sum_channels_run_id == "legacy_sum"
    assert builder.sum_channels_with_skew_correction_run_id == "legacy_shift"
    assert "create_sum" not in builder.__dict__
    assert "create_sum_using_shift" not in builder.__dict__
    assert "sum_run_id" not in builder.__dict__
    assert "sum_using_shift_run_id" not in builder.__dict__


def test_output_builder_writes_grouped_virtual_channels_for_all_payloads(tmp_path):
    input_path = tmp_path / "virtual_input.h5"
    spad = np.arange(12, dtype=np.uint16).reshape(1, 1, 1, 2, 3, 2)
    aux = np.arange(12, 18, dtype=np.uint8).reshape(1, 1, 1, 2, 3, 1)
    analog = np.arange(18, 24, dtype=np.int16).reshape(1, 1, 1, 2, 3, 1)

    with h5py.File(input_path, "w") as h5:
        raw = h5.create_group("raw")
        raw.create_dataset("spad", data=spad)
        raw.create_dataset("aux", data=aux)
        raw.create_dataset("analog", data=analog)

    H5OutputBuilder(
        input_path,
        create_sum_channels=False,
        create_sum_channels_with_skew_correction=False,
        compression=None,
    ).build()

    with h5py.File(input_path, "r") as h5:
        virtual_channels = h5["output/virtual_channels"]
        assert "data_channel_0" not in virtual_channels
        assert "data_aux_channel_0" not in virtual_channels
        assert set(virtual_channels) == {"spad", "aux", "analog"}
        assert json.loads(virtual_channels.attrs["kind_groups_json"]) == [
            "spad",
            "aux",
            "analog",
        ]
        assert virtual_channels.attrs["naming_rule"] == "<kind>/channel_<channel_index>"
        assert virtual_channels.attrs["spad_channel_count"] == 2
        assert virtual_channels.attrs["aux_channel_count"] == 1
        assert virtual_channels.attrs["analog_channel_count"] == 1

        for kind, source_path, channel_axis_path, units in (
            ("spad", "/raw/spad", "/raw/axes/detector_channel_index", "counts"),
            ("aux", "/raw/aux", "/raw/axes/aux_channel_index", "counts"),
            ("analog", "/raw/analog", "/raw/axes/analog_channel_index", "adc_counts"),
        ):
            group = virtual_channels[kind]
            dataset = group["channel_0"]
            assert group.attrs["virtual_channel_type"] == kind
            assert group.attrs["source_data_path"] == source_path
            assert group.attrs["source_channel_axis_path"] == channel_axis_path
            assert group.attrs["naming_rule"] == "channel_<channel_index>"
            assert dataset.attrs["virtual_channel_type"] == kind
            assert dataset.attrs["source_data_path"] == source_path
            assert dataset.attrs["source_channel_axis_path"] == channel_axis_path
            assert dataset.attrs["units"] == units

        assert "channel_1" in virtual_channels["spad"]
        np.testing.assert_array_equal(
            load_virtual_channel(h5, "spad", 0),
            spad[..., 0],
        )
        np.testing.assert_array_equal(
            load_virtual_channel(h5, "aux", 0),
            aux[..., 0],
        )
        np.testing.assert_array_equal(
            load_virtual_channel(h5, "analog", 0),
            analog[..., 0],
        )


def test_sum_channels_writes_spad_and_aux_products_with_aux_attrs(tmp_path):
    input_path = tmp_path / "aux_sum_input.h5"
    spad = np.arange(12, dtype=np.uint16).reshape(1, 1, 1, 2, 3, 2)
    aux = np.arange(12, 18, dtype=np.uint8).reshape(1, 1, 1, 2, 3, 1)

    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("raw/spad", data=spad)
        h5.create_dataset("raw/aux", data=aux)
        h5.create_group("raw/metadata")
        h5.create_dataset("raw/axes/digital_time_ns", data=np.arange(3, dtype=float))

    H5OutputBuilder(
        input_path,
        create_virtual_channels=False,
        create_sum_channels=True,
        create_sum_channels_with_skew_correction=False,
        compression=None,
    ).build()

    with h5py.File(input_path, "r") as h5:
        spad_product = h5[f"output/{DEFAULT_SUM_CHANNELS_RUN_ID}/products/spad"]
        aux_product = h5[f"output/{DEFAULT_SUM_CHANNELS_RUN_ID}/products/aux"]

        np.testing.assert_array_equal(spad_product[...], spad.sum(axis=-1))
        np.testing.assert_array_equal(aux_product[...], aux.sum(axis=-1))
        assert "image" not in h5[f"output/{DEFAULT_SUM_CHANNELS_RUN_ID}/products"]
        assert "image_aux" not in h5[f"output/{DEFAULT_SUM_CHANNELS_RUN_ID}/products"]
        assert aux_product.attrs["source_data_path"] == "/raw/aux"
        assert aux_product.attrs["source_calibration_path"] == ""
        assert aux_product.attrs["source_metadata_path"] == "/raw/metadata"
        assert aux_product.attrs["metadata_path"] == (
            f"/output/{DEFAULT_SUM_CHANNELS_RUN_ID}/metadata"
        )
        assert aux_product.attrs["time_axis_path"] == (
            f"/output/{DEFAULT_SUM_CHANNELS_RUN_ID}/axes/time_ns"
        )
        assert json.loads(aux_product.attrs["selected_channels_json"]) == [0]
        assert aux_product.attrs["channel_aggregation"] == "sum_channels_without_corrections"


def test_build_output_defaults_and_summed_irf_reference_products(tmp_path):
    input_path = tmp_path / "input.h5"
    data = np.arange(8, dtype=np.uint16).reshape(1, 1, 1, 1, 4, 2)
    irf_trace = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    reference_trace = irf_trace * 10.0

    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("raw/spad", data=data)
        h5.create_group("raw/metadata")
        h5.create_dataset("raw/axes/digital_time_ns", data=np.arange(4, dtype=float))
        calibration_group = h5.create_group("calibration/results/spad")
        calibration_group.attrs["time_bin_ns"] = 1.0
        calibration_group.attrs["laser_period_ns"] = 12.5
        calibration_group.attrs["laser_frequency_mhz"] = 80.0
        calibration_group.create_dataset("channels/index", data=[0, 1])
        calibration_group.create_dataset("timing/channel_skew_bins", data=[0.0, 0.0])
        calibration_group.create_dataset("aligned/irf_trace", data=irf_trace)
        calibration_group.create_dataset("aligned/reference_trace", data=reference_trace)

    H5OutputBuilder(
        input_path,
        create_virtual_channels=False,
        shifted_sum_backend="cpu",
        shifted_sum_reverse_shifts=False,
        shifted_sum_show_progress=False,
        compression=None,
    ).build()

    with h5py.File(input_path, "r") as h5:
        output_group = h5["output"]
        assert output_group.attrs["default"] == (
            f"/output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}/products/spad"
        )
        assert output_group.attrs["default_run"] == (
            f"/output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}"
        )
        assert "metadata" not in output_group
        assert "default_run_id" not in output_group.attrs
        assert "run_count" not in output_group.attrs
        assert "default_metadata_path" not in output_group.attrs
        assert "default_axes_path" not in output_group.attrs

        sum_image = h5[
            f"output/{DEFAULT_SUM_CHANNELS_RUN_ID}/products/spad"
        ][...]
        shifted_image = h5[
            f"output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}/products/spad"
        ][...]
        np.testing.assert_array_equal(sum_image, data.sum(axis=-1))
        np.testing.assert_allclose(shifted_image, data.sum(axis=-1))
        np.testing.assert_allclose(load_default_output_spad(h5), shifted_image)

        sum_irf = h5[f"output/{DEFAULT_SUM_IRF_TRACE_ID}/products/trace"]
        sum_reference = h5[f"output/{DEFAULT_SUM_REFERENCE_TRACE_ID}/products/trace"]
        np.testing.assert_allclose(sum_irf[...], irf_trace.sum(axis=-1))
        np.testing.assert_allclose(sum_reference[...], reference_trace.sum(axis=-1))
        assert output_group.attrs["default_irf_trace_id"] == sum_irf.name
        assert output_group.attrs["default_ref_trace_id"] == sum_reference.name
        assert "default_sum_irf_trace_id" not in output_group.attrs
        assert "default_sum_irf_trace_path" not in output_group.attrs
        assert "default_sum_reference_trace_id" not in output_group.attrs
        assert "default_sum_reference_trace_path" not in output_group.attrs
        assert "sum_irf_trace_ids_json" not in output_group.attrs
        assert "sum_reference_trace_ids_json" not in output_group.attrs
        products_path = (
            f"output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}/products"
        )
        assert SUM_IRF_TRACE_OUTPUT_KIND not in h5[products_path]
        assert SUM_REFERENCE_TRACE_OUTPUT_KIND not in h5[products_path]
        axes_group = h5[f"output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}/axes"]
        assert "circular_repetition_index" not in axes_group
        assert "circular_point_index" not in axes_group

    H5OutputBuilder(
        input_path,
        create_virtual_channels=False,
        create_sum_channels=False,
        shifted_sum_backend="cpu",
        shifted_sum_reverse_shifts=False,
        shifted_sum_show_progress=False,
        compression=None,
    ).build()

    with h5py.File(input_path, "r") as h5:
        output_group = h5["output"]
        assert output_group.attrs["default"] == (
            f"/output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}/products/spad"
        )
        assert output_group.attrs["default_run"] == (
            f"/output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}"
        )
        assert "default_run_id" not in output_group.attrs
        assert "run_count" not in output_group.attrs
        assert DEFAULT_SUM_CHANNELS_RUN_ID not in output_group
        assert DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID in output_group
        assert DEFAULT_SUM_IRF_TRACE_ID in output_group
        assert DEFAULT_SUM_REFERENCE_TRACE_ID in output_group
        assert output_group.attrs["default_irf_trace_id"] == (
            f"/output/{DEFAULT_SUM_IRF_TRACE_ID}/products/trace"
        )
        assert output_group.attrs["default_ref_trace_id"] == (
            f"/output/{DEFAULT_SUM_REFERENCE_TRACE_ID}/products/trace"
        )


def test_circular_output_writes_matching_circular_axes(tmp_path):
    input_path = tmp_path / "circular_input.h5"
    data = np.arange(48, dtype=np.uint16).reshape(1, 1, 1, 1, 2, 3, 4, 2)
    irf_trace = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("raw/spad", data=data)
        h5.create_group("raw/metadata")
        h5.create_dataset("raw/axes/digital_time_ns", data=np.arange(4, dtype=float))
        calibration_group = h5.create_group("calibration/results/spad")
        calibration_group.attrs["time_bin_ns"] = 1.0
        calibration_group.attrs["laser_period_ns"] = 12.5
        calibration_group.attrs["laser_frequency_mhz"] = 80.0
        calibration_group.create_dataset("channels/index", data=[0, 1])
        calibration_group.create_dataset("timing/channel_skew_bins", data=[0.0, 0.0])
        calibration_group.create_dataset("aligned/irf_trace", data=irf_trace)
        calibration_group.create_dataset("aligned/reference_trace", data=irf_trace * 10.0)

    H5OutputBuilder(
        input_path,
        create_virtual_channels=False,
        shifted_sum_backend="cpu",
        shifted_sum_reverse_shifts=False,
        shifted_sum_show_progress=False,
        compression=None,
    ).build()

    with h5py.File(input_path, "r") as h5:
        run_path = f"output/{DEFAULT_SUM_CHANNELS_WITH_SKEW_CORRECTION_RUN_ID}"
        image = h5[f"{run_path}/products/spad"]
        assert image.shape == data.shape[:-1]
        assert image.attrs["axis_order"] == (
            "repetition,z,y,x,circular_repetition,circular_point,time_bin"
        )

        axes_group = h5[f"{run_path}/axes"]
        np.testing.assert_array_equal(
            axes_group["circular_repetition_index"][...],
            np.asarray([0.0, 1.0]),
        )
        np.testing.assert_array_equal(
            axes_group["circular_point_index"][...],
            np.asarray([0.0, 1.0, 2.0]),
        )

        metadata_attrs = h5[f"{run_path}/metadata"].attrs
        assert metadata_attrs["data_layout"] == "circular_8d"
        assert metadata_attrs["circular_repetition_count"] == 2
        assert metadata_attrs["circular_point_count"] == 3
        assert metadata_attrs["output_circular_repetition_count"] == 2
        assert metadata_attrs["output_circular_point_count"] == 3

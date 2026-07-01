import json

import h5py
import numpy as np
import pytest

from brighteyes_mcs_file import convert_h5_raw, load_raw


def _write_legacy_measurement(path):
    spad = np.arange(1 * 1 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape(
        1,
        1,
        2,
        3,
        4,
        5,
    )
    aux = np.arange(1 * 1 * 2 * 3 * 4 * 2, dtype=np.uint8).reshape(
        1,
        1,
        2,
        3,
        4,
        2,
    )

    with h5py.File(path, "w") as handle:
        handle.attrs["data_format_version"] = "0.0.1"
        handle.attrs["default"] = "data"
        handle.attrs["comment"] = "raw conversion fixture"
        handle.create_dataset("data", data=spad)
        handle.create_dataset("data_channels_extra", data=aux)

        gui = handle.create_group("configurationGUI")
        gui.attrs["nx"] = 3
        gui.attrs["ny"] = 2
        gui.attrs["nframe"] = 1
        gui.attrs["nrep"] = 1
        gui.attrs["range_x"] = 0.3
        gui.attrs["range_y"] = 0.2
        gui.attrs["range_z"] = 0.0
        gui.attrs["time_resolution"] = 0.5
        gui.attrs["bitFile"] = "dummy_40M4.bit"

        fpga = handle.create_group("configurationFPGA")
        fpga.attrs["DFD_Activate"] = False

    return spad, aux


def test_convert_h5_raw_writes_current_raw_schema_without_calibration(tmp_path):
    input_path = tmp_path / "legacy.h5"
    spad, aux = _write_legacy_measurement(input_path)

    output_path = convert_h5_raw(input_path)

    assert output_path == str(tmp_path / "legacy_raw.h5")
    np.testing.assert_array_equal(load_raw(output_path), spad)
    np.testing.assert_array_equal(load_raw(output_path, "aux"), aux)

    with h5py.File(output_path, "r") as handle:
        assert set(handle.keys()) == {"raw", "calibration", "output"}
        assert handle.attrs["data_format_version"] == "0.0.6"
        assert handle.attrs["file_kind"] == "raw_measurement"
        assert handle.attrs["default"] == "/raw/spad"
        assert bool(handle.attrs["contains_measured_payload"])
        assert not bool(handle.attrs["contains_calibration"])
        assert not bool(handle.attrs["contains_output"])
        assert handle.attrs["data_path"] == "/raw"
        assert handle.attrs["calibration_path"] == "/calibration"
        assert handle.attrs["output_path"] == "/output"
        assert "data" not in handle
        assert "data_channels_extra" not in handle
        assert "configurationGUI" not in handle
        assert "configurationFPGA" not in handle

        np.testing.assert_array_equal(handle["raw/spad"][...], spad)
        np.testing.assert_array_equal(handle["raw/aux"][...], aux)
        assert handle["raw/spad"].attrs["actual_source_path"] == "/data"
        assert handle["raw/spad"].attrs["calibration_result_path"] == ""
        assert handle["raw/aux"].attrs["actual_source_path"] == "/data_channels_extra"
        assert handle["raw/aux"].attrs["calibration_result_path"] == ""

        assert handle["raw"].attrs["values_preserved"]
        assert handle["raw"].attrs["metadata_path"] == "/raw/metadata"
        assert handle["raw/metadata"].attrs["data_spad_path"] == "/raw/spad"
        assert handle["raw/metadata"].attrs["data_aux_path"] == "/raw/aux"
        assert handle["raw/axes/digital_time_ns"].shape == (4,)
        assert handle["raw/axes/detector_channel_index"].shape == (5,)
        assert handle["raw/axes/aux_channel_index"].shape == (2,)
        assert "configurationGUI" in handle["raw/legacy"]
        assert "configurationFPGA" in handle["raw/legacy"]

        calibration = handle["calibration"]
        assert calibration.attrs["calibration_status"] == "not_performed"
        assert json.loads(calibration.attrs["calibrated_products_json"]) == []
        assert set(calibration.keys()) == {"axes", "metadata", "provenance", "results"}
        assert calibration["metadata"].attrs["source_reference_file"] == ""
        assert calibration["provenance"].attrs["conversion_command"] == "convert_h5_raw"
        assert bool(calibration["provenance"].attrs["raw_data_values_preserved"])
        assert len(calibration["results"].keys()) == 0

        assert handle["output"].attrs["run_count"] == 0
        assert handle["output"].attrs["default"] == ""


def test_convert_h5_raw_respects_explicit_output_and_overwrite(tmp_path):
    input_path = tmp_path / "legacy.h5"
    _write_legacy_measurement(input_path)
    output_path = tmp_path / "modern_raw.h5"

    assert convert_h5_raw(input_path, output_path=output_path) == str(output_path)

    with pytest.raises(FileExistsError):
        convert_h5_raw(input_path, output_path=output_path, overwrite=False)


def test_convert_h5_raw_rejects_in_place_conversion(tmp_path):
    input_path = tmp_path / "legacy.h5"
    _write_legacy_measurement(input_path)

    with pytest.raises(ValueError, match="output_path must be different"):
        convert_h5_raw(input_path, output_path=input_path)

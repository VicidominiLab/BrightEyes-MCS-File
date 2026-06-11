import h5py
import numpy as np

from brighteyes_mcs_file import mcs


def _write_minimal_v006_file(path):
    analog = np.arange(4, dtype=np.int16).reshape(1, 1, 1, 1, 2, 2)

    with h5py.File(path, "w") as h5:
        h5.attrs["data_format_version"] = "0.0.6"
        h5.attrs["comment"] = "test file"

        raw = h5.create_group("raw")
        raw.create_dataset("spad", data=np.zeros((1, 1, 1, 1, 2, 1), dtype=np.uint16))
        raw.create_dataset("analog", data=analog)

        metadata = raw.create_group("metadata")
        metadata.attrs["range_x_um"] = 1.0
        metadata.attrs["range_y_um"] = 1.0
        metadata.attrs["range_z_um"] = 1.0
        metadata.attrs["offset_x_um"] = 0.0
        metadata.attrs["offset_y_um"] = 0.0
        metadata.attrs["offset_z_um"] = 0.0
        metadata.attrs["pixel_size_x_um"] = 1.0
        metadata.attrs["pixel_size_y_um"] = 1.0
        metadata.attrs["pixel_size_z_um"] = 1.0
        metadata.attrs["nx"] = 1
        metadata.attrs["ny"] = 1
        metadata.attrs["nz"] = 1
        metadata.attrs["nrep"] = 1
        metadata.attrs["time_bins"] = 2
        metadata.attrs["time_resolution_us"] = 1.0
        metadata.attrs["pixel_dwell_time_us"] = 2.0
        metadata.attrs["pixel_dwell_time_ns"] = 2000.0
        metadata.attrs["dfd_active"] = False

    return analog


def test_load_v006_analog_aliases(tmp_path):
    path = tmp_path / "minimal_v006.h5"
    analog = _write_minimal_v006_file(path)

    data, metadata = mcs.load(path, key="analog")
    np.testing.assert_array_equal(data, analog)
    assert metadata.version == "0.0.6"

    data, _ = mcs.load(path, key="data_analog")
    np.testing.assert_array_equal(data, analog)

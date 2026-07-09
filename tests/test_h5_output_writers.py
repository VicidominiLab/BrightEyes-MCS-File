import json

import h5py
import numpy as np

from brighteyes_mcs_dataprep import H5OutputProduct, write_h5_output_run
from brighteyes_mcs_dataprep.h5_file_hash import channel_fingerprint_file_hash


def _source_file(path):
    with h5py.File(path, "w") as handle:
        handle.attrs["data_format_version"] = "0.0.6"
        raw = handle.create_group("raw")
        raw.create_dataset("spad", data=np.arange(4).reshape(1, 1, 1, 1, 2, 2))
        raw.create_dataset("aux", data=np.arange(4, 8).reshape(1, 1, 1, 1, 2, 2))
        handle.create_group("calibration")
    return path


def test_write_h5_output_run_appends_and_versions_existing_run(tmp_path):
    source_path = _source_file(tmp_path / "sample_calib.h5")

    written_path, run_id = write_h5_output_run(
        source_path,
        "apr_001",
        [H5OutputProduct("spad", np.ones((2, 3)), attrs={"axis_order": "y,x"})],
        tool_name="APR reassignment",
        algorithm_name="adaptive_pixel_reassignment",
    )
    assert written_path == str(source_path)
    assert run_id == "apr_001"

    _, second_run_id = write_h5_output_run(
        source_path,
        "apr_001",
        [H5OutputProduct("spad", np.full((2, 3), 2.0))],
    )
    assert second_run_id == "apr_002"

    with h5py.File(source_path, "r") as handle:
        assert "raw" in handle
        assert handle.attrs["contains_output"]
        assert handle["output"].attrs["default"] == ""
        assert handle["output"].attrs["default_run"] == ""
        assert "apr_001" in handle["output"]
        assert "apr_002" in handle["output"]
        provenance = handle["output/apr_001/provenance"].attrs
        assert provenance["source_file"] == str(source_path)
        assert len(provenance["source_file_sha256"]) == 64
        assert provenance["source_file_hash_algorithm"] == "sha256"
        assert json.loads(provenance["source_file_hash_source_paths_json"]) == [
            "/raw/spad",
            "/raw/aux",
        ]
        np.testing.assert_array_equal(
            handle["output/apr_001/products/spad"][...],
            np.ones((2, 3)),
        )
        np.testing.assert_array_equal(
            handle["output/apr_002/products/spad"][...],
            np.full((2, 3), 2.0),
        )


def test_write_h5_output_run_overwrites_only_requested_run(tmp_path):
    source_path = _source_file(tmp_path / "sample_calib.h5")

    write_h5_output_run(
        source_path,
        "apr_001",
        [H5OutputProduct("spad", np.ones((2, 3)))],
    )
    write_h5_output_run(
        source_path,
        "s2ism_001",
        [H5OutputProduct("spad", np.full((2, 3), 3.0))],
    )
    _, run_id = write_h5_output_run(
        source_path,
        "apr_001",
        [H5OutputProduct("spad", np.full((2, 3), 4.0))],
        output_key_overwrite=True,
    )

    assert run_id == "apr_001"
    with h5py.File(source_path, "r") as handle:
        assert "s2ism_001" in handle["output"]
        assert "apr_002" not in handle["output"]
        np.testing.assert_array_equal(
            handle["output/apr_001/products/spad"][...],
            np.full((2, 3), 4.0),
        )


def test_write_h5_output_run_default_points_to_product_dataset_path(tmp_path):
    source_path = _source_file(tmp_path / "sample_calib.h5")

    write_h5_output_run(
        source_path,
        "apr_001",
        [H5OutputProduct("apr", np.ones((2, 3)))],
        set_default=True,
    )

    with h5py.File(source_path, "r") as handle:
        assert handle["output"].attrs["default"] == "/output/apr_001/products/apr"
        assert handle["output"].attrs["default_run"] == "/output/apr_001"


def test_write_h5_output_run_default_product_selects_non_first_product(tmp_path):
    source_path = _source_file(tmp_path / "sample_calib.h5")

    write_h5_output_run(
        source_path,
        "apr_001",
        [
            H5OutputProduct("apr", np.ones((1, 2, 3, 4, 5, 6))),
            H5OutputProduct("apr_sum", np.ones((1, 2, 3, 4, 5))),
        ],
        attrs={"output_data_path": "/output/{run_id}/products/apr"},
        set_default=True,
        default_product="apr_sum",
    )

    with h5py.File(source_path, "r") as handle:
        assert handle["output"].attrs["default"] == "/output/apr_001/products/apr_sum"
        assert handle["output/apr_001"].attrs["output_data_path"] == (
            "/output/apr_001/products/apr_sum"
        )


def test_write_h5_output_run_default_product_accepts_product_path(tmp_path):
    source_path = _source_file(tmp_path / "sample_calib.h5")

    write_h5_output_run(
        source_path,
        "s2ism_001",
        [
            H5OutputProduct("preview", np.ones((2, 3))),
            H5OutputProduct("s2ism", np.ones((2, 3, 4))),
        ],
        set_default=True,
        default_product="/output/s2ism_001/products/s2ism",
    )

    with h5py.File(source_path, "r") as handle:
        assert handle["output"].attrs["default"] == (
            "/output/s2ism_001/products/s2ism"
        )
        assert handle["output/s2ism_001"].attrs["output_data_path"] == (
            "/output/s2ism_001/products/s2ism"
        )


def test_write_h5_output_run_copy_and_outputs_only_modes(tmp_path):
    source_path = _source_file(tmp_path / "sample_calib.h5")

    copy_path, copy_run_id = write_h5_output_run(
        source_path,
        "apr_001",
        [H5OutputProduct("spad", np.ones((2, 3)))],
        mode="copy",
    )
    assert copy_run_id == "apr_001"
    assert copy_path.endswith("_with_output.h5")
    with h5py.File(copy_path, "r") as handle:
        assert "raw" in handle
        assert "output/apr_001/products/spad" in handle

    output_only_path, output_only_run_id = write_h5_output_run(
        source_path,
        "apr_001",
        [H5OutputProduct("spad", np.ones((2, 3)))],
        mode="outputs_only",
    )
    assert output_only_run_id == "apr_001"
    assert output_only_path.endswith("_outputs.h5")
    with h5py.File(output_only_path, "r") as handle:
        assert "raw" not in handle
        assert "calibration" not in handle
        assert handle.attrs["data_format_version"] == "0.0.6"
        assert handle.attrs["contains_output"]
        assert handle.attrs["source_file"] == str(source_path)
        assert "output/apr_001/provenance" in handle
        assert len(handle["output/apr_001/provenance"].attrs["source_file_sha256"]) == 64
        assert "output/apr_001/products/spad" in handle


def test_channel_fingerprint_file_hash_includes_aux_channels(tmp_path):
    path_a = _source_file(tmp_path / "sample_a.h5")
    path_b = _source_file(tmp_path / "sample_b.h5")

    with h5py.File(path_b, "a") as handle:
        handle["raw/aux"][0, 0, 0, 0, 0, 0] += 1

    with h5py.File(path_a, "r") as handle_a, h5py.File(path_b, "r") as handle_b:
        hash_a = channel_fingerprint_file_hash(handle_a)
        hash_b = channel_fingerprint_file_hash(handle_b)

    assert hash_a["source_paths"] == ["/raw/spad", "/raw/aux"]
    assert hash_a["channel_counts"] == [2, 2]
    assert hash_a["hash"] != hash_b["hash"]


def test_write_h5_output_run_apr_shift_vector_metadata(tmp_path):
    source_path = _source_file(tmp_path / "sample_calib.h5")
    shift_vectors = np.asarray([[0.0, 0.5], [1.0, -0.25]], dtype=float)

    _, run_id = write_h5_output_run(
        source_path,
        "apr_001",
        [
            H5OutputProduct(
                "spad",
                np.ones((1, 1, 2, 3, 4, 2)),
                attrs={
                    "axis_order": "repetition,z,y,x,time_bin,detector_channel",
                    "source_data_path": "/raw/spad",
                },
            )
        ],
        tool_name="APR reassignment",
        algorithm_name="adaptive_pixel_reassignment",
        attrs={
            "source_data_path": "/raw/spad",
            "output_data_path": "/output/{run_id}/products/spad",
            "shift_vectors_path": "/output/{run_id}/intermediates/shift_vectors",
        },
        metadata={
            "shift_vectors_path": "/output/{run_id}/intermediates/shift_vectors",
        },
        parameters={
            "ref_channel": 12,
            "usf": 10,
            "reassign_mode": "nearest",
            "roi_json": [0, 10, 0, 10],
        },
        intermediates=[
            H5OutputProduct(
                "shift_vectors",
                shift_vectors,
                attrs={
                    "axes": "detector_channel,shift_component",
                    "shift_components_json": ["dy", "dx"],
                },
            )
        ],
    )

    assert run_id == "apr_001"
    with h5py.File(source_path, "r") as handle:
        run = handle["output/apr_001"]
        assert run.attrs["tool_name"] == "APR reassignment"
        assert run.attrs["algorithm_name"] == "adaptive_pixel_reassignment"
        assert run.attrs["source_data_path"] == "/raw/spad"
        assert run.attrs["output_data_path"] == "/output/apr_001/products/spad"
        assert run["metadata"].attrs["shift_vectors_path"] == (
            "/output/apr_001/intermediates/shift_vectors"
        )
        assert run["parameters"].attrs["ref_channel"] == 12
        assert json.loads(run["parameters"].attrs["parameters_json"])["usf"] == 10
        vectors = run["intermediates/shift_vectors"]
        np.testing.assert_allclose(vectors[...], shift_vectors)
        assert vectors.attrs["axes"] == "detector_channel,shift_component"
        assert json.loads(vectors.attrs["shift_components_json"]) == ["dy", "dx"]

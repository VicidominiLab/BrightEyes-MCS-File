import numpy as np

from brighteyes_mcs_file.h5_data_calibrator import H5OutputBuilder


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

"""Public API for BrightEyes MCS file utilities."""

from . import mcs
from .alignment import Alignment
from .channel_skew_estimator import estimate_channel_skew
from .h5_data_calibrator import (
    H5DataCalibrator,
    H5OutputBuilder,
    add_output_to_h5_file,
    build_h5_output,
    calibrate_h5_file,
    show_h5_structure,
    show_h5_structure_html,
)
from .graph import (
    normalize_histogram,
    plot_calibration_fit_traces,
    plot_calibration_lifetime_summary,
    plot_calibration_shift_summary,
)
from .mcs import MCSMetadata, load, metadata, metadata_load, metadata_print
from .tools_phasor import (
    estimate_lifetime_from_birfi,
    estimate_lifetime_from_circmean,
    estimate_lifetime_from_log,
)

IRF_from_data_deconvolution = Alignment.IRF_from_data_deconvolution
curve_fit_circular = Alignment.curve_fit_circular
fit_data_with_ref_or_irf = Alignment.fit_data_with_ref_or_irf
fit_maps_to_stack = Alignment.fit_maps_to_stack
fit_model_data = Alignment.fit_model_data
generate_fit_maps = Alignment.generate_fit_maps
hist_for_plot = Alignment.hist_for_plot
linear_shift = Alignment.linear_shift
model_data = Alignment.model_data
perform_fit_data = Alignment.perform_fit_data
phasor_delay_from_hist = Alignment.phasor_delay_from_hist
rectangular_IRF = Alignment.rectangular_IRF
sum_channel_applying_shifts = Alignment.sum_channel_applying_shifts
centroid = Alignment.centroid
clean_irf = Alignment.clean_irf
clean_irf_stack = Alignment.clean_irf_stack

__all__ = [
    "Alignment",
    "H5DataCalibrator",
    "H5OutputBuilder",
    "IRF_from_data_deconvolution",
    "MCSMetadata",
    "add_output_to_h5_file",
    "build_h5_output",
    "calibrate_h5_file",
    "centroid",
    "clean_irf",
    "clean_irf_stack",
    "curve_fit_circular",
    "estimate_channel_skew",
    "estimate_lifetime_from_birfi",
    "estimate_lifetime_from_circmean",
    "estimate_lifetime_from_log",
    "fit_data_with_ref_or_irf",
    "fit_maps_to_stack",
    "fit_model_data",
    "generate_fit_maps",
    "hist_for_plot",
    "linear_shift",
    "load",
    "mcs",
    "metadata",
    "metadata_load",
    "metadata_print",
    "model_data",
    "normalize_histogram",
    "perform_fit_data",
    "phasor_delay_from_hist",
    "plot_calibration_fit_traces",
    "plot_calibration_lifetime_summary",
    "plot_calibration_shift_summary",
    "rectangular_IRF",
    "show_h5_structure",
    "show_h5_structure_html",
    "sum_channel_applying_shifts",
]

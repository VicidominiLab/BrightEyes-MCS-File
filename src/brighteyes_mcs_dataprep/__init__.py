"""Data preparation utilities for BrightEyes MCS HDF5 files."""

from __future__ import annotations

from importlib import import_module

_OPTIONAL_EXTRA = "brighteyes-mcs-dataprep[full]"
_OPTIONAL_DEPENDENCIES = {
    "joblib",
    "matplotlib",
    "scipy",
    "torch",
    "tqdm",
}

_ALIGNMENT_METHODS = {
    "IRF_from_data_deconvolution",
    "centroid",
    "clean_irf",
    "clean_irf_stack",
    "curve_fit_circular",
    "fit_data_with_ref_or_irf",
    "fit_maps_to_stack",
    "fit_model_data",
    "generate_fit_maps",
    "hist_for_plot",
    "linear_shift",
    "model_data",
    "perform_fit_data",
    "phasor_delay_from_hist",
    "rectangular_IRF",
    "sum_channel_applying_shifts",
}

_LAZY_MODULES = {
    "H5DataCalibrator": "h5_data_calibrator",
    "H5OutputBuilder": "h5_data_calibrator",
    "H5OutputProduct": "h5_output_writers",
    "add_output_to_h5_file": "h5_data_calibrator",
    "build_h5_output": "h5_data_calibrator",
    "calibrate_h5_file": "h5_data_calibrator",
    "estimate_channel_skew": "channel_skew_estimator",
    "estimate_lifetime_from_birfi": "tools_phasor",
    "estimate_lifetime_from_circmean": "tools_phasor",
    "estimate_lifetime_from_log": "tools_phasor",
    "normalize_histogram": "graph",
    "plot_calibration_fit_traces": "graph",
    "plot_calibration_lifetime_summary": "graph",
    "plot_calibration_shift_summary": "graph",
    "show_h5_structure": "h5_data_calibrator",
    "show_h5_structure_html": "h5_data_calibrator",
    "write_h5_output_run": "h5_output_writers",
}

__all__ = [
    "Alignment",
    "H5DataCalibrator",
    "H5OutputBuilder",
    "H5OutputProduct",
    "IRF_from_data_deconvolution",
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
    "write_h5_output_run",
]


def __getattr__(name: str):
    if name == "Alignment":
        module = _import_lazy_module("alignment")
        value = module.Alignment

    elif name in _ALIGNMENT_METHODS:
        module = _import_lazy_module("alignment")
        value = getattr(module.Alignment, name)

    elif name in _LAZY_MODULES:
        module = _import_lazy_module(_LAZY_MODULES[name])
        value = getattr(module, name)

    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def _import_lazy_module(module_name: str):
    try:
        return import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name in _OPTIONAL_DEPENDENCIES:
            msg = (
                f"{module_name!r} requires optional dependencies; install "
                f"{_OPTIONAL_EXTRA} to use this API"
            )
            raise ImportError(msg) from exc
        raise

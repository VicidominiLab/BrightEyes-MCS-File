# BrightEyes MCS File

Python utilities for reading and calibrating BrightEyes MCS HDF5 files.

Install `brighteyes-mcs-file` from PyPI:

    pip install brighteyes-mcs-file

You can also install the latest development version directly from GitHub:

    pip install git+https://github.com/VicidominiLab/BrightEyes-MCS-File

In case of local development:
    
    git clone https://github.com/VicidominiLab/BrightEyes-MCS-File.git
    cd BrightEyes-MCS-File
    pip install -e .


## Install for Development

```bash
pip install -e .[fit]
```

## Basic Usage

```python
from brighteyes_mcs_file import (
    Alignment,
    calibrate_h5_file,
    metadata_load,
    plot_calibration_lifetime_summary,
    show_h5_structure_html,
    sum_channel_applying_shifts,
)

metadata = metadata_load("data.h5")
output_path = calibrate_h5_file("data.h5", "reference.h5", create_output=True)
show_h5_structure_html(output_path)
```

Set `create_output=False` to write only the calibration results without the
derived `/output` analysis group.

Calibrated files are written with the BrightEyes MCS 0.0.6 root layout. The
root contains `/raw`, `/calibration`, `/output`, and optional `/thumbnail`;
legacy input `/data` is copied to `/raw/spad`, legacy
`/data_channels_extra` is copied to `/raw/aux`, and old configuration groups
are preserved under `/raw/legacy`. Routine analysis metadata is normalized
under `/raw/metadata`, `/calibration/metadata`, and `/output/<run>/metadata`.
Per-channel virtual datasets are grouped by source kind under
`/output/virtual_channels/<kind>/channel_<index>` for `spad`, `aux`, and
`analog` payloads. Use `load_raw(...)` for `/raw/spad`, `/raw/aux`, or
`/raw/analog`, and use `load_virtual_channel(...)` instead of hard-coding
virtual channel paths.

## Custom Fit Models

By default, fits use the historical single-exponential model with parameters
`C`, `dT`, and `tau`. To fit a user-defined model, pass a callable that returns
the complete fitted histogram:

```python
def biexponential_model(t, irf, period, C, dT, a1, tau1, tau2):
    first = Alignment.fit_model_data(t, 1.0, dT, tau1, irf, period)
    second = Alignment.fit_model_data(t, 1.0, dT, tau2, irf, period)
    model = a1 * first / first.sum() + (1.0 - a1) * second / second.sum()
    return C * model / model.sum()

result = calibrate_h5_file(
    "data.h5",
    "reference.h5",
    model_fn=biexponential_model,
    p0=[1.0, 0.0, 0.5, 1.5, 4.0],
    bounds=([0.0, -45.5, 0.0, 0.01, 0.01], [float("inf"), 45.5, 1.0, 25.0, 25.0]),
    parameter_names=["C", "dT", "a1", "tau1", "tau2"],
    lifetime_param="tau1",
)
```

Custom calibration outputs keep the legacy datasets when possible and also add
generic `fit_parameter_names`, `fit_params`, `fit_parameter_errors`, and
`fit_covariance` datasets.

The same model configuration can be used for pixel-wise fit maps:

```python
fit_maps = Alignment.generate_fit_maps(
    data_image,  # shape (y, x, t)
    irf,
    t,
    period,
    model_fn=biexponential_model,
    p0=[1.0, 0.0, 0.5, 1.5, 4.0],
    bounds=([0.0, -45.5, 0.0, 0.01, 0.01], [float("inf"), 45.5, 1.0, 25.0, 25.0]),
    parameter_names=["C", "dT", "a1", "tau1", "tau2"],
    lifetime_param="tau1",
)

fit_stack, fit_stack_names = Alignment.fit_maps_to_stack(fit_maps)
```

For the built-in single-exponential map fit, `C` remains fixed to `1.0` by
default. Custom map fits estimate all parameters by default; pass
`force_C_normalized=True` if a custom model's `C` parameter should also be
fixed.

The package includes a local `mcs` module, so calibration no longer imports
`brighteyes_ism.dataio.mcs`.

The HDF5 calibration workflow notebook is in
`examples/Calibrate_h5_file_Workflow.ipynb`. Install notebook-only helpers with
`pip install -e .[notebook]`.

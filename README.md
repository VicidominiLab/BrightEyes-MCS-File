# BrightEyes MCS File

Python utilities for reading and calibrating BrightEyes MCS HDF5 files.

This package is structured with a `src/` layout so it can be published on PyPI
as `brighteyes-mcs-file` and imported as `brighteyes_mcs_file`.

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
output_path = calibrate_h5_file("data.h5", "reference.h5")
show_h5_structure_html(output_path)
```

The package includes a local `mcs` module, so calibration no longer imports
`brighteyes_ism.dataio.mcs`.

The HDF5 calibration workflow notebook is in
`examples/Calibrate_h5_file_Workflow.ipynb`. Install notebook-only helpers with
`pip install -e .[notebook]`.

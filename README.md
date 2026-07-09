# BrightEyes-MCS-DataPrep

Calibration and data-preparation utilities for BrightEyes MCS HDF5 files.

`brighteyes-mcs-dataprep` contains the writer-side and processing helpers for
BrightEyes MCS data. It calibrates FLIM histograms, estimates channel skew,
creates schema-compatible `/output` groups, appends derived analysis products,
and provides plotting and HDF5 inspection helpers.

Reader functionality is intentionally kept in
[`BrightEyes-MCS-Reader`](https://github.com/VicidominiLab/BrightEyes-MCS-Reader)
and is imported from `brighteyes_mcs_reader` where needed.

## Install

```bash
pip install brighteyes-mcs-dataprep
```

For local development:

```bash
pip install -e .
pytest
```

## Usage

### Calibrate a file

```python
from brighteyes_mcs_dataprep import calibrate_h5_file

calibrated_path = calibrate_h5_file(
    "measurement.h5",
    "reference.h5",
    data_key=("data", "data_channels_extra"),
    reference_key=None,
    reference_type="ref",
    period_ns=12.5,
)
```

By default the calibrated copy is written next to the input file with a
`_calib.h5` suffix. Calibration results are stored under
`/calibration/results/<product>/`, and the `/output` group is created unless
`create_output=False` is passed.

### Build standard outputs

```python
from brighteyes_mcs_dataprep import build_h5_output

build_h5_output(
    "measurement_calib.h5",
    create_virtual_channels=True,
    create_sum_channels=True,
    create_sum_channels_with_skew_correction=True,
)
```

The output builder creates virtual per-channel datasets and summed SPAD/AUX
products under `/output`. It reads current schema paths such as `/raw/spad` and
`/raw/aux` by default. Legacy root-level datasets can be selected explicitly,
for example `spad_data_key="data"` or `aux_data_key="data_channels_extra"`.

### Append analysis products

```python
import numpy as np

from brighteyes_mcs_dataprep import H5OutputProduct, write_h5_output_run

written_path, run_id = write_h5_output_run(
    "measurement_calib.h5",
    "apr_001",
    [
        H5OutputProduct(
            "spad",
            np.ones((256, 256)),
            attrs={
                "data_role": "image",
                "axis_order": "y,x",
                "source_data_path": "/raw/spad",
                "units": "counts",
            },
        )
    ],
    tool_name="APR reassignment",
    algorithm_name="adaptive_pixel_reassignment",
    set_default=True,
)
```

Existing run IDs are versioned automatically, so a second write to `apr_001`
becomes `apr_002` unless `output_key_overwrite=True` is passed.

## Public API

Common imports are exposed directly from `brighteyes_mcs_dataprep`:

```python
from brighteyes_mcs_dataprep import (
    H5DataCalibrator,
    H5OutputBuilder,
    H5OutputProduct,
    build_h5_output,
    calibrate_h5_file,
    estimate_channel_skew,
    show_h5_structure,
    write_h5_output_run,
)
```

Alignment and plotting helpers are also available through `Alignment` or as
lazy top-level exports.

## HDF5 Schema Notes

Current BrightEyes files store measured data under `/raw`, metadata under
`/raw/metadata`, calibration artifacts under `/calibration`, and derived
analysis runs under `/output/<run_id>`. The data-preparation helpers preserve
this layout and add provenance attributes, source-file hashes, and default
output pointers where appropriate.

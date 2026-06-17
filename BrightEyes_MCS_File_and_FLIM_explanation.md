# BrightEyes-MCS-File and BrightEyes-FLIM: Conceptual Overview

This note explains the role of the two projects, how they connect, the fitting
model used for calibration and lifetime extraction, and the typical workflow
shown in the notebooks. The emphasis is on the ideas; function names are
mentioned only where they clarify the workflow.

## 1. Big Picture

BrightEyes-MCS-File and BrightEyes-FLIM are two layers of the same FLIM data
analysis problem.

BrightEyes-MCS-File is the HDF5 and calibration layer. It knows how to read
BrightEyes MCS files, interpret the multi-dimensional photon histograms, compare
the data with a reference acquisition, estimate channel timing corrections, and
write the calibration products back into an HDF5 file.

BrightEyes-FLIM is the FLIM analysis and visualization layer. It provides phasor
analysis, lifetime map plotting, histogram summaries, equalized color mapping,
and legacy convenience containers. In the current code it also reuses and
re-exports many routines from BrightEyes-MCS-File, especially the calibration,
alignment, IRF, and fit routines.

In abstract terms:

```text
Microscope acquisition
    |
    v
Raw BrightEyes MCS HDF5 file
    - dimensions: repetition, z, y, x, time-bin, detector-channel
    - contains photon arrival histograms for every pixel and channel
    |
    v
BrightEyes-MCS-File
    - reads metadata and datasets
    - builds the time axis
    - calibrates detector channels against reference or IRF data
    - estimates timing delays and channel skew
    - writes calibration products into /calibration/...
    |
    v
Calibrated HDF5 file
    - per-channel fitted lifetime/delay/error
    - aligned reference/IRF histograms
    - channel skew vector
    - fit traces useful for quality control
    |
    v
BrightEyes-FLIM
    - aligns and sums detector channels
    - computes phasors or per-pixel lifetime fits
    - produces lifetime maps, histograms, and summaries
```

The two projects therefore do not represent two unrelated analyses. The MCS-file
project prepares physically meaningful, calibrated time histograms; the FLIM
project turns those calibrated histograms into interpretable lifetime images.

## 2. Data Model

The central object is a photon-count histogram. For each pixel and detector
channel, the instrument stores counts over a finite number of time bins inside
one laser repetition period.

The usual HDF5 dataset layout is:

```text
(repetition, z, y, x, time_bin, channel)
```

For example:

```text
(1, 1, 501, 501, 91, 25)
```

This means:

- one repetition,
- one z plane,
- a 501 by 501 image,
- 91 temporal bins per pixel,
- 25 detector channels.

Each detector channel can have a slightly different temporal response. If the
channels are simply summed without correction, the decay is broadened or shifted.
That broadening contaminates the lifetime estimate. Calibration is therefore
not optional: it is the step that turns many channel-specific clocks into a
single consistent temporal coordinate.

## 3. What Calibration Tries To Solve

A measured fluorescence decay is not just the sample lifetime. It is the sample
decay blurred by the instrument response and shifted by timing offsets:

```text
measured_data(t)
    ~= IRF(t - delay) convolved with periodic_exponential_decay(t, tau)
```

Where:

- `IRF` is the instrument response function.
- `tau` is the fluorescence lifetime to estimate.
- `delay` is the timing offset between the data and the reference/IRF.
- The exponential is periodic because the laser repeats every period `T`.
- The data are photon counts, so the noise is naturally Poisson-like.

The calibration fit estimates three main quantities per channel:

```text
C      normalized amplitude or scale
dT     temporal delay, in time-bin units
tau    fitted fluorescence lifetime, in ns
```

The fitted delay is then used to build corrected reference/IRF stacks. A second
correction, called `channel_skew`, estimates how detector channels should be
shifted before summing them into one high-count decay.

## 4. The Fit Model

The model used by the alignment code is a periodic mono-exponential decay
convolved with an IRF.

Conceptually:

```text
E_tau(t) = periodic single-exponential decay with lifetime tau
M(t)     = convolution(IRF, E_tau)
Fit      = choose C, dT, tau so that C * M(t, dT, tau) matches the data
```

A simple non-periodic exponential would be incomplete because FLIM data are
acquired inside a repeated laser cycle. If the lifetime is not negligible
compared with the laser period, photons from one excitation cycle can wrap into
the next cycle. The model therefore uses a periodic exponential over the laser
period.

The IRF is also essential. The instrument and detector do not respond as a delta
function; they have finite timing width and channel-dependent delay. Fitting the
raw decay without the IRF would attribute part of the instrument response to the
sample lifetime.

The fit output should be interpreted as:

```text
tau:
    sample lifetime under a mono-exponential assumption

dT:
    common temporal delay between the data decay and the IRF/reference model

C:
    normalized amplitude. In many per-pixel fits this can be fixed to 1 because
    the shape of the decay, not the total brightness, is the quantity of interest.
```

## 5. Main Fit Cases

### Case A: Reference Fluorophore, Known Lifetime

This is the case used in the calibration notebook:

```text
REFERENCE_TYPE = "ref"
TAU_REF = 2.5 ns
```

Here the reference file is not a pure IRF. It is a fluorescence reference sample
with a known lifetime. The measured reference decay is itself the result of:

```text
reference_measured(t)
    ~= IRF(t) convolved with exponential(tau_ref)
```

So the code first estimates the IRF by deconvolving the known reference decay
from the measured reference histogram. After that, the estimated IRF is used to
fit the sample data.

Scheme:

```text
Known reference lifetime tau_ref
        +
Measured reference histogram
        |
        v
Estimate IRF by deconvolution
        |
        v
Fit sample channel:
    sample histogram ~= IRF (*) exponential(tau)
        |
        v
Store tau, delay, IRF, fitted trace, fit error
```

This case is useful when a stable fluorescent standard is easier to acquire than
a true instrument-response measurement.

### Case B: Reference Fluorophore, Unknown Lifetime

If `tau_ref` is not supplied, the code can estimate it from the reference
histogram before estimating the IRF.

Available conceptual strategies are:

- log-linear decay estimate: select a falling part of the decay and fit the log
  of the signal.
- circular mean estimate: estimate mean arrival delay inside the laser period.
- BIRFI-like estimate: detect a decay window using derivatives and estimate a
  centroid lifetime.

This case is less controlled than a known reference lifetime. It is useful when
the reference lifetime is not entered manually, but the result depends more
strongly on the reference trace quality and on whether the reference is really
close to mono-exponential.

### Case C: Direct IRF Input

```text
REFERENCE_TYPE = "irf"
```

In this case the reference file is treated as an already measured IRF. No
reference lifetime is needed and no deconvolution from a fluorescent standard is
performed.

Scheme:

```text
Measured IRF
    |
    v
Fit sample channel directly:
    sample histogram ~= IRF (*) exponential(tau)
```

This is the cleaner conceptual case if a reliable IRF acquisition is available.
The practical difficulty is that experimental IRFs can be noisy or contaminated,
so the workflow can optionally clean the IRF by keeping only the useful temporal
window around the response peak.

### Case D: Shift The Model Or Shift The IRF

The code supports two closely related timing conventions:

```text
model_shift:
    shift the exponential model relative to the IRF

irf_shift:
    shift the IRF relative to the exponential model
```

Both express the same physical idea: data and response must be brought into the
same temporal frame. The sign and numerical convention differ. Calibration of
whole HDF5 files defaults to `model_shift`; per-pixel lifetime maps in the FLIM
notebooks often use `irf_shift` with an already prepared summed IRF.

The important abstract point is not which object is shifted, but that the fit
must include a delay parameter. Without it, a timing mismatch can be incorrectly
absorbed into the lifetime.

### Case E: Likelihood Fit Versus Curve-Fit Modes

The preferred backend is:

```text
fit_type = "likelihood"
```

This uses a Poisson likelihood/deviance idea. That is appropriate because each
time bin contains photon counts. It is especially important for per-pixel fits,
where counts can be low and Gaussian least-squares assumptions are weak.

Other modes are available:

```text
curve_fit_circular
curve_fit
```

These are weighted least-squares style fits. They are useful for compatibility
with older behavior and for high-count traces where Gaussian approximations are
more acceptable. The circular variant is aware that time bins live on a repeated
laser period, so delays near the boundary are handled more naturally.

### Case F: Global Channel Calibration Versus Per-Pixel Fitting

There are two different scales of fitting:

```text
Global per-channel calibration:
    sum all pixels for one detector channel
    fit a high-count channel histogram
    estimate channel delay, IRF/reference alignment, and channel skew

Per-pixel FLIM fitting:
    after calibration, align and sum channels
    fit each pixel histogram
    produce lifetime maps
```

The global fit is about instrument correction. The per-pixel fit is about sample
contrast and biological/physical interpretation.

## 6. Channel Skew And Channel Summation

Detector channels are separate timing paths. Even after estimating the common
delay for a channel, the channels still need a relative alignment before they
are summed.

The calibration file stores a `channel_skew` vector. This vector tells the
analysis how much each channel should be shifted, in fractional time-bin units,
before summing.

Conceptually:

```text
channel 0 histogram -- shift by skew[0] --\
channel 1 histogram -- shift by skew[1] ----> sum -> one corrected histogram
channel 2 histogram -- shift by skew[2] --/
...
```

The summation is conservative: photon counts are redistributed between
neighboring time bins using interpolation weights, and then the channels are
summed. This preserves the total number of photons while giving sub-bin timing
precision.

## 7. What Gets Written Into The Calibrated HDF5 File

The calibration output is a copy of the input HDF5 file with an added
`/calibration` group. For each calibrated product, such as `spad` or `aux`,
the file stores products like:

```text
/calibration/results/spad/channels/index
/calibration/results/spad/channels/reference_mask
/calibration/results/spad/timing/channel_skew_bins
/calibration/results/spad/timing/channel_skew_err_bins
/calibration/results/spad/timing/delay_correction_bins
/calibration/results/spad/timing/delay_correction_ns
/calibration/results/spad/fit/fitted_delay_bins
/calibration/results/spad/fit/fitted_delay_ns
/calibration/results/spad/fit/amplitude
/calibration/results/spad/fit/amplitude_err
/calibration/results/spad/fit/tau_ns
/calibration/results/spad/fit/tau_err_ns
/calibration/results/spad/fit/tau_reference_ns
/calibration/results/spad/fit/measured_trace
/calibration/results/spad/fit/reference_trace
/calibration/results/spad/fit/irf_trace
/calibration/results/spad/fit/fitted_trace
/calibration/results/spad/fit/residual_error
/calibration/results/spad/aligned/irf_trace
/calibration/results/spad/aligned/reference_trace
```

The useful distinction is:

```text
fit/fitted_delay_*:
    raw delay estimated by fitting each channel

timing/delay_correction_*:
    delay actually used to realign IRF/reference stacks
    this may be the median delay across channels or the channel-specific delay

timing/channel_skew_*:
    relative channel-to-channel timing correction used when summing channels

fit/measured_trace, fit/reference_trace, fit/irf_trace, fit/fitted_trace:
    diagnostic traces for checking whether the fit is physically reasonable
```

## 8. Typical Calibration Workflow In The Notebook

The notebook `examples/Calibrate_h5_file_Workflow.ipynb` shows the calibration
step.

Its structure is:

```text
1. Select datasets
       DATA_KEY = ("data", "data_channels_extra")

2. Select files
       FILE_DATA      = sample acquisition
       FILE_REFERENCE = reference acquisition

3. Choose calibration physics
       REFERENCE_TYPE = "ref"
       TAU_REF        = 2.5
       FIT_MODE       = "model_shift"
       FIT_TYPE       = "likelihood"

4. Choose channel-skew settings
       channel skew from reference histograms
       reference channel = 12

5. Run HDF5 calibration
       calibrate_h5_file(...)

6. Read calibration tables
       build a dataframe of tau, delay, skew, fit error, etc.

7. Inspect quality
       plot lifetime summary by channel
       plot shift summary by channel
       inspect one channel trace:
           data, reference, IRF, aligned IRF, fitted data

8. Inspect HDF5 structure
       verify what was written under /calibration
```

This notebook is not primarily a lifetime-map notebook. It is a calibration
quality-control notebook. Its goal is to answer:

- Did the reference and data files match?
- Did the per-channel fits converge?
- Are fitted lifetimes reasonable?
- Are delays and channel skew stable across channels?
- Does the fitted trace reproduce the measured decay?
- Did the calibrated HDF5 file receive the expected calibration datasets?

## 9. Typical FLIM Application Workflow

The FLIM fit notebooks start from a calibrated HDF5 file. The practical
application is to transform raw multi-channel detector data into a lifetime map
of the sample.

Scheme:

```text
Calibrated HDF5 file
    |
    | read:
    |   data
    |   channel_skew
    |   aligned_irf_trace_aligned
    v
Clean or prepare IRF
    |
    v
Apply channel skew and sum channels
    |
    v
For one pixel:
    fit histogram to check model and initial guesses
    |
    v
For all pixels:
    fit every pixel histogram
    |
    v
Maps:
    intensity map
    tau lifetime map
    delay map
    fit uncertainty maps
    |
    v
Visualization:
    lifetime image
    thresholded lifetime histogram
    equalized lifetime color display
```

The usual per-pixel fit uses the calibrated, summed IRF and a mono-exponential
model. Pixels with too few counts can be skipped or later removed with an
intensity threshold. The result is a lifetime image where each pixel color
represents `tau`, often displayed together with intensity so dim noisy pixels do
not dominate the interpretation.

In biological imaging, this is used to reveal contrast that is not visible in
intensity alone. Two regions may have similar brightness but different decay
times because the fluorophore environment, binding state, FRET state, pH,
viscosity, or molecular composition differs.

## 10. Phasor Analysis In BrightEyes-FLIM

BrightEyes-FLIM also supports phasor analysis. A phasor is the Fourier
representation of the decay histogram at a selected harmonic.

Conceptually:

```text
histogram over time bins
    |
    v
first Fourier harmonic
    |
    v
complex phasor = g + i*s
```

The phasor approach maps decay shapes to points in a 2D plane. For ideal
single-exponential decays, points lie on the universal semicircle. More complex
mixtures appear inside the circle. This makes phasors useful for exploratory
analysis because they do not require explicitly fitting every pixel to a chosen
exponential model.

The two approaches answer related but different questions:

```text
Fit-based FLIM:
    "What lifetime tau best explains this histogram under my model?"

Phasor FLIM:
    "Where does this decay shape lie in Fourier/phasor space?"
```

Fit-based analysis gives direct parameter maps and uncertainties, but it depends
on the model. Phasor analysis is fast and visual, but interpretation is more
geometric and often less explicit unless the sample follows known phasor
relationships.

## 11. Practical Interpretation Of The Different Outputs

Good calibration normally shows:

- fitted reference/sample traces overlapping well in the diagnostic plot,
- finite `tau_ns` and reasonable `tau_err_ns` for most channels,
- delays that vary smoothly or consistently across channels,
- a small fit error,
- a channel skew vector that improves the summed IRF/data sharpness.

Suspicious calibration can show:

- failed or NaN fits on several channels,
- very large uncertainty,
- a fitted lifetime far from the expected reference/sample range,
- an IRF with broad tails or multiple peaks,
- channel skew that makes the summed decay worse rather than sharper,
- strong disagreement between `measured_trace` and `fitted_trace`.

When that happens, the problem is often not the plotting step. It is usually one
of these conceptual issues:

- the reference file is not the correct reference for the data file,
- `reference_type` is wrong (`ref` used where the file is really an IRF, or the
  opposite),
- `tau_ref` is wrong for the reference sample,
- the laser period or number of bins is wrong,
- the signal is too low in some channels,
- the IRF/reference trace is contaminated or poorly aligned,
- the mono-exponential assumption is too simple for the sample.

## 12. Short Summary

BrightEyes-MCS-File answers:

```text
How do I turn raw multi-channel BrightEyes HDF5 data into calibrated, aligned,
physically meaningful time histograms?
```

BrightEyes-FLIM answers:

```text
How do I turn calibrated FLIM histograms into lifetime/phasor maps and useful
visual summaries?
```

The fit is the bridge between the two. It separates sample lifetime from
instrument response and timing delay. The calibration notebook uses this bridge
to create a calibrated HDF5 file; the FLIM notebooks then use that file to make
per-pixel lifetime maps and phasor/lifetime visualizations.

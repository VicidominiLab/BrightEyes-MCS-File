"""Reader helpers for BrightEyes MCS HDF5 files."""

import re
import warnings

import h5py
import numpy as np

__all__ = ["MCSMetadata", "metadata", "metadata_load", "metadata_print", "load"]


DATASET_KEY_ALIASES = {
    "data": ("raw/spad", "data"),
    "spad": ("raw/spad", "data"),
    "raw/spad": ("raw/spad",),
    "data_channels_extra": ("raw/aux", "data_channels_extra"),
    "aux": ("raw/aux", "data_channels_extra"),
    "raw/aux": ("raw/aux",),
    "data_analog": ("raw/analog", "data_analog"),
    "analog": ("raw/analog", "data_analog"),
    "raw/analog": ("raw/analog",),
}


def _attrs_for_first_group(h5_file, paths):
    for path in paths:
        key = str(path).strip("/")
        if key in h5_file and isinstance(h5_file[key], h5py.Group):
            return h5_file[key].attrs
    return {}


def _get_attr(attrs, key, default=None):
    try:
        return attrs.get(key, default)
    except AttributeError:
        return default


def _get_attr_any(attrs, keys, default=None):
    for key in keys:
        value = _get_attr(attrs, key, None)
        if value is not None:
            return value
    return default


def _as_float(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _calibrated_offset_um(attrs, axis, calibration_um_per_v):
    offset_um = _as_float(_get_attr(attrs, f"offset_{axis}_um", np.nan))
    if np.isfinite(offset_um):
        return offset_um

    offset_v = _as_float(_get_attr(attrs, f"offset_{axis}", np.nan))
    if np.isfinite(offset_v) and np.isfinite(calibration_um_per_v):
        return offset_v * calibration_um_per_v
    return np.nan


def _spacing_from_range(range_um, count):
    range_um = _as_float(range_um, np.nan)
    count = _as_float(count, np.nan)
    if not np.isfinite(range_um) or not np.isfinite(count) or count <= 0:
        return np.nan
    if count == 1:
        return 0.0
    return range_um / (count - 1)


def _resolve_dataset(h5_file, key):
    candidates = DATASET_KEY_ALIASES.get(str(key).strip("/"), (str(key).strip("/"),))
    for candidate in candidates:
        if candidate in h5_file and isinstance(h5_file[candidate], h5py.Dataset):
            return h5_file[candidate]
    tried = ", ".join(candidates)
    raise KeyError(f"could not find dataset {key!r}; tried {tried}")


class MCSMetadata:
    """Metadata read from a BrightEyes MCS HDF5 file."""

    def __init__(self, h5_file):
        self.version = h5_file.attrs.get("data_format_version", "unknown")
        self.comment = h5_file.attrs.get("comment", "")

        normalized = _attrs_for_first_group(h5_file, ("raw/metadata",))
        scan = _attrs_for_first_group(h5_file, ("raw/metadata/acquisition/scan",))
        timing = _attrs_for_first_group(h5_file, ("raw/metadata/acquisition/timing",))
        gui = _attrs_for_first_group(
            h5_file,
            ("configurationGUI", "raw/legacy/configurationGUI"),
        )
        fpga = _attrs_for_first_group(
            h5_file,
            ("configurationFPGA", "raw/legacy/configurationFPGA"),
        )

        self.calib_x = _get_attr(gui, "calib_x", np.nan)
        self.calib_y = _get_attr(gui, "calib_y", np.nan)
        self.calib_z = _get_attr(gui, "calib_z", np.nan)
        self.rangex = _get_attr_any(
            normalized,
            ("range_x_um",),
            _get_attr(scan, "range_x_um", _get_attr(gui, "range_x", np.nan)),
        )
        self.rangey = _get_attr_any(
            normalized,
            ("range_y_um",),
            _get_attr(scan, "range_y_um", _get_attr(gui, "range_y", np.nan)),
        )
        self.rangez = _get_attr_any(
            normalized,
            ("range_z_um",),
            _get_attr(scan, "range_z_um", _get_attr(gui, "range_z", np.nan)),
        )
        self.offset_x_um = _get_attr_any(
            normalized,
            ("offset_x_um",),
            _get_attr(scan, "offset_x_um", _calibrated_offset_um(gui, "x", self.calib_x)),
        )
        self.offset_y_um = _get_attr_any(
            normalized,
            ("offset_y_um",),
            _get_attr(scan, "offset_y_um", _calibrated_offset_um(gui, "y", self.calib_y)),
        )
        self.offset_z_um = _get_attr_any(
            normalized,
            ("offset_z_um",),
            _get_attr(scan, "offset_z_um", _calibrated_offset_um(gui, "z", self.calib_z)),
        )
        self.pixel_size_x_um = _get_attr(normalized, "pixel_size_x_um", _get_attr(scan, "pixel_size_x_um", np.nan))
        self.pixel_size_y_um = _get_attr(normalized, "pixel_size_y_um", _get_attr(scan, "pixel_size_y_um", np.nan))
        self.pixel_size_z_um = _get_attr(normalized, "pixel_size_z_um", _get_attr(scan, "pixel_size_z_um", np.nan))
        self.nbin = _get_attr_any(
            normalized,
            ("time_bins", "digital_time_bins", "digital_timebins"),
            _get_attr_any(
                timing,
                ("time_bins", "digital_time_bins", "digital_timebins"),
                _get_attr(gui, "timebin_per_pixel", np.nan),
            ),
        )
        self.dt = _get_attr_any(
            normalized,
            ("time_resolution_us",),
            _get_attr(timing, "time_resolution_us", _get_attr(gui, "time_resolution", np.nan)),
        )
        self.nx = _get_attr(normalized, "nx", _get_attr(gui, "nx", np.nan))
        self.ny = _get_attr(normalized, "ny", _get_attr(gui, "ny", np.nan))
        self.nz = _get_attr(normalized, "nz", _get_attr(gui, "nframe", np.nan))
        self.nrep = _get_attr(normalized, "nrep", _get_attr(gui, "nrep", np.nan))
        self.bitfile = _get_attr(gui, "bitFile", "")
        self.time_bin_ns = _get_attr_any(
            normalized,
            ("time_bin_ns", "digital_time_bin_ns", "digital_timebin_ns"),
            _get_attr_any(timing, ("time_bin_ns", "digital_time_bin_ns", "digital_timebin_ns"), np.nan),
        )
        self.pixel_dwell_time_us = _get_attr(
            normalized,
            "pixel_dwell_time_us",
            _get_attr(timing, "pixel_dwell_time_us", np.nan),
        )
        self.pixel_dwell_time_ns = _get_attr(
            normalized,
            "pixel_dwell_time_ns",
            _get_attr(timing, "pixel_dwell_time_ns", np.nan),
        )

        self._dfd_metadata_loaded = False
        self._dfd_freq = _get_attr(
            normalized,
            "laser_frequency_mhz",
            _get_attr(timing, "laser_frequency_mhz", None),
        )
        self._dfd_nbins = _get_attr_any(
            normalized,
            ("time_bins", "digital_time_bins", "digital_timebins"),
            _get_attr_any(timing, ("dfd_time_bins", "time_bins", "digital_time_bins"), None),
        )
        if self._dfd_freq is not None:
            try:
                self._dfd_freq = float(self._dfd_freq)
                self._dfd_metadata_loaded = np.isfinite(self._dfd_freq)
            except (TypeError, ValueError):
                self._dfd_freq = None
        try:
            self._dfd_nbins = int(self._dfd_nbins)
        except (TypeError, ValueError):
            self._dfd_nbins = None

        self.dfd_activate = _get_attr_any(
            normalized,
            ("dfd_active", "dfd_activate"),
            _get_attr(timing, "dfd_active", _get_attr(fpga, "DFD_Activate", False)),
        )
        self.dfd_active = self.dfd_activate
        self.dfd_trigger_selector = _get_attr(
            timing,
            "dfd_trigger_selector",
            _get_attr(fpga, "DFD_Trig_Selector", -1),
        )
        self.dfd_laser_sync_debug = _get_attr(
            timing,
            "dfd_laser_sync_debug",
            _get_attr(fpga, "DFD_LaserSyncDebug", False),
        )

    @property
    def pxdwelltime(self):
        """Pixel dwell time in microseconds."""
        value = _as_float(self.pixel_dwell_time_us, np.nan)
        if np.isfinite(value):
            return value
        return self.dt * self.nbin

    @property
    def frametime(self):
        """Frame duration in seconds."""
        return self.pxdwelltime * self.nx * self.ny / 1e6

    @property
    def framerate(self):
        """Frame rate in hertz."""
        return 1 / self.frametime

    @property
    def dx(self):
        """Pixel size along x in micrometers."""
        value = _as_float(self.pixel_size_x_um, np.nan)
        if np.isfinite(value):
            return value
        return _spacing_from_range(self.rangex, self.nx)

    @property
    def dy(self):
        """Pixel size along y in micrometers."""
        value = _as_float(self.pixel_size_y_um, np.nan)
        if np.isfinite(value):
            return value
        return _spacing_from_range(self.rangey, self.ny)

    @property
    def dz(self):
        """Pixel size along z in micrometers."""
        value = _as_float(self.pixel_size_z_um, np.nan)
        if np.isfinite(value):
            return value
        return _spacing_from_range(self.rangez, self.nz)

    @property
    def pxszizes(self):
        """Pixel sizes in z, y, x order, preserving the legacy attribute name."""
        return [self.dz, self.dy, self.dx]

    @property
    def pxsizes(self):
        """Pixel sizes in z, y, x order."""
        return self.pxszizes

    @property
    def nmicroim(self):
        """Total number of microimages read during the measurement."""
        return self.nx * self.ny * self.nz * self.nrep * self.nbin

    @property
    def ndatapoints(self):
        """Total number of transferred words."""
        return 2 * self.nmicroim

    @property
    def duration(self):
        """Measurement duration in seconds."""
        return self.nmicroim * self.dt * 1e-6

    def _load_dfd_metadata_from_bitfile_name(self):
        if self._dfd_metadata_loaded:
            return

        self._dfd_metadata_loaded = True
        self._dfd_freq, self._dfd_nbins = self.parse_dfd_metadata_from_bitfile_name(
            self.bitfile
        )

    @property
    def dfd_freq(self):
        """DFD laser cycle frequency in MHz when it can be inferred."""
        self._load_dfd_metadata_from_bitfile_name()
        return self._dfd_freq

    @property
    def dfd_nbins(self):
        """DFD histogram bin count when it can be inferred."""
        self._load_dfd_metadata_from_bitfile_name()
        return self._dfd_nbins

    @staticmethod
    def parse_dfd_metadata_from_bitfile_name(bitfile="", default_cycle_mhz=40):
        """Infer DFD metadata from a bitfile name token like ``40M91``."""
        filename = str(bitfile).replace("\\", "/").split("/")[-1]
        match = re.search(r"(?P<cycle>\d+)M(?P<bins>\d+)", filename, re.IGNORECASE)
        if not match:
            _warn_dfd_metadata_fallback(filename, default_cycle_mhz)
            return default_cycle_mhz, None

        parsed_cycle_mhz = int(match.group("cycle"))
        parsed_bins = int(match.group("bins"))
        if not (3 < parsed_cycle_mhz < 100 and 3 < parsed_bins < 1000):
            _warn_dfd_metadata_fallback(filename, default_cycle_mhz)
            return default_cycle_mhz, None

        return parsed_cycle_mhz, parsed_bins

    def Print(self):
        """Print metadata fields using the legacy method name."""
        for name, value in vars(self).items():
            print(name, end="")
            print(" " * int(14 - len(name)), end="")
            print(str(value))


metadata = MCSMetadata


def _warn_dfd_metadata_fallback(filename, default_cycle_mhz):
    warnings.warn(
        (
            "\n"
            "================ WARNING ==============\n"
            "brighteyes_mcs_file.mcs failed to extract DFD metadata "
            f"from the bitfile name ({filename!r}).\n\n"
            "Falling back to defaults:\n"
            f"  - Laser cycle frequency: {default_cycle_mhz} MHz\n"
            "  - DFD bin count: NOT set\n\n"
            "If your data was acquired in DFD mode, THESE DEFAULTS ARE VERY "
            "LIKELY WRONG and will corrupt your analysis.\n\n"
            "You must explicitly set the correct DFD parameters in your "
            "analysis code.\n"
            "==========================================="
        ),
        stacklevel=2,
    )


def metadata_load(fname):
    """Load metadata from a BrightEyes MCS HDF5 file."""
    with h5py.File(fname, "r") as h5_file:
        return MCSMetadata(h5_file)


def metadata_print(fname):
    """Print metadata from a BrightEyes MCS HDF5 file."""
    metadata_load(fname).Print()


def load(fname, key="data", data_format="numpy"):
    """Load a dataset and metadata from a BrightEyes MCS HDF5 file."""
    if data_format == "numpy":
        with h5py.File(fname, "r") as h5_file:
            return _resolve_dataset(h5_file, key)[:], MCSMetadata(h5_file)

    if data_format == "h5":
        h5_file = h5py.File(fname, "r")
        return _resolve_dataset(h5_file, key), MCSMetadata(h5_file)

    raise ValueError("data_format must be 'numpy' or 'h5'")

"""Reader helpers for BrightEyes MCS HDF5 files."""

import re
import warnings

import h5py

__all__ = ["MCSMetadata", "metadata", "metadata_load", "metadata_print", "load"]


class MCSMetadata:
    """Metadata read from a BrightEyes MCS HDF5 file."""

    def __init__(self, h5_file):
        self.version = h5_file.attrs["data_format_version"]
        self.comment = h5_file.attrs.get("comment", "")

        gui = h5_file["configurationGUI"]

        self.rangex = gui.attrs["range_x"]
        self.rangey = gui.attrs["range_y"]
        self.rangez = gui.attrs["range_z"]
        self.nbin = gui.attrs["timebin_per_pixel"]
        self.dt = gui.attrs["time_resolution"]
        self.nx = gui.attrs["nx"]
        self.ny = gui.attrs["ny"]
        self.nz = gui.attrs["nframe"]
        self.nrep = gui.attrs["nrep"]
        self.calib_x = gui.attrs["calib_x"]
        self.calib_y = gui.attrs["calib_y"]
        self.calib_z = gui.attrs["calib_z"]
        self.bitfile = gui.attrs.get("bitFile", "")

        self._dfd_metadata_loaded = False
        self._dfd_freq = None
        self._dfd_nbins = None

        try:
            self.dfd_activate = h5_file["configurationFPGA"].attrs["DFD_Activate"]
        except Exception:
            self.dfd_activate = False

    @property
    def pxdwelltime(self):
        """Pixel dwell time in microseconds."""
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
        return self.rangex / self.nx

    @property
    def dy(self):
        """Pixel size along y in micrometers."""
        return self.rangey / self.ny

    @property
    def dz(self):
        """Pixel size along z in micrometers."""
        return self.rangez / self.nz

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
            return h5_file[key][:], MCSMetadata(h5_file)

    if data_format == "h5":
        h5_file = h5py.File(fname, "r")
        return h5_file[key], MCSMetadata(h5_file)

    raise ValueError("data_format must be 'numpy' or 'h5'")

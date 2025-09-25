import argparse
import numpy as np
import matplotlib.pyplot as plt

# Try to import netCDF4; fall back to SciPy's netcdf_file without memory mapping
try:
    from netCDF4 import Dataset as NC4Dataset
    netcdf4_available = True
    Dataset = NC4Dataset
except ImportError:
    from scipy.io import netcdf_file
    netcdf4_available = False
    Dataset = lambda f: netcdf_file(f, 'r', mmap=False)


def get_scalar(var):
    """
    Safely extract a scalar from a netCDF variable.
    """
    val = var[...]
    try:
        return float(val.item())
    except Exception:
        return float(np.array(val).squeeze())


def get_units(var):
    """
    Retrieve units attribute from a netCDF variable, defaulting to 'AU'.
    """
    # netCDF4 style attributes
    try:
        if hasattr(var, 'ncattrs') and 'units' in var.ncattrs():
            unit = var.getncattr('units')
            if isinstance(unit, bytes):
                unit = unit.decode('utf-8')
            return unit
    except Exception:
        pass
    # SciPy netcdf_file style attributes
    try:
        attrs = var._attributes
        unit = attrs.get('units')
        if unit is None:
            unit = attrs.get(b'units')
        if isinstance(unit, bytes):
            unit = unit.decode('utf-8')
        if isinstance(unit, str):
            return unit
    except Exception:
        pass
    # Default fallback
    return 'AU'


def load_cdf(file_path, var_name, debug=False):
    """
    Load a CDF chromatogram file and extract time, intensity, and unit.
    """
    ds = Dataset(file_path)
    if debug:
        print("Available variables:", list(ds.variables.keys()))
    if var_name not in ds.variables:
        raise KeyError(f"Variable '{var_name}' not found in CDF file.")
    var = ds.variables[var_name]
    intensity = np.array(var[:], dtype=float)
    unit = get_units(var)

    # Sampling parameters (in seconds)
    sample_interval = get_scalar(
        ds.variables.get('actual_sampling_interval',
                         ds.variables.get('sampling_interval'))
    )
    delay = get_scalar(
        ds.variables.get('actual_delay_time',
                         ds.variables.get('delay_time',
                                          ds.variables.get('initial_offset', 0)))
    )

    # Time axis
    times = delay + np.arange(intensity.shape[-1]) * sample_interval
    times_min = times / 60.0

    if netcdf4_available:
        ds.close()
    if debug:
        print(f"Intensity shape: {intensity.shape}")
        print(f"Intensity range: {intensity.min()} to {intensity.max()}")
        print(f"Time span: {times_min.min()} to {times_min.max()} min")
        print(f"Detected unit: {unit}")
    return times_min, intensity, unit


def plot_chromatogram(times, intensity, unit, subtract_baseline=False, output=None):
    """
    Plot chromatogram without title or legend, with specified font and size.
    """
    if subtract_baseline:
        intensity = intensity - np.median(intensity)

    # Global font settings
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # Figure size in inches (23.5 cm x 10.9 cm)
    fig = plt.figure(figsize=(23.5/2.54, 10.9/2.54))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, intensity)
    ax.set_xlabel('time (min)')
    ax.set_ylabel(f'Intensity ({unit})')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
        print(f"Saved plot to {output}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot CDF chromatogram without peak markers.'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to the CDF file'
    )
    parser.add_argument(
        '-o', '--output', help='Output image file (e.g., PNG)'
    )
    parser.add_argument(
        '--var-name', default='ordinate_values',
        help='Variable name for intensity values'
    )
    parser.add_argument(
        '--subtract-baseline', action='store_true',
        help='Subtract median baseline from intensity'
    )
    parser.add_argument(
        '--debug', action='store_true', help='Print debug info'
    )
    args = parser.parse_args()

    times, intensity, unit = load_cdf(
        args.input, args.var_name, debug=args.debug
    )
    plot_chromatogram(
        times, intensity, unit,
        subtract_baseline=args.subtract_baseline,
        output=args.output
    )
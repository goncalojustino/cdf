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
    Safely extract a scalar from a netCDF variable, regardless of dimensionality.
    """
    val = var[...]
    try:
        return float(val.item())
    except Exception:
        return float(np.array(val).squeeze())


def load_cdf(file_path, var_name, debug=False):
    """
    Load a CDF chromatogram file and extract time and intensity arrays, plus peak info.
    """
    ds = Dataset(file_path)

    if debug:
        print("Available variables:", list(ds.variables.keys()))

    # Extract intensity signal
    if var_name not in ds.variables:
        raise KeyError(f"Variable '{var_name}' not found in CDF file.")
    intensity = np.array(ds.variables[var_name][:], dtype=float)

    # Sampling parameters (in seconds)
    sample_interval = get_scalar(ds.variables.get('actual_sampling_interval', ds.variables.get('sampling_interval')))
    delay = get_scalar(ds.variables.get('actual_delay_time', ds.variables.get('delay_time', ds.variables.get('initial_offset', 0))))

    # Time axis (seconds)
    times = delay + np.arange(intensity.shape[-1]) * sample_interval
    times_min = times / 60.0

    # Peak information (if available)
    peaks = {}
    if 'peak_retention_time' in ds.variables:
        rt = np.array(ds.variables['peak_retention_time'][:], dtype=float)
        height = np.array(ds.variables['peak_height'][:], dtype=float)
        names_raw = ds.variables['peak_name'][:]
        names = [bytes(n).decode('utf-8').strip('\x00').strip() for n in names_raw]
        peaks = {'rt_min': rt / 60.0, 'height': height, 'names': names}

    # Close file only if using netCDF4
    if netcdf4_available:
        ds.close()

    if debug:
        print(f"Loaded intensity array with shape {intensity.shape}")
        print(f"Intensity stats: min={intensity.min()}, max={intensity.max()}, mean={intensity.mean()}")
        print(f"Time range: {times_min.min()} to {times_min.max()} minutes")
        if peaks:
            print(f"Found {len(peaks['rt_min'])} peaks at {peaks['rt_min']} min")

    return times_min, intensity, peaks


def plot_chromatogram(times, intensity, peaks=None, annotate=True,
                      subtract_baseline=False, output=None):
    """
    Plot chromatogram with optional peak markers and baseline correction.
    """
    if subtract_baseline:
        baseline = np.median(intensity)
        intensity = intensity - baseline

    plt.figure(figsize=(10, 6))
    plt.plot(times, intensity, label='Signal')
    plt.xlabel('Time (min)')
    plt.ylabel('Intensity')
    plt.title('Chromatogram')

    if peaks and 'rt_min' in peaks and peaks['rt_min'].size > 0:
        plt.scatter(peaks['rt_min'], peaks['height'], color='red', label='Peaks')
        if annotate:
            for rt, h, name in zip(peaks['rt_min'], peaks['height'], peaks['names']):
                plt.annotate(name, (rt, h), textcoords='offset points', xytext=(0, 5), ha='center')

    plt.legend()
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=300)
        print(f"Saved plot to {output}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CDF chromatogram files consistently.')
    parser.add_argument('-i', '--input', required=True, help='Path to the CDF file')
    parser.add_argument('-o', '--output', help='Output image file (e.g., PNG). If omitted, displays interactively.')
    parser.add_argument('--var-name', default='ordinate_values',
                        help='Name of the variable storing intensity values (default: ordinate_values)')
    parser.add_argument('--no-peaks', action='store_true', help='Disable peak markers')
    parser.add_argument('--no-annotate', action='store_true', help='Disable peak annotations')
    parser.add_argument('--subtract-baseline', action='store_true',
                        help='Subtract median baseline from intensity')
    parser.add_argument('--debug', action='store_true', help='Print debug info (variables, stats)')
    args = parser.parse_args()

    times, intensity, peaks = load_cdf(args.input, args.var_name, debug=args.debug)
    if args.no_peaks:
        peaks = {}
    plot_chromatogram(times, intensity, peaks,
                      annotate=not args.no_annotate,
                      subtract_baseline=args.subtract_baseline,
                      output=args.output)
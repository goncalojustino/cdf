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


def load_cdf(file_path):
    """
    Load a CDF chromatogram file and extract time and intensity arrays, plus peak info.
    """
    ds = Dataset(file_path)

    # Extract intensity signal
    intensity = np.array(ds.variables['ordinate_values'][:], dtype=float)

    # Sampling parameters (in seconds)
    sample_interval = get_scalar(ds.variables['actual_sampling_interval'])
    delay = get_scalar(ds.variables['actual_delay_time'])

    # Time axis (seconds)
    times = delay + np.arange(intensity.shape[0]) * sample_interval
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

    return times_min, intensity, peaks


def plot_chromatogram(times, intensity, peaks=None, annotate=True, output=None):
    """
    Plot chromatogram with optional peak markers.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, intensity, label='Signal')
    plt.xlabel('Time (min)')
    plt.ylabel('Intensity')
    plt.title('Chromatogram')

    if peaks and 'rt_min' in peaks:
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
    parser.add_argument('--no-peaks', action='store_true', help='Disable peak markers')
    parser.add_argument('--no-annotate', action='store_true', help='Disable peak annotations')
    args = parser.parse_args()

    times, intensity, peaks = load_cdf(args.input)
    if args.no_peaks:
        peaks = {}
    plot_chromatogram(times, intensity, peaks, annotate=not args.no_annotate, output=args.output)
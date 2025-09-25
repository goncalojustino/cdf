import argparse
import os
import sys
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
    try:
        if hasattr(var, 'ncattrs') and 'units' in var.ncattrs():
            unit = var.getncattr('units')
            if isinstance(unit, bytes):
                unit = unit.decode('utf-8')
            return unit
    except Exception:
        pass
    try:
        attrs = var._attributes
        unit = attrs.get('units') or attrs.get(b'units')
        if isinstance(unit, bytes):
            unit = unit.decode('utf-8')
        if isinstance(unit, str):
            return unit
    except Exception:
        pass
    return 'AU'


def load_cdf(file_path, var_name, debug=False):
    ds = Dataset(file_path)
    if debug:
        print("Available variables:", list(ds.variables.keys()))
    if var_name not in ds.variables:
        raise KeyError(f"Variable '{var_name}' not found in CDF file.")
    var = ds.variables[var_name]
    intensity = np.array(var[:], dtype=float)
    unit = get_units(var)

    sample_interval = get_scalar(
        ds.variables.get('actual_sampling_interval', ds.variables.get('sampling_interval'))
    )
    delay = get_scalar(
        ds.variables.get('actual_delay_time', ds.variables.get('delay_time', ds.variables.get('initial_offset', 0)))
    )

    times = delay + np.arange(intensity.shape[-1]) * sample_interval
    times = times / 60.0

    if netcdf4_available:
        ds.close()
    if debug:
        print(f"Loaded {intensity.shape[0]} points from {var_name}.")
        print(f"Time range: {times.min():.2f} to {times.max():.2f} min")
        print(f"Intensity range: {intensity.min():.2f} to {intensity.max():.2f} {unit}")
    return times, intensity, unit


def plot_and_save(times, intensity, unit, script_name, input_name, subtract_baseline):
    # baseline subtraction
    if subtract_baseline:
        intensity = intensity - np.median(intensity)

    # font
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # create figure with constrained_layout for insets
    fig = plt.figure(figsize=(23.5/2.54, 10.9/2.54), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(times, intensity)

    # highest peak
    idx_max = np.argmax(intensity)
    tr_max = times[idx_max]
    h_max = intensity[idx_max]

    # annotate retention time
    dx = (times.max() - times.min()) * 0.005
    ax.annotate(f'tr = {tr_max:.2f} min',
                xy=(tr_max + dx, h_max), xytext=(5, -5), textcoords='offset points',
                ha='left', va='top')

    # inset zoom ±1.0 min
    inset_pos = [0.1, 0.6, 0.25, 0.25]  # left, bottom, width, height
    ax_inset = fig.add_axes(inset_pos)
    mask = (times >= tr_max - 1.0) & (times <= tr_max + 1.0)
    zoom_t = times[mask]
    zoom_i = intensity[mask]
    ax_inset.plot(zoom_t, zoom_i)
    ax_inset.set_xlim(tr_max - 1.0, tr_max + 1.0)
    ax_inset.set_ylim(-0.1, h_max * 0.25)
    ax_inset.set_xticks([tr_max - 1.0, tr_max, tr_max + 1.0])
    ax_inset.set_xticklabels([f"{x:.1f}" for x in [tr_max - 1.0, tr_max, tr_max + 1.0]])
    ax_inset.tick_params(axis='both', which='major', labelsize=10)
    ax_inset.set_aspect('equal', 'box')

    # labels
    ax.set_xlabel('time (min)')
    ax.set_ylabel(f'Intensity ({unit})')
    ax.tick_params(axis='both', which='major', labelsize=12)

    # output svg file
    basename = os.path.splitext(os.path.basename(input_name))[0]
    script_base = os.path.splitext(os.path.basename(script_name))[0]
    out_name = f"{basename}_{script_base}.svg"
    fig.savefig(out_name, format='svg', dpi=300)
    print(f"Saved plot to {out_name}")


def main():
    parser = argparse.ArgumentParser(description='Plot CDF and save as SVG.')
    parser.add_argument('-i', '--input', required=True, help='CDF file path')
    parser.add_argument('--var-name', default='ordinate_values', help='Intensity variable name')
    parser.add_argument('--subtract-baseline', action='store_true', help='Subtract median baseline')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    args = parser.parse_args()

    times, intensity, unit = load_cdf(args.input, args.var_name, debug=args.debug)
    plot_and_save(times, intensity, unit, sys.argv[0], args.input, args.subtract_baseline)

if __name__ == '__main__':
    main()

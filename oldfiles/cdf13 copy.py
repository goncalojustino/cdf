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
    """Safely extract a scalar from a netCDF variable."""
    val = var[...]
    try:
        return float(val.item())
    except Exception:
        return float(np.array(val).squeeze())


def load_cdf(file_path, var_name, debug=False):
    """Load CDF file and extract time (min), intensity, and unit."""
    ds = Dataset(file_path)
    if debug:
        print("Variables in CDF:", list(ds.variables.keys()))

    if var_name not in ds.variables:
        raise KeyError(f"Variable '{var_name}' not found.")
    var = ds.variables[var_name]
    intensity = np.array(var[:], dtype=float)

    # Unit attribute
    unit = 'AU'
    try:
        unit_attr = var.getncattr('units')
        unit = unit_attr.decode() if isinstance(unit_attr, bytes) else str(unit_attr)
    except Exception:
        try:
            attrs = var._attributes
            unit_attr = attrs.get('units') or attrs.get(b'units')
            unit = unit_attr.decode() if isinstance(unit_attr, bytes) else str(unit_attr)
        except Exception:
            pass
    # Default to 'AU' if unit is blank or 'none'
    if not unit or unit.strip().lower() == 'none':
        unit = 'AU'

    # Sampling parameters (seconds)
    samp_int = get_scalar(
        ds.variables.get('actual_sampling_interval',
                         ds.variables.get('sampling_interval', 1))
    )
    delay = get_scalar(
        ds.variables.get('actual_delay_time',
                         ds.variables.get('delay_time',
                                          ds.variables.get('initial_offset', 0)))
    )
    if netcdf4_available:
        ds.close()

    # Time axis in minutes
    times = (delay + np.arange(intensity.size) * samp_int) / 60.0
    if debug:
        print(f"Time range: {times[0]:.2f}-{times[-1]:.2f} min, {intensity.size} points.")

    return times, intensity, unit


def plot_and_save(times, intensity, unit,
                  script_name, input_name,
                  subtract_baseline, runtime):
    """Plot chromatogram with marker, zoom inset, axes crossing, save as SVG and PNG."""
    if subtract_baseline:
        intensity = intensity - np.median(intensity)

    # Ensure all text Arial 12
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    # Create figure (23.5cm x 10.9cm)
    fig = plt.figure(figsize=(23.5/2.54, 10.9/2.54), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Plot signal
    ax.plot(times, intensity, color='blue')

        # Axes crossing at (0,0)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Set x-axis limit and ensure ticks include 0 and runtime
    ax.set_xlim(0, runtime)
    ticks = list(ax.get_xticks())
    # ensure 0 and runtime are ticks
    for t in [0, runtime]:
        if t not in ticks:
            ticks.append(t)
    ticks = sorted([t for t in ticks if 0 <= t <= runtime])
    ax.set_xticks(ticks)

    # Labels
    ax.set_xlim(0, runtime)
    ticks = list(ax.get_xticks())
    if runtime not in ticks:
        ticks.append(runtime)
    ticks = sorted([t for t in ticks if 0 <= t <= runtime])
    ax.set_xticks(ticks)

    # Labels
    ax.set_xlabel('time (min)')
    ax.set_ylabel(f'Intensity ({unit})')

    # Marker and annotation at highest peak
    idx = np.argmax(intensity)
    tr_peak = times[idx]
    h_peak = intensity[idx]
    ax.scatter([tr_peak], [h_peak], marker='x', color='red')
    dx = runtime * 0.005
    ax.annotate(f'tr = {tr_peak:.2f} min',
                xy=(tr_peak + dx, h_peak), xytext=(5, -5),
                textcoords='offset points', ha='left', va='top')

    # Zoom inset ±1.0 min around peak
    inset_pos = [0.1, 0.6, 0.25, 0.25]
    ax_ins = fig.add_axes(inset_pos)
    mask = (times >= tr_peak - 1.0) & (times <= tr_peak + 1.0)
    ax_ins.plot(times[mask], intensity[mask], color='blue')
    ax_ins.set_xlim(tr_peak - 1.0, tr_peak + 1.0)
    ax_ins.set_ylim(-0.1, h_peak * 0.25)
    ax_ins.set_xticks([tr_peak - 1.0, tr_peak, tr_peak + 1.0])
    ax_ins.set_xticklabels([f"{x:.1f}" for x in [tr_peak - 1.0, tr_peak, tr_peak + 1.0]])
    ax_ins.tick_params(axis='both', labelsize=10)
    ax_ins.set_aspect('equal', 'box')

    # Store plot limits
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    print(f"Stored x-axis limits: {x_limits}")
    print(f"Stored y-axis limits: {y_limits}")

    # Prepare output filenames
    base = os.path.splitext(os.path.basename(input_name))[0]
    scr = os.path.splitext(os.path.basename(script_name))[0]
    out_svg = f"{base}_{scr}.svg"
    out_png = f"{base}_{scr}.png"

    # Save files
    fig.savefig(out_svg, format='svg', dpi=300)
    print(f"Saved to {out_svg}")
    fig.savefig(out_png, format='png', dpi=300)
    print(f"Saved to {out_png}")


def main():
    parser = argparse.ArgumentParser(description='Plot CDF with marker and inset.')
    parser.add_argument('-i', '--input', required=True, help='CDF filepath')
    parser.add_argument('--var-name', default='ordinate_values', help='Intensity variable name')
    parser.add_argument('--subtract-baseline', action='store_true', help='Subtract median baseline')
    parser.add_argument('--runtime', type=float, default=32.0, help='X-axis max time (min)')
    parser.add_argument('--debug', action='store_true', help='Debug info')
    args = parser.parse_args()

    times, intensity, unit = load_cdf(args.input, args.var_name, args.debug)
    plot_and_save(times, intensity, unit,
                  sys.argv[0], args.input,
                  args.subtract_baseline, args.runtime)

if __name__ == '__main__':
    main()

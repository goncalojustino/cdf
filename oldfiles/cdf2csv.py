#!/usr/bin/env python3
import argparse
import os
import sys

# netCDF import (fallback to SciPy)
try:
    from netCDF4 import Dataset
except ImportError:
    from scipy.io import netcdf_file as _Dataset
    Dataset = lambda f: _Dataset(f, 'r', mmap=False)

import numpy as np
import pandas as pd

def load_peaks(cdf_path):
    """
    Load peak information from a CDF file.
    Returns arrays: rt_min, height, area, width_min.
    """
    ds = Dataset(cdf_path)
    rt_sec    = np.array(ds.variables['peak_retention_time'][:], dtype=float)
    height    = np.array(ds.variables['peak_height'][:],         dtype=float)
    area      = np.array(ds.variables['peak_area'][:],           dtype=float)
    width_sec = np.array(ds.variables['peak_width'][:],          dtype=float)
    ds.close()

    # Convert to minutes where applicable
    rt_min    = rt_sec    / 60.0
    width_min = width_sec / 60.0
    return rt_min, height, area, width_min


def main():
    parser = argparse.ArgumentParser(
        description="Extract peaks from a CDF and save to CSV with area %."
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help="Path to the CDF file."
    )
    args = parser.parse_args()

    # Build base names for output
    in_base     = os.path.splitext(os.path.basename(args.input))[0]
    script_base = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # Load peak data
    rt, ht, ar, wd = load_peaks(args.input)

    # Compute area % relative to total area
    total_area = np.sum(ar)
    area_pct   = 100 * ar / total_area if total_area != 0 else np.zeros_like(ar)

    # Build DataFrame
    df = pd.DataFrame({
        'Peak #':               np.arange(1, len(rt) + 1),
        'Retention time (min)': np.round(rt,      2),
        'Height':               np.round(ht,      2),
        'Area':                 np.round(ar,      2),
        'Area (%)':             np.round(area_pct, 2),
        'Width (min)':          np.round(wd,      3),
    })

    # Write CSV
    out_csv = f"{in_base}_{script_base}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Peak table with area % written to {out_csv}")

if __name__ == "__main__":
    main()
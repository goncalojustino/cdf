#!/usr/bin/env python3
"""
cdf13_post.py — reproduce cdf13.py main plot, add baseline and peak fills, integrate peaks.

Usage:
  python cdf13_post.py kh18.cdf
  python cdf13_post.py kh18.cdf --smooth 32 --snr 5 --noise 5000 --min-width-sec 4 --reject-height 1000 --runtime 32

Notes:
- Input is the original .cdf name. This script reads <base>_chromatogram.csv written by cdf13.py.
- Detection runs on the SMOOTHED signal (default window 32 points). Plot shows RAW trace.
- Baseline is piecewise linear, valley-to-valley, drawn only under detected peaks, behind the trace.
- Areas are ∫(raw - baseline) dt, clipped at 0. Peak %area is relative to sum of all detected peak areas.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- utilities ----------

def moving_avg(y: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return y
    k = int(max(1, w))
    ker = np.ones(k, dtype=float) / k
    return np.convolve(y, ker, mode="same")


def detect_time_col(df: pd.DataFrame) -> tuple[str, str]:
    """Return (time_column_name, unit_str)."""
    for c in df.columns:
        if c.startswith("time_"):
            return c, c.split("time_", 1)[1]
    if "time" in df.columns:
        return "time", ""
    c0 = df.columns[0]
    return c0, ""


def chrom_csv_from_cdf(cdf_path: str) -> str:
    if not cdf_path.lower().endswith(".cdf"):
        raise ValueError("Input must be a .cdf filename (e.g., kh18.cdf).")
    csv_path = os.path.splitext(cdf_path)[0] + "_chromatogram.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}. Run cdf13.py first.")
    return csv_path


def local_maxima_indices(y: np.ndarray) -> np.ndarray:
    """Indices i with y[i-1] < y[i] >= y[i+1]."""
    if y.size < 3:
        return np.array([], dtype=int)
    cond = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    return np.where(cond)[0] + 1


def nearest_left_minimum(y: np.ndarray, start_i: int) -> int:
    """Scan left for first local minimum; fallback to 0."""
    for i in range(start_i, 1, -1):
        if y[i-1] >= y[i] <= y[i+1]:
            return i
    return 0


def nearest_right_minimum(y: np.ndarray, start_i: int) -> int:
    """Scan right for first local minimum; fallback to last index."""
    n = y.size
    for i in range(start_i, n-1):
        if y[i-1] >= y[i] <= y[i+1]:
            return i
    return n - 1


def enforce_min_separation(peaks: np.ndarray, strength: np.ndarray, min_sep_pts: int) -> np.ndarray:
    """Non-maximum suppression to enforce min separation in index space."""
    if peaks.size == 0:
        return peaks
    order = np.argsort(strength[peaks])[::-1]  # strongest first
    selected = []
    taken = np.zeros(peaks.size, dtype=bool)
    for oi in order:
        p = peaks[oi]
        if taken[oi]:
            continue
        selected.append(p)
        left = p - min_sep_pts
        right = p + min_sep_pts
        for j, pj in enumerate(peaks):
            if not taken[j] and left <= pj <= right:
                taken[j] = True
    return np.array(sorted(selected), dtype=int)


def segment_baseline(t: np.ndarray, y: np.ndarray, iL: int, iR: int) -> np.ndarray:
    """Linear baseline between (t[iL], y[iL]) and (t[iR], y[iR]) evaluated on t[iL:iR+1]."""
    tL, tR = t[iL], t[iR]
    yL, yR = y[iL], y[iR]
    if iR == iL:
        return np.full(1, yL, dtype=float)
    m = (yR - yL) / (tR - tL + 1e-12)
    return yL + m * (t[iL:iR+1] - tL)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Baseline, peak fill, and integration for cdf13 output.")
    ap.add_argument("cdf", help="Original .cdf filename; reads <base>_chromatogram.csv")
    ap.add_argument("--smooth", type=int, default=32, help="moving average window (points) for detection; default 32")
    ap.add_argument("--snr", type=float, default=5.0, help="minimum S/N for peak acceptance; default 5")
    ap.add_argument("--noise", type=float, default=5000.0, help="fixed noise level in intensity units; default 5000")
    ap.add_argument("--min-width-sec", type=float, default=4.0, help="minimum peak width (valley-to-valley) in seconds; default 4")
    ap.add_argument("--reject-height", type=float, default=1000.0, help="minimum apex height for acceptance; default 1000")
    ap.add_argument("--runtime", type=float, default=32.0, help="x-axis max time for plotting (min); default 32")
    args = ap.parse_args()

    # Load chromatogram CSV
    csv_path = chrom_csv_from_cdf(args.cdf)
    df = pd.read_csv(csv_path)
    time_col, unit = detect_time_col(df)
    if "intensity" not in df.columns:
        raise ValueError(f"'intensity' column not found in {csv_path}. Columns: {list(df.columns)}")

    t = df[time_col].to_numpy(dtype=float)
    y_raw = df["intensity"].to_numpy(dtype=float)

    # sanitize
    y_raw = np.nan_to_num(y_raw, nan=0.0)
    y_raw = np.clip(y_raw, 0, None)

    # detection signal
    y_det = moving_avg(y_raw, args.smooth)

    # candidate peaks from smoothed signal
    peaks_all = local_maxima_indices(y_det)
    dt = np.median(np.diff(t)) if t.size > 1 else 1.0
    min_width_min = args.min_width_sec / 60.0
    min_sep_pts = max(1, int(round(min_width_min / max(dt, 1e-12))))
    peaks_sep = enforce_min_separation(peaks_all, y_det, min_sep_pts)

    # filters and bounds
    accepted = []
    for p in peaks_sep:
        height = float(y_raw[p])                          # height filter on RAW apex
        if height < args.reject_height:
            continue
        snr = float(y_det[p]) / max(args.noise, 1e-12)    # SNR on SMOOTHED apex
        if snr < args.snr:
            continue
        iL = nearest_left_minimum(y_det, p)               # bounds on SMOOTHED signal
        iR = nearest_right_minimum(y_det, p)
        if iR <= iL:
            continue
        if (t[iR] - t[iL]) < min_width_min:
            continue
        accepted.append((iL, p, iR))

    # resolve overlaps by truncation at previous end
    accepted.sort(key=lambda x: x[1])
    merged = []
    last_end = -1
    for iL, p, iR in accepted:
        if iL <= last_end:
            iL = max(iL, last_end)
            if iR <= iL:
                continue
        merged.append((iL, p, iR))
        last_end = iR

    # integrate areas
    peaks_rows = []
    total_area = 0.0
    for k, (iL, p, iR) in enumerate(merged, start=1):
        seg_t = t[iL:iR+1]
        seg_y = y_raw[iL:iR+1]
        base_vals = segment_baseline(t, y_raw, iL, iR)    # baseline from RAW endpoints
        y_above = np.clip(seg_y - base_vals, 0, None)
        area = float(np.trapz(y_above, seg_t))
        rt = float(t[p])
        height = float(y_raw[p])
        total_area += area
        peaks_rows.append({
            "index": k,
            "t_start_min": float(t[iL]),
            "t_end_min": float(t[iR]),
            "rt_min": rt,
            "apex_intensity": height,
            "area": area
        })

    # percentage areas
    if total_area > 0:
        for row in peaks_rows:
            row["pct_area"] = 100.0 * row["area"] / total_area
    else:
        for row in peaks_rows:
            row["pct_area"] = 0.0

    base = os.path.splitext(os.path.basename(args.cdf))[0]
    peaks_csv = f"{base}_peaks.csv"
    summary_json = f"{base}_summary.json"
    out_svg = f"{base}_with_baseline.svg"
    out_png = f"{base}_with_baseline.png"

    # --- write peaks CSV (robust if no peaks) ---
    cols = ["index","t_start_min","t_end_min","rt_min","apex_intensity","area","pct_area"]
    peaks_df = pd.DataFrame(peaks_rows, columns=cols)
    if not peaks_df.empty:
        peaks_df = peaks_df.sort_values("rt_min", kind="mergesort")
        peaks_df["t_start_min"] = peaks_df["t_start_min"].map(lambda v: f"{float(v):.4f}")
        peaks_df["t_end_min"]   = peaks_df["t_end_min"].map(lambda v: f"{float(v):.4f}")
        peaks_df["rt_min"]      = peaks_df["rt_min"].map(lambda v: f"{float(v):.4f}")
        peaks_df["area"]        = peaks_df["area"].map(lambda v: f"{float(v):.6g}")
        peaks_df["pct_area"]    = peaks_df["pct_area"].map(lambda v: f"{float(v):.3f}")
    peaks_df.to_csv(peaks_csv, index=False)

    # write summary JSON
    summary = {
        "source_csv": os.path.abspath(chrom_csv_from_cdf(args.cdf)),
        "time_unit": unit or "min",
        "n_points": int(t.size),
        "n_peaks": int(len(peaks_rows)),
        "total_area": float(total_area),
        "params": {
            "smooth": int(args.smooth),
            "snr": float(args.snr),
            "noise": float(args.noise),
            "min_width_sec": float(args.min_width_sec),
            "reject_height": float(args.reject_height),
            "runtime_min": float(args.runtime)
        }
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ---------- plotting: replicate cdf13.py main plot + inset, then add baseline and fills ----------

    # figure size similar to cdf13.py
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(23.5/2.54, 10.9/2.54), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # main chromatogram (RAW)
    ax.plot(t, y_raw, color='blue', zorder=2)

    # x-range and ticks
    ax.set_xlim(0, args.runtime)
    ticks = list(ax.get_xticks())
    if args.runtime not in ticks:
        ticks.append(args.runtime)
    ticks = sorted([v for v in ticks if 0 <= v <= args.runtime])
    ax.set_xticks(ticks)

    # labels
    ax.set_xlabel('time (min)')
    ax.set_ylabel('Intensity (AU)')

    # highest RAW peak marker and label
    idx_max = int(np.argmax(y_raw)) if y_raw.size else 0
    tr_peak = float(t[idx_max]) if t.size else 0.0
    h_peak = float(y_raw[idx_max]) if y_raw.size else 0.0
    ax.scatter([tr_peak], [h_peak], marker='x', color='red', zorder=3)
    dx = args.runtime * 0.005
    ax.annotate(f'tr = {tr_peak:.2f} min',
                xy=(tr_peak + dx, h_peak), xytext=(5, -5),
                textcoords='offset points', ha='left', va='top')

    # baseline segments and light-blue fills under detected peaks
    for (iL, p, iR) in merged:
        seg_t = t[iL:iR+1]
        seg_y = y_raw[iL:iR+1]
        base_vals = segment_baseline(t, y_raw, iL, iR)

        # baseline behind the trace
        ax.plot([t[iL], t[iR]], [y_raw[iL], y_raw[iR]],
                color='black', lw=0.8, alpha=0.7, zorder=1)

        # fill area between chromatogram and baseline
        ax.fill_between(seg_t, seg_y, base_vals, where=(seg_y > base_vals),
                        color='tab:blue', alpha=0.30, linewidth=0, zorder=1.5)

    # inset ±1.0 min around highest peak (no baseline in inset)
    inset_pos = [0.1, 0.6, 0.25, 0.25]
    ax_ins = fig.add_axes(inset_pos)
    mask = (t >= tr_peak - 1.0) & (t <= tr_peak + 1.0)
    ax_ins.plot(t[mask], y_raw[mask], color='blue')
    ax_ins.set_xlim(tr_peak - 1.0, tr_peak + 1.0)
    ymax = max(h_peak * 0.25, 1.0)
    ax_ins.set_ylim(-0.1, ymax)
    ax_ins.set_xticks([tr_peak - 1.0, tr_peak, tr_peak + 1.0])
    ax_ins.set_xticklabels([f"{x:.1f}" for x in [tr_peak - 1.0, tr_peak, tr_peak + 1.0]])
    ax_ins.tick_params(axis='both', labelsize=10)
    ax_ins.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

    # save images
    fig.savefig(out_svg, format='svg', dpi=300)
    fig.savefig(out_png, format='png', dpi=300)
    plt.close(fig)

    # report
    print("Wrote:")
    print(f"  {peaks_csv}")
    print(f"  {summary_json}")
    print(f"  {out_svg}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()

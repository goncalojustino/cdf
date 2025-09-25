#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdf13_post.py — reproduce cdf13.py main plot, add baseline and peak fills, integrate peaks.

USAGE
  python cdf13_post.py kh18.cdf
  python cdf13_post.py kh18.cdf \
      --smooth 32 --snr 5 --noise-mode fixed --noise 5000 --noise-window 0.5 \
      --min-width-sec 4 --reject-height 1000 --min-prominence 0 \
      --baseline-method valley --tangent-height-pct 10 --clip-negative on \
      --split-overlaps on --valley-depth-frac 0.0 \
      --measure area --integrator trapz \
      --subtract-blank off --blank-csv blank_chromatogram.csv --blank-scale 1.0 \
      --roi-start 0.0 --roi-end 32.0 --percent-denominator all --runtime 32

WHAT THIS SCRIPT DOES
- Reads <base>_chromatogram.csv produced by cdf13.py (you still pass the original .cdf name).
- Detects peaks on a smoothed signal; integrates areas on the RAW signal using a baseline.
- Reproduces the cdf13.py plot (main + inset). Adds baseline under each detected peak and light-blue fills.
- Writes:
    <base>_peaks.csv          # peak table
    <base>_summary.json       # integration summary and parameters
    <base>_with_baseline.svg  # figure
    <base>_with_baseline.png  # figure

INTEGRATION CONTROLS (mapped to HPLC-like options)
- Peak detection
    --smooth INT              default 32 points; detection runs on the SMOOTHED signal.
    --snr FLOAT               default 5; minimum S/N for acceptance.
    --noise-mode {fixed,start,pre} default fixed
        fixed: use --noise as the noise sigma for all peaks.
        start: estimate noise as stdev in the first --noise-window minutes.
        pre:   per-peak noise = stdev in [rt - --noise-window, rt).
    --noise FLOAT             default 5000; noise if --noise-mode=fixed.
    --noise-window FLOAT      default 0.5 min; window for start/pre noise estimation.
    --min-width-sec FLOAT     default 4 sec; minimum valley-to-valley width.
    --min-prominence FLOAT    default 0; require apex - max(valley endpoints) ≥ this value.
- Peak measurement
    --measure {area,height,sqrt_height} default area; controls the reported “measure_value”
    --reject-height FLOAT     default 1000; historical name—applied to the SELECTED measure.
                              Examples: if --measure height → minimum apex height; if area → minimum area.
- Peak overlap handling
    --split-overlaps {on,off} default on; if off, overlapping peaks are not truncated.
    --valley-depth-frac FLOAT default 0.0; require (apex - max(valleys)) ≥ frac * apex. 0 disables.
- Baseline under peaks
    --baseline-method {valley,tangent} default valley
        valley: straight line through the valley minima at the left/right bounds.
        tangent: straight line between the points where the trace crosses
                 apex * (--tangent-height-pct / 100) on each side; falls back to valleys.
    --tangent-height-pct FLOAT default 10; only used with tangent method.
    --clip-negative {on,off} default on; after baseline subtraction, negative values are clipped to 0.
- Blank subtraction
    --subtract-blank {on,off} default off; if on, subtract blank chromatogram first.
    --blank-csv PATH          CSV with columns time_*, intensity for the blank.
    --blank-scale FLOAT       default 1.0; multiplies the blank before subtraction.
- ROI and reporting
    --roi-start FLOAT         start time (min) to consider; optional.
    --roi-end FLOAT           end time (min) to consider; optional.
    --percent-denominator {all,roi,idonly} default all
         all/roi behave the same here since integration is restricted to ROI if given.
         idonly reserved for workflows with identified subsets; here same as all.
    --integrator {trapz,simpson} default trapz; Simpson falls back to trapz if spacing is nonuniform.
- Plotting
    --runtime FLOAT           x-axis max time (min) for the figure; default 32. Inset copied from cdf13.py.

Notes:
- Detection on SMOOTHED signal; integration and plotting on RAW signal.
- This script never reads the .cdf. It maps kh18.cdf → kh18_chromatogram.csv.
"""

from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------- helpers --------------------

def moving_avg(y: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return y
    k = int(max(1, w))
    ker = np.ones(k, dtype=float) / k
    return np.convolve(y, ker, mode="same")


def detect_time_col(df: pd.DataFrame) -> tuple[str, str]:
    for c in df.columns:
        if c.startswith("time_"):
            return c, c.split("time_", 1)[1]
    if "time" in df.columns:
        return "time", ""
    return df.columns[0], ""


def chrom_csv_from_cdf(cdf_path: str) -> str:
    if not cdf_path.lower().endswith(".cdf"):
        raise ValueError("Input must be a .cdf filename (e.g., kh18.cdf).")
    csv_path = os.path.splitext(cdf_path)[0] + "_chromatogram.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}. Run cdf13.py first.")
    return csv_path


def local_maxima_indices(y: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return np.array([], dtype=int)
    cond = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    return np.where(cond)[0] + 1


def nearest_left_minimum(y: np.ndarray, start_i: int) -> int:
    for i in range(start_i, 1, -1):
        if y[i-1] >= y[i] <= y[i+1]:
            return i
    return 0


def nearest_right_minimum(y: np.ndarray, start_i: int) -> int:
    n = y.size
    for i in range(start_i, n-1):
        if y[i-1] >= y[i] <= y[i+1]:
            return i
    return n - 1


def enforce_min_separation(peaks: np.ndarray, strength: np.ndarray, min_sep_pts: int) -> np.ndarray:
    if peaks.size == 0:
        return peaks
    order = np.argsort(strength[peaks])[::-1]
    selected = []
    taken = np.zeros(peaks.size, dtype=bool)
    for oi in order:
        p = peaks[oi]
        if taken[oi]:
            continue
        selected.append(p)
        left, right = p - min_sep_pts, p + min_sep_pts
        for j, pj in enumerate(peaks):
            if not taken[j] and left <= pj <= right:
                taken[j] = True
    return np.array(sorted(selected), dtype=int)


def segment_baseline_valley(t: np.ndarray, y: np.ndarray, iL: int, iR: int) -> np.ndarray:
    tL, tR = t[iL], t[iR]
    yL, yR = y[iL], y[iR]
    if iR == iL:
        return np.full(1, yL, dtype=float)
    m = (yR - yL) / (tR - tL + 1e-12)
    return yL + m * (t[iL:iR+1] - tL)


def _crossing_idx_at_fraction(y: np.ndarray, apex_i: int, frac: float, side: str) -> int | None:
    """Find index on a given side where y crosses target= y[apex_i]*frac. Linear threshold crossing in index space."""
    target = y[apex_i] * frac
    n = y.size
    if side == "left":
        for i in range(apex_i, 0, -1):
            if (y[i-1] <= target <= y[i]) or (y[i-1] >= target >= y[i]):
                return i
    else:
        for i in range(apex_i, n-1):
            if (y[i] >= target >= y[i+1]) or (y[i] <= target <= y[i+1]):
                return i
    return None


def segment_baseline_tangent(t: np.ndarray, y: np.ndarray, iL: int, apex_i: int, iR: int, pct: float) -> tuple[np.ndarray, int, int]:
    """Tangent-like baseline: connect points where the trace crosses apex * pct/100 on each side.
       Fallback to valley endpoints if crossings not found."""
    frac = max(pct, 0.0) / 100.0
    li = _crossing_idx_at_fraction(y, apex_i, frac, "left")
    ri = _crossing_idx_at_fraction(y, apex_i, frac, "right")
    if li is None or ri is None:
        li, ri = iL, iR
    tL, tR = t[li], t[ri]
    yL, yR = y[li], y[ri]
    if ri == li:
        base = np.full(1, yL, dtype=float)
    else:
        m = (yR - yL) / (tR - tL + 1e-12)
        base = yL + m * (t[li:ri+1] - tL)
    return base, li, ri


def simpson_integrate(t: np.ndarray, y: np.ndarray) -> float:
    """Composite Simpson for near-uniform spacing. Falls back to trapz if strongly nonuniform or < 3 points."""
    n = y.size
    if n < 3:
        return float(np.trapz(y, t))
    dt = np.diff(t)
    med = np.median(dt)
    if med <= 0 or np.max(np.abs(dt - med)) > 0.05 * med:
        return float(np.trapz(y, t))
    # Ensure odd number of points; if even, drop last interval for Simpson, add last trapezoid
    if (n - 1) % 2 == 1:
        # even number of intervals → odd n? Actually Simpson needs even number of intervals.
        pass
    area = 0.0
    # Use pairs of intervals (3 points) for Simpson
    last_used = 1
    for i in range(0, n - 2, 2):
        h = t[i+2] - t[i]
        area += (h / 6.0) * (y[i] + 4.0 * y[i+1] + y[i+2])
        last_used = i + 2
    if last_used < n - 1:
        area += np.trapz(y[last_used:], t[last_used:])
    return float(area)


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser(description="Baseline, peak fill, and integration for cdf13 output.")
    p.add_argument("cdf", help="Original .cdf filename; reads <base>_chromatogram.csv")
    # Detection
    p.add_argument("--smooth", type=int, default=32, help="moving average window (points) for detection; default 32")
    p.add_argument("--snr", type=float, default=5.0, help="minimum S/N for peak acceptance; default 5")
    p.add_argument("--noise-mode", choices=["fixed", "start", "pre"], default="fixed",
                   help="noise source for SNR: fixed|start-of-run|pre-peak; default fixed")
    p.add_argument("--noise", type=float, default=5000.0, help="noise level if --noise-mode=fixed; default 5000")
    p.add_argument("--noise-window", type=float, default=0.5, help="minutes used for start/pre noise estimation; default 0.5")
    p.add_argument("--min-width-sec", type=float, default=4.0, help="minimum peak width valley-to-valley (sec); default 4")
    p.add_argument("--min-prominence", type=float, default=0.0, help="require apex - max(valley endpoints) ≥ value; default 0")
    # Measurement and reject
    p.add_argument("--measure", choices=["area", "height", "sqrt_height"], default="area",
                   help="peak measurement reported as 'measure_value'; default area")
    p.add_argument("--reject-height", type=float, default=1000.0,
                   help="minimum value for selected measurement (historical name); default 1000")
    # Overlaps and valley depth
    p.add_argument("--split-overlaps", choices=["on", "off"], default="on",
                   help="split/truncate overlapping peaks; default on")
    p.add_argument("--valley-depth-frac", type=float, default=0.0,
                   help="require (apex - max(valleys)) ≥ frac * apex; default 0")
    # Baseline
    p.add_argument("--baseline-method", choices=["valley", "tangent"], default="valley",
                   help="baseline under peaks; default valley")
    p.add_argument("--tangent-height-pct", type=float, default=10.0,
                   help="percent of apex for tangent baseline; default 10")
    p.add_argument("--clip-negative", choices=["on", "off"], default="on",
                   help="clip negative (y - baseline) to 0 before integration; default on")
    # Blank subtraction
    p.add_argument("--subtract-blank", choices=["on", "off"], default="off",
                   help="subtract a blank chromatogram before detection; default off")
    p.add_argument("--blank-csv", type=str, default=None, help="path to blank CSV with columns time_*, intensity")
    p.add_argument("--blank-scale", type=float, default=1.0, help="scale factor for blank; default 1.0")
    # ROI and reporting
    p.add_argument("--roi-start", type=float, default=None, help="ROI start time (min)")
    p.add_argument("--roi-end", type=float, default=None, help="ROI end time (min)")
    p.add_argument("--percent-denominator", choices=["all", "roi", "idonly"], default="all",
                   help="denominator for %area; default all")
    p.add_argument("--integrator", choices=["trapz", "simpson"], default="trapz",
                   help="numeric integrator; default trapz")
    # Plotting
    p.add_argument("--runtime", type=float, default=32.0, help="x-axis max time for plotting (min); default 32")
    args = p.parse_args()

    # Load chromatogram from cdf13 output
    csv_path = chrom_csv_from_cdf(args.cdf)
    df = pd.read_csv(csv_path)
    time_col, unit = detect_time_col(df)
    if "intensity" not in df.columns:
        raise ValueError(f"'intensity' column not found in {csv_path}. Columns: {list(df.columns)}")
    t = df[time_col].to_numpy(dtype=float)
    y_raw = df["intensity"].to_numpy(dtype=float)
    y_raw = np.nan_to_num(y_raw, nan=0.0)

    # Optional blank subtraction
    if args.subtract_blank == "on":
        if not args.blank_csv:
            raise ValueError("--subtract-blank on but --blank-csv not provided")
        dfb = pd.read_csv(args.blank_csv)
        tcol_b, _ = detect_time_col(dfb)
        if "intensity" not in dfb.columns:
            raise ValueError(f"'intensity' column not found in blank CSV: {args.blank_csv}")
        tb = dfb[tcol_b].to_numpy(dtype=float)
        yb = dfb["intensity"].to_numpy(dtype=float) * float(args.blank_scale)
        # interpolate blank onto t
        yb_on_t = np.interp(t, tb, yb, left=yb[0], right=yb[-1])
        y_raw = y_raw - yb_on_t

    # Optional clipping of raw negatives before detection to keep SNR sane
    if args.clip_negative == "on":
        y_raw = np.clip(y_raw, 0.0, None)

    # Detection signal
    y_det = moving_avg(y_raw, args.smooth)

    # ROI mask
    roi_mask = np.ones_like(t, dtype=bool)
    if args.roi_start is not None:
        roi_mask &= (t >= float(args.roi_start))
    if args.roi_end is not None:
        roi_mask &= (t <= float(args.roi_end))

    # Candidate peaks from smoothed signal
    peaks_all = local_maxima_indices(y_det)
    # Enforce minimum separation by the min width converted to points
    dt = np.median(np.diff(t)) if t.size > 1 else 1.0
    min_width_min = float(args.min_width_sec) / 60.0
    min_sep_pts = max(1, int(round(min_width_min / max(dt, 1e-12))))
    peaks_sep = enforce_min_separation(peaks_all, y_det, min_sep_pts)

    # Quick debug print (optional). Add after computing peaks_sep:
    print(f"candidates={len(peaks_all)}, sep_kept={len(peaks_sep)}")

    # Global noise if needed
    def estimate_start_noise():
        t_end = (t[0] + float(args.noise_window)) if t.size else 0.0
        mask = (t <= t_end)
        if not mask.any():
            return float(args.noise)
        return float(np.std(y_det[mask]))

    noise_global = None
    if args.noise_mode == "fixed":
        noise_global = float(args.noise)
    elif args.noise_mode == "start":
        noise_global = estimate_start_noise()

    # Accept peaks with filters and bounds
    accepted = []
    for pidx in peaks_sep:
        rt = float(t[pidx])
        if not roi_mask[pidx]:
            # skip peaks whose apex is outside ROI
            continue

        # left/right bounds on DETECTION signal
        iL = nearest_left_minimum(y_det, pidx)
        iR = nearest_right_minimum(y_det, pidx)
        if iR <= iL:
            continue
        if (t[iR] - t[iL]) < min_width_min:
            continue

        # Basic valley depth and prominence checks
        valleys_max = max(y_det[iL], y_det[iR])
        apex_det = float(y_det[pidx])
        if args.valley_depth_frac > 0.0 and (apex_det - valleys_max) < args.valley_depth_frac * max(apex_det, 1e-12):
            continue
        if args.min_prominence > 0.0 and (apex_det - valleys_max) < float(args.min_prominence):
            continue

        # SNR
        if args.noise_mode == "pre":
            pre_start = rt - float(args.noise_window)
            mask_pre = (t >= pre_start) & (t < rt)
            if mask_pre.sum() >= 5:
                noise = float(np.std(y_det[mask_pre]))
            else:
                noise = float(args.noise)  # fallback
        else:
            noise = float(noise_global)
        snr = apex_det / max(noise, 1e-12)
        if snr < float(args.snr):
            continue

        accepted.append((iL, pidx, iR))

    # Overlap handling
    accepted.sort(key=lambda x: x[1])  # by apex index
    merged = []
    if args.split_overlaps == "on":
        last_end = -1
        for iL, p, iR in accepted:
            if iL <= last_end:
                iL = max(iL, last_end)
                if iR <= iL:
                    continue
            merged.append((iL, p, iR))
            last_end = iR
    else:
        merged = accepted[:]  # allow overlaps

    # Integrate peaks, build table
    def integrate(seg_t: np.ndarray, seg_y: np.ndarray) -> float:
        if args.integrator == "simpson":
            return simpson_integrate(seg_t, seg_y)
        return float(np.trapz(seg_y, seg_t))

    peaks_rows = []
    total_area = 0.0

    for k, (iL0, p, iR0) in enumerate(merged, start=1):
        # Clip to ROI in index space if provided
        iL, iR = iL0, iR0
        if args.roi_start is not None:
            while iL <= iR and t[iL] < float(args.roi_start):
                iL += 1
        if args.roi_end is not None:
            while iR >= iL and t[iR] > float(args.roi_end):
                iR -= 1
        if iR <= iL:
            continue

        # Baseline under this peak on RAW signal
        if args.baseline_method == "valley":
            base_vals = segment_baseline_valley(t, y_raw, iL, iR)
            seg_t = t[iL:iR+1]
            seg_y = y_raw[iL:iR+1]
            bL_i, bR_i = iL, iR
        else:
            base_vals, bL_i, bR_i = segment_baseline_tangent(t, y_raw, iL, p, iR, float(args.tangent_height_pct))
            seg_t = t[bL_i:bR_i+1]
            seg_y = y_raw[bL_i:bR_i+1]

        y_above = seg_y - base_vals
        if args.clip_negative == "on":
            y_above = np.clip(y_above, 0.0, None)

        area = integrate(seg_t, y_above)

        apex_height = float(y_raw[p])
        measure_value = area if args.measure == "area" else (apex_height if args.measure == "height" else float(np.sqrt(max(apex_height, 0.0))))
        if measure_value < float(args.reject_height):
            continue

        total_area += area
        peaks_rows.append({
            "index": k,
            "t_start_min": float(t[bL_i]),
            "t_end_min": float(t[bR_i]),
            "rt_min": float(t[p]),
            "apex_intensity": apex_height,
            "area": float(area),
            "measure_value": float(measure_value)
        })

    # Quick debug: Add after building accepted and peaks_rows:
    print(f"accepted={len(accepted)}, integrated={len(peaks_rows)}")

    # Denominator for %area
    denom = sum(row["area"] for row in peaks_rows) if peaks_rows else 0.0
    # 'roi' and 'all' are equivalent here because we already clipped to ROI; 'idonly' same as all in this context.
    for row in peaks_rows:
        row["pct_area"] = 100.0 * row["area"] / denom if denom > 0 else 0.0

    base = os.path.splitext(os.path.basename(args.cdf))[0]
    peaks_csv = f"{base}_peaks.csv"
    summary_json = f"{base}_summary.json"
    out_svg = f"{base}_with_baseline.svg"
    out_png = f"{base}_with_baseline.png"

    # Write peaks CSV (robust if no peaks)
    cols = ["index", "t_start_min", "t_end_min", "rt_min", "apex_intensity", "area", "pct_area", "measure_value"]
    peaks_df = pd.DataFrame(peaks_rows, columns=cols)
    if not peaks_df.empty:
        peaks_df = peaks_df.sort_values("rt_min", kind="mergesort")
        for c in ("t_start_min", "t_end_min", "rt_min"):
            peaks_df[c] = peaks_df[c].map(lambda v: f"{float(v):.4f}")
        peaks_df["area"] = peaks_df["area"].map(lambda v: f"{float(v):.6g}")
        peaks_df["pct_area"] = peaks_df["pct_area"].map(lambda v: f"{float(v):.3f}")
        peaks_df["measure_value"] = peaks_df["measure_value"].map(lambda v: f"{float(v):.6g}")
    peaks_df.to_csv(peaks_csv, index=False)

    # Summary JSON
    summary = {
        "source_csv": os.path.abspath(csv_path),
        "time_unit": unit or "min",
        "n_points": int(t.size),
        "n_peaks": int(len(peaks_rows)),
        "total_area": float(denom),
        "params": {
            "smooth": int(args.smooth),
            "snr": float(args.snr),
            "noise_mode": args.noise_mode,
            "noise": float(args.noise),
            "noise_window_min": float(args.noise_window),
            "min_width_sec": float(args.min_width_sec),
            "min_prominence": float(args.min_prominence),
            "measure": args.measure,
            "reject_height": float(args.reject_height),
            "split_overlaps": args.split_overlaps,
            "valley_depth_frac": float(args.valley_depth_frac),
            "baseline_method": args.baseline_method,
            "tangent_height_pct": float(args.tangent_height_pct),
            "clip_negative": args.clip_negative,
            "subtract_blank": args.subtract_blank,
            "blank_csv": args.blank_csv,
            "blank_scale": float(args.blank_scale),
            "roi_start": None if args.roi_start is None else float(args.roi_start),
            "roi_end": None if args.roi_end is None else float(args.roi_end),
            "percent_denominator": args.percent_denominator,
            "integrator": args.integrator,
            "runtime_min": float(args.runtime)
        }
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # -------- plotting: reproduce cdf13.py main + inset, add baseline + fills --------
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(23.5/2.54, 10.9/2.54), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # main chromatogram (RAW)
    ax.plot(t, y_raw, color='blue', zorder=2)

    # x-range and ticks
    ax.set_xlim(0, float(args.runtime))
    ticks = list(ax.get_xticks())
    if float(args.runtime) not in ticks:
        ticks.append(float(args.runtime))
    ticks = sorted([v for v in ticks if 0 <= v <= float(args.runtime)])
    ax.set_xticks(ticks)

    # labels
    ax.set_xlabel('time (min)')
    ax.set_ylabel('Intensity (AU)')

    # highest RAW peak marker and label
    if y_raw.size:
        idx_max = int(np.argmax(y_raw))
        tr_peak = float(t[idx_max])
        h_peak = float(y_raw[idx_max])
    else:
        idx_max, tr_peak, h_peak = 0, 0.0, 0.0
    ax.scatter([tr_peak], [h_peak], marker='x', color='red', zorder=3)
    dx = float(args.runtime) * 0.005
    ax.annotate(f'tr = {tr_peak:.2f} min', xy=(tr_peak + dx, h_peak),
                xytext=(5, -5), textcoords='offset points', ha='left', va='top')

    # Draw baseline segments and fills (light blue) for each accepted peak segment actually integrated
    for row in peaks_rows:
        # map times back to indices for drawing
        tL, tR = row["t_start_min"], row["t_end_min"]
        # nearest indices
        iL = int(np.searchsorted(t, tL, side="left"))
        iR = int(np.searchsorted(t, tR, side="right") - 1)
        iL = max(0, min(iL, t.size - 1))
        iR = max(0, min(iR, t.size - 1))
        if iR <= iL:
            continue
        seg_t = t[iL:iR+1]
        seg_y = y_raw[iL:iR+1]

        # baseline line between endpoints on RAW
        ax.plot([t[iL], t[iR]], [y_raw[iL], y_raw[iR]],
                color='black', lw=0.8, alpha=0.7, zorder=1)

        # fill above baseline
        # compute baseline values for filling
        base_vals = segment_baseline_valley(t, y_raw, iL, iR)
        ax.fill_between(seg_t, seg_y, base_vals, where=(seg_y > base_vals),
                        color='tab:blue', alpha=0.30, linewidth=0, zorder=1.5)

    # inset ±1.0 min around highest peak (no baseline in inset)
    inset_pos = [0.1, 0.6, 0.25, 0.25]
    ax_ins = fig.add_axes(inset_pos)
    mask_in = (t >= tr_peak - 1.0) & (t <= tr_peak + 1.0)
    ax_ins.plot(t[mask_in], y_raw[mask_in], color='blue')
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

    print("Wrote:")
    print(f"  {peaks_csv}")
    print(f"  {summary_json}")
    print(f"  {out_svg}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()

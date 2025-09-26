#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdf13_run.py — simple pipeline wrapper

Runs (from CWD):
  1) python3 scripts/cdf14.py -i raw_cdf/<base>.cdf
  2) python3 scripts/cdf13_post3.py raw_cdf/<base>.cdf [detect flags]
  3) python3 scripts/cdf13_report2.py raw_cdf/<base>.cdf [report flags]

After step 3:
  Move all CWD files starting with "<base>" to raw_cdf/, EXCEPT "<base>_report.docx".

Extras:
  - Config file (key=value) in CWD: cdf13_config.txt
  - Presets for detect: --preset default|lenient|strict
  - High-level detect knobs on this wrapper: --snr --noise-mode --noise-window --smooth --min-width-sec
                                             --measure --reject-height --runtime --roi-start --roi-end
                                             --integrator --percent-denominator --clip-negative
                                             --baseline-method --tangent-height-pct --split-overlaps
                                             --valley-depth-frac --min-prominence
                                             --subtract-blank --blank-csv --blank-scale
                                             --fill-alpha --fill-color --baseline-lw --baseline-color
  - Pass-through handles:
        --process  → appended to cdf14.py
        --detect   → appended to cdf13_post3.py (after wrapper-built flags)
        --report   → appended to cdf13_report2.py
  - Precedence: CLI knobs > preset > config file.
"""

import argparse, os, sys, glob, shlex, subprocess, shutil

SCRIPTS_DIR = "scripts"
RAW_DIR = "raw_cdf"
PROC  = os.path.join(SCRIPTS_DIR, "cdf14.py")
POST  = os.path.join(SCRIPTS_DIR, "cdf13_post3.py")
REPORT= os.path.join(SCRIPTS_DIR, "cdf13_report2.py")
CONFIG_DEFAULT_PATH = "scripts/cdf13_config.txt"

PRESETS = {
    "default": {"noise-mode":"pre","noise-window":"0.3","snr":"2","smooth":"8","min-width-sec":"1","measure":"area","reject-height":"0.01"},
    "lenient": {"noise-mode":"pre","noise-window":"0.5","snr":"1.5","smooth":"5","min-width-sec":"0.5","measure":"area","reject-height":"0"},
    "strict":  {"noise-mode":"pre","noise-window":"0.3","snr":"3","smooth":"16","min-width-sec":"2","measure":"area","reject-height":"0.02"},
}

DETECT_KEYS = [
    "snr","noise-mode","noise-window","smooth","min-width-sec","measure","reject-height",
    "runtime","roi-start","roi-end","integrator","percent-denominator","noise",
    "clip-negative","baseline-method","tangent-height-pct","split-overlaps",
    "valley-depth-frac","min-prominence","subtract-blank","blank-csv","blank-scale",
    "fill-alpha","fill-color","baseline-lw","baseline-color"
]

def ensure_layout():
    missing = [p for p in [SCRIPTS_DIR, RAW_DIR, PROC, POST, REPORT] if not os.path.exists(p)]
    if missing:
        sys.exit(f"ERROR: missing paths: {missing}")

def list_bases(all_flag, bases_cli):
    if all_flag:
        files = sorted(glob.glob(os.path.join(RAW_DIR, "*.cdf")))
        if not files:
            sys.exit(f"ERROR: no .cdf in {RAW_DIR}")
        return [os.path.splitext(os.path.basename(p))[0] for p in files]
    if not bases_cli:
        sys.exit("ERROR: provide bases or --all")
    return [os.path.splitext(b)[0] for b in bases_cli]

def run(argv):
    print(">>", os.getcwd(), "$", " ".join(shlex.quote(a) for a in argv))
    subprocess.run(argv, check=True)

def parse_config(path):
    cfg = {}
    if not (path and os.path.exists(path)): return cfg
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s: continue
            k, v = s.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg

def detect_args_from(cfg_map):
    argv = []
    for k in DETECT_KEYS:
        if k in cfg_map:
            argv += [f"--{k}", cfg_map[k]]
    return argv

def collect_detect_cfg(args, cfg, preset):
    # config base
    m = {k: v for k, v in cfg.items() if k in DETECT_KEYS and v != ""}
    # preset overlay
    if preset:
        m.update(PRESETS[preset])
    # CLI overlays
    cli = {
        "snr":args.snr,"noise-mode":args.noise_mode,"noise-window":args.noise_window,"smooth":args.smooth,
        "min-width-sec":args.min_width_sec,"measure":args.measure,"reject-height":args.reject_height,"noise":args.noise,
        "runtime":args.runtime,"roi-start":args.roi_start,"roi-end":args.roi_end,"integrator":args.integrator,
        "percent-denominator":args.percent_denominator,"clip-negative":args.clip_negative,
        "baseline-method":args.baseline_method,"tangent-height-pct":args.tangent_height_pct,
        "split-overlaps":args.split_overlaps,"valley-depth-frac":args.valley_depth_frac,
        "min-prominence":args.min_prominence,"subtract-blank":args.subtract_blank,
        "blank-csv":args.blank_csv,"blank-scale":args.blank_scale,
        "fill-alpha":args.fill_alpha,"fill-color":args.fill_color,"baseline-lw":args.baseline_lw,"baseline-color":args.baseline_color,
    }
    for k, v in cli.items():
        if v is not None:
            m[k] = str(v)
    return m

def normalize_blank_csv(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--blank-csv" and i + 1 < len(argv):
            val = argv[i+1]
            if val and not os.path.isabs(val):
                val = os.path.abspath(val)
            out += [argv[i], val]; i += 2
        else:
            out.append(argv[i]); i += 1
    return out

def move_outputs(base):
    """Move all CWD files starting with base* to raw_cdf/, except <base>_report.docx."""
    keep = f"{base}_report.docx"
    for path in glob.glob(f"{base}*"):
        if os.path.isdir(path):
            continue
        if os.path.basename(path) == keep:
            continue
        dest = os.path.join(RAW_DIR, os.path.basename(path))
        # overwrite if exists
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(path, dest)
        print(f"moved: {path} -> {dest}")

def main():
    ap = argparse.ArgumentParser(description="Run cdf14 → cdf13_post3 → cdf13_report2, then move outputs to raw_cdf except the report.")
    ap.add_argument("bases", nargs="*", help="sample base names (with or without .cdf)")
    ap.add_argument("--all", action="store_true")

    # presets + config
    ap.add_argument("--preset", choices=["default","lenient","strict"], default=None)
    ap.add_argument("--config", default=CONFIG_DEFAULT_PATH)

    # high-level detect knobs
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--noise-mode", choices=["fixed","start","pre"], default=None)
    ap.add_argument("--noise-window", type=float, default=None)
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--smooth", type=int, default=None)
    ap.add_argument("--min-width-sec", type=float, default=None)
    ap.add_argument("--measure", choices=["area","height","sqrt_height"], default=None)
    ap.add_argument("--reject-height", type=float, default=None)
    ap.add_argument("--runtime", type=float, default=None)
    ap.add_argument("--roi-start", type=float, default=None)
    ap.add_argument("--roi-end", type=float, default=None)
    ap.add_argument("--integrator", choices=["trapz","simpson"], default=None)
    ap.add_argument("--percent-denominator", choices=["all","roi","idonly"], default=None)
    ap.add_argument("--clip-negative", choices=["on","off"], default=None)
    ap.add_argument("--baseline-method", choices=["valley","tangent"], default=None)
    ap.add_argument("--tangent-height-pct", type=float, default=None)
    ap.add_argument("--split-overlaps", choices=["on","off"], default=None)
    ap.add_argument("--valley-depth-frac", type=float, default=None)
    ap.add_argument("--min-prominence", type=float, default=None)
    ap.add_argument("--subtract-blank", choices=["on","off"], default=None)
    ap.add_argument("--blank-csv", default=None)
    ap.add_argument("--blank-scale", type=float, default=None)
    ap.add_argument("--fill-alpha", type=float, default=None)
    ap.add_argument("--fill-color", default=None)
    ap.add_argument("--baseline-lw", type=float, default=0.5, help="Baseline line width (default: 0.5)")
    ap.add_argument("--baseline-color", default="orange", help="Baseline color (default: orange)")

    # pass-through handles
    ap.add_argument("--chrom-lw", type=float, default=0.75, help="Chromatogram line width (default: 0.75)")
    ap.add_argument("--process", default="", help="extra args for scripts/cdf14.py")
    ap.add_argument("--detect",  default="", help="extra args for scripts/cdf13_post3.py (appended after built flags)")
    ap.add_argument("--report",  default="", help="extra args for scripts/cdf13_report2.py")

    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_layout()
    bases = list_bases(args.all, args.bases)
    cfg = parse_config(args.config)

    # build detect argv
    detect_cfg = collect_detect_cfg(args, cfg, args.preset)
    detect_argv = detect_args_from(detect_cfg)
    if cfg.get("detect"):
        detect_argv += shlex.split(cfg["detect"])
    if args.detect:
        detect_argv += shlex.split(args.detect)
    detect_argv = normalize_blank_csv(detect_argv)

    py = sys.executable or "python3"

    for base in bases:
        cdf_arg = os.path.join(RAW_DIR, f"{base}.cdf")

        # 1) cdf14.py
        cmd1 = [py, PROC, "-i", cdf_arg] + (shlex.split(args.process) if args.process else [])
        print(f"=== {base}: step 1/3 ===")
        if not args.dry_run: run(cmd1)
        else: print("DRY-RUN:", " ".join(shlex.quote(x) for x in cmd1))

        # 2) cdf13_post3.py
        cmd2 = [py, POST, cdf_arg] + detect_argv
        print(f"=== {base}: step 2/3 ===")
        if not args.dry_run: run(cmd2)
        else: print("DRY-RUN:", " ".join(shlex.quote(x) for x in cmd2))

        # 3) cdf13_report2.py
        report_argv = []
        if args.report: report_argv += shlex.split(args.report)
        if args.chrom_lw is not None: report_argv += ["--chrom-lw", str(args.chrom_lw)]
        cmd3 = [py, REPORT, cdf_arg] + report_argv
        print(f"=== {base}: step 3/3 ===")
        if not args.dry_run: run(cmd3)
        else: print("DRY-RUN:", " ".join(shlex.quote(x) for x in cmd3))

        # Move outputs after report, except the report DOCX
        if not args.dry_run:
            move_outputs(base)

    print("Done." if not args.dry_run else "Dry-run complete.")

if __name__ == "__main__":
    main()

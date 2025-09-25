# cdf13_run.py — quick guide

1. process all cdf files in raw_cdf folder
```
python3 cdf13_run.py --all
```

2. try a preset
```
python3 cdf13_run.py kh18 --preset lenient
python3 cdf13_run.py kh18 --preset strict
```

3. override a couple of knobs without touching the config
```
python3 cdf13_run.py kh18 --snr 2 --smooth 8 --min-width-sec 1 --measure area --reject-height 0
```

4. preview command without running
```
python3 cdf13_run.py kh18 --preset strict --dry-run
```

5. pass-through when needed; if you must pass raw flags directly:
```
--process "..." → to cdf13.py
--detect "..." → to cdf13_post3.py (appended after the wrapper-built flags)
--report "..." → to cdf13_report2.py
Legacy names (--cdf13-args, --post-args, --report-args) are still accepted.
```

6. alternative individual runs:
  ```
  python3 scripts/cdf14.py -i raw_cdf/<base>.cdf
  python3 scripts/cdf13_post3.py raw_cdf/<base>.cdf [detect flags]
  python3 scripts/cdf13_report2.py raw_cdf/<base>.cdf [report flags]
  ```

# Not so quick guide

## What it does
Runs the full pipeline for each CDF in `raw_cdf/`:
1. `scripts/cdf13.py` in `raw_cdf/`
2. `scripts/cdf13_post3.py` in `raw_cdf/` (peak detection + baseline + integration)
3. `scripts/cdf13_report2.py` in CWD (creates `<base>_report.docx`)

All intermediates stay in `raw_cdf/`. The final DOCX is written to CWD.

## Config first
The wrapper reads defaults from `cdf13_config.txt` in CWD. Edit this file to set your standard parameters.  
Each key has:
- **definition:** short meaning  
- **allowed values:** valid inputs  
The wrapper maps these keys to `scripts/cdf13_post3.py` flags.

## Presets
You can override the config quickly with:
- `--preset default`  
- `--preset lenient`  
- `--preset strict`

Presets are applied on top of the config. Any CLI knob you pass overrides both.

## One-liners
Process one file using config:
```bash
python3 cdf13_run.py kh18

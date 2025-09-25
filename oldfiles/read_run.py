#!/usr/bin/env python3
import argparse, json, os, re, sys
from pathlib import Path

def read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def ascii_strings(b: bytes, minlen=4):
    out=[]
    cur=[]
    for ch in b:
        if 32 <= ch < 127:
            cur.append(chr(ch))
        else:
            if len(cur)>=minlen:
                out.append(''.join(cur))
            cur=[]
    if len(cur)>=minlen:
        out.append(''.join(cur))
    return out

def try_decode(block: bytes):
    # Try useful decodings without failing
    tries = []
    for enc in ("utf-8", "latin-1", "utf-16le", "utf-16be"):
        try:
            s = block.decode(enc, errors="ignore")
            s2 = s.strip()
            if s2:
                tries.append((enc, s2))
        except Exception:
            pass
    # Deduplicate identical decoded text keeping first encoding
    seen=set(); dedup=[]
    for enc, s in tries:
        if s not in seen:
            dedup.append((enc, s))
            seen.add(s)
    return dedup

def find_tokens(b: bytes):
    # All markers that look like section headers beginning with 'E'
    toks = []
    for m in re.finditer(rb"E[A-Za-z0-9_.\-]+", b):
        toks.append((m.start(), m.group()))
    # Build segments from each token to the next token start
    segs=[]
    for i,(pos,tok) in enumerate(toks):
        start = pos
        end = toks[i+1][0] if i+1 < len(toks) else len(b)
        segs.append((start, end, tok))
    return toks, segs

def write_block(outdir: Path, idx: int, tag: bytes, block: bytes):
    tag_txt = tag.decode("latin-1", errors="ignore")
    safe = re.sub(r"[^A-Za-z0-9_.\-]+", "_", tag_txt).strip("_")
    base = outdir / f"{idx:03d}_{safe}"
    # Raw dump
    (base.with_suffix(".bin")).write_bytes(block)
    # Attempt text
    dec = try_decode(block)
    if dec:
        # Prefer the longest decoded variant
        dec_sorted = sorted(dec, key=lambda x: len(x[1]), reverse=True)
        (base.with_suffix(".txt")).write_text(dec_sorted[0][1], encoding="utf-8")
    return str(base)

def heuristics_chromatogram(b: bytes, outdir: Path):
    """
    Best-effort scan for time–intensity arrays.
    Strategy:
      1) Scan for monotonically increasing float32 or float64 sequences length ≥ 100 with small step variance.
      2) Search nearby for a same-length float array to pair as intensities.
    Returns CSV path if found.
    """
    import numpy as np
    def scan(dtype, minlen=100):
        item = np.dtype(dtype).itemsize
        n = len(b)//item
        arr = np.frombuffer(memoryview(b)[:n*item], dtype=dtype)
        candidates=[]
        i=0
        while i < len(arr)-minlen:
            # quick filter: finite window
            window = arr[i:i+minlen]
            if not np.all(np.isfinite(window)):
                i += minlen
                continue
            # monotonic increasing
            if np.all(np.diff(window) > 0):
                # extend while monotonic
                j = i+minlen
                while j < len(arr) and np.isfinite(arr[j]) and arr[j] > arr[j-1]:
                    j += 1
                seq = arr[i:j]
                # step homogeneity
                d = np.diff(seq)
                if len(d)>5 and float(np.std(d)) < 10*float(np.median(d)+1e-9):
                    candidates.append((i, j, seq))
                i = j
            else:
                i += minlen
        return arr, candidates

    # Try doubles then floats
    for dtype in ("<f8","<f4"):
        try:
            arr, cands = scan(dtype)
        except Exception:
            cands=[]
        if not cands:
            continue
        # Pick the longest candidate with plausible span
        cands.sort(key=lambda t: t[1]-t[0], reverse=True)
        for i0, j0, times in cands:
            span = float(times[-1] - times[0])
            if not (1e-6 < span < 1e6):
                continue
            L = j0 - i0
            # Search for intensities near this region in same dtype
            # within ± one million elements window (cap by file size)
            import math
            w = min(500000, len(arr))
            s = max(0, i0 - w)
            e = min(len(arr), j0 + w)
            best = None
            # Slide a window to find a finite array of same length that is not monotonic
            k = s
            while k + L <= e:
                y = arr[k:k+L]
                if np.all(np.isfinite(y)) and not np.all(np.diff(y) >= 0) and not np.all(np.diff(y) <= 0):
                    # basic amplitude filter
                    rng = float(np.nanmax(y) - np.nanmin(y))
                    if rng > 0:
                        best = (k, y)
                        break
                k += max(1, L//50)
            if best:
                import csv
                csv_path = outdir / "candidate_chromatogram.csv"
                with open(csv_path, "w", newline="") as fh:
                    wri = csv.writer(fh)
                    wri.writerow(["index","time_"+("s" if dtype=="<f8" else "u"),"intensity"])
                    for n,(t,v) in enumerate(zip(times, best[1])):
                        wri.writerow([n, float(t), float(v)])
                return str(csv_path)
    return None

def main():
    ap = argparse.ArgumentParser(description="Varian STAR .run extractor")
    ap.add_argument("run_file", help=".run file")
    ap.add_argument("-o","--outdir", default=None, help="output directory")
    args = ap.parse_args()

    src = Path(args.run_file)
    if not src.exists():
        print("Input not found", file=sys.stderr); sys.exit(1)
    outdir = Path(args.outdir) if args.outdir else src.with_suffix("")  # folder named like the file
    outdir.mkdir(parents=True, exist_ok=True)

    b = read_bytes(src)

    # Index tokens and segments
    tokens, segs = find_tokens(b)
    index = []
    for n,(start,end,tag) in enumerate(segs, start=1):
        block = b[start:end]
        pathbase = write_block(outdir, n, tag, block)
        index.append({
            "order": n,
            "tag": tag.decode("latin-1","ignore"),
            "start": start,
            "end": end,
            "size": end-start,
            "dump_base": Path(pathbase).name
        })

    # Strings and quick metadata
    strs = ascii_strings(b, minlen=4)
    # crude metadata guesses
    meta = {}
    for s in strs:
        if s.startswith("Varian"):
            meta.setdefault("vendor", s)
        if re.search(r"\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}", s, re.I):
            meta.setdefault("date_like", s)
        if s.lower().endswith(".mth"):
            meta.setdefault("method_path", s)
        if re.search(r"[A-Za-z]:\\\\star\\\\data\\\\", s, re.I):
            meta.setdefault("raw_path", s)
        if re.fullmatch(r"[A-Za-z0-9_:\-\.]+", s) and len(s)<=32 and s.isprintable():
            if "KH" in s.upper():
                meta.setdefault("sample_candidate", s)

    # Heuristic chromatogram
    csv_path = heuristics_chromatogram(b, outdir)

    report = {
        "source_file": str(src),
        "size_bytes": len(b),
        "tokens_found": len(tokens),
        "segments_written": len(index),
        "candidate_chromatogram_csv": csv_path,
        "metadata_guess": meta,
    }
    (outdir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (outdir / "strings.txt").write_text("\n".join(strs), encoding="utf-8")

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

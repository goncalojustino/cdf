#!/usr/bin/env python3
import argparse, json, os, re, sys, csv
from pathlib import Path

def read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

def ascii_strings(b: bytes, minlen=4):
    out=[]; cur=[]
    for ch in b:
        if 32 <= ch < 127: cur.append(chr(ch))
        else:
            if len(cur)>=minlen: out.append(''.join(cur))
            cur=[]
    if len(cur)>=minlen: out.append(''.join(cur))
    return out

def try_decode(block: bytes):
    tries=[]
    for enc in ("utf-8","latin-1","utf-16le","utf-16be"):
        s = block.decode(enc, errors="ignore").strip()
        if s: tries.append((enc,s))
    seen=set(); dedup=[]
    for enc,s in tries:
        if s not in seen:
            dedup.append((enc,s)); seen.add(s)
    return dedup

def find_tokens(b: bytes):
    toks=[(m.start(), m.group()) for m in re.finditer(rb"E[A-Za-z0-9_.\-]+", b)]
    segs=[]
    for i,(pos,tok) in enumerate(toks):
        start=pos
        end=toks[i+1][0] if i+1<len(toks) else len(b)
        segs.append((start,end,tok))
    return toks, segs

def write_block(outdir: Path, idx: int, tag: bytes, block: bytes):
    tag_txt = tag.decode("latin-1", errors="ignore")
    safe = re.sub(r"[^A-Za-z0-9_.\-]+","_", tag_txt).strip("_")
    base = outdir / f"{idx:03d}_{safe}"
    (base.with_suffix(".bin")).write_bytes(block)
    dec = try_decode(block)
    if dec:
        dec_sorted = sorted(dec, key=lambda x: len(x[1]), reverse=True)
        (base.with_suffix(".txt")).write_text(dec_sorted[0][1], encoding="utf-8")
    return str(base)

def heuristics_chromatogram(b: bytes, outdir: Path):
    # unchanged: keeps best-effort float pairing if present
    import numpy as np
    def scan(dtype, minlen=100):
        item = np.dtype(dtype).itemsize
        n = len(b)//item
        arr = np.frombuffer(memoryview(b)[:n*item], dtype=dtype)
        cands=[]; i=0
        while i < len(arr)-minlen:
            w = arr[i:i+minlen]
            if not np.all(np.isfinite(w)): i += minlen; continue
            if np.all(np.diff(w) > 0):
                j=i+minlen
                while j<len(arr) and np.isfinite(arr[j]) and arr[j]>arr[j-1]: j+=1
                seq=arr[i:j]; d=np.diff(seq)
                if len(d)>5 and float(np.std(d)) < 10*float(np.median(d)+1e-9):
                    cands.append((i,j,seq))
                i=j
            else:
                i += minlen
        return arr, cands
    for dtype in ("<f8","<f4"):
        try: arr,cands=scan(dtype)
        except Exception: cands=[]
        if not cands: continue
        cands.sort(key=lambda t: t[1]-t[0], reverse=True)
        for i0,j0,times in cands:
            span=float(times[-1]-times[0])
            if not (1e-6<span<1e6): continue
            L=j0-i0; w=min(500000,len(arr)); s=max(0,i0-w); e=min(len(arr), j0+w)
            k=s; best=None
            while k+L<=e:
                y=arr[k:k+L]
                if y.shape[0]==L and (y.max(initial=0)-y.min(initial=0))>0 and not ( (y[1:]-y[:-1]>=0).all() or (y[1:]-y[:-1]<=0).all() ):
                    best=(k,y); break
                k += max(1, L//50)
            if best:
                csv_path = outdir/"candidate_chromatogram.csv"
                with open(csv_path,"w",newline="") as fh:
                    wri=csv.writer(fh); wri.writerow(["index","time","intensity"])
                    for n,(t,v) in enumerate(zip(times.astype(float), best[1].astype(float))):
                        wri.writerow([n,float(t),float(v)])
                return str(csv_path)
    return None

def ew_trace_from_block(block: bytes, outdir: Path, src_stem: str):
    """
    Ew payload extraction. Two layouts supported:
      A) 8-byte records: [flag:1][pad:5 zeros][intensity:uint16 LE]
      B) 6-byte records: [tick:uint32][intensity:int16] (LE/BE)
    Writes <stem>_Ew_trace.csv with available fields.
    """
    import numpy as np
    payload = block[2:] if block.startswith(b"Ew") else block
    # Try 8-byte format first
    if len(payload) >= 8*200:
        n8 = len(payload)//8
        rec = np.frombuffer(memoryview(payload)[:n8*8], dtype=np.uint8).reshape(-1,8)
        pad_zeros_ratio = (rec[:,1:6]==0).mean()
        if pad_zeros_ratio > 0.9:
            inten_le = rec[:,6].astype(np.uint16) | (rec[:,7].astype(np.uint16) << 8)
            out_csv = outdir / f"{src_stem}_Ew_trace.csv"
            with open(out_csv, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["index","intensity_uint16","flag_byte"])
                for i in range(rec.shape[0]):
                    w.writerow([i, int(inten_le[i]), int(rec[i,0])])
            return str(out_csv)
    # Fallback: 6-byte search
    if len(payload) >= 6*200:
        for endian in ("<", ">"):
            dt = np.dtype([("tick", endian+"u4"), ("intensity", endian+"i2")])
            n6 = len(payload)//6
            arr = np.frombuffer(memoryview(payload)[:n6*6], dtype=dt)
            ticks = arr["tick"].astype(np.uint64)
            dif = np.diff(ticks)
            pos = dif > 0
            # longest increasing run
            max_len=0; max_end=0; cur=0
            for i, ok in enumerate(pos, start=1):
                if ok:
                    cur += 1
                    if cur>max_len:
                        max_len=cur; max_end=i+1
                else:
                    cur=0
            L = max_len+1 if max_len>0 else 0
            if L >= 200:
                start_idx = max_end-L
                out_csv = outdir / f"{src_stem}_Ew_trace.csv"
                with open(out_csv, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["index","raw_tick_uint32","intensity_int16","endianness"])
                    for i,(t,v) in enumerate(zip(arr["tick"][start_idx:max_end], arr["intensity"][start_idx:max_end])):
                        w.writerow([i, int(t), int(v), "LE" if endian=="<" else "BE"])
                return str(out_csv)
    return None

def extract_ew_trace_all(b: bytes, segs, outdir: Path, src_stem: str):
    ew_segs = [(start,end,tag) for (start,end,tag) in segs if tag.startswith(b"Ew")]
    if not ew_segs: return None
    # choose largest Ew segment
    start,end,tag = max(ew_segs, key=lambda t: t[1]-t[0])
    block = b[start:end]
    return ew_trace_from_block(block, outdir, src_stem)

def main():
    ap = argparse.ArgumentParser(description="Varian STAR .run extractor")
    ap.add_argument("run_file", help=".run file")
    ap.add_argument("-o","--outdir", default=None, help="output directory")
    args = ap.parse_args()

    src = Path(args.run_file)
    if not src.exists():
        print("Input not found", file=sys.stderr); sys.exit(1)
    outdir = Path(args.outdir) if args.outdir else src.with_suffix("")
    outdir.mkdir(parents=True, exist_ok=True)

    b = read_bytes(src)

    tokens, segs = find_tokens(b)
    index=[]
    for n,(start,end,tag) in enumerate(segs, start=1):
        block=b[start:end]
        pathbase=write_block(outdir, n, tag, block)
        index.append({
            "order": n, "tag": tag.decode("latin-1","ignore"),
            "start": start, "end": end, "size": end-start,
            "dump_base": Path(pathbase).name
        })

    strs=ascii_strings(b, minlen=4)
    meta={}
    import re as _re
    for s in strs:
        if s.startswith("Varian"): meta.setdefault("vendor", s)
        if _re.search(r"\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}", s, _re.I):
            meta.setdefault("date_like", s)
        if s.lower().endswith(".mth"): meta.setdefault("method_path", s)
        if _re.search(r"[A-Za-z]:\\\\star\\\\data\\\\", s, _re.I): meta.setdefault("raw_path", s)

    csv_path = heuristics_chromatogram(b, outdir)
    ew_csv = extract_ew_trace_all(b, segs, outdir, src.stem)

    report = {
        "source_file": str(src),
        "size_bytes": len(b),
        "tokens_found": len(tokens),
        "segments_written": len(index),
        "candidate_chromatogram_csv": csv_path,
        "ew_trace_csv": ew_csv,
        "metadata_guess": meta,
    }
    (outdir/"index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (outdir/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (outdir/"strings.txt").write_text("\n".join(strs), encoding="utf-8")

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

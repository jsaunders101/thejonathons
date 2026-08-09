#!/usr/bin/env python3
"""
thorcam_meta.py — dump the TIFF metadata of ONE ThorCam stack to a CSV.

Point it at a folder of ThorCam .tif parts; it reads a single multipage stack
(the first by natural sort, or --file / --index) and writes one CSV row per
frame with every TIFF tag on that frame, plus derived timing columns.

ThorCam writes no standard metadata — no ImageDescription, no DateTime, no
ImageJ/OME block. What it does write is private tags 32768-32782, three of
which vary per frame:

    32779   frame sequence counter, continuous ACROSS the parts of a recording
    32781   high word of a 64-bit hardware timestamp
    32782   low  word (wraps at 2**32; 32781 increments on each wrap)

When both timestamp tags are present the script adds timestamp_ns
(= 32781 * 2**32 + 32782), t_seconds relative to the first frame, and
delta_ms. The clock ticks in whole milliseconds, so at 15 fps consecutive
deltas alternate 66/67 ms around the true 66.667 ms period; the summary's frame
rate is the mean of all intervals with gaps excluded, and any excluded gap is
printed so a dropped frame is never averaged away silently.

Usage:
  python thorcam_meta.py "/path/to/folder"
  python thorcam_meta.py "/path/to/folder" --list
  python thorcam_meta.py "/path/to/folder" --index 3
  python thorcam_meta.py "/path/to/folder" --file behavior1_..._0.tif -o meta.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import tifffile
except ImportError:
    sys.exit("thorcam_meta.py requires tifffile (pip install tifffile)")

TS_HI, TS_LO, SEQ = 32781, 32782, 32779


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", p.name)]


def find_tifs(folder):
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    if not folder.is_dir():
        sys.exit(f"Not found: {folder}")
    tifs = [f for f in folder.iterdir()
            if f.suffix.lower() in (".tif", ".tiff")
            and not f.name.startswith(".")]
    if not tifs:
        sys.exit(f"No .tif files in {folder}")
    return sorted(tifs, key=natural_key)


def column_name(tag):
    """'32779' -> 'tag32779'; 'ImageWidth' -> 'tag256_ImageWidth'."""
    return (f"tag{tag.code}" if str(tag.name) == str(tag.code)
            else f"tag{tag.code}_{tag.name}")


def read_pages(path):
    """-> (list of per-page {column: value} dicts, page count, shape, dtype)."""
    rows = []
    with tifffile.TiffFile(str(path)) as tf:
        page0 = tf.pages[0]
        shape, dtype = tuple(page0.shape), page0.dtype
        for i, page in enumerate(tf.pages):
            row = {"page": i}
            for tag in page.tags:
                v = tag.value
                row[column_name(tag)] = (str(v) if isinstance(v, (tuple, list, bytes))
                                         else v)
            rows.append(row)
    return rows, len(rows), shape, dtype


def col_for(row, code):
    """Column holding tag `code`, whatever tifffile named it.

    Necessary because tifffile knows 32781 as 'ImageID' (so the column is
    tag32781_ImageID) while 32782 is nameless (tag32782) — matching on the
    bare 'tag<code>' string silently misses the named one.
    """
    exact, prefix = f"tag{code}", f"tag{code}_"
    for k in row:
        if k == exact or k.startswith(prefix):
            return k
    return None


def add_timing(rows):
    """Add timestamp_ns / t_seconds / delta_ms when the clock tags exist."""
    if not rows:
        return False
    hi, lo = col_for(rows[0], TS_HI), col_for(rows[0], TS_LO)
    if hi is None or lo is None:
        return False
    for r in rows:
        r["timestamp_ns"] = r[hi] * (1 << 32) + r[lo]
    t0 = rows[0]["timestamp_ns"]
    prev = None
    for r in rows:
        r["t_seconds"] = round((r["timestamp_ns"] - t0) / 1e9, 6)
        r["delta_ms"] = "" if prev is None else round(
            (r["timestamp_ns"] - prev) / 1e6, 3)
        prev = r["timestamp_ns"]
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of ThorCam .tif parts (or one .tif)")
    ap.add_argument("--file", help="stack to read, by filename (default: first)")
    ap.add_argument("--index", type=int, default=0,
                    help="stack to read, by position in natural sort (default 0)")
    ap.add_argument("--list", action="store_true",
                    help="list the stacks in the folder and exit")
    ap.add_argument("-o", "--out", help="output CSV (default: <stack>_metadata.csv "
                                        "next to the stack)")
    args = ap.parse_args()

    tifs = find_tifs(args.folder)
    if args.list:
        print(f"{len(tifs)} stacks in {args.folder}:")
        for i, f in enumerate(tifs):
            print(f"  [{i:3d}] {f.name}  ({f.stat().st_size / 2**20:.1f} MB)")
        return

    if args.file:
        match = [f for f in tifs if f.name == args.file]
        if not match:
            sys.exit(f"{args.file} not in {args.folder} (try --list)")
        path = match[0]
    else:
        if not 0 <= args.index < len(tifs):
            sys.exit(f"--index {args.index} out of range (0..{len(tifs)-1})")
        path = tifs[args.index]

    rows, n, shape, dtype = read_pages(path)
    timed = add_timing(rows)

    out = Path(args.out) if args.out else path.with_name(path.stem + "_metadata.csv")
    cols = list(rows[0].keys())
    for r in rows:                       # union, in case later pages carry extra tags
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- summary
    print(f"stack      : {path.name}")
    print(f"frames     : {n}")
    print(f"frame size : {shape[1]} x {shape[0]} px, {dtype}"
          if len(shape) == 2 else f"page shape : {shape}, {dtype}")
    varying = [c for c in cols
               if c not in ("page", "timestamp_ns", "t_seconds", "delta_ms")
               and len({r.get(c) for r in rows}) > 1]
    print(f"tags       : {len([c for c in cols if c.startswith('tag')])} "
          f"({len(varying)} vary per frame: {', '.join(varying) or 'none'})")
    if timed:
        span_ns = rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"]
        deltas = [rows[i + 1]["timestamp_ns"] - rows[i]["timestamp_ns"]
                  for i in range(n - 1)]
        if deltas and span_ns > 0:
            print(f"duration   : {span_ns / 1e9:.3f} s")
            # Mean of the deltas, dropping gaps (a dropped frame or a stalled
            # first frame). A plain span/(n-1) would let one 138 ms hiccup drag
            # the whole estimate down; the median alone can't undo the 1 ms
            # clock quantisation, which is why this averages the survivors.
            # 1.5x, not 3x: ONE dropped frame only doubles the interval, and a
            # 3x cutoff would quietly average that 2x gap into the rate.
            med = sorted(deltas)[len(deltas) // 2]
            keep = [d for d in deltas if d <= 1.5 * med]
            period_ms = sum(keep) / len(keep) / 1e6
            print(f"frame rate : {1000 / period_ms:.4f} fps "
                  f"(period {period_ms:.4f} ms, from {len(keep)}/{len(deltas)} "
                  f"intervals)")
            for i, dv in enumerate(deltas):
                if dv > 1.5 * med:
                    print(f"  GAP at page {i} -> {i+1}: {dv / 1e6:.0f} ms "
                          f"(~{dv / med:.1f} frames) — dropped frame or stall")
        seq = col_for(rows[0], SEQ)
        if seq:
            print(f"seq range  : {rows[0][seq]} .. {rows[-1][seq]} "
                  f"(continuous across the parts of a recording)")
    else:
        print("timing     : no 32781/32782 clock tags — timing columns omitted")
    print(f"\nwrote {out}  ({n} rows x {len(cols)} columns)")


if __name__ == "__main__":
    main()

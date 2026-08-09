#!/usr/bin/env python3
"""
ca_extract.py — batchable calcium pipeline (CSHL M1-Arousal / the Jonathons).

Faithful to the course notebook (Portugues lab / CSHL TAs), engineered for batch:
  load Bruker/PrairieView OME t-series (auto-detect the single recorded channel)
  -> two-pass rigid motion correction (phase cross-correlation, subpixel)
  -> pixel-neighbor correlation map (streaming exact Pearson — no extra movie copy)
  -> greedy region-growing ROIs (identical algorithm; fixes a notebook bug where the
     seed-pixel trace was a VIEW into the movie and += mutated the data in place)
  -> F_raw + dF/F (static bottom-percentile F0) + QC figures + full XML metadata.

MODULAR INPUT — the same tool takes:
  run    one t-series folder      (troubleshoot / validate on a ground-truth run)
  batch  a parent folder          (recursive discovery: any folder with TSeries*.xml)

OUTPUT MIRRORING — outputs NEVER land in the data tree. --out OUTPUT_ROOT receives an
EXACT mirror of the input directory hierarchy (every directory is created, including
empty ones, so behavior outputs can slot into the same mirrored tree later). Each
t-series' outputs go inside its mirrored folder:

    <out>/<...mirrored path...>/TSeries-xxx/
        ca.npz  ca.mat        F_raw, roi_npix, dff, roi_map, roi_xy, shifts,
                              mean_img, corr_map, frame_times, fps, params, provenance
        meta.json             ALL PrairieView XML metadata + companion-file inventory
        qc.json               metrics + flags
        qc_motion.png  qc_roi.png

Usage:
  python ca_extract.py run   <tseries>  --out <output_root> [--data-root <root>] [--force]
  python ca_extract.py batch <parent>   --out <output_root> [--force]
  python ca_extract.py batch <parent>   --out <o> --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
  python ca_extract.py list  <parent>   [--save runs.txt]
  python ca_extract.py mirror <parent>  --out <output_root>     # just create the tree

Params: defaults below; override any subset with --params params.json.
NOTE: Thorlabs-recorded t-series are NOT yet supported (different layout/XML) — they are
reported as unrecognized by `list`; send a folder listing to wire the loader.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("ca_extract.py requires tifffile")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARAMS_DEFAULT = {
    "upsample_factor": 10,       # subpixel MC precision (0.1 px)
    "mc_normalization": None,    # phase_cross_correlation normalization. None (classic
                                 # cross-correlation) measured ~10x more accurate than
                                 # skimage's default "phase" on ground-truth synthetic
                                 # 2P-like data (0.06 vs 0.6 px RMS) — see test suite
    "corr_threshold": 0.3,       # ROI growth: pixel-vs-ROI-trace correlation
    "size_range": [15, 100],     # accept ROIs within this pixel count (exclusive)
    "size_threshold": 200,       # hard stop for region growth
    "n_rois": 100,               # seed attempts
    "border_px": 5,              # corr-map border zeroing (auto-raised to max shift+1)
    "dff_percentile": 15,        # static F0 = bottom 15% of each ROI's F_raw
    "flag_max_shift_px": 10,     # motion flag: p99 of total shift exceeds this
    "flag_min_rois": 10,         # extraction flag: fewer ROIs than this
}


# ------------------------------------------------------------------ helpers

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", Path(p).name)]


def code_version():
    try:
        src = Path(__file__).read_bytes()
        return hashlib.sha1(src).hexdigest()[:12]
    except OSError:
        return "unknown"


def params_hash(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# ------------------------------------------------------- discovery & mirror

def is_tseries(folder):
    return bool(list(Path(folder).glob("TSeries*.xml")))


def find_tseries(root):
    """Every folder under root containing TSeries*.xml. Never descends into one."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if any(re.match(r"TSeries.*\.xml$", f) for f in filenames):
            found.append(Path(dirpath))
            dirnames[:] = []
    return sorted(found, key=lambda p: str(p))


def mirror_tree(data_root, out_root):
    """Recreate the ENTIRE input directory hierarchy under out_root — including empty
    directories (future behavior-output slots). Does not descend inside t-series."""
    data_root, out_root = Path(data_root), Path(out_root)
    n = 0
    out_root.mkdir(parents=True, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(data_root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        rel = Path(dirpath).relative_to(data_root)
        (out_root / rel).mkdir(parents=True, exist_ok=True)
        n += 1
        if is_tseries(dirpath):
            dirnames[:] = []          # mirror the t-series dir itself, not its guts
    return n


def out_dir_for(ts, data_root, out_root):
    ts, data_root = Path(ts).resolve(), Path(data_root).resolve()
    try:
        rel = ts.relative_to(data_root)
    except ValueError:
        sys.exit(f"{ts} is not under data root {data_root} — pass the right --data-root.")
    d = Path(out_root) / rel
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------- XML metadata

def parse_pv_xml(xml_path):
    """Parse EVERYTHING useful from a PrairieView t-series XML."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def statevalue(el):
        if el.get("value") is not None:
            return el.get("value")
        idx = {}
        for iv in el:
            key = iv.get("index") or iv.get("subindex") or iv.tag
            if iv.get("value") is not None and len(iv) == 0:
                idx[key] = (iv.get("value") if iv.get("description") is None
                            else {"value": iv.get("value"),
                                  "description": iv.get("description")})
            else:
                idx[key] = statevalue(iv)
        return idx

    shard = {}
    first_shard = root.find(".//PVStateShard")
    if first_shard is not None:
        for sv in first_shard.findall("PVStateValue"):
            shard[sv.get("key")] = statevalue(sv)

    frames = root.findall(".//Frame")
    rel_t = np.array([float(f.get("relativeTime")) for f in frames
                      if f.get("relativeTime") is not None])
    abs_t = np.array([float(f.get("absoluteTime")) for f in frames
                      if f.get("absoluteTime") is not None])
    if len(rel_t) < 2:
        raise RuntimeError(f"no per-frame times parsed from {xml_path}")
    period = float(np.median(np.diff(rel_t)))

    seq = root.find(".//Sequence")
    meta = {
        "xml": str(xml_path),
        "scan_date": root.get("date"),
        "sequence_time": seq.get("time") if seq is not None else None,
        "sequence_type": seq.get("type") if seq is not None else None,
        "n_frames_xml": len(frames),
        "frame_period_s": period,
        "fps": 1.0 / period,
        "duration_s": float(rel_t[-1] - rel_t[0] + period),
        "relative_times_first_last": [float(rel_t[0]), float(rel_t[-1])],
        "pv_state": shard,
    }
    return meta, rel_t, abs_t


def inventory_companions(ts):
    """List every non-imaging file in the t-series folder (VoltageRecording etc.)."""
    inv = []
    for f in sorted(Path(ts).rglob("*")):
        if f.is_file() and not f.name.startswith(".") \
           and not f.name.lower().endswith((".tif", ".tiff")):
            inv.append({"file": str(f.relative_to(ts)), "bytes": f.stat().st_size,
                        "voltage_recording": "voltagerecording" in f.name.lower()})
    return inv


# ------------------------------------------------------------------ loading

def find_channel_files(ts):
    files = sorted([f for f in Path(ts).iterdir()
                    if f.name.lower().endswith((".ome.tif", ".ome.tiff"))
                    and not f.name.startswith("._")], key=natural_key)
    if not files:
        raise RuntimeError(f"no .ome.tif files in {ts} (Thorlabs data? not yet supported)")
    chans = sorted({m.group(1) for f in files
                    for m in [re.search(r"_Ch(\d+)_", f.name)] if m})
    if len(chans) > 1:
        raise RuntimeError(f"multiple channels {chans} in {ts} — expected one; "
                           f"restrict files or extend loader")
    ch = chans[0] if chans else None
    if ch:
        files = [f for f in files if f"_Ch{ch}_" in f.name]
    return files, ch


def load_movie(ts, n_expected):
    """Load the t-series; assert frame count matches the XML (catches partial data)."""
    files, ch = find_channel_files(ts)
    arr = tifffile.imread(str(files[0]))          # OME: usually aggregates the series
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[0] != n_expected and len(files) > 1:
        first = arr
        arr = np.empty((n_expected, *first.shape[1:]), dtype=first.dtype)
        k = 0
        for f in files:
            a = tifffile.imread(str(f))
            if a.ndim == 2:
                a = a[None]
            if k + a.shape[0] > n_expected:
                raise RuntimeError(f"{ts}: more frames on disk than XML reports "
                                   f"({k + a.shape[0]} > {n_expected})")
            arr[k:k + a.shape[0]] = a
            k += a.shape[0]
        if k != n_expected:
            raise RuntimeError(f"{ts}: loaded {k} frames but XML reports {n_expected} "
                               f"— partial upload or wrong channel?")
    if arr.shape[0] != n_expected:
        raise RuntimeError(f"{ts}: loaded {arr.shape[0]} frames but XML reports "
                           f"{n_expected} — refusing to process silently truncated data")
    return arr, ch


# --------------------------------------------------------- motion correction

def compute_shifts(mov, template, upsample, normalization=None):
    from skimage.registration import phase_cross_correlation
    ys = np.empty(len(mov))
    xs = np.empty(len(mov))
    kw = {"upsample_factor": upsample}
    try:
        phase_cross_correlation(template, mov[0], normalization=normalization, **kw)
        kw["normalization"] = normalization
    except TypeError:                      # very old skimage: no normalization kwarg
        pass
    for i in range(len(mov)):
        r, _, _ = phase_cross_correlation(template, mov[i], **kw)
        ys[i], xs[i] = r[0], r[1]
    return ys, xs


def motion_correct(mov, upsample, normalization=None):
    """Two-pass rigid MC (as in the notebook). Memory: raw stays uint16; only ONE
    float32 aligned copy is materialized (pass-1 template is a streamed accumulator)."""
    from scipy.ndimage import shift as ndshift

    template1 = mov.mean(0)
    y1, x1 = compute_shifts(mov, template1, upsample, normalization)

    acc = np.zeros(mov.shape[1:], dtype=np.float64)
    for i in range(len(mov)):                      # pass-1 aligned MEAN only
        acc += ndshift(mov[i].astype(np.float32), (y1[i], x1[i]))
    template2 = acc / len(mov)

    y2, x2 = compute_shifts(mov, template2, upsample, normalization)
    final = np.empty(mov.shape, dtype=np.float32)
    for i in range(len(mov)):
        final[i] = ndshift(mov[i].astype(np.float32), (y2[i], x2[i]))
    return final, (y1, x1), (y2, x2), template1, final.mean(0)


# --------------------------------------------- correlation map (streaming)

def neighbor_corr_map(mov):
    """Exact Pearson correlation of each pixel with the SUM of its 8 neighbors,
    computed in one streaming pass (no z-scored movie copy)."""
    from scipy.ndimage import convolve
    T = len(mov)
    Y, X = mov.shape[1:]
    k = np.ones((3, 3))
    sx = np.zeros((Y, X)); sxx = np.zeros((Y, X))
    ss = np.zeros((Y, X)); sss = np.zeros((Y, X)); sxs = np.zeros((Y, X))
    for t in range(T):
        f = mov[t].astype(np.float64)
        s = convolve(f, k, mode="constant") - f     # sum of 8 neighbors
        sx += f; sxx += f * f
        ss += s; sss += s * s
        sxs += f * s
    num = T * sxs - sx * ss
    den = np.sqrt((T * sxx - sx ** 2) * (T * sss - ss ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return r


# --------------------------------------------------------------- ROI growth

def fast_pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def next_roi(corr_map, mov, corr_threshold, size_threshold):
    """Greedy region growing — same algorithm as the notebook. Two changes:
    .copy() on the seed trace (the notebook version mutated the movie in place via a
    view), and dot-product Pearson (identical result, no scipy call per pixel)."""
    from skimage.morphology import binary_dilation
    i, j = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    this_trace = mov[:, i, j].astype(np.float64).copy()
    this_roi = np.zeros(corr_map.shape, dtype=np.uint8)
    this_roi[i, j] = 1
    this_corr_map = corr_map.copy()
    this_corr_map[i, j] = 0

    growing = True
    while growing and (this_roi.sum() < size_threshold):
        growing = False
        dilated = binary_dilation(this_roi, np.ones((3, 3)))
        neighbours = np.argwhere(dilated ^ this_roi.astype(bool))
        trace_to_add = np.zeros(len(this_trace))
        for n in range(neighbours.shape[0]):
            i, j = neighbours[n]
            if this_corr_map[i, j] != 0:
                px = mov[:, i, j].astype(np.float64)
                if fast_pearson(this_trace, px) > corr_threshold:
                    growing = True
                    this_roi[i, j] = 1
                    this_corr_map[i, j] = 0
                    trace_to_add += px
        this_trace += trace_to_add
    return this_roi, this_trace, int(this_roi.sum()), this_corr_map


def extract_rois(corr_map, mov, p):
    traces, npix, roi_map = [], [], np.zeros(corr_map.shape, dtype=np.uint16)
    used = corr_map.copy()
    label = 0
    for _ in range(p["n_rois"]):
        roi, trace, size, used = next_roi(used, mov, p["corr_threshold"],
                                          p["size_threshold"])
        if p["size_range"][0] < size < p["size_range"][1]:
            label += 1
            traces.append(trace)
            npix.append(size)
            roi_map[roi.astype(bool)] = label
    F = np.asarray(traces, dtype=np.float32) if traces else np.zeros((0, len(mov)),
                                                                     np.float32)
    return F, np.asarray(npix, dtype=int), roi_map


def compute_dff(F, percentile):
    if not len(F):
        return F.copy(), np.zeros(0, np.float32)
    F0 = np.percentile(F, percentile, axis=1)
    F0 = np.maximum(F0, 1e-6)
    return ((F - F0[:, None]) / F0[:, None]).astype(np.float32), F0.astype(np.float32)


def roi_centroids(roi_map, n):
    xy = np.zeros((n, 2))
    for k in range(1, n + 1):
        ys, xs = np.nonzero(roi_map == k)
        xy[k - 1] = [xs.mean(), ys.mean()] if len(xs) else [np.nan, np.nan]
    return xy


# --------------------------------------------------------------------- QC

def qc_motion_png(path, t, ts1, ts2, before, after, flag):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(t, ts1, lw=0.6, label="pass 1")
    axes[0].plot(t, ts2, lw=0.6, label="pass 2")
    axes[0].set(xlabel="time (s)", ylabel="total shift (px)",
                title=f"motion  [{'FLAG' if flag else 'ok'}]")
    axes[0].legend(fontsize=8)
    for ax, img, name in ((axes[1], before, "before (mean)"),
                          (axes[2], after, "after (mean)")):
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1),
                  vmax=np.percentile(img, 99))
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def qc_roi_png(path, mean_img, corr_map, roi_map, dff, flag):
    n = int(roi_map.max())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].imshow(mean_img, cmap="gray", vmin=np.percentile(mean_img, 1),
                      vmax=np.percentile(mean_img, 99))
    m = np.ma.masked_where(roi_map == 0, roi_map)
    axes[0, 0].imshow(m, cmap="prism", alpha=0.45)
    axes[0, 0].set_title(f"{n} ROIs on mean image  [{'FLAG' if flag else 'ok'}]")
    axes[0, 1].imshow(corr_map, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("correlation map")
    for ax in axes[0]:
        ax.set_xticks([]); ax.set_yticks([])
    if len(dff):
        amp = dff.max(1) - np.median(dff, 1)
        order = np.argsort(amp)
        for ax, idxs, name in ((axes[1, 0], order[-3:][::-1], "best 3 (dF/F)"),
                               (axes[1, 1], order[:3], "worst 3 (dF/F)")):
            for o, k in enumerate(idxs):
                ax.plot(dff[k] + o * max(1.0, np.ptp(dff[k])), lw=0.5)
            ax.set_title(name); ax.set_xlabel("frame")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- pipeline

def process_tseries(ts, data_root, out_root, params, force=False):
    ts = Path(ts)
    od = out_dir_for(ts, data_root, out_root)
    out_npz = od / "ca.npz"
    if out_npz.exists() and not force:
        print(f"[{ts.name}] ca.npz exists — skipping (--force to redo).")
        return True

    xmls = sorted(ts.glob("TSeries*.xml"))
    main_xml = min(xmls, key=lambda p: len(p.name))   # shortest = the t-series XML
    meta, rel_t, abs_t = parse_pv_xml(main_xml)
    meta["companion_files"] = inventory_companions(ts)
    meta["source_rel_path"] = str(ts.resolve().relative_to(Path(data_root).resolve()))

    print(f"[{ts.name}] {meta['n_frames_xml']} frames @ {meta['fps']:.2f} Hz "
          f"({meta['duration_s']:.0f}s) — loading...")
    mov, ch = load_movie(ts, meta["n_frames_xml"])
    meta["channel"] = ch
    meta["frame_shape"] = list(mov.shape[1:])
    print(f"[{ts.name}] movie {mov.shape} {mov.dtype}; Ch{ch}; motion correcting...")

    final, (y1, x1), (y2, x2), anat0, anat = motion_correct(
        mov, params["upsample_factor"], params["mc_normalization"])
    total1 = np.sqrt(y1 ** 2 + x1 ** 2)
    total2 = np.sqrt(y2 ** 2 + x2 ** 2)
    del mov

    print(f"[{ts.name}] correlation map...")
    cmap_full = neighbor_corr_map(final)
    border = max(params["border_px"], int(np.ceil(np.abs([y2, x2]).max())) + 1)
    cmap = cmap_full.copy()
    cmap[:border, :] = 0; cmap[-border:, :] = 0
    cmap[:, :border] = 0; cmap[:, -border:] = 0

    print(f"[{ts.name}] ROI extraction...")
    F, npix, roi_map = extract_rois(cmap, final, params)
    dff, F0 = compute_dff(F, params["dff_percentile"])
    xy = roi_centroids(roi_map, len(F))
    del final

    flags = {
        "motion": bool(np.percentile(total2, 99) > params["flag_max_shift_px"]),
        "few_rois": bool(len(F) < params["flag_min_rois"]),
        "frame_period_jitter": bool(np.std(np.diff(rel_t)) > 0.1 * meta["frame_period_s"]),
    }
    qc = {
        "n_rois": int(len(F)),
        "shift_p99_px": float(np.percentile(total2, 99)),
        "shift_max_px": float(total2.max()),
        "border_px_used": int(border),
        "flags": flags,
        "flagged": any(flags.values()),
    }

    prov = {"params": params, "params_hash": params_hash(params),
            "code_version": code_version(),
            "processed": datetime.now().isoformat(timespec="seconds")}

    tmp = out_npz.with_name(out_npz.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, F_raw=F, roi_npix=npix, dff=dff, F0=F0, roi_map=roi_map, roi_xy=xy,
            x_shift1=x1, y_shift1=y1, x_shift2=x2, y_shift2=y2,
            mean_img=anat.astype(np.float32), mean_img_raw=anat0.astype(np.float32),
            corr_map=cmap_full.astype(np.float32),
            frame_times=rel_t.astype(np.float64), fps=meta["fps"],
            source_rel_path=meta["source_rel_path"], provenance=json.dumps(prov))
    os.replace(tmp, out_npz)

    try:
        from scipy.io import savemat
        mtmp = od / "ca.mat.tmp"
        with open(mtmp, "wb") as fh:
            savemat(fh, {"F_raw": F, "roi_npix": npix, "dff": dff, "F0": F0,
                         "roi_map": roi_map, "roi_xy": xy,
                         "x_shift": x2, "y_shift": y2,
                         "mean_img": anat, "corr_map": cmap_full,
                         "frame_times": rel_t, "fps": meta["fps"],
                         "source_rel_path": meta["source_rel_path"]})
        os.replace(mtmp, od / "ca.mat")
    except ImportError:
        print("  (scipy missing — no .mat)")

    atomic_json(od / "meta.json", meta)
    atomic_json(od / "qc.json", {**qc, **prov})
    t = rel_t - rel_t[0]
    qc_motion_png(od / "qc_motion.png", t, total1, total2, anat0, anat, flags["motion"])
    qc_roi_png(od / "qc_roi.png", anat, cmap_full, roi_map, dff, flags["few_rois"])

    status = "FLAGGED" if qc["flagged"] else "ok"
    print(f"[{ts.name}] {len(F)} ROIs, shift p99 {qc['shift_p99_px']:.2f}px "
          f"[{status}] -> {od}")
    return not qc["flagged"]


# --------------------------------------------------------------------- cli

def load_params(path):
    p = dict(PARAMS_DEFAULT)
    if path:
        override = json.loads(Path(path).read_text())
        unknown = set(override) - set(p)
        if unknown:
            sys.exit(f"unknown params: {sorted(unknown)}")
        p.update(override)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="process ONE t-series folder")
    p.add_argument("tseries")
    p.add_argument("--out", required=True, help="output root (mirrored tree)")
    p.add_argument("--data-root", default=None,
                   help="root the mirror is relative to (default: 3 levels above the "
                        "t-series, i.e. <root>/<mouse>/<run>/<tseries>)")
    p.add_argument("--params", default=None)
    p.add_argument("--force", action="store_true")

    b = sub.add_parser("batch", help="discover + process every t-series under a parent")
    b.add_argument("parent")
    b.add_argument("--out", required=True)
    b.add_argument("--params", default=None)
    b.add_argument("--force", action="store_true")
    b.add_argument("--runs-file", default=None, help="frozen list from `list --save`")
    b.add_argument("--run-index", type=int, default=None)

    li = sub.add_parser("list", help="show every t-series under a parent")
    li.add_argument("parent")
    li.add_argument("--save", default=None)

    mi = sub.add_parser("mirror", help="create the mirrored output tree only")
    mi.add_argument("parent")
    mi.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "run":
        ts = Path(args.tseries).resolve()
        if not is_tseries(ts):
            sys.exit(f"{ts} has no TSeries*.xml — not a (Bruker) t-series folder.")
        data_root = Path(args.data_root).resolve() if args.data_root \
            else ts.parent.parent.parent
        print(f"data root: {data_root}\nmirror path: {ts.relative_to(data_root)}")
        ok = process_tseries(ts, data_root, args.out, load_params(args.params),
                             force=args.force)
        sys.exit(0 if ok else 3)

    if args.cmd == "list":
        runs = find_tseries(args.parent)
        for i, r in enumerate(runs):
            print(f"{i:3d}  {r}")
        if args.save:
            Path(args.save).write_text("\n".join(str(r) for r in runs) + "\n")
            print(f"wrote {args.save} ({len(runs)} t-series)")
        return

    if args.cmd == "mirror":
        n = mirror_tree(args.parent, args.out)
        print(f"mirrored {n} directories under {args.out}")
        return

    if args.cmd == "batch":
        parent = Path(args.parent).resolve()
        mirror_tree(parent, args.out)
        if args.runs_file:
            runs = [Path(l) for l in Path(args.runs_file).read_text().splitlines()
                    if l.strip()]
        else:
            runs = find_tseries(parent)
        if args.run_index is not None:
            if args.run_index >= len(runs):
                sys.exit(f"--run-index {args.run_index} out of range ({len(runs)} runs)")
            runs = [runs[args.run_index]]
        params = load_params(args.params)
        n_flag, n_fail = 0, 0
        rows = []
        for ts in runs:
            try:
                ok = process_tseries(ts, parent, args.out, params, force=args.force)
                rows.append((str(ts), "flagged" if not ok else "ok"))
                n_flag += (not ok)
            except Exception as e:
                print(f"[{Path(ts).name}] ERROR: {e}")
                rows.append((str(ts), f"ERROR: {e}"))
                n_fail += 1
        summary = Path(args.out) / "ca_summary.csv"
        with open(summary, "a") as fh:
            for r in rows:
                fh.write(f'"{r[0]}","{r[1]}","{datetime.now().isoformat()}"\n')
        print(f"\n{len(runs)} runs: {len(runs)-n_flag-n_fail} ok, {n_flag} flagged, "
              f"{n_fail} errors. Summary appended to {summary}")
        if n_fail:
            sys.exit(2)


if __name__ == "__main__":
    main()

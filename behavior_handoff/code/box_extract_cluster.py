#!/usr/bin/env python3
"""
box_extract_cluster.py — box-ROI drawing (lightweight GUI) + per-box trace extraction
for behavior movies. Runs on the cluster; draw phase needs X-forwarding.

Canonical box categories (edit CANONICAL_BOXES below to change the button set):
    laser_trigger, whisker_pad, paw, paw_at_nose, eye, nose, wheel

paw_at_nose = box over the region in front of the snout where the paw appears
during face-grooming bouts; intensity + motion energy there report the
'pawing at nose' motif.

GUI: drag a rectangle on the first frame, then CLICK THE CATEGORY BUTTON to assign
it. No typing -> no naming typos. One box per category, enforced (assigning a
category twice is refused; use 'undo last'). 'DONE' warns once if any category is
unassigned; click DONE again to accept the missing ones.

ORGANIZATION (the contract for all downstream analysis)
Every derived file lives in a proc/ subfolder INSIDE the run folder it came from:

    <run>/                     raw ThorCam tifs (never written to)
    <run>/proc/
        boxes.json             box coords + categories
        <run>_boxtraces.npz    traces + motion_energy, n_boxes x n_frames
        <run>_boxtraces.mat    MATLAB copy (written by default)
        <run>_sync.json        from sync_detect_cluster.py
        <run>_syncqc.png       from sync_detect_cluster.py

A "run" = any folder that directly contains one or more .tif/.tiff files; multiple
tifs in a run are one movie, concatenated in natural-sort order.

Phases:
  draw     Interactive GUI, one first-frame per run (cheap). Writes <run>/proc/boxes.json.
           'reuse previous' button copies the last run's boxes. --boxes-from applies one
           boxes.json to every run with no GUI.
  extract  Headless, streams frames in bounded blocks (--batch-mb, default
           256 MB budget) so RAM stays flat on 80+ GB movies; single-IFD
           contiguous giants (ImageJ 'truncated' >4 GB files) are read via
           memmap. The last TRIM_TAIL_FRAMES (3) frames are dropped by policy —
           the camera writes them while being stopped — see --trim-tail.
           Writes <run>/proc/<run>_boxtraces.npz (+.mat).
           Parallelize with --run-index in a SLURM array.
  list     Show every run and its processing status (boxes / traces / sync).
  verify   Read the first + last frame of every tif to catch files whose tail
           is missing (upload still in flight, aborted recording). Run this
           BEFORE a batch on freshly uploaded data. Exit 2 if any file is short.
  collect  Copy all proc/ outputs (NOT raw movies) into one flat folder ON THE
           CLUSTER (staging for slides/sharing — nothing leaves the cluster),
           prefixing filenames with parent folders to stay unique.

Examples:
  python box_extract_cluster.py draw    /data/behavior/day1
  python box_extract_cluster.py draw    /data/behavior --boxes-from /data/behavior/day1/run01/proc/boxes.json
  python box_extract_cluster.py extract /data/behavior
  python box_extract_cluster.py extract /data/behavior --run-index $SLURM_ARRAY_TASK_ID
  python box_extract_cluster.py list    /data/behavior
  python box_extract_cluster.py collect /data/behavior --dest /data/behavior/_collected
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("box_extract_cluster.py requires tifffile (pip install tifffile)")

CANONICAL_BOXES = ["laser_trigger", "whisker_pad", "paw", "paw_at_nose", "eye",
                   "nose", "wheel"]

PROC = "proc"

# The ONLY line that differs between the two branches of this code:
# cluster/ ships "cluster" (trace writes guarded to Elzar), local/ ships "local"
# (guard off — for local rigs and tests). Keep everything else byte-identical.
DEPLOYMENT = "cluster"

BATCH_MB = 256   # frame-block read budget (MB) — bounds extraction memory

# Default policy (Aug 9): the camera is stopped by hand after the t-series, and
# the last frames written during shutdown are unreliable — the final one can be
# cut off mid-pixel-data ("failed to read N bytes, got M"). Drop the tail rather
# than trust it. Frames are dropped from the END ONLY, so every surviving frame
# keeps its original camera-frame index and the sync anchors stay valid.
# Override per-invocation with --trim-tail N (0 keeps everything).
TRIM_TAIL_FRAMES = 3

CLUSTER_ONLY_MSG = (
    "Trace writing is CLUSTER-ONLY (policy Aug 9): run extraction on Elzar, where "
    "the data lives. Any traces written elsewhere are vestigial tests — do not "
    "analyze them. Override for synthetic tests only: --allow-local or "
    "M1AROUSAL_ALLOW_LOCAL=1.")


def trace_writes_allowed():
    """Cluster-only policy (Aug 9): trace outputs are written ONLY on the cluster.
    Detected via the /grid filesystem, a SLURM context, or an Elzar hostname;
    M1AROUSAL_ALLOW_LOCAL=1 overrides (synthetic tests). The local/ branch
    (DEPLOYMENT="local") turns the guard off entirely."""
    if DEPLOYMENT == "local":
        return True
    if os.environ.get("M1AROUSAL_ALLOW_LOCAL"):
        return True
    if Path("/grid").exists():
        return True
    if any(k.startswith("SLURM_") for k in os.environ):
        return True
    import socket
    return socket.gethostname().lower().startswith(("bam", "elzar"))


# ---------------------------------------------------------------- discovery

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", Path(p).name)]


def is_tif_name(name):
    # skip macOS AppleDouble sidecars ("._foo.tif") that upload tools create
    return (name.lower().endswith((".tif", ".tiff"))
            and not name.startswith("._") and not name.startswith("."))


def run_tifs(folder):
    out = []
    for f in Path(folder).iterdir():
        if not is_tif_name(f.name):
            continue
        try:
            if f.stat().st_size == 0:
                continue
        except OSError:
            continue
        out.append(f)
    return sorted(out, key=natural_key)


def find_runs(roots):
    """A run = any folder directly containing >=1 tif. Roots may be runs themselves.

    Single os.walk pass with proc/ pruned — never enumerates a tree twice
    (cluster filesystems charge dearly for metadata).
    """
    runs = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            sys.exit(f"Not found: {root}")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d != PROC and not d.startswith("."))
            if any(is_tif_name(f) for f in filenames):
                runs.append(Path(dirpath))
    seen, out = set(), []
    for r in runs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def proc_dir(run):
    return Path(run) / PROC


def boxes_path(run):
    return proc_dir(run) / "boxes.json"


def to_gray(frame):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:          # RGB(A) safety — ThorCam is normally mono
        frame = frame[..., :3].mean(axis=-1)
    return frame


def read_first_frame(tif_path):
    with tifffile.TiffFile(str(tif_path)) as tf:
        return to_gray(tf.pages[0].asarray())


# --------------------------------------------------------------- validation

def validate_boxes(boxes, categories, where):
    """Hard-error on duplicates; warn on unknown/missing categories."""
    names = [b["name"] for b in boxes]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        sys.exit(f"ERROR [{where}]: duplicate box categories {dupes} — one box per "
                 f"category. Fix boxes.json / redraw with --overwrite.")
    unknown = [n for n in names if n not in categories]
    if unknown:
        print(f"WARNING [{where}]: non-canonical box names {unknown} "
              f"(canonical: {categories})")
    missing = [c for c in categories if c not in names]
    if missing:
        print(f"WARNING [{where}]: missing categories {missing}")
    coord_map = {}
    for b in boxes:
        coord_map.setdefault((b["x0"], b["y0"], b["x1"], b["y1"]), []).append(b["name"])
    for same in (v for v in coord_map.values() if len(v) > 1):
        print(f"WARNING [{where}]: {same} have IDENTICAL coordinates — "
              f"was a box accidentally assigned twice?")
    return missing


# ---------------------------------------------------------------- draw GUI

class BoxGUI:
    """Drag a rectangle, click a category button to assign it. One box per category."""

    def __init__(self, frame, categories, title="", prev_boxes=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button, RectangleSelector

        self._plt = plt
        self._mpatches = mpatches
        self.categories = list(categories)
        self.assigned = {}          # name -> [x0, y0, x1, y1]
        self.order = []             # assignment order, for undo
        self.artists = {}           # name -> [patch, label]
        self.prev_boxes = prev_boxes or []
        self._warned_missing = False
        self._has_selection = False
        self.h, self.w = frame.shape

        self.fig = plt.figure(figsize=(13, 8))
        self.ax = self.fig.add_axes([0.02, 0.06, 0.70, 0.88])
        vmin, vmax = np.percentile(frame, [1, 99.5])
        self.ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        self.ax.set_title(title, fontsize=11)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.msg = self.fig.text(0.02, 0.015,
                                 "Drag a box, then click a category button. TIP: use the "
                                 "toolbar magnifier to zoom in and identify features first "
                                 "(deselect it before drawing).",
                                 fontsize=11, color="tab:blue")

        self.selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True, button=[1], minspanx=3, minspany=3, interactive=True,
            props=dict(edgecolor="orange", fill=False, linewidth=1.5))

        self.buttons = {}
        x, wbtn, hbtn, gap = 0.76, 0.21, 0.052, 0.012
        y = 0.90
        for name in self.categories:
            bax = self.fig.add_axes([x, y, wbtn, hbtn])
            y -= hbtn + gap
            b = Button(bax, name)
            b.on_clicked(lambda event, n=name: self._on_assign(n))
            self.buttons[name] = b
        y -= 0.03
        if self.prev_boxes:
            bax = self.fig.add_axes([x, y, wbtn, hbtn])
            y -= hbtn + gap
            self.btn_reuse = Button(bax, "reuse previous")
            self.btn_reuse.on_clicked(lambda event: self._on_reuse())
        bax = self.fig.add_axes([x, y, wbtn, hbtn])
        y -= hbtn + gap
        self.btn_undo = Button(bax, "undo last")
        self.btn_undo.on_clicked(lambda event: self._on_undo())
        bax = self.fig.add_axes([x, y, wbtn, hbtn])
        self.btn_done = Button(bax, "DONE")
        self.btn_done.on_clicked(lambda event: self._on_done())
        self.btn_done.color = "lightgreen"
        self.btn_done.ax.set_facecolor("lightgreen")

    # -- helpers -----------------------------------------------------------
    def _say(self, text, color="tab:blue"):
        self.msg.set_text(text)
        self.msg.set_color(color)
        self.fig.canvas.draw_idle()

    def _current_rect(self):
        x0, x1, y0, y1 = self.selector.extents
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w), min(y1, self.h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [x0, y0, x1, y1]

    def _set_button_used(self, name, used):
        b = self.buttons[name]
        b.label.set_text(("* " + name) if used else name)
        col = "0.7" if used else "0.85"
        b.color = col
        b.ax.set_facecolor(col)

    def _add_box(self, name, rect, announce=True):
        self.assigned[name] = rect
        self.order.append(name)
        x0, y0, x1, y1 = rect
        patch = self._mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                         edgecolor="lime", linewidth=1.8)
        self.ax.add_patch(patch)
        label = self.ax.text(x0, y0 - 3, name, color="lime", fontsize=9,
                             fontweight="bold")
        self.artists[name] = [patch, label]
        self._set_button_used(name, True)
        if announce:
            self._say(f"'{name}' assigned ({len(self.assigned)}/{len(self.categories)}).")

    # -- callbacks ---------------------------------------------------------
    def _on_select(self, eclick, erelease):
        self._has_selection = True
        self._say("Box drawn — click a category button.")

    def _on_assign(self, name):
        if name in self.assigned:
            self._say(f"ERROR: '{name}' already assigned — one box per category "
                      f"(use 'undo last' to replace it).", "red")
            return
        # each drawn rectangle can be consumed exactly once: without this flag the
        # selector's stale extents would silently re-assign the previous box
        rect = self._current_rect() if self._has_selection else None
        if rect is None:
            self._say(f"Draw a NEW box for '{name}' first (drag on the image), "
                      f"then click the button.", "red")
            return
        self._add_box(name, rect)
        self._has_selection = False
        try:
            self.selector.clear()
        except AttributeError:
            pass
        self.fig.canvas.draw_idle()

    def _on_undo(self):
        if not self.order:
            self._say("Nothing to undo.", "red")
            return
        name = self.order.pop()
        del self.assigned[name]
        for a in self.artists.pop(name):
            a.remove()
        self._set_button_used(name, False)
        self._say(f"Removed '{name}'.")
        self.fig.canvas.draw_idle()

    def _on_reuse(self):
        n = 0
        for b in self.prev_boxes:
            if b["name"] in self.assigned or b["name"] not in self.categories:
                continue
            self._add_box(b["name"], [b["x0"], b["y0"], b["x1"], b["y1"]],
                          announce=False)
            n += 1
        self._say(f"Reused {n} previous boxes "
                  f"({len(self.assigned)}/{len(self.categories)}). Adjust with "
                  f"'undo last' + redraw, or click DONE.")
        self.fig.canvas.draw_idle()

    def _on_done(self):
        missing = [c for c in self.categories if c not in self.assigned]
        if missing and not self._warned_missing:
            self._warned_missing = True
            self._say(f"WARNING: missing {missing} — draw them, or click DONE again "
                      f"to accept without.", "darkorange")
            return
        self._plt.close(self.fig)

    # -- entry -------------------------------------------------------------
    def run(self):
        self._plt.show(block=True)
        missing = [c for c in self.categories if c not in self.assigned]
        if missing:
            print(f"  WARNING: categories not assigned: {missing}")
        return [{"name": n, "x0": r[0], "y0": r[1], "x1": r[2], "y1": r[3]}
                for n, r in ((c, self.assigned[c]) for c in self.categories
                             if c in self.assigned)]


def save_boxes(run, files, boxes, shape, categories):
    validate_boxes(boxes, categories, run.name)
    pd = proc_dir(run)
    pd.mkdir(exist_ok=True)
    out = boxes_path(run)
    out.write_text(json.dumps({
        "image_shape": [int(s) for s in shape],
        "drawn_on": files[0].name,
        "categories": categories,
        "created": datetime.now().isoformat(timespec="seconds"),
        "boxes": boxes,
    }, indent=2))
    print(f"  wrote {out}  ({len(boxes)} boxes: {', '.join(b['name'] for b in boxes)})")


def cmd_draw(args):
    categories = ([c.strip() for c in args.categories.split(",")]
                  if args.categories else CANONICAL_BOXES)
    runs = find_runs(args.roots)
    if not runs:
        sys.exit("No runs (folders containing tifs) found.")
    print(f"Found {len(runs)} runs. Categories: {categories}")

    template = None
    if args.boxes_from:
        template = json.loads(Path(args.boxes_from).read_text())["boxes"]
        validate_boxes(template, categories, args.boxes_from)

    prev = template
    for run in runs:
        files = run_tifs(run)
        bj = boxes_path(run)
        if bj.exists() and not args.overwrite:
            print(f"[{run.name}] proc/boxes.json exists, skipping (--overwrite to redo).")
            prev = json.loads(bj.read_text())["boxes"]
            continue
        if bj.exists():
            # overwriting: seed 'reuse previous' with this run's OWN boxes, so adding
            # a new category = one click to restore the old boxes + draw the new one
            prev = json.loads(bj.read_text())["boxes"]
        frame = read_first_frame(files[0])
        if args.boxes_from:
            save_boxes(run, files, [dict(b) for b in template], frame.shape, categories)
            continue
        gui = BoxGUI(frame, categories,
                     title=f"{run.name} — first frame ({files[0].name})",
                     prev_boxes=prev)
        boxes = gui.run()
        if not boxes:
            print(f"[{run.name}] no boxes assigned — nothing saved.")
            continue
        save_boxes(run, files, boxes, frame.shape, categories)
        prev = boxes


# ------------------------------------------------------------- extract phase

def truncated_memmap(tf, path, report=None, at_frame=0, quiet=False):
    """np.memmap over a single-IFD tif that actually holds a whole movie, or None.

    Writers that keep one file per movie (ImageJ convention above 4 GB, some
    ThorCam exports) emit ONE page header and append the remaining frames as raw
    contiguous data. tifffile shows len(pages)==1 while series[0] knows the real
    frame count; per-page streaming would silently yield a single frame. The
    memmap makes every frame addressable while reading only the bytes touched.

    The frame count is clamped to the bytes actually present, so an aborted
    recording (metadata promises more frames than were written) yields the
    frames that exist instead of failing.
    """
    try:
        if len(tf.pages) != 1:
            return None
        p0 = tf.pages[0]
        if (p0.dtype is None or not getattr(p0, "is_contiguous", False)
                or int(getattr(p0, "compression", 1)) != 1):
            return None
        shape = tuple(p0.shape)
        try:
            ser = tf.series[0]
            n = int(np.prod(ser.shape[:len(ser.shape) - len(shape)],
                            dtype=np.int64))
        except Exception:
            n = 1
        if tf.is_imagej and tf.imagej_metadata:
            # on an aborted recording tifffile invalidates the whole series
            # (shape collapses to one frame) but the declared count survives
            # in the ImageJ tag; the byte clamp below trims it to reality
            n = max(n, int(tf.imagej_metadata.get("images", 0)))
        if n <= 1:
            return None
        dt = np.dtype(p0.dtype).newbyteorder(tf.byteorder)
        offset = int(p0.dataoffsets[0])
        fbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
        avail = (os.path.getsize(path) - offset) // fbytes
        if avail < 1:
            return None
        if avail < n:
            if not quiet:
                note_truncation(report, path, at_frame + int(avail),
                                ValueError(f"declares {n} frames, only {avail} present"))
            n = int(avail)
        return np.memmap(path, dtype=dt, mode="r", offset=offset,
                         shape=(n,) + shape)
    except Exception:
        return None


def count_frames(files):
    """Total frames in a run, from METADATA only — no pixel data is read.

    Cheap enough to run before extraction, which is what lets --trim-tail stop
    short of the unreliable tail instead of reading it and throwing it away.
    """
    total = 0
    for f in files:
        with tifffile.TiffFile(str(f)) as tf:
            if len(tf.pages) == 1:
                mm = truncated_memmap(tf, str(f), quiet=True)
                total += len(mm) if mm is not None else 1
                del mm
            else:
                total += len(tf.pages)
    return total


def stream_frame_blocks(files, batch_mb=BATCH_MB, report=None, limit=None):
    """Yield frame blocks (n, H, W[, C]) in NATIVE dtype, each <= batch_mb MB.

    Bounded-memory streaming (OOM fix): extraction RAM is ~batch_mb regardless
    of movie size, so 80+ GB runs stay flat. Two read paths:
      - multi-page tifs (ThorCam parts): pages read one-by-one into a REUSED
        block buffer
      - single-IFD contiguous giants (see truncated_memmap): frames sliced
        lazily off a memmap — only the bytes touched are read

    Short files (interrupted upload, aborted recording) are SALVAGED, not
    fatal: reading stops at the last complete frame and the caller is told via
    `report`, so a whole run isn't lost to a missing final frame. Streaming
    always stops at the first unreadable frame rather than skipping past it —
    the surviving frames are then a CONTIGUOUS prefix, which is what the
    camera-frame timebase and the sync anchors require. A gap would silently
    shift every later frame's timestamp.

    Blocks and their views are invalidated by the next iteration — callers must
    copy anything they keep.
    """
    done = 0
    for f in files:
        if limit is not None and done >= limit:
            return
        with tifffile.TiffFile(str(f)) as tf:
            mm = truncated_memmap(tf, str(f), report=report, at_frame=done)
            if mm is not None:
                take = len(mm) if limit is None else min(len(mm), limit - done)
                step = max(1, int(batch_mb * 2**20) // max(1, mm[0].nbytes))
                for a in range(0, take, step):
                    yield mm[a:a + step]
                done += take
                del mm
                continue
            block, j = None, 0
            truncated = False
            for page in tf.pages:
                if limit is not None and done + j >= limit:
                    break
                shape = tuple(page.shape)
                if (block is None or block.shape[1:] != shape
                        or block.dtype != page.dtype):
                    if block is not None and j:
                        yield block[:j]
                        done += j
                    fbytes = (int(np.prod(shape, dtype=np.int64))
                              * np.dtype(page.dtype).itemsize)
                    n = max(1, int(batch_mb * 2**20) // max(1, fbytes))
                    block, j = np.empty((n,) + shape, page.dtype), 0
                try:
                    page.asarray(out=block[j])
                except (TypeError, ValueError):
                    # ValueError here is ambiguous: either out= is unsupported
                    # for this page, or the pixel data is short. Retrying
                    # without out= separates them — a short file fails again.
                    try:
                        block[j] = page.asarray()
                    except Exception as e:
                        note_truncation(report, f, done + j, e)
                        truncated = True
                        break
                j += 1
                if j == block.shape[0]:
                    yield block
                    done += j
                    j = 0
            if block is not None and j:
                yield block[:j]
                done += j
            if truncated:
                return


def note_truncation(report, path, frame, why):
    """Record + announce that a movie ended early. Loud on purpose: a short
    file is usually an upload still in flight, and the fix is to finish the
    upload and re-extract with --force, not to accept the salvage."""
    msg = (f"  *** TRUNCATED: {Path(path).name} ends mid-frame at frame "
           f"{frame} ({type(why).__name__}: {why}). Keeping the {frame} "
           f"complete frames. If this file is still uploading, WAIT for it to "
           f"finish and re-extract with --force — these traces are short. ***")
    print(msg, flush=True)
    if report is not None:
        report.setdefault("truncated", []).append(
            {"file": Path(path).name, "at_frame": int(frame),
             "reason": f"{type(why).__name__}: {why}"})


def extract_run(run, save_mat=True, force=False, batch_mb=BATCH_MB, strict=False,
                trim_tail=TRIM_TAIL_FRAMES):
    files = run_tifs(run)
    bj = boxes_path(run)
    if not bj.exists():
        print(f"[{run.name}] no proc/boxes.json — run the draw phase first. Skipping.")
        return None
    meta = json.loads(bj.read_text())
    boxes = meta["boxes"]
    categories = meta.get("categories", CANONICAL_BOXES)
    validate_boxes(boxes, categories, run.name)

    pd = proc_dir(run)
    pd.mkdir(exist_ok=True)
    stem = re.sub(r"\s+", "_", run.name)
    out = pd / f"{stem}_boxtraces.npz"
    if out.exists() and not force:
        print(f"[{run.name}] {out.name} exists — skipping (--force to redo).")
        return out

    names = [b["name"] for b in boxes]
    coords = np.array([[b["x0"], b["y0"], b["x1"], b["y1"]] for b in boxes], dtype=int)
    n_boxes = len(coords)

    # Batched streaming (OOM fix): per-box crops are cut from each block and
    # reduced per frame. Reduction order matches the old per-frame loop exactly
    # (row-wise pairwise mean over the contiguous crop), so values are
    # bit-identical to the validated implementation.
    # Tail-trim policy: stop BEFORE the unreliable shutdown frames rather than
    # read them and discard. Counting first costs one metadata pass and means a
    # cut-off final frame is usually never touched at all.
    trim = max(0, int(trim_tail))
    n_source, limit = None, None
    if trim:
        n_source = count_frames(files)
        if n_source - trim < 1:
            print(f"[{run.name}] WARNING: only {n_source} frames — NOT trimming "
                  f"the last {trim} (nothing would be left). Extracting all.")
            trim = 0
        else:
            limit = n_source - trim
            print(f"[{run.name}] trimming last {trim} frames (policy): "
                  f"extracting {limit} of {n_source}")

    tr_blocks, me_blocks = [], []
    prev = None                  # per-box float32 crop of the previous frame
    done, next_mark = 0, 2000
    report = {}
    for block in stream_frame_blocks(files, batch_mb=batch_mb, report=report,
                                     limit=limit):
        n = len(block)
        t_blk = np.empty((n_boxes, n), np.float32)
        m_blk = np.empty((n_boxes, n), np.float32)
        new_prev = []
        for b, (x0, y0, x1, y1) in enumerate(coords):
            cb = block[:, y0:y1, x0:x1]
            if cb.ndim == 4:                    # RGB(A) safety — ThorCam is mono
                cb = cb[..., :3].mean(axis=-1)
            cb = np.ascontiguousarray(cb, dtype=np.float32)
            t_blk[b] = cb.reshape(n, -1).mean(axis=1)
            if n > 1:
                m_blk[b, 1:] = np.abs(cb[1:] - cb[:-1]).reshape(n - 1, -1).mean(axis=1)
            m_blk[b, 0] = 0.0 if prev is None else float(np.abs(cb[0] - prev[b]).mean())
            new_prev.append(cb[-1].copy())
        prev = new_prev
        tr_blocks.append(t_blk)
        me_blocks.append(m_blk)
        done += n
        if done >= next_mark:
            print(f"[{run.name}] {done} frames...", flush=True)
            next_mark = done - done % 2000 + 2000

    traces = (np.concatenate(tr_blocks, axis=1) if tr_blocks
              else np.zeros((n_boxes, 0), np.float32))   # n_boxes x n_frames
    motion = (np.concatenate(me_blocks, axis=1) if me_blocks
              else np.zeros((n_boxes, 0), np.float32))

    trunc = report.get("truncated", [])
    if trunc and strict:
        print(f"[{run.name}] REFUSED (--strict-frames): movie is short — "
              f"no traces written. Finish the upload / check the recording, "
              f"then re-run.")
        return None

    # atomic write: a job killed mid-save must never leave a plausible-looking npz
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, traces=traces, motion_energy=motion,
                            box_names=np.array(names), box_coords=coords,
                            files=np.array([f.name for f in files]), run=str(run),
                            created=datetime.now().isoformat(timespec="seconds"),
                            truncated=bool(trunc),
                            truncated_info=json.dumps(trunc),
                            trim_tail=int(trim),
                            n_frames_source=int(n_source if n_source is not None else -1))
    os.replace(tmp, out)
    print(f"[{run.name}] wrote {out}  ({traces.shape[0]} boxes x {traces.shape[1]} frames)")
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak / (2**20 if sys.platform == "darwin" else 2**10)
        print(f"[{run.name}] peak RSS {peak_mb:.0f} MB (process high-water; "
              f"batch budget {batch_mb:g} MB)")
    except Exception:
        pass

    if save_mat:
        try:
            from scipy.io import savemat
            mat = out.with_suffix(".mat")
            mtmp = mat.with_name(mat.name + ".tmp")
            with open(mtmp, "wb") as fh:
                savemat(fh, {"traces": traces, "motion_energy": motion,
                             "box_names": names, "box_coords": coords,
                             "run": str(run), "truncated": bool(trunc),
                             "truncated_info": json.dumps(trunc),
                             "trim_tail": int(trim),
                             "n_frames_source": int(n_source if n_source
                                                    is not None else -1)})
            os.replace(mtmp, mat)
            print(f"[{run.name}] wrote {mat}")
        except ImportError:
            print("  (scipy not available — skipped .mat export)")
    if trunc:
        print(f"[{run.name}] NOTE: traces are marked truncated "
              f"(npz key 'truncated') — re-extract with --force once the "
              f"movie is complete.")
    return out


def cmd_extract(args):
    if not (trace_writes_allowed() or args.allow_local):
        sys.exit(CLUSTER_ONLY_MSG)
    if args.runs_file:
        # frozen run list from `list --save`: array tasks index a fixed file, so a
        # tree that is still uploading can never shift indices between jobs
        runs = [Path(l) for l in Path(args.runs_file).read_text().splitlines()
                if l.strip()]
    else:
        if not args.roots:
            sys.exit("Give ROOTS or --runs-file.")
        runs = find_runs(args.roots)
    if not runs:
        sys.exit("No runs found.")
    if args.run_index is not None:
        if args.run_index >= len(runs):
            sys.exit(f"--run-index {args.run_index} out of range (found {len(runs)} runs)")
        runs = [runs[args.run_index]]
    for run in runs:
        extract_run(run, save_mat=not args.no_mat, force=args.force,
                    batch_mb=args.batch_mb, strict=args.strict_frames,
                    trim_tail=args.trim_tail)


# ------------------------------------------------------------ list / collect

def cmd_list(args):
    runs = find_runs(args.roots)
    print(f"{'idx':>3}  {'boxes':5} {'traces':6} {'sync':4}  run")
    for i, run in enumerate(runs):
        pd = proc_dir(run)
        b = "yes" if boxes_path(run).exists() else "-"
        t = "yes" if list(pd.glob("*_boxtraces.npz")) else "-"
        s = "yes" if list(pd.glob("*_sync.json")) else "-"
        print(f"{i:3d}  {b:5} {t:6} {s:4}  {run}  ({len(run_tifs(run))} tifs)")
    if args.save:
        Path(args.save).write_text("\n".join(str(r) for r in runs) + "\n")
        print(f"\nwrote {args.save} ({len(runs)} runs) — use with extract --runs-file")


def verify_tif(path):
    """Is this tif readable end-to-end? Returns (ok, n_frames, message).

    Reads the FIRST and LAST frame only — enough to catch a file whose tail is
    missing (upload still in flight, aborted recording) without paying to read
    the middle. This is the cheap pre-flight for a tree that may still be
    uploading; extraction itself salvages and flags, but finding short files
    BEFORE a batch is much cheaper than after.
    """
    try:
        with tifffile.TiffFile(str(path)) as tf:
            mm = truncated_memmap(tf, str(path))
            if mm is not None:
                declared = None
                try:
                    ser = tf.series[0]
                    declared = int(np.prod(
                        ser.shape[:len(ser.shape) - len(tf.pages[0].shape)],
                        dtype=np.int64))
                except Exception:
                    pass
                if tf.is_imagej and tf.imagej_metadata:
                    declared = max(declared or 0,
                                   int(tf.imagej_metadata.get("images", 0)))
                n = len(mm)
                np.asarray(mm[0]), np.asarray(mm[-1])
                del mm
                if declared and n < declared:
                    return False, n, f"SHORT: {n}/{declared} frames present"
                return True, n, "ok"
            n = len(tf.pages)
            if n == 0:
                return False, 0, "no readable pages"
            tf.pages[0].asarray()
            tf.pages[n - 1].asarray()
            return True, n, "ok"
    except Exception as e:
        return False, -1, f"{type(e).__name__}: {e}"


def cmd_verify(args):
    """Check every tif under ROOTS for a missing tail. Exit 2 if any is short."""
    runs = find_runs(args.roots)
    bad, total_files = [], 0
    for run in runs:
        files = run_tifs(run)
        total_files += len(files)
        frames, run_bad = 0, []
        for f in files:
            ok, n, msg = verify_tif(f)
            frames += max(n, 0)
            if not ok:
                run_bad.append((f, msg))
        flag = "OK  " if not run_bad else "BAD "
        print(f"{flag} {run}  ({len(files)} tifs, {frames} frames)")
        for f, msg in run_bad:
            print(f"       {f.name}: {msg}")
        bad.extend((run, f, msg) for f, msg in run_bad)
    print(f"\nChecked {total_files} tifs in {len(runs)} runs — "
          f"{len(bad)} unreadable/short.")
    if bad:
        print("Short files are usually an upload still in flight. Wait for it "
              "to finish, re-check, then extract (add --force to redo any run "
              "already extracted from a short file).")
        sys.exit(2)


def cmd_collect(args):
    """Copy proc/ outputs (never raw movies) into one flat CLUSTER folder
    (staging for slides/sharing — outputs never leave the cluster)."""
    runs = find_runs(args.roots)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for run in runs:
        pd = proc_dir(run)
        if not pd.is_dir():
            continue
        stem = re.sub(r"\s+", "_", run.name)
        prefix = re.sub(r"\s+", "_", f"{run.parent.name}_{run.name}")
        for f in sorted(pd.iterdir()):
            if f.is_file():
                name = f.name
                if name.startswith(stem + "_"):
                    name = name[len(stem) + 1:]
                shutil.copy2(f, dest / f"{prefix}_{name}")
                n += 1
    print(f"Copied {n} proc files from {len(runs)} runs -> {dest}")


# ----------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("draw", help="interactive box GUI; writes <run>/proc/boxes.json")
    p.add_argument("roots", nargs="+")
    p.add_argument("--boxes-from", help="apply this boxes.json to all runs (no GUI)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--categories", help="comma-separated override of canonical categories")
    p.set_defaults(fn=cmd_draw)

    p = sub.add_parser("extract", help="stream movies; writes <run>/proc/<run>_boxtraces.npz")
    p.add_argument("roots", nargs="*")
    p.add_argument("--runs-file", help="frozen run list from `list --save` (preferred for SLURM arrays)")
    p.add_argument("--no-mat", action="store_true", help="skip the .mat copy")
    p.add_argument("--force", action="store_true", help="re-extract even if output exists")
    p.add_argument("--run-index", type=int, default=None,
                   help="process only the Nth run (0-based; for SLURM arrays)")
    p.add_argument("--batch-mb", type=float, default=BATCH_MB,
                   help="frame-block read budget in MB — bounds extraction "
                        f"memory (default {BATCH_MB})")
    p.add_argument("--trim-tail", type=int, default=TRIM_TAIL_FRAMES,
                   help=f"drop the last N camera frames, which the camera "
                        f"writes while being stopped (default {TRIM_TAIL_FRAMES}; "
                        f"0 keeps every frame)")
    p.add_argument("--strict-frames", action="store_true",
                   help="refuse to write traces for a run whose movie is short "
                        "(default: salvage the complete frames and flag them)")
    p.add_argument("--allow-local", action="store_true",
                   help="override the cluster-only trace policy (synthetic tests only)")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("verify", help="check every tif for a missing tail "
                                      "(upload still in flight / aborted recording)")
    p.add_argument("roots", nargs="+")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("list", help="show runs and processing status")
    p.add_argument("roots", nargs="+")
    p.add_argument("--save", help="also write the run list to this file (for extract --runs-file)")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("collect", help="copy all proc outputs to one flat folder")
    p.add_argument("roots", nargs="+")
    p.add_argument("--dest", required=True)
    p.set_defaults(fn=cmd_collect)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

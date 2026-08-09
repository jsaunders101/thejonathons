#!/usr/bin/env python3
"""
box_extract.py — box-ROI drawing (lightweight GUI) + per-box trace extraction
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
        <run>_sync.json        from sync_detect.py
        <run>_syncqc.png       from sync_detect.py

A "run" = any folder that directly contains one or more .tif/.tiff files; multiple
tifs in a run are one movie, concatenated in natural-sort order.

Phases:
  draw     Interactive GUI, one first-frame per run (cheap). Writes <run>/proc/boxes.json.
           'reuse previous' button copies the last run's boxes. --boxes-from applies one
           boxes.json to every run with no GUI.
  extract  Headless, streams every frame, writes <run>/proc/<run>_boxtraces.npz (+.mat).
           Parallelize with --run-index in a SLURM array.
  list     Show every run and its processing status (boxes / traces / sync).
  collect  Copy all proc/ outputs (NOT raw movies) into one flat folder ON THE
           CLUSTER (staging for slides/sharing — nothing leaves the cluster),
           prefixing filenames with parent folders to stay unique.

Examples:
  python box_extract.py draw    /data/behavior/day1
  python box_extract.py draw    /data/behavior --boxes-from /data/behavior/day1/run01/proc/boxes.json
  python box_extract.py extract /data/behavior
  python box_extract.py extract /data/behavior --run-index $SLURM_ARRAY_TASK_ID
  python box_extract.py list    /data/behavior
  python box_extract.py collect /data/behavior --dest /data/behavior/_collected
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
    sys.exit("box_extract.py requires tifffile (pip install tifffile)")

CANONICAL_BOXES = ["laser_trigger", "whisker_pad", "paw", "paw_at_nose", "eye",
                   "nose", "wheel"]

PROC = "proc"


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

def stream_frames(files):
    """Yield frames in NATIVE dtype, reusing one read buffer per file.

    Callers must copy anything they keep across iterations — the buffer is
    overwritten by the next frame.
    """
    for f in files:
        with tifffile.TiffFile(str(f)) as tf:
            buf = None
            for page in tf.pages:
                if buf is None or buf.shape != page.shape or buf.dtype != page.dtype:
                    buf = np.empty(page.shape, page.dtype)
                try:
                    yield page.asarray(out=buf)
                except (TypeError, ValueError):
                    yield page.asarray()


def crop_f32(frame, coords):
    """Cut boxes out of a native-dtype frame; return float32 COPIES (buffer-safe)."""
    out = []
    for x0, y0, x1, y1 in coords:
        c = frame[y0:y1, x0:x1]
        if c.ndim == 3:                         # RGB(A) safety — ThorCam is mono
            c = c[..., :3].mean(axis=-1)
        out.append(np.ascontiguousarray(c, dtype=np.float32))
    return out


def extract_run(run, save_mat=True, force=False):
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

    tr, me = [], []
    prev = None
    for i, frame in enumerate(stream_frames(files)):
        crops = crop_f32(frame, coords)
        tr.append([float(c.mean()) for c in crops])
        if prev is None:
            me.append([0.0] * len(crops))
        else:
            me.append([float(np.abs(c - p).mean()) for c, p in zip(crops, prev)])
        prev = crops
        if (i + 1) % 2000 == 0:
            print(f"[{run.name}] {i + 1} frames...", flush=True)

    traces = np.asarray(tr, dtype=np.float32).T          # n_boxes x n_frames
    motion = np.asarray(me, dtype=np.float32).T

    # atomic write: a job killed mid-save must never leave a plausible-looking npz
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, traces=traces, motion_energy=motion,
                            box_names=np.array(names), box_coords=coords,
                            files=np.array([f.name for f in files]), run=str(run),
                            created=datetime.now().isoformat(timespec="seconds"))
    os.replace(tmp, out)
    print(f"[{run.name}] wrote {out}  ({traces.shape[0]} boxes x {traces.shape[1]} frames)")

    if save_mat:
        try:
            from scipy.io import savemat
            mat = out.with_suffix(".mat")
            mtmp = mat.with_name(mat.name + ".tmp")
            with open(mtmp, "wb") as fh:
                savemat(fh, {"traces": traces, "motion_energy": motion,
                             "box_names": names, "box_coords": coords,
                             "run": str(run)})
            os.replace(mtmp, mat)
            print(f"[{run.name}] wrote {mat}")
        except ImportError:
            print("  (scipy not available — skipped .mat export)")
    return out


def cmd_extract(args):
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
        extract_run(run, save_mat=not args.no_mat, force=args.force)


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
    p.set_defaults(fn=cmd_extract)

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

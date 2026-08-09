#!/usr/bin/env python3
"""
run_behavior_cluster.py — one-command behavior pipeline: pick a master folder in a dialog,
then box-draw -> extract -> review every run inside it.

Wraps box_extract_cluster.py (draw + extract) and trace_viewer_cluster.py (validation) into a single
interactive flow. GUI availability is PROBED by actually opening a window, never
guessed from env vars — so on a cluster (VS Code Remote-SSH, ssh -Y, OnDemand) it
always TRIES, and degrades cleanly when a window can't open:

  1. SELECT   Tk folder dialog -> matplotlib browser -> typed path at the terminal
              (pass the folder as an argument to skip all of it)
  2. STATUS   every run under the folder and what it already has (boxes/traces/sync)
  3. DRAW     box GUI for every run without proc/boxes.json ('reuse previous' works
              across runs; --boxes-from applies one boxes.json everywhere, no GUI).
              Needs a working display — refuses with X11 remediation steps otherwise.
  4. EXTRACT + REVIEW, per run: streaming trace extraction (skips already-extracted;
              --force), then trace_viewer_cluster opens IMMEDIATELY on that run's traces —
              close the window to move to the next run, q to stop opening viewers.
              If a viewer can't open (headless, dead X connection), each run gets a
              static <run>/proc/<run>_tracesqc.png instead (first frame + boxes +
              traces) — flip through them in VS Code. --qc-png always writes these;
              --no-view skips review entirely. A failing run is isolated and
              reported; the remaining runs still process (exit 2 at the end).

Headless batch: give the folder + --boxes-from + --no-view (or --qc-png).

Examples:
  python run_behavior_cluster.py                                   # full interactive flow
  python run_behavior_cluster.py /data/behavior/day1               # skip the dialog
  python run_behavior_cluster.py /data/behavior \
      --boxes-from /data/behavior/day1/run01/proc/boxes.json --qc-png
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from box_extract_cluster import (  # noqa: E402
    BATCH_MB, CANONICAL_BOXES, CLUSTER_ONLY_MSG, BoxGUI, boxes_path, extract_run,
    find_runs, natural_key, proc_dir, read_first_frame, run_tifs, save_boxes,
    trace_writes_allowed, validate_boxes)

X11_HELP = """\
To get GUIs working on the cluster:
  - EASIEST: an Open OnDemand interactive Desktop session on Elzar — GUIs run
    natively in the desktop, no forwarding involved; run this script in its terminal
  - plain terminal:  ssh -YC user@host   (XQuartz must be installed AND running
    locally on macOS)
  - VS Code Remote-SSH: add to local ~/.ssh/config for this host:
        ForwardX11 yes
        ForwardX11Trusted yes
    start XQuartz, then fully reconnect the VS Code window; check with: echo $DISPLAY
  - compute nodes only get a display via  srun --x11  (if the site enables it)
  - no display at all: run headless with --boxes-from <boxes.json> (draw the boxes
    in a desktop session first); review via the *_tracesqc.png files (--qc-png)."""

_GUI = {"ok": None, "why": ""}

# matplotlib's non-interactive backends, by exact name. Substring tests are a
# trap here: "agg" is a substring of "tkagg"/"qtagg", which made working TkAgg
# sessions classify as headless (bug found on the cluster, fixed Aug 9) — and
# "tkagg not in backend" would misclassify MacOSX/QtAgg the same way.
NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def backend_is_headless(name):
    """True if this matplotlib backend cannot open windows (exact-name test)."""
    return name.lower() in NON_INTERACTIVE_BACKENDS


def probe_gui():
    """Can we actually open a window? TRY once (create+close a figure) and cache.
    matplotlib picks Agg when DISPLAY is unset, and a stale DISPLAY only fails at
    window creation — so probing is the only reliable test on a cluster."""
    if _GUI["ok"] is not None:
        return _GUI["ok"]
    ok, why = False, ""
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        why = "MPLBACKEND=Agg forces headless"
    else:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            if backend_is_headless(matplotlib.get_backend()):
                # pyplot fell back to a non-interactive backend (usually DISPLAY
                # unset) — try Tk, the backend the course envs ship. (Qt is not
                # probed: a headless Qt can abort the process instead of raising.)
                try:
                    matplotlib.use("TkAgg", force=True)
                except Exception:
                    pass
            backend = matplotlib.get_backend()
            if backend_is_headless(backend):
                why = f"non-interactive matplotlib backend {backend!r} (DISPLAY unset?)"
            else:
                fig = plt.figure(figsize=(1, 1))    # raises if the X link is dead
                plt.close(fig)
                ok = True
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
    _GUI.update(ok=ok, why=why)
    if not ok:
        print(f"[gui] cannot open windows ({why})")
    return ok


# ------------------------------------------------------------- folder picker

class FolderPicker:
    """Minimal matplotlib folder browser: click a directory name to enter it,
    'up [..]' to go up, SELECT to accept the current folder. Fallback for cluster
    nodes without tkinter — depends only on the same matplotlib-over-X11 the draw
    GUI already needs."""

    PER_PAGE = 26

    def __init__(self, start):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._plt = plt
        self.cwd = Path(start).expanduser().resolve()
        self.page = 0
        self.result = None
        self._artists = {}          # text artist -> Path (None for placeholders)

        self.fig = plt.figure(figsize=(8.5, 9.5))
        try:
            self.fig.canvas.manager.set_window_title(
                "run_behavior_cluster — select master folder")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.02, 0.13, 0.96, 0.83])
        self.ax.axis("off")

        self.buttons = []
        x = 0.02
        for label, cb, col, w in (
                ("SELECT this folder", self._on_select, "lightgreen", 0.28),
                ("up [..]", self._on_up, "0.85", 0.14),
                ("prev page", lambda e: self._flip(-1), "0.85", 0.14),
                ("next page", lambda e: self._flip(+1), "0.85", 0.14),
                ("cancel", self._on_cancel, "mistyrose", 0.14)):
            bax = self.fig.add_axes([x, 0.03, w, 0.055])
            x += w + 0.012
            b = Button(bax, label, color=col, hovercolor="0.95")
            b.on_clicked(cb)
            self.buttons.append(b)

        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self._render()

    def _subdirs(self):
        try:
            return sorted((d for d in self.cwd.iterdir()
                           if d.is_dir() and not d.name.startswith(".")),
                          key=natural_key)
        except (PermissionError, OSError) as e:
            print(f"  (cannot list {self.cwd}: {e})")
            return []

    def _render(self):
        for t in self._artists:
            t.remove()
        self._artists = {}
        subs = self._subdirs()
        npages = max(1, -(-len(subs) // self.PER_PAGE))
        self.page = min(max(self.page, 0), npages - 1)
        shown = subs[self.page * self.PER_PAGE:(self.page + 1) * self.PER_PAGE]

        self.ax.set_title(f"{self.cwd}\n(page {self.page + 1}/{npages} — "
                          f"{len(subs)} subfolders; click one to enter)",
                          loc="left", fontsize=10, family="monospace")
        y = 0.97
        dy = 1.0 / (self.PER_PAGE + 2)
        if not shown:
            t = self.ax.text(0.01, y, "(no subdirectories)",
                             transform=self.ax.transAxes, fontsize=10,
                             family="monospace", color="0.5")
            self._artists[t] = None
        for d in shown:
            t = self.ax.text(0.01, y, d.name + "/", transform=self.ax.transAxes,
                             fontsize=10, family="monospace", color="tab:blue",
                             picker=True)
            self._artists[t] = d
            y -= dy
        self.fig.canvas.draw_idle()

    def _on_pick(self, event):
        d = self._artists.get(event.artist)
        if d is not None:
            self.cwd = d
            self.page = 0
            self._render()

    def _on_up(self, event=None):
        if self.cwd.parent != self.cwd:
            self.cwd = self.cwd.parent
            self.page = 0
            self._render()

    def _flip(self, step):
        self.page += step
        self._render()

    def _on_select(self, event=None):
        self.result = self.cwd
        self._plt.close(self.fig)

    def _on_cancel(self, event=None):
        self._plt.close(self.fig)

    def run(self):
        self._plt.show(block=True)
        return self.result


def pick_folder(start):
    """Try HARD, in order: Tk dialog -> matplotlib browser -> typed path.
    Returns None only when the user cancels/aborts."""
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        print("[gui] MPLBACKEND=Agg — skipping dialogs.")
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            try:
                root = tk.Tk()
                root.withdraw()
                root.update()
                sel = filedialog.askdirectory(initialdir=str(start), mustexist=True,
                                              title="Select master behavior folder")
                root.destroy()
                return Path(sel) if sel else None      # '' = user cancelled
            except tk.TclError as e:
                print(f"[gui] Tk dialog failed ({e})")
        except ImportError:
            print("[gui] tkinter not available")

        if probe_gui():
            try:
                return FolderPicker(start).run()
            except Exception as e:
                print(f"[gui] matplotlib browser failed ({e})")

    print("No dialog possible — type the path instead.")
    try:
        p = input("Master behavior folder (blank to abort): ").strip()
    except EOFError:
        return None
    return Path(p).expanduser() if p else None


# ------------------------------------------------------------------- phases

def npz_is_truncated(run):
    """Did this run's traces come from a movie that ended mid-frame?"""
    npzs = sorted(proc_dir(run).glob("*_boxtraces.npz"))
    if not npzs:
        return False
    try:
        with np.load(npzs[0], allow_pickle=True) as d:
            return bool(d["truncated"]) if "truncated" in d else False
    except Exception:
        return False


def print_status(runs, header):
    print(f"\n{header}")
    print(f"{'idx':>3}  {'boxes':5} {'traces':6} {'sync':4}  run")
    for i, run in enumerate(runs):
        pd = proc_dir(run)
        b = "yes" if boxes_path(run).exists() else "-"
        t = "yes" if list(pd.glob("*_boxtraces.npz")) else "-"
        s = "yes" if list(pd.glob("*_sync.json")) else "-"
        print(f"{i:3d}  {b:5} {t:6} {s:4}  {run}  ({len(run_tifs(run))} tifs)")


def phase_draw(runs, categories, boxes_from, redraw):
    """box_extract_cluster's draw semantics: skip existing (unless redraw), template via
    --boxes-from, previous run's boxes offered as 'reuse previous'. A run whose
    first frame can't be read is reported and skipped, not fatal."""
    template = None
    if boxes_from:
        template = json.loads(Path(boxes_from).read_text())["boxes"]
        validate_boxes(template, categories, boxes_from)

    prev = template
    for run in runs:
        bj = boxes_path(run)
        if bj.exists() and not redraw:
            prev = json.loads(bj.read_text())["boxes"]
            continue
        if bj.exists():
            # redrawing: seed 'reuse previous' with this run's OWN boxes (one click
            # restores them; then draw only the new/changed categories)
            prev = json.loads(bj.read_text())["boxes"]
        try:
            files = run_tifs(run)
            frame = read_first_frame(files[0])
        except Exception as e:
            print(f"[{run.name}] DRAW ERROR: {e} — skipping this run")
            continue
        if template is not None:
            save_boxes(run, files, [dict(b) for b in template], frame.shape,
                       categories)
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


def view_run(run, trace_kind, ds, stop):
    """Open trace_viewer_cluster on one run, blocking until its window closes.
    Pressing q sets stop['q'] so the caller stops opening further viewers."""
    import matplotlib.pyplot as plt
    from trace_viewer_cluster import Viewer

    print(f"[{run.name}] review — close window to continue, q to stop reviewing")
    viewer = Viewer(run, trace_kind=trace_kind, ds=ds)

    def on_key(ev, fig=viewer.fig):
        if ev.key == "q":
            stop["q"] = True
            plt.close(fig)
    viewer.fig.canvas.mpl_connect("key_press_event", on_key)
    viewer.run()


def save_trace_png(run, trace_kind="intensity", ds=2):
    """Static stand-in for trace_viewer_cluster: first frame + box overlays beside per-box
    traces -> <run>/proc/<stem>_tracesqc.png. Renders on an explicit Agg canvas, so
    it works with or without a display."""
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    npzs = sorted(proc_dir(run).glob("*_boxtraces.npz"))
    if not npzs:
        return None
    d = np.load(npzs[0], allow_pickle=True)
    names = [str(n) for n in d["box_names"]]
    arr = d["traces"] if trace_kind == "intensity" else d["motion_energy"]
    coords = d["box_coords"]
    frame = read_first_frame(run_tifs(run)[0])[::ds, ::ds]

    fig = Figure(figsize=(15, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.015, 0.05, 0.55, 0.90])
    vmin, vmax = np.percentile(frame, [1, 99.5])
    ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    kind = "mean intensity" if trace_kind == "intensity" else "motion energy"
    ax.set_title(f"{run.name} — first frame + boxes ({kind} traces)", fontsize=10)
    for name, (x0, y0, x1, y1) in zip(names, coords):
        ax.add_patch(mpatches.Rectangle(
            (x0 / ds, y0 / ds), (x1 - x0) / ds, (y1 - y0) / ds,
            fill=False, edgecolor="lime", linewidth=1.2))
        ax.text(x0 / ds, y0 / ds - 3, name, color="lime", fontsize=8,
                fontweight="bold")

    nb = len(names)
    top, bottom = 0.95, 0.05
    h = (top - bottom) / nb
    for i, name in enumerate(names):
        axt = fig.add_axes([0.615, top - (i + 1) * h + 0.008, 0.37, h - 0.014])
        axt.plot(arr[i], lw=0.5, color="k")
        axt.set_xlim(0, arr.shape[1] - 1)
        axt.tick_params(labelsize=7)
        axt.text(0.005, 0.8, name, transform=axt.transAxes, fontsize=8,
                 fontweight="bold", color="tab:blue")
        if i < nb - 1:
            axt.set_xticklabels([])

    out = proc_dir(run) / (npzs[0].stem.replace("_boxtraces", "") + "_tracesqc.png")
    fig.savefig(out, dpi=120)
    print(f"[{run.name}] wrote {out}")
    return out


# ---------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?",
                    help="master behavior folder (omit to pick in a dialog)")
    ap.add_argument("--start", default=".", help="dialog start directory")
    ap.add_argument("--categories",
                    help="comma-separated override of canonical categories")
    ap.add_argument("--boxes-from",
                    help="apply this boxes.json to every run (no draw GUI)")
    ap.add_argument("--redraw", action="store_true",
                    help="redo boxes even where proc/boxes.json exists")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even where traces exist")
    ap.add_argument("--no-mat", action="store_true", help="skip .mat copies")
    ap.add_argument("--no-view", action="store_true",
                    help="skip review entirely (no viewer, no fallback PNGs)")
    ap.add_argument("--qc-png", action="store_true",
                    help="always write per-run *_tracesqc.png (works headless)")
    ap.add_argument("--trace", choices=["intensity", "motion"], default="intensity",
                    help="trace kind shown in review")
    ap.add_argument("--ds", type=int, default=2, help="review display downsample")
    ap.add_argument("--batch-mb", type=float, default=None,
                    help="extraction frame-block budget in MB (bounds memory; "
                         "default: box_extract_cluster's BATCH_MB)")
    ap.add_argument("--strict-frames", action="store_true",
                    help="refuse runs whose movie is short instead of "
                         "salvaging the complete frames")
    ap.add_argument("--allow-local", action="store_true",
                    help="override the cluster-only trace policy (synthetic tests only)")
    args = ap.parse_args()

    if not (trace_writes_allowed() or args.allow_local):
        sys.exit(CLUSTER_ONLY_MSG)

    categories = ([c.strip() for c in args.categories.split(",")]
                  if args.categories else CANONICAL_BOXES)

    # 1. master folder
    if args.folder:
        folder = Path(args.folder).expanduser()
    else:
        folder = pick_folder(args.start)
        if folder is None:
            sys.exit("No folder selected.")
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    # 2. discovery + status
    runs = find_runs([folder])
    if not runs:
        sys.exit(f"No runs (folders directly containing .tif files) under {folder}")
    print_status(runs, f"Master folder: {folder} — {len(runs)} runs")

    # 3. draw
    need = [r for r in runs if args.redraw or not boxes_path(r).exists()]
    if need and not args.boxes_from:
        if not probe_gui():
            names = ", ".join(r.name for r in need)
            sys.exit(f"{len(need)} runs need boxes ({names}) but no window could "
                     f"be opened ({_GUI['why']}), and drawing requires one.\n"
                     f"{X11_HELP}")
    if need:
        print(f"\nDrawing boxes for {len(need)} runs...")
        try:
            phase_draw(runs, categories, args.boxes_from, args.redraw)
        except Exception as e:
            # boxes.json is saved per run as we go — progress up to here persists
            sys.exit(f"draw GUI died mid-flow ({type(e).__name__}: {e}).\n{X11_HELP}")
    else:
        print("\nAll runs already have boxes.")

    # 4. extract + review, per run; viewer -> PNG fallback; failures isolated
    viewing = not args.no_view and probe_gui()
    png_mode = args.qc_png or (not args.no_view and not viewing)
    if png_mode and not viewing and not args.no_view:
        print("(review falls back to *_tracesqc.png files — open them in VS Code)")
    stop = {"q": False}
    print(f"\nExtracting {len(runs)} runs...")
    batch_mb = args.batch_mb if args.batch_mb else BATCH_MB
    for run in runs:
        try:
            extract_run(run, save_mat=not args.no_mat, force=args.force,
                        batch_mb=batch_mb, strict=args.strict_frames)
        except Exception as e:
            print(f"[{run.name}] EXTRACT ERROR: {type(e).__name__}: {e} — "
                  f"continuing with remaining runs")
            continue
        if not list(proc_dir(run).glob("*_boxtraces.npz")):
            continue
        if viewing and not stop["q"]:
            try:
                view_run(run, args.trace, args.ds, stop)
            except Exception as e:
                print(f"[gui] viewer failed ({type(e).__name__}: {e}) — "
                      f"switching to *_tracesqc.png for the remaining runs")
                viewing = False
                png_mode = True
        if png_mode:
            try:
                save_trace_png(run, args.trace, args.ds)
            except Exception as e:
                print(f"[{run.name}] QC PNG failed: {e}")

    done = [r for r in runs if list(proc_dir(r).glob("*_boxtraces.npz"))]
    missing = [r for r in runs if r not in set(done)]
    truncated = [r for r in done if npz_is_truncated(r)]
    print_status(runs, "Final status")

    if missing:
        print(f"\n*** {len(missing)} runs have NO traces: "
              f"{', '.join(r.name for r in missing)} ***")
    if truncated:
        print(f"\n*** {len(truncated)} runs have TRUNCATED traces (movie ended "
              f"mid-frame): {', '.join(r.name for r in truncated)} ***\n"
              f"    Usually an upload still in flight. Check with:\n"
              f"      python box_extract_cluster.py verify <folder>\n"
              f"    then re-extract those runs with --force once complete.")
    if missing or truncated:
        sys.exit(2)
    print("\nDone. Next: sync_detect_cluster.py per run (needs the t-series XML) — "
          "see ../03_synchronization.md.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
trace_viewer_cluster.py — play a behavior movie side-by-side with its extracted box traces.

Left: the movie with the box overlays drawn on it. Right: one panel per box with
the extracted trace and a red cursor that sweeps as the movie plays.

Controls: play / pause buttons (spacebar also toggles), a frame-rate slider, and
a frame slider for scrubbing. Playback loops at the end.

Frames are read from the tifs on demand — nothing is preloaded, so movies of any
size open instantly and RAM stays flat. Display is spatially downsampled by --ds
(default 2) for speed; the traces are untouched.

Usage:
  python trace_viewer_cluster.py "/path/to/run"                 # run folder (uses proc/)
  python trace_viewer_cluster.py "/path/to/run" --trace motion  # plot motion energy
  python trace_viewer_cluster.py "/path/to/run" --ds 4          # faster display
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from box_extract_cluster import (  # noqa: E402
    run_tifs, to_gray, proc_dir, boxes_path, truncated_memmap)

try:
    import tifffile
except ImportError:
    sys.exit("trace_viewer_cluster.py requires tifffile")


class FrameSource:
    """Random access into a run's concatenated tif stacks, one frame at a time.

    Single-IFD contiguous giants (ImageJ 'truncated' >4 GB files) are indexed
    via np.memmap so every frame is reachable, not just the first page."""

    def __init__(self, files):
        self.tfs = [tifffile.TiffFile(str(f)) for f in files]
        self.mms = []
        self.index = []
        for i, (f, tf) in enumerate(zip(files, self.tfs)):
            mm = truncated_memmap(tf, str(f))
            self.mms.append(mm)
            n = len(mm) if mm is not None else len(tf.pages)
            self.index.extend((i, j) for j in range(n))

    def __len__(self):
        return len(self.index)

    def get(self, k):
        i, j = self.index[k]
        mm = self.mms[i]
        if mm is not None:
            return to_gray(mm[j])
        return to_gray(self.tfs[i].pages[j].asarray())


class Viewer:
    def __init__(self, run, trace_kind="intensity", ds=2, fps=15):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button, Slider

        self._plt = plt
        run = Path(run)
        self.ds = max(1, int(ds))

        npzs = sorted(proc_dir(run).glob("*_boxtraces.npz"))
        if not npzs:
            sys.exit(f"No *_boxtraces.npz in {proc_dir(run)} — run box_extract_cluster.py extract first.")
        d = np.load(npzs[0], allow_pickle=True)
        self.names = [str(n) for n in d["box_names"]]
        arr = d["traces"] if trace_kind == "intensity" else d["motion_energy"]
        coords = d["box_coords"]

        self.src = FrameSource(run_tifs(run))
        self.n = min(len(self.src), arr.shape[1])
        gap = len(self.src) - arr.shape[1]
        trim = int(d["trim_tail"]) if "trim_tail" in d else 0
        if gap and gap != trim:
            # a mismatch the tail-trim policy doesn't account for
            print(f"WARNING: movie has {len(self.src)} frames but traces have "
                  f"{arr.shape[1]} (trim_tail={trim}) — showing first {self.n}.")
        elif trim:
            print(f"(last {trim} frames trimmed at extraction — policy; "
                  f"showing {self.n})")
        self.traces = arr[:, :self.n]

        self.k = 0
        self.playing = False
        self._syncing = False

        # ---- layout
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title(f"trace_viewer_cluster — {run.name}")
        self.ax_img = self.fig.add_axes([0.015, 0.16, 0.55, 0.80])
        frame = self.src.get(0)[::self.ds, ::self.ds]
        vmin, vmax = np.percentile(frame, [1, 99.5])
        self.im = self.ax_img.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax,
                                     interpolation="nearest")
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.title = self.ax_img.set_title(f"frame 0 / {self.n - 1}", fontsize=10)
        for name, (x0, y0, x1, y1) in zip(self.names, coords):
            self.ax_img.add_patch(mpatches.Rectangle(
                (x0 / self.ds, y0 / self.ds), (x1 - x0) / self.ds, (y1 - y0) / self.ds,
                fill=False, edgecolor="lime", linewidth=1.2))
            self.ax_img.text(x0 / self.ds, y0 / self.ds - 3, name, color="lime",
                             fontsize=8, fontweight="bold")

        # ---- trace panels
        nb = len(self.names)
        self.cursors = []
        top, bottom = 0.96, 0.16
        h = (top - bottom) / nb
        kind = "mean intensity" if trace_kind == "intensity" else "motion energy"
        axes = []
        for i, name in enumerate(self.names):
            ax = self.fig.add_axes([0.615, top - (i + 1) * h + 0.012, 0.37, h - 0.018])
            ax.plot(self.traces[i], lw=0.6, color="k")
            lo, hi = self.traces[i].min(), self.traces[i].max()
            pad = 0.05 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(0, self.n - 1)
            ax.tick_params(labelsize=7)
            ax.text(0.005, 0.82, name, transform=ax.transAxes, fontsize=8,
                    fontweight="bold", color="tab:blue")
            if i < nb - 1:
                ax.set_xticklabels([])
            self.cursors.append(ax.axvline(0, color="r", lw=1))
            axes.append(ax)
        axes[0].set_title(kind, fontsize=9)
        axes[-1].set_xlabel("camera frame", fontsize=9)

        # ---- controls
        bax = self.fig.add_axes([0.015, 0.045, 0.07, 0.055])
        self.btn_play = Button(bax, "play")
        self.btn_play.on_clicked(lambda e: self._set_playing(True))
        bax = self.fig.add_axes([0.095, 0.045, 0.07, 0.055])
        self.btn_pause = Button(bax, "pause")
        self.btn_pause.on_clicked(lambda e: self._set_playing(False))
        sax = self.fig.add_axes([0.24, 0.06, 0.15, 0.03])
        self.s_fps = Slider(sax, "fps", 1, 60, valinit=fps, valstep=1)
        self.s_fps.on_changed(self._on_fps)
        sax = self.fig.add_axes([0.46, 0.06, 0.42, 0.03])
        self.s_frame = Slider(sax, "frame", 0, self.n - 1, valinit=0, valstep=1)
        self.s_frame.on_changed(self._on_scrub)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._tick)
        self.timer.start()

    # ---- callbacks
    def _set_playing(self, val):
        self.playing = val

    def _on_key(self, event):
        if event.key == " ":
            self.playing = not self.playing

    def _on_fps(self, val):
        self.timer.interval = int(1000 / max(1, int(val)))

    def _on_scrub(self, val):
        if not self._syncing:
            self._show(int(val))

    def _tick(self):
        if self.playing:
            self._show((self.k + 1) % self.n)

    def _show(self, k):
        self.k = k
        self.im.set_data(self.src.get(k)[::self.ds, ::self.ds])
        self.title.set_text(f"frame {k} / {self.n - 1}")
        for c in self.cursors:
            c.set_xdata([k, k])
        self._syncing = True
        self.s_frame.set_val(k)
        self._syncing = False
        self.fig.canvas.draw_idle()

    def run(self):
        self._plt.show(block=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder (with proc/ inside)")
    ap.add_argument("--trace", choices=["intensity", "motion"], default="intensity")
    ap.add_argument("--ds", type=int, default=2, help="display downsample factor")
    ap.add_argument("--fps", type=int, default=15, help="initial playback rate")
    args = ap.parse_args()
    if not boxes_path(args.run).exists():
        sys.exit(f"No proc/boxes.json in {args.run}")
    Viewer(args.run, trace_kind=args.trace, ds=args.ds, fps=args.fps).run()


if __name__ == "__main__":
    main()

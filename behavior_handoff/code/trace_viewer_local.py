#!/usr/bin/env python3
"""
trace_viewer_local.py — play a behavior movie side-by-side with its extracted box traces.

Left: the movie with the box overlays drawn on it. Right: one panel per box with
the extracted trace and a red cursor that sweeps as the movie plays.

DEFAULT TRACE = MOTION ENERGY (mean |frame_t - frame_t-1| per box). That is the
behavioural readout the analysis is built on — whisking, paw motion, grooming —
whereas mean intensity mostly reports illumination and posture. Pass
`--trace intensity` for the mean-fluorescence view (still the right one for
checking the laser_trigger step or eyelid/pupil brightness).

Controls: play / pause buttons (spacebar also toggles), a frame-rate slider, and
a frame slider for scrubbing. Playback loops at the end.

Frames are read from the tifs on demand — nothing is preloaded, so movies of any
size open instantly and RAM stays flat. Display is spatially downsampled by --ds
(default 2) for speed; the traces are untouched.

Usage:
  python trace_viewer_local.py "/path/to/run"                    # motion energy
  python trace_viewer_local.py "/path/to/run" --trace intensity  # mean fluorescence
  python trace_viewer_local.py "/path/to/run" --ds 4             # faster display
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from box_extract_local import (  # noqa: E402
    run_tifs, to_gray, proc_dir, boxes_path, truncated_memmap)

try:
    import tifffile
except ImportError:
    sys.exit("trace_viewer_local.py requires tifffile")


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
    def __init__(self, run, trace_kind="motion", ds=2, fps=15):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button, Slider

        self._plt = plt
        run = Path(run)
        self.ds = max(1, int(ds))

        npzs = sorted(proc_dir(run).glob("*_boxtraces.npz"))
        if not npzs:
            sys.exit(f"No *_boxtraces.npz in {proc_dir(run)} — run box_extract_local.py extract first.")
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
        self.fig.canvas.manager.set_window_title(
            f"trace_viewer_local — {run.name} [{trace_kind}]")
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
        kind = ("mean intensity  F  (ADU)" if trace_kind == "intensity"
                else "motion energy  mean|ΔF|  (ADU / frame)")
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


MOTIF_BOXES = ["whisker_pad", "paw", "wheel"]


class MotifViewer:
    """Movie beside the LOG motion energy of the motif channels only.

    Separate from Viewer on purpose: Viewer shows every box on linear axes, which
    is right for checking box placement. This one shows a chosen few on a log
    axis — the space the bout detector actually works in (log ME + envelope), so
    what you watch here is what whisk_bouts.py sees.

    If <run>/proc/*_whiskbouts.npz exists, detected bouts are shaded on the
    whisker panel and onsets ticked, which makes this the tool for the
    validation gate: scrub to an onset and confirm the animal really started
    whisking there.
    """

    def __init__(self, run, boxes=None, ds=2, fps=15, env_s=0.33, cam_fps=15.0):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Button, Slider

        self._plt = plt
        run = Path(run)
        self.ds = max(1, int(ds))

        npzs = sorted(proc_dir(run).glob("*_boxtraces.npz"))
        if not npzs:
            sys.exit(f"No *_boxtraces.npz in {proc_dir(run)} — extract first.")
        d = np.load(npzs[0], allow_pickle=True)
        all_names = [str(n) for n in d["box_names"]]
        want = boxes or MOTIF_BOXES
        self.names = [n for n in want if n in all_names]
        if not self.names:
            sys.exit(f"none of {want} in this run (has: {all_names})")
        missing = [n for n in want if n not in all_names]
        if missing:
            print(f"(not in this run, skipped: {', '.join(missing)})")
        sel = [all_names.index(n) for n in self.names]
        me = d["motion_energy"][sel].astype(float)
        # motion_energy[:,0] is a definitional zero (no previous frame);
        # log() of it would be a huge false outlier that flattens the y axis.
        # Replace rather than drop, so panel index == movie frame == the index
        # whisk_bouts records its onsets in.
        if me.shape[1] > 1:
            me[:, 0] = me[:, 1]
        coords = d["box_coords"][sel]

        self.src = FrameSource(run_tifs(run))
        self.n = min(len(self.src), me.shape[1])
        self.log_me = np.log(me[:, :self.n] + 1e-3)
        w = max(1, int(round(env_s * cam_fps)))
        pad = w // 2
        self.env = np.stack([
            np.convolve(np.pad(t, pad, mode="edge"), np.ones(w) / w,
                        mode="same")[pad:pad + self.n] for t in self.log_me])

        # detected bouts, if the detector has been run on this run
        self.bouts = []
        wb = sorted(proc_dir(run).glob("*_whiskbouts.npz"))
        if wb:
            b = np.load(wb[0], allow_pickle=True)
            self.bouts = list(zip(b["onsets"].tolist(), b["offsets"].tolist()))
            print(f"({len(self.bouts)} detected whisking bouts overlaid from "
                  f"{wb[0].name})")

        self.k, self.playing, self._syncing = 0, False, False

        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title(f"motif viewer — {run.name}")
        self.ax_img = self.fig.add_axes([0.015, 0.16, 0.52, 0.80])
        frame = self.src.get(0)[::self.ds, ::self.ds]
        vmin, vmax = np.percentile(frame, [1, 99.5])
        self.im = self.ax_img.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax,
                                     interpolation="nearest")
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        self.title = self.ax_img.set_title(f"frame 0 / {self.n - 1}", fontsize=10)
        for name, (x0, y0, x1, y1) in zip(self.names, coords):
            self.ax_img.add_patch(mpatches.Rectangle(
                (x0 / self.ds, y0 / self.ds), (x1 - x0) / self.ds,
                (y1 - y0) / self.ds, fill=False, edgecolor="lime", linewidth=1.5))
            self.ax_img.text(x0 / self.ds, y0 / self.ds - 3, name, color="lime",
                             fontsize=9, fontweight="bold")

        nb = len(self.names)
        self.cursors = []
        top, bottom = 0.96, 0.16
        h = (top - bottom) / nb
        axes = []
        for i, name in enumerate(self.names):
            ax = self.fig.add_axes([0.60, top - (i + 1) * h + 0.012, 0.385,
                                    h - 0.02])
            ax.plot(self.log_me[i], lw=0.5, color="0.65")
            ax.plot(self.env[i], lw=1.2, color="#1f3a68")
            if name == "whisker_pad":
                for s_, e_ in self.bouts:
                    ax.axvspan(s_, e_ + 1, color="tab:green", alpha=0.20, lw=0)
            lo, hi = self.log_me[i].min(), self.log_me[i].max()
            pad_y = 0.05 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad_y, hi + pad_y)
            ax.set_xlim(0, self.n - 1)
            ax.tick_params(labelsize=7)
            ax.text(0.005, 0.84, name, transform=ax.transAxes, fontsize=9,
                    fontweight="bold", color="tab:blue")
            if i < nb - 1:
                ax.set_xticklabels([])
            self.cursors.append(ax.axvline(0, color="r", lw=1))
            axes.append(ax)
        axes[0].set_title("log motion energy (thin = raw, thick = envelope"
                          + (", green = detected bout)" if self.bouts else ")"),
                          fontsize=9)
        axes[-1].set_xlabel("camera frame", fontsize=9)

        bax = self.fig.add_axes([0.015, 0.045, 0.07, 0.055])
        self.btn_play = Button(bax, "play")
        self.btn_play.on_clicked(lambda e: setattr(self, "playing", True))
        bax = self.fig.add_axes([0.095, 0.045, 0.07, 0.055])
        self.btn_pause = Button(bax, "pause")
        self.btn_pause.on_clicked(lambda e: setattr(self, "playing", False))
        sax = self.fig.add_axes([0.24, 0.06, 0.14, 0.03])
        self.s_fps = Slider(sax, "fps", 1, 60, valinit=fps, valstep=1)
        self.s_fps.on_changed(lambda v: setattr(self.timer, "interval",
                                                int(1000 / max(1, int(v)))))
        sax = self.fig.add_axes([0.45, 0.06, 0.42, 0.03])
        self.s_frame = Slider(sax, "frame", 0, self.n - 1, valinit=0, valstep=1)
        self.s_frame.on_changed(self._on_scrub)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._tick)
        self.timer.start()

    def _on_key(self, event):
        """space = play/pause; n / p = jump to next / previous detected onset."""
        if event.key == " ":
            self.playing = not self.playing
        elif event.key in ("n", "p") and self.bouts:
            ons = np.array([s for s, _ in self.bouts])
            nxt = ons[ons > self.k] if event.key == "n" else ons[ons < self.k]
            if len(nxt):
                self.playing = False
                self._show(int(nxt[0] if event.key == "n" else nxt[-1]))

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
    ap.add_argument("--trace", choices=["intensity", "motion"], default="motion",
                    help="which trace to plot (default: motion energy)")
    ap.add_argument("--ds", type=int, default=2, help="display downsample factor")
    ap.add_argument("--fps", type=int, default=15, help="initial playback rate")
    ap.add_argument("--motifs", action="store_true",
                    help="MotifViewer: log motion energy of the motif channels "
                         "only (whisker_pad, paw, wheel) beside the movie, with "
                         "detected whisking bouts shaded if available")
    ap.add_argument("--boxes",
                    help="with --motifs: comma-separated boxes to show "
                         f"(default {','.join(MOTIF_BOXES)})")
    args = ap.parse_args()
    if not boxes_path(args.run).exists():
        sys.exit(f"No proc/boxes.json in {args.run}")
    if args.motifs:
        boxes = [b.strip() for b in args.boxes.split(",")] if args.boxes else None
        MotifViewer(args.run, boxes=boxes, ds=args.ds, fps=args.fps).run()
    else:
        Viewer(args.run, trace_kind=args.trace, ds=args.ds, fps=args.fps).run()


if __name__ == "__main__":
    main()

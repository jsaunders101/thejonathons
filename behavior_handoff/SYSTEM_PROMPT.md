# System prompt — CSHL M1-Arousal: behavior extraction (collaborator copy)

Paste everything below this line into your Claude session (or CLAUDE.md) to initialize it.

---

You are assisting with the **behavior-video arm** of a CSHL ISFNS 2026 summer-course final
project. Four PhD students/postdocs built a 2-photon + behavior rig in one day, collected two
days of data, and have **one analysis day**. The deliverable is a 30–40 min talk with
**poster-level** (not publication-level) figures. Speed and reliability beat elegance;
tested-and-working beats clever.

## The science, in brief
Question: how does arousal state relate to ongoing (spontaneous) activity in mouse M1?
- 2P GCaMP7s in M1 excitatory neurons (Bruker/PrairieView, single plane, 30–50 Hz — varies
  per recording), 2 awake headfixed mice, no task.
- Behavior: high-res IR ThorCam video (face, whiskers, paws, nose, wheel), multi-tif stacks.
- Sniff thermistor on NI DAQ (PrairieView clock). No rotary encoder — paw-box motion energy
  is the locomotion proxy.
- Arousal manipulation: ambient odor (peanut butter vs TMT on a cotton ball, placed BEFORE
  recording) — a per-session context label, never an event to align to.
- Arousal readouts, by reliability: sniff > whisking (box motion energy) > paw motion > eye.

## Your scope in this package
The **box-ROI behavior extraction pipeline and its QC viewer** (`code/`). It is validated
end-to-end on real ThorCam data and is the team's working tool — treat it as trusted
infrastructure to USE and extend carefully, not to rewrite.

**Two versions of every script.** `*_cluster.py` (run these on Elzar; trace writing is
guarded to the cluster) and `*_local.py` (run these on your own machine; guard off).
The code is otherwise identical — the suffix is the only difference, and a
`CLUSTER-ONLY` exit means you picked the wrong twin. Commands below say `_cluster`;
substitute `_local` when you are off-cluster. If you change one version, change the
other identically — the owner's repo has `check_branches.py`, which fails on any
divergence beyond the branch name.

**Explicitly out of scope:** 2P↔camera synchronization (`sync_detect.py`), run↔t-series
pairing, sniff processing, and bundle building. That code exists on the owner's side and is
still being refined — do NOT build your own version; coordinate with Seneca
(seneca_scott@brown.edu) instead. The one sync-relevant thing you MUST preserve: every run
needs a `laser_trigger` box drawn on a **non-saturated** image region (a clipped-white patch
cannot show the 2P laser-onset brightness step that sync detection depends on).

## Inviolable conventions (downstream code depends on these)
1. **The proc/ contract.** Every derived file lives in `<run>/proc/` inside the run folder it
   came from. Raw tif folders are never written to. A "run" = any folder directly containing
   ≥1 .tif; multiple tifs in a run are ONE movie, concatenated in natural-sort order.
2. **Canonical box categories** — `laser_trigger, whisker_pad, paw, paw_at_nose, eye, nose, wheel`
   (`CANONICAL_BOXES` in box_extract_cluster.py). One box per category, enforced (duplicates
   hard-error). Do not rename categories or npz keys (`traces`, `motion_energy`, `box_names`,
   `box_coords`) — MATLAB analysis and the bundle builder consume them as-is.
3. **Outputs are idempotent and atomic** — extraction skips existing outputs (`--force` to
   redo) and writes via tmp+rename. Keep it that way in any change you make.
4. **traces** = mean pixel intensity per box per frame (n_boxes × n_frames, float32);
   **motion_energy** = mean |frame_t − frame_{t−1}| per box, column 0 = 0.

## Working practices
- Environment: Python ≥3.9 with numpy, tifffile (≥2023), matplotlib; scipy optional (.mat
  export). Extraction and testing are headless-safe; only the draw GUI needs a display.
- On a SLURM cluster: freeze the run list first (`list --save runs.txt`), then array-index it
  (`extract --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID`). Never let array tasks
  re-scan a directory tree that may still be uploading. ~4G memory is generous (streaming).
- Before changing box_extract_cluster.py, run the regression suites:
  `MPLBACKEND=Agg python test_box_gui_cluster.py` and
  `MPLBACKEND=Agg python test_extract_stream_cluster.py /tmp/es_test` — and keep them
  passing. If you fix a bug, add a check. The extraction suite pins trace values
  bit-exactly against the original per-frame implementation: if it fails, your change
  altered the science, not just the plumbing.
- Extraction streams in bounded blocks (`--batch-mb`, default 256 MB) so 80+ GB movies
  don't OOM; don't replace it with a whole-movie load.
- macOS-uploaded data contains `._*` AppleDouble junk; discovery already filters it — don't
  weaken that.
- QC is mandatory, not optional: after extraction, spot-check runs in trace_viewer_cluster.py
  (movie + traces side by side) before trusting any downstream analysis. Zoom (toolbar
  magnifier) before drawing boxes — at full-frame scale, facial features are easy to
  misidentify (a real mistake that cost this team an hour: an eye labeled as a nose).

## Quick command reference
```
python box_extract_cluster.py draw    <roots>                  # GUI: drag box → click category button
python box_extract_cluster.py draw    <roots> --boxes-from <run>/proc/boxes.json   # no GUI, reuse boxes
python box_extract_cluster.py extract <roots> [--force]        # stream movies → proc/*_boxtraces.npz + .mat
python box_extract_cluster.py list    <roots> [--save runs.txt]
python box_extract_cluster.py collect <roots> --dest <dir>     # stage proc outputs (never raw movies)
python trace_viewer_cluster.py <run> [--trace intensity] [--ds 4]  # QC playback (default: motion energy)
```

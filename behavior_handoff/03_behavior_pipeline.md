# Behavior pipeline: usage & status

## run_behavior_cluster.py — one-command flow (NEW)

Master wrapper: folder dialog → draw → per-run extract with the trace viewer opening
immediately after each run's extraction (close window = next run; `q` = stop viewers).
GUI availability is probed by actually opening a window, then degrades cleanly:
dialog falls back Tk → matplotlib browser → typed path; review falls back to static
`<run>/proc/*_tracesqc.png` files (`--qc-png` forces them). Failing runs are isolated
and reported; the rest still process. For cluster GUIs use an Elzar OnDemand
interactive Desktop session (native windows, no X forwarding).

```bash
python run_behavior_cluster.py                          # full interactive flow
python run_behavior_cluster.py /data/behavior           # skip the dialog
python run_behavior_cluster.py /data/behavior --boxes-from <run>/proc/boxes.json --qc-png
```

Tests: `MPLBACKEND=Agg python test_run_behavior_cluster.py /tmp/rb_test` (14 checks;
the `_local` twin reports 13 PASS + 1 SKIP — the cluster-guard check is off by design there).
Also: `MPLBACKEND=Agg python test_extract_stream_cluster.py /tmp/es_test` (extraction
equivalence + memory bound).

## box_extract_cluster.py — two phases

**draw** (interactive; needs a display; reads ONE frame per run, so it's cheap anywhere):
```bash
python box_extract_cluster.py draw /data/behavior
python box_extract_cluster.py draw /data/behavior --boxes-from <run>/proc/boxes.json   # no GUI
python box_extract_cluster.py draw /data/behavior --overwrite                          # redraw
```
GUI: drag a rectangle on the first frame, then click a category button to assign it.
- One box per category — re-clicking a used category is refused (use `undo last`)
- A drawn rectangle is consumed once; clicking a second category without a NEW drag is
  refused (regression-tested — prevents silent duplicate boxes)
- `DONE` warns once about missing categories; second click accepts
- `reuse previous` copies the prior run's boxes (fast for many runs, camera unmoved)
- **Zoom with the toolbar magnifier before drawing** — at full-frame scale features are
  easy to misidentify (an eye was once labeled "nose" on this very project)

**extract** (headless, streaming — RAM stays flat at any movie size):
```bash
python box_extract_cluster.py extract /data/behavior                 # skips runs already done
python box_extract_cluster.py extract /data/behavior --force         # redo
python box_extract_cluster.py list    /data/behavior --save runs.txt # status table + frozen list
python box_extract_cluster.py extract --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
python box_extract_cluster.py collect /data/behavior --dest /staging # proc outputs only, flat names
```

SLURM template:
```bash
#SBATCH --array=0-19          # n_runs-1, from list --save
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00       # calibrate from one real run
python box_extract_cluster.py extract --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
```
Freeze runs.txt at submit time — never let array tasks re-scan a tree still uploading.

## trace_viewer_cluster.py — QC (use it, every session batch)
```bash
python trace_viewer_cluster.py <run>                 # movie + per-box traces, synced cursor
python trace_viewer_cluster.py <run> --trace motion  # motion energy instead of intensity
python trace_viewer_cluster.py <run> --ds 4 --fps 30 # faster display / playback
```
Play/pause buttons (spacebar toggles; note plain `q` closes matplotlib windows), fps slider,
frame scrubber; loops at end. Frames are read on demand — 3 GB movies open instantly.
Use it to catch misplaced boxes and drift BEFORE batching a day of data.

## Engineering properties you inherit (keep them true)
- Idempotent: extract skips existing outputs; `--force` to redo
- Atomic: npz/.mat written via tmp+rename — no partial outputs ever look complete
- Short movies salvaged, never silently: a tif ending mid-frame (`failed to read N
  bytes, got M` — usually an upload still in flight) stops reading at the last complete
  frame, flags the npz (`truncated`), and makes the run exit 2. Frames are always a
  contiguous prefix — never skip a bad frame and keep going, that shifts every later
  camera-clock timestamp. Check a fresh upload with
  `python box_extract_cluster.py verify /path/to/behavior` before batching.
- Bounded memory: frames stream in blocks capped by `--batch-mb` (default 256 MB), so
  RAM is flat regardless of movie size. Single-IFD contiguous giants (ImageJ
  'truncated' >4 GB files) are read via `np.memmap` — note the earlier per-page
  reader saw only ONE frame in such files, so if your data is one huge tif per run
  rather than ~1 GB parts, re-extract with this version. Peak RSS prints per run.
- Hot loop: frames stay native uint16; only crops convert to float32
  (verified BIT-IDENTICAL to the naive implementation on real 3 GB data, ~4× less CPU;
  the batched rewrite is equality-tested against the same reference)
- Junk-proof discovery: single os.walk pass, proc/ pruned, AppleDouble/empty/hidden ignored

## Validation status (as handed off)
- Synthetic ground truth: multi-tif concatenation, boundary-exact traces, RGB fallback — pass
- Real data (3×1 GB ThorCam, 1035 frames): GUI, extraction, .mat load, viewer — pass
- GUI regression suite: `MPLBACKEND=Agg python test_box_gui_cluster.py` → 9 checks, incl. the
  stale-selection duplicate-box bug caught on real data
- Extraction equivalence suite (`test_extract_stream_cluster.py`): batched streaming is
  bit-identical to the pre-batching per-frame math on paged / RGB / truncated-giant /
  aborted-giant layouts at several block sizes; a ~400 MB movie at `--batch-mb 16`
  peaks under 60 MB RSS. Pass on py3.10/np1.26 and py3.13/np2.x
- Two fixes landed Aug 9 PM: the GUI probe treated TkAgg as headless (`"agg"` is a
  substring of `"tkagg"`) — now an exact-name backend test; and extraction OOM'd on
  very large movies — now bounded-block streaming
- Untested: GUIs on the cluster — planned route is an **Elzar OnDemand interactive
  Desktop session** (native windows; run run_behavior_cluster.py in its terminal). `ssh -Y`
  X-forwarding is the fallback. Everything stays on the cluster — draw there, not on
  laptops (policy: no local copies of data or outputs).

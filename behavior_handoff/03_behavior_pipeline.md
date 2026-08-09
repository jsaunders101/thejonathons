# Behavior pipeline: usage & status

## box_extract.py — two phases

**draw** (interactive; needs a display; reads ONE frame per run, so it's cheap anywhere):
```bash
python box_extract.py draw /data/behavior
python box_extract.py draw /data/behavior --boxes-from <run>/proc/boxes.json   # no GUI
python box_extract.py draw /data/behavior --overwrite                          # redraw
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
python box_extract.py extract /data/behavior                 # skips runs already done
python box_extract.py extract /data/behavior --force         # redo
python box_extract.py list    /data/behavior --save runs.txt # status table + frozen list
python box_extract.py extract --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
python box_extract.py collect /data/behavior --dest /staging # proc outputs only, flat names
```

SLURM template:
```bash
#SBATCH --array=0-19          # n_runs-1, from list --save
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00       # calibrate from one real run
python box_extract.py extract --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
```
Freeze runs.txt at submit time — never let array tasks re-scan a tree still uploading.

## trace_viewer.py — QC (use it, every session batch)
```bash
python trace_viewer.py <run>                 # movie + per-box traces, synced cursor
python trace_viewer.py <run> --trace motion  # motion energy instead of intensity
python trace_viewer.py <run> --ds 4 --fps 30 # faster display / playback
```
Play/pause buttons (spacebar toggles; note plain `q` closes matplotlib windows), fps slider,
frame scrubber; loops at end. Frames are read on demand — 3 GB movies open instantly.
Use it to catch misplaced boxes and drift BEFORE batching a day of data.

## Engineering properties you inherit (keep them true)
- Idempotent: extract skips existing outputs; `--force` to redo
- Atomic: npz/.mat written via tmp+rename — no partial outputs ever look complete
- Hot loop: frames stay native uint16 in a reused read buffer; only crops convert to float32
  (verified BIT-IDENTICAL to the naive implementation on real 3 GB data, ~4× less CPU)
- Junk-proof discovery: single os.walk pass, proc/ pruned, AppleDouble/empty/hidden ignored

## Validation status (as handed off)
- Synthetic ground truth: multi-tif concatenation, boundary-exact traces, RGB fallback — pass
- Real data (3×1 GB ThorCam, 1035 frames): GUI, extraction, .mat load, viewer — pass
- GUI regression suite: `MPLBACKEND=Agg python test_box_gui.py` → 9 checks, incl. the
  stale-selection duplicate-box bug caught on real data
- Untested: draw GUI over `ssh -Y` (X-forwarding); if laggy, draw locally on downloaded
  first frames and push boxes.json back with `--boxes-from`

# ca_pipeline — batchable calcium extraction

`ca_extract.py` is the course notebook (denoise/MC/corr-map/ROI pipeline) converted to a
tested, batchable CLI. Same algorithms, engineered for the cluster.

## Quick start

```bash
# sanity check the install (~2 min, synthetic ground truth):
python test_ca_extract.py /tmp/ca_test

# ONE t-series (validate/troubleshoot; exit 3 = completed but QC-flagged):
python ca_extract.py run "/path/to/.../TSeries-xxx" --out /path/to/ca_output \
    --data-root "/path/to/data_root"

# everything under a parent folder (recursive; mirrors the full tree first):
python ca_extract.py batch "/path/to/data_root" --out /path/to/ca_output

# SLURM array:
python ca_extract.py list "/path/to/data_root" --save runs.txt
python ca_extract.py batch "/path/to/data_root" --out /path/to/ca_output \
    --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
```

## Output mirroring (the organization contract)
Outputs NEVER touch the data tree. `--out` receives an **exact mirror of the input
hierarchy including empty directories** (slots for behavior outputs later). Each
t-series' outputs land in its mirrored folder: `ca.npz`, `ca.mat`, `meta.json`
(ALL PrairieView XML metadata + companion-file inventory incl. VoltageRecording),
`qc.json` (metrics + flags), `qc_motion.png`, `qc_roi.png`. Batch appends to
`<out>/ca_summary.csv`.

`ca.npz`/`ca.mat` keys: `F_raw` (summed pixel F, n_roi × T), `roi_npix`, `dff`
(static bottom-15% F0), `F0`, `roi_map`, `roi_xy`, shifts (both MC passes),
`mean_img`, `corr_map`, `frame_times` (2P clock), `fps`, provenance (params hash +
code version). **Do not rename keys** — MATLAB analysis and the bundle builder consume them.

## Params
Defaults in `PARAMS_DEFAULT` (top of ca_extract.py); override any subset with
`--params params.json`. Freeze a params.json after validating the ground-truth run and
use it for every batch run.

## Deliberate changes vs the notebook (all ground-truth tested)
1. **`mc_normalization: None`** — skimage's default `normalization="phase"` measured
   ~10× WORSE registration on synthetic 2P-like ground truth (0.6 vs 0.06 px RMS).
   The notebook silently inherits the bad default.
2. **Fixed a real notebook bug**: `this_trace = frames[:, i, j]` is a numpy VIEW —
   the notebook's `this_trace += ...` mutates the movie in place, corrupting seed-pixel
   traces for later ROIs. `.copy()` here; flag this to anyone still running the notebook.
3. Correlation map computed in ONE streaming pass (exact same Pearson values, no extra
   movie copy); `pearsonr` replaced with an identical dot-product formula. Equivalence
   proven in the test suite to 1e-6.
4. Raw movie stays uint16; only one float32 aligned copy exists (pass-1 template is a
   streamed accumulator). ~15-min 512² run ≈ 43 GB peak — request cluster memory accordingly.
5. PCA/Gaussian denoising: computed-but-unused in the notebook → excluded entirely.
6. Frame count is ASSERTED against the XML (partial uploads refuse to process);
   channel is auto-detected (never hardcoded Ch2); frame period/times come from the XML.

## Flags (warn, humans decide)
`motion` (p99 total shift > 10 px), `few_rois` (< 10), `frame_period_jitter`.
Flagged runs exit 3 / appear flagged in ca_summary.csv — review their QC pngs.

## Known limitation
**Thorlabs-recorded t-series are not yet supported** (different layout/XML — discovery
keys on `TSeries*.xml`). Send one Thorlabs folder listing to get the loader wired.

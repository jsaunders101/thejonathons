# ca_extract.py — deep audit (Aug 9 PM, sprint)

Stage-by-stage walkthrough of every computation, with verdicts. Nothing here has been
changed yet — proposed changes are listed at the end as options awaiting team confirmation.

## Stage-by-stage logic

### 0. Discovery & mirroring
- A t-series = any folder containing `TSeries*.xml`; discovery never descends inside one.
  **ASSUMPTION**: PrairieView's default naming. VoltageRecording XMLs also start with
  "TSeries" — the main XML is picked as the SHORTEST filename in the folder. Holds for
  standard PV output; breaks if files are renamed.
- **FLAG (real risk): if `--out` is INSIDE the data root**, `mirror_tree` walks the data
  root and will mirror the output tree into itself on re-runs (out/out/... accumulation).
  Their planned layout (`GECI Project Jonathons/Data to Analyze/ca_output` while batching
  `GECI Project Jonathons/` itself) hits this. Currently NO guard.
- **FLAG: mirror paths depend on the chosen root.** `run` mode defaults data-root to
  3 levels above the t-series (`<root>/<mouse>/<run>/<ts>` — matches current layout);
  `batch` mode uses the parent you pass. Batch over `GECI Project Jonathons/` vs run with
  the 3-up default puts THE SAME t-series at two different mirrored paths ("Old
  Data/jonathans_finale/..." vs "..."), and skip-if-exists won't see across them.
  One canonical data root must be agreed and always passed.

### 1. XML metadata
- Full parse: every `Frame` relativeTime/absoluteTime, PVStateShard (nested
  Indexed/Subindexed values), sequence wall-clock time, scan date. fps = 1/median(Δt) —
  robust to logging jitter. Companion files inventoried (VoltageRecording flagged).
- Minor: frames missing a `relativeTime` attribute are skipped from `frame_times` but
  still counted in `n_frames_xml` — a malformed XML could make the two inconsistent.
- Note for sync stage later: `scan_date` ("8/9/2026 2:11:00 PM") and `sequence_time`
  ("14:11:02.5") are stored raw, not yet combined into one timestamp.

### 2. Loading
- Channel auto-detected from `_Ch(\d+)_`; multiple channels = hard error (correct per
  team: only one channel ever recorded, number varies).
- `tifffile.imread(first_file)` normally aggregates the whole OME series; fallback
  path preallocates and concatenates all files in natural-sort order.
- Frame count is ASSERTED equal to the XML count — partial uploads/aborted saves refuse
  to process. **ASSUMPTION**: no legitimately-short runs exist. An aborted t-series
  (XML logs more frames than were saved) will hard-error — currently no override.
- **ASSUMPTION**: files end `.ome.tif`. Some PV configs write plain `.tif` — currently
  "not supported" error (same message as Thorlabs).
- **OPEN RISK: Thorlabs-recorded t-series entirely unsupported** (different layout/XML).
  Still waiting on one folder listing.

### 3. Motion correction (two-pass rigid, notebook-faithful)
- Pass 1: template = raw mean → per-frame subpixel shifts (0.1 px) → aligned MEAN built
  by streaming accumulator (no full aligned copy). Pass 2: re-register raw frames to
  that mean → one float32 aligned movie.
- `normalization=None` default — measured 10× more accurate than skimage's default
  "phase" on ground truth (0.06 vs 0.6 px RMS). Old-skimage fallback probed at runtime.
- Memory: raw uint16 + ONE float32 copy ≈ 1.5× float32-movie. 512²×27k ≈ **43 GB peak**
  → needs ≥64 GB nodes. 256² ≈ 11 GB. **frames.shape still unknown to us — this is the
  single most important unanswered question.**
- **ASSUMPTION**: rigid motion only (no within-frame/non-rigid correction). Course-level
  standard; belongs on the caveats slide.
- Edge fill after shifting is 0 → handled by zeroing a border = max(5, max|shift|+1) in
  the correlation map.

### 4. Correlation map (streaming exact Pearson)
- Pixel vs sum-of-8-neighbors, single pass, float64 accumulators; proven equal to the
  notebook's pearsonr loop to 1e-6. No extra movie copy.
- Detection runs on a temporally BINNED movie (`detect_bin_frames`, auto ≈ 6 Hz) — fixes
  the shot-noise corr-map collapse observed on real data (0.17 ceiling). Regression test
  reproduces symptom and recovery. Last partial bin block is dropped (detection only).

### 5. ROI growth (greedy, notebook-faithful + fixes)
- Same algorithm; fixes vs notebook: seed-trace `.copy()` (notebook mutates the movie
  in place via a view), dot-product Pearson (identical values), sequential labels so
  `roi_map` label k == row k of every trace array (notebook's labels desync from rows
  after size filtering).
- Growth happens on the binned detection movie; final `F_raw` = sum over member pixels
  of the FULL-RATE movie (order-independent — identical to the grown sum on the same movie).
- Note: `n_rois` (100) counts ATTEMPTS incl. size-rejected ones. With binned detection
  being cheap, a denser FOV may deserve 200–300 attempts.

### 6. dF/F & outputs
- `dff = (F_raw − F0)/F0`, F0 = static bottom-15% per ROI (per decision; sliding nixed).
  npix varies per ROI but dff normalizes the sum scale away; `roi_npix` saved.
- **ASSUMPTION**: no neuropil correction (none in the notebook either) — caveats slide.
- **GAP**: no photobleaching metric — under strong bleach a static F0 sits near the
  late-trace floor and inflates early dff. Cheap check possible (see options).
- Outputs atomic (tmp+rename), idempotent (skip-if-exists / --force), provenance
  (params hash + code hash) embedded.

### QC & batch
- Flags: motion (p99 shift > 10 px — absolute, not FOV-relative), few_rois (<10),
  weak_corr_map (max < threshold), frame-period jitter. Warn-only; humans decide.
- **GAP**: no per-ROI SNR metric; best/worst traces chosen by dff amplitude only.
- `ca_summary.csv` APPENDS with no header → duplicate rows on re-runs; contains only
  ok/flagged, no metrics.
- Batch mirror-then-process; concurrent array tasks race mkdir benignly (exist_ok).
- **INCONSISTENCY with team policy (Aug 9)**: box_extract refuses to write traces
  off-cluster; ca_extract has NO such guard.

## Proposed changes — CONFIRM/REJECT each (nothing applied yet)

**Group A — correctness (recommend all; ~30 min total, tested):**
- A1. Refuse `--out` inside the data root (or auto-exclude it from walk/discovery).
- A2. Stamp the canonical data root into `<out>/.mirror_root` on first use; warn/refuse
  on mismatch so every invocation mirrors to the same relative paths.
- A3. `--allow-short` flag: accept XML>disk frame counts by truncating to disk length
  (for aborted runs), default OFF (strict).
- A4. Accept plain `.tif` Cycle files (filter on `_Cycle`), not just `.ome.tif`.

**Group B — QC depth (recommend B1+B3; ~30 min):**
- B1. Bleach metric: mean-F drift first→last 10% per run, flag if > X% (default 20?).
- B2. Per-ROI SNR (transient amplitude / MAD noise) in qc.json.
- B3. ca_summary.csv: header + columns (n_frames, fps, n_rois, shift_p99, corr_max,
  flags) + rewrite-per-run instead of blind append.

**Group C — policy/UX:**
- C1. Cluster-only write guard matching box_extract (`M1AROUSAL_ALLOW_LOCAL=1` override).
- C2. **v3 notebook** that imports ca_extract's stage functions — one narrated cell per
  stage (load→MC→corrmap→ROIs→dff) with the intermediate figures inline. Interactive
  interpretability, zero duplicated logic. (~30 min)
- C3. Raise default `n_rois` attempts 100 → 300 (binned detection makes it cheap).

**Group D — assumptions we propose to ACCEPT for the sprint (flag on caveats slide):**
rigid-only MC; no neuropil; shortest-XML heuristic; single plane; one channel;
static-percentile F0.

## Open questions (answers unblock the most)
1. **frames.shape of a real run + RAM of the nodes you're using** — decides everything
   about memory strategy and SLURM requests.
2. Did the binned corr map on real data jump above ~0.4? (Confirms the shot-noise
   diagnosis end-to-end; if not, next suspects: bidirectional scan phase, laser power.)
3. Confirm the canonical data root + output root paths (for A1/A2 exactly).
4. Thorlabs t-series: one `ls -R` of a folder. How many runs are Thorlabs?
5. Any aborted/short t-series in the dataset (decides whether A3 matters)?
6. Expected cells per FOV (calibrates n_rois attempts + few_rois flag)?

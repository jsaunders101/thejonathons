# Behavioural motif analysis

Turning per-box **motion energy** (from `../behavior_handoff/`) into behavioural
events: whisking bouts, movement onsets, and low/medium/high movement states —
the layer that sits between trace extraction and the neural analysis.

Everything here is **derived from `<run>/proc/<run>_boxtraces.npz`** and writes new
files. No existing npz key changes, so the frozen extraction schema is untouched.
Outputs stay in **camera frames**; conversion to the 2P clock is the sync stage's job.

## What works today

| File | Status |
|---|---|
| [whisk_bouts.py](whisk_bouts.py) | ✅ **built and validated** — whisking bout + onset detection |
| [test_whisk_bouts.py](test_whisk_bouts.py) | ✅ 7 checks, passes on py3.10/np1.26 and py3.13/np2.x |
| [10_behavior_motifs.md](10_behavior_motifs.md) | plan: onsets + states, feasibility measured |
| [11_whisking_onset.md](11_whisking_onset.md) | deep design for whisking onsets |
| [13_running_onset.md](13_running_onset.md) | ⚠ running onsets — **blocked on a data question** |
| [analysis_probes/](analysis_probes/) | the measurements every design decision rests on |

```bash
python whisk_bouts.py run     "/path/to/run"      # detect -> proc/*_whiskbouts.npz + .json + qc png
python whisk_bouts.py inspect "/path/to/run"      # onset-timing figure for eyeballing
python whisk_bouts.py batch   "/path/to/behavior" --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
MPLBACKEND=Agg python test_whisk_bouts.py /tmp/wb # the suite
```

Frame rate is read from the ThorCam tif tags automatically (measured 15.0000 fps,
no drift); `--fps` overrides. Idempotent, atomic writes, per-run failure isolation —
same engineering contract as the extraction side.

**Validated on a real 9.9-minute session:** d′ = 3.56, 80 bouts (8.1/min, median
1.0 s). The independent threshold detector agrees on 76/80 onsets within 2 frames,
and the onset-triggered average of *raw* log ME puts the half-rise at +1 frame —
i.e. the mark sits at the foot of the rise, which is the intended definition.
On synthetics across 20 noise seeds: hit rate 0.982, 5 false positives, onset bias
−2 frames.

To watch the traces against the video with detected bouts shaded, use the motif
viewer in the extraction package (`n` / `p` jump between onsets):

```bash
python ../behavior_handoff/code/trace_viewer_local.py "/path/to/run" --motifs --ds 3
```

## Five findings that shaped the design

Each was measured, and several contradicted the obvious choice. Reproduce with
`analysis_probes/`.

1. **Whisk cycles are unrecoverable.** Whisking is 8–12 Hz, the camera is 15 fps,
   Nyquist 7.5 Hz. The spectrum is 1/f with no peak. Only the bout *envelope*
   exists, so "onset" always means bout onset, never first protraction.
2. **The whiskers never stop.** The two components in log ME are *low-amplitude
   ongoing whisking* (75%) and *bouts* (25%) — not rest vs movement. Detection is a
   step up from a moving baseline.
3. **Do not detrend.** Subtracting a rolling median destroys separation at every
   window tried (d′ 2.63 → 0.82 at 30 s): at a ~25% duty cycle the median tracks
   activity instead of baseline. Envelope smoothing alone improves d′ monotonically.
4. **Do not regress out the common mode.** `laser_trigger` correlates 0.52 with
   whisker_pad, but that is almost entirely slow (0.09 after removing >1 s), and
   regressing it out degrades d′ monotonically to 0.90. Use it only as a spike veto.
5. **"Motion power" and "motion energy" are the same information.** L2 ≈ L1^1.8,
   Spearman ρ ≈ 0.99; after the log transform d′ is unchanged. Keep L1. A *motion
   **area*** channel (fraction of pixels changing) is a different story — it raises
   wheel d′ from 2.86 to 6.98 and is the biggest available win for the locomotion
   channels, but it needs a new npz key and a re-extract.

## Two traps worth reading before writing analysis code

**State thresholds must be pooled across sessions, never within-session tertiles.**
Within-session tertiles force 33% occupancy in every session by construction and
structurally destroy the PB-vs-TMT arousal comparison while still producing a plot.

**The statistical unit is the bout, not the frame.** dF/F at 10 Hz is heavily
autocorrelated; treating frames as independent inflates tests by roughly an order of
magnitude. Average within bout, then test across bouts.

## Open

Running-onset extraction is **blocked on data, not code** — see
[13_running_onset.md](13_running_onset.md). The gold session contains no sustained
locomotion (0 episodes ≥1 s of concurrent paw+wheel activity outside grooming; 85%
of paw-active frames are grooming), and there is no evidence the `wheel` box sees
the wheel. Step 0 is watching a session in the viewer before building anything.

## Note on cross-links

The `.md` files here cross-reference each other correctly, but also link to docs in
the owner's project folder (`03_synchronization.md`, `05_analysis_plan.md`,
`12_bundle_builder.md`) which are not in this repo. Those links will not resolve
here. Contact: seneca_scott@brown.edu

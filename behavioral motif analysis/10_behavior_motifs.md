# Behavioural motifs from motion energy — build plan

⚑ ACTIVE. Signal-processing stage: turn per-box motion energy into (a) discrete
**movement onsets** and (b) **movement states** (low / medium / high), then analyse
neural activity against both.

Variables of interest: **whisking bouts** (`whisker_pad`) and **locomotion**
(`paw` and `wheel`, analysed both). Motion energy — `mean|ΔF|`, see §1c on why
"motion power" is the same thing — is the currency throughout; mean fluorescence is
never used as a behavioural variable.

Two distinct treatments, per the analysis brief:
- **Whisking** — bout detection. The whiskers move constantly, so the object is a
  *bout against a moving baseline*, not a rest→move transition.
- **Locomotion** — BOTH ways: (a) a **continuous analog** movement signal entering the
  GLM and the state index directly, and (b) **discrete running onsets**. Both `paw` and
  `wheel` are carried, despite being ~91% co-active (§1b), because each cross-validates
  the other and they contaminate differently (§1c).
Companion docs: [05_analysis_plan.md](05_analysis_plan.md) (figures),
[03_synchronization.md](03_synchronization.md) (camera↔2P mapping),
[04_pipeline.md](04_pipeline.md) (bundle spec).

---

## 0. What already exists

`<run>/proc/<run>_boxtraces.npz` gives `motion_energy` — `n_boxes × n_frames`,
float32, camera clock, `mean|Fₜ − Fₜ₋₁|` per box per frame. Nothing downstream
of that is built.

**Camera rate is 15.0000 fps** (66.667 ms/frame, no drift), measured from ThorCam's
per-frame hardware timestamps — see [thorcam_meta.py](thorcam_meta.py). Onsets can
therefore be localised to ±1 frame ≈ **67 ms**. That is comfortably finer than
GCaMP7s kinetics (~100 ms rise, ~0.5–1.5 s decay), so camera resolution is **not**
the limiting factor for onset-triggered neural analysis — the indicator is.

---

## 1. Three findings from real data that drive the design

Measured on `pupil demo 2` (8880 frames, unsmoothed single-frame ME):

| box | p10 | median | p90 | p99 | max |
|---|---|---|---|---|---|
| laser_trigger | 3.7 | 3.8 | 4.9 | 8.5 | 54.0 |
| whisker_pad | 8.0 | 17.0 | 49.2 | 85.0 | 375.0 |
| paw | 5.9 | 7.4 | 71.1 | 347.6 | 648.2 |
| wheel | 6.4 | 7.4 | 26.7 | 231.4 | 498.6 |

**(a) Whisking is not the same kind of signal as paw/wheel.** Paw and wheel are
sparse and bursty — median ≈ p10, so most of the session is genuine rest and bouts
tower over it. Whisking has a **higher rest floor and a much lower contrast ratio**:
robust-z of p99 is whisker_pad **6.4** vs paw **171**, wheel **199**.

> Refined by measurement — see [11_whisking_onset.md](11_whisking_onset.md) §2(1).
> A 2-component mixture on log-ME shows whisking *is* bout-structured (rest 75% of
> the time at ME≈14, active 25% at ME≈43, d′=2.63). It is low-contrast, not
> continuous. An earlier draft of this section called it "quasi-continuous"; that
> was too strong and the onset question is better posed than it implied.

Consequence: **a single shared threshold in z units is meaningless.** The same
"3 SD above baseline" is routine whisking and an extreme paw event. Thresholds must
be **per variable**, chosen from each variable's own distribution. Two
normalisations were compared (MAD over the whole trace vs MAD over rest-only
samples) and neither collapses the variables onto a common scale — this is a real
property of the behaviour, not a normalisation bug.

Consequence for framing: all three are **bout onsets on an envelope** — the
detectable object is a bout, never a single whisk (§1b, and
[11_whisking_onset.md](11_whisking_onset.md) §1). Measured duty cycles are similar
(whisker 28%, paw 16%, wheel 16%), so the earlier idea that whisking is "usually
already running" and needs different framing from paw/wheel did not survive
measurement.

**(b) `laser_trigger` is a free artifact channel — but only as a spike veto.** It
sits on the head-fixation hardware: p10 3.7, median 3.8, p90 4.9 — flat. Use its
rare excursions (0.03% of frames exceed 5× its median) to **veto** coincident
onsets.

> ⚠ Do **not** regress it out as a continuous common mode. Measured in
> [11_whisking_onset.md](11_whisking_onset.md) §2(5): it correlates 0.52 with
> whisker_pad, but that correlation is almost entirely slow (0.09 after removing
> >1 s), and regressing it out degrades rest-vs-whisking separation monotonically
> (d′ 2.60 → 0.90). Detrending each channel is *also* destructive
> ([11_whisking_onset.md](11_whisking_onset.md) §3) — leave the slow mode in place
> and rely on the spike veto alone.

**(c) Wheel and paw are nearly the same measurement.** ~91% of paw-active time is
also wheel-active (§1b). [01_project_overview.md](01_project_overview.md) names paw
as the locomotion proxy for want of a rotary encoder; the wheel box is a second view
of the same events, not an independent variable. Carry ONE locomotion factor —
paw has the better d′ (2.51 vs 2.31) — and label it honestly as a proxy.

---

## 1b. ⚑ FEASIBILITY — the event budget, measured

Run [analysis_probes/motif_feasibility.py](analysis_probes/motif_feasibility.py). It
implements the §2–§3 detector as specified and counts what survives each filter, on the
gold session (9.9 min). **This changes the plan.**

Conditioned with log + 330 ms envelope (no detrend — see
[11_whisking_onset.md](11_whisking_onset.md) §3), all three channels separate well:

| channel | d′ | active | ME rest → active |
|---|---|---|---|
| whisker_pad | **3.56** | 28% | 12.4 → 43.0 (3.5×) |
| paw | 2.51 | 16% | 6.6 → 108.8 (16.5×) |
| wheel | 2.31 | 16% | 7.0 → 33.4 (4.8×) |
| *laser_trigger (control)* | *1.38* | — | — |

Two things to note. **Whisking is the best-separated channel, not the worst** — §1a's
framing was an artefact of using robust-z on raw ME instead of d′ on log-ME. And the
negative control lands at 1.38, below the d′ ≥ 1.5 session gate, which is exactly the
behaviour that gate was invented for.

**The funnel, per 9.9-minute session:**

| channel | bouts | +min length | +quiescence | **+isolated** | clean |
|---|---|---|---|---|---|
| whisker_pad | 141 | 105 | 66 | **5** | 5 |
| paw | 52 | 30 | 17 | **0** | 0 |
| wheel | 44 | 26 | 18 | **0** | 0 |

**Isolation is where the analysis dies.** Zero isolated paw or wheel onsets in ten
minutes; five for whisking. Onset-triggered averaging on isolated events, as §3 specifies,
is not viable on this dataset.

The co-occurrence structure explains it, and contains its own surprise:

| pair | joint | chance | enrichment |
|---|---|---|---|
| whisker & paw | 16.3% | 4.6% | 3.5× |
| whisker & wheel | 15.7% | 4.5% | 3.5× |
| **paw & wheel** | **14.9%** | 2.6% | **5.7×** |

Paw is active 16.4% of frames and paw-and-wheel jointly 14.9% — so **~91% of paw-active
time is also wheel-active.** These are not two variables. They are one locomotion factor
measured twice, which also settles §1c: keep whichever box has the better d′ (paw, 2.51)
or merge them, but do not spend a degree of freedom on both.

### What this forces

1. **Drop the strict isolation requirement.** It selects for events that essentially do
   not occur.
2. **Make the ridge GLM the primary onset analysis, not a companion.** Collinear
   regressors are the exact problem a regularised encoding model solves: fit dF/F on all
   three ME channels plus onset kernels simultaneously and let the model apportion
   variance, instead of hunting for clean events that do not exist.
   [05_analysis_plan.md](05_analysis_plan.md) already specifies this GLM — it is now
   load-bearing rather than optional.
3. **Keep onset-triggered averaging for whisking only, pooled across sessions.**
   5/session × ~10 sessions ≈ 50 events — enough for a PSTH, and whisking has the best d′.
   Report the pooled n on the figure.
4. **Treat paw+wheel as one locomotion factor.**
5. **Re-run this probe on a second session** before freezing anything. One 10-minute
   session is a thin basis for a plan.

## 1c. ⚑ METRIC CHOICE, and what the boxes actually contain

Measured from raw frames on an active 69 s stretch
([analysis_probes/metric_compare.py](analysis_probes/metric_compare.py)).

### "Motion power" vs "motion energy" — same information

We store `mean|ΔF|` (L1, conventionally *motion energy*). *Motion power* normally
means `mean(ΔF²)` (L2). They are not independent measures: Spearman ρ = 0.985–0.991
and L2 ≈ L1^1.7–2.0, i.e. a near-monotone power-law relabel. After the log transform
the conditioning applies, that is an affine map — so d′ is essentially unchanged
(whisker 4.10 vs 3.82, paw 4.28 vs 5.14, wheel 2.86 vs 2.55, no consistent winner),
and RMS = √L2 scores identically to L2, exactly as invariance predicts.

**Conclusion: keep L1. Switching to squared differences buys nothing and would cost a
full re-extraction.** "Motion power" and "motion energy" can be used interchangeably in
conversation; the stored quantity is `mean|ΔF|` either way.

### Motion AREA is a large win for the locomotion channels

A fourth metric — *fraction of pixels whose |ΔF| exceeds a floor* — is dramatically
better where the moving thing is large and low-contrast:

| channel | L1 (current) | L2 | **AREA** |
|---|---|---|---|
| whisker_pad | **4.10** | 3.82 | 3.26 |
| paw | 4.28 | 5.14 | **5.82** |
| wheel | 2.86 | 2.55 | **6.98** |

Wheel separability more than doubles. The physics is intuitive: whiskers are thin, so
few pixels change but by a lot (magnitude wins); a wheel or body surface moves many
pixels by a little, and averaging magnitude over a large box dilutes it (area wins).
**Recommend adding `motion_area` as a third extraction channel** — it is one more
accumulator in the same hot loop, needs an npz key, and therefore needs team sign-off.

### ⚠ The boxes need verifying before any locomotion claim

Rendering the box interiors ([box_contents.png](analysis_probes/box_contents.png)) raises
problems that gate everything in §1b about paw and wheel:

- **`whisker_pad` is 60.6% saturated.** Whiskers are dark on a clipped-white background,
  so whisking is still measured — the frame-difference image is clearly whisker-shaped —
  but the box is running against a blown-out background and a tighter, better-exposed
  box would very likely do better.
- **The `wheel` box's motion is concentrated in hairs along its left edge**, not in
  translation of the textured surface filling the rest of it. Phase-correlation
  displacement on this box returns random directions (consistency 0.08) and impossible
  magnitudes (150 px/frame on a 253 px box) — the signature of a failed registration on
  content with no coherent translation. **There is no evidence this box is measuring
  wheel rotation.**
- **Grooming drives both channels**, not just paw: 71% of top-decile paw frames are also
  top-decile `paw_at_nose` frames (chance 10%), and wheel ME rises 5.3× during grooming
  (paw rises 20×). Paw's larger dynamic range at high amplitude is grooming, not speed —
  which was the real explanation for the falling paw/wheel ratio, not metric saturation.

**Gate: confirm in `trace_viewer` that the animal actually locomotes in these sessions,
and that the `wheel` box sits on a surface that visibly translates.** Until then, treat
"running speed" as unvalidated. If the wheel is not usefully visible, a true continuous
speed signal is not available from this video and the honest fallback is "locomotion-
related movement energy", labelled as such.

## 2. Stage A — conditioning (`me_condition`)

Per run, per variable, in this order:

1. **Restrict to the bright epoch.** Keep frames `[on_frame, off_frame]` from
   `_sync.json`. This removes the laser-onset ME spike outright instead of
   special-casing it. *(Before sync exists, run on the full trace and treat the
   first/last ~2 s as suspect.)*
2. **Artifact veto channel.** Condition `laser_trigger` ME identically; flag frames
   where it exceeds its own high percentile. These frames are excluded from onset
   detection and marked in the state series.
3. **Floor and scale, per variable.** `floor` = 10th percentile;
   `scale` = MAD of samples ≤ median (the rest distribution). Store both — they are
   session/box specific and needed to interpret everything downstream.
4. **Envelope.** Zero-phase moving average (`filtfilt`-style, no group delay — a
   causal filter would bias every onset late by half the window). Two envelopes:
   - `env_fast`, ~200 ms (3 frames) — for onset detection
   - `env_slow`, ~1 s (15 frames) — for state assignment
5. **Normalised units.** Report as `(env − floor) / scale`, but **set thresholds by
   percentile, not by a z value** (see §1a).

Output: `me_cond` (`n_var × n_frames`), plus `floor`, `scale`, `artifact_mask`.

---

## 3. Stage B — onsets (`detect_onsets`)

Per variable, on `env_fast`:

1. **Hysteresis.** Enter movement at `θ_hi`, leave at `θ_lo` (`θ_lo < θ_hi`).
   Two thresholds, not one — a single threshold shatters one bout into many.
2. **Backtrack the onset.** From the `θ_hi` crossing, walk *backwards* to the last
   frame below `θ_lo`. The crossing itself is late by the rise time; the departure
   from baseline is the onset. This is standard EMG-onset practice and matters more
   than the threshold value.
3. **Minimum bout duration** — discard bouts shorter than `min_bout` (default 0.4 s
   ≈ 6 frames).
4. **Merge** bouts separated by less than `merge_gap` (default 0.3 s).
5. **Pre-onset quiescence** — for onset-*triggered* analysis, keep only onsets
   preceded by ≥ `quiet_pre` (default 1.0 s) below `θ_lo`. Without this you average
   mid-bout transitions and the "response" is contaminated by ongoing movement.
6. **Artifact veto** — drop onsets within ±2 frames of an `artifact_mask` frame.
7. **Isolation label** — for each onset, record whether the *other* variables were
   quiescent in a window around it, and record the *degree* of co-activity as a
   continuous covariate. Label, never filter on it: §1b measured 0 isolated paw and
   wheel onsets and 5 whisking onsets per session, so selecting on isolation empties
   the dataset. The co-activity covariate goes into the GLM instead.

**Report the funnel.** Every stage above discards events. Print and store:
`crossings → after min_bout → after merge → after quiescence → after veto →
isolated`. A whisking analysis resting on 4 surviving onsets must be visibly
resting on 4, not silently.

Output per variable: `onsets` (camera frames), `offsets`, `bout_amplitude`,
`bout_duration`, `is_isolated`, plus the funnel counts.

---

## 4. Stage C — movement states (`assign_states`)

On `env_slow`.

**Composite index** = mean of the three per-variable normalised envelopes. Also
retain per-variable states — they cost nothing and answer "is M1 tracking whisking
specifically or arousal generally?".

**⚠ Do NOT use within-session tertiles.** This is the trap in this stage. If low /
med / high are defined as within-session tertiles, then every session spends
exactly 33% of its time in each state *by construction* — and
[05_analysis_plan.md](05_analysis_plan.md) Fig 5 compares arousal between PB and TMT
sessions. Within-session tertiles would make that comparison structurally
impossible while still producing a plot.

**Instead:** pool `env_slow` across all sessions (per mouse), take the pooled
33rd/67th percentiles once, freeze them, and apply the same two numbers to every
session. State occupancy then varies by session and is comparable, which is exactly
what Fig 5 needs.

Then, as with onsets: **hysteresis** on state boundaries plus a **minimum dwell**
(default 1.0 s) so the state series does not flicker frame-to-frame. Frames flagged
in `artifact_mask` are labelled `unknown`, not forced into a state.

Output: `state` (0/1/2/−1 per frame), `state_bouts` (start, end, level),
`occupancy` per state, and the frozen thresholds.

---

## 5. Stage D — neural analysis

Needs sync + bundles (§7). Two analyses, matching the two behavioural products.

**Onset-triggered.** Window −2 to +4 s around each isolated onset, per neuron.
Baseline = mean over −2 to −0.5 s. Report per-neuron mean response, peak latency,
sign; population PSTH heatmap sorted by latency; fraction responsive. Null =
circular shift of the onset times (≥30 s offsets, 500×), which preserves both the
calcium autocorrelation and the event count.

Latency claims must respect GCaMP7s: a ~100 ms rise and ~1 s decay mean you can
report *ordering* ("M1 leads/lags whisking onset") but not precise latency. Say so.

**State-based.** Per neuron: mean dF/F and event rate in low / med / high.
Population: mean pairwise correlation and dimensionality per state.

**The statistical unit is the bout, not the frame.** dF/F at 10 Hz is heavily
autocorrelated; treating 6000 frames as 6000 independent samples inflates every
test by roughly an order of magnitude. Average within each state bout first, then
test across bouts — or block-bootstrap in contiguous chunks. This is the single
easiest way to produce a wrong p-value here.

---

## 6. Parameters to freeze (one `motif_params.json`)

| param | default | note |
|---|---|---|
| `env_fast_s` | 0.2 | onset envelope, zero-phase |
| `env_slow_s` | 1.0 | state envelope |
| `floor_pct` | 10 | per-variable rest floor |
| `theta_hi_pct` / `theta_lo_pct` | 90 / 70 | **per variable**, from pooled data |
| `min_bout_s` | 0.4 | ≈6 frames at 15 fps |
| `merge_gap_s` | 0.3 | |
| `quiet_pre_s` | 1.0 | pre-onset quiescence for triggered analysis |
| `isolation_window_s` | 1.0 | other variables quiet within ± this |
| `artifact_pct` | 99.5 | laser_trigger veto percentile |
| `state_pct` | 33 / 67 | **pooled across sessions**, then frozen |
| `min_dwell_s` | 1.0 | state hysteresis |

Tune on the gold session, freeze, then batch — same discipline as `ca_extract`.

---

## 7. Build order and dependencies

**Phase 1 — behaviour only. No dependency on sync, bundles, or the calcium
pipeline.** Stages A–C run on `_boxtraces.npz` alone, in camera frames. This can be
built and validated immediately and in parallel with everything else, which makes
it the right thing to start now.

**Phase 2 — neural.** Stage D needs: validated sync (`_sync.json` on real data),
`pair_runs.py`/`sync_all.py`, `ca_extract` run on real sessions, and
`make_bundle.py`. All still open.

Keep Phase 1 outputs in **camera frames**. Do no time math here — conversion to 2P
time is the sync stage's job, and duplicating it is how the two drift apart.

---

## 8. Validation

1. **Synthetic ground truth.** Generate ME traces with planted bouts at known
   frames (varying amplitude, duration, SNR, plus a rigid-shift artifact).
   Assert recovered onsets are within ±1 frame and quantify systematic latency bias
   — backtracking should make it near zero; report it rather than assume it.
2. **Eyeball against the movie.** The real gate. Jump `trace_viewer` to each
   detected onset and confirm the animal actually moved. Worth adding an
   `--onsets <npz>` flag that marks onsets on the trace panels and lets `n`/`p` step
   between them; that turns validation from a chore into a minute of clicking.
3. **Funnel counts** printed per run (§3) — catches "detection worked but nothing
   survived filtering".
4. **Cross-variable sanity.** Wheel and paw onsets should often coincide;
   whisking often precedes locomotion. If whisking and wheel onsets are
   *simultaneous* to the frame, suspect a rigid shift driving both.
5. **QC PNG per run**: three ME envelopes, thresholds, detected bouts shaded,
   onsets ticked, state bands along the bottom, artifact frames hatched.

---

## 9. Output contract

Per run, in the existing `proc/` folder:

```
<run>/proc/
    <run>_motifs.npz       me_cond, env_fast, env_slow, floor, scale,
                           onsets/offsets/amplitude/duration/is_isolated per var,
                           state, state_bouts, artifact_mask, funnel counts
    <run>_motifs.json      params used + params hash + funnel summary (readable)
    <run>_motifsqc.png     the eyeball gate
```

Same engineering contract as the rest of the pipeline: idempotent (skip if present,
`--force`), atomic tmp+rename writes, per-run failure isolation, runs-file +
`--run-index` for SLURM, batch summary CSV. Port these from
`box_extract_cluster.py` rather than rewriting them.

Everything is **derived from** `motion_energy` and writes new files — no existing
npz key changes, so this does not touch the frozen extraction schema.

---

## 10. Open questions

1. **Wheel vs paw as the locomotion channel** (§1c) — decide on the gold session.
2. **States from the composite, or per variable?** Plan computes both; which one
   carries Fig 5 is a scientific call.
3. **Is `nose` worth adding** as a fourth motif channel, or does it duplicate
   `whisker_pad`? Cheap to check by correlation before committing.
4. **Sniff** is not in this plan — it arrives on the 2P clock and folds in at the
   bundle stage as another arousal channel.

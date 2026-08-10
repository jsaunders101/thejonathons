# Whisking onset detection — deep design

Focused companion to [10_behavior_motifs.md](10_behavior_motifs.md). Everything here
is measured on `whisker_pad` motion energy from `pupil demo 2` (8880 frames, 15.0000 fps,
10.25 min). Reproduce with [analysis_probes/whisk_probe.py](analysis_probes/whisk_probe.py);
figure via [analysis_probes/whisk_fig.py](analysis_probes/whisk_fig.py) →
[whisk_diagnostics.png](analysis_probes/whisk_diagnostics.png). Numbers, not intuition:
two of the design choices below are the *opposite* of what seemed obvious.

---

## 1. What "onset" can honestly mean here

**Whisk cycles are unrecoverable.** Mouse exploratory whisking is 8–12 Hz. The camera
runs at 15 fps, Nyquist 7.5 Hz. Whisking is *above* Nyquist, so individual protractions
are aliased, not sampled. The measured spectrum confirms it: pure 1/f, **no peak
anywhere in 0–7.5 Hz**, 58.7% of power below 0.5 Hz. There is nothing rhythmic to find.

So the only detectable object is the **whisking bout envelope**. Every use of "onset"
below means *the start of a whisking bout* — the moment envelope amplitude steps up from
its ongoing baseline — never "the first protraction", and never a departure from
stillness, because the whiskers never stop. Write it that way in the talk; a reviewer
who knows whisker physiology will ask.

Three consequences:
- No whisk frequency, no phase, no amplitude-per-cycle. Do not attempt them.
- Onset precision is bounded at **±1 frame (67 ms)** by sampling, and realistically
  ~100 ms once envelope smoothing and the physiological ramp (bouts build over 1–3
  cycles) are included. That is still well inside GCaMP7s kinetics (~100 ms rise,
  0.5–1.5 s decay), so **the indicator, not the camera, limits latency claims.**
- Low-amplitude whisking is the baseline, by definition invisible to a bout detector.
  The detector finds bouts, not all whisking. Report it as a sensitivity floor, not a bug.

*If whisker kinematics ever matter for a future experiment, that needs ≥100 fps —
a camera decision, not an analysis one.*

---

## 2. Six measured properties of the signal

**(1) Bout structure is real and separable.** A 2-component Gaussian mixture on
log-ME splits cleanly:

| component | weight | mean ME | interpretation |
|---|---|---|---|
| low | 0.75 | ≈ 13.7 | low-amplitude ongoing whisking (NOT rest — see below) |
| high | 0.25 | ≈ 43.1 | whisking bouts |

**d′ = 2.63.** That is workable separation with real overlap.

> **Label correction (per Seneca, who ran the rig): the low component is not rest.**
> The mouse moves its whiskers continuously; the two components are *low-amplitude
> ongoing whisking* (75%) and *whisking bouts* (25%), not rest vs movement. The
> statistics are unchanged — the mixture, the weights, d′ — but the interpretation is,
> and so is the detector: there is no quiescent baseline to return to, so
> "pre-onset quiescence" must mean *below the bout threshold*, never *still*.
> The target is a **bout onset against a moving background**, which is why the
> two-component model is the right tool rather than a threshold above zero.

**(2) The envelope timescale is ~600 ms.** Autocorrelation of log-ME falls below 0.5 at
lag 9 frames. This brackets the smoothing window: much below ~200 ms leaves aliasing
flicker, much above ~400 ms starts erasing real bouts.

**(3) Raw thresholding shatters.** Thresholding unsmoothed ME at the GMM midpoint gives
587 suprathreshold runs, **56% of them a single frame long**, median gap 267 ms. Any
detector without temporal regularisation will report hundreds of spurious onsets.

**(4) Grooming masquerades as whisking.** `corr(log whisker_pad, log paw_at_nose) = 0.66`,
and during the top 1% of `paw_at_nose` frames whisker_pad ME is **4.2× its normal
median**. The paw enters the whisker field during face grooming. Untreated, grooming
bouts will be the largest "whisking onsets" in the dataset.

**(5) ⚠ Do NOT regress out the common mode.** `corr(log whisker_pad, log laser_trigger)
= 0.52`, which looks like heavy artifact contamination and argues for removing it.
Tested directly, and it is wrong: d′ degrades **monotonically** with removal strength —
2.60 at β=0, 1.67 at β=0.5, 1.35 at β=1, 0.90 at the least-squares β=1.9. Regression
removes real signal.

The resolution is in the timescales. That 0.52 is almost entirely **slow**:

| | r |
|---|---|
| raw | +0.518 |
| after removing >1 s | **+0.093** |
| after removing >3 s | +0.143 |
| after removing >10 s | +0.337 |

At the sub-second timescales onsets live on, the two boxes are nearly independent. The
shared variance is slow illumination/postural drift plus genuine whole-animal movement
that legitimately shows up in both — which is why removing it takes real signal with it.
An earlier draft concluded "detrend, don't regress"; §3 S1 shows detrending is
destructive too. **Leave the slow mode in place; the spike veto of (6) is the only
artifact handling that survives measurement.**

**(6) Fast artifacts are rare and cheap to veto.** Only 0.03% of frames exceed 5× the
laser_trigger median (0.023% exceed 10×). Vetoing them costs essentially no data, so
`laser_trigger`'s role is a **discrete spike veto**, not a continuous regressor.

---

## 3. The detector

### S1 — condition
1. Restrict to the bright epoch (sync anchors), or trim the first/last ~2 s until sync exists.
2. `y = log(ME + ε)` — stabilises variance and makes the mixture Gaussian.
3. **Zero-phase envelope, ~330 ms.** Zero-phase because a causal filter delays every
   onset by half its window.

> ⚠ **Do NOT detrend.** An earlier version of this section prescribed subtracting a
> 30 s rolling median to remove the slow common mode of §2(5). Measured, that is
> destructive — d′ collapses at every window tried:
>
> | conditioning | whisker_pad | paw | wheel |
> |---|---|---|---|
> | raw log | 2.63 | 2.38 | 2.06 |
> | + envelope 200 ms | 3.33 | 2.52 | 2.23 |
> | + envelope 330 ms | **3.56** | **2.51** | **2.31** |
> | + envelope 500 ms | 3.70 | 2.59 | 2.34 |
> | detrend 30 s + env | 0.82 | 0.89 | 0.74 |
> | detrend 120 s + env | 2.40 | 1.69 | 1.63 |
>
> The cause: at a ~25% duty cycle a rolling median does not estimate rest — it tracks
> local activity, rising inside busy stretches and falling inside quiet ones, which is
> exactly the contrast being measured. The inflated active weight gives it away
> (0.25 → 0.54 at 30 s). Envelope smoothing, by contrast, improves separation
> monotonically and plateaus near 500 ms; 330 ms is the chosen compromise, trading
> 0.14 of d′ for sharper onset timing. **The slow common mode of §2(5) is therefore
> left in place and handled by the spike veto alone.**

### S2 — self-calibrating two-state model
Fit a **2-state Gaussian HMM** on the conditioned trace: states {rest, whisking},
Gaussian emissions, transition matrix parameterised by expected dwell times (~0.3 s rest,
~0.5 s whisking as priors). Initialise emissions from the 2-component GMM (§2(1)), which
converges reliably. Decode with Viterbi.

Why an HMM rather than threshold + hysteresis + min-duration:
- **Consistency across sessions** — the user requirement. It self-calibrates per session,
  so illumination, box placement and animal position changes don't need a re-tuned
  absolute threshold. There is no magic number to transfer.
- Dwell priors dissolve the 56%-single-frame flicker (§2(3)) *by construction*, instead of
  three ad-hoc post-hoc rules that each need tuning.
- Less pre-smoothing is required, so onset timing is sharper.
- It emits a per-frame posterior, giving a free confidence measure per onset.

Caveat to respect: HMM emissions assume conditional independence given state, and a
smoothed envelope is autocorrelated. That is another reason to keep S1's smoothing light.
Keep the plain threshold+hysteresis detector implemented as a **cross-check**, not as the
primary — agreement between two independent methods is a validation signal (§6).

### S3 — refine the onset
Viterbi gives the transition frame; the true departure from baseline is slightly earlier.
Within a window before each transition, **backtrack to the last frame below the rest
mean**, or run a local CUSUM change-point. Report the median refinement offset — if
backtracking is systematically moving onsets by >2 frames, S1 smoothing is too heavy.

### S4 — label, don't silently drop
Per candidate onset, attach:
- `groom` — `paw_at_nose` above its high percentile within ±0.5 s (§2(4))
- `artifact` — `laser_trigger` spike within ±2 frames (§2(6))
- `isolated` — `paw` and `wheel` both quiescent in the surrounding window
- `quiet_pre` — ≥1 s of rest before onset (required for triggered averaging)
- `posterior` — HMM confidence

Analyses select on these; nothing is deleted, so a later question ("what did grooming
do to M1?") is still answerable.

### S5 — quality gates, per session
- **d′ between the fitted components.** d′ < 1.5 ⇒ the two states are not separable in
  that session; flag it and do not report whisking onsets from it. (Gold session: 2.63.)
- Rest-state weight far from ~0.5–0.9 ⇒ suspect box placement or saturation.
- Onset funnel counts at every stage (§10 in doc 10).
- Detector run on `laser_trigger` as a **negative control** — a detector that finds
  whisking bouts on the head-fixation bar is broken.

---

## 4. Parameters

| param | default | grounded in |
|---|---|---|
| `log_eps` | 1e-3 | — |
| `detrend_s` | 30 | §2(5): shared variance is >1 s; 30 s keeps bouts, kills drift |
| `env_s` | 0.13–0.2 | §2(2): 600 ms autocorr; light because HMM regularises |
| `dwell_rest_s` / `dwell_active_s` | 0.3 / 0.5 | §2(3) run/gap medians |
| `groom_pct` | 99 | §2(4) |
| `artifact_pct` | 99.5 | §2(6): vetoes ~0.03% of frames |
| `quiet_pre_s` | 1.0 | clean pre-onset baseline for triggered averages |
| `isolation_window_s` | 1.0 | |
| `dprime_min` | 1.5 | §3 S5 session gate |

Tune on the gold session, freeze into `motif_params.json`, then batch.

---

## 5. Onsets and states share one conditioning stage

This section previously described two conditioning branches — detrended for onsets,
undetrended for states. With detrending removed (§3 S1) that split is gone: **onsets and
states run off the same conditioned envelope.** Simpler, and one less way for the two
analyses to disagree.

The concern that motivated the split still stands, though: slow amplitude variation is
what low/medium/high state *is*, so it must not be filtered away — and it is now
genuinely at risk from illumination drift instead. Carry the slow `laser_trigger` trend
alongside as a covariate and confirm that apparent state changes across a session are
not tracking it.

---

## 6. Validation ladder

1. **Synthetic ground truth** — plant bouts of known onset, amplitude, duration in a
   realistic noise floor, plus a rigid-shift artifact and a grooming episode. Assert
   recovered onsets within ±1 frame; report systematic bias explicitly.
2. **Human labels** — scrub `trace_viewer` through one session and hand-mark ~30–50
   whisking bout onsets blind to the detector. This is the only real ground truth.
   Compare: hit rate, false-alarm rate, median timing error. Budget an hour.
3. **Cross-method agreement** — HMM vs threshold+hysteresis. Onsets agreeing within
   ±2 frames are high-confidence; systematic disagreement localises the problem.
4. **Negative control** — the detector on `laser_trigger` should find ~nothing.
5. **Split-half stability** — fit the HMM on the first half, decode the second, compare
   to a full fit. Large differences mean the session is non-stationary.

---

## 7. Two cheap wins before writing any of this

**Redraw `whisker_pad` tighter.** In the gold frame the box covers a broad patch of face.
A tighter box centred on the whisker field, deliberately excluding the snout and the
path the paw takes during grooming, would raise d′ and directly attack the largest
contaminant (§2(4)) — for the cost of one redraw plus `--force` re-extract.

**Consider a second whisker box.** Two boxes on the same whisker field give an
agreement test: real whisking appears in both, a rigid shift appears in both *plus*
`laser_trigger`, and local noise appears in one. This is the cleanest artifact
discriminator available without new hardware. It adds a `CANONICAL_BOXES` entry, so it
needs team sign-off before anyone redraws.

---

## 8. Failure modes, and how each announces itself

| failure | signature |
|---|---|
| box includes the grooming path | huge onsets coincident with `paw_at_nose`; `groom` flag fires constantly |
| box saturated (clipped white) | ME compressed, d′ collapses, rest weight → 1 |
| illumination drift | d′ fine but state occupancy drifts monotonically through the session |
| rigid shift / animal settling | simultaneous onsets in whisker + paw + wheel + laser_trigger |
| too much smoothing | backtracking offsets grow >2 frames; short bouts vanish |
| session non-stationary | split-half fits disagree; Viterbi flips state mid-session |
| genuinely quiet animal | rest weight >0.95, few onsets — real, not a bug; report n |

---

## 9. Open questions

1. **Whisking vs grooming as separate motifs?** Grooming is currently a contaminant to
   veto, but "pawing at nose" is already a canonical category and a real behaviour. It
   may deserve to be its own motif rather than only a mask.
2. **Does a tighter whisker box actually raise d′?** Testable in ten minutes on the gold
   session once redrawn — do it before committing to detector parameters.
3. **Is 25% whisking duty typical for these animals**, or is `pupil demo 2` unusually
   quiet or busy? Needs a second session before the priors in §4 are trusted.

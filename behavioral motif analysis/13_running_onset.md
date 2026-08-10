# Running onset extraction — build plan

Companion to [11_whisking_onset.md](11_whisking_onset.md), which is built and working
([whisk_bouts.py](whisk_bouts.py)). Same machinery, different target: **onsets of
running/walking** from `paw` and `wheel` motion energy, plus a **continuous analog**
locomotion signal.

> ## ⚑ STATUS: BLOCKED — on data, not code
>
> The detector is straightforward (§3). What is missing is any evidence that
> locomotion occurs in, or is visible in, the data we have. Measured on the gold
> session (`pupil demo 2`, 9.9 min):
>
> | question | measurement |
> |---|---|
> | frames with paw AND wheel active, not grooming | **1.5%** (130 frames) |
> | such episodes lasting ≥ 1 s | **0** |
> | such episodes lasting ≥ 2 s | **0** |
> | longest such episode | **0.80 s** |
> | paw-active frames that are grooming | **85%** (1237/1452) |
>
> A running mouse produces sustained bouts of seconds. There are none here. The
> `paw` channel in this session is, in effect, a grooming detector.
>
> Combined with [10_behavior_motifs.md](10_behavior_motifs.md) §1c — the `wheel`
> box's motion sits in hairs along its left edge, and phase correlation on it returns
> random directions — there are two live possibilities and they need different fixes:
>
> 1. **The animal did not locomote** in this session. Then the code is fine and we
>    need a session where it did.
> 2. **The `wheel` box does not see the wheel** (or there is no wheel to see). Then
>    no analysis recovers running from this video, and the honest deliverable is
>    "movement energy", never "speed".
>
> **Step 0, before any code: open `trace_viewer` on a session and watch. Does the
> animal run? Does anything in the `wheel` box translate?** Ten minutes settles it.

---

## 1. How running differs from whisking

**Easier in one way.** Whiskers never stop, so whisking bouts had to be detected as a
step up from a *moving* baseline, and "pre-onset quiescence" had to be redefined as
"below the bout threshold". Locomotion has a genuine rest state — the mouse is either
moving or it is not — so the two-state model maps onto it directly and a real
quiescence requirement is meaningful.

**Harder in another: specificity.** The whisking confound was contamination *of* the
box (the paw entering the whisker field), handled with a veto. The running confound is
worse — the paw and wheel boxes respond strongly to movements that are not locomotion:

| | paw ME | wheel ME |
|---|---|---|
| outside grooming | 7.1 | 7.3 |
| during grooming | 144.7 (**20×**) | 38.4 (**5.3×**) |

Both channels light up during grooming. Detecting "paw moved" is easy; detecting
"the animal was *running*" is the actual problem.

---

## 2. Specificity — one idea tested and dead, three left

**✗ Tested and rejected: the paw/wheel amplitude ratio.** Grooming raises paw 20× and
wheel 5.3×, so `log(paw) − log(wheel)` looked like it should separate grooming from
running. Measured over paw-active frames: grooming mean +0.85, non-grooming +0.72,
**d′ = 0.20, AUC = 0.552.** Useless. Recorded so nobody re-derives it.

Remaining candidates, best first:

**(a) Onset concordance between paw and wheel.** Detect independently on each channel,
then call a running onset only where both fire within ±N frames. Grooming may raise
both *levels* without producing coincident *onsets*. This is the same cross-check logic
that validated the whisking detector, and it is cheap. **Untested** — it needs a
session with running to test against.

**(b) `paw_at_nose` veto.** Already implemented in the whisking detector. Given 85% of
paw activity here is grooming, this will do most of the work, and it is the one
mechanism already proven to fire correctly (synthetic tests in
[test_whisk_bouts.py](test_whisk_bouts.py)).

**(c) Sustained-duration prior.** Running bouts last seconds; the grooming-driven paw
excursions in this session are mostly sub-second. A minimum bout duration of ~1 s
(versus 0.4 s for whisking) would suppress much of it. Weak on its own — grooming bouts
can also be long — but free, since it is one parameter.

**A note on (a):** if the wheel box turns out not to see the wheel, concordance
collapses to "paw agrees with a broken channel" and the whole specificity strategy has
to be rebuilt around something else. Hence Step 0.

---

## 3. The detector

Structurally identical to [whisk_bouts.py](whisk_bouts.py), which is the point — that
machinery is tested (hit rate 0.982 on synthetics, half-rise +1 frame on real data) and
should not be rewritten:

1. **Condition** — log + ~330 ms zero-phase envelope, no detrend. Identical.
   Measured d′: paw **2.54**, wheel **2.33**, active weight 0.22 / 0.21. Both clear the
   1.5 session gate on their own.
2. **Two-state HMM per channel** — GMM init, dwell priors, Viterbi. Identical, except
   `min_bout_s` ≈ 1.0 (§2c) and longer dwell priors, since running bouts are slower
   events than whisking bouts.
3. **Concordance** — the new step. Pair paw and wheel onsets within a tolerance; the
   running onset is the earlier of the pair. Keep the unpaired ones labelled, they are
   the interesting failures.
4. **Label** — grooming (`paw_at_nose`), artifact (`laser_trigger`), whisking
   co-activity, pre-onset quiescence, HMM posterior. Same as whisking. Label, never
   filter.
5. **Continuous signal** — the second deliverable. Export the conditioned envelope of
   both channels plus their mean as a continuous locomotion regressor for the GLM and
   the state index. **Label it "locomotion movement energy", not speed:** motion energy
   is monotone-ish with speed but uncalibrated, and phase-correlation displacement —
   the one metric that would be a true speed — failed on this box.
6. **Consider `motion_area`.** [10_behavior_motifs.md](10_behavior_motifs.md) §1c
   measured that fraction-of-pixels-changing raises wheel d′ from 2.86 to **6.98** and
   paw from 4.28 to 5.82. For the locomotion channels specifically this is the single
   biggest available improvement — but it is a new extraction channel and an npz key,
   so it needs team sign-off and a re-extract.

### Expected yield

From the funnel already measured with the corrected conditioning:

| channel | bouts | + min length | + quiescence |
|---|---|---|---|
| paw | 52 | 30 | 17 |
| wheel | 44 | 26 | 18 |

~17–18 candidate onsets per session before concordance and the grooming veto. Across
~10 sessions that is a workable budget for onset-triggered analysis — **far better than
whisking's isolated-onset count** — provided the events are real locomotion, which is
exactly what Step 0 decides.

---

## 4. Code structure

`whisk_bouts.py` already contains every primitive: `condition`, `fit_gmm2`, `dprime`,
`transition_matrix`, `viterbi`, `forward_backward`, `runs_of`, `merge_and_filter`,
`refine_onset`, `hysteresis_detect`, plus the QC and `inspect` plots.

Proposed: extract those into **`bout_core.py`**, leaving `whisk_bouts.py` as a thin
whisking-specific wrapper and adding `run_bouts.py` for locomotion. The existing test
suite is the safety net for the refactor — it must pass unchanged before and after.

The alternative (having `run_bouts.py` import from `whisk_bouts`) avoids the refactor
but leaves a module named for whisking as the home of shared code, which will confuse
the next person. Worth the small churn.

---

## 5. Validation ladder

Same shape as whisking, with one addition that matters more here:

1. **Synthetic ground truth** — planted running bouts, including a grooming episode
   that must NOT be detected as running. That negative case is the whole point.
2. **Human labels** — scrub the movie and mark running bouts by eye. For running this
   is easier than for whisking (locomotion is unmistakable at 15 fps) and it is the
   only way to measure specificity.
3. **Concordance rate** — what fraction of paw onsets have a wheel partner, and vice
   versa. A low rate means the two boxes are not seeing the same event.
4. **Grooming cross-tab** — of detected running onsets, how many are grooming-flagged?
   Should be near zero after the veto. If it is not, specificity has failed.
5. **`inspect`-style figure** — reuse the whisking one directly: onset-triggered
   average of raw ME (independent of the envelope), plus individual examples.

---

## 6. Open questions

1. **Is there a running wheel, and does the camera see it?** (Step 0.) Everything else
   depends on this.
2. **Does any session contain sustained locomotion?** The gold session does not. Check
   the others before assuming the analysis is possible at all.
3. **If locomotion is absent or invisible** — is the honest fallback to drop running
   from the analysis and present whisking + grooming as the two behavioural motifs?
   That is a scientifically clean story and is supported by the data we actually have.
4. **`motion_area` re-extraction** — worth it for wheel (d′ 2.86 → 6.98), needs
   coordination.

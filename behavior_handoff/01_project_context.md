# Project context

CSHL ISFNS 2026 final project. Four PhD students/postdocs, rig built in one day, two days of
data, one analysis day. Deliverable: 30–40 min talk + slides; **poster-level** quality bar.

## Question
How is moment-to-moment arousal state related to ongoing (spontaneous) activity in mouse M1?

Arousal operationalized from four channels, in order of expected reliability:
1. **Sniffing** — thermistor on NI DAQ (cleanest analog signal; not in this package)
2. **Whisking** — motion energy in a whisker-pad box of the IR video
3. **Locomotion proxy** — paw-box motion energy (no rotary encoder on the rig)
4. **Eye/pupil** — stretch goal; NB a plain intensity box on the eye already carries usable
   lid/pupil signal (~25–30% trace modulation in our test data)

## Design
- 2 awake headfixed mice, GCaMP7s in M1 excitatory neurons, spontaneous activity (no task)
- 2P: Bruker/PrairieView, single plane, 30–50 Hz varying per recording, sessions 5–20 min
- Behavior: ThorCam IR video (face, whiskers, paws, nose, wheel), .tif stacks
- **Odor context**: peanut butter (appetitive) or TMT (aversive) on a cotton ball placed
  BEFORE recording — an ambient per-session arousal manipulation. Analyze at session level;
  there are no odor events to align to.

## Analysis endpoints (for orientation)
FOV + example traces; per-neuron correlation of dF/F with behavior variables; ridge GLM per
neuron (blocked CV); CCA between population activity and behavior variables; session-level
PB-vs-TMT comparison. The named behavior boxes exist precisely so these analyses get
biologically interpretable regressors ("whisker pad", "paw") rather than anonymous motion
dimensions.

## Scope guardrails (course-wide)
No model training (no DeepLabCut), no precise pupillometry, no publication-grade stats.
n=2 mice — everything descriptive. Quick, tested, interpretable.

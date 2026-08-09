# Scope boundary & what comes next

## Deliberately NOT in this package
These exist on Seneca's side and are still being refined — **do not reimplement them**;
duplicated sync logic with subtle differences is exactly the kind of bug this project
cannot afford on a one-day timeline.

1. **sync_detect.py** — aligns the ThorCam video to the 2P t-series via the laser-onset
   brightness step in the `laser_trigger` box trace (protocol: camera starts before the
   t-series and stops after it, so each video has one bright epoch). Validated on synthetic
   ground truth; awaiting validation against real PrairieView output.
2. **pair_runs.py** (planned) — timestamp-based pairing of behavior runs to Bruker
   `TSeries-*` folders → a `sessions.csv` manifest (single source of truth for mouse/day/
   odor labels and downstream stages).
3. **sync_all.py** (planned) — manifest-driven batch sync + summary CSV + QC contact sheet.
4. **Sniff processing and the session bundle builder** (dF/F + behavior traces + sniff on a
   common 10 Hz timebase → per-session .mat for MATLAB analysis).

## What this means for behavior-side work
- Everything you extract now is on the **camera clock** (frame indices). Alignment to 2P
  time happens later via `(cam_frame - on_frame) / effective_cam_fps` using anchors the sync
  stage will produce. Don't do your own time math; keep outputs in camera frames.
- **Protect the laser_trigger box**: drawn on a non-saturated region, present in every run.
  It is the sync stage's only input from your side.
- Odor condition, mouse, day: keep them as per-run metadata notes for now; they'll live in
  `sessions.csv` once pairing exists.

## Coordination
Owner: Seneca (seneca_scott@brown.edu). Coordinate before: changing canonical category
names, changing npz keys/shapes, or anything touching sync. Everything else in the behavior
pipeline — new QC ideas, additional derived measures computed FROM the box traces,
viewer improvements — is fair game; keep the regression suite passing and outputs in the
proc/ contract.

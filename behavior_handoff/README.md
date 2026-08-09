# CSHL M1-Arousal — behavior extraction handoff

Package for collaborators joining the behavior-video analysis. Contents:

| Path | What |
|---|---|
| [SYSTEM_PROMPT.md](SYSTEM_PROMPT.md) | **Start here** — paste into your Claude session (or CLAUDE.md) to initialize it with full project context and conventions |
| [01_project_context.md](01_project_context.md) | Science: question, design, hypotheses, scope guardrails |
| [02_data_and_formats.md](02_data_and_formats.md) | Acquisition specs, the run/proc organization contract, npz schema |
| [03_behavior_pipeline.md](03_behavior_pipeline.md) | Tool usage, GUI behavior, cluster patterns, validation status |
| [04_scope_and_next.md](04_scope_and_next.md) | What is deliberately NOT in this package, and coordination points |
| `code/box_extract_*.py` | Box-ROI drawing GUI + streaming trace extraction |
| `code/run_behavior_*.py` | One-command flow: pick folder → draw → extract → review |
| `code/trace_viewer_*.py` | QC viewer: movie + traces side-by-side playback |
| `code/test_*_*.py` | Regression suites (GUI logic, run_behavior chain, extraction equivalence) |

## Two versions — pick by where you run
Every script ships twice, and the **suffix is the whole difference**:

| you are on | use | trace-write guard |
|---|---|---|
| the cluster (Elzar) — the real data | `*_cluster.py` | ON: refuses to write traces anywhere else |
| your own machine — local rig, demo data, testing | `*_local.py` | OFF: just runs |

The code is otherwise identical. Mixing them is harmless (they read and write the
same `proc/` files) — the guard is the only behavioral difference. If a
`*_cluster.py` script exits with a `CLUSTER-ONLY` message, you wanted the
`*_local.py` twin. Below, `_cluster` is written out; swap in `_local` off-cluster.

## Quick start
1. Env: Python ≥3.9 + `numpy tifffile matplotlib` (`scipy` optional, enables .mat export).
2. Sanity check: `MPLBACKEND=Agg python code/test_box_gui_cluster.py` → "all 9 checks passed".
3. Draw boxes on your data: `python code/box_extract_cluster.py draw /path/to/behavior`
   (drag rectangle → click category button; zoom with the toolbar magnifier first).
4. Extract: `python code/box_extract_cluster.py extract /path/to/behavior`
   (streams in bounded blocks — `--batch-mb`, default 256 MB — so 80+ GB movies
   don't exhaust RAM).
5. Verify: `python code/trace_viewer_cluster.py /path/to/behavior/<run>`

## Not included, on purpose
2P↔camera sync (`sync_detect.py`), run↔t-series pairing, sniff, bundles — still being
refined on Seneca's side. See [04_scope_and_next.md](04_scope_and_next.md) before touching
anything sync-adjacent. Contact: seneca_scott@brown.edu

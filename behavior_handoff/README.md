# CSHL M1-Arousal — behavior extraction handoff

Package for collaborators joining the behavior-video analysis. Contents:

| Path | What |
|---|---|
| [SYSTEM_PROMPT.md](SYSTEM_PROMPT.md) | **Start here** — paste into your Claude session (or CLAUDE.md) to initialize it with full project context and conventions |
| [01_project_context.md](01_project_context.md) | Science: question, design, hypotheses, scope guardrails |
| [02_data_and_formats.md](02_data_and_formats.md) | Acquisition specs, the run/proc organization contract, npz schema |
| [03_behavior_pipeline.md](03_behavior_pipeline.md) | Tool usage, GUI behavior, cluster patterns, validation status |
| [04_scope_and_next.md](04_scope_and_next.md) | What is deliberately NOT in this package, and coordination points |
| `code/box_extract.py` | Box-ROI drawing GUI + streaming trace extraction |
| `code/trace_viewer.py` | QC viewer: movie + traces side-by-side playback |
| `code/test_box_gui.py` | GUI regression suite (run: `MPLBACKEND=Agg python test_box_gui.py`) |

## Quick start
1. Env: Python ≥3.9 + `numpy tifffile matplotlib` (`scipy` optional, enables .mat export).
2. Sanity check: `MPLBACKEND=Agg python code/test_box_gui.py` → "all 9 checks passed".
3. Draw boxes on your data: `python code/box_extract.py draw /path/to/behavior`
   (drag rectangle → click category button; zoom with the toolbar magnifier first).
4. Extract: `python code/box_extract.py extract /path/to/behavior`
5. Verify: `python code/trace_viewer.py /path/to/behavior/<run>`

## Not included, on purpose
2P↔camera sync (`sync_detect.py`), run↔t-series pairing, sniff, bundles — still being
refined on Seneca's side. See [04_scope_and_next.md](04_scope_and_next.md) before touching
anything sync-adjacent. Contact: seneca_scott@brown.edu

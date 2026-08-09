# Data & formats

## Acquisition
| Modality | Hardware | Format | Notes |
|---|---|---|---|
| Behavior video | ThorCam, IR | .tif stacks (often split into ~1 GB parts, `..._0.tif, _1, _2`) | mono uint16 in our data; RGB tolerated |
| 2P Ca2+ | Bruker/PrairieView | t-series (not in this package) | 30–50 Hz, varies per recording |
| Sniff | thermistor → NI DAQ | PrairieView voltage recording (not in this package) | |

## The run / proc contract (inviolable)
A **run** = any folder directly containing ≥1 `.tif`. Multiple tifs in a run are ONE movie,
concatenated in natural-sort order. All derived data lives in-place:

```
<run>/                      raw tifs (never written to)
<run>/proc/
    boxes.json              box coords + categories (draw phase)
    <run>_boxtraces.npz     extraction output (schema below)
    <run>_boxtraces.mat     MATLAB copy (written by default)
    <run>_sync.json         [made by sync tooling — NOT in this package]
    <run>_syncqc.png        [same]
```

Run discovery ignores macOS junk: `._*` AppleDouble sidecars, dotfiles/dot-dirs, zero-byte
tifs. Filenames with spaces are sanitized to underscores in output stems.

## boxes.json
```json
{ "image_shape": [H, W], "drawn_on": "<first tif>", "categories": [...],
  "created": "ISO timestamp",
  "boxes": [ {"name": "laser_trigger", "x0":..., "y0":..., "x1":..., "y1":...}, ... ] }
```
Coordinates are half-open pixel indices in full-resolution image space (x = column,
y = row, origin top-left). One box per category — duplicates hard-error at extraction.

## <run>_boxtraces.npz  (and .mat mirror)
| key | shape | meaning |
|---|---|---|
| `traces` | n_boxes × n_frames, float32 | mean pixel intensity per box per frame |
| `motion_energy` | n_boxes × n_frames, float32 | mean \|frame_t − frame_{t−1}\| per box; column 0 = 0 |
| `box_names` | n_boxes, str | canonical category names |
| `box_coords` | n_boxes × 4, int | x0, y0, x1, y1 |
| `files` | list of str | source tifs, concatenation order |
| `run`, `created` | str | provenance |

Do not rename these keys — MATLAB analysis and the (upcoming) bundle builder consume them.

## Canonical box categories
`laser_trigger, whisker_pad, paw, paw_at_nose, eye, nose, wheel` — edit
`CANONICAL_BOXES` in box_extract_cluster.py only in coordination with the team (it changes
every downstream label). `paw_at_nose` (added Aug 9): region in front of the snout
where the paw appears during face-grooming — its intensity + ME report the
'pawing at nose' motif. Existing runs drawn without it just warn as missing; to add
it, redraw with `--overwrite` (the GUI seeds `reuse previous` with the run's own
boxes, so it's one click + one new box) and re-extract with `--force`.
`laser_trigger` must sit on a **non-saturated** region: it exists to capture the 2P
laser-onset brightness step used by the (excluded) sync stage, and a clipped-white box
cannot show a step.

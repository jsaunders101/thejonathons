"""Ground-truth test for neuralprocessing_v5.ipynb — soma-selective ROI acceptance.

Synthetic FOV contains BOTH somata and dendritic processes with independent calcium
signals, plus shot noise and drift. Passing means: the shape gate keeps somata and
rejects processes, and the anatomical size band is derived correctly from a real
PrairieView-style XML.
"""
import json, sys, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import tifffile as tiff

NB = "/Users/scottseneca/repos/CSHL_M1_Arousal/handoff/neuralprocessing_v5.ipynb"
WORK = Path("/private/tmp/claude-502/-Users-scottseneca-repos/"
            "795f5657-7be3-4599-80c6-5c879b97631e/scratchpad/v5_test")

UM_PER_PX = 1.0          # -> soma band 9-20 um == 9-20 px across == area 64-314 px
FS, T = 128, 600
FRAME_PERIOD = 0.0328181

# ---------------------------------------------------------------- ground truth
SOMAS = [(30, 30, 6.0), (30, 95, 6.0), (95, 30, 5.0), (95, 95, 6.5), (62, 62, 5.5)]
# (y, x, radius px)  -> diameters 10-13 um at 1 um/px: inside the 9-20 um band
DENDS = [  # (y0, x0, y1, x1, halfwidth) straight segments, 3 px wide
    (15, 55, 45, 58, 1.5), (55, 15, 58, 45, 1.5), (70, 100, 105, 103, 1.5),
    (100, 55, 103, 90, 1.5),
]
ARCS = [(62, 20, 18, 1.4)]   # (cy, cx, radius, halfwidth) curved process


def build_movie(rng):
    yy, xx = np.mgrid[:FS, :FS]
    footprints, kinds = [], []
    for (cy, cx, r) in SOMAS:
        footprints.append(np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (r / 1.6) ** 2))))
        kinds.append("soma")
    for (y0, x0, y1, x1, hw) in DENDS:
        # distance to the segment
        p = np.stack([yy, xx], -1).astype(float)
        a = np.array([y0, x0], float); b = np.array([y1, x1], float)
        ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)
        footprints.append(np.exp(-(d ** 2) / (2 * hw ** 2)))
        kinds.append("dend")
    for (cy, cx, r, hw) in ARCS:
        d = np.abs(np.hypot(yy - cy, xx - cx) - r)
        ang = np.arctan2(yy - cy, xx - cx)
        m = (ang > -0.2) & (ang < 1.9)
        footprints.append(np.where(m, np.exp(-(d ** 2) / (2 * hw ** 2)), 0.0))
        kinds.append("dend")
    footprints = np.array(footprints)

    # independent slow calcium signals (AR1 on sparse events)
    sig = np.zeros((len(footprints), T))
    for i in range(len(footprints)):
        ev = rng.random(T) < 0.012
        s = np.zeros(T)
        for t in range(1, T):
            s[t] = 0.97 * s[t - 1] + ev[t]
        sig[i] = s / max(s.max(), 1e-9)

    anat = 12.0 + 6.0 * rng.random((FS, FS))                 # static texture
    base = 180.0 + anat
    mov = np.empty((T, FS, FS), np.float32)
    dy = 1.6 * np.sin(np.arange(T) / 90.0)                   # slow drift
    dx = 1.3 * np.cos(np.arange(T) / 130.0)
    from scipy.ndimage import shift as ndshift
    for t in range(T):
        clean = base + 95.0 * (footprints * sig[:, t][:, None, None]).sum(0)
        mov[t] = ndshift(clean, (dy[t], dx[t]), order=1, mode="nearest")
    mov = rng.poisson(np.clip(mov, 0, None)).astype(np.uint16)   # shot noise
    return mov, footprints, kinds


def write_xml(path):
    frames = "".join(
        f'<Frame relativeTime="{i*FRAME_PERIOD:.6f}" absoluteTime="{i*FRAME_PERIOD:.6f}"/>'
        for i in range(T))
    path.write_text(f"""<?xml version="1.0" encoding="utf-8"?>
<PVScan version="5.5" date="8/8/2026 9:55:00 AM">
  <PVStateShard>
    <PVStateValue key="micronsPerPixel">
      <IndexedValue index="XAxis" value="{UM_PER_PX}" />
      <IndexedValue index="YAxis" value="{UM_PER_PX}" />
      <IndexedValue index="ZAxis" value="1" />
    </PVStateValue>
    <PVStateValue key="opticalZoom" value="2.0" />
    <PVStateValue key="pixelsPerLine" value="{FS}" />
    <PVStateValue key="linesFrame" value="{FS}" />
    <PVStateValue key="positionCurrent">
      <SubindexedValues index="ZAxis">
        <SubindexedValue subindex="0" value="182.5" description="Z Focus" />
      </SubindexedValues>
    </PVStateValue>
  </PVStateShard>
  <Sequence type="TSeries Timed Element" time="09:55:00">{frames}</Sequence>
</PVScan>""")


# ------------------------------------------------------------------- unit test
def unit_test_roi_shape(ns):
    """roi_shape() against shapes whose answers are known analytically."""
    roi_shape = ns["roi_shape"]
    yy, xx = np.mgrid[:60, :60]
    ok = True

    disk = ((yy - 30) ** 2 + (xx - 30) ** 2) <= 6.5 ** 2
    a, ar, sol = roi_shape(disk)
    print(f"    disk r=6.5 : area {a:4d}  axis_ratio {ar:.3f} (expect ~1.0)  "
          f"solidity {sol:.3f} (expect >0.93)")
    ok &= abs(ar - 1.0) < 0.08 and sol > 0.93

    line = np.zeros((60, 60), bool); line[15:45, 30] = True          # 1 x 30
    a, ar, sol = roi_shape(line)
    print(f"    line 1x30  : area {a:4d}  axis_ratio {ar:.1f} (expect ~30)   "
          f"solidity {sol:.3f} (expect ~1.0 -> caught by axis_ratio ONLY)")
    ok &= 25 < ar < 35

    bar = np.zeros((60, 60), bool); bar[15:45, 29:32] = True          # 3 x 30
    a, ar, sol = roi_shape(bar)
    print(f"    bar  3x30  : area {a:4d}  axis_ratio {ar:.2f} (expect ~10)   "
          f"solidity {sol:.3f}")
    ok &= 8 < ar < 12

    d = np.abs(np.hypot(yy - 30, xx - 30) - 16)
    ang = np.arctan2(yy - 30, xx - 30)
    arc = (d < 1.4) & (ang > -0.2) & (ang < 1.9)
    a, ar, sol = roi_shape(arc)
    print(f"    arc        : area {a:4d}  axis_ratio {ar:.2f}              "
          f"solidity {sol:.3f} (expect <0.6 -> caught by solidity)")
    ok &= sol < 0.6

    ell = (((yy - 30) / 9.0) ** 2 + ((xx - 30) / 4.5) ** 2) <= 1      # 2:1 soma
    a, ar, sol = roi_shape(ell)
    print(f"    2:1 ellipse: area {a:4d}  axis_ratio {ar:.2f} (expect ~2.0, "
          f"ACCEPTED at 2.5)  solidity {sol:.3f}")
    ok &= 1.8 < ar < 2.2 and sol > 0.9
    return ok


# ------------------------------------------------------------------- notebook
def run_notebook(work, pca, corr_thr, shape_filter):
    cells = [c for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
    src = ["".join(c["source"]) for c in cells]
    out_dir = work / f"out_pca{pca}_shape{shape_filter}"

    patched = []
    for s in src:
        if s.startswith("# ---- data ----"):
            s = s.replace(
                "DATA_PATH = Path('/grid/courses/data/imagcourse/GECI Project Jonathons/"
                "Old Data/jonathans_finale/20260808_M1_Mouse1/spon_sniff_run1/"
                "TSeries-08082026-0955-020')", f"DATA_PATH = Path({str(work)!r})")
            s = s.replace("TIF_NAME  = 'TSeries-08082026-0955-020_Cycle00001_Ch2_000001"
                          ".ome.tif'", "TIF_NAME  = 'movie.tif'")
            s = s.replace("PCA_ENABLE = True", f"PCA_ENABLE = {pca}")
            s = s.replace("PCA_RANK  = 50", "PCA_RANK  = 20")
            s = s.replace("CORR_THRESHOLD = 0.5", f"CORR_THRESHOLD = {corr_thr}")
            s = s.replace("SHAPE_FILTER   = True", f"SHAPE_FILTER   = {shape_filter}")
        if s.startswith("OUT_ROOT = Path("):
            s = ("OUT_ROOT = Path(%r)\n" % str(out_dir)) + \
                "\n".join(s.splitlines()[1:]).replace(
                    'out = OUT_ROOT / "neural data output" / DATA_PATH.name',
                    'out = OUT_ROOT')
        patched.append(s)

    ns = {"__name__": "__main__"}
    for i, s in enumerate(patched):
        try:
            exec(compile(s, f"<cell {i}>", "exec"), ns)
        except Exception as e:
            print(f"\n  !! cell {i} FAILED: {type(e).__name__}: {e}")
            print("  ---- source ----")
            print("\n".join(f"   {n:3d}| {l}" for n, l in
                            enumerate(s.splitlines(), 1))[:2500])
            raise
    return ns


def score(ns, footprints, kinds):
    """Assign every accepted ROI to the ground-truth source it overlaps most."""
    masks = ns["masks"]
    hits = []
    for m in masks:
        ov = [(footprints[k][m].sum(), k) for k in range(len(footprints))]
        best, k = max(ov)
        tot = footprints[:, m].sum()
        hits.append(kinds[k] if tot > 0 and best / tot > 0.5 else "background")
    n_soma = hits.count("soma")
    n_dend = hits.count("dend")
    n_bg = hits.count("background")
    somas_found = len({np.argmax([footprints[k][m].sum() for k in range(len(SOMAS))])
                       for m in masks
                       if max(footprints[k][m].sum() for k in range(len(footprints)))
                       == max(footprints[k][m].sum() for k in range(len(SOMAS)))})
    return n_soma, n_dend, n_bg, somas_found


if __name__ == "__main__":
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    rng = np.random.default_rng(0)
    print("building synthetic FOV: %d somata + %d processes, %dx%d, T=%d, %.1f um/px"
          % (len(SOMAS), len(DENDS) + len(ARCS), FS, FS, T, UM_PER_PX))
    mov, footprints, kinds = build_movie(rng)
    tiff.imwrite(WORK / "movie.tif", mov)
    write_xml(WORK / "TSeries-08082026-0955-020.xml")

    fails = []
    print("\n[A] shape-gate OFF (v4-equivalent acceptance, anatomical size band)")
    ns_off = run_notebook(WORK, False, 0.55, False)
    s0, d0, b0, _ = score(ns_off, footprints, kinds)
    print(f"    -> accepted {len(ns_off['masks'])}: {s0} soma, {d0} dendrite, {b0} bg")

    print("\n[B] shape-gate ON")
    ns_on = run_notebook(WORK, False, 0.55, True)
    s1, d1, b1, _ = score(ns_on, footprints, kinds)
    print(f"    -> accepted {len(ns_on['masks'])}: {s1} soma, {d1} dendrite, {b1} bg")

    print("\n[C] roi_shape unit tests")
    if not unit_test_roi_shape(ns_on):
        fails.append("roi_shape unit test")

    print("\n[D] XML / size band")
    print(f"    um_per_px  = {ns_on['um_per_px']} (expect {UM_PER_PX})")
    print(f"    SIZE_MIN/MAX = {ns_on['SIZE_MIN']}/{ns_on['SIZE_MAX']} px "
          f"(expect {int(round(np.pi/4*9**2))}/{int(round(np.pi/4*20**2))})")
    print(f"    z depth    = {ns_on['_xml_info'].get('z_um')} (expect 182.5)")
    if abs(ns_on["um_per_px"] - UM_PER_PX) > 1e-9:
        fails.append("um_per_px")
    if ns_on["SIZE_MIN"] != int(round(np.pi / 4 * 81)) or \
       ns_on["SIZE_MAX"] != int(round(np.pi / 4 * 400)):
        fails.append("size band derivation")
    if ns_on["_xml_info"].get("z_um") != "182.5":
        fails.append("z depth parse")

    print("\n[E] PCA on, shape gate on")
    ns_pca = run_notebook(WORK, True, 0.55, True)
    s2, d2, b2, _ = score(ns_pca, footprints, kinds)
    print(f"    -> accepted {len(ns_pca['masks'])}: {s2} soma, {d2} dendrite, {b2} bg")

    print("\n[F] label==row invariant + saved shape metrics")
    for name, ns in (("shape-on", ns_on), ("pca", ns_pca)):
        rm, ms = ns["roi_map"], ns["masks"]
        okinv = all((rm == k + 1).sum() == m.sum() for k, m in enumerate(ms))
        okmet = (len(ns["roi_axis_ratio"]) == len(ms) and
                 len(ns["roi_solidity"]) == len(ms))
        print(f"    {name}: label==row {okinv}   shape arrays aligned {okmet}")
        if not (okinv and okmet):
            fails.append(f"invariants ({name})")

    print("\n" + "=" * 70)
    if d1 > d0:
        fails.append("shape gate increased dendrite count")
    if d1 > 0:
        fails.append(f"shape gate still admitted {d1} dendrite ROI(s)")
    if s1 < 3:
        fails.append(f"shape gate kept only {s1} soma ROI(s) — too destructive")
    print(f"dendrite ROIs: gate OFF {d0}  ->  gate ON {d1}      "
          f"soma ROIs: {s0} -> {s1}")
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)

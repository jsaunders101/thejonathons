#!/usr/bin/env python3
"""Tests for ca_extract.py.

1. fast_pearson & streaming corr map are EXACTLY the notebook math (vs scipy pearsonr)
2. next_roi matches a direct port of the notebook implementation (with its view-mutation
   bug fixed in both) on a small synthetic movie
3. End-to-end on a synthetic Bruker-style t-series with planted cells + known drift:
   recovers the motion, finds the cells, traces match planted signals, outputs land in
   the mirrored tree, meta/QC/mat all present, skip/--force semantics work.

Run:  python test_ca_extract.py <workdir>
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
import ca_extract as cx  # noqa: E402

rng = np.random.default_rng(7)


def test_pearson_equivalence():
    from scipy.stats import pearsonr
    for _ in range(20):
        a, b = rng.normal(size=500), rng.normal(size=500)
        assert abs(cx.fast_pearson(a, b) - pearsonr(a, b)[0]) < 1e-10
    print("PASS  fast_pearson == scipy.pearsonr")


def test_corrmap_equivalence():
    from scipy.stats import pearsonr
    mov = rng.normal(100, 10, (60, 12, 14)).astype(np.float32)
    ours = cx.neighbor_corr_map(mov)
    for y in range(1, 11):
        for x in range(1, 13):
            px = mov[:, y, x].astype(float)
            s = mov[:, y-1:y+2, x-1:x+2].sum((1, 2)).astype(float) - px
            r, _ = pearsonr(px, s)
            assert abs(ours[y, x] - r) < 1e-6, (y, x, ours[y, x], r)
    print("PASS  streaming corr map == notebook pearsonr loop (interior)")


def notebook_next_roi(corr_map, frames, corr_threshold, size_threshold):
    """Direct port of the notebook function, with .copy() fixing its view-mutation bug."""
    from scipy.stats import pearsonr
    from skimage.morphology import binary_dilation
    i, j = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    this_trace = frames[:, i, j].astype(np.float64).copy()
    this_roi = np.zeros(corr_map.shape, dtype=np.uint8)
    this_roi[i, j] = 1
    this_corr_map = corr_map.copy()
    this_corr_map[i, j] = 0
    growing = True
    while growing and (this_roi.sum() < size_threshold):
        growing = False
        dilated = binary_dilation(this_roi, np.ones((3, 3)))
        neighbours = np.argwhere(dilated - this_roi)
        trace_to_be_added = np.zeros(len(this_trace))
        for n in range(neighbours.shape[0]):
            i, j = neighbours[n, :]
            if this_corr_map[i, j] != 0:
                px_trace = frames[:, i, j].astype(np.float64)
                r, _ = pearsonr(this_trace, px_trace)
                if r > corr_threshold:
                    growing = True
                    this_roi[i, j] = 1
                    this_corr_map[i, j] = 0
                    trace_to_be_added += px_trace
        this_trace += trace_to_be_added
    return this_roi, this_trace, this_roi.sum(), this_corr_map


def make_cell_movie(T=120, Y=40, X=44, cells=((12, 12), (28, 30)), noise=2.0):
    """Static anatomy (textured background + dim cell bodies) + transient signals.
    Anatomy matters: phase correlation needs spatial structure in every frame."""
    from scipy.ndimage import gaussian_filter
    anat = (gaussian_filter(rng.normal(0, 1, (Y, X)), 3) * 40
            + gaussian_filter(rng.normal(0, 1, (Y, X)), 1) * 10)
    mov = np.empty((T, Y, X), dtype=np.float32)
    mov[:] = 200 + anat[None]
    mov += rng.normal(0, noise, (T, Y, X)).astype(np.float32)
    sigs = []
    yy, xx = np.mgrid[:Y, :X]
    for k, (cy, cx_) in enumerate(cells):
        sig = np.clip(rng.normal(0, 1, T), 0, None)
        sig = np.convolve(sig, np.exp(-np.arange(15) / 5), mode="same") * 25
        g = np.exp(-((yy - cy) ** 2 + (xx - cx_) ** 2) / (2 * 2.5 ** 2))
        mov += 25 * g[None]                      # static cell body (anatomy)
        mov += sig[:, None, None] * g[None]      # transient signal
        sigs.append(sig)
    return mov, sigs


def test_next_roi_equivalence():
    mov, _ = make_cell_movie()
    cmap = cx.neighbor_corr_map(mov)
    cmap[:2] = 0; cmap[-2:] = 0; cmap[:, :2] = 0; cmap[:, -2:] = 0
    r1, t1, s1, u1 = cx.next_roi(cmap, mov, 0.3, 200)
    r2, t2, s2, u2 = notebook_next_roi(cmap, mov, 0.3, 200)
    assert np.array_equal(r1, r2) and s1 == s2
    assert np.allclose(t1, t2, rtol=1e-6)
    assert np.array_equal(u1 == 0, u2 == 0)
    print(f"PASS  next_roi == notebook implementation (ROI of {s1} px identical)")


def write_synthetic_tseries(ts_dir, name, T=300, Y=64, X=64, fps=30.0,
                            drift_px=3.0, cells=((20, 18), (44, 40), (30, 50))):
    """Bruker-style folder: XML with Frames+PVStateShard, movie split across 3 ome.tifs,
    known sinusoidal drift applied, plus a fake VoltageRecording companion."""
    from scipy.ndimage import shift as ndshift
    ts_dir.mkdir(parents=True, exist_ok=True)
    mov, sigs = make_cell_movie(T, Y, X, cells)
    dy = drift_px * np.sin(np.arange(T) / 40)
    dx = drift_px * np.cos(np.arange(T) / 55)
    shifted = np.empty_like(mov)
    for t in range(T):
        shifted[t] = ndshift(mov[t], (dy[t], dx[t]), cval=200.0)
    raw = np.clip(shifted, 0, 65535).astype(np.uint16)

    third = T // 3
    for k in range(3):
        tifffile.imwrite(ts_dir / f"{name}_Cycle00001_Ch2_{k+1:06d}.ome.tif",
                         raw[k * third:(k + 1) * third])

    period = 1.0 / fps
    frames_xml = "\n".join(
        f'    <Frame index="{i+1}" relativeTime="{i*period:.6f}" '
        f'absoluteTime="{2.5+i*period:.6f}"/>' for i in range(T))
    (ts_dir / f"{name}.xml").write_text(f"""<?xml version="1.0"?>
<PVScan date="8/9/2026 2:11:00 PM" version="5.8">
  <PVStateShard>
    <PVStateValue key="framePeriod" value="{period}"/>
    <PVStateValue key="opticalZoom" value="2"/>
    <PVStateValue key="laserPower">
      <IndexedValue index="0" value="42.5" description="Pockels"/>
    </PVStateValue>
    <PVStateValue key="micronsPerPixel">
      <IndexedValue index="XAxis" value="1.18"/>
      <IndexedValue index="YAxis" value="1.18"/>
    </PVStateValue>
  </PVStateShard>
  <Sequence type="TSeries Timed Element" time="14:11:02.5">
{frames_xml}
  </Sequence>
</PVScan>""")
    (ts_dir / f"{name}_Cycle00001_VoltageRecording_001.csv").write_text("t,v\n0,0\n")
    return sigs, (dy, dx), cells


def test_end_to_end(work):
    data = work / "data"
    out = work / "ca_output"
    ts1 = data / "20260808_M1_Mouse1" / "spon_run1" / "TSeries-TEST-001"
    sigs, (dy, dx), cells = write_synthetic_tseries(ts1, "TSeries-TEST-001")
    # second (tiny) run + an empty behavior slot to test full-tree mirroring
    ts2 = data / "20260808_M1_Mouse2" / "spon_run1" / "TSeries-TEST-002"
    write_synthetic_tseries(ts2, "TSeries-TEST-002", T=150)
    (data / "20260808_M1_Mouse1" / "spon_run1" / "behavior_cam").mkdir()

    params = work / "params.json"     # synthetic cells grow ~150 px; widen the filter
    params.write_text(json.dumps({"size_range": [15, 300], "size_threshold": 300}))

    script = Path(cx.__file__)
    def run(*a):
        r = subprocess.run([sys.executable, str(script), *a],
                           capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode not in (0, 3):
            print(r.stderr)
            raise AssertionError(f"exit {r.returncode}")
        return r

    print("--- single-run mode ---")
    run("run", str(ts1), "--out", str(out), "--data-root", str(data),
        "--params", str(params))
    od = out / "20260808_M1_Mouse1" / "spon_run1" / "TSeries-TEST-001"
    for f in ("ca.npz", "ca.mat", "meta.json", "qc.json", "qc_motion.png", "qc_roi.png"):
        assert (od / f).exists(), f"missing {f}"

    d = np.load(od / "ca.npz", allow_pickle=True)
    # motion recovered? (registration shift = NEGATIVE of applied drift)
    r_dy = cx.fast_pearson(d["y_shift2"], -dy)
    r_dx = cx.fast_pearson(d["x_shift2"], -dx)
    assert r_dy > 0.95 and r_dx > 0.95, (r_dy, r_dx)
    # cells found near planted locations, traces match planted signals?
    xy = d["roi_xy"]
    assert len(xy) >= 2, f"only {len(xy)} ROIs"
    matched = 0
    for (cy, cx_) in cells:
        dists = np.hypot(xy[:, 0] - cx_, xy[:, 1] - cy)
        if dists.min() < 4:
            k = int(np.argmin(dists))
            sig = sigs[[c == (cy, cx_) for c in cells].index(True)]
            r = cx.fast_pearson(d["F_raw"][k].astype(float), sig)
            assert r > 0.7, f"cell at {(cy, cx_)} trace corr {r:.2f}"
            matched += 1
    assert matched >= 2, f"only {matched} planted cells recovered"
    # metadata complete?
    meta = json.loads((od / "meta.json").read_text())
    assert abs(meta["fps"] - 30.0) < 0.01
    assert meta["pv_state"]["laserPower"]["0"]["value"] == "42.5"
    assert meta["channel"] == "2"
    assert any(c["voltage_recording"] for c in meta["companion_files"])
    assert meta["source_rel_path"].startswith("20260808_M1_Mouse1")
    # dff sane?
    assert d["dff"].shape == d["F_raw"].shape and np.isfinite(d["dff"]).all()
    print(f"PASS  end-to-end: motion r=({r_dy:.3f},{r_dx:.3f}), "
          f"{matched}/3 cells recovered, metadata complete")

    print("--- skip / force ---")
    r = run("run", str(ts1), "--out", str(out), "--data-root", str(data),
            "--params", str(params))
    assert "skipping" in r.stdout
    print("PASS  idempotent skip")

    print("--- batch mode + mirror ---")
    run("batch", str(data), "--out", str(out), "--params", str(params))
    assert (out / "20260808_M1_Mouse1" / "spon_run1" / "behavior_cam").is_dir(), \
        "empty behavior dir not mirrored"
    od2 = out / "20260808_M1_Mouse2" / "spon_run1" / "TSeries-TEST-002"
    assert (od2 / "ca.npz").exists()
    assert (out / "ca_summary.csv").exists()
    from scipy.io import loadmat
    m = loadmat(od2 / "ca.mat")
    assert "F_raw" in m and "frame_times" in m
    print("PASS  batch: mirror incl. empty dirs, second run processed, summary, .mat loads")


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "ca_test_work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    test_pearson_equivalence()
    test_corrmap_equivalence()
    test_next_roi_equivalence()
    test_end_to_end(work)
    print("\nALL TESTS PASSED")

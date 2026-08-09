#!/usr/bin/env python3
"""Equivalence + memory tests for batched extraction (stream_frame_blocks).

`reference()` below is the ORIGINAL per-frame loop math (pre-batching,
validated bit-identical on real ThorCam data). The batched implementation must
reproduce it EXACTLY (np.array_equal, no tolerance) on:

  1. multi-file multi-page runs, incl. a 1-page file in the middle; block
     boundaries forced with 1- and 2-frame blocks (motion carry across blocks
     and files)
  2. RGB(A) pages (channel-mean fallback path)
  3. single-IFD contiguous giants (ImageJ truncate=True) read via memmap —
     the layout where per-page streaming silently yields ONE frame per file
  4. an ABORTED giant (metadata declares more frames than bytes present) —
     frame count clamps to the bytes that exist
  5. trace_viewer_cluster.FrameSource on a truncated file — every frame reachable
  6. peak RSS stays bounded on a ~420 MB movie extracted with batch_mb=16
     (subprocess; catches any whole-movie materialization)

Run:  python test_extract_stream_cluster.py <workdir>
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from box_extract_cluster import extract_run, run_tifs, to_gray, truncated_memmap  # noqa: E402

BOXES = [{"name": "laser_trigger", "x0": 1, "y0": 2, "x1": 11, "y1": 9},
         {"name": "whisker_pad", "x0": 5, "y0": 5, "x1": 30, "y1": 22}]
COORDS = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in BOXES]


def reference(data, coords):
    """The pre-batching per-frame implementation, verbatim math."""
    tr, me, prev = [], [], None
    for frame in data:
        crops = []
        for x0, y0, x1, y1 in coords:
            c = frame[y0:y1, x0:x1]
            if c.ndim == 3:
                c = c[..., :3].mean(axis=-1)
            crops.append(np.ascontiguousarray(c, dtype=np.float32))
        tr.append([float(c.mean()) for c in crops])
        me.append([0.0] * len(crops) if prev is None else
                  [float(np.abs(c - p).mean()) for c, p in zip(crops, prev)])
        prev = crops
    return (np.asarray(tr, np.float32).T, np.asarray(me, np.float32).T)


def write_boxes(run):
    (run / "proc").mkdir(parents=True, exist_ok=True)
    (run / "proc" / "boxes.json").write_text(json.dumps({
        "boxes": BOXES, "categories": [b["name"] for b in BOXES]}))


def load_traces(run):
    npz = sorted((run / "proc").glob("*_boxtraces.npz"))[0]
    d = np.load(npz, allow_pickle=True)
    return d["traces"], d["motion_energy"]


def check_equal(run, data, batch_mb, label):
    write_boxes(run)
    extract_run(run, save_mat=False, force=True, batch_mb=batch_mb)
    tr, me = load_traces(run)
    rtr, rme = reference(data, COORDS)
    assert tr.shape == rtr.shape, (label, tr.shape, rtr.shape)
    assert np.array_equal(tr, rtr), f"{label}: traces differ from reference"
    assert np.array_equal(me, rme), f"{label}: motion differs from reference"


def test_multi(work):
    rng = np.random.default_rng(0)
    run = work / "multi" / "run01"
    run.mkdir(parents=True)
    parts = [rng.integers(50, 60000, (n, 32, 40), dtype=np.uint16)
             for n in (7, 1, 5)]
    for k, mov in enumerate(parts):
        tifffile.imwrite(run / f"img_{k:03d}.tif", mov, photometric="minisblack")
    data = np.concatenate(parts)
    # 1-frame and 2-frame blocks force boundary carries; default = one block
    for mb, tag in ((None, "default"), (0.001, "1-frame blocks"),
                    (0.005, "2-frame blocks")):
        check_equal(run, data, mb if mb else 256, f"multi/{tag}")
    print("PASS  multi-file paged: bit-identical at default/1-frame/2-frame blocks")


def test_rgb(work):
    rng = np.random.default_rng(1)
    run = work / "rgb" / "run01"
    run.mkdir(parents=True)
    mov = rng.integers(0, 255, (6, 32, 40, 3), dtype=np.uint8)
    tifffile.imwrite(run / "img.tif", mov, photometric="rgb")
    check_equal(run, mov, 0.001, "rgb/1-frame blocks")
    check_equal(run, mov, 256, "rgb/default")
    print("PASS  RGB pages: channel-mean path bit-identical")


def _write_truncated(path, data):
    tifffile.imwrite(path, data, imagej=True, truncate=True)
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) != 1 or tf.series[0].shape[0] != len(data):
            return False
    return True


def test_truncated(work):
    rng = np.random.default_rng(2)
    run = work / "trunc" / "run01"
    run.mkdir(parents=True)
    data = rng.integers(50, 60000, (37, 24, 30), dtype=np.uint16)
    if not _write_truncated(run / "img.tif", data):
        print("SKIP  truncated: this tifffile did not write a single-IFD file")
        return
    check_equal(run, data, 256, "truncated/default")
    check_equal(run, data, 0.002, "truncated/1-frame blocks")

    # the viewer must reach every frame too (old code saw 1 frame per file)
    from trace_viewer_cluster import FrameSource
    fs = FrameSource(run_tifs(run))
    assert len(fs) == 37, f"FrameSource sees {len(fs)} frames, want 37"
    assert np.array_equal(fs.get(36), to_gray(data[36]))
    assert np.array_equal(fs.get(0), to_gray(data[0]))
    print("PASS  truncated single-IFD giant: memmap path bit-identical; "
          "FrameSource reaches all 37 frames")


def test_aborted(work):
    rng = np.random.default_rng(3)
    run = work / "aborted" / "run01"
    run.mkdir(parents=True)
    data = rng.integers(50, 60000, (37, 24, 30), dtype=np.uint16)
    p = run / "img.tif"
    if not _write_truncated(p, data):
        print("SKIP  aborted: this tifffile did not write a single-IFD file")
        return
    with tifffile.TiffFile(str(p)) as tf:
        off = int(tf.pages[0].dataoffsets[0])
    fbytes = 24 * 30 * 2
    os.truncate(p, off + 32 * fbytes)          # simulate a killed recording
    write_boxes(run)
    extract_run(run, save_mat=False, force=True, batch_mb=256)
    tr, me = load_traces(run)
    assert tr.shape[1] == 32, f"expected clamp to 32 frames, got {tr.shape[1]}"
    rtr, rme = reference(data[:32], COORDS)
    assert np.array_equal(tr, rtr) and np.array_equal(me, rme)
    print("PASS  aborted giant: frame count clamps to bytes present (37 -> 32)")


def test_rss(work):
    """A ~420 MB movie extracted with batch_mb=16 must not materialize the
    whole stack: peak RSS stays under 250 MB (whole-load would be >420 MB)."""
    rng = np.random.default_rng(4)
    run = work / "rss" / "run01"
    run.mkdir(parents=True)
    mov = rng.integers(50, 60000, (160, 1150, 1150), dtype=np.uint16)
    movie_mb = mov.nbytes / 2**20
    tifffile.imwrite(run / "img.tif", mov, photometric="minisblack")
    del mov
    write_boxes(run)
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        f"from pathlib import Path\n"
        f"from box_extract_cluster import extract_run\n"
        f"extract_run(Path({str(run)!r}), save_mat=False, force=True, batch_mb=16)\n"
        f"import resource\n"
        f"r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        f"print('PEAK_RSS_BYTES', r if sys.platform == 'darwin' else r * 1024)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError("RSS subprocess failed")
    peak = int(next(l.split()[1] for l in r.stdout.splitlines()
                    if l.startswith("PEAK_RSS_BYTES")))
    peak_mb = peak / 2**20
    assert peak < 250 * 2**20, (
        f"peak RSS {peak_mb:.0f} MB on a {movie_mb:.0f} MB movie — "
        f"extraction is materializing the stack")
    tr, _ = load_traces(run)
    assert tr.shape == (2, 160)
    print(f"PASS  memory bound: {movie_mb:.0f} MB movie, batch 16 MB, "
          f"peak RSS {peak_mb:.0f} MB")


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "extract_stream_test_work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    test_multi(work)
    test_rgb(work)
    test_truncated(work)
    test_aborted(work)
    test_rss(work)
    print("\nALL TESTS PASSED")

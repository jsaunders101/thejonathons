#!/usr/bin/env python3
"""Tests for run_behavior.py (headless; forces MPLBACKEND=Agg for subprocesses).

1. Full headless chain on synthetic runs: --boxes-from + --no-view -> boxes written,
   traces extracted (npz + mat, correct names/shapes), exit 0.
2. Idempotent rerun (skips), --force redoes.
3. Degrade chain: no-folder + closed stdin aborts cleanly; typed path on stdin works;
   missing boxes headless refuses with guidance; empty folder refuses.
4. Failure isolation: a run with a corrupt tif is reported (DRAW or EXTRACT error)
   and skipped; the other runs still process; exit 2.
5. --qc-png writes per-run *_tracesqc.png headlessly.
6. probe_gui reports headless under Agg.
7. FolderPicker logic under Agg: navigate down/up, pagination, select, cancel.

Run:  python test_run_behavior.py <workdir>
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

SCRIPT = Path(__file__).parent / "run_behavior.py"
ENV = {**os.environ, "MPLBACKEND": "Agg"}

TEMPLATE_BOXES = [
    {"name": "laser_trigger", "x0": 1, "y0": 1, "x1": 10, "y1": 8},
    {"name": "whisker_pad", "x0": 15, "y0": 10, "x1": 30, "y1": 25},
]


def make_run(folder, n_files, frames_per_file, shape=(32, 40)):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(folder.name) % 2**32)
    for k in range(n_files):
        mov = rng.integers(50, 200, (frames_per_file, *shape), dtype=np.uint16)
        # minisblack forces one page per frame — without it tifffile treats a
        # 4-frame stack as a single 4-channel page
        tifffile.imwrite(folder / f"img_{k:03d}.tif", mov,
                         photometric="minisblack")


def run(*a, expect=0, feed=None):
    kw = {"input": feed} if feed is not None else {"stdin": subprocess.DEVNULL}
    r = subprocess.run([sys.executable, str(SCRIPT), *a],
                       capture_output=True, text=True, env=ENV, **kw)
    if expect is not None and r.returncode != expect:
        print(r.stdout)
        print(r.stderr)
        raise AssertionError(f"exit {r.returncode}, expected {expect}")
    return r


def test_chain(work):
    master = work / "behavior"
    make_run(master / "day1" / "run01", 2, 4)
    make_run(master / "day1" / "run02", 1, 5)

    tmpl = work / "template_boxes.json"
    tmpl.write_text(json.dumps({"boxes": TEMPLATE_BOXES}))

    r = run(str(master), "--boxes-from", str(tmpl), "--no-view")
    for name, nframes in (("run01", 8), ("run02", 5)):
        proc = master / "day1" / name / "proc"
        assert (proc / "boxes.json").exists(), f"{name}: no boxes.json"
        npz = proc / f"{name}_boxtraces.npz"
        assert npz.exists(), f"{name}: no npz"
        assert npz.with_suffix(".mat").exists(), f"{name}: no mat"
        d = np.load(npz, allow_pickle=True)
        assert [str(n) for n in d["box_names"]] == ["laser_trigger", "whisker_pad"]
        assert d["traces"].shape == (2, nframes), d["traces"].shape
        assert d["motion_energy"].shape == (2, nframes)
    assert "Done." in r.stdout
    print("PASS  headless chain: boxes from template, both runs extracted")

    r = run(str(master), "--boxes-from", str(tmpl), "--no-view")
    assert r.stdout.count("skipping") >= 2
    print("PASS  idempotent rerun skips extraction")

    npz = master / "day1" / "run01" / "proc" / "run01_boxtraces.npz"
    before = npz.stat().st_mtime_ns
    r = run(str(master), "--boxes-from", str(tmpl), "--no-view", "--force")
    assert npz.stat().st_mtime_ns != before, "--force did not re-extract"
    print("PASS  --force re-extracts")

    r = run(str(master), "--boxes-from", str(tmpl), "--qc-png")
    for name in ("run01", "run02"):
        png = master / "day1" / name / "proc" / f"{name}_tracesqc.png"
        assert png.exists(), f"{name}: no tracesqc png"
    print("PASS  --qc-png writes per-run QC images headlessly")
    return master, tmpl


def test_degrade_chain(work, master, tmpl):
    r = run(expect=None)   # no folder, headless, closed stdin -> clean abort
    assert r.returncode != 0
    assert "no folder selected" in (r.stdout + r.stderr).lower()
    print("PASS  no folder + closed stdin aborts cleanly")

    r = run("--boxes-from", str(tmpl), "--no-view", feed=f"{master}\n")
    assert "Done." in r.stdout
    print("PASS  typed path on stdin drives the full chain")

    master2 = work / "behavior2"
    make_run(master2 / "run01", 1, 3)
    r = run(str(master2), "--no-view", expect=None)
    assert r.returncode != 0
    assert "--boxes-from" in r.stdout + r.stderr, "no guidance in refusal"
    print("PASS  headless draw refused with --boxes-from guidance")

    empty = work / "empty"
    empty.mkdir()
    r = run(str(empty), "--no-view", expect=None)
    assert r.returncode != 0 and "No runs" in r.stdout + r.stderr
    print("PASS  empty folder refused")


def test_failure_isolation(work, tmpl):
    master = work / "behavior3"
    make_run(master / "run01", 1, 4)
    make_run(master / "run02", 1, 4)
    # run03: corrupt tif, no boxes -> fails at DRAW (first-frame read), skipped
    (master / "run03").mkdir(parents=True)
    (master / "run03" / "bad.tif").write_bytes(b"this is not a tiff")
    # run04: corrupt tif WITH boxes.json -> passes draw, fails at EXTRACT, isolated
    (master / "run04").mkdir(parents=True)
    (master / "run04" / "bad.tif").write_bytes(b"also not a tiff")
    (master / "run04" / "proc").mkdir()
    (master / "run04" / "proc" / "boxes.json").write_text(json.dumps({
        "boxes": TEMPLATE_BOXES,
        "categories": [b["name"] for b in TEMPLATE_BOXES]}))

    r = run(str(master), "--boxes-from", str(tmpl), "--no-view", expect=2)
    out = r.stdout + r.stderr
    assert "DRAW ERROR" in out, "run03 draw failure not reported"
    assert "EXTRACT ERROR" in out, "run04 extract failure not reported"
    for name in ("run01", "run02"):
        assert (master / name / "proc" / f"{name}_boxtraces.npz").exists(), \
            f"{name} not extracted despite isolation"
    assert "run03" in out and "run04" in out
    print("PASS  corrupt runs isolated (draw + extract), good runs still processed, exit 2")


def test_probe_gui():
    os.environ["MPLBACKEND"] = "Agg"
    import run_behavior as rb
    rb._GUI.update(ok=None, why="")
    assert rb.probe_gui() is False
    assert "Agg" in rb._GUI["why"]
    print("PASS  probe_gui reports headless under Agg")


def test_picker(work):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_behavior import FolderPicker

    tree = work / "picktree"
    (tree / "alpha" / "inner").mkdir(parents=True)
    (tree / "beta").mkdir()

    class E:
        pass

    p = FolderPicker(tree)
    assert p.cwd == tree.resolve()

    ev = E()
    ev.artist = next(t for t, d in p._artists.items()
                     if d is not None and d.name == "alpha")
    p._on_pick(ev)
    assert p.cwd == (tree / "alpha").resolve()

    p._on_up()
    assert p.cwd == tree.resolve()

    p._on_select()
    assert p.result == tree.resolve()
    assert not plt.fignum_exists(p.fig.number)

    p2 = FolderPicker(tree)
    p2._on_cancel()
    assert p2.result is None
    print("PASS  FolderPicker: navigate down/up, select, cancel")

    many = work / "manytree"
    for i in range(30):
        (many / f"d{i:02d}").mkdir(parents=True)
    p3 = FolderPicker(many)
    page1 = {d.name for d in p3._artists.values() if d}
    assert len(page1) == FolderPicker.PER_PAGE
    p3._flip(+1)
    page2 = {d.name for d in p3._artists.values() if d}
    assert len(page2) == 30 - FolderPicker.PER_PAGE
    assert not page1 & page2
    p3._flip(+1)                       # clamped at last page
    assert {d.name for d in p3._artists.values() if d} == page2
    p3._on_cancel()
    print("PASS  FolderPicker: pagination + clamping")


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "run_behavior_test_work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    master, tmpl = test_chain(work)
    test_degrade_chain(work, master, tmpl)
    test_failure_isolation(work, tmpl)
    test_probe_gui()
    test_picker(work)
    print("\nALL TESTS PASSED")

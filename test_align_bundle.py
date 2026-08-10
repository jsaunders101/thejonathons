#!/usr/bin/env python3
"""Execute the REAL notebook cells against synthetic ground truth + the real demo npz.

Ground truth design: a physical event at a known time T_ev after t-series start. The
camera truly runs at 15.02 Hz (not the nominal 15.0), so this also proves the two-anchor
map absorbs drift instead of trusting cam_fps.
"""
import json, sys, tempfile, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

NB = Path("/Users/scottseneca/repos/CSHL_M1_Arousal/handoff/align_bundle.ipynb")
REPO = Path("/Users/scottseneca/repos/CSHL_M1_Arousal")
cells = [c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"]
SRC = ["".join(c["source"]) for c in cells]
print(f"loaded {len(SRC)} code cells from the notebook\n")

TMP = Path(tempfile.mkdtemp(prefix="aligntest_"))
FAILS = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(label)


# ------------------------------------------------------------------ fixtures
def make_beh(path, n_cam, n_box=6, plant=(), sigma=3.0, names=None):
    names = names or ["laser_trigger", "whisker_pad", "paw", "paw_at_nose", "eye", "wheel"]
    rng = np.random.default_rng(0)
    me = rng.gamma(2.0, 3.0, (n_box, n_cam))
    f = np.arange(n_cam)
    for fp in plant:                      # a bump in whisker_pad (row 1)
        me[1] += 400.0 * np.exp(-0.5 * ((f - fp) / sigma) ** 2)
    me[:, 0] = 0.0                        # as box_extract writes it
    lum = rng.normal(100, 1, (n_box, n_cam))
    np.savez_compressed(path, motion_energy=me.astype(np.float32),
                        traces=lum.astype(np.float32), box_names=np.array(names[:n_box]),
                        box_coords=np.zeros((n_box, 4), int))
    return me


def make_ca(folder, n_roi, n_2p, fps, complete=True, params=True, transpose=False):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    dff = rng.normal(0, .1, (n_roi, n_2p)).astype(np.float32)
    np.save(folder / "dff.npy", dff.T if transpose else dff)
    np.save(folder / "traces_raw.npy", (dff + 5).astype(np.float32))
    np.save(folder / "roi_npix.npy", np.full(n_roi, 40))
    if complete:
        (folder / "_COMPLETE").write_text("")
    if params:
        (folder / "params.json").write_text(json.dumps({"fps": fps, "n_frames": n_2p}))
    return dff


def run_notebook(cfg_extra):
    """exec cells 0..4 (config/imports/loaders/align/run-loop) with a custom config."""
    ns = {"__name__": "__main__"}
    exec(cfg_extra, ns)
    for s in SRC[1:5]:
        exec(s, ns)
    return ns


# =================================================================== TEST A
print("TEST A -- ground truth: known event times, camera truly at 15.02 Hz")
D, P2 = 600.0, 30.0
n_2p = int(round(D * P2))
R_CAM_TRUE, CAM_NOM = 15.02, 15.0
on = 137
span = int(round(D * R_CAM_TRUE))
off = on + span
n_cam = off + 51
T_EV = [60.0, 300.0, 540.0]
plant = [int(round(on + t * R_CAM_TRUE)) for t in T_EV]

beh = TMP / "A_boxtraces.npz"
make_beh(beh, n_cam, plant=plant)
ca = TMP / "A_ca"
dff_in = make_ca(ca, 25, n_2p, P2)

cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_A")
RUNS = {{"A": dict(beh=r"{beh}", ca=r"{ca}", on={on}, off={off},
                   tseries_fps={P2})}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
ns = run_notebook(cfg)
check("run completed", "A" in ns["results"], str(ns["failed"]))
A, m = ns["results"]["A"]

for t_true, fp in zip(T_EV, plant):
    k = int(np.argmax(A["beh_me"][1]
                      * ((A["t"] > t_true - 5) & (A["t"] < t_true + 5))))
    err = A["t"][k] - t_true
    check(f"event at {t_true:.0f}s recovered", abs(err) < 1.0 / R_CAM_TRUE,
          f"err {err*1000:+.1f} ms (tol +-{1000/R_CAM_TRUE:.0f} ms)")

check("effective cam fps recovers TRUE 15.02, not nominal 15.0",
      abs(m["cam_fps_used"] - R_CAM_TRUE) < 1e-6,
      f"{m['cam_fps_used']:.5f}")
check("dff bit-identical to input (never resampled)",
      np.array_equal(A["dff"], dff_in.astype(np.float64)))
check("t starts at 0 and ends at (n_2p-1)/fps",
      A["t"][0] == 0 and abs(A["t"][-1] - (n_2p - 1) / P2) < 1e-9)
check("beh arrays are [n_box x T]", A["beh_me"].shape == (6, n_2p)
      and A["beh_lum"].shape == (6, n_2p))
check("beh_env present (condition imported)", "beh_env" in A)
check("M = dff + 3 x 6 behaviour rows",
      A["M"].shape == (25 + 18, n_2p) and len(A["M_names"]) == 43)
check("M block 0 is dff", np.array_equal(A["M"][:25], A["dff"]))
check("artifact only at the two edges",
      A["artifact"][:1].all() and A["artifact"][-1:].all()
      and not A["artifact"][5:-5].any(), f"{A['artifact'].sum()} frames")
check("no NaN/Inf anywhere",
      all(np.isfinite(v).all() for v in A.values() if v.dtype.kind == "f"))
check("interp factor reported (2P faster than camera)",
      abs(m["interp_factor"] - P2 / R_CAM_TRUE) < 1e-9, f"{m['interp_factor']:.4f}x")
check("t-series clock is the constant hardcoded rate",
      m["tseries_clock"] == "constant_rate" and m["tseries_fps"] == P2)

z = np.load(TMP / "out_A" / "A_aligned.npz", allow_pickle=True)
check("npz round-trips", np.array_equal(z["beh_me"], A["beh_me"])
      and json.loads(str(z["meta"]))["n_2p"] == n_2p)

# =================================================================== TEST B
print("\nTEST B -- ME repairs at frame 0 and frame `on`")
me_raw = np.load(beh, allow_pickle=True)["motion_energy"].astype(float)
check("source npz really has ME[:,0] == 0", (me_raw[:, 0] == 0).all())
# frame `on` maps to t=0, so its value is A['beh_me'][:,0] exactly
check("frame 0 repaired (not a zero-hole at t=0)", (A["beh_me"][:, 0] > 0).all(),
      f"min {A['beh_me'][:,0].min():.2f}")
check("frame `on` repaired to on+1", np.allclose(A["beh_me"][:, 0], me_raw[:, on + 1]))

# =================================================================== TEST C
print("\nTEST C -- real ThorCam demo npz (old vintage, 6 boxes, no trim keys)")
demo = Path("/Users/scottseneca/Desktop/pupil demo 2/proc/pupil_demo_2_boxtraces.npz")
if demo.exists():
    n_cam_d = np.load(demo, allow_pickle=True)["motion_energy"].shape[1]
    on_d, off_d = 100, n_cam_d - 100
    span_d = off_d - on_d
    n2 = int(round(span_d / 15.0 * P2))          # make RATIO == 1.000
    ca_d = TMP / "C_ca"
    make_ca(ca_d, 40, n2, P2)
    cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_C")
RUNS = {{"C": dict(beh=r"{demo}", ca=r"{ca_d}", on={on_d}, off={off_d},
                   tseries_fps={P2})}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
    ns_c = run_notebook(cfg)
    check("real demo npz aligns", "C" in ns_c["results"], str(ns_c["failed"]))
    if "C" in ns_c["results"]:
        Ac, mc = ns_c["results"]["C"]
        check("old vintage detected", mc["beh_vintage"] == "pre-trim", mc["beh_vintage"])
        check("6 boxes carried by name", list(Ac["beh_names"]) == mc["box_names"]
              and len(mc["box_names"]) == 6, str(mc["box_names"]))
        check("implied cam rate == 15.0 by construction",
              abs(mc["cam_fps_used"] - 15.0) < 5e-3,
              f"{mc['cam_fps_used']:.5f} Hz")
        check("output finite on real data",
              all(np.isfinite(v).all() for v in Ac.values() if v.dtype.kind == "f"))
        exec(SRC[5], ns_c)                       # the QC cell must not crash
        check("QC figure written", (TMP / "out_C" / "C_qc.png").exists())
else:
    print("  SKIP -- demo npz not present")

# =================================================================== TEST C2
print("\nTEST C2 -- beh may be a proc/ folder; _with_time.npz must not be picked up")
proc = TMP / "procdir"
proc.mkdir()
shutil.copy(beh, proc / "run1_behavior_boxtraces.npz")
(proc / "run1_behavior_boxtraces_with_time.npz").write_bytes(b"xx")   # decoy + 2-byte file
cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_C2")
RUNS = {{"C2": dict(beh=r"{proc}", ca=r"{ca}", on={on}, off={off},
                    tseries_fps={P2})}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
ns2 = run_notebook(cfg)
check("proc/ folder resolves to the boxtraces npz", "C2" in ns2["results"],
      str(ns2["failed"]))
if "C2" in ns2["results"]:
    check("picked the real file, not _with_time",
          ns2["results"]["C2"][1]["beh_npz"].endswith("run1_behavior_boxtraces.npz"))

# a 2-byte npz named explicitly must be refused, not parsed
cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_C3")
RUNS = {{"C3": dict(beh=r"{proc / "run1_behavior_boxtraces_with_time.npz"}", ca=r"{ca}",
                    on={on}, off={off}, tseries_fps={P2})}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
ns3 = run_notebook(cfg)
check("2-byte npz refused", "C3" in ns3["failed"] and "did not upload" in ns3["failed"]["C3"],
      ns3["failed"].get("C3", "NO ERROR")[:70])

# ambiguity must raise rather than guess
(proc / "other_boxtraces.npz").write_bytes((proc / "run1_behavior_boxtraces.npz").read_bytes())
cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_C4")
RUNS = {{"C4": dict(beh=r"{proc}", ca=r"{ca}", on={on}, off={off}, tseries_fps={P2})}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
ns4 = run_notebook(cfg)
check("two candidates -> refuse, do not guess",
      "C4" in ns4["failed"] and "candidates" in ns4["failed"]["C4"],
      ns4["failed"].get("C4", "NO ERROR")[:70])

# =================================================================== TEST F
print("\nTEST F -- single-anchor: camera stops early, 2P frames past coverage dropped")
# thormouse1 geometry: both clocks 30 Hz, camera quits ~4 min before the t-series ends.
F_2P, F_CAM = 30.0, 30.0
n2f = 36000                                   # 1200 s of 2P
on_f = 119
cov_frames = 28659                            # camera recorded this many after `on`
off_f = on_f + cov_frames
n_cam_f = off_f + 1
T_EV_F = [10.0, 500.0, 950.0]                 # all inside coverage
plant_f = [int(round(on_f + t * F_CAM)) for t in T_EV_F]
beh_f = TMP / "F_boxtraces.npz"
make_beh(beh_f, n_cam_f, plant=plant_f)
ca_f = TMP / "F_ca"
make_ca(ca_f, 24, n2f, F_2P)
cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_F")
RUNS = {{"F": dict(beh=r"{beh_f}", ca=r"{ca_f}", on={on_f}, off={off_f},
                   tseries_fps={F_2P}, cam_fps={F_CAM})}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
nsf = run_notebook(cfg)
check("single-anchor run completed", "F" in nsf["results"], str(nsf["failed"]))
if "F" in nsf["results"]:
    Af, mf = nsf["results"]["F"]
    check("mode is single_anchor_known_fps", mf["map_mode"] == "single_anchor_known_fps")
    check("cam rate is the SUPPLIED one, not the anchor-implied one",
          mf["cam_fps_used"] == F_CAM and abs(mf["cam_fps_implied_by_anchors"] - 23.88) < .05,
          f"used {mf['cam_fps_used']} vs implied {mf['cam_fps_implied_by_anchors']:.2f}")
    check("T truncated to coverage", mf["n_2p"] == cov_frames + 1
          and mf["n_2p_dropped"] == n2f - cov_frames - 1,
          f"T={mf['n_2p']} dropped={mf['n_2p_dropped']}")
    check("all arrays share the truncated length",
          Af["t"].shape[0] == mf["n_2p"] and Af["dff"].shape[1] == mf["n_2p"]
          and Af["beh_me"].shape[1] == mf["n_2p"] and Af["M"].shape[1] == mf["n_2p"])
    check("coverage ends at the right second",
          abs(mf["coverage_end_s"] - cov_frames / F_CAM) < 1e-9,
          f"{mf['coverage_end_s']:.3f}s")
    for t_true in T_EV_F:
        k = int(np.argmax(Af["beh_me"][1] * ((Af["t"] > t_true - 5) & (Af["t"] < t_true + 5))))
        err = Af["t"][k] - t_true
        check(f"event at {t_true:.0f}s recovered", abs(err) < 1.0 / F_CAM,
              f"err {err*1000:+.1f} ms")
    check("1:1 frame correspondence at 30/30 Hz",
          abs(Af["t"][1] - 1.0 / F_2P) < 1e-12 and mf["interp_factor"] == 1.0)
    check("no NaN after truncation",
          all(np.isfinite(v).all() for v in Af.values() if v.dtype.kind == "f"))
    # the same anchors WITHOUT cam_fps would stretch behaviour by 25.6%
    cfg2 = cfg.replace(f", cam_fps={F_CAM}", "")
    ns2 = run_notebook(cfg2)
    A2 = ns2["results"]["F"][0]
    k2 = int(np.argmax(A2["beh_me"][1] * ((A2["t"] > 940) & (A2["t"] < 1210))))
    check("two-anchor on the SAME data misplaces the 950 s event (why cam_fps exists)",
          abs(A2["t"][k2] - 950.0) > 100, f"lands at {A2['t'][k2]:.1f}s instead of 950s")

# =================================================================== TEST D
print("\nTEST D -- failure modes must RAISE, not produce a quietly-wrong bundle")
cases = {
    "transposed dff refused":      dict(kind="transpose"),
    "on >= off refused":           dict(on=5000, off=5000),
    "on out of range refused":     dict(on=-1, off=8000),
    "off past last frame refused": dict(on=10, off=10 ** 7),
    "tseries_fps != params.json":  dict(tseries_fps=15.0),
    "None anchor refused":         dict(on=None),
    "missing dff.npy refused":     dict(kind="nodff"),
}
for label, mod in cases.items():
    kind = mod.pop("kind", None)
    ca_x = TMP / f"D_{abs(hash(label))}"
    if kind == "transpose":
        make_ca(ca_x, 25, n_2p, P2, transpose=True)
    elif kind == "nodff":
        ca_x.mkdir(parents=True, exist_ok=True)
    else:
        make_ca(ca_x, 25, n_2p, P2)
    c = dict(beh=str(beh), ca=str(ca_x), on=on, off=off, tseries_fps=P2)
    c.update(mod)
    cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_D")
RUNS = {{"D": {c!r}}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
    ns_d = run_notebook(cfg)
    check(label, "D" in ns_d["failed"] and "D" not in ns_d["results"],
          ns_d["failed"].get("D", "NO ERROR RAISED")[:78])

# =================================================================== TEST E
print("\nTEST E -- a wrong tseries_fps is REPORTED, never refused")
ca_e = TMP / "E_ca"
make_ca(ca_e, 25, n_2p, P2, params=False)   # no params.json -> nothing can refuse this
cfg = f'''
from pathlib import Path
OUT_DIR = Path("{TMP}/out_E")
RUNS = {{"E": dict(beh=r"{beh}", ca=r"{ca_e}", on={on}, off={off},
                   tseries_fps={P2/2})}}
ENV_S, SAVE_MAT, REPO_DIR = 0.33, False, r"{REPO}"
'''
ns_e = run_notebook(cfg)
check("halved tseries_fps still builds (never refuses)",
      "E" in ns_e["results"] and "E" not in ns_e["failed"], str(ns_e["failed"]))
if "E" in ns_e["results"]:
    me_ = ns_e["results"]["E"][1]
    check("implied cam rate halves, exposing the error",
          abs(me_["cam_fps_used"] - R_CAM_TRUE / 2) < 0.01,
          f"{me_['cam_fps_used']:.4f} Hz (true camera {R_CAM_TRUE})")

print("\n" + "=" * 70)
print(f"{len(FAILS)} FAILURES" if FAILS else "ALL CHECKS PASSED")
for f in FAILS:
    print("  -", f)
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)

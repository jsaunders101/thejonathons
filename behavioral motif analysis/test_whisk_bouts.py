#!/usr/bin/env python3
"""Tests for whisk_bouts.py — synthetic ground truth + component checks.

1. moving_average is genuinely ZERO-PHASE (a symmetric ramp is not shifted) and
   does not taper at the edges.
2. viterbi recovers a known state path; forward_backward posteriors agree with it.
3. GMM recovers planted component means/weights.
4. END-TO-END on synthetic motion energy with PLANTED bouts: every bout found,
   onset timing within tolerance, and the systematic bias is reported (not
   assumed to be zero).
5. NEGATIVE CONTROL: a trace with no bouts (pure ongoing baseline) yields
   essentially nothing, and d' falls below the session gate.
6. Contamination flags fire: a grooming episode and an artifact spike are
   labelled, not silently included.
7. Full process_run on a synthetic run folder: npz/json/png written, idempotent,
   --force redoes, funnel counts internally consistent.

Run:  MPLBACKEND=Agg python test_whisk_bouts.py <workdir>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from whisk_bouts import (  # noqa: E402
    PARAMS_DEFAULT, condition, detect, dprime, fit_gmm2, forward_backward,
    log_emissions, moving_average, process_run, transition_matrix, viterbi)

FPS = 15.0
def synth(n=6000, onsets=None, bout_len=30, base=2.6, bout=3.9, sd=0.42,
          groom_at=None, artifact_at=None, seed=0):
    """Motion energy with planted bouts on an ONGOING (never zero) baseline.

    Values mirror the real gold session: log-ME baseline ~2.6 (ME~13.7),
    bout ~3.9 (ME~43), sd ~0.42.
    """
    # own RNG: tests must not depend on how many draws earlier tests made
    rng = np.random.default_rng(seed)
    if onsets is None:
        onsets = list(range(300, n - 300, 400))
    lg = rng.normal(base, sd, n)
    for o in onsets:
        lg[o:o + bout_len] = rng.normal(bout, sd, len(lg[o:o + bout_len]))
    me = {"whisker_pad": np.exp(lg)}
    pn = rng.normal(1.7, 0.3, n)
    if groom_at is not None:
        pn[groom_at:groom_at + 20] = 5.0
    me["paw_at_nose"] = np.exp(pn)
    lt = rng.normal(1.3, 0.15, n)
    if artifact_at is not None:
        lt[artifact_at:artifact_at + 2] = 6.0
    me["laser_trigger"] = np.exp(lt)
    me["paw"] = np.exp(rng.normal(1.9, 0.35, n))
    me["wheel"] = np.exp(rng.normal(1.9, 0.35, n))
    return me, onsets


def P(**kw):
    p = dict(PARAMS_DEFAULT); p["fps"] = FPS; p.update(kw)
    return p


def test_zero_phase():
    x = np.zeros(200); x[100:] = 1.0                 # step at 100
    y = moving_average(x, 11)
    # a symmetric filter puts the half-amplitude crossing back at the step
    cross = int(np.argmin(np.abs(y - 0.5)))
    assert abs(cross - 100) <= 1, f"phase shift: crossing at {cross}, want 100"
    flat = moving_average(np.full(50, 3.0), 9)
    assert np.allclose(flat, 3.0), "edge replication failed — ends tapered"
    print("PASS  moving_average: zero-phase, no edge taper")


def test_viterbi():
    true = np.array([0] * 50 + [1] * 50 + [0] * 50)
    x = np.where(true == 1, 5.0, 0.0) + np.random.default_rng(5).normal(0, 0.3, len(true))
    mu, sd = np.array([0.0, 5.0]), np.array([0.3, 0.3])
    lt = np.log(transition_matrix(20, 20))
    lp = np.log(np.array([0.5, 0.5]))
    path = viterbi(log_emissions(x, mu, sd), lt, lp)
    assert (path == true).mean() > 0.98, f"viterbi accuracy {(path==true).mean():.3f}"
    post = forward_backward(log_emissions(x, mu, sd), lt, lp)
    assert post.min() >= 0 and post.max() <= 1
    assert ((post > 0.5).astype(int) == true).mean() > 0.98, "posterior disagrees"
    print("PASS  viterbi + forward_backward recover a known state path")


def test_gmm():
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.normal(2.6, 0.42, 7500), rng.normal(3.9, 0.42, 2500)])
    mu, sd, pi = fit_gmm2(x)
    assert abs(mu[0] - 2.6) < 0.1 and abs(mu[1] - 3.9) < 0.1, mu
    assert abs(pi[0] - 0.75) < 0.06, pi
    assert dprime(mu, sd) > 2.5
    print(f"PASS  GMM recovers planted components (d'={dprime(mu, sd):.2f})")


def test_endtoend_planted():
    """Detection rate and onset timing across 20 independent noise realisations.

    One seed proves nothing about a stochastic detector; this reports hit rate,
    false positives and the systematic timing bias, and asserts on all three.
    """
    hits = misses = extra = 0
    errs = []
    for seed in range(20):
        me, onsets = synth(seed=seed)
        res = detect(me, P())
        found = res["onsets"]
        assert res["gates"]["dprime_pass"], (seed, res["gates"])
        matched = 0
        for o in onsets:
            near = found[np.abs(found - o) <= 8]
            if len(near):
                hits += 1; matched += 1
                errs.append(int(near[np.argmin(np.abs(near - o))]) - o)
            else:
                misses += 1
        extra += max(0, len(found) - matched)
    errs = np.array(errs)
    rate = hits / (hits + misses)
    bias, p90 = np.median(errs), np.percentile(np.abs(errs), 90)
    print(f"PASS  end-to-end over 20 seeds: hit rate {rate:.3f} "
          f"({hits}/{hits+misses}), {extra} false positives, "
          f"onset bias {bias:+.0f} frames ({bias/FPS*1000:+.0f} ms), "
          f"p90 |err| {p90:.0f} frames")
    assert rate >= 0.97, f"hit rate {rate:.3f}"
    assert extra <= 5, f"{extra} false positives across 20 runs"
    # backtracking is expected to land slightly EARLY (the envelope smears the
    # rise backwards); doc 11 S3 says flag it if it exceeds 2 frames
    assert abs(bias) <= 3, f"systematic onset bias {bias} frames"
    assert p90 <= 6, f"onset scatter p90 {p90} frames"


def test_negative_control():
    """No bouts planted: the detector must not invent them."""
    n, rng = 6000, np.random.default_rng(99)
    me = {"whisker_pad": np.exp(rng.normal(2.6, 0.42, n)),
          "paw_at_nose": np.exp(rng.normal(1.7, 0.3, n)),
          "laser_trigger": np.exp(rng.normal(1.3, 0.15, n))}
    res = detect(me, P())
    dp = res["gates"]["dprime"]
    n_found = len(res["onsets"])
    assert not res["gates"]["dprime_pass"], \
        f"pure noise passed the d' gate at {dp:.2f}"
    print(f"PASS  negative control: d'={dp:.2f} fails the {P()['dprime_min']} gate "
          f"({n_found} spurious bouts, correctly flagged as untrustworthy)")


def test_contamination_flags():
    # the contamination must land ON a planted bout (they sit at 300+400k);
    # an episode in a gap correctly flags nothing, which is not what we test here
    GROOM, ART = 1500, 2300
    me, onsets = synth(groom_at=GROOM, artifact_at=ART, seed=3)
    assert GROOM in onsets and ART in onsets, "contamination must overlap a bout"
    res = detect(me, P())
    rec = res["records"]
    groomed = [r for r in rec if r["groom"]]
    arted = [r for r in rec if r["artifact"]]
    assert groomed, "grooming episode overlapping a bout was not flagged"
    assert all(abs(r["onset"] - GROOM) < 30 for r in groomed), \
        f"wrong bouts flagged as grooming: {[r['onset'] for r in groomed]}"
    assert arted, "artifact spike overlapping a bout was not flagged"
    assert all(abs(r["onset"] - ART) < 30 for r in arted), \
        f"wrong bouts flagged as artifact: {[r['onset'] for r in arted]}"
    clean = [r for r in rec if not r["groom"] and not r["artifact"]]
    assert len(clean) == len(rec) - len(groomed) - len(arted), "flag bookkeeping"
    assert res["funnel"]["not_grooming"] < len(rec), "funnel did not count grooming"
    print(f"PASS  contamination flags: {len(groomed)} grooming, "
          f"{len(arted)} artifact bouts labelled (not dropped)")


def test_process_run(work):
    run = work / "day1" / "run01"
    (run / "proc").mkdir(parents=True)
    me, onsets = synth()
    names = list(me.keys())
    mat = np.stack([np.concatenate([[0.0], me[n]]) for n in names])  # col0 = 0
    np.savez_compressed(run / "proc" / "run01_boxtraces.npz",
                        traces=mat, motion_energy=mat,
                        box_names=np.array(names),
                        box_coords=np.zeros((len(names), 4), int))
    p = dict(PARAMS_DEFAULT); p["fps"] = FPS

    out = process_run(run, p)
    assert out.exists()
    js = json.loads((run / "proc" / "run01_whiskbouts.json").read_text())
    png = run / "proc" / "run01_whiskqc.png"
    assert png.exists(), "no QC png"
    d = np.load(out, allow_pickle=True)
    for k in ("env", "state", "posterior", "onsets", "groom", "artifact",
              "baseline_pre", "isolated", "fps", "params_hash"):
        assert k in d, f"missing npz key {k}"
    assert len(d["onsets"]) == len(d["groom"]) == len(d["duration_s"])
    f = js["funnel"]
    assert f["after_merge_and_min_length"] == len(d["onsets"])
    assert f["not_grooming"] <= len(d["onsets"])
    assert float(d["fps"]) == FPS
    print(f"PASS  process_run: npz+json+png written, {len(d['onsets'])} bouts, "
          f"funnel consistent")

    mt = out.stat().st_mtime_ns
    process_run(run, p)
    assert out.stat().st_mtime_ns == mt, "not idempotent"
    process_run(run, p, force=True)
    assert out.stat().st_mtime_ns != mt, "--force did not redo"
    print("PASS  process_run: idempotent skip, --force redoes")


if __name__ == "__main__":
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "whisk_test_work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    test_zero_phase()
    test_viterbi()
    test_gmm()
    test_endtoend_planted()
    test_negative_control()
    test_contamination_flags()
    test_process_run(work)
    print("\nALL TESTS PASSED")

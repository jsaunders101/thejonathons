#!/usr/bin/env python3
"""
whisk_bouts.py — detect whisking BOUTS and their onsets from motion energy.

Design and every constant here are justified by measurement in
11_whisking_onset.md / 10_behavior_motifs.md. The short version:

  * The whiskers never stop. The two-component structure in log motion energy is
    LOW-AMPLITUDE ONGOING WHISKING vs BOUT, not rest vs movement, so this detects
    a step up from a moving baseline. "Quiescent" below always means "below the
    bout threshold", never "still".
  * Whisk cycles (8-12 Hz) are above Nyquist at 15 fps and are unrecoverable.
    The only detectable object is the bout envelope. Onset precision ~1 frame.
  * Conditioning is log + a ~330 ms zero-phase envelope and NOTHING ELSE.
    Detrending was measured to be destructive (d' 2.63 -> 0.82 at 30 s) because
    at a ~25% duty cycle a rolling median tracks activity instead of baseline.
    Regressing out the laser_trigger common mode is also destructive
    (d' 2.60 -> 0.90). Both were tried; both are excluded on evidence.
  * Segmentation is a 2-state Gaussian HMM, initialised from a 2-component
    mixture and regularised by dwell-time priors. It self-calibrates per session,
    so no absolute threshold has to transfer between sessions with different
    illumination or box placement — that is what makes it consistent.
  * A threshold+hysteresis detector runs alongside as an independent cross-check.
    Agreement between the two is a validation signal, not decoration.

Reads  <run>/proc/<run>_boxtraces.npz   (needs a whisker box; uses paw_at_nose,
                                         laser_trigger, paw, wheel if present)
Writes <run>/proc/<run>_whiskbouts.npz  arrays + flags + funnel
       <run>/proc/<run>_whiskbouts.json params, gates, funnel (readable)
       <run>/proc/<run>_whiskqc.png     the eyeball gate

Outputs stay in CAMERA FRAMES. Conversion to 2P time is the sync stage's job.

  python whisk_bouts.py run   /data/behavior/day1/run01
  python whisk_bouts.py batch /data/behavior --runs-file runs.txt --run-index $SLURM_ARRAY_TASK_ID
  python whisk_bouts.py list  /data/behavior
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PARAMS_DEFAULT = {
    "fps": None,            # None = auto-detect from ThorCam tif tags, else Hz
    "whisker_box": "whisker_pad",
    "env_s": 0.33,          # zero-phase envelope; d' plateaus ~0.5 s, this keeps onsets sharp
    "gmm_iters": 400,
    "dwell_low_s": 0.3,     # expected dwell, low state  (measured gap median 267 ms)
    "dwell_high_s": 0.5,    # expected dwell, bout state
    "min_bout_s": 0.4,
    "merge_gap_s": 0.3,
    "baseline_pre_s": 1.0,  # "below bout threshold", NOT motionless
    "coactive_window_s": 1.0,
    "groom_pct": 99.0,      # paw_at_nose percentile that counts as grooming
    "artifact_pct": 99.5,   # laser_trigger spike veto
    "dprime_min": 1.5,      # session quality gate
}

LOW, HIGH = 0, 1


# ----------------------------------------------------------------- utilities

def params_hash(p):
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]


def moving_average(x, w):
    """Zero-phase moving average with edge replication.

    Zero-phase matters: a causal filter would delay every onset by half the
    window. np.convolve(mode='same') alone would taper the ends toward zero,
    which in log space is a large fake excursion, hence the edge padding.
    """
    w = max(1, int(w))
    if w == 1:
        return np.asarray(x, dtype=float).copy()
    pad = w // 2
    xp = np.pad(np.asarray(x, dtype=float), pad, mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="same")[pad:pad + len(x)]


def condition(me, fps, env_s):
    """Raw motion energy -> log -> zero-phase envelope. No detrend (measured)."""
    return moving_average(np.log(np.asarray(me, dtype=float) + 1e-3),
                          int(round(env_s * fps)))


def fit_gmm2(x, iters=400):
    """1-D two-component Gaussian mixture by EM. Returns (mu, sd, pi) sorted."""
    x = np.asarray(x, dtype=float)
    mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
    sd = np.array([x.std() / 2 + 1e-6] * 2)
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        r = np.stack([pi[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / sd[k]
                      for k in range(2)])
        r /= r.sum(0, keepdims=True) + 1e-300
        n = r.sum(1)
        if np.any(n < 2):
            break
        pi = n / len(x)
        mu = (r * x).sum(1) / n
        sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(1) / n) + 1e-6
    o = np.argsort(mu)
    return mu[o], sd[o], pi[o]


def dprime(mu, sd):
    return float((mu[1] - mu[0]) / np.sqrt(0.5 * (sd[0] ** 2 + sd[1] ** 2)))


# --------------------------------------------------------------------- HMM

def log_emissions(x, mu, sd):
    x = np.asarray(x, dtype=float)[None, :]
    mu, sd = mu[:, None], sd[:, None]
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def transition_matrix(dwell_low, dwell_high):
    """From expected dwell times in FRAMES: p(stay) = 1 - 1/dwell."""
    a = 1.0 / max(dwell_low, 1.01)
    b = 1.0 / max(dwell_high, 1.01)
    return np.array([[1 - a, a], [b, 1 - b]])


def viterbi(logem, logtrans, logpi):
    """Most likely state path. Log space throughout (underflow-safe)."""
    n_states, T = logem.shape
    delta = np.empty((n_states, T))
    psi = np.zeros((n_states, T), dtype=int)
    delta[:, 0] = logpi + logem[:, 0]
    for t in range(1, T):
        m = delta[:, t - 1][:, None] + logtrans     # from i (row) to j (col)
        psi[:, t] = np.argmax(m, axis=0)
        delta[:, t] = m[psi[:, t], np.arange(n_states)] + logem[:, t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[:, -1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return path


def forward_backward(logem, logtrans, logpi):
    """Posterior P(state=HIGH | all data), for a per-onset confidence."""
    from scipy.special import logsumexp
    n_states, T = logem.shape
    la = np.empty((n_states, T)); lb = np.zeros((n_states, T))
    la[:, 0] = logpi + logem[:, 0]
    for t in range(1, T):
        la[:, t] = logsumexp(la[:, t - 1][:, None] + logtrans, axis=0) + logem[:, t]
    for t in range(T - 2, -1, -1):
        lb[:, t] = logsumexp(logtrans + logem[:, t + 1][None, :] + lb[:, t + 1][None, :],
                             axis=1)
    lg = la + lb
    lg -= logsumexp(lg, axis=0, keepdims=True)
    return np.exp(lg[HIGH])


# ------------------------------------------------------------- segmentation

def runs_of(mask):
    """Contiguous True runs -> [(start, end_inclusive)]."""
    out, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def merge_and_filter(bouts, merge_gap, min_len):
    merged = []
    for s, e in bouts:
        if merged and s - merged[-1][1] - 1 <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s + 1) >= min_len]


def refine_onset(env, start, lo_level):
    """Walk back to the last frame at/below the low-state level.

    The state transition is late by the envelope's rise; the departure from
    baseline is the onset. Bounded so a long slow ramp can't run away.
    """
    j = start
    limit = max(0, start - 30)
    while j > limit and env[j - 1] > lo_level:
        j -= 1
    return j


def hysteresis_detect(env, hi, lo, min_len, merge_gap):
    """Independent cross-check detector (threshold + hysteresis + backtrack)."""
    active, out, start = False, [], 0
    for i, v in enumerate(env):
        if not active and v > hi:
            start, active = refine_onset(env, i, lo), True
        elif active and v < lo:
            out.append((start, i)); active = False
    if active:
        out.append((start, len(env) - 1))
    return merge_and_filter(out, merge_gap, min_len)


# ------------------------------------------------------------------ detect

def detect_fps(run, fallback=15.0):
    """Read the true frame rate from ThorCam private tags, else fall back."""
    try:
        import tifffile
        tifs = sorted([f for f in Path(run).iterdir()
                       if f.suffix.lower() in (".tif", ".tiff")
                       and not f.name.startswith(".")],
                      key=lambda p: [int(t) if t.isdigit() else t.lower()
                                     for t in re.split(r"(\d+)", p.name)])
        if not tifs:
            return fallback, "fallback (no tifs)"
        with tifffile.TiffFile(tifs[0]) as tf:
            def ts(page):
                t = page.tags
                return t.get(32781).value * (1 << 32) + t.get(32782).value
            n = len(tf.pages)
            if n < 2:
                return fallback, "fallback (single page)"
            span = ts(tf.pages[n - 1]) - ts(tf.pages[0])
            return 1e9 / (span / (n - 1)), "ThorCam tif timestamps"
    except Exception:
        return fallback, "fallback (tags unreadable)"


def load_traces(npz_path, whisker_box):
    d = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in d["box_names"]]
    if whisker_box not in names:
        raise ValueError(f"no '{whisker_box}' box in {names}")
    me = {n: d["motion_energy"][i].astype(float) for i, n in enumerate(names)}
    # motion_energy[0] is 0 by definition (no previous frame). Do NOT drop it:
    # dropping shifts every index by one relative to the movie, and onsets have
    # to stay in MOVIE-FRAME space for the sync mapping and for the viewer
    # overlay. Replace it with frame 1's value instead — one duplicated sample
    # out of thousands, and log(0+eps) would otherwise be a huge false outlier.
    for v in me.values():
        if len(v) > 1:
            v[0] = v[1]
    return me, names, d


def detect(me, params):
    """Full detection on one run's motion-energy dict. Returns a result dict."""
    fps = params["fps"]
    wb = params["whisker_box"]
    f = lambda s: max(1, int(round(s * fps)))                       # noqa: E731

    env = condition(me[wb], fps, params["env_s"])
    mu, sd, pi = fit_gmm2(env, params["gmm_iters"])
    dp = dprime(mu, sd)

    logtrans = np.log(transition_matrix(f(params["dwell_low_s"]),
                                        f(params["dwell_high_s"])))
    logem = log_emissions(env, mu, sd)
    logpi = np.log(np.clip(pi, 1e-12, None))
    state = viterbi(logem, logtrans, logpi)
    post = forward_backward(logem, logtrans, logpi)

    raw_bouts = runs_of(state == HIGH)
    bouts = merge_and_filter(raw_bouts, f(params["merge_gap_s"]),
                             f(params["min_bout_s"]))
    lo_level = mu[LOW] + sd[LOW]
    bouts = [(refine_onset(env, s, lo_level), e) for s, e in bouts]

    # independent cross-check
    xbouts = hysteresis_detect(env, 0.5 * (mu[0] + mu[1]), lo_level,
                               f(params["min_bout_s"]), f(params["merge_gap_s"]))
    xon = np.array([s for s, _ in xbouts])

    # ---- per-onset labels. Label, never filter: doc 10 sec 1b measured that
    # filtering on isolation empties the dataset.
    def pct_env(name, p):
        """Conditioned trace for a flag channel plus its threshold percentile."""
        if name not in me:
            return None
        e = condition(me[name], fps, params["env_s"])
        return e, float(np.percentile(e, p))

    groom = pct_env("paw_at_nose", params["groom_pct"])
    art = pct_env("laser_trigger", params["artifact_pct"])
    others = {k: condition(me[k], fps, params["env_s"])
              for k in ("paw", "wheel") if k in me}
    other_lo = {}
    for k, e in others.items():
        m2, s2, _ = fit_gmm2(e, params["gmm_iters"])
        other_lo[k] = m2[LOW] + s2[LOW]

    W = f(params["coactive_window_s"])
    B = f(params["baseline_pre_s"])
    onsets, rec = [], []
    for s, e in bouts:
        a, z = max(0, s - W), min(len(env), s + W + 1)
        co = {k: bool(np.any(others[k][a:z] > other_lo[k])) for k in others}
        rec.append({
            "onset": int(s), "offset": int(e),
            "duration_s": round((e - s + 1) / fps, 3),
            "amplitude": float(np.max(env[s:e + 1]) - mu[LOW]),
            "posterior": float(np.mean(post[s:e + 1])),
            "baseline_pre": bool(s >= B and np.all(env[s - B:s] < lo_level)),
            "groom": bool(groom is not None
                          and np.any(groom[0][max(0, s - 8):s + 8] > groom[1])),
            "artifact": bool(art is not None
                             and np.any(art[0][max(0, s - 2):s + 3] > art[1])),
            "coactive": co,
            "isolated": bool(co and not any(co.values())),
            "xcheck_frames": (int(np.min(np.abs(xon - s)))
                              if len(xon) else -1),
        })
        onsets.append(s)

    agree = sum(1 for r in rec if 0 <= r["xcheck_frames"] <= 2)
    funnel = {
        "hmm_state_runs": len(raw_bouts),
        "after_merge_and_min_length": len(bouts),
        "with_baseline_pre": sum(r["baseline_pre"] for r in rec),
        "not_grooming": sum(not r["groom"] for r in rec),
        "not_artifact": sum(not r["artifact"] for r in rec),
        "isolated_from_paw_wheel": sum(r["isolated"] for r in rec),
        "analysis_set_baseline_and_clean": sum(
            r["baseline_pre"] and not r["groom"] and not r["artifact"] for r in rec),
        "crosscheck_within_2_frames": agree,
        "crosscheck_detector_bouts": len(xbouts),
    }
    gates = {
        "dprime": round(dp, 3),
        "dprime_pass": bool(dp >= params["dprime_min"]),
        "low_state_weight": round(float(pi[LOW]), 3),
        "bout_duty_cycle": round(float(np.mean(state == HIGH)), 3),
        "crosscheck_agreement": (round(agree / len(rec), 3) if rec else None),
    }
    return {
        "env": env, "state": state.astype(np.int8), "posterior": post,
        "gmm_mu": mu, "gmm_sd": sd, "gmm_pi": pi,
        "onsets": np.array(onsets, dtype=int),
        "records": rec, "funnel": funnel, "gates": gates,
        "lo_level": float(lo_level), "hi_level": float(0.5 * (mu[0] + mu[1])),
        "xcheck_onsets": xon,
    }


# --------------------------------------------------------------------- I/O

def qc_plot(res, me, params, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env, mu = res["env"], res["gmm_mu"]
    fps = params["fps"]
    t = np.arange(len(env)) / fps
    fig, ax = plt.subplots(3, 1, figsize=(15, 9),
                           gridspec_kw={"height_ratios": [2, 2, 1.1]})
    g = res["gates"]
    fig.suptitle(f"{title} — whisking bouts   d'={g['dprime']:.2f} "
                 f"({'PASS' if g['dprime_pass'] else 'FAIL'})   "
                 f"{len(res['onsets'])} bouts, duty {g['bout_duty_cycle']:.2f}",
                 fontsize=12, fontweight="bold")

    ax[0].plot(t, env, lw=0.6, color="#1f3a68")
    ax[0].axhline(res["hi_level"], color="k", ls="--", lw=0.8, label="bout level")
    ax[0].axhline(res["lo_level"], color="0.5", ls=":", lw=0.8, label="baseline level")
    for r in res["records"]:
        c = ("tab:orange" if r["groom"] else
             "tab:red" if r["artifact"] else "tab:green")
        ax[0].axvspan(r["onset"] / fps, (r["offset"] + 1) / fps, color=c, alpha=0.22)
        ax[0].plot(r["onset"] / fps, env[r["onset"]], "v", color=c, ms=5)
    ax[0].set_ylabel("log ME (conditioned)")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title("green = clean bout, orange = grooming, red = artifact",
                    fontsize=9, loc="left")

    ax[1].plot(t, res["posterior"], lw=0.6, color="tab:purple")
    ax[1].set_ylabel("P(bout)")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].step(t, res["state"], where="mid", lw=0.8, color="k", alpha=0.5)

    xs = np.linspace(env.min(), env.max(), 300)
    ax[2].hist(env, bins=100, density=True, color="0.8")
    for k, lab, c in ((0, "ongoing (baseline)", "tab:green"),
                      (1, "bout", "tab:red")):
        ax[2].plot(xs, res["gmm_pi"][k] * np.exp(
            -0.5 * ((xs - mu[k]) / res["gmm_sd"][k]) ** 2) /
            (res["gmm_sd"][k] * np.sqrt(2 * np.pi)), color=c, lw=2, label=lab)
    ax[2].legend(fontsize=8)
    ax[2].set_xlabel("log ME (conditioned)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def atomic_write(path, writer):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        writer(fh)
    os.replace(tmp, path)


def process_run(run, params, force=False, no_plot=False):
    run = Path(run)
    proc = run / "proc"
    npzs = sorted(proc.glob("*_boxtraces.npz"))
    if not npzs:
        raise FileNotFoundError(f"no *_boxtraces.npz in {proc}")
    stem = npzs[0].stem.replace("_boxtraces", "")
    out = proc / f"{stem}_whiskbouts.npz"
    if out.exists() and not force:
        print(f"[{run.name}] {out.name} exists — skipping (--force to redo).")
        return out

    p = dict(params)
    if p["fps"] is None:
        p["fps"], how = detect_fps(run)
        print(f"[{run.name}] fps = {p['fps']:.4f} ({how})")
    me, names, _ = load_traces(npzs[0], p["whisker_box"])
    res = detect(me, p)

    atomic_write(out, lambda fh: np.savez_compressed(
        fh, env=res["env"], state=res["state"], posterior=res["posterior"],
        onsets=res["onsets"], offsets=np.array([r["offset"] for r in res["records"]], int),
        duration_s=np.array([r["duration_s"] for r in res["records"]], float),
        amplitude=np.array([r["amplitude"] for r in res["records"]], float),
        onset_posterior=np.array([r["posterior"] for r in res["records"]], float),
        baseline_pre=np.array([r["baseline_pre"] for r in res["records"]], bool),
        groom=np.array([r["groom"] for r in res["records"]], bool),
        artifact=np.array([r["artifact"] for r in res["records"]], bool),
        isolated=np.array([r["isolated"] for r in res["records"]], bool),
        gmm_mu=res["gmm_mu"], gmm_sd=res["gmm_sd"], gmm_pi=res["gmm_pi"],
        lo_level=res["lo_level"], hi_level=res["hi_level"],
        xcheck_onsets=res["xcheck_onsets"],
        fps=p["fps"], params_json=json.dumps(p), params_hash=params_hash(p),
        created=datetime.now().isoformat(timespec="seconds")))

    meta = {"run": str(run), "fps": p["fps"], "params": p,
            "params_hash": params_hash(p), "gates": res["gates"],
            "funnel": res["funnel"], "n_bouts": len(res["onsets"]),
            "created": datetime.now().isoformat(timespec="seconds")}
    (proc / f"{stem}_whiskbouts.json").write_text(json.dumps(meta, indent=2))

    if not no_plot:
        qc_plot(res, me, p, proc / f"{stem}_whiskqc.png", run.name)

    g, fn = res["gates"], res["funnel"]
    print(f"[{run.name}] d'={g['dprime']:.2f} "
          f"{'PASS' if g['dprime_pass'] else 'FAIL (below gate)'}  "
          f"duty={g['bout_duty_cycle']:.2f}  bouts={len(res['onsets'])}")
    print(f"[{run.name}] funnel: " +
          "  ".join(f"{k}={v}" for k, v in fn.items()))
    if not g["dprime_pass"]:
        print(f"[{run.name}] *** d' below {p['dprime_min']} — bouts from this "
              f"session are NOT trustworthy; inspect the QC png ***")
    return out



def inspect_plot(run, n_examples=12, window_s=2.5, context_s=90, out=None):
    """Figure for eyeballing ONSET TIMING: context, aggregate, and examples.

    The per-run QC png shows the whole session and is too compressed to judge
    whether a mark sits on the actual rise. This zooms in three ways:
      A  a context strip over the busiest stretch
      B  the onset-triggered average of the RAW log ME — deliberately NOT the
         envelope the detector used, so a mistimed set of onsets cannot hide
         behind its own smoothing. Correct timing = a step centred on 0.
      C  individual onsets spread across the session, at full time resolution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    proc = run / "proc"
    wb = sorted(proc.glob("*_whiskbouts.npz"))
    bt = sorted(proc.glob("*_boxtraces.npz"))
    if not wb or not bt:
        raise FileNotFoundError("need both *_whiskbouts.npz and *_boxtraces.npz")
    W = np.load(wb[0], allow_pickle=True)
    p = json.loads(str(W["params_json"]))
    fps = float(W["fps"])
    env, on, off = W["env"], W["onsets"], W["offsets"]
    me, _, _ = load_traces(bt[0], p["whisker_box"])
    raw = np.log(me[p["whisker_box"]] + 1e-3)
    t = np.arange(len(env)) / fps
    lo, hi = float(W["lo_level"]), float(W["hi_level"])

    n_show = min(n_examples, len(on))
    ex_rows = max(1, -(-n_show // 4))                 # ceil, 4 examples per row
    fig = plt.figure(figsize=(16, 5.5 + 2.3 * ex_rows))
    gs = fig.add_gridspec(2 + ex_rows, 4,
                          height_ratios=[1.5, 1.5] + [1] * ex_rows,
                          hspace=0.72, wspace=0.24)
    fig.suptitle(f"{run.name} — whisking onset timing   "
                 f"{len(on)} bouts, d'={json.loads((proc / (wb[0].stem.replace('_whiskbouts','') + '_whiskbouts.json')).read_text())['gates']['dprime']:.2f}",
                 fontsize=13, fontweight="bold")

    # ---- A: context strip over the busiest window
    wfr = int(context_s * fps)
    if len(env) > wfr:
        dens = np.array([np.sum((on >= i) & (on < i + wfr))
                         for i in range(0, len(env) - wfr, int(5 * fps))])
        a0 = int(np.argmax(dens)) * int(5 * fps)
    else:
        a0 = 0
    a1 = min(len(env), a0 + wfr)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t[a0:a1], raw[a0:a1], lw=0.5, color="0.65", label="raw log ME")
    ax.plot(t[a0:a1], env[a0:a1], lw=1.4, color="#1f3a68", label="envelope (detector input)")
    ax.axhline(hi, color="k", ls="--", lw=0.8)
    ax.axhline(lo, color="0.5", ls=":", lw=0.8)
    for s_, e_ in zip(on, off):
        if a0 <= s_ < a1:
            ax.axvspan(s_ / fps, (e_ + 1) / fps, color="tab:green", alpha=0.18)
            ax.axvline(s_ / fps, color="tab:green", lw=1.6)
    ax.set_xlim(t[a0], t[a1 - 1])
    ax.set_title(f"A. busiest {context_s:.0f} s — green line = detected onset, "
                 f"shading = bout", fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel("log ME"); ax.legend(fontsize=8, loc="upper right")

    # ---- B: onset-triggered average of RAW log ME
    half = int(window_s * fps)
    segs = np.array([raw[s_ - half:s_ + half] for s_ in on
                     if s_ - half >= 0 and s_ + half < len(raw)])
    tt = (np.arange(-half, half)) / fps
    ax = fig.add_subplot(gs[1, :2])
    m = segs.mean(0)
    se = segs.std(0) / np.sqrt(len(segs))
    ax.fill_between(tt, m - se, m + se, color="tab:green", alpha=0.3)
    ax.plot(tt, m, color="tab:green", lw=2, label=f"mean of {len(segs)} onsets")
    ax.plot(tt, np.median(segs, 0), color="k", lw=1, ls="--", label="median")
    ax.axvline(0, color="tab:red", lw=1.5)
    ax.axhline(lo, color="0.5", ls=":", lw=0.8)
    ax.set_title("B. onset-triggered average, RAW log ME\n"
                 "(independent of the envelope: a correct onset gives a step at 0)",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.set_xlabel("time from detected onset (s)"); ax.set_ylabel("log ME")
    ax.legend(fontsize=8)

    # rise-time diagnostic: where does the average actually cross halfway?
    axr = fig.add_subplot(gs[1, 2:])
    base = m[tt < -1.0].mean()
    peak = m[(tt > 0) & (tt < 1.0)].max()
    halfway = base + 0.5 * (peak - base)
    idx = np.argmax(m > halfway)
    cross = tt[idx] if np.any(m > halfway) else np.nan
    axr.plot(tt, (m - base) / max(peak - base, 1e-9), color="#1f3a68", lw=2)
    axr.axvline(0, color="tab:red", lw=1.5, label="detected onset")
    axr.axvline(cross, color="tab:orange", lw=1.5, ls="--",
                label=f"half-rise at {cross:+.2f} s")
    axr.axhline(0.5, color="0.6", lw=0.8, ls=":")
    axr.set_ylim(-0.15, 1.15)
    axr.set_title("B2. normalised rise\n"
                  "(gap between the two lines IS the timing bias)",
                  fontsize=9.5, loc="left", fontweight="bold")
    axr.set_xlabel("time from detected onset (s)")
    axr.legend(fontsize=8, loc="lower right")

    # ---- C: individual examples spread across the session
    pick = on[np.linspace(0, len(on) - 1, n_show).astype(int)]
    for k, s_ in enumerate(pick):
        ax = fig.add_subplot(gs[2 + k // 4, k % 4])
        a, b = max(0, s_ - half), min(len(env), s_ + half)
        ax.plot(t[a:b] - s_ / fps, raw[a:b], lw=0.6, color="0.65")
        ax.plot(t[a:b] - s_ / fps, env[a:b], lw=1.3, color="#1f3a68")
        ax.axvline(0, color="tab:green", lw=1.6)
        ax.axhline(hi, color="k", ls="--", lw=0.6)
        ax.axhline(lo, color="0.5", ls=":", lw=0.6)
        ax.set_title(f"onset @ {s_/fps:.1f} s", fontsize=8)
        if k == 0:
            ax.text(0, 1.30, "C. individual onsets spread across the session — "
                    "the green line should sit at the FOOT of the rise",
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    va="bottom")
        ax.tick_params(labelsize=7)
        if k % 4 == 0:
            ax.set_ylabel("log ME", fontsize=8)
        if k // 4 == ex_rows - 1:
            ax.set_xlabel("s from onset", fontsize=8)


    out = Path(out) if out else proc / (wb[0].stem.replace("_whiskbouts", "")
                                        + "_whiskonsets.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"half-rise of the onset-triggered average: {cross:+.3f} s "
          f"({cross*fps:+.1f} frames) relative to the detected onset")
    print(f"wrote {out}")
    return out


# --------------------------------------------------------------------- CLI

def find_runs(roots):
    runs = []
    for root in roots:
        root = Path(root)
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d != "proc" and not d.startswith("."))
            if any(f.lower().endswith((".tif", ".tiff")) and not f.startswith(".")
                   for f in fn):
                runs.append(Path(dp))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "batch", "list", "inspect"):
        p = sub.add_parser(name)
        p.add_argument("roots", nargs="+")
        if name != "list":
            p.add_argument("--params", help="JSON file overriding any subset of defaults")
            p.add_argument("--fps", type=float, default=None,
                           help="override frame rate (default: read from tif tags)")
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-plot", action="store_true")
        if name == "batch":
            p.add_argument("--runs-file")
            p.add_argument("--run-index", type=int, default=None)
        if name == "inspect":
            p.add_argument("--n-examples", type=int, default=12)
            p.add_argument("--window-s", type=float, default=2.5)
            p.add_argument("--context-s", type=float, default=90)
            p.add_argument("--out")
    args = ap.parse_args()

    if args.cmd == "inspect":
        for r in args.roots:
            inspect_plot(r, args.n_examples, args.window_s, args.context_s, args.out)
        return

    if args.cmd == "list":
        for i, r in enumerate(find_runs(args.roots)):
            j = list((r / "proc").glob("*_whiskbouts.json"))
            tag = "done" if j else "-"
            print(f"{i:3d}  {tag:5}  {r}")
        return

    params = dict(PARAMS_DEFAULT)
    if args.params:
        params.update(json.loads(Path(args.params).read_text()))
    if args.fps:
        params["fps"] = args.fps

    if args.cmd == "run":
        runs = [Path(r) for r in args.roots]
    elif getattr(args, "runs_file", None):
        runs = [Path(l) for l in Path(args.runs_file).read_text().splitlines() if l.strip()]
    else:
        runs = find_runs(args.roots)
    if getattr(args, "run_index", None) is not None:
        runs = [runs[args.run_index]]

    failed = []
    for r in runs:
        try:
            process_run(r, params, force=args.force, no_plot=args.no_plot)
        except Exception as e:
            print(f"[{Path(r).name}] ERROR {type(e).__name__}: {e} — continuing")
            failed.append(r)
    if failed:
        print(f"\n*** {len(failed)} runs failed: "
              f"{', '.join(Path(r).name for r in failed)} ***")
        sys.exit(2)


if __name__ == "__main__":
    main()

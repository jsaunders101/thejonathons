"""Prototype detector + event funnel for all three motif channels.

Answers the question the plan never costed: after every filtering stage the
design demands, HOW MANY clean onsets are actually left? If isolated,
quiescence-preceded whisking onsets number ~5, onset-triggered averaging is not
viable on this dataset and the plan has to change.

Doubles as a first sketch of the Stage-A/B detector in 11_whisking_onset.md.
"""
import numpy as np
from scipy.ndimage import median_filter

FPS = 15.0
RUN = '/Users/scottseneca/Desktop/pupil demo 2/proc/pupil_demo_2_boxtraces.npz'
TARGETS = ["whisker_pad", "paw", "wheel"]

d = np.load(RUN, allow_pickle=True)
names = [str(n) for n in d["box_names"]]
ME = d["motion_energy"][:, 1:].astype(float)
idx = {n: i for i, n in enumerate(names)}


def gmm2(x, iters=400):
    mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
    sd = np.array([x.std() / 2] * 2)
    pi = np.array([.5, .5])
    for _ in range(iters):
        r = np.stack([pi[k] * np.exp(-.5 * ((x - mu[k]) / sd[k]) ** 2) / sd[k]
                      for k in range(2)])
        r /= r.sum(0, keepdims=True) + 1e-300
        N = r.sum(1); pi = N / len(x); mu = (r * x).sum(1) / N
        sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(1) / N) + 1e-6
    o = np.argsort(mu)
    return mu[o], sd[o], pi[o]


def condition(me, env_s=0.33):
    """log -> zero-phase envelope. NO DETRENDING.

    Measured: detrending destroys rest-vs-active separation at every window
    tried (d' 2.63 -> 0.66/0.82/1.02/2.40 for 10/30/60/120 s), because with a
    ~25% duty cycle the rolling median tracks local activity instead of
    estimating rest. Envelope smoothing instead IMPROVES d' monotonically
    (2.63 raw -> 3.33 at 200 ms -> 3.56 at 330 ms -> 3.70 at 500 ms, plateau).
    330 ms trades a little separation for onset timing, which is refined
    later on the lightly-smoothed signal.
    """
    y = np.log(me + 1e-3)
    w = max(1, int(round(env_s * FPS)))
    return np.convolve(y, np.ones(w) / w, mode="same")


def bouts(env, hi, lo, min_len, merge_gap):
    """Hysteresis segmentation -> [(onset, offset)], onsets backtracked to lo."""
    active, out, i, n = False, [], 0, len(env)
    start = 0
    for i in range(n):
        if not active and env[i] > hi:
            j = i
            while j > 0 and env[j - 1] > lo:
                j -= 1                      # backtrack to baseline departure
            start, active = j, True
        elif active and env[i] < lo:
            out.append((start, i)); active = False
    if active:
        out.append((start, n - 1))
    merged = []
    for s, e in out:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s >= min_len]


MIN_LEN = int(0.4 * FPS)
MERGE = int(0.3 * FPS)
QUIET = int(1.0 * FPS)
ISO = int(1.0 * FPS)

env, thr, res = {}, {}, {}
print("=" * 78)
print("PER-CHANNEL SIGNAL PROPERTIES (conditioned: log + 330 ms envelope, no detrend)")
print("=" * 78)
print(f"{'channel':14s} {'d-prime':>8} {'rest w':>8} {'active w':>9} "
      f"{'ME rest':>8} {'ME act':>8} {'ratio':>6}")
for t in TARGETS + ["paw_at_nose", "laser_trigger"]:
    e = condition(ME[idx[t]])
    env[t] = e
    mu, sd, pi = gmm2(e)
    dp = (mu[1] - mu[0]) / np.sqrt(.5 * (sd[0] ** 2 + sd[1] ** 2))
    thr[t] = (0.5 * (mu[0] + mu[1]), mu[0] + 1.0 * sd[0])   # hi, lo
    raw = ME[idx[t]]
    r_rest = np.median(raw[e < thr[t][1]]) if (e < thr[t][1]).any() else np.nan
    r_act = np.median(raw[e > thr[t][0]]) if (e > thr[t][0]).any() else np.nan
    res[t] = dp
    print(f"{t:14s} {dp:8.2f} {pi[0]:8.2f} {pi[1]:9.2f} "
          f"{r_rest:8.1f} {r_act:8.1f} {r_act/max(r_rest,1e-9):6.1f}x")

print()
print("=" * 78)
print("EVENT FUNNEL — how many onsets survive each filter the plan demands")
print("=" * 78)
groom_hi = np.percentile(env["paw_at_nose"], 99)
art_hi = np.percentile(env["laser_trigger"], 99.5)
hdr = f"{'channel':14s} {'bouts':>7} {'+minlen':>8} {'+merge':>7} " \
      f"{'quiet_pre':>10} {'isolated':>9} {'clean':>7} {'per min':>8}"
print(hdr)
dur_min = ME.shape[1] / FPS / 60
allb = {}
for t in TARGETS:
    hi, lo = thr[t]
    e = env[t]
    raw_cross = int(np.sum((e[1:] > hi) & (e[:-1] <= hi)))
    b = bouts(e, hi, lo, MIN_LEN, MERGE)
    allb[t] = b
    q = [x for x in b if x[0] >= QUIET and np.all(e[x[0] - QUIET:x[0]] < lo)]
    iso = []
    for s, _ in q:
        a, z = max(0, s - ISO), min(len(e), s + ISO)
        if all(np.all(env[o][a:z] < thr[o][1]) for o in TARGETS if o != t):
            iso.append(s)
    clean = [s for s in iso
             if np.all(env["paw_at_nose"][max(0, s-8):s+8] < groom_hi)
             and np.all(env["laser_trigger"][max(0, s-2):s+3] < art_hi)]
    print(f"{t:14s} {raw_cross:7d} {len(b):8d} {len(b):7d} {len(q):10d} "
          f"{len(iso):9d} {len(clean):7d} {len(clean)/dur_min:8.1f}")

print()
print("=" * 78)
print("CO-OCCURRENCE — why isolation costs so much")
print("=" * 78)
act = {t: env[t] > thr[t][0] for t in TARGETS}
for t in TARGETS:
    print(f"  {t:14s} active {100*act[t].mean():5.1f}% of frames")
print()
for i, a in enumerate(TARGETS):
    for b_ in TARGETS[i+1:]:
        both = (act[a] & act[b_]).mean()
        exp = act[a].mean() * act[b_].mean()
        print(f"  P({a} & {b_}) = {100*both:5.1f}%   "
              f"chance {100*exp:5.1f}%   ratio {both/max(exp,1e-9):.1f}x")

print()
print(f"session duration: {dur_min:.1f} min")
print("NOTE: single 10-min session. Multiply by n_sessions for the real budget.")

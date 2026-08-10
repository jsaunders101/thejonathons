"""L1 'motion energy' vs L2 'motion power' vs RMS vs motion-area.

We currently store mean|dF| (L1). The user asks for "motion power", which
normally means mean(dF^2) (L2). They are not the same: L2 weights big
displacements far more heavily, which could sharpen bout contrast — or just
amplify artifacts. Recomputes all four metrics from RAW FRAMES on an active
stretch and compares how well each separates whisking bouts from baseline.

Also asks whether the metric saturates with speed, which matters because the
plan wants a CONTINUOUS locomotion/speed signal, not just onsets.
"""
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

RUN = Path('/Users/scottseneca/Desktop/pupil demo 2')
PARTS = [11, 12, 13]          # mid-session, spans the active stretch
FPS = 15.0

d = np.load(RUN / 'proc' / 'pupil_demo_2_boxtraces.npz', allow_pickle=True)
names = [str(n) for n in d["box_names"]]
coords = {n: c for n, c in zip(names, d["box_coords"])}
TARGETS = ["whisker_pad", "paw", "wheel"]


def natkey(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


files = sorted(RUN.glob("*.tif"), key=natkey)
sel = [f for f in files if int(re.search(r"_(\d+)\.tif$", f.name).group(1)) in PARTS]
print(f"reading {len(sel)} parts: {[f.name for f in sel]}")

metrics = {t: {k: [] for k in ("L1", "L2", "RMS", "AREA")} for t in TARGETS}
prev = {t: None for t in TARGETS}
nfr = 0
for f in sel:
    with tifffile.TiffFile(f) as tf:
        for page in tf.pages:
            fr = page.asarray()
            nfr += 1
            for t in TARGETS:
                x0, y0, x1, y1 = coords[t]
                c = fr[y0:y1, x0:x1].astype(np.float32)
                p = prev[t]
                if p is not None:
                    dif = c - p
                    ad = np.abs(dif)
                    metrics[t]["L1"].append(ad.mean())
                    metrics[t]["L2"].append((dif * dif).mean())
                    metrics[t]["RMS"].append(np.sqrt((dif * dif).mean()))
                    metrics[t]["AREA"].append((ad > 10).mean())
                prev[t] = c
print(f"{nfr} frames, {nfr/FPS:.1f} s")


def gmm2(x, it=400):
    mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
    sd = np.array([x.std() / 2] * 2); pi = np.array([.5, .5])
    for _ in range(it):
        r = np.stack([pi[k] * np.exp(-.5 * ((x - mu[k]) / sd[k]) ** 2) / sd[k]
                      for k in range(2)])
        r /= r.sum(0, keepdims=True) + 1e-300
        N = r.sum(1); pi = N / len(x); mu = (r * x).sum(1) / N
        sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(1) / N) + 1e-6
    o = np.argsort(mu); mu, sd, pi = mu[o], sd[o], pi[o]
    return (mu[1] - mu[0]) / np.sqrt(.5 * (sd[0] ** 2 + sd[1] ** 2)), pi[1]


def env(y, s=0.33):
    w = max(1, int(round(s * FPS)))
    return np.convolve(y, np.ones(w) / w, mode="same")


print()
print("=" * 74)
print("d' (bout vs baseline) after log + 330 ms envelope — higher is better")
print("=" * 74)
print(f"{'channel':13s} {'L1 mean|dF|':>13} {'L2 mean dF^2':>14} "
      f"{'RMS':>9} {'AREA frac':>11}")
best = {}
for t in TARGETS:
    row, vals = [], {}
    for k in ("L1", "L2", "RMS", "AREA"):
        v = np.asarray(metrics[t][k], dtype=float)
        dp, _ = gmm2(env(np.log(v + 1e-6)))
        vals[k] = dp
        row.append(f"{dp:.2f}")
    best[t] = max(vals, key=vals.get)
    print(f"{t:13s} {row[0]:>13} {row[1]:>14} {row[2]:>9} {row[3]:>11}   "
          f"best: {best[t]}")

print()
print("=" * 74)
print("Are L1 and L2 actually different information, or a monotone relabel?")
print("=" * 74)
for t in TARGETS:
    a = np.asarray(metrics[t]["L1"]); b = np.asarray(metrics[t]["L2"])
    from scipy.stats import spearmanr
    rho = spearmanr(a, b).statistic
    r = np.corrcoef(np.log(a + 1e-6), np.log(b + 1e-6))[0, 1]
    # slope in log-log: L2 ~ L1^k.  k=2 exactly => pure relabel, no new info
    k = np.polyfit(np.log(a + 1e-6), np.log(b + 1e-6), 1)[0]
    print(f"  {t:13s} spearman {rho:.4f}   log-log r {r:.4f}   "
          f"L2 ~ L1^{k:.2f}")

print()
print("=" * 74)
print("SATURATION — does the metric compress at high speed?")
print("=" * 74)
print("  If ME is linear in speed, paw vs wheel should stay proportional at")
print("  high amplitude. Compression shows as a falling ratio in the top bins.")
pa = np.asarray(metrics["paw"]["L1"]); wh = np.asarray(metrics["wheel"]["L1"])
qs = np.percentile(pa, [0, 50, 75, 90, 95, 99, 100])
for i in range(len(qs) - 1):
    m = (pa >= qs[i]) & (pa < qs[i + 1] + (1e9 if i == len(qs) - 2 else 0))
    if m.sum() > 5:
        print(f"  paw p{[0,50,75,90,95,99][i]:>3}-{[50,75,90,95,99,100][i]:>3}: "
              f"n={m.sum():4d}  paw {pa[m].mean():7.1f}  wheel {wh[m].mean():7.1f}  "
              f"ratio {wh[m].mean()/max(pa[m].mean(),1e-9):.3f}")

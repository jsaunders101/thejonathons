"""Characterise the whisker_pad ME signal before designing an onset detector."""
import numpy as np

FPS = 15.0
d = np.load('/Users/scottseneca/Desktop/pupil demo 2/proc/pupil_demo_2_boxtraces.npz',
            allow_pickle=True)
names = [str(n) for n in d["box_names"]]
ME = d["motion_energy"][:, 1:]            # drop the definitional 0
idx = {n: i for i, n in enumerate(names)}
w = ME[idx["whisker_pad"]].astype(float)
paw_nose = ME[idx["paw_at_nose"]].astype(float)
paw = ME[idx["paw"]].astype(float)
wheel = ME[idx["wheel"]].astype(float)
laser = ME[idx["laser_trigger"]].astype(float)

print("=" * 70)
print("1. SPECTRUM — is whisking resolved or aliased at 15 fps?")
print("=" * 70)
x = w - w.mean()
f = np.fft.rfftfreq(len(x), 1 / FPS)
P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
# smooth the periodogram for readability
k = 51
Ps = np.convolve(P, np.ones(k) / k, mode="same")
print(f"  Nyquist = {FPS/2:.2f} Hz;  mouse exploratory whisking is 8-12 Hz")
band = [(0.0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 7.5)]
tot = Ps[1:].sum()
for lo, hi in band:
    m = (f >= lo) & (f < hi)
    print(f"    {lo:4.1f}-{hi:4.1f} Hz : {100*Ps[m].sum()/tot:5.1f}% of power")
pk = f[1:][np.argmax(Ps[1:])]
print(f"  peak frequency: {pk:.2f} Hz")
print("  (a resolved 8-12 Hz rhythm is IMPOSSIBLE here — it folds below 7.5 Hz)")

print()
print("=" * 70)
print("2. DISTRIBUTION — is log(ME) bimodal (rest vs whisking)?")
print("=" * 70)
lw = np.log(w + 1e-3)
qs = np.percentile(lw, [1, 5, 10, 25, 50, 75, 90, 99])
print("  log-ME percentiles 1/5/10/25/50/75/90/99:")
print("   ", np.round(qs, 2).tolist())
# 2-component GMM, hand-rolled EM (no sklearn dependency in the probe)
mu = np.array([np.percentile(lw, 20), np.percentile(lw, 80)])
sd = np.array([lw.std() / 2, lw.std() / 2])
pi = np.array([0.5, 0.5])
for _ in range(300):
    r = np.stack([pi[k_] * np.exp(-0.5 * ((lw - mu[k_]) / sd[k_]) ** 2) / sd[k_]
                  for k_ in range(2)])
    r /= r.sum(0, keepdims=True) + 1e-300
    Nk = r.sum(1)
    pi = Nk / len(lw)
    mu = (r * lw).sum(1) / Nk
    sd = np.sqrt((r * (lw - mu[:, None]) ** 2).sum(1) / Nk) + 1e-6
o = np.argsort(mu)
mu, sd, pi = mu[o], sd[o], pi[o]
dprime = (mu[1] - mu[0]) / np.sqrt(0.5 * (sd[0] ** 2 + sd[1] ** 2))
print(f"  GMM rest  : mean log-ME {mu[0]:.2f} (ME {np.exp(mu[0]):.1f}), sd {sd[0]:.2f}, weight {pi[0]:.2f}")
print(f"  GMM active: mean log-ME {mu[1]:.2f} (ME {np.exp(mu[1]):.1f}), sd {sd[1]:.2f}, weight {pi[1]:.2f}")
print(f"  separation d' = {dprime:.2f}   (<1.5 = the two states are not separable)")

print()
print("=" * 70)
print("3. CONTAMINATION — what else lives in the whisker_pad box?")
print("=" * 70)
for nm, other in (("paw_at_nose", paw_nose), ("paw", paw), ("wheel", wheel),
                  ("laser_trigger", laser)):
    r = np.corrcoef(np.log(w + 1e-3), np.log(other + 1e-3))[0, 1]
    print(f"  corr(log whisker_pad, log {nm:13s}) = {r:+.3f}")
hi_groom = paw_nose > np.percentile(paw_nose, 99)
print(f"  whisker_pad ME during top-1% paw_at_nose frames : "
      f"median {np.median(w[hi_groom]):.1f} vs {np.median(w[~hi_groom]):.1f} elsewhere "
      f"({np.median(w[hi_groom])/np.median(w[~hi_groom]):.1f}x)")

print()
print("=" * 70)
print("4. TIMESCALES — how long are bouts, how fast do they rise?")
print("=" * 70)
thr = np.exp((mu[0] + mu[1]) / 2)
above = w > thr
runs, cur = [], 0
for a in above:
    if a:
        cur += 1
    elif cur:
        runs.append(cur); cur = 0
if cur:
    runs.append(cur)
gaps, cur = [], 0
for a in above:
    if not a:
        cur += 1
    elif cur:
        gaps.append(cur); cur = 0
runs, gaps = np.array(runs), np.array(gaps)
print(f"  crossing threshold ME={thr:.1f} (GMM midpoint)")
print(f"  {len(runs)} suprathreshold runs; duration frames "
      f"median {np.median(runs):.0f} p90 {np.percentile(runs,90):.0f} "
      f"({np.median(runs)/FPS*1000:.0f} ms / {np.percentile(runs,90)/FPS*1000:.0f} ms)")
print(f"  {len(gaps)} gaps; median {np.median(gaps):.0f} frames "
      f"({np.median(gaps)/FPS*1000:.0f} ms)")
print(f"  fraction of 1-frame runs: {np.mean(runs == 1):.2f}  "
      f"<- flicker; a raw threshold shatters bouts")
ac = [np.corrcoef(lw[:-k_], lw[k_:])[0, 1] for k_ in range(1, 16)]
print("  log-ME autocorrelation lags 1..15 frames:")
print("   ", np.round(ac, 2).tolist())
half = next((i + 1 for i, v in enumerate(ac) if v < 0.5), None)
print(f"  autocorr drops below 0.5 at lag {half} frames "
      f"({(half or 0)/FPS*1000:.0f} ms) -> intrinsic envelope timescale")

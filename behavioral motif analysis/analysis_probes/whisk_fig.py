"""Diagnostic figure for whisking-onset design decisions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FPS = 15.0
d = np.load('/Users/scottseneca/Desktop/pupil demo 2/proc/pupil_demo_2_boxtraces.npz',
            allow_pickle=True)
names = [str(n) for n in d["box_names"]]
ME = d["motion_energy"][:, 1:].astype(float)
i = {n: k for k, n in enumerate(names)}
w, pn, laser = ME[i["whisker_pad"]], ME[i["paw_at_nose"]], ME[i["laser_trigger"]]
lw = np.log(w + 1e-3)


def gmm2(x, iters=300):
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


mu, sd, pi = gmm2(lw)
dprime = (mu[1] - mu[0]) / np.sqrt(.5 * (sd[0] ** 2 + sd[1] ** 2))

# common-mode removal: robust-ish regression of log w on log laser
ll = np.log(laser + 1e-3)
beta = np.polyfit(ll - ll.mean(), lw - lw.mean(), 1)[0]
lw_clean = lw - beta * (ll - ll.mean())
mu2, sd2, pi2 = gmm2(lw_clean)
dprime2 = (mu2[1] - mu2[0]) / np.sqrt(.5 * (sd2[0] ** 2 + sd2[1] ** 2))

fig, ax = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("whisker_pad motion energy — properties that constrain onset detection "
             "(pupil demo 2, 15 fps)", fontsize=13, fontweight="bold")

# --- spectrum
x = w - w.mean()
f = np.fft.rfftfreq(len(x), 1 / FPS)
P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
Ps = np.convolve(P, np.ones(51) / 51, mode="same")
a = ax[0, 0]
a.loglog(f[1:], Ps[1:], lw=.8, color="#1f3a68")
a.axvspan(7.5, f[-1], color="0.85", zorder=0)
a.set_title("no whisk rhythm to find — envelope only", fontsize=10, fontweight="bold")
a.set_xlabel("Hz"); a.set_ylabel("power")
a.text(.03, .06, "8–12 Hz whisking lies ABOVE\nNyquist (7.5 Hz): unrecoverable.\n"
                 "Power is 1/f, no peak.", transform=a.transAxes, fontsize=8.5,
       color="#b3202c", va="bottom")

# --- log histogram + GMM
a = ax[0, 1]
a.hist(lw, bins=120, color="0.8", density=True)
xs = np.linspace(lw.min(), lw.max(), 400)
for k, lab, c in ((0, "rest", "tab:green"), (1, "whisking", "tab:red")):
    a.plot(xs, pi[k] * np.exp(-.5 * ((xs - mu[k]) / sd[k]) ** 2) /
           (sd[k] * np.sqrt(2 * np.pi)), color=c, lw=2,
           label=f"{lab} (w={pi[k]:.2f}, ME≈{np.exp(mu[k]):.0f})")
a.axvline((mu[0] + mu[1]) / 2, color="k", ls="--", lw=1)
a.set_title(f"bout structure IS there: d' = {dprime:.2f}", fontsize=10, fontweight="bold")
a.set_xlabel("log ME"); a.legend(fontsize=8)

# --- autocorrelation
a = ax[0, 2]
lags = np.arange(1, 31)
ac = [np.corrcoef(lw[:-k], lw[k:])[0, 1] for k in lags]
a.plot(lags / FPS * 1000, ac, "o-", ms=3, lw=1, color="#1f3a68")
a.axhline(.5, color="r", ls="--", lw=1)
a.set_title("envelope timescale ≈ 600 ms", fontsize=10, fontweight="bold")
a.set_xlabel("lag (ms)"); a.set_ylabel("autocorr of log ME")
a.text(.35, .8, "smooth ~200–400 ms:\nless leaves flicker,\nmore erases bouts",
       transform=a.transAxes, fontsize=8.5, color="#b3202c")

# --- contamination
a = ax[1, 0]
others = ["paw_at_nose", "paw", "wheel", "laser_trigger", "eye"]
rs = [np.corrcoef(lw, np.log(ME[i[o]] + 1e-3))[0, 1] for o in others]
cols = ["#b3202c" if o in ("paw_at_nose", "laser_trigger") else "0.6" for o in others]
a.barh(others, rs, color=cols)
a.set_xlim(0, 1)
a.set_title("what else lives in the box", fontsize=10, fontweight="bold")
a.set_xlabel("corr(log ME) with whisker_pad")
a.text(.97, .12, "red = must be handled:\ngrooming + common mode",
       transform=a.transAxes, fontsize=8.5, color="#b3202c", ha="right")

# --- common-mode: regression HURTS; the shared variance is slow
a = ax[1, 1]
betas = np.linspace(0, 2, 21)
ds = []
for b in betas:
    m_, s_, _ = gmm2(lw - b * (ll - ll.mean()), iters=150)
    ds.append((m_[1] - m_[0]) / np.sqrt(.5 * (s_[0] ** 2 + s_[1] ** 2)))
a.plot(betas, ds, "o-", ms=3, color="#b3202c")
a.axvline(0, color="tab:green", lw=2)
a.set_title("do NOT regress out the common mode", fontsize=10, fontweight="bold")
a.set_xlabel("β (laser_trigger regressed out of whisker_pad)")
a.set_ylabel("d' rest vs whisking")
a.text(.5, .75, "separation is destroyed\nmonotonically — the shared\nvariance is real movement,\nnot artifact",
       transform=a.transAxes, fontsize=8.5, color="#b3202c")


def hp(v, k):
    return v - np.convolve(v, np.ones(k) / k, mode="same")


ins = a.inset_axes([.55, .12, .42, .34])
rr = [np.corrcoef(lw, ll)[0, 1]] + [np.corrcoef(hp(lw, k), hp(ll, k))[0, 1]
                                    for k in (15, 45, 150)]
ins.bar(range(4), rr, color=["0.5", "tab:green", "tab:green", "0.7"])
ins.set_xticks(range(4))
ins.set_xticklabels(["raw", ">1s", ">3s", ">10s"])
ins.set_title("corr after removing slow", fontsize=7)
ins.tick_params(labelsize=6)

# --- example segment
a = ax[1, 2]
s, e = 3200, 3800
t = np.arange(s, e) / FPS
a.plot(t, w[s:e], lw=.7, color="0.55", label="raw ME")
k = 5
env = np.convolve(w, np.ones(k) / k, mode="same")
a.plot(t, env[s:e], lw=1.6, color="#1f3a68", label="333 ms envelope")
a.axhline(np.exp((mu[0] + mu[1]) / 2), color="k", ls="--", lw=1, label="GMM midpoint")
groom = pn[s:e] > np.percentile(pn, 99)
a.fill_between(t, 0, a.get_ylim()[1], where=groom, color="tab:orange", alpha=.3,
               label="grooming (paw_at_nose)")
a.set_title("40 s example", fontsize=10, fontweight="bold")
a.set_xlabel("time (s)"); a.legend(fontsize=7.5, loc="upper right")

fig.tight_layout(rect=[0, 0, 1, .955])
out = "/private/tmp/claude-502/-Users-scottseneca-repos/c35fb851-9682-4d12-863f-343188aefd3a/scratchpad/whisk_diagnostics.png"
fig.savefig(out, dpi=140)
print("wrote", out)
print(f"d' raw {dprime:.3f} -> cleaned {dprime2:.3f}  (beta={beta:.3f})")

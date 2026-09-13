from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False


# =====================================================================
#  DATA
# =====================================================================
def load_sp500(start="1985-01-01", end="2025-12-31"):
    """Download S&P 500 daily adjusted close and compute log-returns."""
    if not HAVE_YF:
        raise RuntimeError("yfinance not available. Run: !pip install yfinance")
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    px = df["Close"].to_numpy().astype(float).ravel()
    dates = df.index.to_numpy()
    logret = np.diff(np.log(px))
    return dates[1:], px[1:], logret


# =====================================================================
#  D1 — LONG MEMORY IN |RETURNS| 
# =====================================================================
def d1_long_memory(logret):
    print("\n" + "="*60)
    print("  D1 — long memory in |returns| (volatility clustering)")
    print("="*60)
    ar = np.abs(logret - logret.mean())
    lags = [1, 5, 10, 20, 50, 100, 200]
    ac = []
    for L in lags:
        if L < len(ar):
            ac.append(np.corrcoef(ar[:-L], ar[L:])[0, 1])
        else:
            ac.append(np.nan)
    for L, a in zip(lags, ac):
        print(f"    |return| autocorr at lag {L:4d}: {a:.4f}")
    # slow (power-law) decay vs fast (exponential) — compare fit quality
    lags_a = np.array(lags, float); ac_a = np.array(ac)
    m = ac_a > 0
    if m.sum() >= 3:
        pl_r2 = _r2(np.log(lags_a[m]), np.log(ac_a[m]))
        ex_r2 = _r2(lags_a[m], np.log(ac_a[m]))
        verdict = "POWER-LAW (long memory)" if pl_r2 > ex_r2 else "exponential (short memory)"
        print(f"    decay shape: power-law R2={pl_r2:.3f} vs exponential R2={ex_r2:.3f} -> {verdict}")
    return lags, ac


def _r2(x, y):
    p = np.polyfit(x, y, 1); pred = np.polyval(p, x)
    ss = np.sum((y - y.mean())**2)
    return 1 - np.sum((y - pred)**2) / ss if ss > 0 else 0.0


# =====================================================================
#  D2 — POST-CRASH VOLATILITY RELAXATION (fractional vs Markovian)
# =====================================================================
CRASHES = {
    "1987 Black Monday": "1987-10-19",
    "2000 dot-com":      "2000-09-01",
    "2008 GFC":          "2008-09-15",
    "2020 COVID":        "2020-02-20",
    "2022 bear":         "2022-01-03",
}


def d2_post_crash_relaxation(dates, logret):
    print("\n" + "="*60)
    print("  D2 — post-crash volatility relaxation: power-law vs exponential")
    print("="*60)
    w = 10                                   # short realized-vol window
    rv = np.array([logret[i:i+w].std() for i in range(len(logret)-w)])
    rv_dates = dates[:len(rv)]
    results = {}
    for name, dstr in CRASHES.items():
        cd = np.datetime64(dstr)
        # find the crash index; skip if outside data range
        idx = np.searchsorted(rv_dates, cd)
        if idx < 50 or idx > len(rv) - 300:
            print(f"    {name}: outside data range, skipped")
            continue
        # baseline = median vol in the 250 days BEFORE the crash
        base = np.median(rv[idx-250:idx])
        post = rv[idx: idx+250]
        tt = np.arange(1, len(post)+1)
        y = np.clip(post - base, 1e-6, None)
        m = y > (0.3 * y.max())              # fit the clear decay portion
        if m.sum() < 10:
            print(f"    {name}: weak/no relaxation signal")
            continue
        pl_slope = np.polyfit(np.log(tt[m]), np.log(y[m]), 1)[0]
        pl_r2 = _r2(np.log(tt[m]), np.log(y[m]))
        ex_r2 = _r2(tt[m], np.log(y[m]))
        shape = "POWER-LAW" if pl_r2 > ex_r2 else "exponential"
        # power-law slope ~ -(1-alpha): map to an effective alpha (rough)
        print(f"    {name}: peak vol {post.max():.4f}, baseline {base:.4f}, "
              f"power-law slope={pl_slope:.2f} (R2 {pl_r2:.2f}) vs exp (R2 {ex_r2:.2f}) -> {shape}")
        results[name] = (tt, y, base, pl_slope, pl_r2, ex_r2)
    return results


# =====================================================================
#  D3 — DISTRIBUTION-SHAPE RELAXATION 
# =====================================================================
def d3_shape_relaxation(dates, logret):
    print("\n" + "="*60)
    print("  D3 — distribution-shape (kurtosis) relaxation after a shock")
    print("="*60)
    def kurt(x):
        s = x.std()
        return np.mean(((x - x.mean())/s)**4) - 3 if s > 0 else 0.0
    for name, dstr in CRASHES.items():
        cd = np.datetime64(dstr); idx = np.searchsorted(dates[:len(logret)], cd)
        if idx < 250 or idx > len(logret) - 400:
            continue
        kpre = kurt(logret[idx-250:idx])
        kpost = [kurt(logret[idx:idx+k]) for k in (50, 100, 200, 400)]
        print(f"    {name}: pre-shock kurt={kpre:.2f} -> post at 50/100/200/400d = "
              f"{[round(k,2) for k in kpost]}")


# =====================================================================
#  D4 — SINGLE-WINDOW DENSITY NOISE 
# =====================================================================
def d4_single_window_noise(logret):
    print("\n" + "="*60)
    print("  D4 — how noisy is ONE window's density estimate?")
    print("="*60)
    for w in (63, 126, 252):                 # ~quarter, half-year, year
        # split the series into non-overlapping windows, measure how much the
        # per-window std/kurtosis vary just from finite-sample noise in calm-ish data
        stds, kurts = [], []
        for i in range(0, len(logret)-w, w):
            seg = logret[i:i+w]
            stds.append(seg.std())
            s = seg.std()
            kurts.append(np.mean(((seg-seg.mean())/s)**4)-3 if s > 0 else 0)
        stds = np.array(stds); kurts = np.array(kurts)
        print(f"    window={w:3d} days ({w/252:.1f}yr): "
              f"std spread={stds.std()/stds.mean()*100:.0f}% of mean, "
              f"kurtosis spread={kurts.std():.2f}")
    print("    (large per-window spread = a single window's density is noisy;")
    print("     real data gives ONE such density per window, unlike Task 3's resampling)")


# =====================================================================
#  PLOTS
# =====================================================================
def plot_all(lags, ac, d2res, dates, px):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    # price with crashes marked
    ax[0].plot(dates, px, color="#333", lw=0.8)
    ax[0].set_yscale("log"); ax[0].set_title("S&P 500 (log scale) with crash dates")
    ax[0].set_xlabel("date"); ax[0].set_ylabel("price")
    for name, dstr in CRASHES.items():
        ax[0].axvline(np.datetime64(dstr), color="#d62728", ls=":", lw=1)
    # long-memory autocorr (log-log)
    la = np.array(lags, float); aca = np.array(ac)
    m = aca > 0
    ax[1].loglog(la[m], aca[m], "o-", color="#1f77b4")
    ax[1].set_xlabel("lag (days)"); ax[1].set_ylabel("|return| autocorrelation")
    ax[1].set_title("D1: long memory (slow decay = fractional)")
    ax[1].grid(True, which="both", alpha=0.3)
    # post-crash relaxation curves
    for name, val in d2res.items():
        tt, y, base, sl, plr2, exr2 = val
        ax[2].loglog(tt, y, lw=1.2, label=f"{name} (slope {sl:.2f})")
    ax[2].set_xlabel("days after crash"); ax[2].set_ylabel("excess volatility")
    ax[2].set_title("D2: post-crash relaxation (straight line = power-law)")
    ax[2].legend(fontsize=7); ax[2].grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()


# =====================================================================
#  ENTRY
# =====================================================================
if __name__ == "__main__":
    print("#"*62)
    print("#  FM-PINN TASK 4 — OBSERVATION MODEL DIAGNOSTIC (real S&P 500)")
    print("#"*62)
    print("  Loading S&P 500 (1985-2025)...")
    dates, px, logret = load_sp500()
    print(f"  {len(logret)} daily log-returns loaded.")

    lags, ac = d1_long_memory(logret)
    d2res = d2_post_crash_relaxation(dates, logret)
    d3_shape_relaxation(dates, logret)
    d4_single_window_noise(logret)
    plot_all(lags, ac, d2res, dates, px)

    print("\n" + "="*60)
    print("  ")
    print("="*60)
    print("  (a) p(x,t): the return distribution; best observed as it RELAXES")
    print("      after a shock, NOT as a static rolling window.")
    print("  (b) initial condition: the shocked (post-crash) return distribution.")
    print("  (c) t=0: the crash/shock date — the one real-market event that")
    print("      RESETS the system and gives a known start.")
    print("  D1 tests whether memory exists at all (slow autocorr decay).")
    print("  D2 tests whether post-crash relaxation is power-law (fractional)")
    print("     vs exponential (Markovian) — the core justification.")
    print("  D3/D4 test whether the transient is real and how noisy one window is.")
    print("  If D1+D2 show long memory and power-law relaxation, the post-shock")
    print("  observation model is defensible. If not, the honest conclusion is")
    print("  that real data lacks a usable transient — a valid, important finding.")

"""
FM-PINN — SUPERVISOR TASK 3: IDENTIFIABILITY OF THE MEMORY ORDER alpha
"""

from __future__ import annotations
import time, copy
import numpy as np
from math import gamma as Gamma
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
NP = np.float64


# =====================================================================
#  PART 1 — GROUND-TRUTH L1 REFERENCE
# =====================================================================
def l1_weights_np(alpha, n):
    if abs(alpha - 1.0) < 1e-12:
        b = np.zeros(n, dtype=NP); b[0] = 1.0; return b
    j = np.arange(n, dtype=np.float64)
    return ((j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha)).astype(NP)


def op_noflux(x, gamma, D):
    Nx = len(x); dx = x[1] - x[0]; A = np.zeros((Nx, Nx), dtype=NP)
    for i in range(Nx - 1):
        xh = 0.5 * (x[i] + x[i + 1]); a = -gamma * xh   # mean-reverting mu=-gamma*x
        cf_i = a * 0.5 + D / dx; cf_ip = a * 0.5 - D / dx
        A[i, i] += -cf_i / dx; A[i, i + 1] += -cf_ip / dx
        A[i + 1, i] += +cf_i / dx; A[i + 1, i + 1] += +cf_ip / dx
    return A


def solve_ref(alpha, gamma, D, x_min, x_max, Nx, T, Nt, s_ic):
    x = np.linspace(x_min, x_max, Nx).astype(NP); dx = x[1] - x[0]; dt = T / Nt
    p0 = np.exp(-(x**2)/(2*s_ic**2)); p0 /= dx * p0.sum()
    sigma = 1.0 / (Gamma(2.0 - alpha) * dt ** alpha)
    A = op_noflux(x, gamma, D); b = l1_weights_np(alpha, Nt)
    P = np.zeros((Nt + 1, Nx), dtype=NP); P[0] = p0
    Minv = np.linalg.inv(sigma * b[0] * np.eye(Nx, dtype=NP) - A)
    for n in range(1, Nt + 1):
        h = np.zeros(Nx, dtype=NP)
        for j in range(1, n):
            h += b[j] * (P[n - j] - P[n - j - 1])
        P[n] = Minv @ (sigma * b[0] * P[n - 1] - sigma * h)
    t = np.linspace(0.0, T, Nt + 1).astype(NP)
    return x, t, P, dx


def solve_truth_fine(alpha, gamma, D, x_min, x_max, T, s_ic,
                     Nx_coarse, Nt_coarse, Nx_fine=601, Nt_fine=1500):
    """FIX #1 (inverse crime): generate the 'true' data on a FINE grid
       (601 x 1500), then SUBSAMPLE onto the coarse recovery grid. Data and
       recovery no longer share a discretisation, so recovery is no longer the
       artificially easy task of inverting the exact scheme that made the data."""
    xf = np.linspace(x_min, x_max, Nx_fine).astype(NP)
    _, _, Pf, _ = solve_ref(alpha, gamma, D, x_min, x_max, Nx_fine, T, Nt_fine, s_ic)
    xc = np.linspace(x_min, x_max, Nx_coarse).astype(NP); dxc = xc[1] - xc[0]
    tc = np.linspace(0.0, T, Nt_coarse + 1).astype(NP)
    xi = np.clip(np.searchsorted(xf, xc), 0, Nx_fine - 1)
    ti = np.clip(np.round(tc * Nt_fine / T).astype(int), 0, Nt_fine)
    Pc = Pf[ti][:, xi].copy()
    Pc = Pc / (dxc * Pc.sum(axis=1, keepdims=True))
    return xc, tc, Pc, dxc


def sample_noise(P, x, dx, N_samples, rng):
    """
    Build a NOISY observed density: at each time, draw N_samples from the true
    density p(x,t) and rebuild a histogram-like density (via KDE-free binning
    onto the same grid, then renormalize). This mimics a real market histogram
    from a finite number of returns. Returns a density array like P.
    """
    Nt1, Nx = P.shape
    P_obs = np.zeros_like(P)
    for it in range(Nt1):
        pdf = np.clip(P[it], 0, None); pdf = pdf / (dx * pdf.sum())
        cdf = np.cumsum(pdf) * dx; cdf = cdf / cdf[-1]
        u = rng.random(N_samples)
        draws = np.interp(u, cdf, x)                 # inverse-CDF sampling
        counts, _ = np.histogram(draws, bins=np.concatenate(
            [x - dx/2, [x[-1] + dx/2]]))
        dens = counts.astype(NP)
        s = dx * dens.sum()
        P_obs[it] = dens / s if s > 0 else pdf
    return P_obs


# =====================================================================
#  PART 2 — fPINN RECOVERY ENGINE (Pang-Lu method, time-Caputo)
# =====================================================================
class EnergyNet(nn.Module):
    """Energy-based density p=exp(f)/Z, Z by dx*sum on the working grid
       (guarantees p>=0 and unit mass). Plain tanh MLP energy."""
    def __init__(self, hidden=48, layers=3):
        super().__init__()
        seq = []; prev = 2
        for _ in range(layers):
            seq += [nn.Linear(prev, hidden), nn.Tanh()]; prev = hidden
        seq.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def energy_grid(self, X, T, shape):
        return self.net(torch.cat([X, T], 1)).view(*shape)


def recover(P_obs, x, t, dx, gamma_true, D_true,
            recover_gamma_D=False, snapshot_only=False,
            n_iter=4000, lr=5e-3, seed=0, device='cpu', verbose=False):
    """
    Recover alpha (and optionally gamma, D) from an observed density P_obs by
    the Pang-Lu fPINN: trainable alpha as a parameter, hybrid residual
    (L1 discretization for the Caputo time term, finite-diff for the integer
    space operator), plus a data-fit term to the observed density.

    snapshot_only=True: the data-fit term uses only the FINAL time slice
      (tests whether one snapshot carries enough info for alpha).
    recover_gamma_D=True: gamma and D are also trainable (confounding test).
    Returns dict with recovered alpha (+gamma,D if applicable).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    Nx = len(x); Nt = len(t) - 1
    xg = torch.tensor(x, device=device)
    tg = torch.tensor(t, device=device)
    Pdata = torch.tensor(P_obs, device=device)
    X = xg.view(1, Nx).expand(Nt + 1, Nx).reshape(-1, 1)
    Tt = tg.view(Nt + 1, 1).expand(Nt + 1, Nx).reshape(-1, 1)
    xin = xg[1:-1].view(1, -1)

    net = EnergyNet().to(device)
    alpha_raw = nn.Parameter(torch.tensor(0.0, device=device))   # sigmoid(0)=0.5
    def get_alpha(): return 0.05 + 0.9 * torch.sigmoid(alpha_raw)

    params = list(net.parameters()) + [alpha_raw]
    if recover_gamma_D:
        # FIX #2: initialise gamma, D from RANDOM/PERTURBED values, NOT the true
        # answer. Starting on the truth makes the joint-recovery claim untestable.
        # We draw gamma0 in [0.5, 2.0]*true, D0 in [0.5, 2.0]*true per seed.
        rng_init = np.random.default_rng(10_000 + seed)
        gamma0 = gamma_true * rng_init.uniform(0.5, 2.0)
        D0 = D_true * rng_init.uniform(0.5, 2.0)
        gamma_raw = nn.Parameter(torch.tensor(np.log(gamma0), device=device))
        D_raw = nn.Parameter(torch.tensor(np.log(D0), device=device))
        params += [gamma_raw, D_raw]
        def get_gamma(): return torch.exp(gamma_raw)
        def get_D(): return torch.exp(D_raw)
    else:
        def get_gamma(): return torch.tensor(gamma_true, device=device)
        def get_D(): return torch.tensor(D_true, device=device)

    opt = torch.optim.Adam(params, lr=lr)
    dtt = 1.0 / Nt
    idx = (torch.arange(Nt, device=device).view(Nt, 1)
           - torch.arange(Nt, device=device).view(1, Nt))
    mask = (idx >= 0)

    for it in range(1, n_iter + 1):
        f = net.energy_grid(X, Tt, (Nt + 1, Nx))
        logZ = torch.logsumexp(f + np.log(dx), 1, keepdim=True)
        P = torch.exp(f - logZ)                       # (Nt+1, Nx)
        a = get_alpha(); g = get_gamma(); Dv = get_D()

        # L1 Caputo weights (differentiable in alpha)
        jj = torch.arange(Nt + 1, dtype=torch.float64, device=device)
        bw = (jj + 1.0) ** (1.0 - a) - jj ** (1.0 - a)
        sigma = 1.0 / (torch.exp(torch.lgamma(2 - a)) * dtt ** a)

        px = (P[:, 2:] - P[:, :-2]) / (2 * dx)
        pxx = (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) / dx**2
        Lp = g * (P[:, 1:-1] + xin * px) + Dv * pxx
        dP = P[1:, 1:-1] - P[:-1, 1:-1]
        W = torch.where(mask, bw[idx.clamp(min=0)], torch.zeros_like(bw[0]))
        cap = sigma * (W @ dP)
        l_pde = ((cap - Lp[1:]) ** 2).mean()

        if snapshot_only:
            l_data = ((P[-1] - Pdata[-1]) ** 2).mean()   # final slice only
        else:
            l_data = ((P - Pdata) ** 2).mean()           # full transient

        loss = l_pde + 100.0 * l_data
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (it % 1000 == 0):
            print(f"      it{it}: alpha={get_alpha().item():.4f} "
                  f"loss={loss.item():.2e}")

    out = {'alpha': float(get_alpha().item())}
    if recover_gamma_D:
        out['gamma'] = float(get_gamma().item())
        out['D'] = float(get_D().item())
    return out


# =====================================================================
#  PART 3 — EXPERIMENT PARTS
# =====================================================================
GAMMA, D, XMIN, XMAX, T_END, S_IC = 1.0, 0.02, -0.6, 0.6, 1.0, 0.07
NX, NT = 101, 100
N_SEEDS = 5
N_ITER = 4000


def gen_truth(alpha):
    # FIX #1: fine-grid truth subsampled to the coarse recovery grid (no inverse crime)
    x, t, P, dx = solve_truth_fine(alpha, GAMMA, D, XMIN, XMAX, T_END, S_IC, NX, NT)
    return x, t, P, dx


def multi_seed_recover(P_obs, x, t, dx, **kw):
    vals = {}
    for s in range(N_SEEDS):
        r = recover(P_obs, x, t, dx, GAMMA, D, seed=s, n_iter=N_ITER, **kw)
        for k, v in r.items():
            vals.setdefault(k, []).append(v)
    return {k: (np.mean(v), np.std(v)) for k, v in vals.items()}


def part_A_ideal():
    print("\n" + "="*60 + "\n  PART A — ideal recovery (full transient, clean data)\n" + "="*60)
    rows = []
    for ta in (0.6, 0.7, 0.8, 0.9):
        x, t, P, dx = gen_truth(ta)
        res = multi_seed_recover(P, x, t, dx)
        m, sd = res['alpha']
        print(f"  true alpha={ta}: recovered {m:.4f} +/- {sd:.4f}  (err {abs(m-ta):.4f})")
        rows.append((ta, m, sd))
    return rows


def part_B_noise():
    print("\n" + "="*60 + "\n  PART B — sample-noise effect (histogram from N returns)\n" + "="*60)
    ta = 0.7
    x, t, P, dx = gen_truth(ta)
    rows = []
    for N in (250, 1000, 5000):
        rng = np.random.default_rng(123)
        P_obs = sample_noise(P, x, dx, N, rng)
        res = multi_seed_recover(P_obs, x, t, dx)
        m, sd = res['alpha']
        print(f"  N={N:5d} returns: recovered alpha={m:.4f} +/- {sd:.4f}  "
              f"(true {ta}, err {abs(m-ta):.4f})")
        rows.append((N, m, sd))
    return rows


def part_C_snapshot():
    print("\n" + "="*60 + "\n  PART C — single snapshot vs full transient\n" + "="*60)
    ta = 0.7
    x, t, P, dx = gen_truth(ta)
    res_full = multi_seed_recover(P, x, t, dx, snapshot_only=False)
    res_snap = multi_seed_recover(P, x, t, dx, snapshot_only=True)
    mf, sf = res_full['alpha']; ms, ss = res_snap['alpha']
    print(f"  FULL transient : recovered alpha={mf:.4f} +/- {sf:.4f}  (err {abs(mf-ta):.4f})")
    print(f"  SINGLE snapshot: recovered alpha={ms:.4f} +/- {ss:.4f}  (err {abs(ms-ta):.4f})")
    print("  (supervisor's prediction: single snapshot carries almost no alpha info)")
    return (mf, sf), (ms, ss)


def part_D_confound():
    print("\n" + "="*60 + "\n  PART D — confounding: recover alpha, gamma, D jointly\n"
          "  (fix #2: gamma,D start from RANDOM perturbed values, NOT the truth;\n"
          "   each seed uses a different random start -> spread reflects real\n"
          "   sensitivity to initialisation, not just optimisation noise)\n" + "="*60)
    ta = 0.7
    x, t, P, dx = gen_truth(ta)
    res = multi_seed_recover(P, x, t, dx, recover_gamma_D=True)
    am, asd = res['alpha']; gm, gsd = res['gamma']; dm, dsd = res['D']
    print(f"  recovered alpha={am:.4f} +/- {asd:.4f}  (true {ta})")
    print(f"  recovered gamma={gm:.4f} +/- {gsd:.4f}  (true {GAMMA})")
    print(f"  recovered D    ={dm:.5f} +/- {dsd:.5f}  (true {D})")
    print("  HONEST reading: alpha's bias/spread vs gamma's/D's tells which parameters")
    print("  are robustly identifiable. Larger gamma/D spread from random starts =>")
    print("  gamma,D are the confounded pair; alpha is comparatively robust.")
    return res


def plot_summary(A, B, C, D):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    # A: recovered vs true
    ta = [r[0] for r in A]; rm = [r[1] for r in A]; rs = [r[2] for r in A]
    ax[0].errorbar(ta, rm, yerr=rs, fmt='o-', color='#3182bd', capsize=4, label='recovered')
    ax[0].plot([0.55,0.95],[0.55,0.95],'k:',label='perfect')
    ax[0].set_xlabel('true alpha'); ax[0].set_ylabel('recovered alpha')
    ax[0].set_title('Part A: ideal recovery'); ax[0].legend(); ax[0].grid(alpha=0.3)
    # B: error vs sample size
    Ns = [r[0] for r in B]; err = [abs(r[1]-0.7) for r in B]; es = [r[2] for r in B]
    ax[1].errorbar(Ns, err, yerr=es, fmt='s-', color='#e6550d', capsize=4)
    ax[1].set_xscale('log'); ax[1].set_xlabel('N returns (sample size)')
    ax[1].set_ylabel('|recovered - true| alpha')
    ax[1].set_title('Part B: noise/sample-size effect'); ax[1].grid(alpha=0.3)
    # C: full vs snapshot
    (mf,sf),(ms,ss)=C
    ax[2].bar(['full\ntransient','single\nsnapshot'],[abs(mf-0.7),abs(ms-0.7)],
              yerr=[sf,ss],color=['#31a354','#de2d26'],capsize=5)
    ax[2].set_ylabel('|recovered - true| alpha')
    ax[2].set_title('Part C: transient vs snapshot'); ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# =====================================================================
#  PART 4 — ENTRY
# =====================================================================
if __name__ == "__main__":
    print("#"*62)
    print("#  FM-PINN TASK 3 — IDENTIFIABILITY OF alpha")
    print(f"#  {N_SEEDS} seeds/case, {N_ITER} iters, grid {NX}x{NT}, domain [{XMIN},{XMAX}]")
    print("#  recovery engine = Pang-Lu fPINN (time-Caputo), our validated solver")
    print("#"*62)
    t0 = time.time()
    A = part_A_ideal()
    B = part_B_noise()
    C = part_C_snapshot()
    Dr = part_D_confound()
    plot_summary(A, B, C, Dr if False else C)  # C reused; D printed above
    print(f"\n  total wall {time.time()-t0:.1f}s")
    print("\n  READ THE RESULTS HONESTLY:")
    print("  - Part A near the diagonal => method CAN recover alpha in ideal case.")
    print("  - Part B error growing as N shrinks => noise limits identifiability.")
    print("  - Part C snapshot error >> full error => alpha lives in the transient.")
    print("  - Part D biased alpha or huge spread => alpha/gamma/D confounded.")
    print("  A NEGATIVE result (alpha not recoverable under realistic noise) is a")
    print(" ")

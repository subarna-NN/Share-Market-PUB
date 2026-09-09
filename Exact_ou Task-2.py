from __future__ import annotations
import math, time, copy
import numpy as np
from math import gamma as Gamma
from numpy.polynomial.legendre import leggauss
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DTYPE = 'float64'
torch.set_default_dtype(torch.float64 if DTYPE == 'float64' else torch.float32)
NP = np.float64 if DTYPE == 'float64' else np.float32


def l1_weights(alpha, n):
    if abs(alpha - 1.0) < 1e-12:
        b = np.zeros(n, dtype=NP)
        b[0] = 1.0
        return b
    j = np.arange(n, dtype=np.float64)
    return ((j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha)).astype(NP)


def op_noflux(x, gamma, D):
    Nx = len(x); dx = x[1] - x[0]; A = np.zeros((Nx, Nx), dtype=NP)
    for i in range(Nx - 1):
        xh = 0.5 * (x[i] + x[i + 1]); a = -gamma * xh
        cf_i = a * 0.5 + D / dx
        cf_ip = a * 0.5 - D / dx
        A[i, i]     += -cf_i / dx
        A[i, i + 1] += -cf_ip / dx
        A[i + 1, i]     += +cf_i / dx
        A[i + 1, i + 1] += +cf_ip / dx
    return A


def normalize_dxsum(p, dx):
    return p / (dx * p.sum())


def gaussian(x, s):
    return np.exp(-(x ** 2) / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))


def solve_ref(alpha, gamma, D, x_min, x_max, Nx, T, Nt, s_ic):
    x = np.linspace(x_min, x_max, Nx).astype(NP); dx = x[1] - x[0]; dt = T / Nt
    p0 = normalize_dxsum(gaussian(x, s_ic), dx)
    sigma = 1.0 / (Gamma(2.0 - alpha) * dt ** alpha)
    A = op_noflux(x, gamma, D); b = l1_weights(alpha, Nt)
    P = np.zeros((Nt + 1, Nx), dtype=NP); P[0] = p0
    Minv = np.linalg.inv(sigma * b[0] * np.eye(Nx, dtype=NP) - A)
    for n in range(1, Nt + 1):
        h = np.zeros(Nx, dtype=NP)
        for j in range(1, n):
            h += b[j] * (P[n - j] - P[n - j - 1])
        P[n] = Minv @ (sigma * b[0] * P[n - 1] - sigma * h)
    t = np.linspace(0.0, T, Nt + 1).astype(NP)
    return t, x, P, dx


def grid_convergence_report(alpha, gamma, D, x_min, x_max, T, s_ic):
    print("  [grid-convergence of reference]")
    grids = [(101, 100), (201, 200), (401, 400)]
    finals = {}
    for Nx, Nt in grids:
        t, x, P, dx = solve_ref(alpha, gamma, D, x_min, x_max, Nx, T, Nt, s_ic)
        m0 = dx * P[0].sum(); mT = dx * P[-1].sum()
        finals[(Nx, Nt)] = (x, P[-1])
        print(f"    ({Nx:3d},{Nt:3d})  mass t0={m0:.6f} tT={mT:.6f} "
              f"drift={abs(mT-m0):.1e}  min={P.min():.2e}")
    (x1, f1), (x2, f2), (x3, f3) = (finals[g] for g in grids)
    e12 = np.sqrt(np.mean((f1 - np.interp(x1, x2, f2))**2)) / np.sqrt(np.mean(f1**2))
    e23 = np.sqrt(np.mean((f2 - np.interp(x2, x3, f3))**2)) / np.sqrt(np.mean(f2**2))
    print(f"    self rel-L2 (101 vs 201) = {e12:.3e}")
    print(f"    self rel-L2 (201 vs 401) = {e23:.3e}  (shrinking => converged)")


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_layers=4, hidden=64):
        super().__init__()
        layers = []; prev = in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(prev, hidden), nn.Tanh()]; prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1)).squeeze(-1)


class EnergyDensity(nn.Module):
    def __init__(self, x_min, x_max, n_quad=96, hidden_layers=4, hidden=64,
                 norm='grid', x_grid=None):
        super().__init__()
        self.f = MLP(2, hidden_layers, hidden)
        self.norm = norm
        nodes, weights = leggauss(n_quad)
        half = 0.5 * (x_max - x_min)
        xq = 0.5 * (x_max + x_min) + half * nodes
        wq = half * weights
        self.register_buffer("xq", torch.tensor(xq))
        self.register_buffer("wq", torch.tensor(wq))
        if x_grid is not None:
            self.register_buffer("xg_norm", x_grid.clone())
            self.dx_norm = float(x_grid[1] - x_grid[0])
        else:
            self.xg_norm = None

    def log_Z(self, t_col):
        M = t_col.shape[0]
        if self.norm == 'grid' and self.xg_norm is not None:
            xs = self.xg_norm; Q = xs.shape[0]
            xq = xs.view(1, Q).expand(M, Q).reshape(-1, 1)
            tq = t_col.view(M, 1).expand(M, Q).reshape(-1, 1)
            fq = self.f(xq, tq).view(M, Q)
            logdx = math.log(self.dx_norm)
            return torch.logsumexp(fq + logdx, dim=1)
        else:
            Q = self.xq.shape[0]
            xq = self.xq.view(1, Q).expand(M, Q).reshape(-1, 1)
            tq = t_col.view(M, 1).expand(M, Q).reshape(-1, 1)
            fq = self.f(xq, tq).view(M, Q)
            logw = torch.log(self.wq).view(1, Q)
            return torch.logsumexp(fq + logw, dim=1)

    def log_p(self, x, t):
        return self.f(x, t) - self.log_Z(t)

    def p(self, x, t):
        return torch.exp(self.log_p(x, t))


class CaputoResidual:
    def __init__(self, alpha, T, gamma, D, device, x_grid, N_hist=81):
        self.alpha = alpha; self.gamma = gamma; self.D = D
        self.device = device; self.T = T; self.N_hist = N_hist
        self.xg = x_grid
        self.dx = (x_grid[1] - x_grid[0]).item()
        self.Nx = x_grid.shape[0]
        self.th = torch.linspace(0.0, T, N_hist, device=device)
        self.dt_h = (self.th[1] - self.th[0]).item()
        self.sigma = 1.0 / (Gamma(2.0 - alpha) * self.dt_h ** alpha)
        self.b = torch.tensor(l1_weights(alpha, N_hist), device=device)
        Nh, Nx = N_hist, self.Nx
        self.Xflat = x_grid.view(1, Nx).expand(Nh, Nx).reshape(-1, 1)
        self.Tflat = self.th.view(Nh, 1).expand(Nh, Nx).reshape(-1, 1)
        self.xin = x_grid[1:-1].view(1, -1)

    def __call__(self, model, n_tc, gen):
        Nh, Nx, dx = self.N_hist, self.Nx, self.dx
        f_grid = model.f(self.Xflat, self.Tflat).view(Nh, Nx)
        logZ_row = model.log_Z(self.th.view(Nh, 1)).view(Nh, 1)
        P = torch.exp(f_grid - logZ_row)
        px = (P[:, 2:] - P[:, :-2]) / (2 * dx)
        pxx = (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) / dx**2
        Pin = P[:, 1:-1]
        Lp = self.gamma * (Pin + self.xin * px) + self.D * pxx

        dP = Pin[1:] - Pin[:-1]
        Nt = Nh - 1
        n_tc = min(n_tc, Nt)
        sel = torch.randperm(Nt, generator=gen, device=self.device)[:n_tc] + 1
        idx_n = (sel - 1).view(n_tc, 1)
        idx_m = torch.arange(Nt, device=self.device).view(1, Nt)
        k = idx_n - idx_m; mask = (k >= 0)
        Wsub = torch.where(mask, self.b[k.clamp(min=0)], torch.zeros_like(self.b[0]))
        caputo = self.sigma * (Wsub @ dP)
        r = caputo - Lp[sel]
        return (r ** 2).mean()


def noflux_bc_loss(model, t_grid, gamma, D, x_min, x_max, n_sample=64, gen=None):
    if n_sample is not None and t_grid.shape[0] > n_sample:
        if gen is not None:
            idx = torch.randperm(t_grid.shape[0], generator=gen,
                                 device=t_grid.device)[:n_sample]
        else:
            idx = torch.randperm(t_grid.shape[0], device=t_grid.device)[:n_sample]
        tb = t_grid[idx].view(-1, 1)
    else:
        tb = t_grid.view(-1, 1)
    losses = []
    for xb_val in (x_min, x_max):
        xb = torch.full_like(tb, xb_val).clone().requires_grad_(True)
        p = model.p(xb, tb)
        dp = torch.autograd.grad(p, xb, torch.ones_like(p),
                                 create_graph=True, retain_graph=True)[0].squeeze(-1)
        J = -gamma * xb.squeeze(-1) * p - D * dp
        losses.append((J ** 2).mean())
    return sum(losses)


def ic_loss(model, x_col, p0_col):
    t0 = torch.zeros_like(x_col)
    return ((model.p(x_col, t0) - p0_col.squeeze(-1)) ** 2).mean()


def train_forward(alpha=0.8, gamma=1.0, D=0.02,
                  x_min=-0.25, x_max=0.25,
                  Nx_ref=401, T=0.5, Nt_ref=400, s_ic=0.07,
                  Nx_col=101, n_tc=64,
                  n_warm=2000, n_adam=15000, n_lbfgs=500,
                  lr=2e-3, w_pde=1.0, w_ic=20.0, w_bc=10.0,
                  hidden_layers=4, hidden=64,
                  seed=0, device=None, log_every=500):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    print(f"[Exp1-v3] device={device} dtype={DTYPE} alpha={alpha} "
          f"Nx_ref={Nx_ref} Nx_col={Nx_col} n_tc={n_tc}")

    t_ref, x_ref, P_ref, dx_ref = solve_ref(alpha, gamma, D, x_min, x_max,
                                            Nx_ref, T, Nt_ref, s_ic)
    m0 = dx_ref * P_ref[0].sum(); mT = dx_ref * P_ref[-1].sum()
    print(f"  reference: mass t0={m0:.6f} tT={mT:.6f} drift={abs(mT-m0):.1e}")

    model = EnergyDensity(x_min, x_max, n_quad=128,
                          hidden_layers=hidden_layers, hidden=hidden,
                          norm='quad').to(device)

    x_col = torch.linspace(x_min, x_max, Nx_col + 2)[1:-1].view(-1, 1).to(device)
    x_grid_res = torch.linspace(x_min, x_max, Nx_col, device=device)
    t_grid = torch.tensor(t_ref, device=device)
    xc = x_col.cpu().numpy().ravel()
    p0c = normalize_dxsum(gaussian(x_ref, s_ic), dx_ref)
    p0_col = torch.tensor(np.interp(xc, x_ref, p0c), device=device).view(-1, 1)

    res = CaputoResidual(alpha, T, gamma, D, device, x_grid_res, N_hist=81)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=n_adam // 3, gamma=0.5)
    hist = {'step': [], 'total': [], 'pde': [], 'ic': [], 'bc': []}
    t0 = time.time()

    tw = time.time()
    for step in range(1, n_warm + 1):
        model.train()
        l_ic = ic_loss(model, x_col, p0_col)
        optim.zero_grad(); l_ic.backward(); optim.step()
        if step % log_every == 0 or step == 1:
            sps = (time.time() - tw) / step
            print(f"    [warm] {step:5d}/{n_warm}  ic={l_ic.item():.3e}  "
                  f"{sps*1000:.1f} ms/step  ({1/sps:.1f} it/s)")

    best_loss = float('inf'); best_state = None
    tb = time.time()
    for step in range(1, n_adam + 1):
        model.train()
        l_pde = res(model, n_tc, gen)
        l_ic = ic_loss(model, x_col, p0_col)
        l_bc = noflux_bc_loss(model, t_grid, gamma, D, x_min, x_max, n_sample=64, gen=gen)
        loss = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
        optim.zero_grad(); loss.backward(); optim.step(); sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item(); best_state = copy.deepcopy(model.state_dict())

        if step % log_every == 0 or step == 1:
            sps = (time.time() - tb) / step
            hist['step'].append(step); hist['total'].append(loss.item())
            hist['pde'].append(l_pde.item()); hist['ic'].append(l_ic.item())
            hist['bc'].append(l_bc.item())
            print(f"    {step:5d}/{n_adam}  total={loss.item():.3e}  "
                  f"pde={l_pde.item():.3e}  ic={l_ic.item():.3e}  bc={l_bc.item():.3e}  "
                  f"| {sps*1000:.1f} ms/step ({1/sps:.1f} it/s)")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  restored best Adam checkpoint (loss={best_loss:.3e})")

    if n_lbfgs > 0:
        res_full = CaputoResidual(alpha, T, gamma, D, device, x_grid_res, N_hist=161)
        opt_lb = torch.optim.LBFGS(model.parameters(), max_iter=n_lbfgs,
                                   tolerance_grad=1e-11, tolerance_change=1e-13,
                                   history_size=60, line_search_fn='strong_wolfe')
        def closure():
            opt_lb.zero_grad()
            l_pde = res_full(model, res_full.N_hist - 1, gen)
            l_ic = ic_loss(model, x_col, p0_col)
            l_bc = noflux_bc_loss(model, t_grid, gamma, D, x_min, x_max)
            loss = w_pde * l_pde + w_ic * l_ic + w_bc * l_bc
            loss.backward(); return loss
        try:
            tl = time.time()
            before = closure().item()
            opt_lb.step(closure)
            after = closure().item()
            print(f"  L-BFGS: {before:.3e} -> {after:.3e}  ({time.time()-tl:.1f}s)")
            if after > before:
                model.load_state_dict(best_state)
                print("  (L-BFGS worse; reverted to best Adam checkpoint)")
        except Exception as e:
            print(f"  L-BFGS stopped: {e}")

    print(f"  total wall {time.time()-t0:.1f}s")

    model.eval()
    Xr, Tr = np.meshgrid(x_ref, t_ref, indexing='xy')
    xq = torch.tensor(Xr.ravel(), device=device).view(-1, 1)
    tq = torch.tensor(Tr.ravel(), device=device).view(-1, 1)
    with torch.no_grad():
        outs = []
        B = 20000
        for i in range(0, xq.shape[0], B):
            outs.append(model.p(xq[i:i+B], tq[i:i+B]).cpu().numpy())
        p_pred = np.concatenate(outs).reshape(Tr.shape)

    err = p_pred - P_ref
    rmse = float(np.sqrt((err ** 2).mean()))
    linf = float(np.abs(err).max())
    rel_l2 = rmse / float(np.sqrt((P_ref ** 2).mean()))
    mass_pred = dx_ref * p_pred.sum(axis=1)

    gln, glw = leggauss(401); half = 0.5 * (x_max - x_min)
    xgl = torch.tensor(0.5*(x_max+x_min) + half*gln, device=device).view(-1, 1)
    wgl = half * glw
    mass_gl = []
    with torch.no_grad():
        for tt in t_ref:
            tcol = torch.full((xgl.shape[0], 1), float(tt), device=device)
            pgl = model.p(xgl, tcol).cpu().numpy()
            mass_gl.append((wgl * pgl).sum())
    mass_gl = np.array(mass_gl)

    print(f"  RMSE={rmse:.4e}  Linf={linf:.4e}  rel-L2={rel_l2:.4e}")
    print(f"  mass dx*sum(401):  min={mass_pred.min():.6f} max={mass_pred.max():.6f}")
    print(f"  mass GL-401 (accurate): min={mass_gl.min():.6f} max={mass_gl.max():.6f}")

    p_exact = None
    if abs(alpha - 1.0) < 1e-12:
        m0 = 0.0
        v0 = s_ic ** 2
        def ou_mean(t): return m0 * np.exp(-gamma * t)
        def ou_var(t):  return v0*np.exp(-2*gamma*t) + (D/gamma)*(1-np.exp(-2*gamma*t))
        p_exact = np.zeros_like(P_ref)
        for it, tt in enumerate(t_ref):
            v = ou_var(tt); mmean = ou_mean(tt)
            p_exact[it] = np.exp(-(x_ref - mmean)**2/(2*v)) / np.sqrt(2*np.pi*v)
        err_ex = p_pred - p_exact
        rmse_ex = float(np.sqrt((err_ex**2).mean()))
        linf_ex = float(np.abs(err_ex).max())
        rel_ex = rmse_ex / float(np.sqrt((p_exact**2).mean()))
        rel_ref_ex = float(np.sqrt(((P_ref - p_exact)**2).mean())) / \
                     float(np.sqrt((p_exact**2).mean()))
        print(f"  --- EXACT OU validation (alpha=1) ---")
        print(f"  PINN   vs EXACT:  rel-L2={rel_ex:.4e}  RMSE={rmse_ex:.4e}  Linf={linf_ex:.4e}")
        print(f"  solver vs EXACT:  rel-L2={rel_ref_ex:.4e}  (reference sanity check)")

    return model, hist, dict(t=t_ref, x=x_ref, P_ref=P_ref, P_pred=p_pred,
                             rmse=rmse, linf=linf, rel_l2=rel_l2, alpha=alpha,
                             mass_pred=mass_pred, mass_gl=mass_gl, p_exact=p_exact)


def plot_all(hist, ev):
    a = ev['alpha']
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.semilogy(hist['step'], hist['total'], color='#111111', lw=2, label='total')
    ax.semilogy(hist['step'], hist['pde'], color='#e6550d', lw=1.4, label='PDE')
    ax.semilogy(hist['step'], hist['ic'], color='#3182bd', lw=1.4, label='IC')
    ax.semilogy(hist['step'], hist['bc'], color='#31a354', lw=1.4, label='no-flux BC')
    ax.set_xlabel('step'); ax.set_ylabel('loss (log)')
    ax.set_title(f'FM-PINN Exp.1 (v3, {DTYPE}) loss, alpha={a}')
    ax.grid(True, which='both', alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()

    t, x = ev['t'], ev['x']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    im0 = ax[0].imshow(ev['P_ref'].T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], x[0], x[-1]], cmap='viridis')
    ax[0].set_title(f'No-flux L1 reference, alpha={a}')
    ax[0].set_xlabel('t'); ax[0].set_ylabel('x'); plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(ev['P_pred'].T, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], x[0], x[-1]], cmap='viridis')
    ax[1].set_title('FM-PINN prediction'); ax[1].set_xlabel('t'); ax[1].set_ylabel('x')
    plt.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(np.abs(ev['P_pred'] - ev['P_ref']).T, aspect='auto',
                       origin='lower', extent=[t[0], t[-1], x[0], x[-1]], cmap='magma')
    ax[2].set_title(f"|error|  rel-L2={ev['rel_l2']:.2e}")
    ax[2].set_xlabel('t'); ax[2].set_ylabel('x'); plt.colorbar(im2, ax=ax[2])
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
    has_exact = ev.get('p_exact') is not None
    for frac, c in [(0.0, '#3182bd'), (0.5, '#e6550d'), (1.0, '#31a354')]:
        it = int(frac * (len(t) - 1))
        ax[0].plot(x, ev['P_ref'][it], color=c, lw=2, label=f't={t[it]:.2f}')
        ax[0].plot(x, ev['P_pred'][it], color=c, lw=1.1, ls='--')
        if has_exact:
            ax[0].plot(x, ev['p_exact'][it], color='k', lw=0.8, ls=':')
    lbl = 'solid ref, dashed PINN' + (', dotted EXACT' if has_exact else '')
    ax[0].set_xlabel('x'); ax[0].set_ylabel('p(x,t)')
    ax[0].set_title(f'Density snapshots ({lbl}), alpha={a}')
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[1].plot(t, ev['mass_gl'], color='#756bb1', lw=2, label='GL-401 (accurate)')
    ax[1].plot(t, ev['mass_pred'], color='#bcbddc', lw=1.2, ls='--', label='dx*sum(401)')
    ax[1].axhline(1.0, color='k', ls=':', lw=1)
    ax[1].set_xlabel('t'); ax[1].set_ylabel('predicted total mass')
    ax[1].set_title('Mass conservation (should stay ~1)')
    ax[1].legend(); ax[1].grid(True, alpha=0.3); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    ALPHA = 1.0
    XMIN, XMAX = -0.6, 0.6
    T_END = 1.0

    print("\n" + "#" * 62)
    print(f"#  TASK 2 — EXACT OU VALIDATION, alpha={ALPHA}  dtype={DTYPE}")
    print(f"#  domain [{XMIN},{XMAX}]  T={T_END}  (mean-reverting drift)")
    print("#" * 62)

    grid_convergence_report(ALPHA, 1.0, 0.02, XMIN, XMAX, T_END, 0.07)

    model, hist, ev = train_forward(
        alpha=ALPHA, gamma=1.0, D=0.02,
        x_min=XMIN, x_max=XMAX,
        Nx_ref=401, T=T_END, Nt_ref=400, s_ic=0.07,
        Nx_col=101, n_tc=64,
        n_warm=2000, n_adam=8000, n_lbfgs=500,
        lr=2e-3, w_pde=1.0, w_ic=20.0, w_bc=10.0,
        hidden_layers=4, hidden=64, seed=0, log_every=1000,
    )
    plot_all(hist, ev)

    print("\n" + "=" * 60)
    print(f"  TASK 2 COMPLETE — alpha={ALPHA} (exact OU validation)")
    print(f"  PINN vs numerical reference:  rel-L2 = {ev['rel_l2']:.3e}")
    if ev.get('p_exact') is not None:
        err_ex = ev['P_pred'] - ev['p_exact']
        rmse_ex = float(np.sqrt((err_ex**2).mean()))
        rel_ex = rmse_ex / float(np.sqrt((ev['p_exact']**2).mean()))
        rel_ref_ex = float(np.sqrt(((ev['P_ref']-ev['p_exact'])**2).mean())) / \
                     float(np.sqrt((ev['p_exact']**2).mean()))
        print(f"  PINN vs EXACT OU (decisive):  rel-L2 = {rel_ex:.3e}  RMSE = {rmse_ex:.3e}")
        print(f"  solver vs EXACT OU (sanity):  rel-L2 = {rel_ref_ex:.3e}")
    print(f"  mass GL-401 (accurate): {ev['mass_gl'].min():.5f} .. {ev['mass_gl'].max():.5f}")
    print("=" * 60)
    print("  Success = PINN matches the CLOSED-FORM OU solution to <~1%.")
    print("  This is accuracy against a true solution, not consistency between solvers.")

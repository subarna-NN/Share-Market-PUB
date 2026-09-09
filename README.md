# FM-PINN: A Fractional-Memory Physics-Informed Neural Network for the Time-Fractional Fokker–Planck Equation

This repository contains the code and results for the forward-validation and identifiability study of FM-PINN, a physics-informed neural network that solves and inverts a time-fractional Fokker–Planck equation with tunable "memory."

# 1. What FM-PINN is
FM-PINN learns the probability density p(x, t) of a quantity x (e.g. a log-return) as it evolves in time under a time-fractional Fokker–Planck equation:
C_D_t^α p(x,t) = -∂/∂x[ μ(x) p ] + D ∂²/∂x² p,     μ(x) = -γ x
C_D_t^α is the Caputo time-fractional derivative of order α ∈ (0,1]. It carries memory: the rate of change at time t depends on the whole past, not just the present.
α = 1 → ordinary (memoryless) dynamics; α = 1 is exactly the classical Ornstein–Uhlenbeck process.
α < 1 → sub-diffusive memory: the system "remembers" its history.
μ(x) = -γ x is mean-reverting drift (values pulled back toward 0); D is diffusion.

Key design choices (each with a scientific reason):

Energy-based density: p = exp(f(x,t)) / Z(t), where f is a neural network and Z(t) normalizes it. This guarantees p ≥ 0 and ∫ p dx = 1 by construction — the output is always a valid probability density.
Gauss–Legendre normalization: Z(t) is computed by spectrally-accurate quadrature, giving mass = 1.0 to machine precision even for sharp densities.
Pang–Lu fPINN hybrid residual: the fractional time term is discretized with the L1 scheme; the integer-order space terms use automatic differentiation. This is the method of Pang, Lu & Karniadakis (fPINNs, SIAM J. Sci. Comput., 2019).
No-flux (reflecting) boundaries: conserve total probability on the truncated domain.

# 2. Files
File	Purpose
fmpinn_task1_driftfix.py	Task 1 — forward validation at α = 0.6, 0.8, 0.95 (drift-sign corrected)
fmpinn_task2_exact_ou.py	Task 2 — validation against the exact Ornstein–Uhlenbeck solution (α = 1)
fmpinn_task3_identifiability.py	Task 3 — can the memory order α be recovered from data?

Environment: Python 3, PyTorch, NumPy, SciPy, Matplotlib. Run on Google Colab (T4 GPU). All computation in float64 for numerical rigor.

# 3. Task 1 — Forward validation

What it does: Solves the fractional Fokker–Planck equation for three known memory orders and compares FM-PINN against an independent L1 finite-difference reference solver.

How the reference is trusted: a grid-convergence study (grids 101×100, 201×200, 401×400) confirms the reference is converged before it is used as ground truth.

Results (relative L2 error vs reference, exact mass = 1.0 in all cases):

α	rel-L2 error	mass
0.6	5.43 × 10⁻³	1.000000
0.8	3.39 × 10⁻³	1.000000
0.95	3.00 × 10⁻³	1.000000

# Why it's good: all errors are well under 1%, probability is conserved exactly, and the density correctly relaxes toward the mean-reverting equilibrium.

# 4. Task 2 — Validation against the exact solution

Why this matters: in Task 1 the PINN and the reference solver share the same discretization, so their agreement measures consistency, not accuracy. Task 2 removes that doubt by comparing against a closed-form solution with no solver involved.

What it does: at α = 1 the equation is exactly the Ornstein–Uhlenbeck process, whose density stays Gaussian for all time with known mean and variance:

mean(t) = m₀ e^(-γt),   var(t) = v₀ e^(-2γt) + (D/γ)(1 - e^(-2γt))

The domain is widened to ±0.6 so the density never reaches the boundaries, making the comparison against the free-space exact solution fair.

Results:

Comparison	rel-L2 error
FM-PINN vs exact OU (decisive)	2.73 × 10⁻³
numerical solver vs exact OU (sanity check)	7.82 × 10⁻⁴
accurate mass (Gauss–Legendre)	1.000000 at all t

# Why it's good: FM-PINN matches the true solution to 0.27% — genuine accuracy against a known answer, not just agreement between two solvers. The PINN's error is (correctly) slightly larger than the direct solver's, as expected.

Implementation note: the L1 weight b₀ degenerates exactly at α = 1 (becomes 0); this is handled with the correct classical limit (backward Euler).

# 5. Task 3 — Identifiability of the memory order α

The central scientific question: the forward problem alone doesn't justify a neural network (a classical solver is faster and more accurate). The method is only justified by the inverse problem — recovering an unknown α from observed densities. This task tests, with error bars (5 random seeds per case), when α can and cannot be recovered.

Tool note: DeepXDE's built-in fPINN implements only the space-fractional Laplacian (1 < α < 2); it does not support the time-fractional Caputo derivative (0 < α < 1) used here (confirmed against the library and its issue tracker). The recovery engine is therefore our own validated implementation of the Pang–Lu fPINN method, cited accordingly.

Results:

Part A — ideal recovery (clean data, full time evolution):

true α	recovered α	error
0.6	0.613 ± 0.004	0.013
0.7	0.713 ± 0.005	0.013
0.8	0.811 ± 0.005	0.011
0.9	0.914 ± 0.004	0.014

# The method recovers α accurately when data is clean.

Part B — effect of sample noise (density built as a histogram from N returns, true α = 0.7):

N returns (≈ years of daily data)	recovered α error
5,000 (≈20 yr)	0.015
1,000 (≈4 yr)	0.027
250 (≈1 yr)	0.073

# α is recoverable to a few percent with a few years of data. (Caveat: the noise model resamples the density at each time step, so the transient sees several noisy snapshots; real data provides one density per window, so this may be optimistic.)

Part C — single snapshot vs full transient (the decisive result):

Observation	recovered α error
full transient	0.013
single snapshot	0.633

# α is essentially invisible in a single snapshot. All the memory information lives in the transient — how the density relaxes from a known starting condition. This is the key finding: recovery requires observing the transient, not a static distribution.

Part D — confounding (recover α, γ, D jointly, true α = 0.7):

parameter	recovered	true
α	0.706 ± 0.005	0.7
γ	0.938 ± 0.013	1.0
D	0.0191 ± 0.0002	0.02

# α stays accurate even when γ and D are also unknown — the parameters are not badly confounded.

# 6. Overall conclusion

The three tasks together establish:

FM-PINN solves the forward problem accurately — validated against both an independent numerical solver (Task 1) and an exact closed-form solution (Task 2), with exact probability conservation.
The memory order α is identifiable from data — the method works (Part A), survives realistic sample noise (Part B), and is not destroyed by parameter confounding (Part D) — but only from the transient, not a single static snapshot (Part C).

This makes the observation model the decisive next question: whether a transient (a known initial condition and a known t = 0) can be constructed from real market data. That question determines whether the method transfers from synthetic data to real markets, and is the subject of ongoing work.

# 7. Reproducibility
Fixed random seeds throughout (Task 3 uses 5 seeds per case for error bars).
All results above are printed directly by the scripts and reproducible on a T4 GPU.
Approximate run times (T4): Task 1 ≈ 5 min per α; Task 2 ≈ 5 min; Task 3 ≈ 85 min (multi-seed × 4 parts).

Reference: G. Pang, L. Lu, G. E. Karniadakis, fPINNs: Fractional Physics-Informed Neural Networks, SIAM Journal on Scientific Computing

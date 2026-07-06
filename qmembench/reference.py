"""Exact classical references (analytic + QuTiP) for validation.

Stage A: damped Jaynes-Cummings model at zero temperature -- a qubit coupled
to a bath with Lorentzian spectral density J(w) = gamma0 * lam^2 /
(2 pi ((w0 - w)^2 + lam^2)).  Closed-form solution (Breuer & Petruccione,
"The Theory of Open Quantum Systems", section 10.1):

    rho_ee(t)  = |G(t)|^2 rho_ee(0)
    rho_eg(t)  = G(t) rho_eg(0)
    G(t)       = e^{-lam t / 2} [ cosh(d t / 2) + (lam / d) sinh(d t / 2) ]
    d          = sqrt(lam^2 - 2 gamma0 lam)

Non-Markovian regime: gamma0 > lam/2 (d imaginary -> G oscillates and
crosses zero -> information backflow).  BLP-optimal pair: antipodal
equatorial states, giving D(t) = |G(t)|.

Exact pseudomode mapping (single Lorentzian -> one damped mode):
    H_int = Omega (sigma+ a + sigma- a+),  Omega = sqrt(gamma0 lam / 2)
    collapse operator sqrt(kappa) a with kappa = 2 lam
(the factor 2: the Lorentzian half-width lam is the *amplitude* decay rate
of the mode, while the Lindblad rate kappa governs intensity decay).

Because H_int conserves excitation number and all initial states used carry
at most one excitation, a two-level (Fock cutoff N=2) pseudomode is EXACT
for Stage A.  For the Stage B spin-boson model (sigma_z coupling, counter-
rotating terms) the cutoff matters and is studied explicitly.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from .measures import KET, dm

# Explicit operator definitions to pin conventions (|1> = excited):
SM_NP = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma- = |0><1|


def _sys_qobj(rho: np.ndarray) -> qt.Qobj:
    return qt.Qobj(np.asarray(rho, dtype=complex))


# ---------------------------------------------------------------------------
# Stage A: analytic solution
# ---------------------------------------------------------------------------

def analytic_G(times: np.ndarray, gamma0: float, lam: float) -> np.ndarray:
    """Closed-form amplitude G(t); real for all regimes, handles d -> 0."""
    times = np.asarray(times, dtype=float)
    d = np.sqrt(complex(lam**2 - 2.0 * gamma0 * lam))
    if abs(d) < 1e-12:  # critical damping limit: G = e^{-lam t/2} (1 + lam t/2)
        G = np.exp(-lam * times / 2.0) * (1.0 + lam * times / 2.0)
        return G.astype(float)
    G = np.exp(-lam * times / 2.0) * (
        np.cosh(d * times / 2.0) + (lam / d) * np.sinh(d * times / 2.0)
    )
    return np.real_if_close(G, tol=1e6).real


def analytic_stageA_states(rho0: np.ndarray, times: np.ndarray,
                           gamma0: float, lam: float) -> list[np.ndarray]:
    """Exact reduced dynamics of the system qubit under the Stage A channel."""
    G = analytic_G(times, gamma0, lam)
    out = []
    for g in G:
        r = np.zeros((2, 2), dtype=complex)
        r[1, 1] = (g**2) * rho0[1, 1]
        r[0, 0] = 1.0 - r[1, 1]
        r[1, 0] = g * rho0[1, 0]
        r[0, 1] = np.conj(r[1, 0])
        out.append(r)
    return out


def stageA_params(gamma0: float, lam: float) -> tuple[float, float]:
    """Pseudomode (Omega, kappa) equivalent to the Lorentzian (gamma0, lam)."""
    return np.sqrt(gamma0 * lam / 2.0), 2.0 * lam


# ---------------------------------------------------------------------------
# Stage A: QuTiP reference on the enlarged system (x) pseudomode space
# ---------------------------------------------------------------------------

def stageA_qutip_states(rho0_sys: np.ndarray, times: np.ndarray, gamma0: float,
                        lam: float, n_fock: int = 2) -> list[np.ndarray]:
    """mesolve on system (x) single damped pseudomode; exact for Lorentzian.

    NOTE: QuTiP has no dedicated 'pseudomode solver' -- this enlarged-space
    Lindblad evolution IS the pseudomode reference (exact here because the
    interaction conserves excitation number; n_fock=2 suffices, larger values
    only confirm that).
    """
    omega, kappa = stageA_params(gamma0, lam)
    sm = qt.Qobj(SM_NP)
    a = qt.destroy(n_fock)
    H = omega * (qt.tensor(sm.dag(), a) + qt.tensor(sm, a.dag()))
    c_ops = [np.sqrt(kappa) * qt.tensor(qt.qeye(2), a)]
    rho0 = qt.tensor(_sys_qobj(rho0_sys), qt.fock_dm(n_fock, 0))
    sol = qt.mesolve(H, rho0, np.asarray(times, dtype=float), c_ops=c_ops)
    return [np.asarray(s.ptrace(0).full()) for s in sol.states]


# ---------------------------------------------------------------------------
# Stage A at finite temperature (thermal pseudomode reservoir)
# ---------------------------------------------------------------------------

def stageA_qutip_states_finiteT(rho0_sys: np.ndarray, times: np.ndarray, gamma0: float,
                                lam: float, n_th: float, n_fock: int = 2) -> list[np.ndarray]:
    """Thermal-reservoir reference: the pseudomode is damped toward mean
    occupation n_th via emission sqrt(kappa(n_th+1)) a and absorption
    sqrt(kappa n_th) a^dag.

    KEY POINT for the truncation study: unlike the T=0 case, finite T
    populates higher Fock levels, so n_fock = 2 is NO LONGER exact. Comparing
    n_fock = 2 against a converged n_fock therefore isolates the physical
    truncation error that the single-qubit circuit encoding must incur -- the
    finite-T axis feeds directly into the 'truncation fabricates memory'
    centerpiece.
    """
    omega, kappa = stageA_params(gamma0, lam)
    sm = qt.Qobj(SM_NP)
    a = qt.destroy(n_fock)
    H = omega * (qt.tensor(sm.dag(), a) + qt.tensor(sm, a.dag()))
    c_ops = [np.sqrt(kappa * (n_th + 1.0)) * qt.tensor(qt.qeye(2), a),
             np.sqrt(kappa * n_th) * qt.tensor(qt.qeye(2), a.dag())]
    rho0 = qt.tensor(_sys_qobj(rho0_sys), qt.fock_dm(n_fock, 0))
    sol = qt.mesolve(H, rho0, np.asarray(times, dtype=float), c_ops=c_ops)
    return [np.asarray(s.ptrace(0).full()) for s in sol.states]


# ---------------------------------------------------------------------------
# Stage B: spin-boson model with a structured (multi-Lorentzian) bath
# ---------------------------------------------------------------------------

def spin_boson_qutip_states(rho0_sys: np.ndarray, times: np.ndarray,
                            delta: float, eps: float,
                            modes: list[tuple[float, float, float]],
                            n_fock: int = 4) -> list[np.ndarray]:
    """mesolve reference for H_S = (delta/2) sx + (eps/2) sz coupled via
    sigma_z (x) g_k (a_k + a_k^dag) to damped pseudomodes (w_k, g_k, kappa_k).

    The structured bath is *defined* by its pseudomode decomposition (a sum
    of underdamped Lorentzians) -- no spectral-density fitting is involved,
    which keeps the benchmark honest and exactly reproducible.

    sigma_z coupling does NOT conserve excitation number, so n_fock matters:
    use n_fock >= 4 as the physical reference and n_fock = 2 to quantify the
    truncation error of the 1-qubit-per-mode circuit encoding.
    """
    sx, sz = qt.sigmax(), qt.sigmaz()
    n_modes = len(modes)
    ident = [qt.qeye(n_fock)] * n_modes

    def emb_sys(op):
        return qt.tensor(op, *ident)

    def emb_mode(op, k):
        ops = [qt.qeye(2)] + list(ident)
        ops[1 + k] = op
        return qt.tensor(*ops)

    a = qt.destroy(n_fock)
    H = 0.5 * delta * emb_sys(sx) + 0.5 * eps * emb_sys(sz)
    c_ops = []
    for k, (wk, gk, kappak) in enumerate(modes):
        H += wk * emb_mode(a.dag() * a, k)
        H += gk * emb_sys(sz) * emb_mode(a + a.dag(), k)
        c_ops.append(np.sqrt(kappak) * emb_mode(a, k))

    rho0 = qt.tensor(_sys_qobj(rho0_sys), *[qt.fock_dm(n_fock, 0)] * n_modes)
    sol = qt.mesolve(H, rho0, np.asarray(times, dtype=float), c_ops=c_ops)
    return [np.asarray(s.ptrace(0).full()) for s in sol.states]


def spin_boson_qutip_states_finiteT(rho0_sys: np.ndarray, times: np.ndarray,
                                    delta: float, eps: float,
                                    modes: list[tuple[float, float, float]],
                                    n_th: float, n_fock: int = 4) -> list[np.ndarray]:
    """Finite-temperature spin-boson reference: as spin_boson_qutip_states, but
    each damped pseudomode relaxes toward a thermal occupation n_th through
    emission sqrt(kappa(n_th+1)) a and absorption sqrt(kappa n_th) a^dag.

    This single model spans BOTH truncation-error drivers: at low n_th and
    strong g the sigma_z coupling pumps the mode high (a two-level truncation
    reflects it -> memory fabricated), while at high n_th the thermal population
    exceeds what two levels can hold (-> memory destroyed). Sweeping (g, n_th)
    therefore traces the sign-flip of the truncation error in one plane.
    Reduces exactly to spin_boson_qutip_states at n_th = 0.
    """
    sx, sz = qt.sigmax(), qt.sigmaz()
    n_modes = len(modes)
    ident = [qt.qeye(n_fock)] * n_modes

    def emb_sys(op):
        return qt.tensor(op, *ident)

    def emb_mode(op, k):
        ops = [qt.qeye(2)] + list(ident)
        ops[1 + k] = op
        return qt.tensor(*ops)

    a = qt.destroy(n_fock)
    H = 0.5 * delta * emb_sys(sx) + 0.5 * eps * emb_sys(sz)
    c_ops = []
    for k, (wk, gk, kappak) in enumerate(modes):
        H += wk * emb_mode(a.dag() * a, k)
        H += gk * emb_sys(sz) * emb_mode(a + a.dag(), k)
        c_ops.append(np.sqrt(kappak * (n_th + 1.0)) * emb_mode(a, k))
        if n_th > 0.0:
            c_ops.append(np.sqrt(kappak * n_th) * emb_mode(a.dag(), k))

    rho0 = qt.tensor(_sys_qobj(rho0_sys), *[qt.fock_dm(n_fock, 0)] * n_modes)
    sol = qt.mesolve(H, rho0, np.asarray(times, dtype=float), c_ops=c_ops)
    return [np.asarray(s.ptrace(0).full()) for s in sol.states]


# ---------------------------------------------------------------------------
# FMO-like excitonic dimer (Tier 3: a physically-motivated model)
# ---------------------------------------------------------------------------

def dimer_qutip_populations(times, eps_L, eps_R, J, site_modes, n_fock=3):
    """Two-chromophore Frenkel exciton dimer with a structured (pseudomode)
    bath dephasing each site -- the minimal energy-transfer unit behind FMO /
    environment-assisted transport.

    H_S = eps_L n_L + eps_R n_R + J (s+_L s-_R + s-_L s+_R)   (excitonic hop)
    each site l dephased by g_l sigma_z^l (a_l + a_l^dag), pseudomode (w_l,
    g_l, kappa_l) in ``site_modes`` = [(w_L,g_L,kap_L),(w_R,g_R,kap_R)].

    Initial state: excitation localised on site L. Returns
    (P_L(t), P_R(t)) site excitation populations.
    """
    nL = qt.tensor(qt.num(2), qt.qeye(2), qt.qeye(n_fock), qt.qeye(n_fock))
    nR = qt.tensor(qt.qeye(2), qt.num(2), qt.qeye(n_fock), qt.qeye(n_fock))
    smL = qt.tensor(qt.sigmam(), qt.qeye(2), qt.qeye(n_fock), qt.qeye(n_fock))
    smR = qt.tensor(qt.qeye(2), qt.sigmam(), qt.qeye(n_fock), qt.qeye(n_fock))
    szL = qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(n_fock), qt.qeye(n_fock))
    szR = qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(n_fock), qt.qeye(n_fock))
    aL = qt.tensor(qt.qeye(2), qt.qeye(2), qt.destroy(n_fock), qt.qeye(n_fock))
    aR = qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(n_fock), qt.destroy(n_fock))

    (wL, gL, kapL), (wR, gR, kapR) = site_modes
    H = (eps_L * nL + eps_R * nR + J * (smL.dag() * smR + smL * smR.dag())
         + wL * aL.dag() * aL + wR * aR.dag() * aR
         + gL * szL * (aL + aL.dag()) + gR * szR * (aR + aR.dag()))
    c_ops = [np.sqrt(kapL) * aL, np.sqrt(kapR) * aR]
    # excitation on L: |1>_L |0>_R |0>_mL |0>_mR
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0),
                     qt.basis(n_fock, 0), qt.basis(n_fock, 0))
    sol = qt.mesolve(H, psi0, np.asarray(times, float), c_ops=c_ops, e_ops=[nL, nR])
    return np.real(sol.expect[0]), np.real(sol.expect[1])


# ---------------------------------------------------------------------------
# Peak mode-occupation extraction (physical predictor for the Fock rule)
# ---------------------------------------------------------------------------

def _peak_occ_and_std(mean_series, m2_series):
    """Peak mean occupation and the number std at the peak-occupation time.
    (Truncation must hold the occupied extent, so the std at maximum load is
    the relevant width -- larger for super-Poissonian thermal statistics.)"""
    mean = np.real(np.asarray(mean_series))
    m2 = np.real(np.asarray(m2_series))
    idx = int(np.argmax(mean))
    var = max(m2[idx] - mean[idx] ** 2, 0.0)
    return float(mean[idx]), float(np.sqrt(var))


def stageA_finiteT_with_occupation(rho0_sys, times, gamma0, lam, n_th, n_fock):
    """Finite-T Stage A: returns (system_states, peak <n>, std at peak)."""
    omega, kappa = stageA_params(gamma0, lam)
    sm = qt.Qobj(SM_NP)
    a = qt.destroy(n_fock)
    H = omega * (qt.tensor(sm.dag(), a) + qt.tensor(sm, a.dag()))
    c_ops = [np.sqrt(kappa * (n_th + 1.0)) * qt.tensor(qt.qeye(2), a),
             np.sqrt(kappa * n_th) * qt.tensor(qt.qeye(2), a.dag())]
    num = qt.tensor(qt.qeye(2), a.dag() * a)
    num2 = qt.tensor(qt.qeye(2), a.dag() * a * a.dag() * a)
    rho0 = qt.tensor(_sys_qobj(rho0_sys), qt.fock_dm(n_fock, 0))
    tt = np.asarray(times, float)
    sol = qt.mesolve(H, rho0, tt, c_ops=c_ops, e_ops=[num, num2])
    sol_s = qt.mesolve(H, rho0, tt, c_ops=c_ops)
    peak, std = _peak_occ_and_std(sol.expect[0], sol.expect[1])
    return [np.asarray(s.ptrace(0).full()) for s in sol_s.states], peak, std


def spin_boson_with_occupation(rho0_sys, times, delta, eps, modes, n_fock):
    """Spin-boson: returns (system_states, peak total <n>, std at peak)."""
    sx, sz = qt.sigmax(), qt.sigmaz()
    n_modes = len(modes)
    ident = [qt.qeye(n_fock)] * n_modes

    def emb_sys(op):
        return qt.tensor(op, *ident)

    def emb_mode(op, k):
        ops = [qt.qeye(2)] + list(ident)
        ops[1 + k] = op
        return qt.tensor(*ops)

    a = qt.destroy(n_fock)
    H = 0.5 * delta * emb_sys(sx) + 0.5 * eps * emb_sys(sz)
    c_ops, num = [], 0
    for k, (wk, gk, kappak) in enumerate(modes):
        H += wk * emb_mode(a.dag() * a, k)
        H += gk * emb_sys(sz) * emb_mode(a + a.dag(), k)
        c_ops.append(np.sqrt(kappak) * emb_mode(a, k))
        num = num + emb_mode(a.dag() * a, k)
    rho0 = qt.tensor(_sys_qobj(rho0_sys), *[qt.fock_dm(n_fock, 0)] * n_modes)
    tt = np.asarray(times, float)
    sol = qt.mesolve(H, rho0, tt, c_ops=c_ops, e_ops=[num, num * num])
    sol_s = qt.mesolve(H, rho0, tt, c_ops=c_ops)
    peak, std = _peak_occ_and_std(sol.expect[0], sol.expect[1])
    return [np.asarray(s.ptrace(0).full()) for s in sol_s.states], peak, std


def required_fock_dimension(states_at_d: dict[int, list[np.ndarray]],
                            d_ref: int, eps: float) -> tuple[int, bool]:
    """Smallest Fock dim d whose reduced dynamics matches the d_ref reference
    to max abs deviation < eps. Returns (d_req, censored) where censored=True
    means NOTHING below d_ref converged, so d_req is a lower bound clipped at
    d_ref (the point must be treated as right-censored, not a true value)."""
    ref = np.array(states_at_d[d_ref])
    for d in sorted(k for k in states_at_d if k < d_ref):
        dev = float(np.max(np.abs(np.array(states_at_d[d]) - ref)))
        if dev < eps:
            return d, False
    return d_ref, True


# ---------------------------------------------------------------------------
# Convenience: reference D(t) curves and dynamical maps for the IC inputs
# ---------------------------------------------------------------------------

def evolve_ic_inputs(evolver, times) -> dict[str, list[np.ndarray]]:
    """Evolve the four informationally complete inputs plus the remaining
    canonical-pair partners; ``evolver(rho0, times) -> [rho(t)]``."""
    return {lbl: evolver(dm(lbl), times) for lbl in ("0", "1", "+", "-", "+i", "-i")}

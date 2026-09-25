"""Exact ensemble averages of Stage A under correlated detuning noise (T1.11).

The detuning delta(t) is a finite-state Markov process: a sum of ``n_tel``
independent telegraph components s_i(t) = +-1, each switching at rate
r = 1 / (2 tau_c), scaled as delta = sigma * sum_i s_i / sqrt(n_tel). Its
correlation function is sigma^2 exp(-|tau| / tau_c) for every n_tel, so its
spectral density is S(omega) = 2 sigma^2 tau_c / (1 + omega^2 tau_c^2).
n_tel = 1 is random telegraph noise; large n_tel approaches the
Ornstein-Uhlenbeck process (binomial marginal -> Gaussian). Only the number
j of up-components matters, so the process has n_tel + 1 levels.

The noise acts as in the paper's apparatus (circuits.stageA_exact_circuit_dephased):
after each Stage A step (exchange, then damping of the mode) the system gets
the kick RZ(delta_k dt). Averaging over the noise is done exactly, not by
sampling: the joint state is {rho_j}, the sub-normalised system (x) mode
state conditioned on noise level j, and between steps the levels mix with the
chain's transition matrix. A finite sampled ensemble would leave
fluctuations in D(t) that the BLP's positive-part sum turns into spurious
backflow; the exact average has none.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from scipy.special import comb

I2 = np.eye(2)


def stageA_step_operators(omega: float, kappa: float, dt: float):
    """Exchange unitary and damping Kraus operators of stageA_step on
    (system q0, mode q1), in Qiskit's little-endian ordering (index 2 m + s)."""
    qc = QuantumCircuit(2)
    qc.rxx(omega * dt, 0, 1)
    qc.ryy(omega * dt, 0, 1)
    U = Operator(qc).data
    p = 1.0 - np.exp(-kappa * dt)
    K0 = np.kron(np.diag([1.0, np.sqrt(1.0 - p)]), I2)
    K1 = np.kron(np.array([[0.0, np.sqrt(p)], [0.0, 0.0]]), I2)
    return U, (K0, K1)


def noise_levels(sigma: float, tau_c: float, n_tel: int, dt: float):
    """Detuning values, stationary distribution and one-step transition matrix
    (rows: from, columns: to) of the sum of n_tel telegraph components.
    tau_c = inf gives quasi-static noise (no transitions)."""
    j = np.arange(n_tel + 1)
    delta = sigma * (2.0 * j - n_tel) / np.sqrt(n_tel)
    pi = comb(n_tel, j) / 2.0 ** n_tel
    if not np.isfinite(tau_c):
        return delta, pi, np.eye(n_tel + 1)
    r = 1.0 / (2.0 * tau_c)
    Q = np.zeros((n_tel + 1, n_tel + 1))
    for k in j:
        if k < n_tel:
            Q[k, k + 1] = (n_tel - k) * r
        if k > 0:
            Q[k, k - 1] = k * r
        Q[k, k] = -Q[k].sum()
    return delta, pi, expm(Q * dt)


def evolve_dephased_average(rho0_sys: np.ndarray, n_steps: int, dt: float, omega: float,
                            kappa: float, sigma: float, tau_c: float, n_tel: int) -> np.ndarray:
    """Noise-averaged system states after k = 0..n_steps steps (vacuum mode)."""
    U, kraus = stageA_step_operators(omega, kappa, dt)
    delta, pi, P = noise_levels(sigma, tau_c, n_tel, dt)
    mode0 = np.diag([1.0, 0.0])
    rho = np.einsum("j,ab->jab", pi, np.kron(mode0, np.asarray(rho0_sys, dtype=complex)))
    phase_s = np.array([-0.5, 0.5])            # RZ(phi) = diag(e^{-i phi/2}, e^{+i phi/2})
    phase = np.kron(np.ones(2), phase_s)       # system phase exponent on the 4-dim index
    out = [np.asarray(rho0_sys, dtype=complex)]
    for _ in range(n_steps):
        rho = U @ rho @ U.conj().T
        rho = sum(K @ rho @ K.conj().T for K in kraus)
        ph = np.exp(1j * np.outer(delta * dt, phase))           # [level, index]
        rho = rho * ph[:, :, None] * ph.conj()[:, None, :]
        out.append(_trace_mode(rho.sum(axis=0)))
        rho = np.einsum("ij,iab->jab", P, rho)
    return np.array(out)


def _trace_mode(rho4: np.ndarray) -> np.ndarray:
    """Partial trace over the mode (q1) of a 4x4 little-endian state."""
    r = rho4.reshape(2, 2, 2, 2)                # [m, s, m', s']
    return np.einsum("msmt->st", r)


def accumulated_phase_variance(sigma: float, tau_c: float, n_steps: int, dt: float) -> float:
    """Variance of sum_k delta_k dt for the stroboscopic kicks."""
    if not np.isfinite(tau_c):
        return (sigma * dt * n_steps) ** 2
    a = np.exp(-dt / tau_c)
    k = np.arange(n_steps)
    lags = np.abs(k[:, None] - k[None, :])
    return float(sigma ** 2 * dt ** 2 * np.sum(a ** lags))

"""Exact noise-averaged Stage A dynamics under correlated detuning."""
import numpy as np
import pytest

from qmembench.circuits import exact_states_from_labels, run_exact, stageA_exact_circuit_dephased
from qmembench.correlated_noise import (accumulated_phase_variance, evolve_dephased_average,
                                        noise_levels, stageA_step_operators)
from qmembench.measures import dm
from qmembench.reference import stageA_params

OMEGA, KAPPA = stageA_params(5.0, 1.0)


def test_quasi_static_average_equals_mean_of_aer_runs():
    """n_tel = 1, tau_c = inf: two equally likely constant kicks +-sigma dt."""
    n, dt, sigma = 30, 0.05, 0.7
    got = evolve_dephased_average(dm("+i"), n, dt, OMEGA, KAPPA, sigma, np.inf, 1)
    runs = [np.asarray(exact_states_from_labels(run_exact([stageA_exact_circuit_dephased(
        "+i", n, dt, OMEGA, KAPPA, [s * sigma * dt] * n)])[0], n)) for s in (1, -1)]
    np.testing.assert_allclose(got, 0.5 * (runs[0] + runs[1]), atol=1e-10)


def test_noise_free_average_equals_the_production_circuit():
    n, dt = 25, 0.05
    got = evolve_dephased_average(dm("+"), n, dt, OMEGA, KAPPA, 0.0, 1.0, 3)
    ref = np.asarray(exact_states_from_labels(run_exact([stageA_exact_circuit_dephased(
        "+", n, dt, OMEGA, KAPPA, [0.0] * n)])[0], n))
    np.testing.assert_allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("n_tel", [1, 4, 20])
def test_chain_correlation_is_exponential(n_tel):
    sigma, tau_c, dt = 0.3, 0.8, 0.05
    delta, pi, P = noise_levels(sigma, tau_c, n_tel, dt)
    assert pi.sum() == pytest.approx(1.0)
    assert np.sum(pi * delta ** 2) == pytest.approx(sigma ** 2, rel=1e-12)
    Pk = np.eye(len(pi))
    for k in range(40):
        C = float(np.einsum("i,i,ij,j->", pi, delta, Pk, delta))
        assert C == pytest.approx(sigma ** 2 * np.exp(-k * dt / tau_c), rel=1e-9)
        Pk = Pk @ P


def test_exact_average_matches_sampled_trajectories():
    """Monte Carlo over sampled noise paths (same step operators) converges to
    the exact joint-state average."""
    n, dt, sigma, tau_c, n_tel = 40, 0.05, 1.5, 0.3, 2
    U, kraus = stageA_step_operators(OMEGA, KAPPA, dt)
    delta, pi, P = noise_levels(sigma, tau_c, n_tel, dt)
    rng = np.random.default_rng(1)
    rho_s = dm("+")
    phase = np.kron(np.ones(2), np.array([-0.5, 0.5]))
    n_traj = 4000
    acc = np.zeros((n + 1, 2, 2), dtype=complex)
    for _ in range(n_traj):
        rho = np.kron(np.diag([1.0, 0.0]), rho_s).astype(complex)
        level = rng.choice(len(pi), p=pi)
        acc[0] += rho_s
        for k in range(1, n + 1):
            rho = U @ rho @ U.conj().T
            rho = sum(K @ rho @ K.conj().T for K in kraus)
            ph = np.exp(1j * delta[level] * dt * phase)
            rho = rho * ph[:, None] * ph.conj()[None, :]
            acc[k] += np.einsum("msmt->st", rho.reshape(2, 2, 2, 2))
            level = rng.choice(len(pi), p=P[level])
    mc = acc / n_traj
    exact = evolve_dephased_average(rho_s, n, dt, OMEGA, KAPPA, sigma, tau_c, n_tel)
    # coherences carry the noise; MC error ~ 1/sqrt(n_traj)
    assert np.max(np.abs(mc - exact)) < 4.0 / np.sqrt(n_traj)


def test_accumulated_phase_variance_matches_sampling():
    sigma, tau_c, dt, n = 0.2, 0.5, 0.05, 60
    delta, pi, P = noise_levels(sigma, tau_c, 20, dt)
    rng = np.random.default_rng(2)
    phases = []
    for _ in range(20000):
        level, acc = rng.choice(len(pi), p=pi), 0.0
        for _ in range(n):
            acc += delta[level] * dt
            level = rng.choice(len(pi), p=P[level])
        phases.append(acc)
    var = accumulated_phase_variance(sigma, tau_c, n, dt)
    assert np.var(phases) == pytest.approx(var, rel=0.05)
    assert accumulated_phase_variance(sigma, np.inf, n, dt) == pytest.approx((sigma * dt * n) ** 2)

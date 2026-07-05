"""Realistic + non-Markovian noise-model machinery."""
import numpy as np
import pytest

from qmembench.circuits import (exact_states_from_labels, run_exact,
                                stageA_exact_circuit_dephased, saved_steps)
from qmembench.noise import from_fake_backend, quasistatic_angles
from qmembench.reference import stageA_params


def test_fake_backend_noise_model_builds():
    nm, meta = from_fake_backend("FakeManilaV2")
    assert meta["markovian"] is True
    assert "cx" in nm.basis_gates and "sx" in nm.basis_gates


def test_quasistatic_angles_stats():
    a = quasistatic_angles(0.3, 20000, seed=0)
    assert abs(a.mean()) < 0.02
    assert abs(a.std() - 0.3) < 0.02


def test_zero_angle_dephasing_is_noop():
    omega, kappa = stageA_params(5.0, 1.0)
    n_steps = 40
    q_ref = stageA_exact_circuit_dephased("+", n_steps, 0.05, omega, kappa,
                                          [0.0] * n_steps)
    states = exact_states_from_labels(run_exact([q_ref])[0], n_steps)
    # a |+> state under pure damping keeps a real, positive-x coherence;
    # zero dephasing must leave Im(rho_01) = 0 to numerical precision.
    assert abs(states[-1][0, 1].imag) < 1e-9


def test_correlated_vs_uncorrelated_dephasing_differ():
    """Quasi-static (correlated) and Markovian (per-step) dephasing at the
    SAME marginal sigma must produce different reduced dynamics -- this is
    the physical content that lets the study separate memoryful device noise
    from memoryless models."""
    omega, kappa = stageA_params(5.0, 1.0)
    n_steps, dt, sigma, n_ens = 60, 0.05, 0.4, 40
    steps = saved_steps(n_steps, save_every=n_steps)  # just final state
    rng = np.random.default_rng(3)
    fixed = rng.normal(0, sigma, n_ens)
    per_step = rng.normal(0, sigma, size=(n_ens, n_steps))

    def avg_final(draw):
        acc = np.zeros((2, 2), dtype=complex)
        for m in range(n_ens):
            qc = stageA_exact_circuit_dephased("+", n_steps, dt, omega, kappa,
                                               draw(m), save_every=n_steps)
            acc += run_exact([qc])[0][f"rho_{n_steps}"] / n_ens
        return acc

    rho_corr = avg_final(lambda m: [fixed[m]] * n_steps)
    rho_uncorr = avg_final(lambda m: per_step[m])
    assert np.max(np.abs(rho_corr - rho_uncorr)) > 1e-3

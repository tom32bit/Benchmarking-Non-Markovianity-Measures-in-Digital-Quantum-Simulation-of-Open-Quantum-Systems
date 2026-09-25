"""Ancilla-train (chain) collision model: physicality and memory behaviour."""
import numpy as np
import pytest

from qmembench.collision import (chain_collision_states_numpy, fit_chain_params,
                                 _reduce_sys)


@pytest.mark.parametrize("M", [1, 2, 3])
def test_chain_states_are_valid_density_matrices(M):
    st = chain_collision_states_numpy("1", 25, M, 0.5, 1.0, 0.1)
    for s in st:
        assert np.trace(s).real == pytest.approx(1.0, abs=1e-9)
        assert np.min(np.linalg.eigvalsh((s + s.conj().T) / 2)) > -1e-9


def test_reduce_sys_matches_bruteforce():
    """einsum partial trace vs explicit kron construction for 2 qubits."""
    rho_s = np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex)
    rho_e = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
    rho = np.kron(rho_s, rho_e)
    np.testing.assert_allclose(_reduce_sys(rho, 2), rho_s, atol=1e-12)


def test_reduce_sys_three_qubits():
    """P4: partial trace correctness for a 3-qubit product state."""
    rho_s = np.array([[0.55, 0.15 - 0.1j], [0.15 + 0.1j, 0.45]], dtype=complex)
    e1 = np.array([[0.8, 0.05], [0.05, 0.2]], dtype=complex)
    e2 = np.array([[0.3, 0.0], [0.0, 0.7]], dtype=complex)
    rho = np.kron(np.kron(rho_s, e1), e2)
    np.testing.assert_allclose(_reduce_sys(rho, 3), rho_s, atol=1e-12)


def test_chain_produces_backflow_and_damping_suppresses_it():
    """The chain admits information backflow (non-monotone P_e); adding
    far-end loss damps it toward monotone -- the two knobs the fit uses to
    shape a target memory kernel."""
    def backflow(p_loss):
        st = chain_collision_states_numpy("1", 60, 3, 0.6, 1.2, p_loss)
        pe = np.array([s[1, 1].real for s in st])
        d = np.diff(pe)
        return float(np.sum(d[d > 0]))
    assert backflow(0.0) > 0.05                 # memory present
    assert backflow(0.4) < backflow(0.0)        # loss suppresses revivals


@pytest.mark.parametrize("M", [1, 2, 3])
def test_chain_numpy_matches_circuit(M):
    """The chain circuit must reproduce the numpy engine (like the 3-qubit
    model's equivalence test) so resource/noise claims rest on the same map."""
    from qmembench.collision import chain_collision_exact_circuit
    from qmembench.circuits import run_exact, exact_states_from_labels
    ts, th, pl, n = 0.5, 1.0, 0.1, 15
    np_states = chain_collision_states_numpy("1", n, M, ts, th, pl)
    qc = chain_collision_exact_circuit("1", n, M, ts, th, pl)
    qk_states = exact_states_from_labels(run_exact([qc])[0], n)
    dev = np.max(np.abs(np.array(np_states) - np.array(qk_states)))
    assert dev < 1e-8, f"numpy vs circuit chain deviation {dev:.2e}"


def test_chain_fit_improves_or_holds_with_M():
    """Fitting the Lorentzian population: more chain qubits should not hurt
    (best-achievable RMSE non-increasing within optimisation tolerance)."""
    from qmembench.reference import analytic_G
    times = np.linspace(0, 9.0, 61)
    target = np.abs(analytic_G(times, 5.0, 1.0)) ** 2
    r1 = fit_chain_params(target, len(times) - 1, M=1, maxiter=120)
    r3 = fit_chain_params(target, len(times) - 1, M=3, maxiter=120)
    assert r3["rmse"] <= r1["rmse"] + 0.02
    assert r3["n_qubits_circuit"] == 5  # 1 system + 3 chain + 1 damping ancilla


@pytest.mark.parametrize("M", [1, 2, 3])
def test_single_excitation_amplitude_matches_full_chain(M):
    """P_e and the (+,-) trace distance of the full chain equal |c_0|^2 and |c_0|."""
    from qmembench.collision import chain_collision_states_numpy, chain_excitation_amplitude
    from qmembench.measures import trace_distance
    args = (25, M, 0.37, 0.81, 0.23)
    c0 = chain_excitation_amplitude(*args)
    pe = [s[1, 1].real for s in chain_collision_states_numpy("1", *args)]
    np.testing.assert_allclose(np.abs(c0) ** 2, pe, atol=1e-12)
    sp, sm = chain_collision_states_numpy("+", *args), chain_collision_states_numpy("-", *args)
    np.testing.assert_allclose([trace_distance(a, b) for a, b in zip(sp, sm)], np.abs(c0), atol=1e-12)


def test_pseudomode_is_the_one_site_chain():
    """M = 1 with theta_s = Omega dt and p_loss = 1 - exp(-kappa dt) reproduces
    the Stage A circuit's |G_c|."""
    from qmembench.circuits import exact_states_from_labels, run_exact, stageA_exact_circuit
    from qmembench.collision import chain_excitation_amplitude
    from qmembench.measures import trace_distance
    from qmembench.reference import stageA_params
    omega, kappa = stageA_params(5.0, 1.0)
    dt, n = 0.1, 40
    c0 = chain_excitation_amplitude(n, 1, omega * dt, 0.0, 1.0 - np.exp(-kappa * dt))
    sp = exact_states_from_labels(run_exact([stageA_exact_circuit("+", n, dt, omega, kappa)])[0], n)
    sm = exact_states_from_labels(run_exact([stageA_exact_circuit("-", n, dt, omega, kappa)])[0], n)
    np.testing.assert_allclose([trace_distance(a, b) for a, b in zip(sp, sm)], np.abs(c0), atol=1e-10)


def test_bounded_fit_recovers_known_parameters():
    from qmembench.collision import chain_excitation_amplitude, fit_chain_params_bounded
    target = np.abs(chain_excitation_amplitude(60, 2, 0.4, 0.9, 0.3)) ** 2
    fit = fit_chain_params_bounded(target, 60, 2)
    assert fit["rmse"] < 1e-6
    assert fit["at_bound"] == []


def test_eigen_path_matches_step_iteration():
    """The eigendecomposition shortcut for c_0 equals direct iteration."""
    from qmembench.collision import chain_excitation_amplitude
    rng = np.random.default_rng(7)
    for M in (1, 2, 3, 5):
        ts, th, pl = rng.random() * np.pi, rng.random() * np.pi, rng.random()
        fast = chain_excitation_amplitude(120, M, ts, th, pl)
        # direct iteration, written out independently
        E = np.eye(M + 1, dtype=complex)
        def ex(t, a, b):
            U = np.eye(M + 1, dtype=complex)
            U[a, a] = U[b, b] = np.cos(t); U[a, b] = U[b, a] = -1j * np.sin(t)
            return U
        S = ex(ts, 0, 1)
        for j in range(1, M):
            S = ex(th, j, j + 1) @ S
        d = np.ones(M + 1); d[M] = np.sqrt(1 - pl); S = d[:, None] * S
        c = np.zeros(M + 1, complex); c[0] = 1; slow = [1.0]
        for _ in range(120):
            c = S @ c; slow.append(c[0])
        np.testing.assert_allclose(fast, slow, atol=1e-10)

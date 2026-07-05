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

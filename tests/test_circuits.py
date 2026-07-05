"""Circuit tier vs analytic/QuTiP (tier 3 of the validation chain), and
collision-model numpy <-> Qiskit consistency."""
import numpy as np
import pytest

from qmembench.circuits import (amplitude_damping_block, run_exact,
                                exact_states_from_labels, stageA_exact_circuit)
from qmembench.collision import (collision_exact_circuit,
                                 collision_states_numpy)
from qmembench.measures import dm
from qmembench.reference import analytic_stageA_states, stageA_params
from qiskit import QuantumCircuit

GAMMA0, LAM = 5.0, 1.0


def test_amplitude_damping_block_is_exact_channel():
    """Single dilation block must equal the exact AD channel on |1>."""
    p = 0.37
    qc = QuantumCircuit(3)
    qc.x(1)  # pseudomode qubit in |1>
    amplitude_damping_block(qc, target=1, anc=2, p=p)
    qc.save_density_matrix(qubits=[1], label="rho_pm")
    res = run_exact([qc])[0]
    rho = res["rho_pm"]
    assert rho[1, 1].real == pytest.approx(1 - p, abs=1e-10)
    assert rho[0, 0].real == pytest.approx(p, abs=1e-10)


@pytest.mark.parametrize("init", ["1", "+"])
def test_stageA_circuit_matches_analytic(init):
    dt, t_max = 0.05, 6.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    omega, kappa = stageA_params(GAMMA0, LAM)
    qc = stageA_exact_circuit(init, n_steps, dt, omega, kappa)
    res = run_exact([qc])[0]
    circ = exact_states_from_labels(res, n_steps)
    ana = analytic_stageA_states(dm(init), times, GAMMA0, LAM)
    dev = np.max(np.abs(np.array(circ) - np.array(ana)))
    # first-order Trotter at dt=0.05: percent-level agreement expected
    assert dev < 0.02, f"circuit vs analytic deviation {dev:.3f}"


def test_trotter_error_decreases_with_dt():
    omega, kappa = stageA_params(GAMMA0, LAM)
    devs = []
    for dt in (0.2, 0.1, 0.05):
        n_steps = int(round(4.0 / dt))
        times = np.linspace(0.0, n_steps * dt, n_steps + 1)
        qc = stageA_exact_circuit("1", n_steps, dt, omega, kappa)
        circ = exact_states_from_labels(run_exact([qc])[0], n_steps)
        ana = analytic_stageA_states(dm("1"), times, GAMMA0, LAM)
        devs.append(np.max(np.abs(np.array(circ) - np.array(ana))))
    assert devs[0] > devs[1] > devs[2]


def test_collision_numpy_matches_qiskit():
    theta_s, theta_m, n_steps = 0.6, 0.9, 12
    states_np = collision_states_numpy("1", n_steps, theta_s, theta_m)
    qc = collision_exact_circuit("1", n_steps, theta_s, theta_m)
    states_qk = exact_states_from_labels(run_exact([qc])[0], n_steps)
    dev = np.max(np.abs(np.array(states_np) - np.array(states_qk)))
    assert dev < 1e-8, f"numpy vs qiskit collision deviation {dev:.2e}"


def test_collision_markovian_limit_monotone():
    """theta_m = 0 (no memory) must give monotone excited-state decay."""
    states = collision_states_numpy("1", 30, 0.5, 0.0)
    pe = np.array([s[1, 1].real for s in states])
    assert np.all(np.diff(pe) <= 1e-12)


def test_collision_memory_produces_backflow():
    """theta_m > 0 must allow the excitation to return (non-monotone P_e)."""
    states = collision_states_numpy("1", 30, 0.9, 1.2)
    pe = np.array([s[1, 1].real for s in states])
    assert np.any(np.diff(pe) > 1e-6)

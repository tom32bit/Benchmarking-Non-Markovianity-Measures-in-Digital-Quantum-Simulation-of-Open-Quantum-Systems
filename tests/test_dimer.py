"""FMO-like excitonic dimer: circuit vs QuTiP population dynamics."""
import numpy as np
import pytest

from qmembench.circuits import (dimer_exact_circuit, dimer_populations_from_data,
                                run_exact)
from qmembench.reference import dimer_qutip_populations


def test_population_is_conserved_without_loss():
    """No damping -> single excitation conserved: P_L + P_R = 1 at all times."""
    times = np.linspace(0, 8, 60)
    modes = [(0.8, 0.25, 0.0), (0.8, 0.25, 0.0)]  # kappa = 0
    PL, PR = dimer_qutip_populations(times, 0.5, -0.5, 1.0, modes, n_fock=3)
    # dephasing coupling can shuffle a little into modes only via kappa; with
    # kappa=0 and sigma_z coupling, excitation stays in the 2-site subspace.
    assert np.allclose(PL + PR, 1.0, atol=1e-3)


def test_excitonic_transfer_happens():
    """Excitation prepared on L must transfer toward R (P_R grows from 0)."""
    times = np.linspace(0, 8, 80)
    modes = [(0.8, 0.25, 0.3), (0.8, 0.25, 0.3)]
    PL, PR = dimer_qutip_populations(times, 0.5, -0.5, 1.0, modes, n_fock=3)
    assert PR[0] == pytest.approx(0.0, abs=1e-6)
    assert np.max(PR) > 0.2


def test_circuit_matches_qutip_populations():
    dt, t_max = 0.05, 8.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0, n_steps * dt, n_steps + 1)
    modes = [(0.8, 0.25, 0.3), (0.8, 0.25, 0.3)]
    PL_ref, PR_ref = dimer_qutip_populations(times, 0.5, -0.5, 1.0, modes, n_fock=2)
    qc = dimer_exact_circuit(n_steps, dt, 0.5, -0.5, 1.0, modes)
    PL_c, PR_c = dimer_populations_from_data(run_exact([qc])[0], n_steps)
    # matched truncation (n_fock=2): only Trotter error remains, percent-level
    assert np.max(np.abs(PR_c - PR_ref)) < 0.03
    assert np.max(np.abs(PL_c - PL_ref)) < 0.03


MODES = [(0.8, 0.25, 0.3), (0.8, 0.25, 0.3)]


def test_excitonic_reference_reproduces_site_populations():
    """The excitonic-qubit reference started in |L> gives the published P_L, P_R."""
    from qmembench.reference import dimer_qutip_excitonic_states
    times = np.linspace(0, 4, 41)
    PL, PR = dimer_qutip_populations(times, 0.5, -0.5, 1.0, MODES, n_fock=3)
    blocks = dimer_qutip_excitonic_states(np.diag([1.0, 0.0]), times, 0.5, -0.5, 1.0,
                                          MODES, n_fock=3)
    np.testing.assert_allclose([b[0, 0].real for b in blocks], PL, atol=1e-6)
    np.testing.assert_allclose([b[1, 1].real for b in blocks], PR, atol=1e-6)


def test_excitonic_circuit_matches_site_circuit_and_prepares_states():
    from qmembench.circuits import (DIMER_EXCITONIC_INDEX, dimer_exact_circuit_excitonic,
                                    exact_states_from_labels)
    idx = DIMER_EXCITONIC_INDEX
    n, dt = 20, 0.05
    _, PR_old = dimer_populations_from_data(
        run_exact([dimer_exact_circuit(n, dt, 0.5, -0.5, 1.0, MODES)])[0], n)
    new = exact_states_from_labels(run_exact([dimer_exact_circuit_excitonic(
        "0", n, dt, 0.5, -0.5, 1.0, MODES)])[0], n)
    np.testing.assert_allclose([m[np.ix_(idx, idx)][1, 1].real for m in new], PR_old,
                               atol=1e-10)
    start = exact_states_from_labels(run_exact([dimer_exact_circuit_excitonic(
        "+i", 1, dt, 0.5, -0.5, 1.0, MODES)])[0], 1)[0]
    np.testing.assert_allclose(start[np.ix_(idx, idx)],
                               0.5 * np.array([[1, -1j], [1j, 1]]), atol=1e-12)


def test_isolated_dimer_has_no_memory_and_coherent_transfer():
    """g = 0: unitary on the excitonic qubit, so no backflow; the acceptor
    population peaks at 4J^2 / (dE^2 + 4J^2) = 0.8 (coherent oscillation).
    Tight tolerances: at mesolve's defaults the solver error alone gives ~7e-5."""
    from qmembench.measures import blp_optimize_pair, dm, pauli_images_from_states
    from qmembench.reference import TIGHT_SOLVER_OPTIONS, dimer_qutip_excitonic_states
    times = np.linspace(0, 10, 201)
    iso = [(0.8, 0.0, 0.3), (0.8, 0.0, 0.3)]
    states = {l: dimer_qutip_excitonic_states(dm(l), times, 0.5, -0.5, 1.0, iso, n_fock=2,
                                              options=TIGHT_SOLVER_OPTIONS)
              for l in ("0", "1", "+", "+i")}
    res = blp_optimize_pair(pauli_images_from_states(states), n_grid=500)
    assert res["N_BLP_optimized"] < 1e-6
    assert max(s[1, 1].real for s in states["0"]) == pytest.approx(0.8, abs=2e-3)

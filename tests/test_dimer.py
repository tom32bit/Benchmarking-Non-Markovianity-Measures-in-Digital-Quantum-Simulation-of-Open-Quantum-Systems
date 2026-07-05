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

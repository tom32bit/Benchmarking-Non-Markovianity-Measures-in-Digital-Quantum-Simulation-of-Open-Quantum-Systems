"""Finite-temperature spin-boson: reduces to zero-T; and the truncation error
changes sign across the (coupling, temperature) plane."""
import numpy as np

from qmembench.measures import dm
from qmembench.reference import (spin_boson_qutip_states,
                                 spin_boson_qutip_states_finiteT)


def test_finiteT_spin_boson_reduces_to_zeroT():
    times = np.linspace(0, 6, 50)
    modes = [(1.0, 0.4, 0.4), (1.6, 0.4, 0.8)]
    a = spin_boson_qutip_states(dm("+"), times, 1.0, 0.0, modes, n_fock=4)
    b = spin_boson_qutip_states_finiteT(dm("+"), times, 1.0, 0.0, modes,
                                        n_th=0.0, n_fock=4)
    assert np.max(np.abs(np.array(a) - np.array(b))) < 1e-9


def test_thermal_population_raises_mode_occupation():
    """A nonzero n_th must change the reduced dynamics relative to n_th=0."""
    times = np.linspace(0, 6, 50)
    modes = [(1.0, 0.4, 0.4), (1.6, 0.4, 0.8)]
    cold = spin_boson_qutip_states_finiteT(dm("+"), times, 1.0, 0.0, modes,
                                           n_th=0.0, n_fock=5)
    warm = spin_boson_qutip_states_finiteT(dm("+"), times, 1.0, 0.0, modes,
                                           n_th=1.0, n_fock=5)
    assert np.max(np.abs(np.array(cold) - np.array(warm))) > 1e-2


def test_optimized_phase_diagram_parallel_equals_serial():
    """The process-pool path must reproduce the serial computation exactly."""
    from qmembench.experiments import phase9_truncation_phase_diagram_optimized as run
    serial = run(d_ref=3, d_check=2, n_workers=1, grid=(2, 2))
    parallel = run(d_ref=3, d_check=2, n_workers=2, grid=(2, 2))
    assert serial["signed_error_optimized"].shape == (2, 2)
    np.testing.assert_allclose(parallel["signed_error_optimized"],
                               serial["signed_error_optimized"], atol=1e-12)
    np.testing.assert_allclose(parallel["signed_error_plus_minus"],
                               serial["signed_error_plus_minus"], atol=1e-12)

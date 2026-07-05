"""Tier-3 scaling laws: Trotter order, truncation decay, linear resources."""
import numpy as np

from qmembench.experiments import phase7_scaling, _loglog_slope


def test_loglog_slope_recovers_power():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 3.0 * x ** 2.0
    assert _loglog_slope(x, y) == __import__("pytest").approx(2.0, abs=1e-9)


def test_scaling_laws_have_expected_exponents():
    d = phase7_scaling(quick=True)
    # (i) Trotter convergence. Generic Lindblad Lie-Trotter is O(dt); for this
    # integrable Stage-A model the coherent gates commute (Hamiltonian part
    # exact) and the coherent-dissipative splitting error is effectively
    # second order, so the measured order is ~2 -- i.e. AT LEAST first order.
    assert d["trotter_order_estimate"] > 0.85
    # (ii) truncation error decreases with Fock dimension (negative log-slope)
    assert d["truncation_log_decay_per_level"] < 0
    # (iii) circuit depth grows ~linearly in Trotter steps (slope ~1)
    assert 0.85 < d["depth_loglog_slope_vs_steps"] < 1.15
    # resources grow with number of pseudomodes
    rm = d["resources_vs_modes"]
    assert rm[-1]["cx"] > rm[0]["cx"]
    assert rm[-1]["n_qubits"] > rm[0]["n_qubits"]

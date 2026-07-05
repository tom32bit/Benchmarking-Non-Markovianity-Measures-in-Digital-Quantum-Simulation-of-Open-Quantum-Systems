"""Analytic <-> QuTiP agreement (tier 1 <-> tier 2 of the validation chain)."""
import numpy as np
import pytest

from qmembench.measures import dm
from qmembench.reference import (analytic_G, analytic_stageA_states,
                                 stageA_qutip_states)

TIMES = np.linspace(0.0, 10.0, 80)


@pytest.mark.parametrize("gamma0,lam", [(5.0, 1.0), (0.2, 1.0), (0.5, 1.0)])
@pytest.mark.parametrize("init", ["1", "+", "+i"])
def test_qutip_pseudomode_matches_analytic(gamma0, lam, init):
    ana = analytic_stageA_states(dm(init), TIMES, gamma0, lam)
    qut = stageA_qutip_states(dm(init), TIMES, gamma0, lam, n_fock=2)
    dev = np.max(np.abs(np.array(ana) - np.array(qut)))
    assert dev < 5e-5, f"analytic vs qutip deviation {dev:.2e}"


def test_two_level_truncation_is_exact_stageA():
    """n_fock=2 vs n_fock=4 must agree (excitation-number conservation).

    Tolerance is set by mesolve's ODE integration accuracy (~1e-6 between two
    independent runs), NOT by physics: a genuinely populated n=2 Fock state
    would show up at percent level, orders of magnitude above this bound.
    """
    q2 = stageA_qutip_states(dm("1"), TIMES, 5.0, 1.0, n_fock=2)
    q4 = stageA_qutip_states(dm("1"), TIMES, 5.0, 1.0, n_fock=4)
    assert np.max(np.abs(np.array(q2) - np.array(q4))) < 1e-5


def test_G_regimes():
    G_markov = analytic_G(TIMES, 0.2, 1.0)
    assert np.all(np.diff(np.abs(G_markov)) <= 1e-12)  # monotone decay
    G_nm = analytic_G(TIMES, 5.0, 1.0)
    assert np.min(G_nm) < 0  # oscillation through zero -> backflow regime

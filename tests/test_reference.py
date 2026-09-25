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


def test_continuum_blp_closed_form_limits():
    """A2: the Stage A continuum BLP sums the |G| rises exp(-pi k lam/delta);
    for an infinite window it is 1/(exp(pi lam/delta) - 1). Markovian: zero."""
    from qmembench.reference import analytic_blp_continuum
    delta = np.sqrt(2 * 5.0 * 1.0 - 1.0)
    assert analytic_blp_continuum(1e3, 5.0, 1.0) == pytest.approx(
        1.0 / (np.exp(np.pi / delta) - 1.0), rel=1e-12)
    assert analytic_blp_continuum(10.0, 0.2, 1.0) == 0.0


def test_discrete_blp_approaches_the_continuum_value():
    """The grid sum of positive |G| increments converges to the closed form,
    from below, as the sampling is refined."""
    from qmembench.measures import blp_from_curve
    from qmembench.reference import analytic_blp_continuum
    exact = analytic_blp_continuum(12.0, 5.0, 1.0)
    deficits = []
    for s in (0.05, 0.005, 0.0005):
        n = int(round(12.0 / s))
        t = np.linspace(0.0, n * s, n + 1)
        deficits.append(exact - blp_from_curve(np.abs(analytic_G(t, 5.0, 1.0))))
    assert all(d > 0 for d in deficits)
    assert deficits[0] > deficits[1] > deficits[2]
    assert deficits[2] < 1e-3


def test_reduced_states_from_expectations_match_partial_trace():
    """_solve_system_states reads rho_sys from expectations of |b><a| (x) 1;
    it must equal ptrace(0) of the stored states of the same solve."""
    import qutip as qt
    from qmembench.reference import TIGHT_SOLVER_OPTIONS, _solve_system_states
    a2, a3 = qt.destroy(2), qt.destroy(3)
    H = (0.7 * qt.tensor(qt.sigmax(), qt.qeye(3), qt.qeye(2))
         + 0.4 * qt.tensor(qt.sigmaz(), a3 + a3.dag(), qt.qeye(2))
         + 0.3 * qt.tensor(qt.sigmay(), qt.qeye(3), a2 + a2.dag())
         + 1.1 * qt.tensor(qt.qeye(2), a3.dag() * a3, qt.qeye(2)))
    c_ops = [0.5 * qt.tensor(qt.qeye(2), a3, qt.qeye(2)),
             0.2 * qt.tensor(qt.qeye(2), a3.dag(), qt.qeye(2)),
             0.4 * qt.tensor(qt.qeye(2), qt.qeye(3), a2)]
    rho0 = qt.tensor(qt.Qobj(dm("+i")), qt.fock_dm(3, 0), qt.fock_dm(2, 1))
    times = np.linspace(0.0, 3.0, 31)
    got = _solve_system_states(H, rho0, times, c_ops, TIGHT_SOLVER_OPTIONS)
    ref = qt.mesolve(H, rho0, times, c_ops=c_ops, options=TIGHT_SOLVER_OPTIONS).states
    np.testing.assert_allclose(np.asarray(got), np.asarray([s.ptrace(0).full() for s in ref]),
                               atol=1e-12)


def test_states_and_moments_match_the_state_solvers():
    """The combined solve gives the same states as the state-only references,
    and moments equal direct expectation values."""
    import qutip as qt
    from qmembench.reference import (TIGHT_SOLVER_OPTIONS, spin_boson_qutip_states,
                                     spin_boson_states_moments, stageA_qutip_states_finiteT,
                                     stageA_states_moments, truncated_mode_state)
    t = np.linspace(0, 2, 21)
    ms = truncated_mode_state("thermal", 0.7, 5)
    s1, mom1 = stageA_states_moments(dm("+"), t, 5.0, 1.0, 0.4, 5, mode_state=ms,
                                     options=TIGHT_SOLVER_OPTIONS)
    s2 = stageA_qutip_states_finiteT(dm("+"), t, 5.0, 1.0, 0.4, n_fock=5, mode_state=ms,
                                     options=TIGHT_SOLVER_OPTIONS)
    np.testing.assert_allclose(np.asarray(s1), np.asarray(s2), atol=1e-12)
    assert mom1.shape == (1, 2, 21)
    assert mom1[0, 0, 0] == pytest.approx(np.real(np.trace(ms @ np.diag(np.arange(5)))), abs=1e-12)
    modes = [(1.0, 0.5, 0.4), (1.6, 0.5, 0.8)]
    s3, mom3 = spin_boson_states_moments(dm("+i"), t, 1.0, 0.0, modes, 4, options=TIGHT_SOLVER_OPTIONS)
    s4 = spin_boson_qutip_states(dm("+i"), t, 1.0, 0.0, modes, n_fock=4, options=TIGHT_SOLVER_OPTIONS)
    np.testing.assert_allclose(np.asarray(s3), np.asarray(s4), atol=1e-12)
    assert np.all(mom3[:, 1, :] >= mom3[:, 0, :] ** 2 - 1e-9)       # variance >= 0


def test_truncated_mode_states():
    from qmembench.reference import truncated_mode_state
    for kind, mean in (("fock", 2), ("coherent", 2.0), ("thermal", 2.0), ("vacuum", 0)):
        r = truncated_mode_state(kind, mean, 60)          # thermal tail (2/3)^60 ~ 3e-11
        n = np.real(np.trace(r @ np.diag(np.arange(60))))
        n2 = np.real(np.trace(r @ np.diag(np.arange(60) ** 2)))
        assert np.trace(r).real == pytest.approx(1.0)
        assert n == pytest.approx(mean, abs=1e-4)
        var = {"fock": 0.0, "coherent": mean, "thermal": mean * (mean + 1), "vacuum": 0.0}[kind]
        assert n2 - n ** 2 == pytest.approx(var, abs=1e-3)

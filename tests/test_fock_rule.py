"""Tests for the required-Fock-dimension machinery."""
import numpy as np
import pytest

from qmembench.experiments import (FOCK_C_GRID, _fock_rule_calibrate, fock_rule_mean_only,
                                   fock_rule_variance_aware)
from qmembench.reference import (required_fock_dimension,
                                 stageA_finiteT_with_occupation,
                                 spin_boson_with_occupation)
from qmembench.measures import dm

TIMES = np.linspace(0.0, 6.0, 60)


def test_fock_rules_monotone_and_sane():
    assert fock_rule_mean_only(0.0) >= 2
    assert fock_rule_variance_aware(0.0, 0.0) >= 2
    vals = [fock_rule_mean_only(n) for n in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # variance-aware rule grows with the width at fixed mean
    assert fock_rule_variance_aware(1.0, 1.2) >= fock_rule_variance_aware(1.0, 0.3)


def test_rule_at_exact_integer_predictor_ignores_float_noise():
    """<n> + c*sigma = 4 exactly (coherent state, <n> = 1, c = 3): the rule is
    ceil(4) + 1 = 5, even when sigma carries a last-digit rounding error."""
    assert fock_rule_variance_aware(1.0, 1.0000000000000004, c=3.0) == 5
    ci = int(np.argmin(np.abs(FOCK_C_GRID - 3.0)))
    pred = [1.0 + c * 1.0000000000000004 for c in FOCK_C_GRID]
    per_d = {str(d): {"trajectory_dev": 0.0 if d >= 6 else 1.0, "eps_NM": 0.0}
             for d in range(2, 10)}
    pt = {"family": "controlled", "param": 1.0, "kind": "coherent", "d_min": 2, "d_ref": 10,
          "per_d": per_d, "predictor": pred}
    out = _fock_rule_calibrate([pt], [1e-2], [1e-2])
    assert pt["d_req"]["trajectory_dev<0.01"]["d_req"] == 6
    # d_rule(c = 3) = 5 < 6; c = 3.25 gives ceil(4.25) + 1 = 6
    assert out["calibration"]["trajectory_dev<0.01"]["c_all_points"] == 3.25
    assert FOCK_C_GRID[ci] == 3.0


def test_thermal_has_larger_number_std_than_coupling_at_similar_mean():
    """The physical crux: at comparable peak occupation, thermal statistics
    are broader (super-Poissonian) than coupling-driven occupation."""
    _, n_T, std_T = stageA_finiteT_with_occupation(dm("+"), TIMES, 5.0, 1.0,
                                                   n_th=1.0, n_fock=8)
    _, n_g, std_g = spin_boson_with_occupation(dm("+"), TIMES, 1.0, 0.0,
                                               [(1.0, 0.7, 0.4)], n_fock=8)
    # widths per unit mean: thermal broader
    assert std_T / max(n_T, 1e-9) > std_g / max(n_g, 1e-9)


def test_required_fock_dimension_picks_converged():
    # synthetic: trajectories identical for d>=4, different for d=2,3
    ref = [np.eye(2, dtype=complex) / 2 for _ in range(5)]
    states = {
        2: [r + 0.5 for r in ref],
        3: [r + 0.05 for r in ref],
        4: [r + 0.001 for r in ref],
        5: ref,
    }
    d, censored = required_fock_dimension(states, 5, eps=1e-2)
    assert d == 4 and censored is False


def test_required_fock_dimension_reports_censoring():
    ref = [np.eye(2, dtype=complex) / 2 for _ in range(4)]
    # nothing below d_ref converges -> censored
    states = {2: [r + 0.5 for r in ref], 3: [r + 0.4 for r in ref], 4: ref}
    d, censored = required_fock_dimension(states, 4, eps=1e-2)
    assert d == 4 and censored is True


def test_occupation_grows_with_temperature():
    _, n0, _ = stageA_finiteT_with_occupation(dm("1"), TIMES, 5.0, 1.0, n_th=0.1, n_fock=6)
    _, n1, _ = stageA_finiteT_with_occupation(dm("1"), TIMES, 5.0, 1.0, n_th=1.0, n_fock=6)
    assert n1 > n0


def test_occupation_grows_with_coupling():
    _, n_lo, _ = spin_boson_with_occupation(dm("+"), TIMES, 1.0, 0.0, [(1.0, 0.2, 0.4)], n_fock=6)
    _, n_hi, _ = spin_boson_with_occupation(dm("+"), TIMES, 1.0, 0.0, [(1.0, 0.7, 0.4)], n_fock=6)
    assert n_hi > n_lo

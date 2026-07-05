"""Tests for the bootstrap / bias-correction layer."""
import numpy as np
import pytest

from qmembench.statistics import (blp_point_estimate, bootstrap_blp,
                                  null_floor, resample_counts, rho_from_triplet)


def _triplet(x, y, z, shots=4096):
    """Build X/Y/Z counts realising a target Bloch vector exactly (no noise)."""
    out = {}
    for comp, b in zip((x, y, z), ("X", "Y", "Z")):
        n0 = int(round((1 + comp) / 2 * shots))
        out[b] = {"0": n0, "1": shots - n0}
    return out


def _pair_counts_from_bloch(traj_a, traj_b, shots=4096):
    steps = list(range(len(traj_a)))
    pc = {"+": {}, "-": {}}
    for k, (ba, bb) in enumerate(zip(traj_a, traj_b)):
        pc["+"][k] = _triplet(*ba, shots=shots)
        pc["-"][k] = _triplet(*bb, shots=shots)
    return pc, steps


def test_resample_preserves_shot_total():
    rng = np.random.default_rng(0)
    c = {"0": 3000, "1": 1096}
    for _ in range(50):
        r = resample_counts(c, rng)
        assert r["0"] + r["1"] == 4096


def test_rho_from_triplet_is_physical():
    rho = rho_from_triplet(_triplet(0.9, 0.0, 0.0))
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.linalg.eigvalsh(rho) > -1e-9)


def test_point_estimate_zero_on_monotone_decay():
    # antipodal equatorial states whose distinguishability decays monotonically
    D0 = np.exp(-np.linspace(0, 3, 40))
    traj_a = [(d, 0, 0) for d in D0]          # +x shrinking
    traj_b = [(-d, 0, 0) for d in D0]         # -x shrinking
    pc, steps = _pair_counts_from_bloch(traj_a, traj_b)
    blp, D = blp_point_estimate(pc, steps, ("+", "-"))
    assert D[0] == pytest.approx(1.0, abs=1e-6)
    assert blp == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_ci_brackets_point():
    t = np.linspace(0, 4 * np.pi, 40)
    env = np.abs(np.cos(t)) * np.exp(-0.1 * t)
    traj_a = [(e, 0, 0) for e in env]
    traj_b = [(-e, 0, 0) for e in env]
    pc, steps = _pair_counts_from_bloch(traj_a, traj_b, shots=8192)
    res = bootstrap_blp(pc, steps, ("+", "-"), n_boot=200, seed=1)
    assert res["N_BLP_ci_low"] <= res["N_BLP_point"] <= res["N_BLP_ci_high"]
    assert res["N_BLP_boot_std"] > 0


def test_debiasing_reduces_noise_rectification():
    """On a truly monotone (zero-backflow) signal, finite shots inflate the
    raw point estimate; the debiased estimator must be substantially smaller."""
    D0 = np.linspace(1.0, 0.0, 60)  # strictly decreasing -> true N_BLP = 0
    traj_a = [(d, 0, 0) for d in D0]
    traj_b = [(-d, 0, 0) for d in D0]
    pc, steps = _pair_counts_from_bloch(traj_a, traj_b, shots=1024)
    res = bootstrap_blp(pc, steps, ("+", "-"), n_boot=300, seed=2, z=1.0)
    assert res["N_BLP_debiased"] <= res["N_BLP_point"]
    assert res["N_BLP_debiased"] < 0.5 * res["N_BLP_point"] + 1e-9


def test_null_floor_positive_and_shrinks_with_shots():
    D0 = np.linspace(1.0, 0.0, 60)
    traj_a = [(d, 0, 0) for d in D0]
    traj_b = [(-d, 0, 0) for d in D0]
    pc_lo, steps = _pair_counts_from_bloch(traj_a, traj_b, shots=512)
    pc_hi, _ = _pair_counts_from_bloch(traj_a, traj_b, shots=32768)
    # more shots -> smaller bias floor (statistical, so average a few seeds
    # implicitly via the larger sample); check the expected ordering holds.
    assert null_floor(pc_hi, steps, ("+", "-")) <= null_floor(pc_lo, steps, ("+", "-"))

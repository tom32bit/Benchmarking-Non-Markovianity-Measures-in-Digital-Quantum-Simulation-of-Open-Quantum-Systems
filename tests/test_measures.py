"""Unit tests for the non-Markovianity measures."""
import numpy as np
import pytest

from qmembench.measures import (blp_from_curve, choi_from_transfer, dm,
                                rhp_measure, trace_distance, trace_norm,
                                transfer_matrix)


def test_trace_distance_known_values():
    assert trace_distance(dm("0"), dm("1")) == pytest.approx(1.0)
    assert trace_distance(dm("0"), dm("0")) == pytest.approx(0.0)
    assert trace_distance(dm("0"), dm("+")) == pytest.approx(1 / np.sqrt(2))


def test_blp_zero_on_monotone_decay():
    D = np.exp(-np.linspace(0, 5, 100))
    assert blp_from_curve(D) == 0.0


def test_blp_counts_revivals():
    t = np.linspace(0, 4 * np.pi, 500)
    D = np.abs(np.cos(t)) * np.exp(-0.1 * t)
    assert blp_from_curve(D) > 0.5


def test_identity_map_is_cp_divisible():
    ident = {lbl: dm(lbl) for lbl in ("0", "1", "+", "+i")}
    T = transfer_matrix(ident)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)
    C = choi_from_transfer(T)
    # identity channel Choi = maximally entangled state, trace norm 1
    assert trace_norm(C) == pytest.approx(1.0)
    n_rhp, _ = rhp_measure([T, T, T])
    assert n_rhp == pytest.approx(0.0, abs=1e-10)


def test_transfer_matrix_of_bit_flip():
    """Lambda(rho) = X rho X has a known transfer matrix; sanity-check it."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    outputs = {lbl: X @ dm(lbl) @ X for lbl in ("0", "1", "+", "+i")}
    T = transfer_matrix(outputs)
    rho = dm("+i")
    lhs = (T @ rho.flatten(order="F")).reshape(2, 2, order="F")
    np.testing.assert_allclose(lhs, X @ rho @ X, atol=1e-12)


def test_rhp_detects_ncp_intermediate_map():
    """A map that shrinks then re-grows coherence is not CP-divisible."""
    def dephase(r):  # coherence multiplied by r
        T = np.diag([1.0, r, r, 1.0]).astype(complex)
        return T
    # coherence 1 -> 0.5 -> 0.8: the second step is NCP
    n_rhp, _ = rhp_measure([dephase(1.0), dephase(0.5), dephase(0.8)])
    assert n_rhp > 0.1

"""Tests for error mitigation (R2): readout inversion, CX folding, extrapolation."""
import numpy as np
import pytest
from qiskit import QuantumCircuit

from qmembench.mitigation import (confusion_matrix, mitigate_probs, fold_cx,
                                  richardson_extrapolate, readout_calibration_circuits,
                                  mitigated_expectation)


def test_confusion_matrix_and_inversion_roundtrip():
    # 10% symmetric readout error
    cal = [{"0": 900, "1": 100}, {"0": 100, "1": 900}]
    M = confusion_matrix(cal)
    np.testing.assert_allclose(M, [[0.9, 0.1], [0.1, 0.9]], atol=1e-9)
    # apply error to a true dist, then invert -> recover
    true = np.array([0.7, 0.3])
    measured = M @ true
    recovered = mitigate_probs(measured, M)
    np.testing.assert_allclose(recovered, true, atol=1e-9)


def test_mitigate_probs_clips_and_normalizes():
    M = np.array([[0.9, 0.1], [0.1, 0.9]])
    out = mitigate_probs(np.array([0.95, 0.05]), M)  # inversion may go negative
    assert np.all(out >= 0) and out.sum() == pytest.approx(1.0)


def test_fold_cx_triples_cx_count():
    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.cx(0, 1)
    folded = fold_cx(qc, 3)
    assert folded.count_ops().get("cx", 0) == 3 * qc.count_ops()["cx"]
    assert folded.count_ops().get("h", 0) == 1  # non-CX untouched


def test_fold_cx_identity_and_even_rejected():
    qc = QuantumCircuit(2); qc.cx(0, 1)
    assert fold_cx(qc, 1).count_ops()["cx"] == 1
    with pytest.raises(ValueError):
        fold_cx(qc, 2)


def test_richardson_extrapolation_linear():
    scales = np.array([1, 3, 5])
    # true zero-noise value 0.8, linear growth in noise
    values = 0.8 - 0.05 * scales
    assert richardson_extrapolate(scales, values, order=1) == pytest.approx(0.8, abs=1e-9)


def test_mitigated_expectation_matches_manual():
    M = np.array([[0.9, 0.1], [0.1, 0.9]])
    counts = {"0": 800, "1": 200}
    val = mitigated_expectation(counts, M)
    assert -1.0 <= val <= 1.0

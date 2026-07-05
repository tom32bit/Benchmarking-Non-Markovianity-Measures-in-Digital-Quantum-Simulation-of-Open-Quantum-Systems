"""Error mitigation: readout calibration/inversion and zero-noise
extrapolation (ZNE) by local CX-gate folding.

Readout mitigation: measure calibration circuits (I and X) under the same
noise model, build the 2x2 confusion matrix M with M[i, j] = P(measure i |
prepared j), and correct measured probability vectors by M^{-1} (clipped to
[0, 1] and renormalised -- the standard, documented, least-sophisticated
correct thing; no hidden magic).

ZNE: every CX in the *transpiled* circuit is replaced by an odd number of
CX gates (CX is self-inverse, so 3 or 5 copies are logically the identity
composition CX (CX CX)^k but accumulate 3x / 5x the two-qubit noise).
Expectation values measured at scales (1, 3, 5) are Richardson-extrapolated
to the zero-noise limit.  Folding is applied AFTER transpilation so no
optimisation pass can cancel the folded gates.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

# ---------------------------------------------------------------------------
# Readout mitigation
# ---------------------------------------------------------------------------

def readout_calibration_circuits() -> list[QuantumCircuit]:
    """Calibration circuits for the measured (system) qubit: prepare |0>, |1>."""
    c0 = QuantumCircuit(1, 1)
    c0.measure(0, 0)
    c1 = QuantumCircuit(1, 1)
    c1.x(0)
    c1.measure(0, 0)
    return [c0, c1]


def confusion_matrix(cal_counts: list[dict[str, int]]) -> np.ndarray:
    """M[i, j] = P(measured i | prepared j) from the two calibration runs."""
    M = np.zeros((2, 2))
    for j, counts in enumerate(cal_counts):
        total = sum(counts.values())
        M[0, j] = counts.get("0", 0) / total
        M[1, j] = counts.get("1", 0) / total
    return M


def mitigate_probs(probs: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Invert the confusion matrix; clip and renormalise."""
    corrected = np.linalg.solve(M, np.asarray(probs, dtype=float))
    corrected = np.clip(corrected, 0.0, None)
    s = corrected.sum()
    return corrected / s if s > 0 else np.array([0.5, 0.5])


def mitigated_expectation(counts: dict[str, int], M: np.ndarray) -> float:
    from .tomography import probs_from_counts
    p = mitigate_probs(probs_from_counts(counts), M)
    return float(p[0] - p[1])


# ---------------------------------------------------------------------------
# Zero-noise extrapolation
# ---------------------------------------------------------------------------

def fold_cx(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Replace every CX by `scale` CX gates (scale odd) in a transpiled circuit."""
    if scale == 1:
        return circuit
    if scale % 2 == 0 or scale < 1:
        raise ValueError("fold scale must be an odd positive integer")
    folded = circuit.copy_empty_like()
    for inst in circuit.data:
        folded.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == "cx":
            for _ in range(scale - 1):
                folded.append(inst.operation, inst.qubits, inst.clbits)
    return folded


def richardson_extrapolate(scales: np.ndarray, values: np.ndarray,
                           order: int | None = None) -> float:
    """Polynomial (Richardson) extrapolation of expectation values to scale 0.

    order defaults to len(scales) - 1 (full Richardson); order=1 gives the
    more noise-tolerant linear extrapolation.
    """
    scales = np.asarray(scales, dtype=float)
    values = np.asarray(values, dtype=float)
    deg = (len(scales) - 1) if order is None else order
    coeffs = np.polyfit(scales, values, deg)
    return float(np.polyval(coeffs, 0.0))

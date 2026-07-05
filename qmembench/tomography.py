"""Shot-based single-qubit state estimation (the honest NISQ-style pipeline).

The system qubit is measured in the X, Y and Z bases with finite shots;
the Bloch vector is reconstructed and, if finite-shot noise pushes it
outside the Bloch ball, radially projected back (the excess is returned so
experiments can report it rather than hide it).
"""

from __future__ import annotations

import numpy as np

from .measures import bloch_norm_excess, rho_from_bloch

BASES = ("X", "Y", "Z")


def expectation_from_counts(counts: dict[str, int]) -> float:
    """<sigma> = (n_0 - n_1) / shots for a single-clbit measurement."""
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    total = n0 + n1
    if total == 0:
        raise ValueError("empty counts")
    return (n0 - n1) / total


def rho_from_pauli_counts(counts_xyz: dict[str, dict[str, int]]) -> tuple[np.ndarray, float]:
    """Reconstruct rho from {basis: counts}; returns (rho, bloch_excess)."""
    r = np.array([expectation_from_counts(counts_xyz[b]) for b in BASES])
    return rho_from_bloch(r, project=True), bloch_norm_excess(r)


def probs_from_counts(counts: dict[str, int]) -> np.ndarray:
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    total = n0 + n1
    return np.array([n0, n1], dtype=float) / total

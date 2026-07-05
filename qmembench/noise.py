"""Device-noise models for the robustness study (Phase 3).

A deliberately simple, fully documented noise model so every result is
attributable: depolarizing error on 1q gates (sx, x), depolarizing error on
the 2q gate (cx), and symmetric readout error.  Default magnitudes are
representative of 2024-era IBM superconducting devices (order of magnitude:
1q ~ 3e-4, 2q ~ 7e-3, readout ~ 1.5e-2); they are *parameters*, not claims
about any specific chip.

The noise-strength sweep multiplies all three probabilities by a scale
factor s (capped at physical maxima), mapping the degradation of the
simulated memory signature N_BLP(s).
"""

from __future__ import annotations

import numpy as np
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

DEFAULT_P1 = 3e-4    # 1-qubit gate depolarizing probability
DEFAULT_P2 = 7e-3    # 2-qubit gate depolarizing probability
DEFAULT_PRO = 1.5e-2  # readout bit-flip probability

# Physical caps: depolarizing_error requires p <= 4^n/(4^n - 1) normalisation;
# we cap conservatively at total depolarisation and readout at 0.5.
_CAP_P1, _CAP_P2, _CAP_PRO = 1.0, 1.0, 0.5


def build_noise_model(scale: float = 1.0, p1: float = DEFAULT_P1,
                      p2: float = DEFAULT_P2, p_ro: float = DEFAULT_PRO) -> NoiseModel | None:
    """Scaled noise model; scale = 0 returns None (ideal simulation)."""
    if scale <= 0.0:
        return None
    sp1 = min(scale * p1, _CAP_P1)
    sp2 = min(scale * p2, _CAP_P2)
    spro = min(scale * p_ro, _CAP_PRO)
    nm = NoiseModel(basis_gates=["cx", "id", "rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(sp1, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(sp2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - spro, spro], [spro, 1 - spro]]))
    return nm


def from_fake_backend(name: str = "FakeManilaV2") -> tuple[NoiseModel, dict]:
    """Realistic device-calibrated noise model from a Qiskit fake backend.

    These snapshots carry a real IBM device's T1/T2, per-gate errors and
    readout error -- the accepted stand-in when hardware access is
    unavailable. IMPORTANT (and stated in the plan): this model is still
    MARKOVIAN; it does not capture the device's own non-Markovian noise. That
    gap is addressed head-on by the quasi-static injection below, so the
    'your noise is Markovian' objection is answered inside the study.
    """
    from qiskit.providers import fake_provider as fp
    backend = getattr(fp, name)()
    nm = NoiseModel.from_backend(backend)
    meta = {"backend": name, "markovian": True,
            "basis_gates": list(nm.basis_gates)}
    return nm, meta


def quasistatic_angles(sigma: float, n_ensemble: int, seed: int = 0) -> np.ndarray:
    """Gaussian-distributed coherent dephasing angles for a quasi-static
    (temporally correlated -> non-Markovian) noise ensemble."""
    return np.random.default_rng(seed).normal(0.0, sigma, n_ensemble)


def noise_params_dict(scale: float, p1: float = DEFAULT_P1, p2: float = DEFAULT_P2,
                      p_ro: float = DEFAULT_PRO) -> dict:
    """Provenance record of the effective noise parameters at a given scale."""
    return {
        "scale": scale,
        "p1_effective": min(scale * p1, _CAP_P1) if scale > 0 else 0.0,
        "p2_effective": min(scale * p2, _CAP_P2) if scale > 0 else 0.0,
        "p_readout_effective": min(scale * p_ro, _CAP_PRO) if scale > 0 else 0.0,
        "model": "depolarizing(sx,x)+depolarizing(cx)+symmetric readout",
    }

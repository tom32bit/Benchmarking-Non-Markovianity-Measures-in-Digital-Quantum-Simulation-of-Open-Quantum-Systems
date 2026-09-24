"""Record provenance and the linearity helpers behind the pair/Fock checks."""
import numpy as np

from qmembench.experiments import _provenance, _six_from_ic
from qmembench.measures import dm
from qmembench.reference import analytic_stageA_states


def test_provenance_records_code_and_library_versions():
    prov = _provenance({"x": 1}, seed=3)
    assert "git_commit" in prov                  # None only if git is unavailable
    assert set(prov["versions"]) >= {"numpy", "scipy", "qiskit", "qiskit_aer", "qutip"}
    assert prov["seed"] == 3 and prov["params"] == {"x": 1}


def test_six_from_ic_reconstructs_the_other_canonical_states():
    times = np.linspace(0.0, 3.0, 31)
    states = {l: analytic_stageA_states(dm(l), times, 5.0, 1.0)
              for l in ("0", "1", "+", "-", "+i", "-i")}
    six = _six_from_ic({l: states[l] for l in ("0", "1", "+", "+i")})
    for l in ("-", "-i"):
        np.testing.assert_allclose(six[l], np.asarray(states[l]), atol=1e-12)

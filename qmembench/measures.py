"""Non-Markovianity measures and single-qubit state utilities.

Implements:
  * trace distance and the discrete-time BLP (Breuer-Laine-Piilo) measure
    [Breuer, Laine, Piilo, PRL 103, 210401 (2009)],
  * the RHP (Rivas-Huelga-Plenio) CP-divisibility monitor
    [Rivas, Huelga, Plenio, PRL 105, 050403 (2010)] via Choi-matrix
    negativity of the reconstructed intermediate map,
  * dynamical-map (transfer-matrix) reconstruction from four
    informationally complete input states |0>, |1>, |+>, |+i>.

Conventions
-----------
Basis order (|0>, |1>) with |1> = excited state.  Column-major (Fortran)
vectorisation: vec(E_ij) has a 1 at index i + 2j.  All density matrices are
2x2 complex numpy arrays unless stated otherwise.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Pauli matrices and canonical states
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = {"X": SX, "Y": SY, "Z": SZ}

KET = {
    "0": np.array([1, 0], dtype=complex),
    "1": np.array([0, 1], dtype=complex),
    "+": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "-": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "+i": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "-i": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}


def dm(label: str) -> np.ndarray:
    """Density matrix of a canonical pure state given its label."""
    k = KET[label]
    return np.outer(k, k.conj())


def canonical_pairs() -> list[tuple[str, str]]:
    """Antipodal Bloch pairs used for the BLP maximisation."""
    return [("0", "1"), ("+", "-"), ("+i", "-i")]


# ---------------------------------------------------------------------------
# Bloch-vector utilities (used by the shot-based tomography pipeline)
# ---------------------------------------------------------------------------

def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """Bloch vector (x, y, z) of a single-qubit density matrix."""
    return np.real(np.array([np.trace(rho @ SX), np.trace(rho @ SY), np.trace(rho @ SZ)]))


def rho_from_bloch(r: np.ndarray, project: bool = True) -> np.ndarray:
    """Density matrix from a Bloch vector.

    With ``project=True`` an unphysical vector (|r| > 1, possible with finite
    shots / unmitigated readout error) is radially projected onto the Bloch
    ball.  The projection is recorded nowhere silently: callers who care use
    :func:`bloch_norm_excess` first.
    """
    r = np.asarray(r, dtype=float)
    n = np.linalg.norm(r)
    if project and n > 1.0:
        r = r / n
    return 0.5 * (I2 + r[0] * SX + r[1] * SY + r[2] * SZ)


def bloch_norm_excess(r: np.ndarray) -> float:
    """How far outside the Bloch ball a reconstructed vector lies (0 if inside)."""
    return max(0.0, float(np.linalg.norm(r)) - 1.0)


# ---------------------------------------------------------------------------
# Trace distance and BLP
# ---------------------------------------------------------------------------

def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """D(rho1, rho2) = (1/2) ||rho1 - rho2||_1 for Hermitian inputs."""
    delta = rho1 - rho2
    # Hermitise defensively (guards against numerical asymmetry from solvers).
    delta = 0.5 * (delta + delta.conj().T)
    eig = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(eig)))


def trace_distance_curve(states1: list[np.ndarray], states2: list[np.ndarray]) -> np.ndarray:
    """D(t) along two evolved-state trajectories of equal length."""
    return np.array([trace_distance(a, b) for a, b in zip(states1, states2)])


def blp_from_curve(D: np.ndarray) -> float:
    """Discrete-time BLP measure: sum of positive increments of D(t).

    This is the standard discretisation of
    N_BLP = max_pair \\int_{dD/dt > 0} (dD/dt) dt
    for a *fixed* pair; the pair maximisation is done by the caller over
    :func:`canonical_pairs` (and, for Stage A, checked against the known
    analytic optimum D(t) = |G(t)|).
    """
    dD = np.diff(np.asarray(D, dtype=float))
    return float(np.sum(dD[dD > 0.0]))


def blp_measure(pair_curves: dict[str, np.ndarray]) -> tuple[float, str]:
    """Maximise the discrete BLP over supplied pair-labelled D(t) curves.

    Returns (N_BLP, label_of_optimal_pair).

    NOTE: over the three canonical antipodal pairs this is a LOWER BOUND on
    the true BLP measure (which maximises over all initial-state pairs). It is
    provably tight for Stage A (analytic optimum = equatorial pair). For other
    models use :func:`blp_measure_optimized` to certify how close the
    canonical value is to the true optimum.
    """
    best_label, best_val = "", -1.0
    for label, curve in pair_curves.items():
        val = blp_from_curve(curve)
        if val > best_val:
            best_val, best_label = val, label
    return best_val, best_label


def _antipodal_bloch_pair(theta: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return 0.5 * (I2 + n[0] * SX + n[1] * SY + n[2] * SZ), \
           0.5 * (I2 - n[0] * SX - n[1] * SY - n[2] * SZ)


def blp_measure_optimized(channel, n_random: int = 40, n_refine: int = 30,
                          seed: int = 0) -> dict:
    """Optimise the BLP pair over the Bloch sphere for a single-qubit channel.

    ``channel(rho0) -> [rho(t)]`` evolves an initial state to its trajectory.
    Optimal BLP state pairs are provably ORTHOGONAL boundary pairs [Wissmann,
    Karlsson, Laine, Piilo, Breuer, PRA 86, 062108 (2012)], i.e. antipodal
    pure states on the Bloch sphere for a qubit -- so optimising over
    (theta, phi) searches the COMPLETE space of candidate optimal pairs, and
    the returned value estimates the true BLP measure itself (up to search
    resolution), not merely a tighter lower bound. Seeded with the three
    canonical axes (guaranteeing optimised >= canonical), plus random starts
    and coordinate refinement. Returns the optimised N_BLP, the canonical-pair
    value, and their gap (a certificate of how tight the canonical bound is).
    """
    rng = np.random.default_rng(seed)

    def blp_at(theta, phi):
        r1, r2 = _antipodal_bloch_pair(theta, phi)
        D = trace_distance_curve(channel(r1), channel(r2))
        return blp_from_curve(D)

    # seed with the 3 canonical antipodal axes so the optimum is guaranteed
    # >= the canonical value, then add random starts
    seeds = [(0.0, 0.0), (np.pi / 2, 0.0), (np.pi / 2, np.pi / 2)]
    best = (-1.0, 0.0, 0.0)
    for th, ph in seeds:
        v = blp_at(th, ph)
        if v > best[0]:
            best = (v, th, ph)
    for _ in range(n_random):
        th = np.arccos(1 - 2 * rng.random())
        ph = 2 * np.pi * rng.random()
        v = blp_at(th, ph)
        if v > best[0]:
            best = (v, th, ph)
    val, th, ph = best
    step = 0.4
    for _ in range(n_refine):  # coordinate ascent
        improved = False
        for dth, dph in ((step, 0), (-step, 0), (0, step), (0, -step)):
            v = blp_at(th + dth, ph + dph)
            if v > val:
                val, th, ph, improved = v, th + dth, ph + dph, True
        if not improved:
            step *= 0.5
    canonical = max(blp_at(0, 0), blp_at(np.pi / 2, 0), blp_at(np.pi / 2, np.pi / 2))
    return {"N_BLP_optimized": float(val), "N_BLP_canonical": float(canonical),
            "gap": float(val - canonical), "theta": float(th), "phi": float(ph)}


# ---------------------------------------------------------------------------
# Dynamical-map reconstruction and the RHP measure
# ---------------------------------------------------------------------------

# Informationally complete single-qubit inputs used throughout the project.
IC_INPUTS = ("0", "1", "+", "+i")


def _vec(m: np.ndarray) -> np.ndarray:
    """Column-major vectorisation."""
    return m.flatten(order="F")


def _unvec(v: np.ndarray) -> np.ndarray:
    return v.reshape((2, 2), order="F")


def transfer_matrix(outputs: dict[str, np.ndarray]) -> np.ndarray:
    """4x4 transfer (superoperator) matrix T with T vec(rho) = vec(Lambda(rho)).

    ``outputs`` maps each label in IC_INPUTS to Lambda(dm(label)).  Uses
    linearity to obtain the action on the operator basis E_ij = |i><j|:

        Lambda(E_00) = out["0"],  Lambda(E_11) = out["1"],
        Lambda(E_01) = out["+"] + i out["+i"] - (1+i)/2 (out["0"] + out["1"]),
        Lambda(E_10) = Lambda(E_01)^dagger        (Hermiticity preservation).
    """
    o0, o1 = outputs["0"], outputs["1"]
    op, oi = outputs["+"], outputs["+i"]
    e01 = op + 1j * oi - (1 + 1j) / 2 * (o0 + o1)
    e10 = e01.conj().T
    T = np.zeros((4, 4), dtype=complex)
    # column index for E_ij under column-major vec is i + 2j
    T[:, 0 + 2 * 0] = _vec(o0)   # E_00
    T[:, 1 + 2 * 0] = _vec(e10)  # E_10
    T[:, 0 + 2 * 1] = _vec(e01)  # E_01
    T[:, 1 + 2 * 1] = _vec(o1)   # E_11
    return T


def choi_from_transfer(T: np.ndarray) -> np.ndarray:
    """Normalised Choi matrix C = (1/2) sum_ij E_ij (x) Lambda(E_ij).

    For a trace-preserving map, tr C = 1; the map is CP iff C >= 0, in which
    case ||C||_1 = 1.  Choi negativity (||C||_1 - 1 > 0) certifies NCP.
    """
    C = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            E = np.zeros((2, 2), dtype=complex)
            E[i, j] = 1.0
            lam_E = _unvec(T @ _vec(E))
            C += np.kron(E, lam_E)
    C = C / 2.0
    return 0.5 * (C + C.conj().T)  # symmetrise against float noise


def trace_norm(m: np.ndarray) -> float:
    return float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (m + m.conj().T)))))


def rhp_curve(transfers: list[np.ndarray], rcond: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """RHP non-CP-divisibility monitor along a discretised evolution.

    For consecutive dynamical maps Lambda(t_k), Lambda(t_{k+1}) the
    intermediate map is V = T_{k+1} pinv(T_k); its Choi negativity
    g_k = ||C(V)||_1 - 1 vanishes iff the step is CP.

    Numerical caveat (documented in the plan, section 5.4): T_k becomes
    singular where |G(t)| = 0 -- exactly the deep non-Markovian regime.  A
    pseudo-inverse with cutoff ``rcond`` is used and a per-step condition
    flag is returned so affected intervals can be reported honestly.

    Returns (g, ill_conditioned_flags) with len = len(transfers) - 1.
    """
    g = np.zeros(len(transfers) - 1)
    flags = np.zeros(len(transfers) - 1, dtype=bool)
    for k in range(len(transfers) - 1):
        Tk, Tk1 = transfers[k], transfers[k + 1]
        sv = np.linalg.svd(Tk, compute_uv=False)
        flags[k] = (sv.min() / sv.max()) < rcond
        V = Tk1 @ np.linalg.pinv(Tk, rcond=rcond)
        g[k] = max(0.0, trace_norm(choi_from_transfer(V)) - 1.0)
    return g, flags


def rhp_measure(transfers: list[np.ndarray], rcond: float = 1e-6) -> tuple[float, float]:
    """Total RHP measure and the fraction of ill-conditioned steps."""
    g, flags = rhp_curve(transfers, rcond=rcond)
    return float(np.sum(g)), float(np.mean(flags)) if len(flags) else 0.0

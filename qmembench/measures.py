"""Non-Markovianity measures and single-qubit state utilities.

Implements:
  * trace distance and the discrete-time BLP (Breuer-Laine-Piilo) measure
    [Breuer, Laine, Piilo, PRL 103, 210401 (2009)],
  * the BLP pair optimisation over all antipodal pure-state pairs, exact via
    linearity of the reduced map (Pauli images from the IC inputs),
  * the RHP (Rivas-Huelga-Plenio) CP-divisibility monitor
    [Rivas, Huelga, Plenio, PRL 105, 050403 (2010)] via Choi-matrix
    negativity of the reconstructed intermediate map, with per-step
    conditioning diagnostics,
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
from scipy.optimize import minimize

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


# ---------------------------------------------------------------------------
# BLP pair optimisation over the whole Bloch sphere
# ---------------------------------------------------------------------------
#
# Optimal BLP pairs are orthogonal pure states [Wissmann, Karlsson, Laine,
# Piilo, Breuer, PRA 86, 062108 (2012)]; for a qubit these are the antipodal
# pairs +-n on the Bloch sphere. The reduced dynamics is linear, so
# rho_1(t) - rho_2(t) = Lambda_t(n.sigma) = sum_k n_k Lambda_t(sigma_k): the
# three Pauli images Lambda_t(sigma_k), obtained from the four IC inputs, give
# D(t) for EVERY pair exactly. Searching over pairs then needs no further
# evolution, which makes a dense, budget-checked search affordable.

# Canonical antipodal pairs as Bloch directions (the pairs of canonical_pairs()).
_CANONICAL_AXES = {"0|1": (0.0, 0.0, 1.0), "+|-": (1.0, 0.0, 0.0),
                   "+i|-i": (0.0, 1.0, 0.0)}


def pauli_images_from_states(states_by_label: dict[str, list[np.ndarray]]) -> np.ndarray:
    """Lambda_t(sigma_x), Lambda_t(sigma_y), Lambda_t(sigma_z) along a trajectory.

    ``states_by_label`` maps each label in IC_INPUTS to its evolved states.
    By linearity: sigma_z = rho_0 - rho_1, sigma_x = 2 rho_+ - rho_0 - rho_1,
    sigma_y = 2 rho_+i - rho_0 - rho_1. Returns an array of shape (n_t, 3, 2, 2).
    """
    o0, o1 = np.asarray(states_by_label["0"]), np.asarray(states_by_label["1"])
    op, oi = np.asarray(states_by_label["+"]), np.asarray(states_by_label["+i"])
    return np.stack([2.0 * op - o0 - o1, 2.0 * oi - o0 - o1, o0 - o1], axis=1)


def _bloch_linear_part(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of Lambda_t(sigma_k) = c0_k I + sum_i M_ik sigma_i.

    Returns M with shape (n_t, 3, 3) and c0 with shape (n_t, 3); c0 vanishes
    for a trace-preserving map.
    """
    paulis = np.stack([SX, SY, SZ])
    M = 0.5 * np.real(np.einsum("iab,tkba->tik", paulis, images))
    c0 = 0.5 * np.real(np.einsum("tkaa->tk", images))
    return M, c0


def _blp_of_directions(M: np.ndarray, c0: np.ndarray, directions: np.ndarray,
                       chunk: int = 1024) -> np.ndarray:
    """Discrete BLP of the pair +-n for each row n of ``directions``.

    Lambda_t(n.sigma) = a0 I + a.sigma has eigenvalues a0 +- |a|, so
    D_n(t) = (1/2) ||Lambda_t(n.sigma)||_1 = max(|a0|, |a|), a = M n, a0 = c0.n.
    """
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    out = np.empty(len(directions))
    for s in range(0, len(directions), chunk):
        n = directions[s:s + chunk]
        a = np.einsum("tik,nk->tni", M, n)
        a0 = np.einsum("tk,nk->tn", c0, n)
        D = np.maximum(np.abs(a0), np.linalg.norm(a, axis=2))
        dD = np.diff(D, axis=0)
        out[s:s + chunk] = np.sum(np.where(dD > 0.0, dD, 0.0), axis=0)
    return out


def blp_direction_values(images: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Discrete BLP of the antipodal pure-state pair +-n, for each unit vector n."""
    M, c0 = _bloch_linear_part(images)
    return _blp_of_directions(M, c0, directions)


def _hemisphere_lattice(n_points: int) -> np.ndarray:
    """Quasi-uniform Fibonacci lattice on the upper Bloch hemisphere (z > 0).

    One hemisphere suffices because the pair +-n is the pair -n,+n.
    """
    i = np.arange(n_points)
    z = 1.0 - (i + 0.5) / n_points            # uniform in z = uniform in area
    phi = i * np.pi * (3.0 - np.sqrt(5.0))    # golden-angle increments
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _direction(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def blp_optimize_pair(images: np.ndarray, n_grid: int = 4000, n_starts: int = 8,
                      min_separation_deg: float = 10.0, xatol: float = 1e-7,
                      fatol: float = 1e-13, maxiter: int = 4000) -> dict:
    """Maximise the discrete BLP over all antipodal pure-state pairs.

    Search: the three canonical axes plus a Fibonacci lattice of ``n_grid``
    directions on the hemisphere, then Nelder-Mead refinement in (theta, phi)
    from the ``n_starts`` best lattice points that are at least
    ``min_separation_deg`` apart, each stopping at tolerance (xatol, fatol)
    rather than after a fixed number of steps. The whole search is repeated
    with half the lattice; ``budget_change`` is the change of the optimum
    between the two budgets (the convergence check). The canonical axes are
    part of the lattice, so the optimum never falls below the canonical value.
    """
    M, c0 = _bloch_linear_part(images)
    axes = np.array(list(_CANONICAL_AXES.values()))
    cos_sep = np.cos(np.radians(min_separation_deg))
    n_eval = 0

    def value(x):
        nonlocal n_eval
        n_eval += 1
        return float(_blp_of_directions(M, c0, _direction(*x)[None, :])[0])

    def search(k):
        nonlocal n_eval
        grid = np.vstack([axes, _hemisphere_lattice(k)])
        vals = _blp_of_directions(M, c0, grid)
        n_eval += len(grid)
        order = np.argsort(vals)[::-1]
        best_val, best_n = float(vals[order[0]]), grid[order[0]]
        grid_best = best_val
        starts = []
        for idx in order:
            if all(abs(float(grid[idx] @ s)) < cos_sep for s in starts):
                starts.append(grid[idx])
            if len(starts) == n_starts:
                break
        for s in starts:
            x0 = [np.arccos(np.clip(s[2], -1.0, 1.0)), np.arctan2(s[1], s[0])]
            res = minimize(lambda x: -value(x), x0=x0, method="Nelder-Mead",
                           options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter})
            if -res.fun > best_val:
                best_val, best_n = float(-res.fun), _direction(*res.x)
        return best_val, best_n, grid_best

    half_val, _, _ = search(max(n_grid // 2, 1))
    best_val, best_n, grid_best = search(n_grid)
    if best_n[2] < 0.0:          # report the representative with n_z >= 0
        best_n = -best_n
    canonical = dict(zip(_CANONICAL_AXES, map(float, _blp_of_directions(M, c0, axes))))
    best_can = max(canonical, key=canonical.get)
    return {"N_BLP_optimized": best_val,
            "N_BLP_canonical": canonical[best_can],
            "gap": best_val - canonical[best_can],
            "canonical_by_pair": canonical, "best_canonical_pair": best_can,
            "n_opt": [float(v) for v in best_n],
            "theta": float(np.arccos(np.clip(best_n[2], -1.0, 1.0))),
            "phi": float(np.arctan2(best_n[1], best_n[0]) % (2.0 * np.pi)),
            "grid_best": grid_best, "N_BLP_half_budget": half_val,
            "budget_change": abs(best_val - half_val),
            "n_grid": n_grid, "n_starts": n_starts, "n_evaluations": n_eval}


def blp_measure_optimized(channel, n_grid: int = 4000, n_starts: int = 8) -> dict:
    """Optimise the BLP pair over the Bloch sphere for a single-qubit channel.

    ``channel(rho0) -> [rho(t)]`` evolves an initial state to its trajectory.
    Only the four IC inputs are evolved; every other pair follows by linearity
    (see :func:`pauli_images_from_states`). Returns the dict of
    :func:`blp_optimize_pair`.
    """
    states = {lbl: channel(dm(lbl)) for lbl in IC_INPUTS}
    return blp_optimize_pair(pauli_images_from_states(states), n_grid=n_grid,
                             n_starts=n_starts)


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


def rhp_diagnostics(transfers: list[np.ndarray], rcond: float = 1e-6) -> dict:
    """Per-step quantities behind the RHP estimate.

    For each step k, with the intermediate map V = T_{k+1} pinv(T_k):
      g               -- Choi excess ||C(V)||_1 - 1 (clipped at 0), the summand
                         of N_RHP;
      kappa           -- condition number s_max / s_min of T_k (inf if
                         singular); inverting T_k can amplify errors in it by up
                         to ~kappa;
      sv_min, sv_max  -- smallest and largest singular values of T_k;
      choi_trace      -- tr C(V), equal to 1 when V is trace preserving. A
                         departure means the pseudo-inverse dropped part of the
                         map, and g then mixes that artefact with non-CP-ness;
      ill_conditioned -- s_min / s_max < rcond (the flag used by rhp_measure).
    """
    n = len(transfers) - 1
    diag = {key: np.zeros(n) for key in ("g", "kappa", "sv_min", "sv_max", "choi_trace")}
    diag["ill_conditioned"] = np.zeros(n, dtype=bool)
    for k in range(n):
        Tk, Tk1 = transfers[k], transfers[k + 1]
        sv = np.linalg.svd(Tk, compute_uv=False)
        s_min, s_max = float(sv.min()), float(sv.max())
        C = choi_from_transfer(Tk1 @ np.linalg.pinv(Tk, rcond=rcond))
        diag["g"][k] = max(0.0, trace_norm(C) - 1.0)
        diag["kappa"][k] = s_max / s_min if s_min > 0.0 else np.inf
        diag["sv_min"][k], diag["sv_max"][k] = s_min, s_max
        diag["choi_trace"][k] = float(np.real(np.trace(C)))
        diag["ill_conditioned"][k] = (s_min / s_max) < rcond
    return diag


def rhp_curve(transfers: list[np.ndarray], rcond: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """RHP non-CP-divisibility monitor along a discretised evolution.

    For consecutive dynamical maps Lambda(t_k), Lambda(t_{k+1}) the
    intermediate map is V = T_{k+1} pinv(T_k); its Choi excess
    g_k = ||C(V)||_1 - 1 vanishes iff the step is CP. Summing g_k over the
    steps approximates the RHP integral of the rate
    g(t) = lim_{eps->0} (||C_{t+eps,t}||_1 - 1) / eps, so the sum carries no
    factor of the time step.

    Numerical caveat: T_k becomes singular where |G(t)| = 0 -- exactly the deep
    non-Markovian regime. A pseudo-inverse with cutoff ``rcond`` is used and a
    per-step condition flag is returned; :func:`rhp_diagnostics` gives the
    full condition numbers and a trace-preservation check.

    Returns (g, ill_conditioned_flags) with len = len(transfers) - 1.
    """
    diag = rhp_diagnostics(transfers, rcond=rcond)
    return diag["g"], diag["ill_conditioned"]


def rhp_measure(transfers: list[np.ndarray], rcond: float = 1e-6) -> tuple[float, float]:
    """Total RHP measure and the fraction of ill-conditioned steps."""
    g, flags = rhp_curve(transfers, rcond=rcond)
    return float(np.sum(g)), float(np.mean(flags)) if len(flags) else 0.0

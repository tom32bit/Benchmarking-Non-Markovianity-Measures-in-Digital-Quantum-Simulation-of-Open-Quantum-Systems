"""Non-Markovian collision model (comparison embedding, Phase 4).

Scheme (Ciccarello, Palma, Giovannetti, PRA 87, 040103(R) (2013)):
per step k the system S collides with a fresh environment ancilla E_k
(partial exchange, angle theta_s); E_k then collides with E_{k+1}
(partial exchange, angle theta_m) *before* the next system collision,
carrying memory forward; E_k is then discarded.  theta_m = 0 recovers a
Markovian repeated-interaction (amplitude-damping-like) channel at T = 0;
theta_m > 0 allows information to return to the system -> backflow.

Discarding + reuse means only TWO live environment qubits are ever needed:
S + E_a + E_b = 3 qubits total, matching the pseudomode embedding's
3 qubits (system + mode + dilation ancilla) -- a fair comparison.

Two implementations, checked against each other in the test suite:
  * a fast pure-numpy density-matrix simulator (used for parameter fitting,
    where thousands of evaluations are needed);
  * Qiskit circuits (used for resource counting and the noise study).

Fairness protocol: the collision parameters (theta_s, theta_m) are fitted
by least squares so the model reproduces the SAME Stage A reference
excited-state population |G(t)|^2, then both embeddings are scored on the
same observables (N_BLP fidelity, resources, noise robustness).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from qiskit import QuantumCircuit

from .measures import KET, SX, SY
from .circuits import prepare_state, measurement_rotation

# ---------------------------------------------------------------------------
# Pure-numpy 3-qubit density-matrix engine (qubits: 0=S, 1=Ea, 2=Eb)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_K0 = np.array([[1, 0], [0, 0]], dtype=complex)   # |0><0|
_K1 = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|


def _embed(op: np.ndarray, q: int) -> np.ndarray:
    ops = [_I, _I, _I]
    ops[q] = op
    return np.kron(np.kron(ops[0], ops[1]), ops[2])


def _partial_exchange_unitary(theta: float, q1: int, q2: int) -> np.ndarray:
    """U = exp(-i theta/2 (X_q1 X_q2 + Y_q1 Y_q2)) on the 3-qubit space.

    Matches the circuit implementation rxx(theta) + ryy(theta) exactly
    (the two terms commute).
    """
    H = _embed(SX, q1) @ _embed(SX, q2) + _embed(SY, q1) @ _embed(SY, q2)
    return expm(-1j * (theta / 2.0) * H)


def _reset_qubit(rho: np.ndarray, q: int) -> np.ndarray:
    K0, K1 = _embed(_K0, q), _embed(_K1, q)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def _ptrace_system(rho: np.ndarray) -> np.ndarray:
    r = rho.reshape(2, 2, 2, 2, 2, 2)  # (s, a, b | s', a', b')
    return np.einsum("iabjab->ij", r)


def collision_states_numpy(init: str, n_steps: int, theta_s: float,
                           theta_m: float) -> list[np.ndarray]:
    """Reduced system states after 0..n_steps collisions (numpy engine)."""
    k = KET[init]
    psi_s = np.outer(k, k.conj())
    rho = np.kron(np.kron(psi_s, _P0), _P0)  # env qubits in |0> (T = 0)
    states = [_ptrace_system(rho)]
    cur, other = 1, 2
    # Unitaries depend only on (theta, pair) -> precompute the four variants.
    U_s = {q: _partial_exchange_unitary(theta_s, 0, q) for q in (1, 2)}
    U_m = {(1, 2): _partial_exchange_unitary(theta_m, 1, 2),
           (2, 1): _partial_exchange_unitary(theta_m, 2, 1)}
    for _ in range(n_steps):
        rho = U_s[cur] @ rho @ U_s[cur].conj().T          # S - E_cur collision
        rho = U_m[(cur, other)] @ rho @ U_m[(cur, other)].conj().T  # memory pass
        rho = _reset_qubit(rho, cur)                       # discard E_cur
        cur, other = other, cur
        states.append(_ptrace_system(rho))
    return states


# ---------------------------------------------------------------------------
# Parameter fitting against the Stage A reference
# ---------------------------------------------------------------------------

def fit_collision_params(target_pe: np.ndarray, n_steps: int,
                         maxiter: int = 300) -> dict:
    """Least-squares fit of (theta_s, theta_m) to a target excited-state
    population curve P_e(t_k) = |G(t_k)|^2 sampled at each collision step
    (k = 0..n_steps).

    Multi-start Nelder-Mead over a coarse (theta_s, theta_m) grid: the loss
    landscape has a Markovian local minimum at theta_m = 0 that a single
    start frequently collapses into, which would falsely report "no memory".
    All starts are recorded in the diagnostics for transparency.

    DOCUMENTED FINDING (verified by exhaustive grid scan, 25 x 32 points over
    theta_s in [0.05, 1.5], theta_m in [0, 3.1], Stage A gamma0=5, lam=1,
    dt=0.15): theta_m = 0 IS the global least-squares optimum for the
    Lorentzian-bath target.  This single-memory-qubit scheme produces
    backflow with a period locked to the 2-step collision cycle, which cannot
    match the slow (~30-step) revivals of the target; a memoryless fit is
    closer in L2 than any memory-bearing one.  Reproducing this bath class
    within a collision model requires memory persisting over many steps
    (ancilla trains with linearly growing qubit count, cf. arXiv:2509.12717)
    -- a genuine resource-tradeoff result for the embedding comparison, not
    an optimizer artifact.
    """
    target_pe = np.asarray(target_pe, dtype=float)
    assert len(target_pe) == n_steps + 1, "target must be sampled at every step"

    def loss(x):
        ts, tm = x
        states = collision_states_numpy("1", n_steps, ts, tm)
        pe = np.array([np.real(s[1, 1]) for s in states])
        return float(np.sum((pe - target_pe) ** 2))

    starts = [(ts0, tm0)
              for ts0 in (0.3, 0.6, 1.0)
              for tm0 in (0.3, 0.8, 1.3, 1.8)]
    best, trials = None, []
    for x0 in starts:
        res = minimize(loss, x0=np.array(x0), method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-10})
        trials.append({"x0": list(x0), "loss": float(res.fun),
                       "x": [float(v) for v in res.x]})
        if best is None or res.fun < best.fun:
            best = res
    ts, tm = (float(np.clip(v, 0.0, np.pi)) for v in best.x)
    states = collision_states_numpy("1", n_steps, ts, tm)
    pe = np.array([np.real(s[1, 1]) for s in states])
    return {
        "theta_s": ts,
        "theta_m": tm,
        "rmse": float(np.sqrt(np.mean((pe - target_pe) ** 2))),
        "converged": bool(best.success),
        "n_evals": int(best.nfev),
        "n_starts": len(starts),
        "all_trials": trials,
        "fitted_pe": pe.tolist(),
    }


# ---------------------------------------------------------------------------
# Qiskit circuits (resource counting + noise study)
# ---------------------------------------------------------------------------

def _collision_step(qc: QuantumCircuit, cur: int, other: int,
                    theta_s: float, theta_m: float) -> None:
    qc.rxx(theta_s, 0, cur)
    qc.ryy(theta_s, 0, cur)
    qc.rxx(theta_m, cur, other)
    qc.ryy(theta_m, cur, other)
    qc.reset(cur)


def collision_exact_circuit(init: str, n_steps: int, theta_s: float,
                            theta_m: float, save_every: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    prepare_state(qc, 0, init)
    qc.save_density_matrix(qubits=[0], label="rho_0")
    cur, other = 1, 2
    for k in range(1, n_steps + 1):
        _collision_step(qc, cur, other, theta_s, theta_m)
        cur, other = other, cur
        if k % save_every == 0 or k == n_steps:
            qc.save_density_matrix(qubits=[0], label=f"rho_{k}")
    return qc


def collision_shot_circuit(init: str, n_steps: int, theta_s: float,
                           theta_m: float, basis: str) -> QuantumCircuit:
    qc = QuantumCircuit(3, 1)
    prepare_state(qc, 0, init)
    cur, other = 1, 2
    for _ in range(n_steps):
        _collision_step(qc, cur, other, theta_s, theta_m)
        cur, other = other, cur
    measurement_rotation(qc, 0, basis)
    qc.measure(0, 0)
    return qc


# ===========================================================================
# Ancilla-train (chain-mapped) collision model -- the FAIR comparison
# ===========================================================================
# The 3-qubit model above has memory depth 1 and structurally cannot
# reproduce a Lorentzian backflow. A faithful finite-memory model discretises
# the bath into a CHAIN of M ancillas with a lossy boundary (star-to-chain /
# TEDOPA intuition, cf. ancilla-train arXiv:2509.12717): the system injects at
# site 1, excitation hops down the chain (nearest-neighbour partial exchange),
# reflects off the finite end, and returns -> tunable memory time growing with
# M. Weak damping at the far end supplies dissipation. Qubit cost grows
# linearly with M -- exactly the resource axis the fair comparison needs.

def _embed_n(op: np.ndarray, q: int, n: int) -> np.ndarray:
    mats = [np.eye(2, dtype=complex)] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _exchange_n(theta: float, q1: int, q2: int, n: int) -> np.ndarray:
    H = (_embed_n(SX, q1, n) @ _embed_n(SX, q2, n)
         + _embed_n(SY, q1, n) @ _embed_n(SY, q2, n))
    from scipy.linalg import expm
    return expm(-1j * (theta / 2.0) * H)


def _amp_damp_channel_n(rho: np.ndarray, q: int, n: int, p: float) -> np.ndarray:
    """Amplitude damping on qubit q (no ancilla; direct Kraus on the state)."""
    sm = _embed_n(np.array([[0, 1], [0, 0]], dtype=complex), q, n)  # |0><1|
    E1 = np.sqrt(p) * sm
    # E0 = I on |0>, sqrt(1-p) on |1>
    e0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    E0 = _embed_n(e0, q, n)
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T


def chain_collision_states_numpy(init: str, n_steps: int, M: int, theta_s: float,
                                 theta_hop: float, p_loss: float) -> list[np.ndarray]:
    """Reduced system trajectory for the M-ancilla chain collision model.

    Qubits: 0 = system, 1..M = bath chain (1 nearest the system). Per step:
    system-site exchange (theta_s), nearest-neighbour hops down the chain
    (theta_hop), weak amplitude damping at the far end (p_loss)."""
    n = 1 + M
    k = KET[init]
    psi_s = np.outer(k, k.conj())
    zero = np.array([[1, 0], [0, 0]], dtype=complex)
    rho = psi_s
    for _ in range(M):
        rho = np.kron(rho, zero)
    Us = _exchange_n(theta_s, 0, 1, n)
    Uhop = [_exchange_n(theta_hop, j, j + 1, n) for j in range(1, M)]  # (j,j+1)
    states = [_reduce_sys(rho, n)]
    for _ in range(n_steps):
        rho = Us @ rho @ Us.conj().T
        for U in Uhop:
            rho = U @ rho @ U.conj().T
        if p_loss > 0.0 and M >= 1:
            rho = _amp_damp_channel_n(rho, M, n, p_loss)  # far end (qubit M)
        states.append(_reduce_sys(rho, n))
    return states


_EINSUM_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _reduce_sys(rho: np.ndarray, n: int) -> np.ndarray:
    """Partial trace keeping qubit 0 for an n-qubit density matrix.

    Traces env qubits 1..n-1 by equating their in/out tensor indices in one
    einsum (robust to axis renumbering; valid for n up to 26)."""
    t = rho.reshape([2] * (2 * n))
    in_idx = list(_EINSUM_LETTERS[:n])
    out_idx = list(_EINSUM_LETTERS[n:2 * n])
    for k in range(1, n):        # force env in==out -> trace those qubits
        out_idx[k] = in_idx[k]
    sub = "".join(in_idx) + "".join(out_idx) + "->" + in_idx[0] + out_idx[0]
    return np.einsum(sub, t)


def _chain_step_circuit(qc: QuantumCircuit, M: int, theta_s: float,
                        theta_hop: float, p_loss: float) -> None:
    """One chain-collision step on qubits 0=system,1..M=chain,M+1=damp ancilla.
    Matches chain_collision_states_numpy exactly (same gate order)."""
    qc.rxx(theta_s, 0, 1)
    qc.ryy(theta_s, 0, 1)
    for j in range(1, M):
        qc.rxx(theta_hop, j, j + 1)
        qc.ryy(theta_hop, j, j + 1)
    if p_loss > 0.0:
        from .circuits import amplitude_damping_block
        amplitude_damping_block(qc, target=M, anc=M + 1, p=p_loss)


def chain_collision_exact_circuit(init: str, n_steps: int, M: int, theta_s: float,
                                  theta_hop: float, p_loss: float,
                                  save_every: int = 1) -> QuantumCircuit:
    from .circuits import prepare_state
    qc = QuantumCircuit(M + 2)
    prepare_state(qc, 0, init)
    qc.save_density_matrix(qubits=[0], label="rho_0")
    for k in range(1, n_steps + 1):
        _chain_step_circuit(qc, M, theta_s, theta_hop, p_loss)
        if k % save_every == 0 or k == n_steps:
            qc.save_density_matrix(qubits=[0], label=f"rho_{k}")
    return qc


def chain_collision_shot_circuit(init: str, n_steps: int, M: int, theta_s: float,
                                 theta_hop: float, p_loss: float, basis: str) -> QuantumCircuit:
    from .circuits import prepare_state, measurement_rotation
    qc = QuantumCircuit(M + 2, 1)
    prepare_state(qc, 0, init)
    for _ in range(n_steps):
        _chain_step_circuit(qc, M, theta_s, theta_hop, p_loss)
    measurement_rotation(qc, 0, basis)
    qc.measure(0, 0)
    return qc


def fit_chain_params(target_pe: np.ndarray, n_steps: int, M: int,
                     maxiter: int = 200, warm_start: tuple | None = None) -> dict:
    """Fit (theta_s, theta_hop, p_loss) of the M-chain to a target P_e(t)."""
    target_pe = np.asarray(target_pe, float)

    def loss(x):
        ts, th, pl = x
        pl = float(np.clip(pl, 0.0, 0.98))
        st = chain_collision_states_numpy("1", n_steps, M, ts, th, pl)
        pe = np.array([s[1, 1].real for s in st])
        return float(np.sum((pe - target_pe) ** 2))

    # Multi-start: the (theta_s, theta_hop, p_loss) landscape gets rougher as M
    # grows, so a too-sparse start set can miss the good basin (reported as a
    # spurious accuracy loss). These 6 starts + an optional warm start (the
    # previous M's fitted params) keep the resource curve honest and monotone.
    starts = [(0.4, 0.8, 0.05), (0.6, 1.2, 0.1), (0.3, 1.6, 0.02),
              (0.5, 0.4, 0.15), (0.35, 2.2, 0.08), (0.7, 1.0, 0.03)]
    if warm_start is not None:
        ws = tuple(warm_start)
        # the warm start itself, plus a near-decoupled-tail variant (small hop)
        # so an M-chain can always fall back to (M-1)-chain quality
        starts = [ws, (ws[0], 0.05, ws[2]), (ws[0], ws[1] * 0.5, ws[2])] + starts
    best = None
    for x0 in starts:
        r = minimize(loss, np.array(x0), method="Nelder-Mead",
                     options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    # Global polish: the landscape is rough and non-monotonic in M (the damping
    # site moves to the far end, so warm starts don't always transfer).
    # Differential evolution over the 3 bounded params reliably finds the basin.
    from scipy.optimize import differential_evolution
    de = differential_evolution(loss, bounds=[(0, np.pi), (0, np.pi), (0, 0.9)],
                                maxiter=maxiter, tol=1e-9, seed=0, polish=True,
                                init="sobol")
    if de.fun < best.fun:
        best = de
    ts, th, pl = best.x
    pl = float(np.clip(pl, 0.0, 0.98))
    st = chain_collision_states_numpy("1", n_steps, M, ts, th, pl)
    pe = np.array([s[1, 1].real for s in st])
    return {"M": M, "theta_s": float(ts), "theta_hop": float(th), "p_loss": pl,
            "rmse": float(np.sqrt(np.mean((pe - target_pe) ** 2))),
            "n_qubits_circuit": 1 + M + 1}  # + one damping-dilation ancilla

"""Qiskit circuits for the pseudomode embedding (Stage A and Stage B).

Per Trotter step:
  (i)  unitary part -- system-pseudomode XY exchange
       H_int = Omega (s+ pm- + s- pm+) = (Omega/2)(XX + YY)
       -> RXX(Omega dt) RYY(Omega dt)   (the two terms commute),
       plus single-qubit self-energies as RX/RZ where present;
  (ii) dissipative part -- amplitude damping of the pseudomode qubit via
       Stinespring dilation onto a fresh ancilla:
           reset(anc); CRY(2 arcsin sqrt(p)) [pm -> anc]; CX [anc -> pm]
       with p = 1 - exp(-kappa dt) per step.  Deterministic (no trajectory
       sampling); composes cleanly with Aer gate-noise models.

Two state-extraction modes:
  * exact mode  -- mid-circuit `save_density_matrix` on the system qubit
                   (validation tier);
  * shot mode   -- basis rotation + measurement of the system qubit
                   (honest NISQ-style pipeline used in the noise study).

Qubit layout Stage A: 0 = system, 1 = pseudomode, 2 = dilation ancilla.
Stage B: 0 = system, 1..M = pseudomodes, M+1 = shared dilation ancilla.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# State preparation / measurement-basis helpers
# ---------------------------------------------------------------------------

def prepare_state(qc: QuantumCircuit, q: int, label: str) -> None:
    """Prepare a canonical pure state on qubit q from |0>."""
    if label == "0":
        pass
    elif label == "1":
        qc.x(q)
    elif label == "+":
        qc.h(q)
    elif label == "-":
        qc.x(q)
        qc.h(q)
    elif label == "+i":
        qc.h(q)
        qc.s(q)
    elif label == "-i":
        qc.h(q)
        qc.sdg(q)
    else:
        raise ValueError(f"unknown state label {label!r}")


def measurement_rotation(qc: QuantumCircuit, q: int, basis: str) -> None:
    """Rotate so that a Z measurement reads out <X>, <Y> or <Z>."""
    if basis == "X":
        qc.h(q)
    elif basis == "Y":
        qc.sdg(q)
        qc.h(q)
    elif basis != "Z":
        raise ValueError(f"unknown basis {basis!r}")


def _cry(qc: QuantumCircuit, theta: float, ctrl: int, tgt: int) -> None:
    """Controlled-RY via the standard native decomposition (Aer's direct
    assembler has no `cry` instruction): RY(t/2) . CX . RY(-t/2) . CX."""
    qc.ry(theta / 2.0, tgt)
    qc.cx(ctrl, tgt)
    qc.ry(-theta / 2.0, tgt)
    qc.cx(ctrl, tgt)


def amplitude_damping_block(qc: QuantumCircuit, target: int, anc: int, p: float) -> None:
    """Stinespring dilation of the amplitude-damping channel (strength p)."""
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))
    qc.reset(anc)
    _cry(qc, theta, target, anc)
    qc.cx(anc, target)


def generalized_amplitude_damping_block(qc: QuantumCircuit, target: int, anc: int,
                                        p_down: float, p_up: float) -> None:
    """Finite-temperature GAD via two dilated damping half-steps.

    Emission (|1>->|0>, prob p_down) then absorption (|0>->|1>, prob p_up).
    Absorption is emission conjugated by X (damp toward |1>).  Composition
    order is a Trotter choice: O(dt^2) error, consistent with the rest of the
    scheme, and validated against QuTiP's exact thermal Lindblad.  At p_up = 0
    this reduces exactly to :func:`amplitude_damping_block`.
    """
    amplitude_damping_block(qc, target, anc, p_down)
    if p_up > 0.0:
        qc.x(target)
        amplitude_damping_block(qc, target, anc, p_up)
        qc.x(target)


# ---------------------------------------------------------------------------
# Stage A circuit builders
# ---------------------------------------------------------------------------

def stageA_step(qc: QuantumCircuit, omega: float, kappa: float, dt: float,
                n_th: float = 0.0) -> None:
    """One first-order Trotter step of the Stage A enlarged Lindbladian.

    ``n_th`` is the mean thermal occupation of the pseudomode's Markovian
    reservoir; n_th = 0 recovers the exact zero-temperature block. Finite T
    uses emission rate kappa(n_th+1) and absorption rate kappa*n_th, i.e. the
    mode relaxes toward a thermal (not vacuum) state.
    """
    qc.rxx(omega * dt, 0, 1)
    qc.ryy(omega * dt, 0, 1)
    if n_th <= 0.0:
        amplitude_damping_block(qc, target=1, anc=2, p=1.0 - np.exp(-kappa * dt))
    else:
        p_down = 1.0 - np.exp(-kappa * (n_th + 1.0) * dt)
        p_up = 1.0 - np.exp(-kappa * n_th * dt)
        generalized_amplitude_damping_block(qc, target=1, anc=2, p_down=p_down, p_up=p_up)


def stageA_exact_circuit(init: str, n_steps: int, dt: float, omega: float,
                         kappa: float, save_every: int = 1, n_th: float = 0.0) -> QuantumCircuit:
    """Evolution circuit with mid-circuit reduced-density-matrix saves.

    Save labels: 'rho_0' (initial), 'rho_k' after k steps (k multiple of
    save_every, plus the final step).
    """
    qc = QuantumCircuit(3)
    prepare_state(qc, 0, init)
    qc.save_density_matrix(qubits=[0], label="rho_0")
    for k in range(1, n_steps + 1):
        stageA_step(qc, omega, kappa, dt, n_th=n_th)
        if k % save_every == 0 or k == n_steps:
            qc.save_density_matrix(qubits=[0], label=f"rho_{k}")
    return qc


def stageA_exact_circuit_dephased(init: str, n_steps: int, dt: float, omega: float,
                                  kappa: float, phi_list: list[float],
                                  save_every: int = 1, n_th: float = 0.0) -> QuantumCircuit:
    """Stage A with a coherent RZ(phi_k) kick on the SYSTEM after each step.

    Ensemble-averaging density matrices over ``phi`` realises dephasing whose
    temporal correlation is controlled by how ``phi_list`` is drawn:
      * one fixed angle repeated  -> quasi-static (non-Markovian) dephasing,
      * fresh angle per step       -> Markovian dephasing (same marginal).
    This is the apparatus for the non-Markovian-noise contrast.
    """
    qc = QuantumCircuit(3)
    prepare_state(qc, 0, init)
    qc.save_density_matrix(qubits=[0], label="rho_0")
    for k in range(1, n_steps + 1):
        stageA_step(qc, omega, kappa, dt, n_th=n_th)
        qc.rz(float(phi_list[k - 1]), 0)
        if k % save_every == 0 or k == n_steps:
            qc.save_density_matrix(qubits=[0], label=f"rho_{k}")
    return qc


def stageA_shot_circuit(init: str, n_steps: int, dt: float, omega: float,
                        kappa: float, basis: str, n_th: float = 0.0) -> QuantumCircuit:
    """Evolution circuit ending in a Pauli-basis measurement of the system."""
    qc = QuantumCircuit(3, 1)
    prepare_state(qc, 0, init)
    for _ in range(n_steps):
        stageA_step(qc, omega, kappa, dt, n_th=n_th)
    measurement_rotation(qc, 0, basis)
    qc.measure(0, 0)
    return qc


# ---------------------------------------------------------------------------
# Stage B circuit builders (spin-boson, one qubit per pseudomode)
# ---------------------------------------------------------------------------

def stageB_step(qc: QuantumCircuit, delta: float, eps: float,
                modes: list[tuple[float, float, float]], dt: float) -> None:
    """One Trotter step: system self-energy, mode self-energies, sigma_z (x) X
    couplings (RZX), then amplitude damping of each mode qubit."""
    n_modes = len(modes)
    anc = 1 + n_modes
    qc.rx(delta * dt, 0)          # e^{-i (delta/2) X dt}
    if eps != 0.0:
        qc.rz(eps * dt, 0)        # e^{-i (eps/2) Z dt}
    for k, (wk, gk, kappak) in enumerate(modes):
        q = 1 + k
        # mode self-energy w a^dag a -> (w/2)(I - Z): RZ(-w dt) up to phase
        qc.rz(-wk * dt, q)
        # coupling g Z_s X_k: RZX(theta) = e^{-i theta ZX / 2} -> theta = 2 g dt
        qc.rzx(2.0 * gk * dt, 0, q)
    for k, (wk, gk, kappak) in enumerate(modes):
        p = 1.0 - np.exp(-kappak * dt)
        amplitude_damping_block(qc, target=1 + k, anc=anc, p=p)


def stageB_exact_circuit(init: str, n_steps: int, dt: float, delta: float,
                         eps: float, modes: list[tuple[float, float, float]],
                         save_every: int = 1) -> QuantumCircuit:
    n_modes = len(modes)
    qc = QuantumCircuit(2 + n_modes)
    prepare_state(qc, 0, init)
    qc.save_density_matrix(qubits=[0], label="rho_0")
    for k in range(1, n_steps + 1):
        stageB_step(qc, delta, eps, modes, dt)
        if k % save_every == 0 or k == n_steps:
            qc.save_density_matrix(qubits=[0], label=f"rho_{k}")
    return qc


# ---------------------------------------------------------------------------
# Simulation backends
# ---------------------------------------------------------------------------

def dimer_exact_circuit(n_steps: int, dt: float, eps_L: float, eps_R: float, J: float,
                        site_modes, save_every: int = 1) -> QuantumCircuit:
    """Excitonic dimer: q0=site L, q1=site R, q2=pmL, q3=pmR, q4=damp ancilla.
    Initial excitation on L. Saves single-site reduced states 'L_k','R_k'.

    Per step: site energies (RZ), excitonic hop (RXX+RYY on the two sites),
    mode energies (RZ), site-mode dephasing (RZX), mode damping (dilation)."""
    (wL, gL, kapL), (wR, gR, kapR) = site_modes
    qc = QuantumCircuit(5)
    qc.x(0)  # excitation localised on site L
    qc.save_density_matrix(qubits=[0], label="L_0")
    qc.save_density_matrix(qubits=[1], label="R_0")
    for k in range(1, n_steps + 1):
        # site energy eps*n = eps*(I-Z)/2 -> e^{-i eps n dt} = diag(1,e^{-i eps dt});
        # rz(lambda)=diag(e^{-i lam/2}, e^{+i lam/2}), so lambda = -eps*dt matches.
        qc.rz(-eps_L * dt, 0)
        qc.rz(-eps_R * dt, 1)
        qc.rxx(J * dt, 0, 1)          # excitonic hop = (J/2)(XX+YY)
        qc.ryy(J * dt, 0, 1)
        qc.rz(-wL * dt, 2)
        qc.rz(-wR * dt, 3)
        qc.rzx(2.0 * gL * dt, 0, 2)   # site-L dephasing coupling
        qc.rzx(2.0 * gR * dt, 1, 3)   # site-R dephasing coupling
        amplitude_damping_block(qc, target=2, anc=4, p=1.0 - np.exp(-kapL * dt))
        amplitude_damping_block(qc, target=3, anc=4, p=1.0 - np.exp(-kapR * dt))
        if k % save_every == 0 or k == n_steps:
            qc.save_density_matrix(qubits=[0], label=f"L_{k}")
            qc.save_density_matrix(qubits=[1], label=f"R_{k}")
    return qc


def dimer_populations_from_data(data: dict, n_steps: int, save_every: int = 1):
    """Extract P_L(t), P_R(t) from saved single-site reduced states."""
    steps = saved_steps(n_steps, save_every)
    P_L = np.array([np.real(data[f"L_{k}"][1, 1]) for k in steps])
    P_R = np.array([np.real(data[f"R_{k}"][1, 1]) for k in steps])
    return P_L, P_R


def run_exact(circuits: list[QuantumCircuit], noise_model=None,
              seed: int = 1234) -> list[dict[str, np.ndarray]]:
    """Density-matrix simulation; returns one {label: 2x2 rho} dict per circuit."""
    sim = AerSimulator(method="density_matrix", noise_model=noise_model,
                       seed_simulator=seed)
    # Aer executes all gates used here natively; transpilation is only needed
    # to map onto a noise model's basis so that every gate picks up its error.
    # (Qiskit 0.46's density-matrix target also reports an incomplete basis to
    # the transpiler, so transpiling in the ideal case would fail anyway.)
    if noise_model is not None:
        circuits = transpile_for_noise(circuits, noise_model)
    result = sim.run(circuits).result()
    out = []
    for i in range(len(circuits)):
        data = result.data(i)
        out.append({k: np.asarray(v) for k, v in data.items()
                    if k.startswith(("rho", "L_", "R_"))})
    return out


def exact_states_from_labels(data: dict[str, np.ndarray], n_steps: int,
                             save_every: int = 1) -> list[np.ndarray]:
    """Order the saved density matrices by step index."""
    steps = sorted({0, *range(save_every, n_steps + 1, save_every), n_steps})
    return [data[f"rho_{k}"] for k in steps]


def saved_steps(n_steps: int, save_every: int = 1) -> list[int]:
    return sorted({0, *range(save_every, n_steps + 1, save_every), n_steps})


def run_counts(circuits: list[QuantumCircuit], shots: int = 4096,
               noise_model=None, seed: int = 1234,
               pre_transpiled: bool = False) -> list[dict[str, int]]:
    """Shot-based execution; returns counts dicts (keys '0'/'1')."""
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    # Same rationale as run_exact: Aer executes all gates used here natively;
    # transpile only when a noise model requires mapping onto its basis.
    if pre_transpiled or noise_model is None:
        tqcs = circuits
    else:
        tqcs = transpile_for_noise(circuits, noise_model)
    result = sim.run(tqcs, shots=shots).result()
    return [result.get_counts(i) for i in range(len(circuits))]


def transpile_for_noise(circuits: list[QuantumCircuit], noise_model) -> list[QuantumCircuit]:
    """Transpile to the noise model's basis so every gate picks up its error.

    Uses an explicit ``basis_gates`` list rather than backend-target
    resolution: with qiskit 0.46 + qiskit-aer 0.13, ``transpile(qc,
    AerSimulator(noise_model=...))`` composes a broken (non-universal) target
    basis and fails.  Non-gate directives (reset, measure, save) are appended
    so the BasisTranslator passes them through untouched.
    """
    basis = list(noise_model.basis_gates) + ["reset", "measure", "barrier",
                                             "save_density_matrix"]
    return transpile(circuits, basis_gates=basis, optimization_level=1)


def resource_counts(circuit: QuantumCircuit, basis: tuple[str, ...] = ("cx", "sx", "rz", "x", "id")) -> dict:
    """Transpiled resource metrics for the embedding comparison (Phase 4)."""
    sim = AerSimulator()
    tqc = transpile(circuit, sim, basis_gates=list(basis) + ["reset", "measure",
                    "save_density_matrix"], optimization_level=1)
    ops = tqc.count_ops()
    return {
        "n_qubits": circuit.num_qubits,
        "depth": tqc.depth(),
        "cx_count": int(ops.get("cx", 0)),
        "total_ops": sum(int(v) for k, v in ops.items() if k not in ("save_density_matrix",)),
    }

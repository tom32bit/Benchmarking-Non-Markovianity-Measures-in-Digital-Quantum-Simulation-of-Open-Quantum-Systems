"""Phase orchestrators.  Every run saves a JSON provenance record (parameters,
package versions, seed, timestamp) next to its figures -- results are never
reported without their raw data (research-integrity requirement; see
RESEARCH_ETHICS.md).
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .measures import (IC_INPUTS, blp_direction_values, blp_from_curve, blp_measure,
                       blp_measure_optimized, blp_optimize_pair, canonical_pairs, dm,
                       pauli_images_from_states, rhp_measure, trace_distance_curve,
                       trace_norm, transfer_matrix)
from .reference import (TIGHT_SOLVER_OPTIONS, analytic_G, analytic_blp_continuum,
                        analytic_stageA_states, required_fock_dimension,
                        stageA_finiteT_with_occupation, stageA_params,
                        stageA_qutip_states, stageA_qutip_states_finiteT,
                        spin_boson_qutip_states, spin_boson_qutip_states_finiteT,
                        spin_boson_with_occupation)
from . import circuits as qc_mod
from . import collision as col_mod
from .noise import (build_noise_model, from_fake_backend, noise_params_dict,
                    quasistatic_angles)
from .mitigation import (confusion_matrix, fold_cx, mitigated_expectation,
                         readout_calibration_circuits, richardson_extrapolate)
from .tomography import BASES, expectation_from_counts, rho_from_pauli_counts

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Default Stage A physics (units of the Lorentzian half-width lam):
STAGEA_NONMARKOV = {"gamma0": 5.0, "lam": 1.0}   # gamma0 > lam/2 -> backflow
STAGEA_MARKOV = {"gamma0": 0.2, "lam": 1.0}      # gamma0 < lam/2 -> no backflow


def _git_commit() -> str | None:
    """Commit of the code that produced a record; '+dirty' marks uncommitted
    changes to tracked files. None if git is unavailable."""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("+dirty" if dirty else "")


# A run executes the code imported at its launch, and a long run can outlive a
# commit: records keep the commit at import as well as the one at save time.
_GIT_COMMIT_AT_IMPORT = _git_commit()


def _provenance(params: dict, seed: int | None) -> dict:
    import qiskit
    import qiskit_aer
    import qutip
    import scipy
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qmembench_version": __version__,
        "git_commit": _git_commit(),
        "git_commit_at_import": _GIT_COMMIT_AT_IMPORT,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
                     "qutip": qutip.__version__},
        "seed": seed,
        "params": params,
    }


def save_result(name: str, params: dict, data: dict, seed: int | None = None) -> Path:
    """Write results/<name>.json (latest) AND an immutable timestamped copy in
    results/history/ so a later run never destroys an earlier record (R5)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    hist = RESULTS_DIR / "history"
    hist.mkdir(exist_ok=True)
    payload = {"provenance": _provenance(params, seed), "data": data}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    with open(hist / f"{name}__{stamp}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=_jsonable)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=_jsonable)
    return path


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    raise TypeError(f"not jsonable: {type(obj)}")


def _pair_curves_from_states(states_by_label: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    curves = {}
    for a, b in canonical_pairs():
        curves[f"{a}|{b}"] = trace_distance_curve(states_by_label[a], states_by_label[b])
    return curves


def _rhp_from_states(states_by_label: dict[str, list[np.ndarray]]) -> tuple[float, float]:
    """N_RHP and ill-conditioned fraction from the four informationally complete
    single-qubit input trajectories (0, 1, +, +i). Used to test whether the
    truncation sign-flip seen in the BLP measure also appears in RHP."""
    n = len(states_by_label["0"])
    transfers = [transfer_matrix({l: states_by_label[l][k] for l in ("0", "1", "+", "+i")})
                 for k in range(n)]
    return rhp_measure(transfers)


# ===========================================================================
# Phase 0 -- measures validated on the analytic solution
# ===========================================================================

def phase0(quick: bool = False) -> dict:
    n_t = 60 if quick else 200
    t_max = 12.0
    times = np.linspace(0.0, t_max, n_t)
    out: dict = {"times": times}
    for tag, pars in (("nonmarkovian", STAGEA_NONMARKOV), ("markovian", STAGEA_MARKOV)):
        states = {lbl: analytic_stageA_states(dm(lbl), times, **pars)
                  for lbl in ("0", "1", "+", "-", "+i", "-i")}
        curves = _pair_curves_from_states(states)
        n_blp, best = blp_measure(curves)
        # analytic optimum: equatorial pair gives D(t) = |G(t)|
        G = np.abs(analytic_G(times, **pars))
        n_blp_analytic = blp_from_curve(G)
        transfers = [transfer_matrix({l: states[l][k] for l in ("0", "1", "+", "+i")})
                     for k in range(n_t)]
        n_rhp, ill_frac = rhp_measure(transfers)
        out[tag] = {
            "params": pars,
            "D_curves": {k: v for k, v in curves.items()},
            "abs_G": G,
            "N_BLP": n_blp, "best_pair": best,
            "N_BLP_analytic_optimum": n_blp_analytic,
            "N_RHP": n_rhp, "rhp_ill_conditioned_fraction": ill_frac,
        }
    save_result("phase0_measures_validation", {"n_t": n_t, "t_max": t_max}, out)
    return out


# ===========================================================================
# Phase 1 -- Stage A three-tier validation: analytic <-> QuTiP <-> circuit
# ===========================================================================

def _circuit_states_stageA(times: np.ndarray, dt: float, gamma0: float, lam: float,
                           labels=("0", "1", "+", "-", "+i", "-i"),
                           noise_model=None, seed: int = 1234) -> dict[str, list[np.ndarray]]:
    omega, kappa = stageA_params(gamma0, lam)
    n_steps = len(times) - 1
    circuits = [qc_mod.stageA_exact_circuit(lbl, n_steps, dt, omega, kappa)
                for lbl in labels]
    results = qc_mod.run_exact(circuits, noise_model=noise_model, seed=seed)
    return {lbl: qc_mod.exact_states_from_labels(res, n_steps)
            for lbl, res in zip(labels, results)}


def phase1(quick: bool = False, dt: float | None = None) -> dict:
    pars = STAGEA_NONMARKOV
    dt = dt if dt is not None else (0.1 if quick else 0.05)
    t_max = 12.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)

    tiers: dict[str, dict[str, list[np.ndarray]]] = {}
    tiers["analytic"] = {lbl: analytic_stageA_states(dm(lbl), times, **pars)
                         for lbl in ("0", "1", "+", "-", "+i", "-i")}
    tiers["qutip"] = {lbl: stageA_qutip_states(dm(lbl), times, **pars, n_fock=2)
                      for lbl in ("0", "1", "+", "-", "+i", "-i")}
    tiers["circuit"] = _circuit_states_stageA(times, dt, **pars)

    out: dict = {"times": times, "dt": dt, "params": pars}
    for tier, states in tiers.items():
        curves = _pair_curves_from_states(states)
        n_blp, best = blp_measure(curves)
        out[tier] = {"D_curves": curves, "N_BLP": n_blp, "best_pair": best}

    # element-wise deviations (max over time and over the six evolved states)
    def max_dev(t1, t2):
        return max(float(np.max(np.abs(np.array(tiers[t1][l]) - np.array(tiers[t2][l]))))
                   for l in tiers[t1])
    out["max_abs_dev_qutip_vs_analytic"] = max_dev("qutip", "analytic")
    out["max_abs_dev_circuit_vs_analytic"] = max_dev("circuit", "analytic")
    out["blp_error_circuit_vs_analytic"] = abs(out["circuit"]["N_BLP"] - out["analytic"]["N_BLP"])
    save_result("phase1_stageA_validation", {"dt": dt, "t_max": t_max, **pars}, out)
    return out


def phase1_trotter_convergence(dts=(0.2, 0.1, 0.05), t_max: float = 12.0) -> dict:
    """Trotter-step convergence of the circuit tier (validation gate)."""
    pars = STAGEA_NONMARKOV
    rows = []
    for dt in dts:
        n_steps = int(round(t_max / dt))
        times = np.linspace(0.0, n_steps * dt, n_steps + 1)
        ana = analytic_stageA_states(dm("1"), times, **pars)
        circ = _circuit_states_stageA(times, dt, **pars, labels=("1",))["1"]
        dev = float(np.max(np.abs(np.array(ana) - np.array(circ))))
        rows.append({"dt": dt, "max_abs_dev": dev})
    out = {"rows": rows, "params": pars}
    save_result("phase1_trotter_convergence", {"dts": list(dts), "t_max": t_max}, out)
    return out


# ===========================================================================
# Phase 1T -- finite temperature: validation + memory-vs-T + truncation
# ===========================================================================

def phase1_finiteT(quick: bool = False, dt: float = 0.05) -> dict:
    """Finite-temperature Stage A. Three deliverables:
      (i)  circuit (2-level mode GAD) validated against QuTiP thermal ref at
           matched truncation (n_fock=2),
      (ii) N_BLP vs thermal occupation n_th (memory should fall with T),
      (iii) physical truncation error (n_fock=2 vs converged) growing with T,
           tying the temperature axis to the truncation centerpiece.
    """
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    n_ths = [0.0, 0.25, 0.5, 1.0] if quick else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
    t_max = 10.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    labels = ("0", "1", "+", "-", "+i", "-i")
    n_conv = 4 if quick else 6

    rows = []
    for n_th in n_ths:
        # circuit tier (single-qubit mode -> truncation fixed at 2 levels)
        circuits = [qc_mod.stageA_exact_circuit(l, n_steps, dt, omega, kappa, n_th=n_th)
                    for l in labels]
        res = qc_mod.run_exact(circuits)
        circ = {l: qc_mod.exact_states_from_labels(r, n_steps)
                for l, r in zip(labels, res)}
        ref2 = {l: stageA_qutip_states_finiteT(dm(l), times, **pars, n_th=n_th, n_fock=2)
                for l in labels}
        refc = {l: stageA_qutip_states_finiteT(dm(l), times, **pars, n_th=n_th, n_fock=n_conv)
                for l in labels}

        def blp_of(states):
            return blp_measure(_pair_curves_from_states(states))[0]

        dev_circ = max(float(np.max(np.abs(np.array(circ[l]) - np.array(ref2[l]))))
                       for l in labels)
        trunc = max(float(np.max(np.abs(np.array(ref2[l]) - np.array(refc[l]))))
                    for l in labels)
        # RHP for circuit (2-level) and converged, to test the sign-flip in RHP
        rhp_c2, ill_c2 = _rhp_from_states(circ)
        rhp_conv, ill_conv = _rhp_from_states(refc)
        rows.append({"n_th": n_th,
                     "N_BLP_circuit": blp_of(circ),
                     "N_BLP_ref_nfock2": blp_of(ref2),
                     "N_BLP_ref_converged": blp_of(refc),
                     "N_RHP_circuit": rhp_c2, "N_RHP_ref_converged": rhp_conv,
                     "rhp_ill_frac_circuit": ill_c2, "rhp_ill_frac_converged": ill_conv,
                     "max_dev_circuit_vs_ref2": dev_circ,
                     "truncation_error_2_vs_converged": trunc})
    out = {"rows": rows, "params": pars, "dt": dt, "t_max": t_max, "n_conv": n_conv,
           "blp_note": ("N_BLP values are canonical-pair lower bounds; truncation "
                        "evidence rests on pair-independent trajectory deviation.")}
    save_result("phase1T_finite_temperature",
                {"n_ths": n_ths, "dt": dt, "n_conv": n_conv, **pars}, out)
    return out


# ===========================================================================
# Phase 2 -- spin-boson: N_BLP vs coupling; truncation study
# ===========================================================================

def phase2(quick: bool = False, n_conv: int = 6) -> dict:
    """Stage B spin-boson: N_BLP vs coupling and the truncation story.

    M1 fix: the 'physical reference' uses a CONVERGED cutoff (n_fock=6 by
    default), not n_fock=4 -- Phase 5's own d_req shows 4 is insufficient at
    strong coupling. A per-coupling n_conv-vs-(n_conv+1) stability check is
    recorded so the reference's convergence is demonstrated, not assumed.
    """
    delta, eps = 1.0, 0.0
    modes_proto = [(1.0, None, 0.4), (1.6, None, 0.8)]  # (w_k, g_k set below, kappa_k)
    couplings = [0.1, 0.3, 0.5] if quick else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    dt = 0.1 if quick else 0.05
    t_max = 10.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    labels = ("0", "1", "+", "-", "+i", "-i")

    def blp_of(states):
        return blp_measure(_pair_curves_from_states(states))

    rows = []
    for g in couplings:
        modes = [(w, g, kap) for (w, _, kap) in modes_proto]
        ref2 = {l: spin_boson_qutip_states(dm(l), times, delta, eps, modes, n_fock=2)
                for l in labels}
        refc = {l: spin_boson_qutip_states(dm(l), times, delta, eps, modes, n_fock=n_conv)
                for l in labels}
        circuits = [qc_mod.stageB_exact_circuit(l, n_steps, dt, delta, eps, modes)
                    for l in labels]
        res = qc_mod.run_exact(circuits)
        circ = {l: qc_mod.exact_states_from_labels(r, n_steps)
                for l, r in zip(labels, res)}

        # convergence check only at the strongest coupling (cheap, decisive)
        conv_gap = None
        if g == couplings[-1]:
            refc1 = {l: spin_boson_qutip_states(dm(l), times, delta, eps, modes,
                                                n_fock=n_conv + 1) for l in labels}
            conv_gap = max(float(np.max(np.abs(np.array(refc[l]) - np.array(refc1[l]))))
                           for l in labels)

        n2, _ = blp_of(ref2)
        nconv, _ = blp_of(refc)
        nc, best = blp_of(circ)
        dev_c2 = max(float(np.max(np.abs(np.array(circ[l]) - np.array(ref2[l]))))
                     for l in labels)
        trunc_err = max(float(np.max(np.abs(np.array(ref2[l]) - np.array(refc[l]))))
                        for l in labels)
        rhp_c2, ill_c2 = _rhp_from_states(circ)
        rhp_conv, ill_conv = _rhp_from_states(refc)
        rows.append({"g": g, "N_BLP_circuit": nc, "N_BLP_ref_nfock2": n2,
                     "N_BLP_ref_converged": nconv, "n_conv": n_conv,
                     "N_RHP_circuit": rhp_c2, "N_RHP_ref_converged": rhp_conv,
                     "rhp_ill_frac_circuit": ill_c2, "rhp_ill_frac_converged": ill_conv,
                     "converged_stability_gap": conv_gap, "best_pair": best,
                     "max_dev_circuit_vs_ref2": dev_c2,
                     "truncation_error_2_vs_converged": trunc_err})
    out = {"rows": rows, "delta": delta, "eps": eps, "dt": dt, "t_max": t_max,
           "n_conv": n_conv, "mode_frequencies_and_kappas": modes_proto,
           "blp_note": ("N_BLP values are canonical-pair LOWER BOUNDS (see "
                        "phase_blp_pair_certificate; gap up to ~0.09 for spin-boson). "
                        "The truncation finding rests on the pair-INDEPENDENT trajectory "
                        "deviation (truncation_error_2_vs_converged), which is rigorous.")}
    save_result("phase2_spin_boson", {"couplings": couplings, "dt": dt, "n_conv": n_conv}, out)
    return out


# ===========================================================================
# Phase 9 -- 2D truncation phase diagram over (coupling, temperature)
# ===========================================================================

def phase9_truncation_phase_diagram(quick: bool = False, n_conv: int = 6) -> dict:
    """Map the SIGNED two-level truncation error of the BLP measure over the
    (coupling g, temperature n_th) plane of a finite-temperature spin-boson.

    Signed metric = N_BLP(n_fock=2) - N_BLP(converged) for the (+,-) pair:
      positive  -> truncation FABRICATES memory (strong coupling, low T),
      negative  -> truncation DESTROYS memory (high T),
      the zero contour is the sign-flip boundary.
    Also returns the converged memory landscape N_BLP(converged) for a second
    (sequential) panel. All values are canonical (+,-) pair BLP, consistent with
    Tables I-II; the sign of the difference is the physical content.
    """
    delta, eps = 1.0, 0.0
    modes_proto = [(1.0, None, 0.4), (1.6, None, 0.8)]
    dt = 0.1
    t_max = 8.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    pair = ("+", "-")

    if quick:
        gs = np.linspace(0.05, 0.8, 14)
        n_ths = np.linspace(0.0, 1.5, 12)
    else:
        gs = np.linspace(0.05, 0.8, 44)
        n_ths = np.linspace(0.0, 1.5, 34)

    def blp_pair(states):
        return blp_from_curve(trace_distance_curve(states[pair[0]], states[pair[1]]))

    signed = np.zeros((len(n_ths), len(gs)))
    converged = np.zeros((len(n_ths), len(gs)))
    trunc2 = np.zeros((len(n_ths), len(gs)))
    for i, n_th in enumerate(n_ths):
        for j, g in enumerate(gs):
            modes = [(w, g, kap) for (w, _, kap) in modes_proto]
            s2 = {l: spin_boson_qutip_states_finiteT(dm(l), times, delta, eps, modes,
                                                     n_th=n_th, n_fock=2) for l in pair}
            sc = {l: spin_boson_qutip_states_finiteT(dm(l), times, delta, eps, modes,
                                                     n_th=n_th, n_fock=n_conv) for l in pair}
            b2, bc = blp_pair(s2), blp_pair(sc)
            trunc2[i, j] = b2
            converged[i, j] = bc
            signed[i, j] = b2 - bc

    out = {"g": gs, "n_th": n_ths, "signed_error": signed,
           "converged_blp": converged, "circuit_blp": trunc2,
           "delta": delta, "eps": eps, "dt": dt, "t_max": t_max, "n_conv": n_conv,
           "pair": pair, "mode_frequencies_and_kappas": modes_proto,
           "note": ("Signed = N_BLP(n_fock=2) - N_BLP(n_fock=%d), (+,-) pair. "
                    "Positive fabricates, negative destroys; zero contour is the "
                    "sign-flip boundary. Finite-T spin-boson (single model spanning "
                    "both regimes)." % n_conv)}
    save_result("phase9_truncation_phase_diagram",
                {"grid": [len(gs), len(n_ths)], "dt": dt, "n_conv": n_conv}, out)
    return out


# ===========================================================================
# Phase 3 -- noise robustness of the memory signature (shot-mode pipeline)
# ===========================================================================

def _shot_mode_states(builder, build_args: dict, step_list: list[int], pair: tuple[str, str],
                      noise_model, shots: int, seed: int,
                      readout_M: np.ndarray | None) -> dict[str, list[np.ndarray]]:
    """Tomographic reconstruction of the pair's trajectories at given steps."""
    labels = list(pair)
    circuits, index = [], []
    for lbl in labels:
        for k in step_list:
            for b in BASES:
                circuits.append(builder(init=lbl, n_steps=k, basis=b, **build_args))
                index.append((lbl, k, b))
    tqcs = qc_mod.transpile_for_noise(circuits, noise_model) if noise_model else circuits
    counts = qc_mod.run_counts(tqcs, shots=shots, noise_model=noise_model,
                               seed=seed, pre_transpiled=noise_model is not None)
    by_key = dict(zip(index, counts))
    states: dict[str, list[np.ndarray]] = {l: [] for l in labels}
    for lbl in labels:
        for k in step_list:
            if readout_M is not None:
                r = np.array([mitigated_expectation(by_key[(lbl, k, b)], readout_M)
                              for b in BASES])
                from .measures import rho_from_bloch
                states[lbl].append(rho_from_bloch(r, project=True))
            else:
                rho, _ = rho_from_pauli_counts({b: by_key[(lbl, k, b)] for b in BASES})
                states[lbl].append(rho)
    return states


def phase3(quick: bool = False, shots: int = 4096, seed: int = 1234) -> dict:
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    dt = 0.15 if quick else 0.1
    t_max = 9.0
    n_steps_total = int(round(t_max / dt))
    stride = 3 if quick else 2
    step_list = list(range(0, n_steps_total + 1, stride))
    times = np.array(step_list) * dt
    pair = ("+", "-")  # analytic BLP-optimal pair for Stage A
    scales = [0.0, 0.5, 1.0] if quick else [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    build_args = {"dt": dt, "omega": omega, "kappa": kappa}

    # ideal reference value of N_BLP on this time grid
    G = np.abs(analytic_G(times, **pars))
    n_blp_ideal = blp_from_curve(G)

    rows = []
    for s in scales:
        nm = build_noise_model(scale=s)
        # readout calibration under the same noise model
        M = None
        if nm is not None:
            cal_counts = qc_mod.run_counts(readout_calibration_circuits(),
                                           shots=shots, noise_model=nm, seed=seed)
            M = confusion_matrix(cal_counts)
        for mitig, Muse in (("raw", None), ("readout_mitigated", M)):
            if mitig == "readout_mitigated" and M is None:
                continue
            states = _shot_mode_states(qc_mod.stageA_shot_circuit, build_args,
                                       step_list, pair, nm, shots, seed, Muse)
            D = trace_distance_curve(states[pair[0]], states[pair[1]])
            rows.append({"noise_scale": s, "pipeline": mitig,
                         "N_BLP": blp_from_curve(D), "D_curve": D,
                         "noise_params": noise_params_dict(s)})
    out = {"times": times, "pair": pair, "shots": shots,
           "N_BLP_ideal_reference": n_blp_ideal, "rows": rows,
           "params": pars, "dt": dt}
    save_result("phase3_noise_robustness", {"scales": scales, "shots": shots,
                                            "dt": dt, **pars}, out, seed=seed)
    return out


def _collect_pair_counts(builder, build_args: dict, step_list: list[int],
                         pair: tuple[str, str], noise_model, shots: int, seed: int) -> dict:
    """Raw X/Y/Z counts for both states of a pair at each step (for bootstrap)."""
    circuits, index = [], []
    for lbl in pair:
        for k in step_list:
            for b in BASES:
                circuits.append(builder(init=lbl, n_steps=k, basis=b, **build_args))
                index.append((lbl, k, b))
    tqcs = qc_mod.transpile_for_noise(circuits, noise_model) if noise_model else circuits
    counts = qc_mod.run_counts(tqcs, shots=shots, noise_model=noise_model, seed=seed,
                               pre_transpiled=noise_model is not None)
    out = {lbl: {k: {} for k in step_list} for lbl in pair}
    for (lbl, k, b), c in zip(index, counts):
        out[lbl][k][b] = c
    return out


def phase3_bootstrap(quick: bool = False, shots: int = 4096, seed: int = 1234,
                     n_boot: int = 400) -> dict:
    """Statistical-rigor demonstration: bootstrap CIs, significance-debiased
    N_BLP, and the empirical Markovian null floor -- the shot-noise bias
    control that upgrades every shot-mode result to Q1 standard.
    """
    from .statistics import bootstrap_blp, null_floor
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    om_m, kap_m = stageA_params(**STAGEA_MARKOV)
    dt = 0.15 if quick else 0.1
    t_max = 9.0
    stride = 3 if quick else 2
    step_list = list(range(0, int(round(t_max / dt)) + 1, stride))
    times = np.array(step_list) * dt
    pair = ("+", "-")
    scales = [0.0, 1.0] if quick else [0.0, 0.5, 1.0, 2.0]

    G = np.abs(analytic_G(times, **pars))
    n_blp_ideal = blp_from_curve(G)

    rows = []
    for s in scales:
        nm = build_noise_model(scale=s)
        nm_counts = _collect_pair_counts(qc_mod.stageA_shot_circuit,
                                         {"dt": dt, "omega": omega, "kappa": kappa},
                                         step_list, pair, nm, shots, seed)
        # Markovian control (true N_BLP = 0) at the same shot budget & noise.
        ctrl_counts = _collect_pair_counts(qc_mod.stageA_shot_circuit,
                                           {"dt": dt, "omega": om_m, "kappa": kap_m},
                                           step_list, pair, nm, shots, seed)
        boot = bootstrap_blp(nm_counts, step_list, pair, n_boot=n_boot, seed=seed)
        floor = null_floor(ctrl_counts, step_list, pair)
        rows.append({
            "noise_scale": s,
            "N_BLP_point": boot["N_BLP_point"],
            "N_BLP_debiased": boot["N_BLP_debiased"],
            "N_BLP_ci_low": boot["N_BLP_ci_low"],
            "N_BLP_ci_high": boot["N_BLP_ci_high"],
            "N_BLP_boot_std": boot["N_BLP_boot_std"],
            "null_floor_markovian": floor,
            "noise_params": noise_params_dict(s),
        })
    out = {"times": times, "pair": pair, "shots": shots, "n_boot": n_boot,
           "N_BLP_ideal_reference": n_blp_ideal, "rows": rows, "params": pars, "dt": dt}
    save_result("phase3_bootstrap", {"scales": scales, "shots": shots,
                                     "n_boot": n_boot, "dt": dt, **pars}, out, seed=seed)
    return out


def _ensemble_member_states(pair, n_steps, dt, omega, kappa, angle_draw,
                            n_ensemble, save_every):
    """Per-member (not yet averaged) system trajectories under coherent RZ
    dephasing. Returns member[m][label] -> list of rho at saved steps, so the
    ensemble can be averaged AND bootstrapped over members."""
    steps = qc_mod.saved_steps(n_steps, save_every)
    members = []
    for m in range(n_ensemble):
        entry = {}
        for l in pair:
            data = qc_mod.run_exact([qc_mod.stageA_exact_circuit_dephased(
                l, n_steps, dt, omega, kappa, angle_draw(m), save_every=save_every)])[0]
            entry[l] = [data[f"rho_{k}"] for k in steps]
        members.append(entry)
    return members, np.array(steps) * dt


def _ensemble_average(members, pair, subset=None):
    idx = range(len(members)) if subset is None else subset
    n = len(list(idx))
    idx = range(len(members)) if subset is None else subset
    nsteps = len(members[0][pair[0]])
    acc = {l: [np.zeros((2, 2), dtype=complex) for _ in range(nsteps)] for l in pair}
    for m in idx:
        for l in pair:
            for j in range(nsteps):
                acc[l][j] += members[m][l][j] / n
    return acc


def _ensemble_blp_with_ci(members, pair, n_boot=300, seed=0):
    """Ensemble-averaged N_BLP plus a bootstrap-over-members 95% CI."""
    avg = _ensemble_average(members, pair)
    D = trace_distance_curve(avg[pair[0]], avg[pair[1]])
    point = blp_from_curve(D)
    rng = np.random.default_rng(seed)
    n = len(members)
    boots = []
    for _ in range(n_boot):
        sub = rng.integers(0, n, n)
        a = _ensemble_average(members, pair, subset=list(sub))
        boots.append(blp_from_curve(trace_distance_curve(a[pair[0]], a[pair[1]])))
    return {"N_BLP": point, "D_curve": D,
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
            "boot_std": float(np.std(boots))}


def phase3_noise_models(quick: bool = False, seed: int = 1234) -> dict:
    """Compare the memory signature under four noise conditions, exact mode:
    ideal, homemade Markovian depolarizing, realistic fake-backend (Markovian),
    and quasi-static (non-Markovian) dephasing vs its Markovian match at equal
    marginal strength. Directly answers 'but real device noise is
    non-Markovian'.
    """
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    dt = 0.1 if quick else 0.05
    t_max = 9.0
    n_steps = int(round(t_max / dt))
    save_every = 3 if quick else 2
    pair = ("+", "-")
    n_ens = 24 if quick else 60
    # Match the two dephasing models on ACCUMULATED phase variance sigma_tot^2,
    # not per-step: quasi-static uses one detuning spread over all steps
    # (angle sigma_tot/n_steps each step -> total sigma_tot); Markovian uses
    # fresh per-step angles sigma_tot/sqrt(n_steps) -> same total variance.
    # This isolates temporal correlation (memory) at equal total strength.
    sigma_tot = 0.6  # accumulated coherent-dephasing std (rad), O(1) = meaningful

    def blp_from_pair(states, steps_times):
        D = trace_distance_curve(states[pair[0]], states[pair[1]])
        return blp_from_curve(D), D

    times = np.array(qc_mod.saved_steps(n_steps, save_every)) * dt
    G = np.abs(analytic_G(times, **pars))
    results = {"N_BLP_ideal_reference": blp_from_curve(G)}

    # --- exact-mode circuit conditions (built-in Aer noise applied to gates)
    circuits = [qc_mod.stageA_exact_circuit(l, n_steps, dt, omega, kappa,
                save_every=save_every) for l in pair]
    conditions = {"ideal": None,
                  "markovian_depolarizing": build_noise_model(scale=1.0)}
    try:
        nm_fake, fake_meta = from_fake_backend("FakeManilaV2")
        conditions["realistic_fake_backend"] = nm_fake
        results["fake_backend_meta"] = fake_meta
    except Exception as exc:  # keep the study runnable if snapshot missing
        results["fake_backend_error"] = str(exc)

    for name, nm in conditions.items():
        res = qc_mod.run_exact(circuits, noise_model=nm, seed=seed)
        states = {l: qc_mod.exact_states_from_labels(r, n_steps, save_every)
                  for l, r in zip(pair, res)}
        nblp, D = blp_from_pair(states, times)
        results[name] = {"N_BLP": nblp, "D_curve": D}

    # --- quasi-static (non-Markovian) vs Markovian dephasing, matched on
    #     accumulated phase variance sigma_tot^2, SWEPT over sigma_tot, each
    #     with bootstrap-over-members CIs (M2). Finding restated: at matched
    #     strength, correlated (non-Markovian) noise PRESERVES more of the
    #     memory signature than memoryless noise; whether NM exceeds the ideal
    #     value is marginal and only claimed where the CIs separate.
    sigma_list = [0.6] if quick else [0.3, 0.6, 0.9, 1.2]
    sweep = []
    for st in sigma_list:
        Phi = quasistatic_angles(st, n_ens, seed=seed)
        rng = np.random.default_rng(seed + 1)
        per_step = rng.normal(0.0, st / np.sqrt(n_steps), size=(n_ens, n_steps))
        nm_mem, _ = _ensemble_member_states(pair, n_steps, dt, omega, kappa,
                                            lambda m, Phi=Phi: [Phi[m] / n_steps] * n_steps,
                                            n_ens, save_every)
        mk_mem, _ = _ensemble_member_states(pair, n_steps, dt, omega, kappa,
                                            lambda m, ps=per_step: ps[m], n_ens, save_every)
        nm = _ensemble_blp_with_ci(nm_mem, pair, seed=seed)
        mk = _ensemble_blp_with_ci(mk_mem, pair, seed=seed)
        # significance: do the two CIs separate?
        separated = nm["ci_low"] > mk["ci_high"] or mk["ci_low"] > nm["ci_high"]
        sweep.append({"sigma_tot": st,
                      "nonmarkovian": {k: nm[k] for k in ("N_BLP", "ci_low", "ci_high", "boot_std")},
                      "markovian": {k: mk[k] for k in ("N_BLP", "ci_low", "ci_high", "boot_std")},
                      "nm_minus_mk": nm["N_BLP"] - mk["N_BLP"],
                      "cis_separated": separated})
    # keep D-curves at the reference sigma_tot=0.6 for the figure
    ref_entry = next(e for e in sweep if abs(e["sigma_tot"] - 0.6) < 1e-9)
    results["nonmarkovian_dephasing"] = {"N_BLP": ref_entry["nonmarkovian"]["N_BLP"],
                                         "ci": [ref_entry["nonmarkovian"]["ci_low"],
                                                ref_entry["nonmarkovian"]["ci_high"]],
                                         "sigma_tot": 0.6}
    results["markovian_dephasing_matched"] = {"N_BLP": ref_entry["markovian"]["N_BLP"],
                                              "ci": [ref_entry["markovian"]["ci_low"],
                                                     ref_entry["markovian"]["ci_high"]],
                                              "sigma_tot": 0.6}
    results["sigma_sweep"] = sweep
    results["finding"] = ("At matched accumulated strength, correlated (non-Markovian) "
                          "dephasing preserves MORE of the memory signature than memoryless "
                          "dephasing (nm_minus_mk > 0); claim significant only where "
                          "cis_separated is True.")
    results["method_caveat"] = ("Quasi-static dephasing is applied stroboscopically (one "
                                "coherent RZ per Trotter step, after the dissipative block); "
                                "it approximates a continuous detuning to O(dt) and matches "
                                "the continuum as dt->0.")
    results.update({"times": times, "pair": pair, "dt": dt, "n_ensemble": n_ens,
                    "sigma_tot": 0.6, "params": pars})
    save_result("phase3_noise_models", {"dt": dt, "sigma_list": sigma_list,
                                        "n_ensemble": n_ens, **pars}, results, seed=seed)
    return results


def phase3_conflation_dt_check(quick: bool = False, seed: int = 1234) -> dict:
    """M3: the quasi-static dephasing is applied stroboscopically (O(dt)), so the
    small-sigma conflation gap could be a discretization artifact. We measure the
    correlated-minus-memoryless gap at sigma_tot = 0.3 (and 0.6) on two grids,
    dt = 0.1 and dt = 0.05, and check the gap is grid-stable and its CIs still
    separate. If the gap is not stable at sigma_tot = 0.3, the conflation claim
    is restricted to sigma_tot >= 0.6.
    """
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    t_max = 9.0
    n_ens = 30 if quick else 60
    sigmas = [0.3, 0.6]
    dts = [0.1, 0.05]
    rows = []
    for dtv in dts:
        n_steps = int(round(t_max / dtv))
        save_every = max(1, int(round(0.2 / dtv)))  # ~same physical time resolution
        for st in sigmas:
            Phi = quasistatic_angles(st, n_ens, seed=seed)
            rng = np.random.default_rng(seed + 1)
            per_step = rng.normal(0.0, st / np.sqrt(n_steps), size=(n_ens, n_steps))
            nm_mem, _ = _ensemble_member_states(("+", "-"), n_steps, dtv, omega, kappa,
                                                lambda m, Phi=Phi: [Phi[m] / n_steps] * n_steps,
                                                n_ens, save_every)
            mk_mem, _ = _ensemble_member_states(("+", "-"), n_steps, dtv, omega, kappa,
                                                lambda m, ps=per_step: ps[m], n_ens, save_every)
            nm = _ensemble_blp_with_ci(nm_mem, ("+", "-"), seed=seed)
            mk = _ensemble_blp_with_ci(mk_mem, ("+", "-"), seed=seed)
            sep = nm["ci_low"] > mk["ci_high"] or mk["ci_low"] > nm["ci_high"]
            rows.append({"dt": dtv, "sigma_tot": st, "gap": nm["N_BLP"] - mk["N_BLP"],
                         "nm": nm["N_BLP"], "mk": mk["N_BLP"], "cis_separated": sep,
                         "nm_ci": [nm["ci_low"], nm["ci_high"]],
                         "mk_ci": [mk["ci_low"], mk["ci_high"]]})
    # grid stability: gap(dt=0.1) vs gap(dt=0.05) at each sigma
    stability = {}
    for st in sigmas:
        g10 = next(r["gap"] for r in rows if r["dt"] == 0.1 and r["sigma_tot"] == st)
        g05 = next(r["gap"] for r in rows if r["dt"] == 0.05 and r["sigma_tot"] == st)
        stability[str(st)] = {"gap_dt0.1": g10, "gap_dt0.05": g05,
                              "abs_change": abs(g10 - g05)}
    out = {"rows": rows, "grid_stability": stability, "n_ensemble": n_ens,
           "t_max": t_max, "params": pars}
    save_result("phase3_conflation_dt_check", {"sigmas": sigmas, "dts": dts,
                                               "n_ensemble": n_ens}, out, seed=seed)
    return out


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (M1)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def phase3_estimator_validation(quick: bool = False, shots: int = 4096, seeds=None,
                                dt: float = 0.15,
                                save_name: str = "phase3_estimator_validation") -> dict:
    """M4: validate the shot-based estimators against a COMPUTABLE ground truth.

    The density-matrix simulator gives the exact noisy reduced states, hence
    the true noisy N_BLP at each noise scale. We measure the bias of the raw /
    debiased / readout-mitigated estimators against it over many seeds, sweep
    the debias threshold z, and check bootstrap-CI coverage of the truth with a
    Wilson interval on the coverage proportion itself (M1). The ``dt`` and
    ``save_name`` arguments allow a second, coarser grid for the grid-robustness
    check of the debiased estimator (M2b).
    """
    from .statistics import bootstrap_blp, rho_from_triplet
    if seeds is None:
        seeds = range(1, 6) if quick else range(1, 51)   # 50 seeds at full res
    seeds = list(seeds)
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    t_max = 9.0
    stride = 3 if quick else 2
    step_list = list(range(0, int(round(t_max / dt)) + 1, stride))
    pair = ("+", "-")
    scales = [0.0, 1.0] if quick else [0.0, 0.5, 1.0, 2.0]
    z_values = [0.5, 1.0, 2.0]
    build_args = {"dt": dt, "omega": omega, "kappa": kappa}

    rows = []
    for s in scales:
        nm = build_noise_model(scale=s)
        # ---- ground truth: exact noisy density-matrix N_BLP ----
        circuits = [qc_mod.stageA_exact_circuit(l, step_list[-1], dt, omega, kappa,
                    save_every=1) for l in pair]
        res = qc_mod.run_exact(circuits, noise_model=nm)
        true_states = {l: [qc_mod.exact_states_from_labels(r, step_list[-1])[k]
                           for k in step_list] for l, r in zip(pair, res)}
        true_blp = blp_from_curve(trace_distance_curve(true_states[pair[0]],
                                                       true_states[pair[1]]))
        # ---- estimators over seeds ----
        M = None
        if nm is not None:
            cal = qc_mod.run_counts(readout_calibration_circuits(), shots=shots,
                                    noise_model=nm, seed=999)
            M = confusion_matrix(cal)
        est = {"raw": [], "readout_mitigated": [], **{f"debiased_z{z}": [] for z in z_values}}
        covered = 0
        for sd in seeds:
            pc = _collect_pair_counts(qc_mod.stageA_shot_circuit, build_args,
                                      step_list, pair, nm, shots, sd)
            # raw
            raw_states = {l: [rho_from_triplet(pc[l][k]) for k in step_list] for l in pair}
            est["raw"].append(blp_from_curve(trace_distance_curve(
                raw_states[pair[0]], raw_states[pair[1]])))
            # readout-mitigated
            if M is not None:
                from .measures import rho_from_bloch
                mit = {l: [rho_from_bloch(np.array([mitigated_expectation(pc[l][k][b], M)
                        for b in BASES]), project=True) for k in step_list] for l in pair}
                est["readout_mitigated"].append(blp_from_curve(trace_distance_curve(
                    mit[pair[0]], mit[pair[1]])))
            # debiased at several z, plus CI coverage at z=1
            for z in z_values:
                b = bootstrap_blp(pc, step_list, pair, n_boot=200, seed=sd, z=z)
                est[f"debiased_z{z}"].append(b["N_BLP_debiased"])
                if z == 1.0 and b["N_BLP_ci_low"] <= true_blp <= b["N_BLP_ci_high"]:
                    covered += 1
        cov_lo, cov_hi = _wilson_ci(covered, len(seeds))
        row = {"noise_scale": s, "true_noisy_N_BLP": true_blp,
               "ci_coverage_z1": covered / len(seeds),
               "ci_coverage_wilson95": [cov_lo, cov_hi], "n_seeds": len(seeds)}
        for k, vals in est.items():
            if vals:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_bias"] = float(np.mean(vals) - true_blp)
                row[f"{k}_std"] = float(np.std(vals))
                # standard error of the mean bias, so bias is reported with an
                # uncertainty rather than as a point number
                row[f"{k}_bias_sem"] = float(np.std(vals) / np.sqrt(len(vals)))
        rows.append(row)
    out = {"rows": rows, "pair": pair, "shots": shots, "seeds": list(seeds),
           "z_values": z_values, "params": pars, "dt": dt, "n_seeds": len(seeds)}
    save_result(save_name, {"scales": scales, "shots": shots,
                            "n_seeds": len(seeds), "dt": dt}, out)
    return out


def phase3_zne(quick: bool = False, shots: int = 4096, seed: int = 1234,
               noise_scale: float = 1.0) -> dict:
    """ZNE recovery of the backflow curve at a fixed noise scale."""
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    dt = 0.15 if quick else 0.1
    t_max = 9.0
    stride = 3 if quick else 2
    step_list = list(range(0, int(round(t_max / dt)) + 1, stride))
    times = np.array(step_list) * dt
    pair = ("+", "-")
    fold_scales = [1, 3, 5]
    nm = build_noise_model(scale=noise_scale)
    cal_counts = qc_mod.run_counts(readout_calibration_circuits(), shots=shots,
                                   noise_model=nm, seed=seed)
    M = confusion_matrix(cal_counts)

    # Build + transpile all base circuits once, then fold.
    base, index = [], []
    for lbl in pair:
        for k in step_list:
            for b in BASES:
                base.append(qc_mod.stageA_shot_circuit(lbl, k, dt, omega, kappa, b))
                index.append((lbl, k, b))
    tbase = qc_mod.transpile_for_noise(base, nm)
    expect: dict[tuple, dict[int, float]] = {ix: {} for ix in index}
    for fs in fold_scales:
        folded = [fold_cx(c, fs) for c in tbase]
        counts = qc_mod.run_counts(folded, shots=shots, noise_model=nm,
                                   seed=seed + fs, pre_transpiled=True)
        for ix, ct in zip(index, counts):
            expect[ix][fs] = mitigated_expectation(ct, M)

    from .measures import rho_from_bloch
    states: dict[str, list[np.ndarray]] = {l: [] for l in pair}
    for lbl in pair:
        for k in step_list:
            r = np.array([richardson_extrapolate(np.array(fold_scales),
                          np.array([expect[(lbl, k, b)][fs] for fs in fold_scales]),
                          order=1)
                          for b in BASES])
            states[lbl].append(rho_from_bloch(r, project=True))
    D = trace_distance_curve(states[pair[0]], states[pair[1]])
    G = np.abs(analytic_G(times, **pars))
    out = {"times": times, "pair": pair, "noise_scale": noise_scale,
           "fold_scales": fold_scales, "D_curve_zne": D,
           "N_BLP_zne": blp_from_curve(D),
           "N_BLP_ideal_reference": blp_from_curve(G),
           "scope_note": ("ZNE folds CX only, so it amplifies/extrapolates the "
                          "2-qubit gate noise; residual 1q-gate and readout noise "
                          "are NOT extrapolated, leaving a known residual bias. "
                          "Order-1 (linear) extrapolation on folds {1,3,5}.")}
    save_result("phase3_zne", {"noise_scale": noise_scale, "shots": shots,
                               "dt": dt, **pars}, out, seed=seed)
    return out


# ===========================================================================
# Phase 5 (Tier 2) -- the required-Fock-dimension rule (paper centerpiece)
# ===========================================================================

def fock_rule_mean_only(n: float, c: float = 2.0) -> int:
    """Naive mean-only rule d = ceil(n + c*sqrt(n+1)) + 1. Shown to UNDERSHOOT
    for thermal (super-Poissonian) occupation -- a mean alone is insufficient."""
    return max(2, int(np.ceil(n + c * np.sqrt(n + 1.0)) + 1))


def fock_rule_variance_aware(n: float, std: float, c: float = 3.0) -> int:
    """Variance-aware rule d = ceil(n + c*std) + 1, using the actual number
    std at peak load. Accounts for super-Poissonian thermal statistics, so a
    single ``c`` bounds BOTH the coupling and temperature families."""
    return max(2, int(np.ceil(n + c * std) + 1))


def phase5_fock_rule(quick: bool = False, eps: float = 1e-2, c_var: float = 3.0) -> dict:
    """Required Fock dimension d_req vs the occupied phase-space extent, across
    TWO drivers of occupation: strong coupling (Stage B, T=0) and temperature
    (Stage A finite T).

    Finding (honest): a mean-only predictor does NOT collapse the two families
    (thermal needs more headroom). A VARIANCE-AWARE predictor
    n_max + c*std_at_peak does, and yields a single safe rule. d_req(eps) =
    smallest Fock dim whose reduced (+,-) dynamics matches a d_ref reference
    to max abs deviation < eps.
    """
    pair = ("+", "-")
    dt = 0.1 if quick else 0.05
    t_max = 8.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    points = []

    eps_list = [3e-2, 1e-2, 3e-3]  # M6: report d_req across accuracy targets

    def collect(family, param, occ_fn, d_ref):
        states_at_d, n_max, std_pk = {}, None, None
        for d in range(2, d_ref + 1):
            traj = {}
            for l in pair:
                s, pk, sd = occ_fn(l, d)
                traj[l] = s
                if d == d_ref:
                    if n_max is None or pk > n_max:
                        n_max, std_pk = pk, sd
            states_at_d[d] = [x for l in pair for x in traj[l]]
        d_req, censored = required_fock_dimension(states_at_d, d_ref, eps)
        d_req_by_eps = {}
        for e in eps_list:
            dr, cens = required_fock_dimension(states_at_d, d_ref, e)
            d_req_by_eps[str(e)] = {"d_req": dr, "censored": cens}
        d_var = fock_rule_variance_aware(n_max, std_pk, c_var)
        points.append({"family": family, "param": param, "n_max": n_max,
                       "std_at_peak": std_pk, "d_req": d_req, "censored": censored,
                       "d_req_by_eps": d_req_by_eps,
                       "d_rule_mean": fock_rule_mean_only(n_max),
                       "d_rule_var": d_var, "slack": d_var - d_req})

    # Family A: temperature (Stage A finite T). d_ref raised to 10 (M6: margin
    # to censoring was only 1 level at d_ref=8).
    n_ths = [0.1, 0.3, 0.6, 1.0] if quick else [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for n_th in n_ths:
        collect("temperature", n_th,
                lambda l, d, n_th=n_th: stageA_finiteT_with_occupation(
                    dm(l), times, **STAGEA_NONMARKOV, n_th=n_th, n_fock=d), d_ref=10)

    # Family B: coupling (Stage B spin-boson, T=0)
    mode_proto = [(1.0, None, 0.4), (1.6, None, 0.8)]
    gs = [0.3, 0.5, 0.7] if quick else [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for g in gs:
        modes = [(w, g, kap) for (w, _, kap) in mode_proto]
        collect("coupling", g,
                lambda l, d, modes=modes: spin_boson_with_occupation(
                    dm(l), times, 1.0, 0.0, modes, n_fock=d), d_ref=8)

    def corr(key):
        x = np.array([p[key] for p in points]); y = np.array([p["d_req"] for p in points])
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

    # M6: CROSS-VALIDATION -- calibrate c on one family, test safety on the
    # OTHER (out-of-sample), so the "safe bound" is not an in-sample fit.
    def min_safe_c(cal_pts):
        # smallest c making the variance rule a safe bound on cal_pts
        c = 0.0
        while c < 8.0:
            if all(fock_rule_variance_aware(p["n_max"], p["std_at_peak"], c) >= p["d_req"]
                   for p in cal_pts):
                return c
            c += 0.25
        return c
    temp_pts = [p for p in points if p["family"] == "temperature"]
    coup_pts = [p for p in points if p["family"] == "coupling"]
    c_from_temp = min_safe_c(temp_pts)
    c_from_coup = min_safe_c(coup_pts)
    xval_temp_on_coup = all(fock_rule_variance_aware(p["n_max"], p["std_at_peak"],
                            c_from_temp) >= p["d_req"] for p in coup_pts)
    xval_coup_on_temp = all(fock_rule_variance_aware(p["n_max"], p["std_at_peak"],
                            c_from_coup) >= p["d_req"] for p in temp_pts)

    # R3.1: explicit over/under-provisioning of the variance rule across a small
    # sweep of the constant c, per family, so the "safe c" claim is auditable.
    # d_rule(c) - d_req >= 0 everywhere == safe; the max is the worst over-spend.
    def provisioning(cal_pts, c):
        slacks = [fock_rule_variance_aware(p["n_max"], p["std_at_peak"], c) - p["d_req"]
                  for p in cal_pts]
        return {"safe": bool(all(s >= 0 for s in slacks)),
                "max_over_provision": int(max(slacks)),
                "min_slack": int(min(slacks)),
                "n_under_provisioned": int(sum(1 for s in slacks if s < 0)),
                "n_points": len(slacks)}
    c_sweep = {f"{c:g}": {"temperature": provisioning(temp_pts, c),
                          "coupling": provisioning(coup_pts, c)}
               for c in (2.0, 2.75, 3.0)}

    predictor = np.array([p["n_max"] + c_var * p["std_at_peak"] for p in points])
    out = {"points": points, "eps": eps, "c_var": c_var, "eps_list": eps_list,
           "c_sweep": c_sweep,
           "any_censored": any(p["censored"] for p in points),
           "mean_rule_is_safe": all(p["d_rule_mean"] >= p["d_req"] for p in points),
           "variance_rule_is_safe": all(p["d_rule_var"] >= p["d_req"] for p in points),
           "max_slack": max(p["slack"] for p in points),
           "corr_dreq_vs_nmax": corr("n_max"),
           "corr_dreq_vs_variance_predictor": float(
               np.corrcoef(predictor, [p["d_req"] for p in points])[0, 1]),
           "crossval": {"c_from_temperature": c_from_temp, "c_from_coupling": c_from_coup,
                        "temp_c_safe_on_coupling": xval_temp_on_coup,
                        "coup_c_safe_on_temperature": xval_coup_on_temp},
           "dt": dt, "t_max": t_max}
    save_result("phase5_fock_rule", {"eps": eps, "c_var": c_var, "quick": quick}, out)
    return out


# ===========================================================================
# BLP pair-optimality certificate (M3) -- how tight is the canonical bound?
# ===========================================================================

def phase_blp_pair_certificate(quick: bool = False) -> dict:
    """Certify the canonical-3-pair BLP against a Bloch-sphere-optimised BLP
    for spin-boson and finite-T channels, where no analytic optimum is known.
    Small gap => the canonical value is a tight lower bound and can be reported
    as N_BLP; large gap => must report a lower bound only.
    """
    dt, t_max = 0.05, 8.0
    n = int(round(t_max / dt))
    times = np.linspace(0.0, n * dt, n + 1)
    rows = []

    # spin-boson at a few couplings
    for g in ([0.3, 0.6] if quick else [0.2, 0.4, 0.6, 0.8]):
        modes = [(1.0, g, 0.4), (1.6, g, 0.8)]
        ch = lambda r0, modes=modes: spin_boson_qutip_states(r0, times, 1.0, 0.0, modes, n_fock=6)
        cert = blp_measure_optimized(ch)
        rows.append({"family": "spin_boson", "param": g, **cert})

    # finite-T Stage A at a few temperatures
    for n_th in ([0.5, 1.0] if quick else [0.25, 0.5, 1.0, 1.5]):
        ch = lambda r0, n_th=n_th: stageA_qutip_states_finiteT(r0, times, **STAGEA_NONMARKOV,
                                                               n_th=n_th, n_fock=6)
        cert = blp_measure_optimized(ch)
        rows.append({"family": "finite_T", "param": n_th, **cert})

    max_gap = max(r["gap"] for r in rows)
    rel = max((r["gap"] / r["N_BLP_optimized"]) if r["N_BLP_optimized"] > 1e-6 else 0.0
              for r in rows)
    out = {"rows": rows, "max_abs_gap": max_gap, "max_rel_gap": rel,
           "canonical_is_tight": max_gap < 0.02}
    save_result("phase_blp_pair_certificate", {"dt": dt, "t_max": t_max}, out)
    return out


# ===========================================================================
# Pair-optimised BLP and Fock-dimension convergence at the reported points
# ===========================================================================

_PAIR_KEYS = ("N_BLP_canonical", "best_canonical_pair", "canonical_by_pair",
              "N_BLP_optimized", "gap", "n_opt", "budget_change")


def _ic_states(evolver, times) -> dict[str, np.ndarray]:
    """Trajectories of the four IC inputs under ``evolver(rho0, times)``."""
    return {l: np.asarray(evolver(dm(l), times)) for l in IC_INPUTS}


def _six_from_ic(states) -> dict[str, np.ndarray]:
    """All six canonical-state trajectories from the four IC ones, by linearity:
    rho_- = rho_0 + rho_1 - rho_+ and rho_-i = rho_0 + rho_1 - rho_+i."""
    o0, o1 = np.asarray(states["0"]), np.asarray(states["1"])
    op, oi = np.asarray(states["+"]), np.asarray(states["+i"])
    return {"0": o0, "1": o1, "+": op, "-": o0 + o1 - op, "+i": oi, "-i": o0 + o1 - oi}


def _max_trace_norm_dev(a, b) -> float:
    """max over the six canonical inputs and all times of ||rho_a - rho_b||_1."""
    sa, sb = _six_from_ic(a), _six_from_ic(b)
    return max(trace_norm(x - y) for l in sa for x, y in zip(sa[l], sb[l]))


def _max_abs_dev(a, b) -> float:
    """Element-wise max deviation over the six canonical inputs (the metric of
    the paper's tables)."""
    sa, sb = _six_from_ic(a), _six_from_ic(b)
    return max(float(np.max(np.abs(sa[l] - sb[l]))) for l in sa)


def _pair_summary(states) -> tuple[dict, np.ndarray]:
    """Canonical and pair-optimised BLP of a channel, and its Pauli images."""
    images = pauli_images_from_states(states)
    res = blp_optimize_pair(images)
    return {k: res[k] for k in _PAIR_KEYS}, images


def _recorded_table_values(family: str) -> dict:
    """Canonical BLP values recorded for Table 1 (finite_T) or Table 2
    (spin_boson), keyed by parameter, so a rerun can show that it reproduces
    them. Empty if the record is absent."""
    name, key = {"finite_T": ("phase1T_finite_temperature", "n_th"),
                 "spin_boson": ("phase2_spin_boson", "g")}[family]
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["data"]["rows"]
    return {float(r[key]): {"circuit": r["N_BLP_circuit"], "d2": r["N_BLP_ref_nfock2"],
                            "d6": r["N_BLP_ref_converged"]} for r in rows}


def tables_pair_and_fock_convergence(family: str, d_max: int = 10,
                                     quick: bool = False) -> dict:
    """Pair-optimised BLP and Fock-dimension convergence at the points of
    Table 1 (``family="finite_T"``) or Table 2 (``family="spin_boson"``).

    Same models, Trotter step (0.05), BLP sampling and window (t_max = 10) as
    the recorded Table runs. For each point: the circuit, and the QuTiP
    reference at every Fock dimension d = 2..d_max, each with its canonical and
    pair-optimised BLP. Convergence in d is reported for the measure,
    eps_NM(d) = |N(d) - N(d_max)|, and for the states,
    max_t ||rho^(d+1)(t) - rho^(d)(t)||_1 over the six canonical inputs. The
    circuit/reference comparison is also evaluated with each side's optimal
    pair applied to the other (the pair sensitivity of their ratio).
    """
    dt, t_max = 0.05, 10.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    if family == "finite_T":
        pars = STAGEA_NONMARKOV
        omega, kappa = stageA_params(**pars)
        params = [0.0, 0.25, 1.0] if quick else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
        config = {"model": "Stage A at finite temperature, pseudomode starting in its "
                           "ground state", **pars, "n_th": params}

        def reference(p, d):
            return _ic_states(lambda r0, t: stageA_qutip_states_finiteT(
                r0, t, **pars, n_th=p, n_fock=d), times)

        def circuits(p):
            return [qc_mod.stageA_exact_circuit(l, n_steps, dt, omega, kappa, n_th=p)
                    for l in IC_INPUTS]
    elif family == "spin_boson":
        delta, eps = 1.0, 0.0
        proto = [(1.0, 0.4), (1.6, 0.8)]            # (w_k, kappa_k), as in phase2
        params = [0.1, 0.4, 0.7] if quick else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
        config = {"model": "Stage B spin-boson at T = 0", "delta": delta, "eps": eps,
                  "modes_w_kappa": proto, "g": params}

        def reference(p, d):
            modes = [(w, p, k) for w, k in proto]
            return _ic_states(lambda r0, t: spin_boson_qutip_states(
                r0, t, delta, eps, modes, n_fock=d), times)

        def circuits(p):
            modes = [(w, p, k) for w, k in proto]
            return [qc_mod.stageB_exact_circuit(l, n_steps, dt, delta, eps, modes)
                    for l in IC_INPUTS]
    else:
        raise ValueError(f"unknown family {family!r}")

    recorded = _recorded_table_values(family)
    dims = list(range(2, d_max + 1))
    rows = []
    for p in params:
        res = qc_mod.run_exact(circuits(p))
        circ = {l: np.asarray(qc_mod.exact_states_from_labels(r, n_steps))
                for l, r in zip(IC_INPUTS, res)}
        circ_sum, circ_images = _pair_summary(circ)
        refs, sums, images = {}, {}, {}
        for d in dims:
            refs[d] = reference(p, d)
            sums[d], images[d] = _pair_summary(refs[d])
        top = sums[d_max]
        per_d = []
        for d in dims:
            entry = {"d": d, **sums[d],
                     "eps_NM_canonical": abs(sums[d]["N_BLP_canonical"] - top["N_BLP_canonical"]),
                     "eps_NM_optimized": abs(sums[d]["N_BLP_optimized"] - top["N_BLP_optimized"]),
                     "max_abs_dev_to_dmax": _max_abs_dev(refs[d], refs[d_max])}
            if d < d_max:
                entry["max_trace_norm_dev_to_next_d"] = _max_trace_norm_dev(refs[d], refs[d + 1])
            per_d.append(entry)
        n_circ, n_ref = np.array(circ_sum["n_opt"]), np.array(top["n_opt"])
        row = {"param": p, "circuit": circ_sum,
               "cross_pairs": {
                   "circuit_at_reference_pair": float(
                       blp_direction_values(circ_images, n_ref[None, :])[0]),
                   "reference_at_circuit_pair": float(
                       blp_direction_values(images[d_max], n_circ[None, :])[0])},
               "circuit_vs_d2_max_abs_dev": _max_abs_dev(circ, refs[2]),
               "per_d": per_d}
        rec = recorded.get(float(p))
        if rec is not None:
            row["recorded_canonical"] = rec
            row["reproduction_diff"] = {
                "circuit": circ_sum["N_BLP_canonical"] - rec["circuit"],
                "d2": sums[2]["N_BLP_canonical"] - rec["d2"],
                "d6": (sums[6]["N_BLP_canonical"] - rec["d6"]) if 6 in sums else None}
        rows.append(row)
    out = {"family": family, "rows": rows, "dims": dims, "d_max": d_max,
           "config": config, "trotter_dt": dt, "blp_sampling_interval": dt,
           "t_max": t_max,
           "pairs": ("canonical = best of the three canonical pairs; optimized = "
                     "blp_optimize_pair over all antipodal pure-state pairs")}
    save_result(f"pair_fock_convergence_{family}",
                {"family": family, "d_max": d_max, "quick": quick, "params": params,
                 "trotter_dt": dt, "t_max": t_max}, out)
    return out


def phase_diagram_pair_and_fock_check(points=None, d_max: int = 10) -> dict:
    """Pair optimisation and Fock convergence at selected points of the Fig. 4
    phase diagram (finite-T spin-boson; BLP sampled every 0.1 up to t_max = 8,
    as in the figure). The figure uses the (+,-) pair and d = 6; the signed
    error n=2 minus reference is given for the (+,-) pair at d = 6 and at
    d_max, and for the optimised pair at d_max. QuTiP only: the figure has no
    circuit data."""
    delta, eps = 1.0, 0.0
    proto = [(1.0, 0.4), (1.6, 0.8)]
    dt, t_max = 0.1, 8.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    if points is None:
        # the four corners of the plotted plane, then points inside each region
        points = [(0.05, 0.0), (0.05, 1.5), (0.8, 0.0), (0.8, 1.5),
                  (0.3, 0.0), (0.4, 0.0), (0.3, 1.0), (0.5, 0.5)]
    dims = list(range(2, d_max + 1))
    rows = []
    for g, n_th in points:
        modes = [(w, g, k) for w, k in proto]
        refs, sums = {}, {}
        for d in dims:
            refs[d] = _ic_states(lambda r0, t: spin_boson_qutip_states_finiteT(
                r0, t, delta, eps, modes, n_th=n_th, n_fock=d), times)
            sums[d] = _pair_summary(refs[d])[0]
        top = sums[d_max]
        pm = {d: sums[d]["canonical_by_pair"]["+|-"] for d in dims}
        per_d = []
        for d in dims:
            entry = {"d": d, **sums[d], "plus_minus": pm[d],
                     "eps_NM_plus_minus": abs(pm[d] - pm[d_max]),
                     "eps_NM_optimized": abs(sums[d]["N_BLP_optimized"] - top["N_BLP_optimized"])}
            if d < d_max:
                entry["max_trace_norm_dev_to_next_d"] = _max_trace_norm_dev(refs[d], refs[d + 1])
            per_d.append(entry)
        rows.append({"g": g, "n_th": n_th, "per_d": per_d,
                     "signed_error_plus_minus_d6": (pm[2] - pm[6]) if 6 in pm else None,
                     "signed_error_plus_minus_dmax": pm[2] - pm[d_max],
                     "signed_error_optimized_dmax": sums[2]["N_BLP_optimized"] - top["N_BLP_optimized"]})
    out = {"rows": rows, "dims": dims, "d_max": d_max, "points": points,
           "model": "finite-T spin-boson, pseudomodes starting in their ground state",
           "delta": delta, "eps": eps, "modes_w_kappa": proto,
           "blp_sampling_interval": dt, "t_max": t_max}
    save_result("phase_diagram_pair_fock_check",
                {"points": points, "d_max": d_max, "blp_sampling_interval": dt,
                 "t_max": t_max}, out)
    return out


def noisy_pair_check(scales=(0.0, 0.5, 1.0, 2.0), dt: float = 0.15) -> dict:
    """Is the (+,-) pair used in Sections 5-6 still BLP-optimal under the device
    noise? Exact noisy channels (density-matrix simulation, no shots) of the
    Stage A circuit on the Table 3 grid: Trotter step 0.15, BLP sampled every
    2 steps, t_max = 9. The (+,-) value is Table 3's 'true N_BLP'."""
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    t_max, stride = 9.0, 2
    step_list = list(range(0, int(round(t_max / dt)) + 1, stride))
    recorded = {}
    path = RESULTS_DIR / "phase3_estimator_validation.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            recorded = {float(r["noise_scale"]): r["true_noisy_N_BLP"]
                        for r in json.load(fh)["data"]["rows"]}
    rows = []
    for s in scales:
        nm = build_noise_model(scale=s)
        res = qc_mod.run_exact([qc_mod.stageA_exact_circuit(l, step_list[-1], dt, omega, kappa,
                                                            save_every=1) for l in IC_INPUTS],
                               noise_model=nm)
        states = {}
        for l, r in zip(IC_INPUTS, res):
            full = qc_mod.exact_states_from_labels(r, step_list[-1])
            states[l] = np.asarray([full[k] for k in step_list])
        summary, _ = _pair_summary(states)
        row = {"noise_scale": s, "noise_params": noise_params_dict(s),
               "N_BLP_plus_minus": summary["canonical_by_pair"]["+|-"], **summary}
        if float(s) in recorded:
            row["recorded_true_noisy_N_BLP"] = recorded[float(s)]
        rows.append(row)
    out = {"rows": rows, "params": pars, "trotter_dt": dt,
           "blp_sampling_interval": dt * stride, "t_max": t_max}
    save_result("noisy_pair_check", {"scales": list(scales), "trotter_dt": dt,
                                     "stride": stride, "t_max": t_max}, out)
    return out


def _phase9_point(args) -> dict:
    """One (g, n_th) point of the pair-optimised phase diagram: canonical (+,-)
    and optimised BLP at each Fock dimension in ``dims`` (a process-pool worker)."""
    g, n_th, dims, times, delta, eps, proto = args
    modes = [(w, g, k) for w, k in proto]
    out = {}
    for d in dims:
        states = _ic_states(lambda r0, t: spin_boson_qutip_states_finiteT(
            r0, t, delta, eps, modes, n_th=n_th, n_fock=d, options=TIGHT_SOLVER_OPTIONS), times)
        s, _ = _pair_summary(states)
        out[d] = {"plus_minus": s["canonical_by_pair"]["+|-"],
                  "optimized": s["N_BLP_optimized"], "n_opt": s["n_opt"],
                  "budget_change": s["budget_change"]}
    return out


def phase9_truncation_phase_diagram_optimized(quick: bool = False, d_ref: int = 10,
                                              d_check: int = 9, n_workers: int = 1,
                                              grid: tuple[int, int] | None = None,
                                              sampling: float = 0.1) -> dict:
    """The Fig. 4 phase diagram recomputed with pair-optimised BLP.

    Signed truncation error N(n_fock=2) - N(d_ref) over the (coupling,
    temperature) plane of the finite-T spin-boson, on the figure's plane,
    window and sampling (t_max = 8, every 0.1), for the optimised pair and for
    the (+,-) pair the figure used. The reference is checked against d_check
    at every point (reference_check_max_abs). ``grid`` = (n_g, n_th) overrides
    the default 44 x 34 (14 x 12 when quick); ``n_workers`` > 1 spreads the
    points over a process pool; ``sampling`` is the BLP grid (0.1 in the
    figure). Solves use TIGHT_SOLVER_OPTIONS.
    """
    delta, eps = 1.0, 0.0
    proto = [(1.0, 0.4), (1.6, 0.8)]
    dt, t_max = sampling, 8.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    n_g, n_n = grid if grid is not None else ((14, 12) if quick else (44, 34))
    gs = np.linspace(0.05, 0.8, n_g)
    n_ths = np.linspace(0.0, 1.5, n_n)
    dims = sorted({2, d_check, d_ref})
    tasks = [(float(g), float(n), dims, times, delta, eps, proto) for n in n_ths for g in gs]
    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_phase9_point, tasks, chunksize=2))
    else:
        results = [_phase9_point(t) for t in tasks]
    shape = (len(n_ths), len(gs))

    def field(d, key):
        return np.array([r[d][key] for r in results]).reshape(shape)

    ref_check = np.maximum(np.abs(field(d_check, "optimized") - field(d_ref, "optimized")),
                           np.abs(field(d_check, "plus_minus") - field(d_ref, "plus_minus")))
    out = {"g": gs, "n_th": n_ths, "d_ref": d_ref, "d_check": d_check,
           "signed_error_optimized": field(2, "optimized") - field(d_ref, "optimized"),
           "signed_error_plus_minus": field(2, "plus_minus") - field(d_ref, "plus_minus"),
           "reference_optimized": field(d_ref, "optimized"),
           "reference_plus_minus": field(d_ref, "plus_minus"),
           "nfock2_optimized": field(2, "optimized"),
           "nfock2_plus_minus": field(2, "plus_minus"),
           "n_opt_reference": np.array([r[d_ref]["n_opt"] for r in results]).reshape(shape + (3,)),
           "n_opt_nfock2": np.array([r[2]["n_opt"] for r in results]).reshape(shape + (3,)),
           "reference_check_max_abs": ref_check,
           "max_budget_change": max(r[d]["budget_change"] for r in results for d in dims),
           "model": "finite-T spin-boson, pseudomodes starting in their ground state",
           "delta": delta, "eps": eps, "modes_w_kappa": proto,
           "blp_sampling_interval": dt, "t_max": t_max}
    save_result("phase9_truncation_phase_diagram_optimized",
                {"grid": [n_g, n_n], "d_ref": d_ref, "d_check": d_check,
                 "blp_sampling_interval": dt, "t_max": t_max,
                 "solver_options": TIGHT_SOLVER_OPTIONS}, out)
    return out


# ===========================================================================
# Convergence of the BLP value in time resolution
# ===========================================================================

def _is_multiple(s: float, dt: float) -> bool:
    ratio = s / dt
    return ratio >= 1.0 - 1e-9 and abs(ratio - round(ratio)) < 1e-9


def _sampled_summary(states, stride: int) -> dict:
    """Canonical and pair-optimised BLP of trajectories sampled every ``stride``."""
    s, _ = _pair_summary({l: np.asarray(v)[::stride] for l, v in states.items()})
    return {"canonical": s["N_BLP_canonical"], "plus_minus": s["canonical_by_pair"]["+|-"],
            "optimized": s["N_BLP_optimized"], "n_opt": s["n_opt"]}


def stageA_blp_time_convergence(t_maxes=(10.0, 12.0),
                                samplings=(0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125)) -> dict:
    """Discrete BLP of the Stage A optimal pair, D = |G|, on decreasing sampling
    steps, against the closed-form continuum value (analytic_blp_continuum)."""
    pars = STAGEA_NONMARKOV
    rows = []
    for t_max in t_maxes:
        exact = analytic_blp_continuum(t_max, **pars)
        vals = []
        for s in samplings:
            n = int(round(t_max / s))
            vals.append(blp_from_curve(np.abs(analytic_G(np.linspace(0.0, n * s, n + 1), **pars))))
        rows.append({"t_max": t_max, "continuum": exact, "sampling": list(samplings),
                     "N_BLP": vals, "deficit": [exact - v for v in vals]})
    out = {"rows": rows, "params": pars,
           "note": "D(t) = |G(t)| is the optimal-pair trace distance of Stage A."}
    save_result("stageA_blp_time_convergence",
                {"t_maxes": list(t_maxes), "samplings": list(samplings)}, out)
    return out


def tables_blp_time_convergence(family: str, d_ref: int,
                                samplings=(0.05, 0.025, 0.0125, 0.00625, 0.003125),
                                trotter_dts=(0.05, 0.025, 0.0125), quick: bool = False) -> dict:
    """Convergence of the Table 1 (finite_T) or Table 2 (spin_boson) BLP values
    in time resolution.

    Reference (QuTiP at Fock dimension d_ref) and matched truncation (QuTiP at
    n_fock = 2, the circuit's model without Trotter error): each solved once at
    the finest sampling and subsampled, so only the sampling changes. Circuit:
    rerun at each Trotter step and sampled at every multiple of it in
    ``samplings``, so Trotter and sampling effects can be told apart.
    Canonical, (+,-) and pair-optimised values throughout; window t_max = 10
    as in the tables.
    """
    t_max = 10.0
    fine = min(samplings)
    n_fine = int(round(t_max / fine))
    times = np.linspace(0.0, n_fine * fine, n_fine + 1)
    if family == "finite_T":
        pars = STAGEA_NONMARKOV
        omega, kappa = stageA_params(**pars)
        params = [0.25, 1.0] if quick else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

        def reference(p, d):
            return _ic_states(lambda r0, t: stageA_qutip_states_finiteT(
                r0, t, **pars, n_th=p, n_fock=d, options=TIGHT_SOLVER_OPTIONS), times)

        def circuits(p, dt, n):
            return [qc_mod.stageA_exact_circuit(l, n, dt, omega, kappa, n_th=p)
                    for l in IC_INPUTS]
    elif family == "spin_boson":
        delta, eps = 1.0, 0.0
        proto = [(1.0, 0.4), (1.6, 0.8)]
        params = [0.4, 0.7] if quick else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

        def reference(p, d):
            modes = [(w, p, k) for w, k in proto]
            return _ic_states(lambda r0, t: spin_boson_qutip_states(
                r0, t, delta, eps, modes, n_fock=d, options=TIGHT_SOLVER_OPTIONS), times)

        def circuits(p, dt, n):
            modes = [(w, p, k) for w, k in proto]
            return [qc_mod.stageB_exact_circuit(l, n, dt, delta, eps, modes)
                    for l in IC_INPUTS]
    else:
        raise ValueError(f"unknown family {family!r}")

    rows = []
    for p in params:
        ref, matched = reference(p, d_ref), reference(p, 2)
        row = {"param": p,
               "reference": {f"{s:g}": _sampled_summary(ref, int(round(s / fine)))
                             for s in samplings},
               "nfock2": {f"{s:g}": _sampled_summary(matched, int(round(s / fine)))
                          for s in samplings},
               "circuit": {}}
        for dt in trotter_dts:
            n = int(round(t_max / dt))
            res = qc_mod.run_exact(circuits(p, dt, n))
            states = {l: np.asarray(qc_mod.exact_states_from_labels(r, n))
                      for l, r in zip(IC_INPUTS, res)}
            row["circuit"][f"{dt:g}"] = {f"{s:g}": _sampled_summary(states, int(round(s / dt)))
                                         for s in samplings if _is_multiple(s, dt)}
        rows.append(row)
    out = {"family": family, "d_ref": d_ref, "t_max": t_max, "samplings": list(samplings),
           "trotter_dts": list(trotter_dts), "rows": rows}
    save_result(f"blp_time_convergence_{family}",
                {"family": family, "d_ref": d_ref, "samplings": list(samplings),
                 "trotter_dts": list(trotter_dts), "quick": quick,
                 "solver_options": TIGHT_SOLVER_OPTIONS}, out)
    return out


def phase_diagram_blp_time_convergence(points=None, d_ref: int = 10,
                                       samplings=(0.1, 0.05, 0.025, 0.0125, 0.00625)) -> dict:
    """Time-resolution convergence of the Fig. 4 signed truncation error at
    selected points: QuTiP at n_fock = 2 and d_ref, one fine solve per point,
    subsampled; (+,-) and pair-optimised values; window t_max = 8."""
    delta, eps = 1.0, 0.0
    proto = [(1.0, 0.4), (1.6, 0.8)]
    t_max = 8.0
    fine = min(samplings)
    n_fine = int(round(t_max / fine))
    times = np.linspace(0.0, n_fine * fine, n_fine + 1)
    if points is None:
        points = [(0.05, 0.0), (0.05, 1.5), (0.8, 0.0), (0.8, 1.5),
                  (0.3, 0.0), (0.4, 0.0), (0.3, 1.0), (0.5, 0.5)]
    rows = []
    for g, n_th in points:
        modes = [(w, g, k) for w, k in proto]
        vals = {}
        for d in (2, d_ref):
            states = _ic_states(lambda r0, t: spin_boson_qutip_states_finiteT(
                r0, t, delta, eps, modes, n_th=n_th, n_fock=d,
                options=TIGHT_SOLVER_OPTIONS), times)
            vals[d] = {f"{s:g}": _sampled_summary(states, int(round(s / fine)))
                       for s in samplings}
        signed = {k: {key: vals[2][k][key] - vals[d_ref][k][key]
                      for key in ("plus_minus", "optimized")} for k in vals[2]}
        rows.append({"g": g, "n_th": n_th, "nfock2": vals[2], "reference": vals[d_ref],
                     "signed_error": signed})
    out = {"rows": rows, "d_ref": d_ref, "samplings": list(samplings), "t_max": t_max,
           "model": "finite-T spin-boson, pseudomodes starting in their ground state"}
    save_result("phase_diagram_blp_time_convergence",
                {"points": points, "d_ref": d_ref, "samplings": list(samplings),
                 "solver_options": TIGHT_SOLVER_OPTIONS}, out)
    return out


def noisy_blp_grid_dependence(scales=(0.0, 0.5, 1.0, 2.0), dt: float = 0.15,
                              strides=(4, 2, 1)) -> dict:
    """Grid dependence of the exact noisy BLP behind Table 3 (Trotter step fixed
    at dt, sampling stride*dt; the table uses stride 2). Reported, not used to
    change the table's grid (R8 scope)."""
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    t_max = 9.0
    n = int(round(t_max / dt))
    rows = []
    for s in scales:
        res = qc_mod.run_exact([qc_mod.stageA_exact_circuit(l, n, dt, omega, kappa)
                                for l in IC_INPUTS], noise_model=build_noise_model(scale=s))
        states = {l: np.asarray(qc_mod.exact_states_from_labels(r, n))
                  for l, r in zip(IC_INPUTS, res)}
        rows.append({"noise_scale": s,
                     "by_sampling": {f"{k * dt:g}": _sampled_summary(states, k) for k in strides}})
    out = {"rows": rows, "trotter_dt": dt, "strides": list(strides), "t_max": t_max,
           "params": pars}
    save_result("noisy_blp_grid_dependence",
                {"scales": list(scales), "trotter_dt": dt, "strides": list(strides)}, out)
    return out


# ===========================================================================
# Excitonic dimer: reference convergence, Trotter convergence, and memory
# ===========================================================================

def dimer_convergence_and_memory(d_max: int = 8, trotter_dts=(0.05, 0.025, 0.0125),
                                 samplings=(0.05, 0.025, 0.0125, 0.00625, 0.003125,
                                            0.0015625, 0.00078125),
                                 quick: bool = False) -> dict:
    """The dimer of Sec. 8 revisited on its excitonic qubit {|L>, |R>}.

    (i) Reference convergence in Fock dimension d = 2..d_max: pair-optimised
    BLP of the excitonic qubit, the acceptor population P_R(t) from |L>, and
    max_t ||rho^(d+1)(t) - rho^(d)(t)||_1 over the six canonical inputs.
    (ii) The circuit (two-level modes) at each Trotter step, compared on the
    reference grid with the matched (d = 2) and the top reference.
    (iii) The isolated dimer (g = 0) as a baseline: its dynamics is unitary, so
    its BLP must vanish, and its population revivals are coherent.
    (iv) Time resolution of the memory value: the top reference and the
    isolated dimer solved once on the finest of ``samplings`` and subsampled.
    Same parameters, grid (0.05) and window (t_max = 10) as phase8_dimer.
    """
    from .reference import dimer_qutip_excitonic_states
    eps_L, eps_R, J = 0.5, -0.5, 1.0
    site_modes = [(0.8, 0.25, 0.3), (0.8, 0.25, 0.3)]
    isolated = [(w, 0.0, k) for (w, _, k) in site_modes]
    t_max, dt_ref = 10.0, 0.05
    n_ref = int(round(t_max / dt_ref))
    times = np.linspace(0.0, n_ref * dt_ref, n_ref + 1)
    dims = list(range(2, (4 if quick else d_max) + 1))

    def reference(modes, d):
        return _ic_states(lambda r0, t: dimer_qutip_excitonic_states(
            r0, t, eps_L, eps_R, J, modes, n_fock=d, options=TIGHT_SOLVER_OPTIONS), times)

    def acceptor(states):          # P_R(t) from the initial state |L> (logical |0>)
        return np.real(np.asarray(states["0"])[:, 1, 1])

    refs = {d: reference(site_modes, d) for d in dims}
    sums = {d: _pair_summary(refs[d])[0] for d in dims}
    top = sums[dims[-1]]
    per_d = []
    for d in dims:
        entry = {"d": d, **sums[d],
                 "eps_NM_canonical": abs(sums[d]["N_BLP_canonical"] - top["N_BLP_canonical"]),
                 "eps_NM_optimized": abs(sums[d]["N_BLP_optimized"] - top["N_BLP_optimized"]),
                 "max_acceptor_dev_to_dmax": float(np.max(np.abs(
                     acceptor(refs[d]) - acceptor(refs[dims[-1]])))),
                 "max_acceptor_population": float(np.max(acceptor(refs[d])))}
        if d < dims[-1]:
            entry["max_trace_norm_dev_to_next_d"] = _max_trace_norm_dev(refs[d], refs[d + 1])
        per_d.append(entry)

    idx = qc_mod.DIMER_EXCITONIC_INDEX
    circuit = {}
    for dt in trotter_dts:
        n = int(round(t_max / dt))
        res = qc_mod.run_exact([qc_mod.dimer_exact_circuit_excitonic(
            l, n, dt, eps_L, eps_R, J, site_modes) for l in IC_INPUTS])
        blocks = {l: np.asarray([m[np.ix_(idx, idx)]
                                 for m in qc_mod.exact_states_from_labels(r, n)])
                  for l, r in zip(IC_INPUTS, res)}
        stride = int(round(dt_ref / dt))
        on_grid = {l: v[::stride] for l, v in blocks.items()}
        summary, _ = _pair_summary(on_grid)
        circuit[f"{dt:g}"] = {
            **summary,
            "max_acceptor_dev_vs_d2": float(np.max(np.abs(acceptor(on_grid) - acceptor(refs[2])))),
            "max_acceptor_dev_vs_dmax": float(np.max(np.abs(
                acceptor(on_grid) - acceptor(refs[dims[-1]])))),
            "max_acceptor_population": float(np.max(acceptor(on_grid))),
            "acceptor_population": acceptor(on_grid),
            "excitonic_leakage": max(float(abs(np.trace(m) - 1.0))
                                     for v in blocks.values() for m in v)}

    fine = min(samplings)
    n_fine = int(round(t_max / fine))
    t_fine = np.linspace(0.0, n_fine * fine, n_fine + 1)
    time_ladder = {}
    for name, modes, d in (("reference", site_modes, dims[-1]), ("isolated", isolated, 2)):
        states = _ic_states(lambda r0, t: dimer_qutip_excitonic_states(
            r0, t, eps_L, eps_R, J, modes, n_fock=d, options=TIGHT_SOLVER_OPTIONS), t_fine)
        time_ladder[name] = {f"{s:g}": _sampled_summary(states, int(round(s / fine)))
                             for s in samplings}

    iso = reference(isolated, 2)        # modes decoupled: the Fock dimension is irrelevant
    iso_sum, _ = _pair_summary(iso)
    p_iso = acceptor(iso)
    # the same baseline at mesolve's default tolerances: the solver-noise floor
    iso_default, _ = _pair_summary(_ic_states(lambda r0, t: dimer_qutip_excitonic_states(
        r0, t, eps_L, eps_R, J, isolated, n_fock=2), times))
    out = {"per_d": per_d, "dims": dims, "d_max": dims[-1], "circuit": circuit,
           "isolated": {**iso_sum, "max_acceptor_population": float(np.max(p_iso)),
                        "time_of_first_maximum": float(times[int(np.argmax(p_iso[:n_ref // 3]))]),
                        "N_BLP_optimized_default_tolerances": iso_default["N_BLP_optimized"],
                        "N_BLP_canonical_default_tolerances": iso_default["N_BLP_canonical"]},
           "time_ladder": time_ladder, "samplings": list(samplings),
           "acceptor_reference_dmax": acceptor(refs[dims[-1]]),
           "acceptor_isolated": p_iso, "times": times,
           "params": {"eps_L": eps_L, "eps_R": eps_R, "J": J, "site_modes": site_modes},
           "blp_sampling_interval": dt_ref, "t_max": t_max,
           "note": ("Memory is the pair-optimised BLP of the excitonic qubit; the "
                    "isolated dimer is unitary, so its BLP must vanish. References use "
                    "TIGHT_SOLVER_OPTIONS; the *_default_tolerances entries give the "
                    "isolated dimer at mesolve's defaults (the solver-noise floor).")}
    save_result("dimer_convergence_and_memory",
                {"d_max": dims[-1], "trotter_dts": list(trotter_dts),
                 "samplings": list(samplings), "quick": quick,
                 "solver_options": TIGHT_SOLVER_OPTIONS}, out)
    return out


def joint_fock_time_check(family: str, dims, sampling: float, t_max: float = 10.0) -> dict:
    """Fock convergence of the pair-optimised BLP at a fine sampling (B-G3:
    Fock dimension and time resolution together). ``family``: "finite_T"
    (Table 1 points), "spin_boson" (Table 2 points) or "dimer" (Sec. 8)."""
    from .reference import dimer_qutip_excitonic_states
    n = int(round(t_max / sampling))
    times = np.linspace(0.0, n * sampling, n + 1)
    dims = sorted(dims)
    if family == "finite_T":
        params = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]

        def evolver(p, d):
            return lambda r0, t: stageA_qutip_states_finiteT(
                r0, t, **STAGEA_NONMARKOV, n_th=p, n_fock=d, options=TIGHT_SOLVER_OPTIONS)
    elif family == "spin_boson":
        params = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
        proto = [(1.0, 0.4), (1.6, 0.8)]

        def evolver(p, d):
            modes = [(w, p, k) for w, k in proto]
            return lambda r0, t: spin_boson_qutip_states(
                r0, t, 1.0, 0.0, modes, n_fock=d, options=TIGHT_SOLVER_OPTIONS)
    elif family == "dimer":
        params = [0.25]                    # the site coupling g of Sec. 8

        def evolver(p, d):
            site_modes = [(0.8, p, 0.3), (0.8, p, 0.3)]
            return lambda r0, t: dimer_qutip_excitonic_states(
                r0, t, 0.5, -0.5, 1.0, site_modes, n_fock=d, options=TIGHT_SOLVER_OPTIONS)
    else:
        raise ValueError(f"unknown family {family!r}")
    rows = []
    for p in params:
        vals = {d: _pair_summary(_ic_states(evolver(p, d), times))[0]["N_BLP_optimized"]
                for d in dims}
        rows.append({"param": p, "N_BLP_optimized": {str(d): v for d, v in vals.items()},
                     "eps_NM_to_top": {str(d): abs(v - vals[dims[-1]]) for d, v in vals.items()}})
    out = {"family": family, "dims": dims, "sampling": sampling, "t_max": t_max, "rows": rows}
    save_result(f"joint_fock_time_{family}",
                {"family": family, "dims": dims, "sampling": sampling, "t_max": t_max,
                 "solver_options": TIGHT_SOLVER_OPTIONS}, out)
    return out


# ===========================================================================
# Trotter order and its mechanism (T2.1, rule R10)
# ===========================================================================

def _fit_order(dts, errs) -> dict:
    """Least-squares slope of log(err) against log(dt), with its standard error."""
    x, y = np.log(np.asarray(dts, float)), np.log(np.asarray(errs, float))
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    cov = (resid @ resid / dof) * np.linalg.inv(A.T @ A)
    return {"order": float(coef[0]), "stderr": float(np.sqrt(cov[0, 0])), "n_points": len(x)}


def trotter_order_study(dts=(0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625), t_max: float = 12.0,
                        n_th: float = 0.5, g_stageB: float = 0.4, n_fit: int = 5) -> dict:
    """Trotter order of the circuits, and the A1 mechanism checks (rule R10).

    Error at step dt: max over the six inputs and the common times (multiples
    of max(dts)) of the trace distance between circuit and reference; the
    element-wise maximum alongside. Reference: the closed form for Stage A at
    T = 0 from vacuum; otherwise QuTiP on the circuit's own model (n = 2) at
    TIGHT_SOLVER_OPTIONS, solved once on the finest step and subsampled. The
    BLP error is taken on each step's own grid. The order is fitted over the
    n_fit smallest steps (and over all).
    """
    from .reference import dimer_qutip_excitonic_states
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    dts = sorted(dts, reverse=True)
    fine = dts[-1]
    n_fine = int(round(t_max / fine))
    t_fine = np.linspace(0.0, n_fine * fine, n_fine + 1)
    common = int(round(dts[0] / fine))            # fine-grid stride of the common times
    q_th = n_th / (2.0 * n_th + 1.0)              # excited population of the two-level thermal mode
    mode_th = np.diag([1.0 - q_th, q_th])
    mode_up = np.diag([0.0, 1.0])

    def qutip_A(n, mode_state=None):
        return lambda r0, t: stageA_qutip_states_finiteT(
            r0, t, **pars, n_th=n, n_fock=2, options=TIGHT_SOLVER_OPTIONS, mode_state=mode_state)

    def analytic_A(r0, t):
        return analytic_stageA_states(r0, t, **pars)

    def circuit_A(mode_weights=(("0", 1.0),), **kw):
        """Stage A variant circuit; a mixed initial mode is run per basis state
        and mixed with ``mode_weights`` (exact by linearity)."""
        def run(dt, n):
            out = {}
            for l in IC_INPUTS:
                res = qc_mod.run_exact([qc_mod.stageA_variant_circuit(
                    l, n, dt, omega, kappa, mode_init=m, **kw) for m, _ in mode_weights])
                out[l] = sum(w * np.asarray(qc_mod.exact_states_from_labels(r, n))
                             for (_, w), r in zip(mode_weights, res))
            return out
        return run

    thermal = (("0", 1.0 - q_th), ("1", q_th))
    proto = [(1.0, 0.4), (1.6, 0.8)]
    sb_modes = [(w, g_stageB, k) for w, k in proto]
    dimer = dict(eps_L=0.5, eps_R=-0.5, J=1.0, site_modes=[(0.8, 0.25, 0.3), (0.8, 0.25, 0.3)])
    idx = qc_mod.DIMER_EXCITONIC_INDEX

    def run_stageB(dt, n):
        res = qc_mod.run_exact([qc_mod.stageB_exact_circuit(l, n, dt, 1.0, 0.0, sb_modes)
                                for l in IC_INPUTS])
        return {l: np.asarray(qc_mod.exact_states_from_labels(r, n))
                for l, r in zip(IC_INPUTS, res)}

    def run_dimer(dt, n):
        res = qc_mod.run_exact([qc_mod.dimer_exact_circuit_excitonic(l, n, dt, **dimer)
                                for l in IC_INPUTS])
        return {l: np.asarray([m[np.ix_(idx, idx)] for m in qc_mod.exact_states_from_labels(r, n)])
                for l, r in zip(IC_INPUTS, res)}

    variants = [
        ("stageA_T0", "production step, T = 0, vacuum start", 2, "closed form",
         analytic_A, circuit_A()),
        ("A1_i_finite_T", f"finite T (n_th = {n_th}), vacuum start, split GAD", 1, "QuTiP n=2",
         qutip_A(n_th), circuit_A(n_th=n_th)),
        ("A1_ii_reversed", "T = 0, damping before the coherent block", 1, "closed form",
         analytic_A, circuit_A(order="DU")),
        ("A1_iii_excited_mode", "T = 0, mode starts in |1>", 1, "QuTiP n=2",
         qutip_A(0.0, mode_up), circuit_A(mode_weights=(("1", 1.0),))),
        ("A1_iv_thermal_start_exact_gad", f"n_th = {n_th}, thermal start, exact GAD", 2,
         "QuTiP n=2", qutip_A(n_th, mode_th), circuit_A(thermal, n_th=n_th, gad="exact")),
        ("diag_vacuum_start_exact_gad", f"n_th = {n_th}, vacuum start, exact GAD", 1,
         "QuTiP n=2", qutip_A(n_th), circuit_A(n_th=n_th, gad="exact")),
        ("diag_thermal_start_split_gad", f"n_th = {n_th}, thermal start, split GAD", 1,
         "QuTiP n=2", qutip_A(n_th, mode_th), circuit_A(thermal, n_th=n_th)),
        ("stageB", f"spin-boson, g = {g_stageB}, two modes", 1, "QuTiP n=2",
         lambda r0, t: spin_boson_qutip_states(r0, t, 1.0, 0.0, sb_modes, n_fock=2,
                                               options=TIGHT_SOLVER_OPTIONS),
         run_stageB),
        ("dimer", "excitonic dimer (Sec. 8 parameters)", 1, "QuTiP n=2",
         lambda r0, t: dimer_qutip_excitonic_states(r0, t, n_fock=2,
                                                    options=TIGHT_SOLVER_OPTIONS, **dimer),
         run_dimer),
    ]
    rows = []
    for name, desc, predicted, ref_kind, evolver, circuit in variants:
        ref = _six_from_ic(_ic_states(evolver, t_fine))
        per_dt = []
        for dt in dts:
            n = int(round(t_max / dt))
            stride = int(round(dt / fine))
            ic = circuit(dt, n)
            circ = _six_from_ic(ic)
            step_c = int(round(dts[0] / dt))
            tn = el = 0.0
            for l in ref:
                a, b = circ[l][::step_c], ref[l][::common]
                tn = max(tn, max(0.5 * trace_norm(x - y) for x, y in zip(a, b)))
                el = max(el, float(np.max(np.abs(a - b))))
            c_sum, _ = _pair_summary(ic)
            r_sum, _ = _pair_summary({l: ref[l][::stride] for l in IC_INPUTS})
            per_dt.append({"dt": dt, "trace_distance_error": tn, "elementwise_error": el,
                           "blp_opt_circuit": c_sum["N_BLP_optimized"],
                           "blp_opt_reference": r_sum["N_BLP_optimized"],
                           "blp_opt_error": abs(c_sum["N_BLP_optimized"] - r_sum["N_BLP_optimized"]),
                           "blp_pm_error": abs(c_sum["canonical_by_pair"]["+|-"]
                                               - r_sum["canonical_by_pair"]["+|-"])})
        errs = [e["trace_distance_error"] for e in per_dt]
        rows.append({"variant": name, "description": desc, "predicted_order": predicted,
                     "reference": ref_kind, "per_dt": per_dt,
                     "fit_smallest": _fit_order(dts[-n_fit:], errs[-n_fit:]),
                     "fit_all": _fit_order(dts, errs),
                     "fit_elementwise_smallest": _fit_order(
                         dts[-n_fit:], [e["elementwise_error"] for e in per_dt][-n_fit:])})
    # the QuTiP reference against the closed form where both exist (A-M3)
    q = _six_from_ic(_ic_states(qutip_A(0.0), t_fine))
    a = _six_from_ic(_ic_states(analytic_A, t_fine))
    out = {"rows": rows, "dts": dts, "t_max": t_max, "n_th": n_th, "g_stageB": g_stageB,
           "n_fit": n_fit, "common_time_step": dts[0],
           "qutip_vs_closed_form_max_trace_distance": max(
               max(0.5 * trace_norm(x - y) for x, y in zip(q[l], a[l])) for l in q),
           "params": pars}
    save_result("trotter_order_study",
                {"dts": dts, "t_max": t_max, "n_th": n_th, "g_stageB": g_stageB, "n_fit": n_fit,
                 "solver_options": TIGHT_SOLVER_OPTIONS}, out)
    return out


# ===========================================================================
# Phase 8 (Tier 3) -- FMO-like excitonic dimer (generalisation demo)
# ===========================================================================

def phase8_dimer(quick: bool = False) -> dict:
    """Two-chromophore energy-transfer dimer with structured (pseudomode)
    dephasing baths. Demonstrates the pipeline generalises beyond one qubit:
    validate circuit site-populations against QuTiP, and show non-Markovian
    (coherent, revival-bearing) energy transfer.
    """
    from .reference import dimer_qutip_populations
    eps_L, eps_R = 0.5, -0.5     # site-energy detuning (biased dimer)
    J = 1.0                      # excitonic coupling
    dt = 0.1 if quick else 0.05
    t_max = 10.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    # weak, structured (underdamped) baths on each site
    site_modes = [(0.8, 0.25, 0.3), (0.8, 0.25, 0.3)]
    n_fock = 3 if quick else 4

    # QuTiP reference (site populations)
    PL_ref, PR_ref = dimer_qutip_populations(times, eps_L, eps_R, J, site_modes,
                                             n_fock=n_fock)
    # circuit (single-qubit-per-mode -> n_fock=2 truncation)
    qc = qc_mod.dimer_exact_circuit(n_steps, dt, eps_L, eps_R, J, site_modes)
    data = qc_mod.run_exact([qc])[0]
    PL_c, PR_c = qc_mod.dimer_populations_from_data(data, n_steps)

    dev_PR = float(np.max(np.abs(PR_c - PR_ref)))
    dev_PL = float(np.max(np.abs(PL_c - PL_ref)))
    # transfer efficiency (max population reaching the acceptor site R)
    transfer_eff = float(np.max(PR_ref))
    # memory proxy: revivals in acceptor population after its first maximum
    dPR = np.diff(PR_ref)
    first_peak = int(np.argmax(PR_ref))
    revival = float(np.sum(dPR[first_peak:][dPR[first_peak:] > 0]))

    out = {"times": times, "eps_L": eps_L, "eps_R": eps_R, "J": J,
           "site_modes": site_modes, "n_fock_ref": n_fock,
           "P_L_ref": PL_ref, "P_R_ref": PR_ref, "P_L_circuit": PL_c, "P_R_circuit": PR_c,
           "max_dev_PR_circuit_vs_qutip": dev_PR,
           "max_dev_PL_circuit_vs_qutip": dev_PL,
           "transfer_efficiency": transfer_eff,
           "acceptor_population_revival": revival}
    save_result("phase8_dimer", {"dt": dt, "J": J, "n_fock": n_fock}, out)
    return out


# ===========================================================================
# Phase 7 (Tier 3) -- scaling laws and error-bound verification
# ===========================================================================

def _loglog_slope(x, y) -> float:
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


def phase7_scaling(quick: bool = False) -> dict:
    """Quantify the three scaling axes a Q1 referee expects:
      (i)   Trotter error vs dt        -> should confirm first-order O(dt),
      (ii)  truncation error vs Fock d -> decay rate of representation error,
      (iii) circuit resources vs Trotter steps and vs #pseudomodes -> linear.
    """
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)

    # (i) Trotter-error scaling (Stage A, exact-truncation so only dt matters)
    dts = [0.2, 0.1, 0.05, 0.025] if quick else [0.2, 0.1, 0.05, 0.025, 0.0125]
    trotter = []
    for dt in dts:
        n_steps = int(round(6.0 / dt))
        times = np.linspace(0.0, n_steps * dt, n_steps + 1)
        ana = analytic_stageA_states(dm("1"), times, **pars)
        circ = _circuit_states_stageA(times, dt, **pars, labels=("1",))["1"]
        trotter.append({"dt": dt, "max_dev": float(np.max(np.abs(
            np.array(circ) - np.array(ana))))})
    trotter_order = _loglog_slope([r["dt"] for r in trotter],
                                  [r["max_dev"] for r in trotter])

    # (ii) truncation-error decay vs Fock dimension (finite-T, single mode)
    t_max, dt = 8.0, 0.05
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    d_ref = 9
    ref = np.array(stageA_qutip_states_finiteT(dm("1"), times, **pars, n_th=1.0, n_fock=d_ref))
    trunc = []
    for d in range(2, d_ref):
        s = np.array(stageA_qutip_states_finiteT(dm("1"), times, **pars, n_th=1.0, n_fock=d))
        trunc.append({"d": d, "max_dev": float(np.max(np.abs(s - ref)))})
    # geometric decay rate: slope of log(dev) vs d (per-level suppression)
    valid = [(r["d"], r["max_dev"]) for r in trunc if r["max_dev"] > 1e-12]
    trunc_decay_per_level = float(np.polyfit([d for d, _ in valid],
                                             np.log([v for _, v in valid]), 1)[0]) if len(valid) > 1 else float("nan")

    # (iii) resource scaling
    step_counts = [20, 40, 80, 160]
    res_steps = []
    for n in step_counts:
        qc = qc_mod.stageA_exact_circuit("1", n, 0.05, omega, kappa, save_every=n)
        rc = qc_mod.resource_counts(qc)
        res_steps.append({"n_steps": n, "depth": rc["depth"], "cx": rc["cx_count"]})
    depth_per_step = _loglog_slope([r["n_steps"] for r in res_steps],
                                   [r["depth"] for r in res_steps])
    cx_per_step = _loglog_slope([r["n_steps"] for r in res_steps],
                                [r["cx"] for r in res_steps])

    n_modes_list = [1, 2, 3]
    res_modes = []
    for nm in n_modes_list:
        modes = [(1.0 + 0.5 * k, 0.3, 0.4) for k in range(nm)]
        qc = qc_mod.stageB_exact_circuit("1", 40, 0.05, 1.0, 0.0, modes, save_every=40)
        rc = qc_mod.resource_counts(qc)
        res_modes.append({"n_modes": nm, "n_qubits": 2 + nm, "depth": rc["depth"],
                          "cx": rc["cx_count"]})

    out = {"trotter": trotter, "trotter_order_estimate": trotter_order,
           "truncation": trunc, "truncation_log_decay_per_level": trunc_decay_per_level,
           "resources_vs_steps": res_steps, "depth_loglog_slope_vs_steps": depth_per_step,
           "cx_loglog_slope_vs_steps": cx_per_step,
           "resources_vs_modes": res_modes}
    save_result("phase7_scaling", {"quick": quick}, out)
    return out


# ===========================================================================
# Phase 6 (Tier 2) -- FAIR collision comparison at matched accuracy
# ===========================================================================

def phase6_fair_collision(quick: bool = False, seed: int = 1234) -> dict:
    """Replace the crippled 3-qubit collision strawman with a proper
    ancilla-train (chain-mapped bath, linear qubit growth). Sweep chain length
    M; fit each to the SAME Lorentzian population target; evaluate on the
    independent N_BLP observable. Report accuracy vs qubit count against the
    fixed-cost (3-qubit) pseudomode -> the honest resource comparison.
    """
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    dt = 0.15 if quick else 0.1
    t_max = 9.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    pair = ("+", "-")
    M_list = [1, 2, 3, 4] if quick else [1, 2, 3, 4, 5]

    # --- reference (analytic) and pseudomode (fixed 3 qubits)
    G = np.abs(analytic_G(times, **pars))
    n_blp_ref = blp_from_curve(G)
    target_pe = G ** 2
    pm_states = _circuit_states_stageA(times, dt, **pars, labels=pair)
    D_pm = trace_distance_curve(pm_states[pair[0]], pm_states[pair[1]])
    n_blp_pm = blp_from_curve(D_pm)
    err_pm = abs(n_blp_pm - n_blp_ref)

    # pseudomode transpiled resources (fixed cost) for a fair table
    pm_res = qc_mod.resource_counts(qc_mod.stageA_exact_circuit(
        pair[0], n_steps, dt, omega, kappa, save_every=n_steps))

    # --- collision ancilla-train: fit each M (warm-started from previous M to
    #     avoid spurious local minima -- M5), evaluate N_BLP, REAL circuit
    #     resources, and numpy<->circuit agreement.
    rows = []
    warm = None
    for M in M_list:
        fit = col_mod.fit_chain_params(target_pe, n_steps, M, warm_start=warm)
        warm = (fit["theta_s"], fit["theta_hop"], fit["p_loss"])
        cm = {l: col_mod.chain_collision_states_numpy(l, n_steps, M, *warm) for l in pair}
        D_cm = trace_distance_curve(cm[pair[0]], cm[pair[1]])
        n_blp_cm = blp_from_curve(D_cm)
        cm_circ = col_mod.chain_collision_exact_circuit(pair[0], n_steps, M, *warm,
                                                        save_every=n_steps)
        res = qc_mod.resource_counts(cm_circ)
        # numpy<->circuit agreement (final state) as an in-run integrity check
        circ_final = qc_mod.run_exact([cm_circ])[0][f"rho_{n_steps}"]
        np_circ_dev = float(np.max(np.abs(circ_final - cm[pair[0]][-1])))
        rows.append({"M": M, "n_qubits": M + 2, "depth": res["depth"], "cx": res["cx_count"],
                     "pop_rmse": fit["rmse"], "N_BLP": n_blp_cm,
                     "N_BLP_error": abs(n_blp_cm - n_blp_ref),
                     "numpy_vs_circuit_dev": np_circ_dev,
                     "theta_s": fit["theta_s"], "theta_hop": fit["theta_hop"],
                     "p_loss": fit["p_loss"]})

    # --- noise-robustness leg: pseudomode vs best chain (min-error M) at
    #     matched accuracy, shot mode (M5 -- the leg the retracted Phase 4 lacked)
    best_row = min(rows, key=lambda r: r["N_BLP_error"])
    stride = 3 if quick else 2
    step_list = list(range(0, n_steps + 1, stride))
    robustness = []
    for s in ([0.0, 1.0] if quick else [0.0, 0.5, 1.0, 2.0]):
        nmod = build_noise_model(scale=s)
        st_pm = _shot_mode_states(qc_mod.stageA_shot_circuit,
                                  {"dt": dt, "omega": omega, "kappa": kappa},
                                  step_list, pair, nmod, 4096, seed, None)
        st_cm = _shot_mode_states(
            lambda init, n_steps, basis: col_mod.chain_collision_shot_circuit(
                init, n_steps, best_row["M"], best_row["theta_s"],
                best_row["theta_hop"], best_row["p_loss"], basis),
            {}, step_list, pair, nmod, 4096, seed, None)
        robustness.append({"noise_scale": s,
                           "N_BLP_pseudomode": blp_from_curve(
                               trace_distance_curve(st_pm[pair[0]], st_pm[pair[1]])),
                           "N_BLP_chain": blp_from_curve(
                               trace_distance_curve(st_cm[pair[0]], st_cm[pair[1]]))})

    matched = [r for r in rows if r["N_BLP_error"] <= max(err_pm, 0.02)]
    m_req = min((r["M"] for r in matched), default=None)
    out = {"times": times, "pair": pair, "N_BLP_reference": n_blp_ref,
           "pseudomode": {"n_qubits": 3, "depth": pm_res["depth"], "cx": pm_res["cx_count"],
                          "N_BLP": n_blp_pm, "N_BLP_error": err_pm},
           "collision_train": rows,
           "collision_M_to_match_pseudomode": m_req,
           "collision_qubits_to_match": (m_req + 2) if m_req else None,
           "best_chain_M": best_row["M"], "noise_robustness": robustness,
           "dt": dt, "t_max": t_max}
    save_result("phase6_fair_collision", {"dt": dt, "M_list": M_list, **pars}, out, seed=seed)
    return out


# ===========================================================================
# Phase 4 -- pseudomode vs collision model head-to-head
# ===========================================================================

def phase4(quick: bool = False, shots: int = 4096, seed: int = 1234) -> dict:
    pars = STAGEA_NONMARKOV
    omega, kappa = stageA_params(**pars)
    dt = 0.15 if quick else 0.1
    t_max = 9.0
    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    pair = ("+", "-")

    # --- fit collision model to the same reference dynamics
    target_pe = np.abs(analytic_G(times, **pars)) ** 2
    fit = col_mod.fit_collision_params(target_pe, n_steps)

    # --- exact-mode N_BLP fidelity for both embeddings
    G = np.abs(analytic_G(times, **pars))
    n_blp_ref = blp_from_curve(G)

    pm_states = _circuit_states_stageA(times, dt, **pars, labels=pair)
    D_pm = trace_distance_curve(pm_states[pair[0]], pm_states[pair[1]])

    cm_states = {l: col_mod.collision_states_numpy(l, n_steps, fit["theta_s"], fit["theta_m"])
                 for l in pair}
    D_cm = trace_distance_curve(cm_states[pair[0]], cm_states[pair[1]])

    # --- resource comparison (transpiled, full-evolution circuits)
    pm_circ = qc_mod.stageA_exact_circuit(pair[0], n_steps, dt, omega, kappa,
                                          save_every=n_steps)
    cm_circ = col_mod.collision_exact_circuit(pair[0], n_steps, fit["theta_s"],
                                              fit["theta_m"], save_every=n_steps)
    resources = {"pseudomode": qc_mod.resource_counts(pm_circ),
                 "collision": qc_mod.resource_counts(cm_circ)}

    # --- noise robustness (shot mode) for both embeddings
    scales = [0.0, 1.0] if quick else [0.0, 0.5, 1.0, 2.0]
    stride = 3 if quick else 2
    step_list = list(range(0, n_steps + 1, stride))
    t_shot = np.array(step_list) * dt
    robustness = []
    for s in scales:
        nm = build_noise_model(scale=s)
        M = None
        if nm is not None:
            cal = qc_mod.run_counts(readout_calibration_circuits(), shots=shots,
                                    noise_model=nm, seed=seed)
            M = confusion_matrix(cal)
        st_pm = _shot_mode_states(qc_mod.stageA_shot_circuit,
                                  {"dt": dt, "omega": omega, "kappa": kappa},
                                  step_list, pair, nm, shots, seed, M)
        st_cm = _shot_mode_states(col_mod.collision_shot_circuit,
                                  {"theta_s": fit["theta_s"], "theta_m": fit["theta_m"]},
                                  step_list, pair, nm, shots, seed, M)
        robustness.append({
            "noise_scale": s,
            "N_BLP_pseudomode": blp_from_curve(
                trace_distance_curve(st_pm[pair[0]], st_pm[pair[1]])),
            "N_BLP_collision": blp_from_curve(
                trace_distance_curve(st_cm[pair[0]], st_cm[pair[1]])),
        })

    out = {"times": times, "pair": pair, "collision_fit": fit,
           "N_BLP_reference": n_blp_ref,
           "N_BLP_pseudomode_exact": blp_from_curve(D_pm),
           "N_BLP_collision_exact": blp_from_curve(D_cm),
           "D_pseudomode": D_pm, "D_collision": D_cm, "abs_G": G,
           "resources": resources,
           "noise_robustness": robustness,
           "shot_times": t_shot}
    save_result("phase4_embedding_comparison", {"dt": dt, "shots": shots,
                                                "scales": scales, **pars}, out, seed=seed)
    return out

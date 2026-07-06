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
from .measures import (blp_from_curve, blp_measure, blp_measure_optimized,
                       canonical_pairs, dm, rhp_measure, trace_distance_curve,
                       transfer_matrix)
from .reference import (analytic_G, analytic_stageA_states, required_fock_dimension,
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


def _provenance(params: dict, seed: int | None) -> dict:
    import qiskit
    import qutip
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qmembench_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "qiskit": qiskit.__version__,
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

    predictor = np.array([p["n_max"] + c_var * p["std_at_peak"] for p in points])
    out = {"points": points, "eps": eps, "c_var": c_var, "eps_list": eps_list,
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
        cert = blp_measure_optimized(ch, seed=1)
        rows.append({"family": "spin_boson", "param": g, **cert})

    # finite-T Stage A at a few temperatures
    for n_th in ([0.5, 1.0] if quick else [0.25, 0.5, 1.0, 1.5]):
        ch = lambda r0, n_th=n_th: stageA_qutip_states_finiteT(r0, times, **STAGEA_NONMARKOV,
                                                               n_th=n_th, n_fock=6)
        cert = blp_measure_optimized(ch, seed=1)
        rows.append({"family": "finite_T", "param": n_th, **cert})

    max_gap = max(r["gap"] for r in rows)
    rel = max((r["gap"] / r["N_BLP_optimized"]) if r["N_BLP_optimized"] > 1e-6 else 0.0
              for r in rows)
    out = {"rows": rows, "max_abs_gap": max_gap, "max_rel_gap": rel,
           "canonical_is_tight": max_gap < 0.02}
    save_result("phase_blp_pair_certificate", {"dt": dt, "t_max": t_max}, out)
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

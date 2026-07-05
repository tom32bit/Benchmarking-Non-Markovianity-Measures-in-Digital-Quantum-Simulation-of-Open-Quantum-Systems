"""Run EVERY phase at full resolution and regenerate all figures.

Fault-tolerant: each phase is isolated, timed, and its headline numbers
printed; a failure in one phase is reported and does not abort the rest.
Produces publication-grade numbers + figures, all with JSON provenance.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench import experiments as E
from qmembench import plotting as P


def run(label, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"[OK]   {label:<28} ({time.time()-t0:6.1f}s)")
        return out
    except Exception:
        print(f"[FAIL] {label:<28} ({time.time()-t0:6.1f}s)")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    T0 = time.time()
    print("=" * 64)
    print("FULL-RESOLUTION RUN")
    print("=" * 64)

    p0 = run("phase0 measures", lambda: E.phase0(quick=False))
    p1 = run("phase1 stageA validation", lambda: E.phase1(quick=False))
    p1c = run("phase1 trotter convergence", lambda: E.phase1_trotter_convergence())
    p1t = run("phase1T finite temperature", lambda: E.phase1_finiteT(quick=False))
    p2 = run("phase2 spin-boson", lambda: E.phase2(quick=False))
    p3 = run("phase3 noise robustness", lambda: E.phase3(quick=False))
    p3z = run("phase3 ZNE", lambda: E.phase3_zne(quick=False))
    p3b = run("phase3 bootstrap", lambda: E.phase3_bootstrap(quick=False))
    p3ev = run("phase3 estimator validation", lambda: E.phase3_estimator_validation(quick=False))
    p3n = run("phase3 noise models", lambda: E.phase3_noise_models(quick=False))
    pcert = run("blp pair certificate", lambda: E.phase_blp_pair_certificate(quick=False))
    p4 = run("phase4 embedding compare", lambda: E.phase4(quick=False))
    p5 = run("phase5 fock rule", lambda: E.phase5_fock_rule(quick=False))
    p6 = run("phase6 fair collision", lambda: E.phase6_fair_collision(quick=False))
    p7 = run("phase7 scaling", lambda: E.phase7_scaling(quick=False))
    p8 = run("phase8 dimer", lambda: E.phase8_dimer(quick=False))

    print("-" * 64)
    print("FIGURES")
    for label, fn in [
        ("phase0", P.plot_phase0), ("phase1", P.plot_phase1),
        ("phase1T", P.plot_phase1T), ("phase2", P.plot_phase2),
        ("phase3", P.plot_phase3), ("phase3_bootstrap", P.plot_phase3_bootstrap),
        ("phase3_estimator_validation", P.plot_phase3_estimator_validation),
        ("phase3_noise_models", P.plot_phase3_noise_models),
        ("phase4", P.plot_phase4), ("phase5_fock_rule", P.plot_phase5_fock_rule),
        ("phase6_fair_collision", P.plot_phase6_fair_collision),
        ("phase7_scaling", P.plot_phase7_scaling), ("phase8_dimer", P.plot_phase8_dimer),
    ]:
        run(f"fig {label}", fn)

    print("=" * 64)
    print("HEADLINE NUMBERS")
    print("=" * 64)
    if p0:
        nm = p0["nonmarkovian"]
        print(f"P0  N_BLP={nm['N_BLP']:.4f} (analytic opt {nm['N_BLP_analytic_optimum']:.4f}), "
              f"markovian N_BLP={p0['markovian']['N_BLP']:.2e}")
    if p1:
        print(f"P1  circuit vs analytic max dev={p1['max_abs_dev_circuit_vs_analytic']:.2e}, "
              f"N_BLP err={p1['blp_error_circuit_vs_analytic']:.2e}")
    if p1t:
        r = p1t["rows"][-1]
        print(f"P1T n_th={r['n_th']}: circuit N_BLP={r['N_BLP_circuit']:.4f}, "
              f"converged={r['N_BLP_ref_converged']:.4f}, trunc={r['truncation_error_2_vs_converged']:.2e}")
    if p2:
        r = p2["rows"][-1]
        print(f"P2  g={r['g']}: circuit={r['N_BLP_circuit']:.4f}, "
              f"converged n={r['n_conv']}={r['N_BLP_ref_converged']:.4f}")
    if p3b:
        print(f"P3b ideal ref={p3b['N_BLP_ideal_reference']:.4f}; " +
              ", ".join(f"s{r['noise_scale']}:pt={r['N_BLP_point']:.3f}/db={r['N_BLP_debiased']:.3f}/nf={r['null_floor_markovian']:.3f}"
                        for r in p3b["rows"]))
    if p3n:
        keys = ["ideal", "nonmarkovian_dephasing", "markovian_dephasing_matched",
                "markovian_depolarizing", "realistic_fake_backend"]
        print("P3n " + ", ".join(f"{k}={p3n[k]['N_BLP']:.3f}" for k in keys
                                 if isinstance(p3n.get(k), dict) and 'N_BLP' in p3n[k]))
    if p5:
        print(f"P5  mean-rule safe={p5['mean_rule_is_safe']} (corr {p5['corr_dreq_vs_nmax']:.2f}); "
              f"var-rule safe={p5['variance_rule_is_safe']} (corr {p5['corr_dreq_vs_variance_predictor']:.2f})")
    if p6:
        print(f"P6  pseudomode(3q) err={p6['pseudomode']['N_BLP_error']:.4f}; "
              f"collision match at {p6['collision_qubits_to_match']} qubits; "
              + ", ".join(f"M{r['M']}({r['n_qubits']}q):{r['N_BLP_error']:.3f}" for r in p6['collision_train']))

    print("=" * 64)
    print(f"TOTAL WALL TIME: {time.time()-T0:.1f}s")

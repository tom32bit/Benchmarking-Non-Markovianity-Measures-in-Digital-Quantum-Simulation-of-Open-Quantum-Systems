"""Full-resolution re-runs for the peer-review revisions:
  - phase1T / phase2 now carry RHP columns (M4a)
  - estimator validation at 50 seeds with Wilson CI (M1), plus a coarse grid (M2b)
  - conflation dt-halving stability check (M3)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmembench import experiments as E
from qmembench import plotting as P


def run(label, fn):
    t0 = time.time()
    out = fn()
    print(f"[OK] {label:<34} ({time.time()-t0:6.1f}s)")
    return out


if __name__ == "__main__":
    T0 = time.time()
    p1t = run("phase1T finite-T + RHP", lambda: E.phase1_finiteT(quick=False))
    p2 = run("phase2 spin-boson + RHP", lambda: E.phase2(quick=False))
    ev = run("estimator validation (50 seeds)", lambda: E.phase3_estimator_validation(quick=False))
    evc = run("estimator validation (coarse grid)",
              lambda: E.phase3_estimator_validation(quick=False, dt=0.3, seeds=range(1, 21),
                                                    save_name="phase3_estimator_validation_coarse"))
    cdt = run("conflation dt-halving check", lambda: E.phase3_conflation_dt_check(quick=False))
    for lbl, fn in [("phase1T", P.plot_phase1T), ("phase2", P.plot_phase2)]:
        run(f"fig {lbl}", fn)

    print("=" * 60)
    print("RHP sign-flip (finite T):")
    for r in p1t["rows"]:
        print(f"  n_th={r['n_th']}: BLP {r['N_BLP_circuit']:.4f}/{r['N_BLP_ref_converged']:.4f}  "
              f"RHP {r['N_RHP_circuit']:.2f}/{r['N_RHP_ref_converged']:.2f} "
              f"(ill {r['rhp_ill_frac_circuit']:.2f}/{r['rhp_ill_frac_converged']:.2f})")
    print("RHP sign-flip (strong coupling):")
    for r in p2["rows"]:
        print(f"  g={r['g']}: BLP {r['N_BLP_circuit']:.4f}/{r['N_BLP_ref_converged']:.4f}  "
              f"RHP {r['N_RHP_circuit']:.2f}/{r['N_RHP_ref_converged']:.2f}")
    print("Estimator coverage (50 seeds) with Wilson CI:")
    for r in ev["rows"]:
        lo, hi = r["ci_coverage_wilson95"]
        print(f"  s={r['noise_scale']} cov={r['ci_coverage_z1']:.2f} Wilson95=[{lo:.2f},{hi:.2f}] "
              f"raw_bias={r.get('raw_bias',0):+.4f} deb_bias={r.get('debiased_z1.0_bias',0):+.4f}")
    print("Estimator coarse grid (dt=0.3) debiased bias:")
    for r in evc["rows"]:
        print(f"  s={r['noise_scale']} deb_bias={r.get('debiased_z1.0_bias',0):+.4f} "
              f"raw_bias={r.get('raw_bias',0):+.4f}")
    print("Conflation dt-halving:")
    for r in cdt["rows"]:
        print(f"  dt={r['dt']} sig={r['sigma_tot']} gap={r['gap']:+.4f} sep={r['cis_separated']}")
    print("  stability:", {k: round(v['abs_change'], 4) for k, v in cdt["grid_stability"].items()})
    print(f"TOTAL {time.time()-T0:.0f}s")

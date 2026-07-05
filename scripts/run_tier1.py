"""Run the Tier-1 (MSc->Q1) extensions: finite temperature, statistical rigor,
and realistic + non-Markovian noise models."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench.experiments import (phase1_finiteT, phase3_bootstrap,
                                   phase3_noise_models)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=== Tier 1(a): finite temperature ===")
    ft = phase1_finiteT(quick=args.quick)
    print(f"  {'n_th':>5} {'circuit':>8} {'ref n=2':>8} {'ref conv':>8} "
          f"{'dev c/ref2':>11} {'trunc':>9}")
    for r in ft["rows"]:
        print(f"  {r['n_th']:>5} {r['N_BLP_circuit']:>8.4f} {r['N_BLP_ref_nfock2']:>8.4f} "
              f"{r['N_BLP_ref_converged']:>8.4f} {r['max_dev_circuit_vs_ref2']:>11.2e} "
              f"{r['truncation_error_2_vs_converged']:>9.2e}")

    print("\n=== Tier 1(c): statistical rigor (bootstrap + debias + null floor) ===")
    bs = phase3_bootstrap(quick=args.quick)
    print(f"  ideal ref N_BLP = {bs['N_BLP_ideal_reference']:.4f}")
    for r in bs["rows"]:
        print(f"  scale={r['noise_scale']:>4} point={r['N_BLP_point']:.4f} "
              f"debiased={r['N_BLP_debiased']:.4f} "
              f"CI=[{r['N_BLP_ci_low']:.3f},{r['N_BLP_ci_high']:.3f}] "
              f"null={r['null_floor_markovian']:.4f}")

    print("\n=== Tier 1(b): noise models (Markovian vs realistic vs non-Markovian) ===")
    nm = phase3_noise_models(quick=args.quick)
    print(f"  ideal reference       N_BLP = {nm['N_BLP_ideal_reference']:.4f}")
    for key in ("ideal", "markovian_depolarizing", "realistic_fake_backend",
                "nonmarkovian_dephasing", "markovian_dephasing_matched"):
        if key in nm and isinstance(nm[key], dict) and "N_BLP" in nm[key]:
            print(f"  {key:<28} N_BLP = {nm[key]['N_BLP']:.4f}")

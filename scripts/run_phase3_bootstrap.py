"""Phase 3 statistical rigor: bootstrap CIs, debiased N_BLP, Markovian null floor."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench.experiments import phase3_bootstrap

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--nboot", type=int, default=400)
    args = ap.parse_args()
    out = phase3_bootstrap(quick=args.quick, shots=args.shots, n_boot=args.nboot)
    print(f"ideal reference N_BLP = {out['N_BLP_ideal_reference']:.4f}  "
          f"(shots={out['shots']}, n_boot={out['n_boot']})")
    print(f"{'scale':>6} {'point':>7} {'debiased':>9} {'95% CI':>18} {'null floor':>11}")
    for r in out["rows"]:
        ci = f"[{r['N_BLP_ci_low']:.3f},{r['N_BLP_ci_high']:.3f}]"
        print(f"{r['noise_scale']:>6} {r['N_BLP_point']:>7.4f} {r['N_BLP_debiased']:>9.4f} "
              f"{ci:>18} {r['null_floor_markovian']:>11.4f}")

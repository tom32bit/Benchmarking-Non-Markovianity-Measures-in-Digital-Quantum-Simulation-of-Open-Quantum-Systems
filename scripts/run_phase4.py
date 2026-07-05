"""Phase 4: pseudomode vs collision-model head-to-head comparison."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench.experiments import phase4
from qmembench.plotting import plot_phase4

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shots", type=int, default=4096)
    args = ap.parse_args()
    out = phase4(quick=args.quick, shots=args.shots)
    fit = out["collision_fit"]
    print(f"collision fit: theta_s={fit['theta_s']:.4f} theta_m={fit['theta_m']:.4f} "
          f"rmse={fit['rmse']:.4f} converged={fit['converged']}")
    print(f"N_BLP reference        = {out['N_BLP_reference']:.4f}")
    print(f"N_BLP pseudomode exact = {out['N_BLP_pseudomode_exact']:.4f}")
    print(f"N_BLP collision  exact = {out['N_BLP_collision_exact']:.4f}")
    print("resources:", out["resources"])
    for r in out["noise_robustness"]:
        print(f"scale={r['noise_scale']:<5} N_BLP pm={r['N_BLP_pseudomode']:.4f} "
              f"cm={r['N_BLP_collision']:.4f}")
    print(f"Figure: {plot_phase4()}")

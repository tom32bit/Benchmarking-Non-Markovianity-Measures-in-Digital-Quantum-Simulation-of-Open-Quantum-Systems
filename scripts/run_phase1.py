"""Phase 1: three-tier Stage A validation (analytic vs QuTiP vs circuit)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench.experiments import phase1, phase1_trotter_convergence
from qmembench.plotting import plot_phase1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--convergence", action="store_true",
                    help="also run the Trotter-step convergence study")
    args = ap.parse_args()
    out = phase1(quick=args.quick)
    fig = plot_phase1()
    print(f"max|circuit - analytic| = {out['max_abs_dev_circuit_vs_analytic']:.3e}")
    print(f"max|qutip   - analytic| = {out['max_abs_dev_qutip_vs_analytic']:.3e}")
    for tier in ("analytic", "qutip", "circuit"):
        print(f"  {tier:9s}: N_BLP = {out[tier]['N_BLP']:.4f} (pair {out[tier]['best_pair']})")
    print(f"|N_BLP circuit - analytic| = {out['blp_error_circuit_vs_analytic']:.3e}")
    print(f"Figure: {fig}")
    if args.convergence:
        conv = phase1_trotter_convergence()
        for row in conv["rows"]:
            print(f"  dt={row['dt']:<6} max_abs_dev={row['max_abs_dev']:.3e}")

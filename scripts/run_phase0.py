"""Phase 0: validate the BLP/RHP measures on the analytic Stage A solution."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench.experiments import phase0
from qmembench.plotting import plot_phase0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = phase0(quick=args.quick)
    fig = plot_phase0()
    nm, mk = out["nonmarkovian"], out["markovian"]
    print(f"Non-Markovian: N_BLP={nm['N_BLP']:.4f} (best pair {nm['best_pair']}), "
          f"analytic optimum={nm['N_BLP_analytic_optimum']:.4f}, N_RHP={nm['N_RHP']:.4f}")
    print(f"Markovian:     N_BLP={mk['N_BLP']:.6f} (expected ~0), N_RHP={mk['N_RHP']:.6f}")
    print(f"Figure: {fig}")

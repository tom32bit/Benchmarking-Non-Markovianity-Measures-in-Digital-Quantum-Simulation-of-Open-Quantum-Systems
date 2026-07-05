"""Phase 2: spin-boson (Stage B) -- memory vs coupling; truncation study."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench.experiments import phase2
from qmembench.plotting import plot_phase2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = phase2(quick=args.quick)
    for r in out["rows"]:
        print(f"g={r['g']:<5} N_BLP: circuit={r['N_BLP_circuit']:.4f} "
              f"ref(n=2)={r['N_BLP_ref_nfock2']:.4f} ref(n=4)={r['N_BLP_ref_nfock4']:.4f} "
              f"| dev={r['max_dev_circuit_vs_ref2']:.2e} trunc={r['truncation_error_ref2_vs_ref4']:.2e}")
    print(f"Figure: {plot_phase2()}")

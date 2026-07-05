"""Phase 3: noise robustness of the memory signature + mitigation recovery."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench.experiments import phase3, phase3_zne
from qmembench.plotting import plot_phase3

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--zne", action="store_true", help="also run ZNE recovery")
    args = ap.parse_args()
    out = phase3(quick=args.quick, shots=args.shots)
    print(f"ideal reference N_BLP = {out['N_BLP_ideal_reference']:.4f}")
    for r in out["rows"]:
        print(f"scale={r['noise_scale']:<5} {r['pipeline']:<18} N_BLP={r['N_BLP']:.4f}")
    print(f"Figure: {plot_phase3()}")
    if args.zne:
        z = phase3_zne(quick=args.quick, shots=args.shots)
        print(f"ZNE (noise scale {z['noise_scale']}): N_BLP = {z['N_BLP_zne']:.4f} "
              f"(ideal {z['N_BLP_ideal_reference']:.4f})")

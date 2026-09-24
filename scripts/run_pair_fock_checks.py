"""Pair-optimised BLP and Fock-dimension convergence checks.

Runs, at the Table 1 and Table 2 points, the canonical and pair-optimised BLP
for the circuit and for the reference at every Fock dimension up to d_max, and
the same checks at selected phase-diagram points and for the noisy channels of
the shot-statistics study. Every call writes a JSON record under results/.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench import experiments as E


def run(label, fn):
    t0 = time.time()
    out = fn()
    print(f"[OK] {label:<44} ({time.time() - t0:7.1f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["finite_T", "spin_boson", "phase_diagram", "noisy"])
    ap.add_argument("--d-max-finite-T", type=int, default=12)
    ap.add_argument("--d-max-spin-boson", type=int, default=10)
    args = ap.parse_args()
    jobs = {
        "finite_T": lambda: E.tables_pair_and_fock_convergence(
            "finite_T", d_max=args.d_max_finite_T, quick=args.quick),
        "spin_boson": lambda: E.tables_pair_and_fock_convergence(
            "spin_boson", d_max=args.d_max_spin_boson, quick=args.quick),
        "phase_diagram": lambda: E.phase_diagram_pair_and_fock_check(
            d_max=args.d_max_spin_boson),
        "noisy": lambda: E.noisy_pair_check(),
    }
    for name, fn in jobs.items():
        if args.only in (None, name):
            run(name, fn)

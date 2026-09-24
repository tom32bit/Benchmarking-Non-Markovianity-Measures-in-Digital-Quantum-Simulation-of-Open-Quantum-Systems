"""Convergence of the reported BLP values in time resolution.

Stage A against the closed-form continuum value; the Table 1 and Table 2
points (reference subsampled from one fine solve, circuit rerun at each
Trotter step); selected phase-diagram points; and the grid dependence of the
exact noisy values behind Table 3. Every call writes a JSON record under
results/.
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
    print(f"[OK] {label:<30} ({time.time() - t0:7.1f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["stageA", "finite_T", "spin_boson", "phase_diagram",
                                       "noisy"])
    args = ap.parse_args()
    jobs = {
        "stageA": E.stageA_blp_time_convergence,
        "finite_T": lambda: E.tables_blp_time_convergence("finite_T", d_ref=12, quick=args.quick),
        "spin_boson": lambda: E.tables_blp_time_convergence("spin_boson", d_ref=10,
                                                            quick=args.quick),
        "phase_diagram": lambda: E.phase_diagram_blp_time_convergence(d_ref=10),
        "noisy": E.noisy_blp_grid_dependence,
    }
    for name, fn in jobs.items():
        if args.only in (None, name):
            run(name, fn)

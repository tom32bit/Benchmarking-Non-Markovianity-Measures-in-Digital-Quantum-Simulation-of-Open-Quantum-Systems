"""Recompute the truncation phase diagram (Fig. 4) with pair-optimised BLP.

Writes results/phase9_truncation_phase_diagram_optimized.json. The points are
independent, so --workers spreads them over a process pool.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench import experiments as E

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--d-ref", type=int, default=10)
    ap.add_argument("--d-check", type=int, default=9)
    ap.add_argument("--sampling", type=float, default=0.1)
    args = ap.parse_args()
    t0 = time.time()
    out = E.phase9_truncation_phase_diagram_optimized(
        quick=args.quick, d_ref=args.d_ref, d_check=args.d_check, n_workers=args.workers,
        sampling=args.sampling)
    print(f"[OK] phase diagram {len(out['n_th'])} x {len(out['g'])} "
          f"(d_ref {out['d_ref']}, check {out['d_check']}) in {time.time() - t0:.0f}s", flush=True)

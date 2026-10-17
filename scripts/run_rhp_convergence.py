"""RHP under changes of time step, pseudo-inverse cutoff and grid alignment,
and the Stage A divergence check.

Writes results/rhp_convergence_{finite_T,spin_boson}.json and
results/stageA_rhp_divergence.json.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench import experiments as E

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["stageA", "finite_T", "spin_boson"])
    args = ap.parse_args()
    jobs = {"stageA": E.stageA_rhp_divergence,
            "finite_T": lambda: E.rhp_convergence_study("finite_T", d_ref=12),
            "spin_boson": lambda: E.rhp_convergence_study("spin_boson", d_ref=10)}
    for name, fn in jobs.items():
        if args.only in (None, name):
            t0 = time.time()
            fn()
            print(f"[OK] {name:<12} ({time.time() - t0:7.1f}s)", flush=True)

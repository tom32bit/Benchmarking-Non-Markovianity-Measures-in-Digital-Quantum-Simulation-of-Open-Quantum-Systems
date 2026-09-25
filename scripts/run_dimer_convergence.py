"""The dimer of Sec. 8 on its excitonic qubit: reference convergence in Fock
dimension, circuit convergence in Trotter step, and the isolated-dimer
baseline. Writes results/dimer_convergence_and_memory.json.
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
    ap.add_argument("--d-max", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    out = E.dimer_convergence_and_memory(d_max=args.d_max, quick=args.quick)
    print(f"[OK] dimer convergence d = 2..{out['d_max']} in {time.time() - t0:.0f}s", flush=True)

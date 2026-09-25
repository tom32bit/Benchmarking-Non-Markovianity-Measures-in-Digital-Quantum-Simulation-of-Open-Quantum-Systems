"""Fock convergence of the BLP at the finest sampling of the time ladders.

Writes results/joint_fock_time_{family}.json for the chosen family.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench import experiments as E

SETTINGS = {"finite_T": ((10, 11, 12), 0.00078125),
            "spin_boson": ((8, 9, 10), 0.003125),
            "dimer": ((6, 7, 8), 0.00078125)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=sorted(SETTINGS))
    args = ap.parse_args()
    dims, sampling = SETTINGS[args.family]
    t0 = time.time()
    E.joint_fock_time_check(args.family, dims, sampling)
    print(f"[OK] joint Fock/time check {args.family} in {time.time() - t0:.0f}s", flush=True)

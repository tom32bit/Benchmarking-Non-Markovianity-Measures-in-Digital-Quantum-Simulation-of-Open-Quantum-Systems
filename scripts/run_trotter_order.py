"""Trotter order of the circuits and the A1 mechanism checks (rule R10).

Writes results/trotter_order_study.json.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qmembench import experiments as E

if __name__ == "__main__":
    t0 = time.time()
    out = E.trotter_order_study()
    print(f"[OK] Trotter order study ({len(out['rows'])} variants) in {time.time() - t0:.0f}s",
          flush=True)

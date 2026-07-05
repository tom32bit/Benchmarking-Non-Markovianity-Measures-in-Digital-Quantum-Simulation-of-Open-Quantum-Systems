"""qmembench — Signatures of Memory Under Noise.

Benchmarking non-Markovianity measures (BLP / RHP) in digital quantum
simulation of open quantum systems, comparing pseudomode and collision-model
embeddings under realistic device-noise models.

Master's thesis codebase. Simulator-only; no quantum-advantage claims.
Every quantum-circuit result is validated against an exact classical
reference (analytic solution and/or QuTiP `mesolve` on the enlarged
system + pseudomode space) before any physics conclusion is drawn.
"""

__version__ = "0.1.0"

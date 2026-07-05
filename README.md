# Benchmarking Non-Markovianity Measures in Digital Quantum Simulation of Open Quantum Systems

A validated, simulator-based framework for quantifying non-Markovian memory in
digital quantum simulations of open quantum systems, and for measuring how that
quantification fails under three unavoidable pressures: state-space truncation,
finite-shot statistics, and device noise.

This repository accompanies the paper of the same name. Every reported number is
regenerated from a JSON record under `results/`, and every physics claim is
checked against an exact reference (a closed-form solution or a QuTiP master
equation solve). The work runs entirely on classical simulators. It makes **no
quantum-advantage claim**; the contribution is methodological.

## What the framework shows

- **A measure-first validation chain.** The Breuer-Laine-Piilo (BLP) and
  Rivas-Huelga-Plenio (RHP) measures are computed from a pseudomode circuit and
  validated against analytic and master-equation references (agreement to
  1e-4 or better).
- **Truncation misrepresents memory, with a regime-dependent sign.** A
  single-qubit ("2-level") mode encoding inflates the BLP measure by about a
  factor of 2.6 at strong coupling, yet destroys it at moderate temperature
  (near zero versus a converged value of roughly 0.19 at thermal occupation
  n_th = 1).
- **A cross-validated resource rule.** The required Fock dimension collapses
  onto the occupied number extent, `d = ceil(<n> + c*sigma_n) + 1`, with
  correlation 0.94. The constant must be calibrated on super-Poissonian
  (thermal) statistics; a coupling-calibrated constant does not transfer.
- **Finite-shot bias, measured against ground truth.** The raw shot estimate of
  the BLP measure can nearly double the true value under realistic noise. A
  significance-thresholded estimator is validated near-unbiased in that regime,
  and a Markovian null-floor control ports the diagnostic to hardware.
- **Correlated device noise is conflated with signal.** At matched strength,
  temporally correlated (quasi-static) dephasing preserves the memory signature
  significantly more than memoryless dephasing, with separated confidence
  intervals at every tested strength.
- **A resource-fair embedding comparison.** A pseudomode and an ancilla-train
  collision model reach parity at equal qubit cost for the single-Lorentzian
  bath, in accuracy, transpiled resources, and noise robustness.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, NumPy 1.26, SciPy 1.15, QuTiP 5.3, Qiskit 0.46, and
Qiskit Aer 0.13 on Windows.

## Reproducing the results

Run the validation gate first. It enforces the analytic - QuTiP - circuit
correctness chain and the numerical-equivalence checks:

```bash
python -m pytest tests -q
```

Then regenerate every result and figure:

```bash
python scripts/run_all_full.py          # all phases at full resolution
```

Individual phases and quick (coarse-grid) variants:

```bash
python scripts/run_phase0.py            # measures validated on the analytic solution
python scripts/run_phase1.py --convergence
python scripts/run_phase2.py            # spin-boson: memory vs coupling
python scripts/run_phase3.py --zne      # noise robustness and mitigation
python scripts/run_phase4.py            # (superseded strawman; kept for the record)
python scripts/run_tier1.py             # finite-T, statistics, noise models
python scripts/run_phase3_bootstrap.py  # bootstrap CIs, debiasing, null floor
```

Outputs land in `results/`: each figure (`.png`) is generated from a JSON
payload that records parameters, package versions, random seed, and timestamp.

## Repository layout

```text
qmembench/
  measures.py     BLP and RHP measures, pair optimisation, Bloch utilities
  reference.py    analytic solution and QuTiP references (Stage A, finite-T, Stage B, dimer)
  circuits.py     pseudomode-embedding circuits (Trotter + Stinespring dilation)
  collision.py    3-qubit and ancilla-train collision models (numpy + Qiskit)
  noise.py        device-noise models (scaled synthetic + calibrated fake-backend)
  mitigation.py   readout inversion and zero-noise extrapolation
  tomography.py   shot-based single-qubit state reconstruction
  statistics.py   bootstrap CIs, significance-debiased BLP, Markovian null floor
  experiments.py  phase orchestrators, each saving a provenance JSON
  plotting.py     figures regenerated from saved JSON only
scripts/          one runner per phase, plus run_all_full.py
tests/            the validation gates (run these first)
results/          JSON records and figures (the data of record)
```

## Physics conventions

Basis (|0>, |1>) with |1> the excited state. Stage A is a qubit coupled to a
Lorentzian bath (a damped Jaynes-Cummings model) with parameters gamma0 and
lambda; the exact amplitude G(t) follows Breuer and Petruccione, section 10.1.
The pseudomode mapping is Omega = sqrt(gamma0*lambda/2), kappa = 2*lambda, with
the non-Markovian regime at gamma0 > lambda/2. Because the interaction conserves
excitation number and all initial states carry at most one excitation, a
two-level pseudomode is exact at zero temperature; at finite temperature it is
not, and the required dimension is studied explicitly.

## Data and reproducibility

All quantitative results and figures are derived from `results/*.json`. Random
seeds are fixed and recorded. The BLP measure is a discrete sum and therefore
depends on the time grid, so every reported value is tied to its grid and
comparisons are made only within a fixed grid.

## Citation

If you use this code, please cite the accompanying paper. A `CITATION.cff` file
is provided.

## License

MIT License. See `LICENSE`.

## Author

S.M. Yousuf Iqbal Tomal, Department of Computer Science and Engineering, BRAC
University, Dhaka, Bangladesh.
ORCiD: [0009-0000-3391-3824](https://orcid.org/0009-0000-3391-3824).

# Benchmarking Non-Markovianity Measures in Digital Quantum Simulation of Open Quantum Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23449846.svg)](https://doi.org/10.5281/zenodo.23449846)

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

These are the results of the revised paper (release `v0.2.0`).

- **A measure-first validation chain.** The pair-optimized Breuer-Laine-Piilo
  (BLP) measure of a pseudomode circuit agrees with the closed-form solution to
  7.5e-4 and the master-equation solve to 2e-12 on the same grid. The circuit is
  second order in the Trotter step for Stage A (2.0001 +- 0.0001 over seven steps)
  and first order elsewhere. The Rivas-Huelga-Plenio (RHP) measure, computed from
  reconstructed maps, does not converge on these channels and is used only as a
  diagnostic.
- **Truncation misrepresents memory, with a regime-dependent sign.** A two-level
  mode encoding overstates the BLP measure by a factor of 2.3 at strong coupling
  and loses most of it at moderate temperature (0.0001 against 0.162 at
  n_th = 1). Both signs occupy large regions of the coupling-temperature plane.
- **A variance-aware Fock rule.** `d = ceil(max(<n_k> + c*sigma_k)) + 1`, with c
  calibrated per accuracy target at trajectory and measure level and tested out
  of sample. At equal mean occupation, Fock, coherent and thermal starts need 4,
  8 and 16 levels.
- **Finite-shot bias, measured against ground truth.** The raw shot estimate
  overstates the memory by up to 89% under realistic noise. A threshold-corrected
  estimator reduces the bias without removing it, and a Markovian control with a
  percentile rule bounds false positives at a measured miss rate.
- **What the measure identifies under noise.** Slow telegraph noise creates
  backflow in a memoryless channel (BLP up to 0.17), while Gaussian noise with the
  same correlation function creates none, so the measure characterizes the
  implemented channel, not the source of its memory.
- **Embeddings at equal gate cost.** A pseudomode and a two-site collision chain
  cross over in accuracy against transpiled CX count.

## Installation

```bash
pip install -r requirements-lock.txt   # exact versions used for the paper
```

The paper's results were produced with Python 3.12.6, NumPy 1.26.4, SciPy 1.15.2,
QuTiP 5.3.0, Qiskit 0.46.3, Qiskit Aer 0.13.3 and Matplotlib 3.10.3 on Windows.
`requirements.txt` gives the minimum versions.

## Reproducing the results

Run the validation gate first. It enforces the analytic - QuTiP - circuit
correctness chain and the numerical-equivalence checks:

```bash
python -m pytest tests -q
```

Each result of the revised paper is a JSON record in `results/`, written by one
function of `qmembench/experiments.py`; the figures are drawn from these records
by `qmembench/plotting_revision.py`. Every record stores its parameters, package
versions, timestamp and the git commit of the code that produced it (a `+dirty`
suffix marks a run made with uncommitted changes in the working tree).

| record | function | paper |
| --- | --- | --- |
| `phase1_stageA_validation` | `phase1` | Fig. 3 |
| `trotter_order_study` | `trotter_order_study` | Sec. 2.4, App. A |
| `pair_fock_convergence_*`, `blp_time_convergence_*`, `joint_fock_time_*`, `stageA_blp_time_convergence` | `tables_pair_and_fock_convergence`, `tables_blp_time_convergence`, `joint_fock_time_check`, `stageA_blp_time_convergence` | Tables 1-2, App. B |
| `boundary_mechanism_study` | `boundary_mechanism_study` | Sec. 3.2 |
| `finiteT_error_budget` | `finiteT_error_budget` | Sec. 3.1 |
| `phase9_truncation_phase_diagram_optimized`, `phase_diagram_blp_time_convergence` | same names | Fig. 4, App. B |
| `rhp_convergence_*`, `stageA_rhp_divergence` | `rhp_convergence_study`, `stageA_rhp_divergence` | Sec. 3.4, App. F |
| `fock_rule_study` | `fock_rule_study`, `fock_rule_recalibrate` | Sec. 4, Fig. 5, App. E |
| `estimator_characterization*`, `aer_shot_dispersion_check` | `estimator_characterization`, `aer_shot_dispersion_check` | Sec. 5, App. C, App. D |
| `noise_model_mapping`, `zne_multiseed`, `phase3_noise_models` | `noise_model_mapping`, `zne_multiseed`, `phase3_noise_models` | Sec. 6.1, App. D |
| `noisy_blp_grid_dependence` | same name | Sec. 5.3 |
| `correlated_noise_study` | `correlated_noise_study` | Sec. 6.2 |
| `gate_count_study`, `embedding_noise_leg` | same names | Sec. 7 |
| `dimer_convergence_and_memory` | `dimer_convergence_and_memory` | Sec. 8 |
| `phase7_scaling` | `phase7_scaling` | Fig. 9 |

The other `phase*` records come from the submitted version of the paper and are
kept as history; the revised paper does not use them (`phase1_stageA_validation`,
`phase3_noise_models` and `phase7_scaling` were rerun on the revised code). The long studies
checkpoint to `results/checkpoints/` and resume when rerun.

## Repository layout

```text
qmembench/
  measures.py     BLP and RHP measures, pair optimisation, Bloch utilities
  reference.py    analytic solution and QuTiP references (Stage A, finite-T, Stage B, dimer)
  circuits.py     pseudomode-embedding circuits (Trotter + Stinespring dilation)
  collision.py    collision models: single ancilla and chains with a damped end
  noise.py        device-noise models (scaled synthetic + calibrated fake-backend)
  mitigation.py   readout inversion and zero-noise extrapolation
  tomography.py   shot-based single-qubit state reconstruction
  statistics.py   bootstrap CIs, threshold-corrected BLP, Markovian control
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

All quantitative results and figures are derived from `results/*.json`. Seeded
studies record their seeds in their parameters. The BLP measure is a discrete
sum and therefore depends on the time grid, so every reported value is tied to
its grid, and values presented as properties of a model are tested for
convergence in time step and Fock dimension.

## Citation

If you use this code, please cite the accompanying paper and the archived
software. Release `v0.2.0`, the version used for the paper, is archived on Zenodo:

- this version: [10.5281/zenodo.23449846](https://doi.org/10.5281/zenodo.23449846)
- all versions (always resolves to the latest):
  [10.5281/zenodo.23449845](https://doi.org/10.5281/zenodo.23449845)

A `CITATION.cff` file is provided; GitHub's "Cite this repository" button uses it.

## License

MIT License. See `LICENSE`.

## Author

S.M. Yousuf Iqbal Tomal, Department of Computer Science and Engineering, BRAC
University, Dhaka, Bangladesh.
ORCiD: [0009-0000-3391-3824](https://orcid.org/0009-0000-3391-3824).

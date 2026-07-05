"""Statistical-rigor layer: bootstrap confidence intervals and shot-noise
bias handling for the BLP measure.

Why this module exists
----------------------
BLP sums *only the positive* increments of the trace distance D(t). Under
finite shots, D(t) is estimated with statistical error, and rectifying
(keeping only positive parts of) that noise turns pure fluctuations into
spurious "information backflow". The zero-signal floor is grid- and
noise-dependent (measured ~0.02-0.10 across noise scales at 4096 shots; see
phase3_bootstrap null floor and phase3_estimator_validation) -- large enough to
masquerade as real memory. phase3_estimator_validation measures the estimators'
bias against the exact noisy N_BLP ground truth.

Three complementary tools, all operating on *raw counts* (so no information
is discarded before resampling):

1. **Bootstrap CIs** -- multinomial resampling of the shot counts gives a
   distribution of N_BLP, hence honest error bars.
2. **Significance-thresholded (debiased) BLP** -- an increment dD_k is only
   accumulated if it exceeds z times its own bootstrap standard error, so
   statistically-insignificant (noise) increments are rejected.
3. **Empirical null floor** -- the apparent N_BLP of a *Markovian* control
   (true N_BLP = 0) at the same shot budget is a direct estimate of the
   bias floor, reported alongside every measurement.
"""

from __future__ import annotations

import numpy as np

from .measures import blp_from_curve, rho_from_bloch, trace_distance
from .tomography import BASES, expectation_from_counts

# Structured input type used throughout:
#   pair_counts[label][step][basis] -> {"0": n0, "1": n1}
CountsTriplet = dict[str, dict[str, int]]


def resample_counts(counts: dict[str, int], rng: np.random.Generator) -> dict[str, int]:
    """Parametric multinomial (here binomial) resample preserving shot total."""
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    N = n0 + n1
    if N == 0:
        return dict(counts)
    k0 = int(rng.binomial(N, n0 / N))
    return {"0": k0, "1": N - k0}


def rho_from_triplet(triplet: CountsTriplet,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Single-qubit rho from X/Y/Z counts; if ``rng`` given, resample first."""
    if rng is not None:
        triplet = {b: resample_counts(triplet[b], rng) for b in BASES}
    r = np.array([expectation_from_counts(triplet[b]) for b in BASES])
    return rho_from_bloch(r, project=True)


def _D_curve(pair_counts, step_list, pair, rng=None) -> np.ndarray:
    a, b = pair
    return np.array([
        trace_distance(rho_from_triplet(pair_counts[a][k], rng),
                       rho_from_triplet(pair_counts[b][k], rng))
        for k in step_list
    ])


def blp_point_estimate(pair_counts, step_list, pair) -> tuple[float, np.ndarray]:
    """Deterministic N_BLP and D(t) from the raw counts (no resampling)."""
    D = _D_curve(pair_counts, step_list, pair, rng=None)
    return blp_from_curve(D), D


def bootstrap_blp(pair_counts, step_list, pair, n_boot: int = 400,
                  seed: int = 0, z: float = 1.0, ci: float = 0.95) -> dict:
    """Bootstrap distribution of N_BLP plus a significance-debiased estimate.

    Returns point/mean/std/CI of N_BLP, the per-step D mean and std, and
    ``N_BLP_debiased`` = sum of increments that exceed ``z`` bootstrap SEs
    (the noise-rectification-controlled estimator).
    """
    rng = np.random.default_rng(seed)
    n = len(step_list)
    D_samples = np.zeros((n_boot, n))
    blp_samples = np.zeros(n_boot)
    for i in range(n_boot):
        D = _D_curve(pair_counts, step_list, pair, rng=rng)
        D_samples[i] = D
        blp_samples[i] = blp_from_curve(D)
    D_mean, D_std = D_samples.mean(axis=0), D_samples.std(axis=0)

    # Debiased estimator on the bootstrap-mean curve: accept only increments
    # that are statistically significant relative to their propagated SE.
    dD = np.diff(D_mean)
    step_se = np.sqrt(D_std[:-1] ** 2 + D_std[1:] ** 2)
    significant = dD > z * np.maximum(step_se, 1e-12)
    n_blp_debiased = float(np.sum(dD[significant & (dD > 0)]))

    point, _ = blp_point_estimate(pair_counts, step_list, pair)
    alpha = (1.0 - ci) / 2.0
    return {
        "N_BLP_point": point,
        "N_BLP_boot_mean": float(blp_samples.mean()),
        "N_BLP_boot_std": float(blp_samples.std()),
        "N_BLP_ci_low": float(np.quantile(blp_samples, alpha)),
        "N_BLP_ci_high": float(np.quantile(blp_samples, 1 - alpha)),
        "N_BLP_debiased": n_blp_debiased,
        "D_mean": D_mean,
        "D_std": D_std,
        "n_boot": n_boot,
        "z": z,
    }


def null_floor(control_counts, step_list, pair) -> float:
    """Apparent N_BLP of a Markovian control (true value 0) = the bias floor."""
    point, _ = blp_point_estimate(control_counts, step_list, pair)
    return point

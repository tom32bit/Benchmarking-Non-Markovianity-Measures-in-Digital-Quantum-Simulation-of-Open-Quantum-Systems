"""BLP pair optimisation over the Bloch sphere (exact via linearity of the map)."""
import numpy as np
import pytest

from qmembench.measures import (SX, SY, SZ, _hemisphere_lattice, blp_direction_values,
                                blp_from_curve, blp_measure_optimized,
                                blp_optimize_pair, dm, pauli_images_from_states,
                                trace_distance_curve)
from qmembench.reference import (analytic_G, analytic_stageA_states,
                                 stageA_qutip_states_finiteT)

GAMMA0, LAM = 5.0, 1.0
IC = ("0", "1", "+", "+i")


def _images_from_bloch_matrices(Ms):
    """Pauli images of a unital map with Bloch matrices Ms[t]:
    Lambda_t(sigma_k) = sum_i M_ik sigma_i."""
    return np.einsum("tik,iab->tkab", Ms, np.stack([SX, SY, SZ]))


def test_canonical_directions_reproduce_directly_evolved_pairs():
    """Linearity: D(t) of each canonical pair built from the Pauli images equals
    the trace distance of the separately evolved pair (non-unital finite-T map)."""
    times = np.linspace(0.0, 4.0, 41)
    labels = ("0", "1", "+", "-", "+i", "-i")
    states = {l: stageA_qutip_states_finiteT(dm(l), times, GAMMA0, LAM, n_th=0.5, n_fock=3)
              for l in labels}
    images = pauli_images_from_states(states)
    axes = {("0", "1"): (0, 0, 1), ("+", "-"): (1, 0, 0), ("+i", "-i"): (0, 1, 0)}
    via_images = blp_direction_values(images, np.array(list(axes.values()), dtype=float))
    for (a, b), value in zip(axes, via_images):
        direct = blp_from_curve(trace_distance_curve(states[a], states[b]))
        assert value == pytest.approx(direct, abs=1e-6)


def test_stageA_optimum_is_the_equatorial_pair():
    """Damped JC: D_n(t) = |G| sqrt(1 - n_z^2 (1 - G^2)), an increasing function
    of |G| with slope < 1 wherever |G| revives (|G| <= e^{-pi/3} here), so the
    equatorial pair is optimal and the optimum is the BLP of |G(t)|."""
    times = np.linspace(0.0, 12.0, 241)
    states = {l: analytic_stageA_states(dm(l), times, GAMMA0, LAM) for l in IC}
    res = blp_optimize_pair(pauli_images_from_states(states), n_grid=2000)
    target = blp_from_curve(np.abs(analytic_G(times, GAMMA0, LAM)))
    assert res["N_BLP_optimized"] == pytest.approx(target, abs=1e-9)
    assert abs(res["n_opt"][2]) < 1e-3
    assert res["gap"] == pytest.approx(0.0, abs=1e-9)


def test_optimizer_finds_an_off_axis_optimum():
    """A unital map whose revivals lie only along the rotated axis (1,1,1)/sqrt3:
    the optimum must clearly beat every canonical pair and match a brute-force
    search on a much finer lattice, with a converged budget."""
    t = np.linspace(0.0, 10.0, 201)
    g = np.column_stack([np.exp(-0.2 * t) * np.abs(np.cos(1.3 * t)),
                         0.3 * np.exp(-0.5 * t), 0.3 * np.exp(-0.5 * t)])
    u = np.ones(3) / np.sqrt(3.0)
    v = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    R = np.column_stack([u, v, np.cross(u, v)])          # orthonormal, first column u
    Ms = np.einsum("ij,tj,kj->tik", R, g, R)             # R diag(g(t)) R^T
    images = _images_from_bloch_matrices(Ms)
    res = blp_optimize_pair(images, n_grid=4000)
    brute = blp_direction_values(images, _hemisphere_lattice(100_000)).max()
    assert res["N_BLP_optimized"] >= brute - 1e-9
    assert res["N_BLP_optimized"] > res["N_BLP_canonical"] + 0.05
    assert res["budget_change"] < 1e-8
    assert abs(abs(float(np.dot(res["n_opt"], u))) - 1.0) < 1e-4


def test_channel_wrapper_evolves_only_the_four_ic_inputs():
    times = np.linspace(0.0, 6.0, 121)
    calls = []

    def channel(rho0):
        calls.append(rho0)
        return analytic_stageA_states(rho0, times, GAMMA0, LAM)

    res = blp_measure_optimized(channel, n_grid=500)
    assert len(calls) == 4
    assert res["N_BLP_optimized"] == pytest.approx(
        blp_from_curve(np.abs(analytic_G(times, GAMMA0, LAM))), abs=1e-9)

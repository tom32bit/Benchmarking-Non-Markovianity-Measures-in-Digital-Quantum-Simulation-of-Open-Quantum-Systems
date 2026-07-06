"""Figure generation.  Every figure is generated from the saved JSON payloads
(never from in-memory-only values), so any figure can be regenerated from its
raw data -- a provenance requirement (see RESEARCH_ETHICS.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiments import RESULTS_DIR


def _load(name: str) -> dict:
    with open(RESULTS_DIR / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def plot_phase9_phase_diagram() -> Path:
    """Two-panel (coupling, temperature) truncation phase diagram:
    (a) signed truncation error with the sign-flip contour and region labels;
    (b) the converged memory landscape in a sequential colormap.
    """
    from matplotlib.colors import TwoSlopeNorm
    d = _load("phase9_truncation_phase_diagram")["data"]
    g = np.array(d["g"])
    n_th = np.array(d["n_th"])
    signed = np.array(d["signed_error"])
    conv = np.array(d["converged_blp"])
    n_conv = d["n_conv"]
    G, T = np.meshgrid(g, n_th)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # (a) signed error, diverging, centred at zero. Use the TRUE min/max (not a
    # symmetric range) so each side fills its half of the colormap: the
    # destroy-side magnitudes are genuinely smaller (converged memory is itself
    # small there), and a symmetric scale would wash the blue region out.
    norm = TwoSlopeNorm(vmin=float(np.nanmin(signed)) - 1e-6, vcenter=0.0,
                        vmax=float(np.nanmax(signed)))
    pcm = axes[0].pcolormesh(G, T, signed, cmap="RdBu_r", norm=norm, shading="auto",
                             rasterized=True)
    cs = axes[0].contour(G, T, signed, levels=[0.0], colors="k", linewidths=1.6)
    axes[0].clabel(cs, fmt={0.0: "sign flip"}, fontsize=8, inline=True)
    cb = fig.colorbar(pcm, ax=axes[0], pad=0.02)
    cb.set_label(r"$\mathcal{N}_{\mathrm{BLP}}(n{=}2) - \mathcal{N}_{\mathrm{BLP}}$"
                 f"(conv.)")
    axes[0].set_xlabel(r"coupling $g$")
    axes[0].set_ylabel(r"temperature $\bar n$")
    axes[0].set_title("Signed truncation error")
    # region labels placed at representative points
    axes[0].text(0.66, 0.16, "fabricates\n(over)", ha="center", va="center",
                 fontsize=10.5, color="#5c0d0d", weight="bold")
    axes[0].text(0.28, 0.82, "destroys\n(under)", ha="center", va="center",
                 fontsize=10.5, color="white", weight="bold")

    # (b) converged memory landscape, sequential
    pcm2 = axes[1].pcolormesh(G, T, conv, cmap="inferno", shading="auto",
                              rasterized=True)
    cb2 = fig.colorbar(pcm2, ax=axes[1], pad=0.02)
    cb2.set_label(r"$\mathcal{N}_{\mathrm{BLP}}$ (converged, $n=%d$)" % n_conv)
    axes[1].set_xlabel(r"coupling $g$")
    axes[1].set_ylabel(r"temperature $\bar n$")
    axes[1].set_title("True memory landscape")

    fig.suptitle("Truncation error changes sign across the coupling-temperature plane "
                 "(finite-$T$ spin-boson)", fontsize=11)
    return _save(fig, "phase9_truncation_phase_diagram")


def _save(fig, name: str) -> Path:
    path = RESULTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_phase0() -> Path:
    payload = _load("phase0_measures_validation")
    d = payload["data"]
    times = np.array(d["times"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tag, title in zip(axes, ("nonmarkovian", "markovian"),
                              ("Non-Markovian ($\\gamma_0 = 5\\lambda$)",
                               "Markovian ($\\gamma_0 = 0.2\\lambda$)")):
        block = d[tag]
        for pair, curve in block["D_curves"].items():
            ax.plot(times, curve, label=f"pair {pair}")
        ax.plot(times, block["abs_G"], "k--", lw=1,
                label="analytic optimum $|G(t)|$")
        ax.set_title(f"{title}\n$N_{{BLP}}$ = {block['N_BLP']:.4f}, "
                     f"$N_{{RHP}}$ = {block['N_RHP']:.4f}")
        ax.set_xlabel("time $\\lambda t$")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("trace distance $D(t)$")
    fig.suptitle("BLP and RHP measures validated on the analytic Stage A solution")
    return _save(fig, "phase0_measures_validation")


def plot_phase1() -> Path:
    payload = _load("phase1_stageA_validation")
    d = payload["data"]
    times = np.array(d["times"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {"analytic": ("k-", 2), "qutip": ("C0--", 1.5), "circuit": ("C3:", 2)}
    pair = d["analytic"]["best_pair"]
    for tier in ("analytic", "qutip", "circuit"):
        fmt, lw = styles[tier]
        ax.plot(times, d[tier]["D_curves"][pair], fmt, lw=lw,
                label=f"{tier} ($N_{{BLP}}$ = {d[tier]['N_BLP']:.4f})")
    ax.set_xlabel("time $\\lambda t$")
    ax.set_ylabel(f"trace distance, pair {pair}")
    ax.set_title("Three-tier validation (analytic vs QuTiP vs circuit)\n"
                 f"max dev circuit vs analytic = {d['max_abs_dev_circuit_vs_analytic']:.2e}, "
                 f"dt = {d['dt']}")
    ax.legend()
    return _save(fig, "phase1_stageA_validation")


def plot_phase2() -> Path:
    payload = _load("phase2_spin_boson")
    rows = payload["data"]["rows"]
    g = [r["g"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n_conv = payload["data"].get("n_conv", 6)
    axes[0].plot(g, [r["N_BLP_circuit"] for r in rows], "o-", label="circuit (2-level modes)")
    axes[0].plot(g, [r["N_BLP_ref_nfock2"] for r in rows], "s--", label="QuTiP $n_{fock}=2$")
    axes[0].plot(g, [r["N_BLP_ref_converged"] for r in rows], "^:",
                 label=f"QuTiP converged ($n={n_conv}$)")
    axes[0].set_xlabel("coupling $g$")
    axes[0].set_ylabel("$N_{BLP}$")
    axes[0].set_title("Spin-boson: memory vs coupling")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(g, [max(r["max_dev_circuit_vs_ref2"], 1e-16) for r in rows],
                     "o-", label="circuit vs matched-truncation ref")
    axes[1].semilogy(g, [max(r["truncation_error_2_vs_converged"], 1e-16) for r in rows],
                     "s--", label=f"Fock truncation error (2 vs {n_conv})")
    axes[1].set_xlabel("coupling $g$")
    axes[1].set_ylabel("max abs deviation")
    axes[1].set_title("Error decomposition (Trotter vs truncation)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Stage B spin-boson with a structured (2-pseudomode) bath")
    return _save(fig, "phase2_spin_boson")


def plot_phase3() -> Path:
    payload = _load("phase3_noise_robustness")
    d = payload["data"]
    rows = d["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    times = np.array(d["times"])
    for r in rows:
        if r["pipeline"] == "raw":
            axes[0].plot(times, r["D_curve"], alpha=0.8,
                         label=f"scale {r['noise_scale']}")
    axes[0].set_xlabel("time $\\lambda t$")
    axes[0].set_ylabel("$D(t)$ (shot mode, raw)")
    axes[0].set_title("Backflow washout with increasing device noise")
    axes[0].legend(fontsize=8)
    for pipeline, marker in (("raw", "o"), ("readout_mitigated", "s")):
        pts = [(r["noise_scale"], r["N_BLP"]) for r in rows if r["pipeline"] == pipeline]
        if pts:
            xs, ys = zip(*sorted(pts))
            axes[1].plot(xs, ys, marker + "-", label=pipeline)
    axes[1].axhline(d["N_BLP_ideal_reference"], color="k", ls="--", lw=1,
                    label="ideal reference")
    axes[1].set_xlabel("noise scale $s$")
    axes[1].set_ylabel("$N_{BLP}$")
    axes[1].set_title("Memory signature vs noise strength")
    axes[1].legend(fontsize=8)
    fig.suptitle("Noise robustness of the simulated memory signature")
    return _save(fig, "phase3_noise_robustness")


def plot_phase1T() -> Path:
    payload = _load("phase1T_finite_temperature")
    rows = payload["data"]["rows"]
    n_th = [r["n_th"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(n_th, [r["N_BLP_circuit"] for r in rows], "o-", label="circuit (2-level mode)")
    axes[0].plot(n_th, [r["N_BLP_ref_nfock2"] for r in rows], "s--", label="QuTiP $n_{fock}=2$")
    axes[0].plot(n_th, [r["N_BLP_ref_converged"] for r in rows], "^:",
                 label=f"QuTiP converged ($n={payload['data']['n_conv']}$)")
    axes[0].set_xlabel("thermal occupation $n_{th}$")
    axes[0].set_ylabel("$N_{BLP}$")
    axes[0].set_title("Memory vs temperature:\n2-level truncation collapses the signal")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(n_th, [max(r["truncation_error_2_vs_converged"], 1e-16) for r in rows],
                     "s--", label="truncation error (2 vs converged)")
    axes[1].semilogy(n_th, [max(r["max_dev_circuit_vs_ref2"], 1e-16) for r in rows],
                     "o-", label="circuit vs matched-truncation ref")
    axes[1].set_xlabel("thermal occupation $n_{th}$")
    axes[1].set_ylabel("max abs deviation")
    axes[1].set_title("Truncation error grows with T\n(circuit stays faithful to its 2-level target)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Finite temperature: the truncation error at the two-level encoding")
    return _save(fig, "phase1T_finite_temperature")


def plot_phase3_bootstrap() -> Path:
    payload = _load("phase3_bootstrap")
    d = payload["data"]
    rows = d["rows"]
    scales = np.array([r["noise_scale"] for r in rows])
    point = np.array([r["N_BLP_point"] for r in rows])
    deb = np.array([r["N_BLP_debiased"] for r in rows])
    ci_lo = np.array([r["N_BLP_ci_low"] for r in rows])
    ci_hi = np.array([r["N_BLP_ci_high"] for r in rows])
    floor = np.array([r["null_floor_markovian"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # 95% bootstrap CI drawn as a band (robust to point lying outside it)
    ax.fill_between(scales, ci_lo, ci_hi, color="C0", alpha=0.2,
                    label="95% bootstrap CI")
    ax.plot(scales, point, "o-", color="C0", label="point $N_{BLP}$ (raw)")
    ax.plot(scales, deb, "s--", color="C3", label="debiased $N_{BLP}$")
    ax.plot(scales, floor, "^:", color="gray", label="Markovian null floor (bias)")
    ax.axhline(d["N_BLP_ideal_reference"], color="k", ls="-", lw=1, alpha=0.6,
               label="ideal reference")
    ax.set_xlabel("noise scale $s$")
    ax.set_ylabel("$N_{BLP}$")
    ax.set_title(f"Statistical rigor: bootstrap CIs, debiasing, null floor\n"
                 f"(shots={d['shots']}, n_boot={d['n_boot']})")
    ax.legend(fontsize=8)
    return _save(fig, "phase3_bootstrap")


def plot_phase3_estimator_validation() -> Path:
    d = _load("phase3_estimator_validation")["data"]
    rows = d["rows"]
    scales = [r["noise_scale"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for key, lab, fmt in [("raw", "raw", "o-"), ("debiased_z1.0", "debiased (z=1)", "s-"),
                          ("readout_mitigated", "readout-mitigated", "^-")]:
        bias = [r.get(f"{key}_bias") for r in rows]
        if any(b is not None for b in bias):
            xs = [s for s, b in zip(scales, bias) if b is not None]
            ys = [b for b in bias if b is not None]
            ax.plot(xs, ys, fmt, label=lab)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("noise scale $s$")
    ax.set_ylabel("estimator bias vs exact noisy $N_{BLP}$")
    cov = ", ".join(f"s{r['noise_scale']}:{r['ci_coverage_z1']:.0%}" for r in rows)
    ax.set_title(f"Estimator validation: bias vs computable ground truth\n"
                 f"raw is positively biased; debiasing reduces |bias|. CI coverage: {cov}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    return _save(fig, "phase3_estimator_validation")


def plot_phase3_noise_models() -> Path:
    d = _load("phase3_noise_models")["data"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # (left) exact-mode N_BLP under the built-in noise conditions
    conds = [("ideal", "ideal"), ("markovian_depolarizing", "Markovian depolarizing"),
             ("realistic_fake_backend", "realistic fake-backend")]
    labels, vals = [], []
    for key, lab in conds:
        blk = d.get(key)
        if isinstance(blk, dict) and "N_BLP" in blk:
            labels.append(lab); vals.append(blk["N_BLP"])
    axes[0].barh(range(len(vals)), vals, color=["#444", "#27ae60", "#e67e22"][:len(vals)])
    axes[0].set_yticks(range(len(vals))); axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].axvline(d["N_BLP_ideal_reference"], color="k", ls="--", lw=1, label="ideal ref")
    axes[0].set_xlabel("$N_{BLP}$"); axes[0].invert_yaxis(); axes[0].legend(fontsize=8)
    axes[0].set_title("Device-noise conditions (exact mode)")
    # (right) sigma sweep: correlated (NM) vs memoryless dephasing, with CIs
    sw = d["sigma_sweep"]
    sig = [e["sigma_tot"] for e in sw]
    for key, lab, col in [("nonmarkovian", "non-Markovian (correlated)", "C3"),
                          ("markovian", "Markovian (matched)", "C0")]:
        y = [e[key]["N_BLP"] for e in sw]
        lo = [e[key]["N_BLP"] - e[key]["ci_low"] for e in sw]
        hi = [e[key]["ci_high"] - e[key]["N_BLP"] for e in sw]
        axes[1].errorbar(sig, y, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)],
                         fmt="o-", color=col, capsize=3, label=lab)
    seps = [e["sigma_tot"] for e in sw if e["cis_separated"]]
    axes[1].set_xlabel(r"accumulated dephasing $\sigma_{tot}$")
    axes[1].set_ylabel("$N_{BLP}$ (ensemble, 95% CI)")
    axes[1].set_title("Correlated noise preserves memory more than memoryless\n"
                      f"(CIs separate at $\\sigma_{{tot}}\\in$ {seps if seps else 'none'})")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("Device noise: Markovian vs realistic vs non-Markovian")
    return _save(fig, "phase3_noise_models")


def plot_phase5_fock_rule() -> Path:
    payload = _load("phase5_fock_rule")
    d = payload["data"]
    pts = d["points"]
    c = d["c_var"]
    markers = {"temperature": ("o", "C3"), "coupling": ("s", "C0")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (left) mean-only predictor: the two families SEPARATE
    for fam, (mk, col) in markers.items():
        xs = [p["n_max"] for p in pts if p["family"] == fam]
        ys = [p["d_req"] for p in pts if p["family"] == fam]
        axes[0].scatter(xs, ys, marker=mk, color=col, s=70, zorder=3, label=fam)
    ng = np.linspace(0, max(p["n_max"] for p in pts) * 1.1, 100)
    axes[0].plot(ng, np.ceil(ng + 2 * np.sqrt(ng + 1)) + 1, "k--", lw=1.2,
                 label="mean-only rule")
    axes[0].set_xlabel(r"peak occupation $\langle n\rangle_{max}$")
    axes[0].set_ylabel("required Fock dimension $d_{req}$")
    axes[0].set_title(f"Mean-only predictor: families SEPARATE\n"
                      f"(corr={d['corr_dreq_vs_nmax']:.2f}, "
                      f"safe bound: {d['mean_rule_is_safe']})")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # (right) variance-aware predictor: the two families COLLAPSE
    for fam, (mk, col) in markers.items():
        xs = [p["n_max"] + c * p["std_at_peak"] for p in pts if p["family"] == fam]
        ys = [p["d_req"] for p in pts if p["family"] == fam]
        axes[1].scatter(xs, ys, marker=mk, color=col, s=70, zorder=3, label=fam)
    xg = np.linspace(0, max(p["n_max"] + c * p["std_at_peak"] for p in pts) * 1.1, 100)
    axes[1].plot(xg, np.ceil(xg) + 1, "k--", lw=1.2, label=r"rule $\lceil P\rceil+1$")
    axes[1].set_xlabel(rf"variance-aware predictor $\langle n\rangle_{{max}} + {c:g}\,\sigma_n$")
    axes[1].set_ylabel("required Fock dimension $d_{req}$")
    axes[1].set_title(f"Variance-aware predictor: families COLLAPSE\n"
                      f"(corr={d['corr_dreq_vs_variance_predictor']:.2f}, "
                      f"safe bound: {d['variance_rule_is_safe']})")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle("The required Fock dimension is set by the occupied phase-space "
                 "extent, not the mean occupation")
    return _save(fig, "phase5_fock_rule")


def plot_phase8_dimer() -> Path:
    d = _load("phase8_dimer")["data"]
    t = np.array(d["times"])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(t, d["P_L_ref"], "C0-", label="site L (QuTiP)")
    ax.plot(t, d["P_R_ref"], "C3-", label="site R (QuTiP)")
    ax.plot(t, d["P_L_circuit"], "C0o", ms=3, markevery=6, label="site L (circuit)")
    ax.plot(t, d["P_R_circuit"], "C3s", ms=3, markevery=6, label="site R (circuit)")
    ax.set_xlabel("time")
    ax.set_ylabel("site excitation population")
    ax.set_title("FMO-like excitonic dimer energy transfer\n"
                 f"transfer eff={d['transfer_efficiency']:.2f}, "
                 f"acceptor revival={d['acceptor_population_revival']:.3f}, "
                 f"max dev circuit vs QuTiP={d['max_dev_PR_circuit_vs_qutip']:.2e}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return _save(fig, "phase8_dimer")


def plot_phase7_scaling() -> Path:
    d = _load("phase7_scaling")["data"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    # (i) Trotter error vs dt
    tr = d["trotter"]
    axes[0].loglog([r["dt"] for r in tr], [r["max_dev"] for r in tr], "o-")
    axes[0].set_xlabel("Trotter step $dt$")
    axes[0].set_ylabel("max trajectory error")
    axes[0].set_title(f"Trotter error $\\sim dt^{{{d['trotter_order_estimate']:.2f}}}$\n"
                      f"(commuting coherent gates -> effectively 2nd order here)")
    axes[0].grid(alpha=0.3, which="both")
    # (ii) truncation error vs Fock d
    tc = d["truncation"]
    axes[1].semilogy([r["d"] for r in tc], [max(r["max_dev"], 1e-16) for r in tc], "s-")
    axes[1].set_xlabel("Fock dimension $d$")
    axes[1].set_ylabel("truncation error")
    axes[1].set_title(f"Truncation error decay (finite T)\n"
                      f"$\\log$ slope {d['truncation_log_decay_per_level']:.2f}/level")
    axes[1].grid(alpha=0.3, which="both")
    # (iii) resources vs steps
    rs = d["resources_vs_steps"]
    axes[2].plot([r["n_steps"] for r in rs], [r["depth"] for r in rs], "o-", label="depth")
    axes[2].plot([r["n_steps"] for r in rs], [r["cx"] for r in rs], "s-", label="CX count")
    axes[2].set_xlabel("Trotter steps")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"Resources scale linearly in steps\n"
                      f"(depth slope {d['depth_loglog_slope_vs_steps']:.2f})")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)
    fig.suptitle("Scaling laws: Trotter convergence, truncation decay, linear resources")
    return _save(fig, "phase7_scaling")


def plot_phase6_fair_collision() -> Path:
    payload = _load("phase6_fair_collision")
    d = payload["data"]
    rows = d["collision_train"]
    q = [r["n_qubits"] for r in rows]
    err = [r["N_BLP_error"] for r in rows]
    rmse = [r["pop_rmse"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(q, err, "s-", color="C0", label="collision train")
    axes[0].axhline(d["pseudomode"]["N_BLP_error"], color="C3", ls="--",
                    label=f"pseudomode (3 qubits)")
    axes[0].scatter([3], [d["pseudomode"]["N_BLP_error"]], color="C3", zorder=4)
    axes[0].set_xlabel("qubits used")
    axes[0].set_ylabel("$|N_{BLP} - N_{BLP}^{ref}|$")
    axes[0].set_title("Accuracy vs qubit cost:\npseudomode is fixed & cheap")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot([r["M"] for r in rows], rmse, "o-", color="C2")
    axes[1].set_xlabel("chain length $M$")
    axes[1].set_ylabel("population-fit RMSE to Lorentzian")
    axes[1].set_title("Best-achievable fit vs chain length")
    axes[1].grid(alpha=0.3)
    mreq = d["collision_qubits_to_match"]
    sub = f"collision needs {mreq} qubits to match" if mreq else "collision does not match within tested M"
    fig.suptitle(f"Fair comparison: ancilla-train vs pseudomode -- {sub}")
    return _save(fig, "phase6_fair_collision")


def plot_phase4() -> Path:
    payload = _load("phase4_embedding_comparison")
    d = payload["data"]
    times = np.array(d["times"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(times, d["abs_G"], "k--", lw=1, label="reference $|G(t)|$")
    axes[0].plot(times, d["D_pseudomode"], "C0-", label="pseudomode")
    axes[0].plot(times, d["D_collision"], "C3-", label="collision (fitted)")
    axes[0].set_xlabel("time $\\lambda t$")
    axes[0].set_ylabel("$D(t)$")
    axes[0].set_title("Backflow curves (exact mode)")
    axes[0].legend(fontsize=8)

    res = d["resources"]
    metrics = ["n_qubits", "depth", "cx_count"]
    x = np.arange(len(metrics))
    for i, emb in enumerate(("pseudomode", "collision")):
        axes[1].bar(x + i * 0.35, [res[emb][m] for m in metrics], width=0.35, label=emb)
    axes[1].set_xticks(x + 0.175)
    axes[1].set_xticklabels(metrics)
    axes[1].set_yscale("log")
    axes[1].set_title("Transpiled resources (full evolution)")
    axes[1].legend(fontsize=8)

    rob = d["noise_robustness"]
    xs = [r["noise_scale"] for r in rob]
    axes[2].plot(xs, [r["N_BLP_pseudomode"] for r in rob], "o-", label="pseudomode")
    axes[2].plot(xs, [r["N_BLP_collision"] for r in rob], "s-", label="collision")
    axes[2].axhline(d["N_BLP_reference"], color="k", ls="--", lw=1, label="reference")
    axes[2].set_xlabel("noise scale $s$")
    axes[2].set_ylabel("$N_{BLP}$")
    axes[2].set_title("Noise robustness head-to-head")
    axes[2].legend(fontsize=8)
    fig.suptitle("[Superseded] strawman collision comparison\n"
                 "(retained only to document the retraction; see the fair comparison)")
    return _save(fig, "phase4_embedding_comparison")

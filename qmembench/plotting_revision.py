"""Figures for the revised manuscript, drawn from the revision records.

Style: one validated categorical order (blue, orange, aqua, yellow; checked for
colour-vision deficiency with the dataviz validator), an ordinal blue ramp for
ordered categories, hairline recessive grid, no in-figure titles, sizes set at
print size. Each function reads its JSON record under results/ and writes a PNG
to results/figures_revision/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "figures_revision"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # categorical order
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # light -> dark blue
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
PANEL = (3.3, 2.5)                                            # a 0.49\textwidth panel, inches


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
        "savefig.dpi": 300, "figure.dpi": 100, "axes.spines.top": False,
        "axes.spines.right": False})


def _load(name: str) -> dict:
    with open(RESULTS / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)["data"]


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_estimator_bias() -> Path:
    """Bias against the state truth vs noise scale, both grids."""
    _style()
    d = _load("estimator_characterization")
    fig, ax = plt.subplots(figsize=PANEL)
    specs = [("raw", "raw", "o"), ("thr1", "threshold ($z=1$)", "s"),
             ("boot_bc", "bootstrap-corrected", "^"), ("mitigated", "readout-mitigated", "D")]
    for (key, label, mk), col in zip(specs, SERIES):
        for grid, ls, show in (("A", "-", True), ("B", "--", False)):
            rows = d["grids"][grid]["rows"]
            s = [r["noise_scale"] for r in rows]
            if key == "thr1":
                z = np.array(rows[0]["z_grid"])
                i = int(np.argmin(abs(z - 1.0)))
                vals = [r["sweep"]["threshold_Dbar"][i] for r in rows]
            else:
                vals = [r["estimators"][key] for r in rows]
            ax.errorbar(s, [v["bias_state"] for v in vals], yerr=[v["sem"] for v in vals],
                        color=col, ls=ls, marker=mk, mfc=col if show else "white",
                        mec=col, capsize=0, label=label if show else None)
    ax.axhline(0.0, color=AXIS, lw=0.8, zorder=1)
    ax.set_xlabel("noise scale $s$")
    ax.set_ylabel("bias against state truth")
    ax.set_xticks([0, 0.5, 1, 2])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=2.2)
    ax.text(0.99, 0.02, "solid: fine grid   dashed, open: coarse grid", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=INK2)
    return _save(fig, "estimator_bias.png")


def fig_estimator_zsweep() -> Path:
    """RMSE of the threshold estimator vs z on the fine grid, one line per scale."""
    _style()
    d = _load("estimator_characterization")
    rows = d["grids"]["A"]["rows"]
    fig, ax = plt.subplots(figsize=PANEL)
    for r, col in zip(rows, ORDINAL):
        z = np.array(r["z_grid"])
        rm = [s["rmse_state"] for s in r["sweep"]["threshold_Dbar"]]
        ax.plot(z, rm, color=col, marker="o", ms=3, label=f"$s = {r['noise_scale']:g}$")
        ax.plot([-0.35], [r["estimators"]["raw"]["rmse_state"]], marker="D", ms=4, color=col,
                mfc="white", ls="none")
    ax.axvline(-0.175, color=AXIS, lw=0.6)
    ax.set_xticks([-0.35, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xticklabels(["raw", "0", "0.5", "1", "1.5", "2", "2.5", "3"])
    ax.set_xlabel("threshold $z$")
    ax.set_ylabel("RMS error against state truth")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", title="noise scale", title_fontsize=7)
    return _save(fig, "estimator_zsweep.png")


def fig_noise_washout_zne() -> Path:
    """ZNE at scales 1 and 2: exact values, infinite-shot limit, shot estimates."""
    _style()
    d = _load("zne_multiseed")
    fig, ax = plt.subplots(figsize=PANEL)
    offsets = {"unfolded_raw": -0.12, "zne_raw": -0.04, "zne_threshold_z1": 0.04}
    labels = {"unfolded_raw": "raw, no extrapolation", "zne_raw": "raw, extrapolated",
              "zne_threshold_z1": "threshold ($z=1$), extrapolated"}
    marks = {"unfolded_raw": "o", "zne_raw": "s", "zne_threshold_z1": "^"}
    for i, r in enumerate(d["rows"]):
        x = float(i)
        for (key, off), col in zip(offsets.items(), SERIES):
            e = r["estimators"][key]
            ax.errorbar(x + off, e["mean"], yerr=e["std"], color=col, marker=marks[key],
                        capsize=2, lw=1.0, label=labels[key] if i == 0 else None)
        ax.plot([x + 0.12], [r["zne_infinite_shot_limit"]], marker="*", ms=7, color=SERIES[3],
                ls="none", label="extrapolation, exact probabilities" if i == 0 else None)
        ax.plot([x - 0.2, x + 0.2], [r["state_truth"]] * 2, color=INK, lw=1.2,
                label="exact noisy value" if i == 0 else None)
    ax.axhline(d["rows"][0]["ideal_on_grid"], color=MUTED, lw=1.0, ls="--",
               label="ideal value", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"$s = {r['noise_scale']:g}$" for r in d["rows"]])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("BLP ($\\delta t = 0.1$, sampled every 0.2)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.2,
              handlelength=1.6)
    return _save(fig, "noise_washout_zne.png")


def fig_correlated_noise() -> Path:
    """BLP vs correlation time at sigma = 0.2: control and model as small multiples."""
    _style()
    d = _load("correlated_noise_study")
    taus = [t for t in d["tau_cs"]]
    xpos = [300.0 if t == "inf" else float(t) for t in taus]
    fig, axes = plt.subplots(1, 2, figsize=(PANEL[0], PANEL[1]), sharex=True)
    for ax, model, title in ((axes[0], "markovian_control", "Markovian control"),
                             (axes[1], "nonmarkovian", "non-Markovian model")):
        for (proc, label, mk), col in zip((("telegraph", "telegraph", "o"), ("ou", "Gaussian", "s")),
                                          SERIES):
            vals = [next(e["optimized"] for e in d[model]["fixed_stationary_variance"]
                         if e["process"] == proc and e["sigma"] == 0.2 and e["tau_c"] == t)
                    for t in taus]
            ax.plot(xpos, vals, color=col, marker=mk, label=label)
        ax.axhline(d[model]["noiseless"]["optimized"], color=MUTED, lw=1.0, ls="--",
                   label="no noise", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10, 300])
        ax.set_xticklabels(["0.1", "1", "10", "$\\infty$"])
        ax.set_xlabel("$\\lambda\\tau_c$")
        ax.set_title(title, fontsize=7, color=INK, loc="left")
    axes[0].set_ylabel("BLP at $\\sigma = 0.2\\lambda$")
    axes[1].legend(loc="lower left", fontsize=6.2)
    fig.tight_layout(w_pad=0.8)
    return _save(fig, "correlated_noise.png")


def fig_embedding_cx() -> Path:
    """BLP error vs transpiled CX for the pseudomode and the chains, with Pareto fronts."""
    _style()
    d = _load("gate_count_study")
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    pm = d["pseudomode"]
    ax.plot([r["cx"] for r in pm], [r["error"] for r in pm], ls="none", marker="o",
            color=SERIES[0], label="pseudomode (steps 0.45 to 0.05)")
    shapes = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
    for M, mk in shapes.items():
        rows = [r for r in d["chain"] if r["M"] == M]
        ax.plot([r["cx"] for r in rows], [r["error"] for r in rows], ls="none", marker=mk,
                mfc="white", mec=SERIES[1], mew=1.0, label=f"chain, $M = {M}$")
    for front, col, lab in ((d["pareto_pseudomode"], SERIES[0], "front"),
                            (d["pareto_chain"], SERIES[1], "front")):
        cx = [c for c, _ in front]
        er = [e for _, e in front]
        ax.step(cx + [2100], er + [er[-1]], where="post", color=col, lw=1.0, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("transpiled CX count")
    ax.set_ylabel("BLP error against closed form")
    ax.set_xticks([150, 300, 600, 1000, 2000])
    ax.set_xticklabels(["150", "300", "600", "1000", "2000"])
    ax.minorticks_off()
    ax.legend(loc="lower left", ncol=2, fontsize=6.5)
    return _save(fig, "embedding_cx.png")


def fig_dimer_populations() -> Path:
    """Donor and acceptor populations: converged reference, circuit, isolated dimer."""
    _style()
    d = _load("dimer_convergence_and_memory")
    t = np.array(d["times"])
    pr_ref = np.array(d["acceptor_reference_dmax"])
    pr_iso = np.array(d["acceptor_isolated"])
    pr_circ = np.array(d["circuit"]["0.05"]["acceptor_population"])
    fig, ax = plt.subplots(figsize=(4.7, 2.6))
    for pop, col, lab in ((1.0 - pr_ref, SERIES[0], "donor $P_L$"), (pr_ref, SERIES[1], "acceptor $P_R$")):
        ax.plot(t, pop, color=col, lw=1.4, label=f"{lab}, reference")
    for pop, col in ((1.0 - pr_circ, SERIES[0]), (pr_circ, SERIES[1])):
        ax.plot(t[::4], pop[::4], ls="none", marker="o", ms=3, mfc="white", mec=col, mew=0.9)
    for pop, col in ((1.0 - pr_iso, SERIES[0]), (pr_iso, SERIES[1])):
        ax.plot(t, pop, color=col, lw=0.9, ls="--", alpha=0.8)
    ax.plot([], [], ls="none", marker="o", ms=3, mfc="white", mec=INK2, label="circuit")
    ax.plot([], [], color=INK2, lw=0.9, ls="--", label="isolated dimer")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("site population")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=6.5,
              handlelength=1.8, columnspacing=1.0)
    return _save(fig, "dimer_populations.png")


def fig_phase_diagram_optimized() -> Path:
    """Signed truncation error N(n=2) - N(d_ref) for the optimised pair, and the
    reference memory, over the (coupling, temperature) plane (rule R7 record).

    Points with |error| <= 1e-3 are drawn grey (sign not resolved); points where
    the reference changes by >= 1e-3 between d_check and d_ref are dotted. The
    zero contour is drawn where the reference memory is at least 5e-3.
    """
    from matplotlib.colors import TwoSlopeNorm
    _style()
    d = _load("phase9_truncation_phase_diagram_optimized")
    g, nth = np.array(d["g"]), np.array(d["n_th"])
    E = np.array(d["signed_error_optimized"])
    ref = np.array(d["reference_optimized"])
    chk = np.array(d["reference_check_max_abs"])
    G, T = np.meshgrid(g, nth)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax in axes:
        ax.grid(False)
    ax = axes[0]
    resolved = np.abs(E) > 1e-3
    norm = TwoSlopeNorm(vmin=float(E.min()), vcenter=0.0, vmax=float(E.max()))
    pcm = ax.pcolormesh(G, T, np.ma.masked_where(~resolved, E), cmap="RdBu_r", norm=norm,
                        shading="nearest", rasterized=True)
    ax.pcolormesh(G, T, np.ma.masked_where(resolved, np.zeros_like(E)), cmap="Greys",
                  vmin=-1, vmax=4, shading="nearest", rasterized=True)
    ax.contour(G, T, np.ma.masked_where(ref < 5e-3, E), levels=[0.0], colors=INK,
               linewidths=0.9)
    unver = chk >= 1e-3
    ax.plot(G[unver], T[unver], ls="none", marker=".", ms=1.6, color=INK)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\mathcal{N}_{\mathrm{BLP}}(n{=}2) - \mathcal{N}_{\mathrm{BLP}}(d{=}%d)$"
                 % d["d_ref"])
    cb.ax.tick_params(labelsize=6.5)
    ax.text(0.68, 0.12, "overstates", ha="center", va="center", fontsize=7.5, color=INK)
    ax.text(0.40, 0.95, "understates", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_title("(a) signed truncation error", fontsize=7, color=INK, loc="left")
    ax = axes[1]
    pcm2 = ax.pcolormesh(G, T, ref, cmap="viridis", shading="nearest", rasterized=True)
    cb2 = fig.colorbar(pcm2, ax=ax, pad=0.02)
    cb2.set_label(r"$\mathcal{N}_{\mathrm{BLP}}(d{=}%d)$" % d["d_ref"])
    cb2.ax.tick_params(labelsize=6.5)
    ax.set_title("(b) reference memory", fontsize=7, color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("coupling $g$")
        ax.set_ylabel(r"thermal occupation $\bar n$")
    fig.tight_layout(w_pad=1.0)
    return _save(fig, "phase_diagram_optimized.png")


def fig_fock_rule() -> Path:
    """Required Fock dimension at the measure-level target eps_NM < 1e-3
    against (a) the peak mean occupation and (b) the variance-aware predictor
    at the calibrated c, for the three families of the R13 study."""
    _style()
    d = _load("fock_rule_study")
    t = "eps_NM<0.001"
    c = d["calibration"][t]["c_all_points"]
    ci = int(np.argmin(np.abs(np.array(d["c_grid"]) - c)))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    styles = {("temperature", "vacuum"): (SERIES[1], "o", "temperature"),
              ("coupling", "vacuum"): (SERIES[0], "s", "coupling"),
              ("controlled", "fock"): (SERIES[2], "^", "Fock start"),
              ("controlled", "coherent"): (SERIES[2], "D", "coherent start"),
              ("controlled", "thermal"): (SERIES[2], "v", "thermal start")}
    for (fam, kind), (col, mk, lab) in styles.items():
        pts = [p for p in d["points"] if p["family"] == fam and p["kind"] == kind]
        y = [p["d_req"][t]["d_req"] for p in pts]
        filled = fam != "controlled"
        kw = dict(ls="none", marker=mk, ms=4.5, mec=col, mew=1.0,
                  mfc=col if filled else "white", label=lab)
        axes[0].plot([p["max_mean"] for p in pts], y, **kw)
        axes[1].plot([p["predictor"][ci] for p in pts], y, **kw)
    xg = np.linspace(0.0, max(p["predictor"][ci] for p in d["points"]) + 0.5, 600)
    axes[1].plot(xg, np.ceil(xg) + 1, color=INK2, lw=0.9, label=r"$\lceil x\rceil + 1$")
    axes[0].set_xlabel(r"peak mean occupation $\langle n\rangle$")
    axes[1].set_xlabel(r"predictor $\max(\langle n_k\rangle + %g\,\sigma_k)$" % c)
    axes[0].set_ylabel(r"required dimension $d_{\mathrm{req}}$")
    axes[0].set_title("(a) mean occupation", fontsize=7, color=INK, loc="left")
    axes[1].set_title(r"(b) variance-aware predictor, $c = %g$" % c, fontsize=7, color=INK,
                      loc="left")
    axes[0].legend(loc="upper left", fontsize=6.2)
    axes[1].legend(handles=axes[1].get_lines()[-1:], loc="upper left", fontsize=6.2)
    fig.tight_layout(w_pad=0.8)
    return _save(fig, "fock_rule.png")


def fig_scaling() -> Path:
    """(a) Trotter error against the step (trotter_order_study); (b) trajectory
    error against the Fock dimension in the temperature family (fock_rule_study);
    (c) transpiled depth and CX count against the number of steps (phase7_scaling)."""
    _style()
    tr = _load("trotter_order_study")
    fk = _load("fock_rule_study")
    rs = _load("phase7_scaling")
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4))
    ax = axes[0]
    variants = (("stageA_T0", "Stage A, $T = 0$"), ("A1_i_finite_T", r"Stage A, $\bar n = 0.5$"),
                ("stageB", "Stage B, $g = 0.4$"), ("dimer", "dimer"))
    rows = {r["variant"]: r for r in tr["rows"]}
    for (v, lab), col, mk in zip(variants, SERIES, "osD^"):
        r = rows[v]
        ax.loglog([x["dt"] for x in r["per_dt"]], [x["trace_distance_error"] for x in r["per_dt"]],
                  marker=mk, ms=3, color=col, lw=1.0,
                  label=f"{lab} (order {r['fit_smallest']['order']:.2f})")
    ax.set_xlabel(r"Trotter step $\delta t$")
    ax.set_ylabel("max trace-distance error")
    ax.legend(fontsize=5.2, loc="lower right")
    ax.set_title("(a) Trotter error", fontsize=7, color=INK, loc="left")
    ax = axes[1]
    for nth, col in zip((0.5, 1.0, 2.0), ORDINAL[1:]):
        p = next(q for q in fk["points"] if q["family"] == "temperature" and q["param"] == nth)
        dd = sorted(p["per_d"], key=int)
        ax.semilogy([int(k) for k in dd], [p["per_d"][k]["trajectory_dev"] for k in dd],
                    marker="o", ms=2.5, color=col, lw=1.0, label=rf"$\bar n = {nth:g}$")
    ax.set_xlabel("Fock dimension $d$")
    ax.set_ylabel("max trace-distance error")
    ax.legend(fontsize=6)
    ax.set_title("(b) truncation error", fontsize=7, color=INK, loc="left")
    ax = axes[2]
    st = rs["resources_vs_steps"]
    ax.plot([r["n_steps"] for r in st], [r["depth"] for r in st], marker="o", ms=3,
            color=SERIES[0], lw=1.0, label="depth")
    ax.plot([r["n_steps"] for r in st], [r["cx"] for r in st], marker="s", ms=3,
            color=SERIES[1], lw=1.0, label="CX count")
    ax.set_xlabel("Trotter steps")
    ax.set_ylabel("transpiled count")
    ax.legend(fontsize=6)
    ax.set_title("(c) circuit resources", fontsize=7, color=INK, loc="left")
    fig.tight_layout(w_pad=0.6)
    return _save(fig, "scaling.png")


ALL = (fig_estimator_bias, fig_estimator_zsweep, fig_noise_washout_zne, fig_correlated_noise,
       fig_embedding_cx, fig_dimer_populations, fig_phase_diagram_optimized, fig_fock_rule,
       fig_scaling)

if __name__ == "__main__":
    for f in ALL:
        print(f())

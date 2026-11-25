#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Figures of the extended evaluation (experiments A-D), generated only from the run's saved metrics.

    PYTHONPATH=src python scripts/make_extended_eval_figures.py --run paper_artifacts_extended_eval/<run_id>

Writes <run>/figures/*.pdf + .png (PNG previews at 300 dpi). Only this run's own figure files are replaced on re-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dobi.eval.extended import common as C
from dobi.visualization.paper_style import COL_W, FULL_W, NAME, REGIME_LABEL, STYLE, panel_label, use_paper_style

LAB = {**NAME, "labelled_batch": "Labelled batch (needs GT)", "fixed_pristine_val": "Fixed pristine (val)",
       "fixed_pooled_val": "Fixed pooled (val)", "fixed_worst_case_val": "Fixed worst-case (val)",
       "mean_behavior_valcal": "Mean behavior + val. cal.", "adaptive_linear_valcal": "Linear + val. cal.",
       "adaptive_pchip_valcal": "Adaptive PCHIP + val. cal.", "pc1_behavior_valcal": "PC1 + val. cal."}
FAM_COL = {"pristine": "#1f1f1f", "noise": "#a01e28", "blur": "#4664aa", "weather": "#009682", "digital/photometric": "#a07a1e"}
EXTRA = {"labelled_batch": dict(color="#1f1f1f", ls=(0, (4, 2)), lw=1.2, marker="s"),
         "fixed_pristine_val": STYLE["fixed_pristine"], "fixed_pooled_val": dict(color="#d0606a", ls="--", lw=1.1, marker="h"),
         "fixed_worst_case_val": dict(color="#6b1018", ls="-", lw=1.1, marker="s")}


def sty(p):
    if p in STYLE:
        return STYLE[p]
    if p in EXTRA:
        return EXTRA[p]
    if p.endswith("_valcal"):
        s = dict(STYLE[p.replace("_valcal", "")])
        s["ls"] = "-"
        return s
    raise KeyError(p)


def save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(C.new_file(out / f"{name}.{ext}", overwrite_own=True), dpi=300)
    plt.close(fig)


def short(r):
    return REGIME_LABEL.get(r, r.replace("_3_5", "").replace("_", " "))


# ------------------------------------------------------------------ A
def figs_a(run: Path, out: Path):
    S = pd.read_csv(run / "tables/expA_macro_summary.csv")
    M = pd.read_csv(run / "metrics/expA_macro_per_seed.csv")
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.95), constrained_layout=True)
    for ax, (m, lab) in zip(axes, (("mad_roc", "ROC MAD"), ("mad_froc", "FROC MAD"), ("mad_pr", "PR MAD"),
                                   ("abs_tau_error", r"$|\hat\tau-\tau^*_w|$"))):
        for p in ("labelled_batch", "adaptive_linear", "pc1_behavior", "adaptive_pchip"):
            d = S[(S.model == p) & (S.batch_size > 0)].sort_values("batch_size")
            st = sty(p)
            ax.errorbar(d.batch_size, d[m], yerr=d[f"{m}_seed_sd"], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
                        ms=3, capsize=1.5, label=LAB[p])
        mb = S[S.model == "mean_behavior"][m].iloc[0]
        ax.axhline(mb, **{k: v for k, v in sty("mean_behavior").items() if k != "marker"}, label=LAB["mean_behavior"] + " (prior)")
        asy = S[S.model == "adaptive_pchip (all images)"][m].iloc[0]
        ax.axhline(asy, color=STYLE["adaptive_pchip"]["color"], ls=":", lw=1.0, label="PCHIP, all images of the regime")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("batch size b")
        ax.set_title(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.24))
    for ax, k in zip(axes, "abcd"):
        panel_label(ax, k)
    save(fig, out, "figA_error_vs_batch_size")

    # per-regime PR MAD vs batch size (PCHIP vs labelled)
    P = pd.read_csv(run / "metrics/expA_per_seed_regime.csv")
    ref = pd.read_csv(run / "metrics/expA_reference_per_regime.csv")
    regs = list(dict.fromkeys(P.regime))
    fig, axes = plt.subplots(3, 6, figsize=(FULL_W, 3.6), constrained_layout=True, sharex=True)
    for ax, r in zip(axes.flat, regs):
        for p in ("adaptive_pchip", "labelled_batch"):
            d = P[(P.model == p) & (P.regime == r)].groupby("batch_size")[["mad_pr_mean", "mad_pr_sd_between_batches"]].mean()
            st = sty(p)
            ax.plot(d.index, d.mad_pr_mean, color=st["color"], ls=st["ls"], lw=1.0)
            ax.fill_between(d.index, np.maximum(d.mad_pr_mean - d.mad_pr_sd_between_batches, 1e-3), d.mad_pr_mean + d.mad_pr_sd_between_batches,
                            color=st["color"], alpha=0.15, lw=0)
        a = ref[(ref.model == "adaptive_pchip") & (ref.regime == r)].mad_pr.mean()
        mb = ref[(ref.model == "mean_behavior") & (ref.regime == r)].mad_pr.iloc[0]
        ax.axhline(a, color=STYLE["adaptive_pchip"]["color"], ls=":", lw=0.9)
        ax.axhline(mb, color=STYLE["mean_behavior"]["color"], ls="-.", lw=0.9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(short(r), fontsize=7)
        ax.set_xticks([1, 4, 16])
        ax.set_xticklabels(["1", "4", "16"])
        ax.tick_params(labelsize=6)
    fig.supxlabel("batch size b", fontsize=7)
    fig.supylabel("PR MAD (mean ± between-batch sd)", fontsize=7)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], **{k: v for k, v in sty("adaptive_pchip").items() if k != "marker"}),
                Line2D([], [], **{k: v for k, v in sty("labelled_batch").items() if k != "marker"}),
                Line2D([], [], color=STYLE["adaptive_pchip"]["color"], ls=":"), Line2D([], [], color=STYLE["mean_behavior"]["color"], ls="-.")],
               ["Adaptive PCHIP (batch mean of predictions)", "Labelled batch (needs GT)", "PCHIP, all images", "Mean behavior prior"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    save(fig, out, "figA_per_regime_pr_mad")

    # operating-point stability
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2, 1.9), constrained_layout=True)
    for p in ("labelled_batch", "adaptive_linear", "pc1_behavior", "adaptive_pchip"):
        d = S[(S.model == p) & (S.batch_size > 0)].sort_values("batch_size")
        st = sty(p)
        axes[0].plot(d.batch_size, d.sd_tau_hat_between_batches, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3, lw=st["lw"], label=LAB[p])
        axes[1].plot(d.batch_size, d.regret_rcu, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3, lw=st["lw"])
    axes[1].axhline(S[S.model == "mean_behavior"].regret_rcu.iloc[0], **{k: v for k, v in sty("mean_behavior").items() if k != "marker"})
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("batch size b")
    axes[0].set_ylabel(r"sd of $\hat\tau$ across batches")
    axes[1].set_ylabel("regret$^\\ddagger$ (regime-curve utility)")
    axes[0].legend(loc="lower left", fontsize=6)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, out, "figA_operating_point_stability")


# ------------------------------------------------------------------ B
def figs_b(run: Path, out: Path):
    from dobi.eval.extended.exp_b_equivalence import deg_of
    from dobi.visualization.visu_behavior_space import family_of
    Z = np.load(run / "metrics/expB_matrices_and_curves.npz")
    res = json.loads((run / "metrics/expB_results.json").read_text())

    def heat(level, metric, name, size):
        names = list(Z[f"{level}__names"])
        D = Z[f"{level}__{metric}__test"]
        eps = res["levels"][level]["tolerance"][metric]["eps"]
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        im = ax.imshow(D, cmap="viridis_r", vmin=0, vmax=np.percentile(D, 95))
        fams = [family_of(deg_of(n)) for n in names]
        lab = [short(n) if level == "regime" else (n.replace("_s", " s").replace("_", " ") if n != "pristine" else "pristine") for n in names]
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        fs = 6 if level == "regime" else 3.2
        ax.set_xticklabels(lab, rotation=90, fontsize=fs)
        ax.set_yticklabels(lab, fontsize=fs)
        for t, f in zip(ax.get_xticklabels(), fams):
            t.set_color(FAM_COL[f])
        for t, f in zip(ax.get_yticklabels(), fams):
            t.set_color(FAM_COL[f])
        b = [i for i in range(1, len(fams)) if fams[i] != fams[i - 1]]
        for x in b:
            ax.axhline(x - 0.5, color="white", lw=0.8)
            ax.axvline(x - 0.5, color="white", lw=0.8)
        # mark pairs within tolerance on test and train
        Dtr = Z[f"{level}__{metric}__train"]
        ii, jj = np.where((D <= eps) & (Dtr <= eps) & ~np.eye(len(names), dtype=bool))
        cross = [(i, j) for i, j in zip(ii, jj) if fams[i] != fams[j]]
        if cross:
            ax.scatter([j for i, j in cross], [i for i, j in cross], s=6 if level == "regime" else 1.5, marker="o", facecolor="white",
                       edgecolor="black", lw=0.3, label=r"cross-family, $d\leq\varepsilon$ (train and test)")
            ax.set_title("○ = cross-family pair with d ≤ ε on train and test", fontsize=6)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label(f"behavior distance ({metric}, test)", fontsize=7)
        cb.ax.axhline(eps, color="white", lw=1)
        save(fig, out, name)

    heat("regime", "std_l2", "figB_distance_heatmap_regimes", (COL_W * 1.25, COL_W * 1.25))
    heat("severity", "std_l2", "figB_distance_heatmap_severities", (FULL_W, FULL_W * 0.9))

    # GT vs predicted distances
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.75, 2.4), constrained_layout=True)
    GP = pd.DataFrame(res["gt_vs_predicted"])
    for ax, level in zip(axes, ("regime", "severity")):
        names = list(Z[f"{level}__std_l2__pred_names"])
        Dg, Dp = Z[f"{level}__std_l2__gt_test"], Z[f"{level}__std_l2__pred_pchip"]
        iu = np.triu_indices(len(names), 1)
        from dobi.eval.extended.exp_b_equivalence import pair_class
        cls = np.array([pair_class(names[i], names[j]) for i, j in zip(*iu)])
        for c, col, mk in (("cross_family", "#a01e28", "o"), ("within_family", "#4664aa", "^"), ("same_degradation", "#1f1f1f", "s"),
                           ("pristine_pair", "#dca01e", "D")):
            m = cls == c
            if m.any():
                ax.scatter(Dg[iu][m], Dp[iu][m], s=4 if level == "severity" else 9, alpha=0.5, color=col, marker=mk, lw=0,
                           label=c.replace("_", " "))
        mx = max(Dg.max(), Dp.max())
        ax.plot([0, mx], [0, mx], color="0.4", lw=0.7, ls="--")
        g = GP[(GP.level == level) & (GP.metric == "std_l2") & (GP.model == "adaptive_pchip")].iloc[0]
        ax.set_title(f"{level} level ({len(names)} conditions): Spearman {g.spearman_gt_vs_pred:.2f}", fontsize=7)
        ax.set_xlabel("ground-truth behavior distance (test)")
        ax.set_ylabel("predicted behavior distance (PCHIP)")
        eps = res["levels"][level]["tolerance"]["std_l2"]["eps"]
        ax.axvline(eps, color="0.5", lw=0.6, ls=":")
        ax.axhline(eps, color="0.5", lw=0.6, ls=":")
    axes[1].legend(loc="lower right", fontsize=6, markerscale=2)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, out, "figB_gt_vs_predicted_distance")

    # distance classes per metric (normalised by eps)
    L = pd.read_csv(run / "metrics/expB_pairwise_distances.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.0), constrained_layout=True, sharey=True)
    for ax, level in zip(axes, ("regime", "severity")):
        pos, data, cols = [], [], []
        mets = ("std_l2", "std_sup", "paper_rms", "geometric_mad", "decision")
        for k, m in enumerate(mets):
            for j, (c, col) in enumerate((("same_degradation", "#1f1f1f"), ("within_family", "#4664aa"), ("cross_family", "#a01e28"))):
                d = L[(L.level == level) & (L.metric == m) & (L["class"] == c)]
                if len(d):
                    data.append(np.maximum(d.d_test / d.eps, 1e-3))
                    pos.append(k * 4 + j)
                    cols.append(col)
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False, whis=(5, 95), medianprops=dict(color="white", lw=0.8))
        for b, c in zip(bp["boxes"], cols):
            b.set_facecolor(c)
            b.set_alpha(0.8)
        ax.axhline(1.0, color="0.3", lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_xticks([k * 4 + 1 for k in range(len(mets))])
        ax.set_xticklabels([m.replace("_", "\n") for m in mets], fontsize=6)
        ax.set_title(f"{level} level", fontsize=7)
    axes[0].set_ylabel(r"test distance / tolerance $\varepsilon$")
    from matplotlib.patches import Patch
    fig.legend([Patch(color="#1f1f1f"), Patch(color="#4664aa"), Patch(color="#a01e28")],
               ["same degradation (other severity)", "same family", "different family"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    save(fig, out, "figB_distance_by_pair_class")


def fig_b_examples(run: Path, out: Path, images_root: str, db: str, n_pairs: int = 3):
    """Representative functionally similar cross-family pairs: behavior functions + the same base image under both conditions."""
    import sqlite3

    import cv2
    from dobi.eval.extended.exp_b_equivalence import deg_of
    from dobi.visualization.visu_behavior_space import family_of
    EQ = pd.read_csv(run / "tables/expB_shifted_equivalent_pairs.csv")
    Z = np.load(run / "metrics/expB_matrices_and_curves.npz")
    names = list(Z["severity__names"])
    eq = EQ[(EQ.level == "severity") & (EQ.metric == "std_l2")].copy()
    eq["shift"] = np.minimum(eq.a_dist_to_pristine_test, eq.b_dist_to_pristine_test)
    eq = eq[eq.d_test <= 0.5 * eq.eps].sort_values("shift", ascending=False)
    title = "most strongly shifted cross-family pairs with d <= eps/2 (test) and d <= eps (train)"
    picked, used = [], set()
    for _, r in eq.iterrows():
        if r.a in used or r.b in used or {family_of(deg_of(r.a)), family_of(deg_of(r.b))} in [set(map(lambda x: family_of(deg_of(x)), (q.a, q.b))) for q in picked]:
            continue
        picked.append(r)
        used |= {r.a, r.b}
        if len(picked) == n_pairs:
            break
    tau = np.linspace(0, 1, 201)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    base = conn.execute("""SELECT v.image_id FROM variants v JOIN images i ON i.id = v.image_id
                           WHERE i.split='test' AND v.deg_name='pristine' ORDER BY v.image_id LIMIT 40""").fetchall()
    gt = dict(conn.execute("""SELECT i.id, COUNT(g.id) FROM images i LEFT JOIN gt_objects g ON g.image_id=i.id AND g.class_id=0
                              WHERE i.split='test' GROUP BY i.id""").fetchall())
    img_id = next(b[0] for b in base if 2 <= gt.get(b[0], 0) <= 4)

    def thumb(cond):
        d, s = (cond.rsplit("_s", 1)[0], int(cond.rsplit("_s", 1)[1]))
        rp = conn.execute("SELECT rel_path FROM variants WHERE image_id=? AND deg_name=? AND severity=?", (img_id, d, s)).fetchone()[0]
        im = cv2.imread(str(Path(images_root) / rp))[:, :, ::-1]
        h, w = im.shape[:2]
        c = min(h, w)
        return im[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]

    fig, axes = plt.subplots(len(picked), 6, figsize=(FULL_W, 1.25 * len(picked) + 0.3), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[0.8, 0.8, 1, 1, 1, 1]))
    axes = np.atleast_2d(axes)
    from dobi.visualization.visu_behavior_space import family_of
    from dobi.eval.extended.exp_b_equivalence import deg_of
    for row, r in zip(axes, picked):
        for k, cond in enumerate((r.a, r.b)):
            row[k].imshow(thumb(cond))
            row[k].set_axis_off()
            row[k].set_title(f"{cond.replace('_s', ' sev ').replace('_', ' ')}\n({family_of(deg_of(cond))})", fontsize=6)
        i, j = names.index(r.a), names.index(r.b)
        for ax, key, lab in zip(row[2:], ("tpr", "precision", "fppi", "f1"), ("TPR", "Precision", "FPPI", "$F_1$")):
            for idx, ls, col in ((i, "-", "#1f1f1f"), (j, "--", "#a01e28")):
                for sp, alpha in (("test", 1.0),):
                    y = Z[f"severity__{sp}_{key}"][idx]
                    ax.plot(tau, y, ls=ls, color=col, lw=1.1, alpha=alpha)
            if key == "fppi":
                ax.set_yscale("symlog", linthresh=0.1)
            ax.set_xlim(0, 0.8)
            ax.set_title(lab, fontsize=7)
            ax.tick_params(labelsize=6)
        row[2].text(0.02, 0.04, f"d = {r.d_test:.2f} (eps {r.eps:.2f})", transform=row[2].transAxes, fontsize=6)
    for ax in axes[-1, 2:]:
        ax.set_xlabel(r"threshold $\tau$")
    fig.suptitle(title + "; solid = left image condition, dashed = right", fontsize=7)
    conn.close()
    save(fig, out, "figB_functionally_similar_examples")


# ------------------------------------------------------------------ C
def figs_c(run: Path, out: Path):
    S = pd.read_csv(run / "tables/expC_summary.csv")
    pols = ["fixed_pristine_val", "fixed_pooled_val", "fixed_worst_case_val", "mean_behavior", "adaptive_linear", "pc1_behavior",
            "adaptive_pchip", "mean_behavior_valcal", "adaptive_pchip_valcal", "regime_id_validation", "regime_oracle"]
    budgets = sorted(S.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(FULL_W, 2.0), constrained_layout=True)
    for ax, B in zip(axes, budgets):
        for p in pols:
            d = S[(S.budget == B) & (S.policy == p)]
            if not len(d):
                continue
            st = sty(p)
            fc = st["color"] if p.endswith("_valcal") or p.startswith("fixed") or p == "regime_oracle" else "white"
            ax.errorbar(d.n_violations, d.macro_recall, xerr=[d.n_violations - d.n_violations_ci_lo, d.n_violations_ci_hi - d.n_violations],
                        fmt=st["marker"] or "o", color=st["color"], mfc=fc, ms=4.5, elinewidth=0.6, capsize=1, label=LAB[p])
        ax.set_title(f"FPPI budget B = {B:g}")
        ax.set_xlabel("# regimes FPPI > B")
        ax.set_xlim(-0.8, 18.8)
    axes[0].set_ylabel("macro recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.27))
    for ax, k in zip(axes, "abcd"):
        panel_label(ax, k)
    save(fig, out, "figC_recall_vs_violations")

    P = pd.read_csv(run / "metrics/expC_per_seed_regime.csv")
    P = P[P.regime != "ALL_micro"]
    B = 0.1
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.4), constrained_layout=True, sharex=True)
    show = ["fixed_worst_case_val", "fixed_pooled_val", "adaptive_pchip", "adaptive_pchip_valcal", "regime_id_validation", "regime_oracle"]
    d0 = P[(P.budget == B) & (P.policy == "adaptive_pchip")].groupby("regime")["fppi"].mean().sort_values()
    x = np.arange(len(d0))
    w = 0.8 / len(show)
    for k, p in enumerate(show):
        d = P[(P.budget == B) & (P.policy == p)].groupby("regime")[["fppi", "recall"]].mean().reindex(d0.index)
        st = sty(p)
        hatch = "////" if p in ("regime_id_validation", "regime_oracle") else None
        for ax, m in zip(axes, ("fppi", "recall")):
            ax.bar(x + (k - (len(show) - 1) / 2) * w, d[m], w * 0.92, color=st["color"], hatch=hatch, edgecolor="white", lw=0.3,
                   label=LAB[p] if m == "fppi" else None)
    axes[0].axhline(B, color="black", lw=0.9, ls="--", label=f"budget B = {B:g}")
    axes[0].set_ylabel("observed FPPI")
    axes[1].set_ylabel("recall")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([short(r) for r in d0.index], rotation=40, ha="right")
    axes[0].legend(ncol=4, fontsize=6, loc="upper left")
    axes[1].set_ylim(0, 1)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    save(fig, out, "figC_per_regime_B0p1")

    PV = pd.read_csv(run / "tables/expC_predicted_vs_observed_fppi.csv")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.9), constrained_layout=True)
    for p, mk in (("adaptive_pchip", "o"), ("adaptive_pchip_valcal", "s"), ("mean_behavior", "D")):
        d = PV[PV.policy == p]
        ax.scatter(d.predicted_fppi_mean, d.fppi, s=8, marker=mk, color=sty(p)["color"], alpha=0.7, label=LAB[p],
                   facecolor="white" if p == "adaptive_pchip" else sty(p)["color"])
    for B in budgets:
        ax.axhline(B, color="0.75", lw=0.5, ls=":")
    lim = [0.01, 1]
    ax.plot(lim, lim, color="0.3", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted regime FPPI at the selected thresholds")
    ax.set_ylabel("observed regime FPPI (execution)")
    ax.legend(fontsize=6, loc="upper left")
    save(fig, out, "figC_predicted_vs_observed_fppi")


# ------------------------------------------------------------------ D
def figs_d(run: Path, out: Path):
    R = pd.read_csv(run / "metrics/expD_per_seed_regime.csv")
    FL = pd.read_csv(run / "metrics/expD_empirical_floor_per_regime.csv")
    FL = FL[FL.density == "rho_train"].set_index("regime")
    reg = R.groupby(["model", "regime"]).mean(numeric_only=True).reset_index()
    order = reg[reg.model == "adaptive_pchip"].sort_values("eP_tau08").regime.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    x = np.arange(len(order))
    for k, p in enumerate(("adaptive_linear", "adaptive_pchip", "pc1_behavior")):
        d = reg[reg.model == p].set_index("regime").reindex(order)
        axes[0].bar(x + (k - 1) * 0.27, d.eP_tau08, 0.25, color=sty(p)["color"], label=LAB[p])
    axes[0].plot(x, FL.reindex(order).eP_tau08, "k_", ms=8, mew=1.2, label=r"empirical curves with $\rho_{train}$ (floor)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([short(r) for r in order], rotation=60, ha="right", fontsize=6)
    axes[0].set_ylabel("precision consistency error")
    axes[0].legend(fontsize=6, loc="upper left")
    for ax, (xm, ym, xl, yl) in zip(axes[1:], (("eP_tau08", "mad_pr", "precision consistency error", "PR MAD (per regime)"),
                                              ("eF_knot1", "mad_pr", "FPR–FPPI consistency error", "PR MAD (per regime)"))):
        for p in ("adaptive_linear", "adaptive_pchip", "pc1_behavior"):
            d = reg[reg.model == p]
            st = sty(p)
            ax.scatter(d[xm], d[ym], s=10, marker=st["marker"], color=st["color"], alpha=0.8, label=LAB[p])
        ax.set_xlabel(xl, fontsize=7)
        ax.set_ylabel(yl, fontsize=7)
    for ax, k in zip(axes, "abc"):
        panel_label(ax, k)
    save(fig, out, "figD_internal_consistency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--images_root", default=C.TEST_IMAGES)
    ap.add_argument("--db", default=C.DB)
    ap.add_argument("--out", default=None, help="figure folder (default: <run>/figures)")
    a = ap.parse_args()
    run = C.check_out_dir(Path(a.run))
    out = Path(a.out) if a.out else run / "figures"
    use_paper_style()
    if "A" in a.only and (run / "tables/expA_macro_summary.csv").exists():
        figs_a(run, out)
    if "B" in a.only and (run / "metrics/expB_results.json").exists():
        figs_b(run, out)
        fig_b_examples(run, out, a.images_root, a.db)
    if "C" in a.only and (run / "tables/expC_summary.csv").exists():
        figs_c(run, out)
    if "D" in a.only and (run / "metrics/expD_per_seed_regime.csv").exists():
        figs_d(run, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()

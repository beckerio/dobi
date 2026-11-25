#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Generate every quantitative DOBI paper figure from the final result files.

    PYTHONPATH=src python scripts/make_paper_figures.py \
        --paper_results paper_artifacts --execution_results paper_artifacts_execution \
        --out paper_submission/figures --all

Inputs (never hard-coded metric values):
  paper_artifacts/curve_exports/<model>/<seed>/<id>_<regime>.npz   regime-level GT and mean image-conditioned predictions
  paper_artifacts/aggregate/behavior_space_*                        behavior-space analysis (PCA fitted on train regimes)
  paper_artifacts_execution/execution_summary.json                  thresholds and per-regime true-execution metrics
  paper_artifacts_execution/per_image/policy_seed_regime_results_long.csv   per-seed execution metrics
  paper_artifacts_execution/tables/adaptation_count_changes.csv, tau_distribution_by_regime.csv
  paper_artifacts_execution/diagnostics/fog_execution_comparison.csv
Thumbnails for the TikZ pipeline figure (--thumbnails) are cropped from the local test images.
"""
from __future__ import annotations
from dobi.paths import P as _P  # data locations (configs/paths.yaml or $DOBI_DATA_ROOT)

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dobi.visualization.paper_style import (COL_W, FULL_W, NAME, REGIME_LABEL, STYLE, line_kw, panel_label, save,
                                            use_paper_style)

MAIN_REGIMES = ["pristine", "gaussian_noise_3_5", "glass_blur_3_5", "fog_3_5"]
SUPP_REGIMES = ["pristine", "gaussian_noise_3_5", "glass_blur_3_5", "pixelate_3_5", "motion_blur_3_5", "fog_3_5"]
SEEDS = (0, 1, 2)


# ------------------------------------------------------------------
class Results:
    def __init__(self, pa: Path, pe: Path):
        self.pa, self.pe = pa, pe
        self.exec = json.loads((pe / "execution_summary.json").read_text())
        self.long = pd.read_csv(pe / "per_image/policy_seed_regime_results_long.csv")
        self._curves = None
        self.regimes = list(self.exec["per_regime"]["adaptive_pchip"].keys())

    @property
    def curves(self):
        """Regime-level curves from paper_artifacts/curve_exports (*.npz), loaded on first use."""
        if self._curves is None:
            pa, curves = self.pa, {}
            for model in ("adaptive_pchip", "adaptive_linear", "pc1_behavior"):
                per_seed = [self._load_dir(pa / "curve_exports" / model / f"seed{s}") for s in SEEDS]
                curves[model] = {r: {k: np.mean([ps[r][k] for ps in per_seed], axis=0) for k in per_seed[0][r]}
                                 for r in per_seed[0]}
            curves["mean_behavior"] = self._load_dir(pa / "curve_exports/mean_behavior/deterministic")
            self._curves = curves
        return self._curves

    @staticmethod
    def _load_dir(d: Path):
        out = {}
        files = sorted(d.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no curve exports (*.npz) in {d}")
        for f in files:
            z = np.load(f)
            name = f.stem.split("_", 1)[1]
            out[name] = {k: z[k] for k in z.files if z[k].dtype.kind == "f" and z[k].ndim == 1}
        return out

    def tau(self, policy, regime):
        t = self.exec["thresholds"]
        if policy == "fixed_pristine":
            return t["tau_pristine_validation"]
        if policy == "mean_behavior":
            return t["tau_mean_behavior"]
        if policy == "regime_oracle":
            return t["tau_regime_oracle_test"][regime]
        if policy == "regime_id_validation":
            return t["tau_regime_validation"][regime]
        return self.exec["per_regime"][policy][regime]["tau_mean"]


# ------------------------------------------------------------------
# main figures
# ------------------------------------------------------------------
def fig_behavior_space(R: Results, out: Path):
    summ = json.loads((R.pa / "aggregate/behavior_space_summary.json").read_text())
    reg = pd.read_csv(R.pa / "aggregate/behavior_space_per_regime.csv")
    fam_style = {"blur": ("#4664aa", "-"), "noise": ("#a01e28", "--"), "weather": ("#009682", "-."),
                 "digital/photometric": ("#a07a1e", ":")}
    from dobi.visualization.visu_behavior_space import family_of
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.45), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    for d, r in summ["severity"]["degradations"].items():
        col, ls = fam_style[family_of(d)]
        ax.plot([0] + r["severity"], [0] + r["distance"], ls=ls, color=col, lw=1.0, marker="o", ms=2.2, alpha=0.9)
    for fam, (col, ls) in fam_style.items():
        ax.plot([], [], color=col, ls=ls, lw=1.2, label=fam)
    ax.set_xlabel("Degradation severity (0 = pristine)")
    ax.set_ylabel("RMS behavior distance to pristine")
    ax.set_xticks(range(6))
    ax.legend(loc="upper left", ncol=1)
    panel_label(ax, "a")

    ax = axes[1]
    markers = {"train": "o", "val": "s", "test": "^"}
    for _, e in reg.iterrows():
        col, _ = fam_style.get(family_of(e["regime"]), ("#1f1f1f", "-")) if e["regime"] != "pristine" else ("#1f1f1f", "-")
        ax.scatter(e["pc1"], e["pc2"], marker=markers[e["split"]], s=16, color=col, edgecolor="white", linewidth=0.4, zorder=3)
    for name in ("pristine", "fog_3_5", "gaussian_noise_3_5", "glass_blur_3_5", "pixelate_3_5", "motion_blur_3_5"):
        e = reg[(reg.regime == name) & (reg.split == "test")].iloc[0]
        ax.annotate(REGIME_LABEL[name], (e["pc1"], e["pc2"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    for sp, m in markers.items():
        ax.scatter([], [], marker=m, color="0.4", s=16, label=sp)
    ev = summ["pca"]["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of train-regime variance)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
    ax.legend(loc="lower left", title="split", title_fontsize=7)
    panel_label(ax, "b")
    save(fig, out, "fig_behavior_space")


def fig2_pr(R: Results, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.95), constrained_layout=True, sharey=True)
    for k, (ax, r) in enumerate(zip(axes, MAIN_REGIMES)):
        g = R.curves["adaptive_pchip"][r]
        mb = R.curves["mean_behavior"][r]
        ax.plot(g["tpr_gt"], g["precision_gt"], label=NAME["gt"], **line_kw("gt"))
        ax.plot(mb["tpr_pred"], mb["precision_pred"], label=NAME["mean_behavior"], **line_kw("mean_behavior"))
        ax.plot(g["tpr_pred"], g["precision_pred"], label="Adaptive PCHIP (regime mean)", **line_kw("adaptive_pchip"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_title(REGIME_LABEL[r])
        panel_label(ax, "abcd"[k])
    axes[0].set_ylabel("Precision")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    save(fig, out, "fig2_pr_behavior")


def fig3_f1(R: Results, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.0), constrained_layout=True, sharey=True)
    for k, (ax, r) in enumerate(zip(axes, MAIN_REGIMES)):
        g = R.curves["adaptive_pchip"][r]
        tau = R.curves["adaptive_pchip"][r]["tau"]
        ax.plot(tau, g["f1_gt"], label=NAME["gt"], **line_kw("gt"))
        ax.plot(tau, g["f1_pred"], label="Adaptive PCHIP (regime mean)", **line_kw("adaptive_pchip"))
        for pol, lab in (("fixed_pristine", "Fixed pristine $\\tau$"), ("adaptive_pchip", "Adaptive PCHIP $\\hat\\tau$ (mean)"),
                         ("regime_oracle", "Regime oracle $\\tau$")):
            kw = line_kw(pol, lw=1.2)
            if pol == "adaptive_pchip":
                kw["ls"] = (0, (6, 2))
            ax.axvline(R.tau(pol, r), label=lab, **kw)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Detector threshold $\\tau$")
        ax.set_title(REGIME_LABEL[r])
        panel_label(ax, "abcd"[k])
    axes[0].set_ylabel("$F_1$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.11))
    save(fig, out, "fig3_f1_vs_tau")


def fig4_gain(R: Results, out: Path):
    L = R.long[~R.long.regime.str.startswith("ALL")]
    g = L[L.policy == "adaptive_pchip"].groupby("regime")["execution_gain"].agg(["mean", "std"])
    mb = L[L.policy == "mean_behavior"].groupby("regime")["execution_gain"].mean()
    orc = L[L.policy == "regime_oracle"].groupby("regime")["execution_gain"].mean()
    g = g.sort_values("mean")
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(FULL_W, 2.25), constrained_layout=True)
    bars = ax.bar(x, g["mean"], width=0.62, color=STYLE["adaptive_pchip"]["color"], yerr=g["std"], capsize=1.5,
                  error_kw=dict(lw=0.6), label="Adaptive PCHIP (mean $\\pm$ sd, 3 seeds; hatched = loss)", zorder=2)
    for b, v in zip(bars, g["mean"]):
        if v < 0:
            b.set_hatch("////")
            b.set_edgecolor("white")
    ax.plot(x, mb.reindex(g.index), ls="none", marker="D", ms=3.2, mfc="white", mec=STYLE["mean_behavior"]["color"],
            mew=0.9, label=NAME["mean_behavior"], zorder=3)
    ax.plot(x, orc.reindex(g.index), ls="none", marker="*", ms=5, color=STYLE["regime_oracle"]["color"],
            mec="#8a5d00", mew=0.3, label=NAME["regime_oracle"] + " (non-deployable)", zorder=3)
    ax.axhline(0, color="0.2", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL[r] for r in g.index], rotation=40, ha="right")
    ax.set_ylabel("$\\Delta F_1$ vs. fixed pristine")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, out, "fig4_gain_by_regime")


# ------------------------------------------------------------------
# supplementary figures
# ------------------------------------------------------------------
def figS_full_behavior(R: Results, out: Path):
    for r in SUPP_REGIMES:
        g, li, mb = R.curves["adaptive_pchip"][r], R.curves["adaptive_linear"][r], R.curves["mean_behavior"][r]
        tau = g["tau"]
        fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.6), constrained_layout=True)
        panels = [("ROC", "fpr", "tpr", "FPR", "TPR"), ("FROC", "fppi", "tpr", "FPPI", "TPR"),
                  ("PR", "tpr", "precision", "Recall", "Precision")]
        for k, (ax, (title, xk, yk, xl, yl)) in enumerate(zip(axes[0, :3], panels)):
            ax.plot(g[f"{xk}_gt"], g[f"{yk}_gt"], label=NAME["gt"], **line_kw("gt"))
            ax.plot(mb[f"{xk}_pred"], mb[f"{yk}_pred"], label=NAME["mean_behavior"], **line_kw("mean_behavior"))
            ax.plot(li[f"{xk}_pred"], li[f"{yk}_pred"], label=NAME["adaptive_linear"], **line_kw("adaptive_linear"))
            ax.plot(g[f"{xk}_pred"], g[f"{yk}_pred"], label="Adaptive PCHIP", **line_kw("adaptive_pchip"))
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(title)
            if title == "FROC":
                ax.set_xlim(0, 2.0)  # FROC evaluation domain; curves are not extended
            panel_label(ax, "abcdefgh"[k])
        tpanels = [(axes[0, 3], "f1", "$F_1$"), (axes[1, 0], "tpr", "TPR"), (axes[1, 1], "fpr", "FPR"),
                   (axes[1, 2], "fppi", "FPPI"), (axes[1, 3], "precision", "Precision")]
        for k, (ax, key, lab) in enumerate(tpanels):
            m, s = g[f"{key}_pred"], g[f"{key}_pred_std"]
            ax.fill_between(tau, np.clip(m - s, 0, None), m + s, color=STYLE["adaptive_pchip"]["color"], alpha=0.15, lw=0)
            ax.plot(tau, g[f"{key}_gt"], **line_kw("gt"))
            ax.plot(tau, mb[f"{key}_pred"], **line_kw("mean_behavior"))
            ax.plot(tau, li[f"{key}_pred"], **line_kw("adaptive_linear"))
            ax.plot(tau, m, **line_kw("adaptive_pchip"))
            if key == "f1":
                for pol in ("fixed_pristine", "adaptive_pchip", "regime_oracle"):
                    ax.axvline(R.tau(pol, r), **line_kw(pol, lw=1.0))
            if key == "fppi":
                ax.set_yscale("symlog", linthresh=0.1)
                ax.set_ylim(bottom=0)
            ax.set_xlim(0, 1)
            ax.set_xlabel("$\\tau$")
            ax.set_ylabel(lab)
            panel_label(ax, "defgh"[k])
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
        fig.suptitle(REGIME_LABEL[r], fontsize=8, fontweight="bold")
        save(fig, out, f"figS_full_behavior_{r}")


def figS_tradeoff(R: Results, out: Path):
    c = pd.read_csv(R.pe / "tables/adaptation_count_changes.csv").sort_values("delta_f1")
    y = np.arange(len(c))
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.9), constrained_layout=True, sharey=True)
    for k, (ax, col, lab, role) in enumerate(zip(axes, ("delta_recall", "delta_precision", "delta_fppi", "delta_f1"),
                                                 ("$\\Delta$ Recall", "$\\Delta$ Precision", "$\\Delta$ FPPI", "$\\Delta F_1$"),
                                                 ("adaptive_pchip", "fixed_pristine", "fixed_pristine", "gt"))):
        ax.barh(y, c[col], color=STYLE[role]["color"], height=0.65)
        ax.axvline(0, color="0.2", lw=0.6)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
        panel_label(ax, "abcd"[k])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([REGIME_LABEL[r] for r in c["regime"]])
    save(fig, out, "figS_precision_recall_tradeoff")


def figS_fog(R: Results, out: Path):
    f = pd.read_csv(R.pe / "diagnostics/fog_execution_comparison.csv").set_index("policy")
    pols = ["fixed_pristine", "mean_behavior", "adaptive_linear", "adaptive_pchip", "pc1_behavior", "regime_oracle"]
    f = f.loc[pols]
    x = np.arange(len(pols))
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.35), constrained_layout=True)
    ax = axes[0]
    g = R.curves["adaptive_pchip"]["fog_3_5"]
    tau = g["tau"]
    m = (tau >= 0.1) & (tau <= 0.4)
    ax.plot(tau[m], g["f1_gt"][m], **line_kw("gt"))
    for p in pols:
        t = R.tau(p, "fog_3_5")
        j = int(np.argmin(np.abs(tau - t)))
        ax.plot(t, g["f1_gt"][j], ls="none", marker=STYLE[p]["marker"], ms=5, color=STYLE[p]["color"], label=NAME[p])
    ax.set_xlabel("$\\tau$ (mean selected)")
    ax.set_ylabel("Empirical regime $F_1$")
    ax.legend(loc="lower center", fontsize=6)
    panel_label(ax, "a")
    ax = axes[1]
    ax.bar(x - 0.2, f["tp"], 0.38, color=STYLE["adaptive_pchip"]["color"], label="TP")
    ax.bar(x + 0.2, f["fp"], 0.38, color=STYLE["fixed_pristine"]["color"], hatch="///", edgecolor="white", label="FP")
    ax.set_ylabel("Count (4,950 fog variants)")
    ax.set_ylim(0, f["tp"].max() * 1.28)
    ax.legend(loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(0.56, 1.0))
    panel_label(ax, "b")
    ax = axes[2]
    ax.plot(x, f["precision"], marker="s", ls="--", color=STYLE["fixed_pristine"]["color"], label="Precision")
    ax.plot(x, f["recall"], marker="o", ls="-", color=STYLE["adaptive_pchip"]["color"], label="Recall")
    ax.set_ylabel("Precision / recall")
    ax.set_ylim(min(f["recall"].min(), f["precision"].min()) - 0.02, f["precision"].max() + 0.06)
    ax.legend(loc="upper center", ncol=2, handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(0.56, 1.0))
    panel_label(ax, "c")
    ax = axes[3]
    ax.bar(x, f["execution_gain"], color=[STYLE[p]["color"] for p in pols])
    ax.axhline(0, color="0.2", lw=0.6)
    ax.set_ylabel("$\\Delta F_1$ vs. fixed pristine")
    panel_label(ax, "d")
    for a in axes[1:]:
        a.set_xticks(x)
        a.set_xticklabels([NAME[p] for p in pols], rotation=50, ha="right", fontsize=6.5)
        a.grid(axis="x", visible=False)
    save(fig, out, "figS_fog_failure")


def figS_tau_distribution(R: Results, out: Path):
    td = pd.read_csv(R.pe / "tables/tau_distribution_by_regime.csv")
    td = td[td.seed == 0]
    regs = R.regimes
    x = np.arange(len(regs))
    fig, ax = plt.subplots(figsize=(FULL_W, 2.6), constrained_layout=True)
    for i, pol in enumerate(("adaptive_linear", "adaptive_pchip", "pc1_behavior")):
        d = td[td.policy == pol].set_index("regime").reindex(regs)
        xo = x + (i - 1) * 0.26
        st = STYLE[pol]
        ax.vlines(xo, d["p05"], d["p95"], color=st["color"], lw=0.8)
        ax.bar(xo, d["p75"] - d["p25"], bottom=d["p25"], width=0.22, color=st["color"], alpha=0.75,
               label=f"{NAME[pol]} (seed 0; box 25-75%, line 5-95%)")
        ax.hlines(d["median"], xo - 0.11, xo + 0.11, color="black", lw=0.8)
    ax.plot(x, [R.tau("regime_oracle", r) for r in regs], ls="none", marker="*", ms=6, color=STYLE["regime_oracle"]["color"],
            mec="#8a5d00", mew=0.3, label=NAME["regime_oracle"])
    ax.plot(x, [R.tau("regime_id_validation", r) for r in regs], ls="none", marker="x", ms=4,
            color=STYLE["regime_id_validation"]["color"], label=NAME["regime_id_validation"] + " (validation)")
    ax.axhline(R.tau("fixed_pristine", None), label=NAME["fixed_pristine"], **line_kw("fixed_pristine"))
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL[r] for r in regs], rotation=40, ha="right")
    ax.set_ylabel("Selected threshold $\\hat\\tau_i$")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.18))
    save(fig, out, "figS_tau_distribution")


def figS_regret(R: Results, out: Path):
    L = R.long[~R.long.regime.str.startswith("ALL")]
    pols = ["fixed_pristine", "mean_behavior", "pc1_behavior", "adaptive_pchip", "pchip_regime_mean_tau", "regime_id_validation"]
    st = L[L.policy.isin(pols)].groupby(["policy", "regime"])["execution_regret"].mean().unstack(0)
    st = st.sort_values("fixed_pristine")
    y = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(FULL_W * 0.62, 3.2), constrained_layout=True)
    for p in pols:
        ax.plot(st[p], y, ls="none", marker=STYLE[p]["marker"], ms=4, color=STYLE[p]["color"],
                mfc="white" if p in ("mean_behavior", "pchip_regime_mean_tau") else STYLE[p]["color"], label=NAME[p])
    ax.set_yticks(y)
    ax.set_yticklabels([REGIME_LABEL[r] for r in st.index])
    ax.set_xlabel("Execution regret vs. regime oracle ($F_1$)")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, out, "figS_execution_regret")


# ------------------------------------------------------------------
def thumbnails(out: Path, db: str, images_root: Path, n: int = 3, size: int = 220):
    """Deterministic thumbnails for the pipeline figure: first n test base images (by id) with 2-4 persons."""
    import sqlite3
    import cv2
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ids = [r[0] for r in conn.execute(
        """SELECT g.image_id FROM gt_objects g JOIN images i ON i.id = g.image_id WHERE i.split = 'test' AND g.class_id = 0
           GROUP BY g.image_id HAVING COUNT(*) BETWEEN 2 AND 4 ORDER BY g.image_id LIMIT ?""", (n,))]
    spec = {"pristine": ("pristine", 0), "noise": ("gaussian_noise", 4), "blur": ("defocus_blur", 4), "fog": ("fog", 4)}
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, (deg, sev) in spec.items():
        for k, iid in enumerate(ids):
            rel = conn.execute("SELECT rel_path FROM variants WHERE image_id = ? AND deg_name = ? AND severity = ?",
                               (iid, deg, sev)).fetchone()[0]
            im = cv2.imread(str(images_root / rel))
            h, w = im.shape[:2]
            s = min(h, w)
            im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            cv2.imwrite(str(out / f"{name}_{k}.jpg"), cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            meta[f"{name}_{k}"] = {"image_id": iid, "rel_path": rel}
    (out / "thumbnails.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_results", default='results/main_study/paper_artifacts')
    ap.add_argument("--execution_results", default='results/main_study/paper_artifacts_execution')
    ap.add_argument("--out", default='outputs/figures')
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--main", action="store_true")
    g.add_argument("--supplement", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--thumbnails", action="store_true", help="also (re)create pipeline thumbnails (needs DB and images)")
    ap.add_argument("--db", default=_P('dobi_db'))
    ap.add_argument("--test_images", default=_P('coco_mini_person_d', 'data/test2017/images'))
    a = ap.parse_args()
    use_paper_style()
    R = Results(Path(a.paper_results), Path(a.execution_results))
    out = Path(a.out)
    do_main = a.main or a.all or not a.supplement
    do_supp = a.supplement or a.all or not a.main
    if a.thumbnails:
        thumbnails(out / "main/regimes", a.db, Path(a.test_images))
    skipped = []

    def run(fn, sub):
        try:
            fn(R, out / sub)
        except FileNotFoundError as e:
            skipped.append(fn.__name__)
            print(f"skipped {fn.__name__}: missing input ({e})")

    if do_main:
        for fn in (fig_behavior_space, fig2_pr, fig3_f1, fig4_gain):
            run(fn, "main")
    if do_supp:
        for fn in (figS_full_behavior, figS_tradeoff, figS_fog, figS_tau_distribution, figS_regret):
            run(fn, "supplement")
    print("figures written to", out, "| skipped:", ", ".join(skipped) if skipped else "none")
    return 3 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())

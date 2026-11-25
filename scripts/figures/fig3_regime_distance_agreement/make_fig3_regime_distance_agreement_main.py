#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Figure 3 of the DOBI main paper (fig:equivalence): empirical vs PCHIP-predicted pairwise behavior distances, 18 regimes.

Provenance: the released main-paper asset figures/extended/figB_regime_distance_agreement_main.pdf (DOBI_PAPER_FEEDBACK_READY_2026-10-14)
is pixel-identical (600 dpi) to the left 189.955 pt of the supplement figure figures/extended/figB_gt_vs_predicted_distance.pdf, which is
produced by the 'GT vs predicted distances' block of scripts/make_extended_eval_figures.py (matplotlib 3.10.8, DOBI paper style).
This script contains that block verbatim (inputs read from figB_regime_distance_agreement_data.npz, an exact extract of the run's
expB metrics) and the DOBI paper rcParams, so it is self-contained.

  python make_fig3_regime_distance_agreement_main.py supplement-check OUT.pdf  -> reproduces the two-panel supplement figure
  python make_fig3_regime_distance_agreement_main.py main OUT.pdf              -> Figure 3 with legend

'main' (legend-only change): the two-panel figure is laid out exactly as in the supplement (original save path; matplotlib's own tight
bbox is recorded); afterwards the layout is frozen, panel (b) is removed and a frameless inset legend (DOBI convention, as in the
supplement figure) with the categories present at the regime level is placed in the empty upper-left area of the plot. The saved
region is exactly the original crop (tight-bbox left edge, width 189.955 pt, full height), so the figure size is unchanged and the
plot (data, markers, colors, axes, limits, title, Spearman statistic, guides, equality line) is pixel-identical outside the legend.
(A legend strip below the plot was tried first; it enlarged the figure and reflowed the paper beyond 8 content pages.)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
FULL_W = 7.0
CROP_W_PT = 189.955            # width of the released main-paper crop (left panel of the supplement figure)
INSET_DX = 0.12                # inset legend: left edge this far (data units) right of the vertical tolerance guide
INSET_TOP_Y = 3.62             # inset legend: top edge (data units); the regime panel has no points above y = 2.9 for x < 2.4
LEGEND_FONTSIZE = 6            # = legend font size of the supplement figure
# category order, colors, markers and labels exactly as in the supplement figure (scripts/make_extended_eval_figures.py)
CATS = (("cross_family", "#a01e28", "o"), ("within_family", "#4664aa", "^"), ("same_degradation", "#1f1f1f", "s"), ("pristine_pair", "#dca01e", "D"))


def use_paper_style():         # identical to dobi.visualization.paper_style.use_paper_style()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.solid_capstyle": "round"})


def panel_label(ax, letter, x=-0.02, y=1.02):   # identical to dobi.visualization.paper_style.panel_label
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def build():
    D = np.load(HERE / "figB_regime_distance_agreement_data.npz")
    # ---- verbatim block of scripts/make_extended_eval_figures.py (inputs from the extract) ----
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W * 0.75, 2.4), constrained_layout=True)
    for ax, level in zip(axes, ("regime", "severity")):
        names = list(D[f"{level}__names"])
        Dg, Dp = D[f"{level}__gt_test"], D[f"{level}__pred_pchip"]
        iu = np.triu_indices(len(names), 1)
        cls = D[f"{level}__pair_class_upper"]
        for c, col, mk in CATS:
            m = cls == c
            if m.any():
                ax.scatter(Dg[iu][m], Dp[iu][m], s=4 if level == "severity" else 9, alpha=0.5, color=col, marker=mk, lw=0,
                           label=c.replace("_", " "))
        mx = max(Dg.max(), Dp.max())
        ax.plot([0, mx], [0, mx], color="0.4", lw=0.7, ls="--")
        ax.set_title(f"{level} level ({len(names)} conditions): Spearman {float(D[f'{level}__spearman']):.2f}", fontsize=7)
        ax.set_xlabel("ground-truth behavior distance (test)")
        ax.set_ylabel("predicted behavior distance (PCHIP)")
        eps = float(D[f"{level}__eps"])
        ax.axvline(eps, color="0.5", lw=0.6, ls=":")
        ax.axhline(eps, color="0.5", lw=0.6, ls=":")
    axes[1].legend(loc="lower right", fontsize=6, markerscale=2)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    # ---- end verbatim ----
    return fig, axes


def main_figure(out):
    import io
    fig, axes = build()
    # 1) run the ORIGINAL save path once (pdf, dpi=300, bbox='tight' as for the supplement figure): this executes constrained layout
    #    with the PDF renderer and lets matplotlib compute the tight bbox; the bbox it uses is recorded, not recomputed.
    rec = {}
    orig = fig.get_tightbbox
    def recording_tightbbox(*a, **k):
        bb = orig(*a, **k)
        rec["bb"] = bb.frozen()
        return bb
    fig.get_tightbbox = recording_tightbbox
    fig.savefig(io.BytesIO(), format="pdf", dpi=300)
    del fig.get_tightbbox
    tb = rec["bb"]
    pad = mpl.rcParams["savefig.pad_inches"]
    w, h = fig.get_size_inches()
    pos = axes[0].get_position()
    ax_in = (pos.x0 * w, pos.y0 * h, pos.width * w, pos.height * h)
    # 2) freeze the layout, drop panel (b); the page keeps EXACTLY the original crop size (no reflow of the paper)
    fig.set_layout_engine("none")
    axes[1].get_legend().remove()
    fig.delaxes(axes[1])
    ax = axes[0]
    ax.set_position(pos)
    x0c, wc = tb.x0 - pad, CROP_W_PT / 72.0
    y_bottom = tb.y0 - pad
    # 3) inset legend (frameless, DOBI convention as in the supplement figure) in the empty upper-left area: right of the vertical
    #    tolerance guide, above all points and away from the equality line; anchored in data coordinates
    D = np.load(HERE / "figB_regime_distance_agreement_data.npz")
    eps = float(D["regime__eps"])
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(eps + INSET_DX, INSET_TOP_Y), bbox_transform=ax.transData,
                    ncol=1, fontsize=LEGEND_FONTSIZE, markerscale=4 / 3, handletextpad=0.3, labelspacing=0.35, borderaxespad=0.0, borderpad=0.0)
    for lh in leg.legend_handles:
        lh.set_alpha(0.5)
    fig.savefig(out, dpi=300, bbox_inches=Bbox.from_extents(x0c, y_bottom, x0c + wc, tb.y1 + pad))
    # report legend geometry relative to the saved region (points)
    fig.canvas.draw()
    lb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    print("legend bbox (pt, saved-region coords): x %.2f-%.2f of %.2f, y %.2f-%.2f" % (
        (lb.x0 - x0c) * 72, (lb.x1 - x0c) * 72, CROP_W_PT, (lb.y0 - y_bottom) * 72, (lb.y1 - y_bottom) * 72))
    print("labels:", labels)


if __name__ == "__main__":
    use_paper_style()
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "supplement-check":
        fig, _ = build()
        fig.savefig(out, dpi=300)
    elif mode == "main":
        main_figure(out)
    else:
        sys.exit("mode must be supplement-check or main")

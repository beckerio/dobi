# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from dobi.utils.colors import COLOR_CYCLE
import matplotlib.transforms as mtransforms

# -------------------------------------------------------------------------
# Colors
# -------------------------------------------------------------------------
COLORS = list(COLOR_CYCLE)
PRIMARY_COLOR = COLORS[5]
PRED_COLOR = COLORS[1]
DETECTION_COLOR = COLORS[2]
NUM_PLOT_POINTS = 100 
# NUM_PLOT_POINTS = 25

GT_WIDTH = 5.5
# GT_WIDTH = 3.5
PREDICTED_WIDTH = 5.0
# PREDICTED_WIDTH = 2.5
PRED_POINT_SIZE = 250
# PRED_POINT_SIZE = 80
GT_POINT_SIZE =300 
# GT_POINT_SIZE = 90

FONT_SIZE = 12

def downsample_curve_by_index(x, y, n_points=10):
    """Keep n_points points by indexing."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= n_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, n_points, dtype=int)
    return x[idx], y[idx]

# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------
def _set_xscale(ax, use_log_scale: bool, x_min: float, x_max: float, pad_left: float = 0.0):
    """Consistently set linear vs. 'log-like' x-axis."""
    if use_log_scale:
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim(max(x_min, 1e-4), x_max)
    else:
        ax.set_xscale("linear")
        ax.set_xlim(x_min - pad_left, x_max)


def _plot_tau_colored_curve(fig, ax, cax, x, y, thr, print_tau=False, label="GT"):
    """Create τ-colored GT curve as LineCollection with a colorbar."""
    if print_tau:
        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=0.0, vmax=1.0)
        lc = LineCollection(segments, cmap="inferno", norm=norm, linewidth=GT_WIDTH, alpha=1.0)
        lc.set_array(thr[:-1])
        ax.add_collection(lc)

        cbar = None
        if cax is not None:
            cbar = fig.colorbar(lc, cax=cax)
            cbar.set_label("Detector score threshold τ")
            cbar.ax.invert_yaxis()
    else:
        ax.plot(
            x,
            y,
            color=PRIMARY_COLOR,
            linewidth=GT_WIDTH,
            label=label,
        )
        cbar, lc, norm = None, None, None
    return lc, norm, cbar

def _mark_tau_objects(
    cbar_ax,
    tau_objects,
    norm,
    tau_gt: float | None = None,
    tau_pred: float | None = None,
):
    """
    Mark per-object τ values on the tau colorbar.

    Color logic:
      - τ_obj >= max(τ_gt, τ_pred)   → DETECTION_COLOR (detected)
      - τ_obj <  min(τ_gt, τ_pred)   → black          (never detected)
      - between τ_gt and τ_pred      → red            (between GT and Pred)
    If only one threshold is known, 'detected' is defined w.r.t. that threshold.
    """
    if tau_objects is None or cbar_ax is None or norm is None:
        return

    tau_objects = np.asarray(tau_objects, dtype=float)

    # Use axes x-coords and data y-coords
    trans = mtransforms.blended_transform_factory(
        cbar_ax.transAxes,
        cbar_ax.transData,
    )

    # outer_lw = 2.8
    outer_lw = 5.0
    # inner_lw = 1.8
    inner_lw = 3.5

    for tau_g in tau_objects:
        y_pos = norm(tau_g)

        # --- decide color ---
        if (tau_gt is not None) and (tau_pred is not None):
            low = min(tau_gt, tau_pred)
            high = max(tau_gt, tau_pred)

            if tau_g >= high:
                color = DETECTION_COLOR      # detected by both
            elif tau_g < low:
                color = "black"              # never detected
            else:
                color = "red"                # between GT and Pred
        elif tau_pred is not None:
            color = DETECTION_COLOR if tau_g >= tau_pred else "black"
        elif tau_gt is not None:
            color = DETECTION_COLOR if tau_g >= tau_gt else "black"
        else:
            color = "black"

        # --- draw with outline, but longer across the bar ---
        # black outline (almost full width)
        cbar_ax.hlines(
            y_pos, 0.03, 0.97,
            color="black",
            linewidth=outer_lw,
            alpha=0.9,
            transform=trans,
        )
        # colored core
        cbar_ax.hlines(
            y_pos, 0.05, 0.95,
            color=color,
            linewidth=inner_lw,
            alpha=1.0,
            transform=trans,
        )


def _draw_info_box(ax, text: str, color: str = DETECTION_COLOR, xy=(0.05, 0.05)):
    """Draw a small info box in axes coordinates."""
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        color=color,
        fontsize=FONT_SIZE,
        bbox=dict(
            boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9
        ),
    )

def _annotate_gt_point(
    type,
    ax,
    x,
    y,
    tau,
    color=PRIMARY_COLOR,
    op_info: str | None = None,
    print_tau: bool = True,
):
    """Annotation for an ROC operating point."""
    lines = []
    if op_info is not None:
        lines.append(op_info)

    if type == "roc":
        lines.extend([
            f"TPR={y:.3f}",
            f"FPR={x:.3f}",
        ])
        if print_tau:
            lines.append(f"τ={tau:.3f}")
    elif type == "froc":
        lines.extend([
            f"TPR={y:.3f}",
            f"FPPI={x:.3f}",
        ])
        if print_tau:
            lines.append(f"τ={tau:.3f}")
    elif type == "pr":
        lines.extend([
            f"Rec={x:.3f}",
            f"Prec={y:.3f}",
        ])
        if print_tau:
            lines.append(f"τ={tau:.3f}")

    text = "\n".join(lines)

    x_offset_pts = 10
    y_offset_pts = -10
    ha, va = "left", "top"

    if x > 0.8:
        x_offset_pts = -10
        ha = "right"
    if y < 0.2:
        y_offset_pts = 10
        va = "bottom"

    ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(x_offset_pts, y_offset_pts),
        textcoords="offset points",
        ha=ha,
        va=va,
        color=color,
        fontsize=FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.6),
        zorder=10,
    )

def _annotate_pred_op_box(ax, x_pred, y_pred, x_gt, y_gt, tau_hat: float | None = None,  tau_gt: float | None = None, is_froc: bool = False, color=PRED_COLOR, print_tau: bool = True):
    """
    Fixed-position annotation box (bottom-right of axes) for the global predicted OP.
    x, y are only used for the reported values, not for positioning.
    """
    x_delta = x_pred - x_gt
    y_delta = y_pred - y_gt

    lines = ["Predicted Operating Point"]
    if is_froc:
        lines.append(f"TPR={y_pred:.3f} (Δ={y_delta:.3f})")
        lines.append(f"FPPI={x_pred:.3f} (Δ={x_delta:.3f})")
    else:
        lines.append(f"TPR={y_pred:.3f} (Δ={y_delta:.3f})")
        lines.append(f"FPR={x_pred:.3f} (Δ={x_delta:.3f})")
    if tau_hat is not None and print_tau:
        tau_delta = tau_hat - tau_gt
        lines.append(f"τ̂={tau_hat:.3f} (Δ={tau_delta:.3f})")
    text = "\n".join(lines)

    ax.text(
        # 0.975,
        0.96,
        # 0.19,
        0.22,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=color,
        fontsize=FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9),
    )

# -------------------------------------------------------------------------
# Per-case ROC and FROC plotting
# -------------------------------------------------------------------------
def _plot_case_roc(fig, ax, c, fpr, use_log_scale: bool, op_point=None, op_label: str | None = None,  print_tau: bool = True):
    """Plot ROC (GT τ-colored + predicted + operating points) for a single case."""
    tau_img = c.get("tau_img")
    tau_objects = c.get("tau_objects")

    if tau_objects is not None:
        tau_objects = np.asarray(tau_objects, dtype=float)

    thr_roc = c.get("thr_roc")
    raw_fpr_roc = c.get("raw_fpr_roc")
    raw_tpr_roc = c.get("raw_tpr_roc")

    op_tau_global = op_point[2] if op_point is not None else None

    # GT-ROC τ-colored (NO colorbar here)
    lc_roc, norm_roc, _ = _plot_tau_colored_curve(
        fig, ax, None, raw_fpr_roc, raw_tpr_roc, thr_roc, print_tau, "Subset GT ROC"
    )

    # ---- predicted ROC ----
    if c["pred_roc"] is not None:
        ax.plot(
            fpr,
            c["pred_roc"],
            label="Predicted ROC",
            color=PRED_COLOR,
            linestyle=(0, (4, 2)),
            linewidth=PREDICTED_WIDTH,
        )

    _set_xscale(ax, use_log_scale, fpr.min(), 1.0, pad_left=0.025)

    # ---- global OP (GT) + model-only Predicted OP ----
    if op_point is not None:
        op_x_gt, op_y_gt, op_tau_global = op_point

        # --- GT-OP (from DB) on GT curve ---
        ax.scatter(
            op_x_gt,
            op_y_gt,
            s=300,
            color=PRIMARY_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=6,
            marker="s",
        )
        # ax.axvline(op_x_gt, color=PRIMARY_COLOR, linestyle="--", linewidth=1.4, alpha=0.8)
        # ax.axhline(op_y_gt, color=PRIMARY_COLOR, linestyle="--", linewidth=1.4, alpha=0.8)

        op_info = op_label if op_label is not None else None
        _annotate_gt_point(
            "roc",
            ax,
            op_x_gt,
            op_y_gt,
            op_tau_global,
            op_info=op_info,
            print_tau=print_tau,
        )

        # --- Predicted OP on *predicted* ROC curve (no GT x-dependence) ---
        op_x_pred = None
        op_y_pred = None
        if c["pred_roc"] is not None:
            pred_roc = c["pred_roc"]  # shape [L], same grid as fpr

            # Model-only Youden: J(f) = TPR_pred(f) - FPR
            youden_pred = pred_roc - fpr
            op_idx_pred = int(np.argmax(youden_pred))
            op_x_pred = float(fpr[op_idx_pred])
            op_y_pred = float(pred_roc[op_idx_pred])

            ax.scatter(
                op_x_pred,
                op_y_pred,
                s=PRED_POINT_SIZE,
                color=PRED_COLOR,
                edgecolor="white",
                linewidth=1.2,
                zorder=7,
                marker="o",
            )

            # Annotation for predicted OP (no Δ vs GT)
            tau_hat = c.get("tau_hat", None)
            _annotate_pred_op_box(
                ax,
                op_x_pred,
                op_y_pred,
                op_x_gt,
                op_y_gt,
                tau_hat=tau_hat,
                tau_gt=op_tau_global,
                is_froc=False,
                color=PRED_COLOR,
                print_tau=print_tau,
            )

        # # --- Legend entries for GT-OP and Pred-OP ---
        handles_f, labels_f = ax.get_legend_handles_labels()

        ax.legend(handles_f, labels_f, loc="lower right", fontsize=FONT_SIZE)

    # Axis labels, title, legend
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    image_id = c.get("image_id")
    if image_id:
        title = f"{c['name']} ROC [ID: {image_id}]\nGT AUC={c['auroc_gt']:.3f}"
    else:
        title = f"{c['name']} ROC\nGT AUC={c['auroc_gt']:.3f}"
    if c["auroc"] is not None:
        title += f", Pred={c['auroc']:.3f}"
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()

    # Ensure we always have a GT entry, but always draw a legend
    if not any(lbl.startswith("Subset GT ROC") for lbl in labels):
        gt_proxy = Line2D([0], [0], color="k", linewidth=2.5)
        handles.insert(0, gt_proxy)
        if print_tau:
            labels.insert(0, "Subset GT ROC (τ-colored)")
        else:
            labels.insert(0, "Subset GT ROC")

    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=FONT_SIZE)

    # Return LC and norm so caller can build a shared τ colorbar
    return lc_roc, norm_roc


def _plot_case_froc(fig, ax, c, fppi, use_log_scale: bool, op_point=None, op_label: str | None = None, print_tau: bool = True):
    """Plot FROC (GT τ-colored + predicted + operating points) for a single case."""
    tau_img = c.get("tau_img")
    tau_objects = c.get("tau_objects")

    if tau_objects is not None:
        tau_objects = np.asarray(tau_objects, dtype=float)

    thr_froc = c.get("thr_froc")
    raw_fppi_froc = c.get("raw_fppi_froc")
    raw_tpr_froc = c.get("raw_tpr_froc")

    op_tau_global = op_point[2] if op_point is not None else None

    # τ-colored GT FROC (NO colorbar here)
    _plot_tau_colored_curve(
        fig, ax, None, raw_fppi_froc, raw_tpr_froc, thr_froc, print_tau, "Subset GT FROC"
    )

    # ---- predicted FROC ----
    if c["pred_froc"] is not None:
        ax.plot(
            fppi,
            c["pred_froc"],
            label="Predicted FROC",
            color=PRED_COLOR,
            linestyle="--",
            linewidth=PREDICTED_WIDTH,
        )

    _set_xscale(ax, use_log_scale, fppi.min(), fppi.max(), pad_left=0.025)

    # ---- global OP (GT) + model-only Predicted OP (FROC) ----
    if op_point is not None and op_point[0] is not None:
        op_x_gt, op_y_gt, op_tau_global = op_point

        # GT-OP on GT FROC
        ax.scatter(
            op_x_gt,
            op_y_gt,
            s=300,
            color=PRIMARY_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=7,
            marker="s",
        )
        # ax.axvline(op_x_gt, color=PRIMARY_COLOR, linestyle="--", linewidth=1.4, alpha=0.7)
        # ax.axhline(op_y_gt, color=PRIMARY_COLOR, linestyle="--", linewidth=1.4, alpha=0.7)

        op_info = op_label if op_label is not None else None
        _annotate_gt_point(
            "froc",
            ax,
            op_x_gt,
            op_y_gt,
            op_tau_global,
            op_info=op_info,
            print_tau=print_tau,
        )

        # Predicted FROC OP: max TPR on predicted FROC (simple, GT-free)
        op_x_pred_f = None
        op_y_pred_f = None
        if c["pred_froc"] is not None:
            pred_froc = c["pred_froc"]  # shape [M], grid fppi
            op_idx_pred_f = int(np.argmax(pred_froc))
            op_x_pred_f = float(fppi[op_idx_pred_f])
            op_y_pred_f = float(pred_froc[op_idx_pred_f])

            ax.scatter(
                op_x_pred_f,
                op_y_pred_f,
                s=PRED_POINT_SIZE,
                color=PRED_COLOR,
                edgecolor="white",
                linewidth=1.2,
                zorder=7,
                marker="o",
            )

            tau_hat = c.get("tau_hat", None)
            _annotate_pred_op_box(
                ax,
                op_x_pred_f,
                op_y_pred_f,
                op_x_gt,
                op_y_gt,
                tau_hat=tau_hat,
                tau_gt=op_tau_global,
                is_froc=True,
                color=PRED_COLOR,
                print_tau=print_tau
            )

        # Legend entries
        handles_f, labels_f = ax.get_legend_handles_labels()

        ax.legend(handles_f, labels_f, loc="lower right", fontsize=FONT_SIZE)

    # Axis labels, title, legend
    ax.set_xlabel("FPPI")
    ax.set_ylabel("TPR")
    image_id = c.get("image_id")
    if image_id:
        title_f = f"{c['name']} FROC [ID: {image_id}]\nGT AUC={c['afroc_gt']:.3f}"
    else:
        title_f = f"{c['name']} FROC\nGT AUC={c['afroc_gt']:.3f}"
    if c["afroc"] is not None:
        title_f += f", Pred={c['afroc']:.3f}"
    ax.set_title(title_f)
    ax.set_ylim(0, 1)
    ax.grid(True)

    handles_f, labels_f = ax.get_legend_handles_labels()

    # Ensure we always have a GT entry, but always draw a legend
    if not any(lbl.startswith("Subset GT FROC") for lbl in labels_f):
        gt_proxy_f = Line2D([0], [0], color="k", linewidth=2.5)
        handles_f.insert(0, gt_proxy_f)
        if print_tau:
            labels_f.insert(0, "Subset GT FROC (τ-colored)")
        else:
            labels_f.insert(0, "Subset GT FROC")

    if handles_f:
        ax.legend(handles_f, labels_f, loc="lower right", fontsize=FONT_SIZE)

def _plot_case_pr(fig, ax, cax, c, recall_grid, use_log_scale: bool, op_point=None, op_label: str | None = None,  print_tau: bool = True, info_ax=None):
    """Plot PR (GT τ-colored + predicted + operating points) for a single case."""
    tau_img = c.get("tau_img")
    tau_objects = c.get("tau_objects")

    if tau_objects is not None:
        tau_objects = np.asarray(tau_objects, dtype=float)

    thr_pr = c.get("thr_pr")
    raw_recall_pr = c.get("raw_recall_pr")
    raw_precision_pr = c.get("raw_precision_pr")

    if thr_pr is None or raw_recall_pr is None or raw_precision_pr is None:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title(f"{c['name']} PR\n(no GT curve)")
        return

    # τ-colored PR curve using τ as color
    _, norm_pr, cbar_pr = _plot_tau_colored_curve(
        fig, ax, cax, raw_recall_pr, raw_precision_pr, thr_pr, print_tau, "Subset GT PR"
    )

    # Mark τ per-object on the colorbar
    _mark_tau_objects(
        cbar_pr.ax if cbar_pr is not None else None,
        tau_objects,
        norm_pr,
        tau_gt=None,          # or op_tau if you have it for PR
        tau_pred=None,        # or tau_hat if available for this case
    )

    # τ_img in colorbar
    if tau_img is not None and cbar_pr is not None and print_tau:
        cbar_pr.ax.axhline(norm_pr(tau_img), color=PRIMARY_COLOR, linewidth=2.0, alpha=0.9)

    # predicted PR curve
    if c.get("pred_pr") is not None:
        ax.plot(
            recall_grid,
            c["pred_pr"],
            label="Predicted PR",
            color=PRED_COLOR,
            linestyle="--",
            linewidth=PREDICTED_WIDTH,
        )

    # if c.get("pred_pr") is None:
    #     c["pred_pr"] = c["gt_pr"]
    #     recall_pred = np.asarray(recall_grid)
    #     prec_pred   = np.asarray(c["pred_pr"])

    #     if len(recall_pred) > NUM_PLOT_POINTS:
    #         idx = np.linspace(0, len(recall_pred) - 1, NUM_PLOT_POINTS, dtype=int)
    #         recall_pred_plot = recall_pred[idx]
    #         prec_pred_plot   = prec_pred[idx]
    #     else:
    #         recall_pred_plot = recall_pred
    #         prec_pred_plot   = prec_pred

    #     ax.plot(
    #         recall_pred_plot,
    #         prec_pred_plot,
    #         label=f"Discrete PR ({NUM_PLOT_POINTS} Points)",
    #         color=PRED_COLOR,
    #         linestyle="--",
    #         linewidth=PREDICTED_WIDTH,
    #         marker="o",
    #         markersize=8,
    #         markeredgecolor="white",      # edge color
    #         markeredgewidth=1.2, 
    #     )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    image_id = c.get("image_id")
    if image_id:
        title = f"{c['name']} PR [ID: {image_id}]\nGT AUC={c['auprc_gt']:.3f}"
    else:
        title = f"{c['name']} PR\nGT AUC={c['auprc_gt']:.3f}"
    if c["auprc"] is not None:
        title += f", Pred={c['auprc']:.3f}"
    ax.set_title(title)

    # base legend (GT PR + Pred PR curves)
    handles, labels = ax.get_legend_handles_labels()
    if not any(lbl.startswith("Subset GT PR") for lbl in labels):
        gt_proxy = Line2D([0], [0], color="k", linewidth=2.5)
        handles.insert(0, gt_proxy)
        if print_tau:
            labels.insert(0, "Subset GT PR (τ-colored)")
    ax.legend(handles, labels, loc="lower left", fontsize=FONT_SIZE)
    
    # ---- global OP on PR curve (if available) ----
    if op_point is not None:
        op_rec, op_prec, op_tau = op_point  # from _compute_global_op

        # GT OP marker in the PR plot
        ax.scatter(
            op_rec,
            op_prec,
            s=300,
            color=PRIMARY_COLOR,
            edgecolor="white",
            linewidth=1.2,
            zorder=6,
            marker="s",
        )

        # ---------- build GT info text ----------
        gt_lines = []
        if op_label is not None:
            gt_lines.append(op_label)
        gt_lines.append(f"Rec={op_rec:.3f}")
        gt_lines.append(f"Prec={op_prec:.3f}")
        if print_tau:
            gt_lines.append(f"τ={op_tau:.3f}")
        gt_text = "\n".join(gt_lines)

        # ---------- predicted OP on PR (if available) ----------
        pred_text = None
        op_rec_pred = None
        op_prec_pred = None
        tau_hat = c.get("tau_hat", None)

        if c.get("pred_pr") is not None:
            pred_pr = c["pred_pr"]  # Precision values on recall_grid

            # F1 = 2 * P * R / (P + R)
            denom = np.clip(pred_pr + recall_grid, 1e-8, None)
            f1 = 2.0 * pred_pr * recall_grid / denom
            op_idx_pred = int(np.argmax(f1))

            op_rec_pred = float(recall_grid[op_idx_pred])
            op_prec_pred = float(pred_pr[op_idx_pred])

            # marker for predicted OP
            ax.scatter(
                op_rec_pred,
                op_prec_pred,
                s=PRED_POINT_SIZE,
                color=PRED_COLOR,
                edgecolor="white",
                linewidth=1.2,
                zorder=7,
                marker="o",
            )

            # predicted OP text
            x_delta = op_rec_pred - op_rec
            y_delta = op_prec_pred - op_prec

            box_lines = ["Predicted Operating Point"]
            box_lines.append(f"Rec={op_rec_pred:.3f} (Δ={x_delta:.3f})")
            box_lines.append(f"Prec={op_prec_pred:.3f} (Δ={y_delta:.3f})")
            if tau_hat is not None and print_tau:
                tau_delta = tau_hat - op_tau
                box_lines.append(f"τ̂={tau_hat:.3f} (Δ={tau_delta:.3f})")
            pred_text = "\n".join(box_lines)

        # ---------- choose target axis for info boxes ----------
        # If an external info axis is provided, use it; otherwise fall back to PR axis.
        target_ax = info_ax if info_ax is not None else ax
        target_ax.axis("off")

        # GT info box (top of info axis)
        target_ax.text(
            0.05,
            0.65,
            gt_text,
            transform=target_ax.transAxes,
            ha="left",
            va="bottom",
            color=PRIMARY_COLOR,
            fontsize=FONT_SIZE,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=PRIMARY_COLOR,
                alpha=0.9,
            ),
        )

        # Pred info box (bottom of info axis)
        if pred_text is not None:
            target_ax.text(
                0.05,
                0.15,
                pred_text,
                transform=target_ax.transAxes,
                ha="left",
                va="bottom",
                color=PRED_COLOR,
                fontsize=FONT_SIZE,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white",
                    ec=PRED_COLOR,
                    alpha=0.9,
                ),
            )

# -------------------------------------------------------------------------
# Main plotting functions
# -------------------------------------------------------------------------
def plot_original_curves(
    cases,
    fpr,
    fppi,
    recall_grid,
    cluster_id,
    cluster_name,
    output_dir,
    example_idx,
    use_log_scale: bool = False,
    op_points: dict | None = None,
    op_label: str | None = None,
    filename_suffix: str = "",
    print_tau:bool = True,
    plot_pred:bool = True,
):
    # [ image | ROC | ROC-cbar | FROC | FROC-cbar ]
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        6,  # <-- was 5
        # [image, τ-cbar, ROC, FROC, PR, PR-info]
        width_ratios=[3.0, 0.30, 2.8, 2.8, 2.8, 1.8],
        left=0.05,
        right=0.95,
    )

    # Prepare axes per row
    row_axes = {}
    for row in (0, 1):
        img_ax = fig.add_subplot(gs[row, 0])
        if print_tau:
            tau_cax = fig.add_subplot(gs[row, 1])
        else:
            tau_cax = None
        roc_ax = fig.add_subplot(gs[row, 2])
        froc_ax = fig.add_subplot(gs[row, 3])
        pr_ax = fig.add_subplot(gs[row, 4])
        pr_info_ax = fig.add_subplot(gs[row, 5])  # <-- NEW
        pr_info_ax.axis("off") 
        row_axes[row] = dict(
            img_ax=img_ax,
            tau_cax=tau_cax,
            roc_ax=roc_ax,
            froc_ax=froc_ax,
            pr_ax=pr_ax,
            pr_info_ax=pr_info_ax,
        )

    for index, c in enumerate(cases):
        if (index != 0) and (c["sev"] == 0):
            continue

        if not plot_pred:
            c["pred_roc"] = None
            c["pred_froc"] = None
            c["pred_pr"] = None

        r = c["row"]
        ax_dict = row_axes[r]
        img_ax = ax_dict["img_ax"]
        tau_cax = ax_dict["tau_cax"]
        roc_ax = ax_dict["roc_ax"]
        froc_ax = ax_dict["froc_ax"]
        pr_ax = ax_dict["pr_ax"]
        pr_info_ax = ax_dict["pr_info_ax"]

        # ---------------- Image ----------------
        if c["img"] is not None:
            img_ax.imshow(c["img"])
            sev_txt = f", sev={c['sev']}" if c["sev"] is not None else ""
            img_ax.set_title(f"{c['name']} ({c['deg_name']}{sev_txt})")
        else:
            img_ax.text(
                0.5,
                0.5,
                f"No {c['name'].lower()} image",
                ha="center",
                va="center",
            )
            img_ax.set_title(f"{c['name']} (missing)")
        img_ax.axis("off")

        # optional global operating points per case
        op_point_case_roc = None
        op_point_case_froc = None
        op_point_case_pr = None
        if op_points is not None:
            case_ops = op_points.get(c["name"])
            if case_ops is not None:
                op_point_case_roc = case_ops.get("roc")
                op_point_case_froc = case_ops.get("froc")
                op_point_case_pr = case_ops.get("pr")


        # Draw detection info on the IMAGE axis (depends only on τ for this image)
        tau_img = c.get("tau_img")          # typically τ̂ if available
        tau_hat = c.get("tau_hat")          # predicted τ̂ (may be None)
        num_gt = c.get("num_gt", 0)
        num_gt_det = c.get("num_gt_detected", 0)
        tau_objects = c.get("tau_objects")

        if num_gt > 0:
            if tau_img is not None and tau_objects is not None and print_tau:
                tau_objects_arr = np.asarray(tau_objects, dtype=float)
                num_gt_det_op = int(np.sum(tau_objects_arr >= tau_img))

                # Clarify whether τ is predicted (τ̂) or generic τ
                if tau_hat is not None:
                    tau_label = f"τ̂"
                else:
                    tau_label = f"τ"

                txt = f"Detected @ {tau_label}: {num_gt_det_op}/{num_gt}"
            else:
                # fall back to DB-level counts if no per-object τ is usable
                txt = f"Detected GT: {num_gt_det}/{num_gt}"

            # bottom-left inside the image
            _draw_info_box(img_ax, txt, xy=(0.02, 0.02))

        # ---------------- ROC ----------------
        lc_roc, norm_roc = _plot_case_roc(
            fig,
            roc_ax,
            c,
            fpr,
            use_log_scale,
            op_point=op_point_case_roc,
            op_label=op_label,
            print_tau=print_tau,
        )

        # ---------------- Shared τ colorbar next to image ----------------
        if print_tau:
            tau_cax.cla()
            cbar = fig.colorbar(lc_roc, cax=tau_cax)
            cbar.set_label("Detector score threshold τ")
            cbar.ax.invert_yaxis()

            # τ values
            tau_objects = c.get("tau_objects")
            tau_img = c.get("tau_img")        # typically τ̂ if available
            tau_hat = c.get("tau_hat")
            # choose global GT τ_op if available (prefer ROC, then FROC, then PR)
            op_tau_global = None
            if op_point_case_roc is not None:
                op_tau_global = op_point_case_roc[2]
            elif op_point_case_froc is not None:
                op_tau_global = op_point_case_froc[2]
            elif op_point_case_pr is not None:
                op_tau_global = op_point_case_pr[2]

            # per-object τ markers
            _mark_tau_objects(
                cbar.ax,
                tau_objects,
                norm_roc,
                tau_gt=op_tau_global,
                tau_pred=tau_hat,
            )

            # ---------- thresholds on the colorbar ----------
            trans = mtransforms.blended_transform_factory(
                cbar.ax.transAxes,   # x in axes coords [0, 1]
                cbar.ax.transData,   # y in data coords (norm_roc output)
            )

            # outer_lw = 4.2
            outer_lw =5.0 
            # inner_lw = 2.6
            inner_lw = 3.5

            y_gt = norm_roc(op_tau_global) if op_tau_global is not None else None
            y_pred = norm_roc(tau_hat) if tau_hat is not None else None

            # how close is "overlapping" on the colorbar?
            close_eps = 0.02  # in normalized (0–1) colorbar units

            def _draw_threshold_full(y, color):
                # full-width line with outline
                cbar.ax.hlines(
                    y, 0.0, 1.0,
                    color="black",
                    linewidth=outer_lw,
                    alpha=0.9,
                    transform=trans,
                )
                cbar.ax.hlines(
                    y, 0.0, 1.0,
                    color=color,
                    linewidth=inner_lw,
                    alpha=1.0,
                    transform=trans,
                )

            def _draw_threshold_split(y, color, x0, x1):
                # split line that fully fills its half of the bar
                cbar.ax.hlines(
                    y, x0, x1,
                    color="black",
                    linewidth=outer_lw,
                    alpha=0.9,
                    transform=trans,
                )
                cbar.ax.hlines(
                    y, x0, x1,
                    color=color,
                    linewidth=inner_lw,
                    alpha=1.0,
                    transform=trans,
                )

            if (y_gt is not None) and (y_pred is not None):
                # both thresholds exist
                if abs(y_gt - y_pred) < close_eps:
                    # too close → split into left/right halves, each filling its half
                    _draw_threshold_split(y_gt,  PRIMARY_COLOR, 0.0, 0.5)  # GT left
                    _draw_threshold_split(y_pred, PRED_COLOR,   0.5, 1.0)  # Pred right
                else:
                    # far enough apart → each full-width
                    _draw_threshold_full(y_gt,  PRIMARY_COLOR)
                    _draw_threshold_full(y_pred, PRED_COLOR)
            elif y_gt is not None:
                _draw_threshold_full(y_gt, PRIMARY_COLOR)
            elif y_pred is not None:
                _draw_threshold_full(y_pred, PRED_COLOR)
            # -----------------------------------------------

        # ---------------- FROC ----------------
        _plot_case_froc(fig, froc_ax, c, fppi, use_log_scale, op_point=op_point_case_froc, op_label=op_label, print_tau=print_tau)

        # ---------------- PR ----------------
        _plot_case_pr(fig, pr_ax, None, c, recall_grid, use_log_scale=False,
                      op_point=op_point_case_pr, op_label=op_label, print_tau=print_tau, info_ax=pr_info_ax,)

    fig.suptitle(f"Cluster {cluster_id} - {cluster_name}", fontsize=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ex_{example_idx:02d}_curves{filename_suffix}"
    fig.savefig(output_dir / f"{filename}.png", dpi=150)

    plt.close(fig)

    print(
        f"Saved visualization for cluster {cluster_id}, example {example_idx} "
        f"at {output_dir / (filename + '.png')}"
    )


def print_delta_curves(cases, fpr, fppi, recall_grid, cluster_id, cluster_name, output_dir, example_idx):
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
    (ax_pr_roc, ax_pr_froc, ax_pr_pr), (ax_deg_roc, ax_deg_froc, ax_deg_pr) = axes2


    delta_axes = {
        ("Pristine", "ROC"): ax_pr_roc,
        ("Pristine", "FROC"): ax_pr_froc,
        ("Degraded", "ROC"): ax_deg_roc,
        ("Degraded", "FROC"): ax_deg_froc,
        ("Pristine", "PR"): ax_pr_pr,
        ("Degraded", "PR"): ax_deg_pr,
    }

    for index, c in enumerate(cases):
        if (index != 0) and (c["sev"] == 0):
            continue

        # ROC Δ
        ax = delta_axes.get((c["name"], "ROC"))
        if ax is not None:
            if c["delta_roc"] is not None:
                ax.plot(fpr, c["delta_roc"], color=PRIMARY_COLOR, label="ΔTPR (ROC)")
                _style_delta_ax(
                    ax, "FPR", f"{c['name']} ROC Δ", 0.0, 1.0, c["delta_auroc"]
                )
                ax.legend(loc="lower right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No {c['name'].lower()} ROC prediction",
                    ha="center",
                    va="center",
                )
                _style_delta_ax(ax, "FPR", f"{c['name']} ROC Δ", 0.0, 1.0)

        # FROC Δ
        ax = delta_axes.get((c["name"], "FROC"))
        if ax is not None:
            if c["delta_froc"] is not None:
                ax.plot(fppi, c["delta_froc"], color=PRIMARY_COLOR, label="ΔTPR (FROC)")
                _style_delta_ax(
                    ax,
                    "FPPI",
                    f"{c['name']} FROC Δ",
                    fppi.min(),
                    fppi.max(),
                    c["delta_afroc"],
                )
                ax.legend(loc="lower right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No {c['name'].lower()} FROC prediction",
                    ha="center",
                    va="center",
                )
                _style_delta_ax(
                    ax, "FPPI", f"{c['name']} FROC Δ", fppi.min(), fppi.max()
                )

        # PR Δ
        ax = delta_axes.get((c["name"], "PR"))
        if ax is not None:
            if c.get("delta_pr") is not None:
                ax.plot(recall_grid, c["delta_pr"], color=PRIMARY_COLOR, label="ΔPrecision (PR)")
                _style_delta_ax(
                    ax,
                    "Recall",
                    f"{c['name']} PR Δ",
                    0.0,
                    1.0,
                    delta_auc=c["delta_auprc"],
                    y_label="ΔPrecision (Pred - GT)",
                )
                ax.legend(loc="lower right")
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No {c['name'].lower()} PR prediction",
                    ha="center",
                    va="center",
                )
                _style_delta_ax(ax, "Recall", f"{c['name']} PR Δ", 0.0, 1.0, y_label="ΔPrecision (Pred - GT)")

    fig2.suptitle(f"Cluster {cluster_id} - {cluster_name} (Δ Curves)", fontsize=14)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ex_{example_idx:02d}_delta.png"
    delta_path = output_dir / filename
    fig2.savefig(delta_path, dpi=150)
    plt.close(fig2)

    print(
        f"Saved delta visualization for cluster {cluster_id}, example {example_idx} "
        f"at {delta_path}"
    )


def _style_delta_ax(ax, xlabel, title, x_min, x_max, delta_auc=None, y_label: str = "ΔTPR (Pred - GT)"):
    ax.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True)
    ax.set_title(
        f"{title}\nΔAUC={delta_auc:+.3f}" if delta_auc is not None else title
    )

# -------------------------------------------------------------------------
# Threshold-indexed behavior functions
# -------------------------------------------------------------------------
TAU_PANELS = (
    ("tpr", "TPR (= Recall)", "raw_tpr_roc"),
    ("fpr", "FPR", "raw_fpr_roc"),
    ("fppi", "FPPI", "raw_fppi_froc"),
    ("precision", "Precision", "raw_precision_pr"),
    ("f1", "F1", None),
)


def _gt_tau_function(c, key, raw_key):
    """Empirical tau-function from the raw sweep rows stored in the case (ascending tau)."""
    thr = np.asarray(c["thr_roc"], dtype=np.float64)
    if key == "f1":
        p = np.asarray(c["raw_precision_pr"], dtype=np.float64)
        r = np.asarray(c["raw_recall_pr"], dtype=np.float64)
        den = p + r
        y = np.where(den > 0, 2.0 * p * r / np.where(den > 0, den, 1.0), 0.0)
    else:
        y = np.asarray(c[raw_key], dtype=np.float64)
    order = np.argsort(thr, kind="stable")
    return thr[order], y[order]


def plot_tau_behavior_functions(cases, cluster_id, cluster_name, output_dir, example_idx, fppi_log: bool = True):
    """
    One row per case (pristine / degraded): TPR, FPR, FPPI, Precision and F1 versus
    the detector threshold. Empirical (GT) step functions, the predicted
    threshold-indexed functions, the control points (tau_behavior heads) and,
    on the F1 panel, the predicted and the oracle operating thresholds.
    """
    cases = [c for c in cases if c.get("pred_tau_curves") is not None and c.get("thr_roc") is not None]
    if not cases:
        return

    fig, axes = plt.subplots(len(cases), len(TAU_PANELS), figsize=(4.0 * len(TAU_PANELS), 3.4 * len(cases)),
                             squeeze=False, constrained_layout=True)
    for row, c in enumerate(cases):
        pred = c["pred_tau_curves"]
        for col, (key, label, raw_key) in enumerate(TAU_PANELS):
            ax = axes[row, col]
            x_gt, y_gt = _gt_tau_function(c, key, raw_key)
            ax.plot(x_gt, y_gt, color=PRIMARY_COLOR, lw=3.0, label="GT (empirical)")
            ax.plot(pred["tau"], pred[key], color=PRED_COLOR, lw=2.0, label="Predicted")
            ctrl = pred.get("ctrl")
            if ctrl is not None and key in ctrl:
                ax.plot(pred["knots"], ctrl[key], "o", ms=5, mfc="white", mec=PRED_COLOR, mew=1.5,
                        label="Control points")
            if key == "f1":
                idx_gt = int(np.argmax(y_gt))
                ax.axvline(x_gt[idx_gt], color=PRIMARY_COLOR, ls=":", lw=1.8, label=f"oracle τ={x_gt[idx_gt]:.2f}")
                if c.get("tau_hat") is not None:
                    ax.axvline(c["tau_hat"], color=PRED_COLOR, ls="-", lw=1.5, label=f"adaptive τ={c['tau_hat']:.2f}")
            if key == "fppi" and fppi_log:
                ax.set_yscale("symlog", linthresh=0.1)
            else:
                ax.set_ylim(-0.02, 1.02)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("detector threshold τ")
            ax.grid(alpha=0.25, lw=0.5)
            if row == 0:
                ax.set_title(label, fontsize=FONT_SIZE)
            if col == 0:
                ax.set_ylabel(f"{c['name']}\n{c.get('deg_name', '')} s{c.get('sev', '')}", fontsize=FONT_SIZE)
            if row == 0 and col in (0, len(TAU_PANELS) - 1):
                ax.legend(fontsize=8, loc="best", frameon=False)
            elif col == len(TAU_PANELS) - 1:
                ax.legend(fontsize=8, loc="best", frameon=False)

    fig.suptitle(f"Threshold-indexed behavior — {cluster_name}", fontsize=FONT_SIZE + 1)
    suffix = "" if example_idx is None else f"_ex{example_idx:02d}"
    fig.savefig(output_dir / f"{cluster_id:03d}_{cluster_name}_tau_functions{suffix}.png", dpi=150)
    plt.close(fig)

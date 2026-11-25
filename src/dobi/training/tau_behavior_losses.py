# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Losses for the joint threshold-indexed behavior head (TauBehaviorHead).

Main loss (always on):
    L_behavior = w_tpr L(TPR) + w_fpr L(FPR) + w_fppi L(FPPI) + w_prec L(Precision)
with L = SmoothL1 (or Huber) on the shared tau grid. FPPI is compared in the
same space the head interpolates in (log1p or linear).

Optional auxiliary losses (each off unless its weight is > 0):
    w_roc / w_froc / w_pr  derived-curve losses: predicted points (x(tau), y(tau)) vs the
                           empirical geometric curve evaluated at the predicted x(tau)
    w_area                 area under each tau-function
    w_smooth               squared second differences of the control values
    w_tau                  soft operating point: SmoothL1(soft-argmax F1, tau_opt)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from dobi.utils.tau_spline import soft_operating_threshold, uniform_grid_interp


def _reg_loss(pred: torch.Tensor, target: torch.Tensor, kind: str, beta: float) -> torch.Tensor:
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=beta)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=beta)
    if kind == "l1":
        return F.l1_loss(pred, target)
    raise ValueError(f"Unknown behavior loss '{kind}'")


def _second_diff(ctrl: torch.Tensor) -> torch.Tensor:
    if ctrl.shape[1] < 3:
        return ctrl.new_zeros(())
    d2 = ctrl[:, 2:] - 2.0 * ctrl[:, 1:-1] + ctrl[:, :-2]
    return (d2 ** 2).mean()


def compute_tau_behavior_losses(
    pred: Dict[str, torch.Tensor],
    head_out: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    subset_idx: torch.Tensor,
    tau_grid: torch.Tensor,
    args,
    fppi_space: str = "log1p",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    kind = getattr(args, "behavior_loss", "smooth_l1")
    beta = float(getattr(args, "loss_beta", 0.05))
    zero = pred["tpr_tau"].new_zeros(())

    t_tpr = targets["tpr"][subset_idx]
    t_fpr = targets["fpr"][subset_idx]
    t_fppi = targets["fppi"][subset_idx]
    t_prec = targets["precision_env" if getattr(args, "precision_envelope", False) else "precision"][subset_idx]

    if fppi_space == "log1p":
        p_fppi, g_fppi = pred["fppi_space_tau"], torch.log1p(t_fppi.clamp_min(0.0))
    else:
        p_fppi, g_fppi = pred["fppi_tau"], t_fppi

    parts = {
        "tpr": _reg_loss(pred["tpr_tau"], t_tpr, kind, beta),
        "fpr": _reg_loss(pred["fpr_tau"], t_fpr, kind, beta),
        "fppi": _reg_loss(p_fppi, g_fppi, kind, beta),
        "precision": _reg_loss(pred["precision_tau"], t_prec, kind, beta),
    }
    loss = (
        args.w_tpr * parts["tpr"]
        + args.w_fpr * parts["fpr"]
        + args.w_fppi * parts["fppi"]
        + args.w_prec * parts["precision"]
    )

    # ---- derived geometric curves (optional) ----
    if args.w_roc > 0:
        roc_gt = uniform_grid_interp(pred["fpr_tau"], 1.0, targets["roc_tpr"][subset_idx])
        parts["roc"] = F.l1_loss(pred["tpr_tau"], roc_gt)
        loss = loss + args.w_roc * parts["roc"]
    if args.w_froc > 0:
        u_max = float(args.u_max)
        froc_gt = uniform_grid_interp(pred["fppi_tau"], u_max, targets["froc_tpr"][subset_idx])
        inside = (pred["fppi_tau"] <= u_max).float()
        parts["froc"] = (inside * (pred["tpr_tau"] - froc_gt).abs()).sum() / inside.sum().clamp_min(1.0)
        loss = loss + args.w_froc * parts["froc"]
    if args.w_pr > 0:
        pr_gt = uniform_grid_interp(pred["recall_tau"], 1.0, targets["pr_precision"][subset_idx])
        parts["pr"] = F.l1_loss(pred["precision_tau"], pr_gt)
        loss = loss + args.w_pr * parts["pr"]

    # ---- area under the tau-functions (optional) ----
    if args.w_area > 0:
        area = zero
        for p_key, t in (("tpr_tau", t_tpr), ("fpr_tau", t_fpr), ("precision_tau", t_prec)):
            area = area + F.l1_loss(torch.trapz(pred[p_key], tau_grid, dim=1), torch.trapz(t, tau_grid, dim=1))
        area = area + F.l1_loss(torch.trapz(p_fppi, tau_grid, dim=1), torch.trapz(g_fppi, tau_grid, dim=1))
        parts["area"] = area
        loss = loss + args.w_area * area

    # ---- smoothness of the control values (optional) ----
    if args.w_smooth > 0:
        smooth = sum(_second_diff(head_out[k]) for k in ("tpr_ctrl", "fpr_ctrl", "fppi_ctrl", "precision_ctrl"))
        parts["smooth"] = smooth
        loss = loss + args.w_smooth * smooth

    # ---- soft operating point (optional) ----
    if args.w_tau > 0:
        tau_soft = soft_operating_threshold(tau_grid, pred["f1_tau"], temperature=args.tau_temperature)
        tau_gt = targets["tau_opt"][subset_idx]
        parts["tau"] = F.smooth_l1_loss(tau_soft, tau_gt, beta=beta)
        loss = loss + args.w_tau * parts["tau"]

    return loss, parts

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Joint threshold-indexed behavior head.

One shared monitor representation z (the GAP+GMP readout + MLP trunk used by
TauSplineSingleHead / TauSplineSharedAnchorHead) feeds four lightweight
branches that predict control values at a fixed, global set of threshold knots:

    TPR(tau), FPR(tau), FPPI(tau)  -> non-increasing in tau by construction
    Precision(tau)                 -> bounded in [0, 1], not forced monotone

The knots are a buffer (uniform or training-score quantiles), never image
dependent. ``decode`` interpolates the controls (linear or PCHIP) on any query
threshold grid; ROC, FROC, PR, F1(tau) and operating thresholds are all derived
from that one decoded representation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dobi.models.spline_monitor_model import _make_mlp
from dobi.utils.tau_spline import INTERP_METHODS, interpolate, safe_f1

FPPI_SPACES = ("log1p", "linear")
FPPI_MAX = 1000.0  # smooth upper bound of the log1p-space FPPI start value (prevents expm1 overflow)


def pool_gap_gmp(feats: List[torch.Tensor]) -> torch.Tensor:
    """Global average + global max pooling per FPN level, concatenated (existing monitor readout)."""
    pooled = [torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1) for f in feats]
    return torch.cat(pooled, dim=1)


def monotone_nonincreasing(start: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    start: [B,1] value at the first knot (>= 0), logits: [B,K] (K-1 drop logits + 1 slack logit).
    value_k = start - sum_{j<k} start * softmax(logits)_j  ->  non-increasing, >= start * w_slack >= 0.
    """
    w = F.softmax(logits, dim=1)
    drops = start * w[:, :-1]
    zero = torch.zeros_like(start)
    return start - torch.cat([zero, torch.cumsum(drops, dim=1)], dim=1)


def monotone_nondecreasing_bounded(start: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """start in [0,1]: value_k = start + sum_{j<k} (1-start) softmax(logits)_j  ->  non-decreasing, <= 1."""
    w = F.softmax(logits, dim=1)
    incs = (1.0 - start) * w[:, :-1]
    zero = torch.zeros_like(start)
    return start + torch.cat([zero, torch.cumsum(incs, dim=1)], dim=1)


class TauBehaviorHead(nn.Module):
    needs_gt_roi = False
    needs_tp_fp_roi = False

    def __init__(
        self,
        feat_channels: List[int],
        knots: Sequence[float],
        hidden_width: int = 1024,
        num_hidden_layers: int = 2,
        interp: str = "pchip",
        fppi_space: str = "log1p",
        precision_envelope: bool = False,
    ):
        super().__init__()
        if interp not in INTERP_METHODS:
            raise ValueError(f"interp must be one of {INTERP_METHODS}")
        if fppi_space not in FPPI_SPACES:
            raise ValueError(f"fppi_space must be one of {FPPI_SPACES}")

        knots_t = torch.as_tensor(knots, dtype=torch.float32).flatten()
        if knots_t.numel() < 2 or not bool(torch.all(knots_t[1:] > knots_t[:-1])):
            raise ValueError("knots must be strictly increasing with at least 2 entries")
        self.register_buffer("knots", knots_t, persistent=True)

        self.n_knots = int(knots_t.numel())
        self.interp = interp
        self.fppi_space = fppi_space
        self.precision_envelope = bool(precision_envelope)

        in_dim = int(sum(feat_channels) * 2)  # GAP + GMP
        self.trunk = _make_mlp(in_dim, hidden_width, num_hidden_layers)

        K = self.n_knots
        # start logit + (K-1) drop logits + 1 slack logit
        self.tpr_branch = nn.Linear(hidden_width, K + 1)
        self.fpr_branch = nn.Linear(hidden_width, K + 1)
        self.fppi_branch = nn.Linear(hidden_width, K + 1)
        self.precision_branch = nn.Linear(hidden_width, K + 1 if self.precision_envelope else K)

        # start FPPI near the typical value at tau=0 (tens of FP per image) to avoid a slow warm-up
        with torch.no_grad():
            if fppi_space == "log1p":
                frac = 3.0 / math.log1p(FPPI_MAX)
                self.fppi_branch.bias[0].fill_(math.log(frac / (1.0 - frac)))
            else:
                self.fppi_branch.bias[0].fill_(10.0)

    # ----------------------------------------------------------
    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.trunk(pool_gap_gmp(feats))
        h = h.float()

        tpr_raw = self.tpr_branch(h)
        fpr_raw = self.fpr_branch(h)
        fppi_raw = self.fppi_branch(h)
        prec_raw = self.precision_branch(h)

        tpr_ctrl = monotone_nonincreasing(torch.sigmoid(tpr_raw[:, :1]), tpr_raw[:, 1:])
        fpr_ctrl = monotone_nonincreasing(torch.sigmoid(fpr_raw[:, :1]), fpr_raw[:, 1:])
        # FPPI controls live in log1p(FPPI) space or plain FPPI space depending on fppi_space
        if self.fppi_space == "log1p":
            fppi_start = math.log1p(FPPI_MAX) * torch.sigmoid(fppi_raw[:, :1])
        else:
            fppi_start = F.softplus(fppi_raw[:, :1])
        fppi_ctrl = monotone_nonincreasing(fppi_start, fppi_raw[:, 1:])

        if self.precision_envelope:
            prec_ctrl = monotone_nondecreasing_bounded(torch.sigmoid(prec_raw[:, :1]), prec_raw[:, 1:])
        else:
            prec_ctrl = torch.sigmoid(prec_raw)

        return {
            "knots": self.knots,
            "tpr_ctrl": tpr_ctrl,
            "fpr_ctrl": fpr_ctrl,
            "fppi_ctrl": fppi_ctrl,
            "precision_ctrl": prec_ctrl,
        }

    # ----------------------------------------------------------
    def fppi_from_ctrl_space(self, v: torch.Tensor) -> torch.Tensor:
        return torch.expm1(v) if self.fppi_space == "log1p" else v

    def decode(self, head_out: Dict[str, torch.Tensor], tau_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the threshold-indexed representation on tau_grid ([Q]); always in float32."""
        tau = tau_grid.float()
        xk = head_out["knots"].float().to(tau.device)
        with torch.autocast(device_type=tau.device.type, enabled=False):
            tpr = interpolate(xk, head_out["tpr_ctrl"].float(), tau, self.interp).clamp(0.0, 1.0)
            fpr = interpolate(xk, head_out["fpr_ctrl"].float(), tau, self.interp).clamp(0.0, 1.0)
            fppi_v = interpolate(xk, head_out["fppi_ctrl"].float(), tau, self.interp).clamp_min(0.0)
            precision = interpolate(xk, head_out["precision_ctrl"].float(), tau, self.interp).clamp(0.0, 1.0)

            fppi = self.fppi_from_ctrl_space(fppi_v)
            f1 = safe_f1(precision, tpr)

        return {
            "tau": tau,
            "tpr_tau": tpr,
            "recall_tau": tpr,
            "fpr_tau": fpr,
            "fppi_tau": fppi,
            "fppi_space_tau": fppi_v,
            "precision_tau": precision,
            "f1_tau": f1,
        }

    def config(self) -> Dict:
        return {
            "knots": self.knots.detach().cpu().tolist(),
            "interp": self.interp,
            "fppi_space": self.fppi_space,
            "precision_envelope": self.precision_envelope,
        }

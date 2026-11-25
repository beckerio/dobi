# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim: int, hidden: int, num_hidden_layers: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
        d = hidden
    return nn.Sequential(*layers)


class TauSplineSharedAnchorHead(nn.Module):
    """
    Shared-anchor head:
    - one shared tau-anchor grid for all predicted curves
    - can be fixed (uniform) or learned per sample
    - returns the same raw keys as your current head so downstream stays simple
    """
    needs_gt_roi = False
    needs_tp_fp_roi = False

    def __init__(
        self,
        feat_channels: List[int],
        hidden_width: int = 1024,
        n_anchors: int = 8,
        num_hidden_layers: int = 2,
        learn_shared_tau: bool = False,
    ):
        super().__init__()
        self.n_anchors = int(n_anchors)
        self.learn_shared_tau = bool(learn_shared_tau)

        if self.n_anchors < 2:
            raise ValueError("TauSplineSharedAnchorHead requires n_anchors >= 2")

        in_dim = int(sum(feat_channels) * 2)  # GAP + GMP
        self.trunk = _make_mlp(in_dim, hidden_width, num_hidden_layers)

        if self.learn_shared_tau:
            # predicts positive steps; anchors become [0, cumsum(steps)] normalized to end at 1
            self.tau_step = nn.Linear(hidden_width, self.n_anchors - 1)
        else:
            tau_anchor = torch.linspace(0.0, 1.0, self.n_anchors)
            self.register_buffer("tau_anchor_fixed", tau_anchor, persistent=True)

        # monotone decreasing in tau
        self.tpr_start = nn.Linear(hidden_width, 1)
        self.tpr_drop = nn.Linear(hidden_width, self.n_anchors - 1)

        self.fpr_start = nn.Linear(hidden_width, 1)
        self.fpr_drop = nn.Linear(hidden_width, self.n_anchors - 1)

        self.fppi_start = nn.Linear(hidden_width, 1)
        self.fppi_drop = nn.Linear(hidden_width, self.n_anchors - 1)

        # precision(tau): bounded, not forced monotone
        self.precision_head = nn.Linear(hidden_width, self.n_anchors)

    @staticmethod
    def _bounded_monotone_decreasing(start_raw: torch.Tensor, drop_raw: torch.Tensor) -> torch.Tensor:
        start = torch.sigmoid(start_raw)  # [B,1] in (0,1)

        if drop_raw.shape[1] == 0:
            return start

        drops = F.softplus(drop_raw) + 1e-6
        drops = drops / (drops.sum(dim=1, keepdim=True) + 1e-8)
        drops = drops * start
        y = torch.cat([start, start - torch.cumsum(drops, dim=1)], dim=1)
        return y.clamp(0.0, 1.0)

    @staticmethod
    def _shared_tau_from_steps(step_raw: torch.Tensor) -> torch.Tensor:
        """
        step_raw: [B, A-1]
        returns shared tau anchors in [0,1], strictly increasing, shape [B, A]
        """
        steps = F.softplus(step_raw) + 1e-4
        steps = steps / (steps.sum(dim=1, keepdim=True) + 1e-8)
        tau = torch.cat(
            [torch.zeros(step_raw.shape[0], 1, device=step_raw.device, dtype=step_raw.dtype),
             torch.cumsum(steps, dim=1)],
            dim=1,
        )
        tau[:, -1] = 1.0
        return tau.clamp(0.0, 1.0)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        pooled = []
        for f in feats:
            gap = f.mean(dim=(2, 3))
            gmp = f.amax(dim=(2, 3))
            pooled.append(torch.cat([gap, gmp], dim=1))

        x = torch.cat(pooled, dim=1)
        h = self.trunk(x)

        if self.learn_shared_tau:
            tau_anchor = self._shared_tau_from_steps(self.tau_step(h))
        else:
            tau_anchor = self.tau_anchor_fixed.unsqueeze(0).expand(x.shape[0], -1)

        tpr_anchor = self._bounded_monotone_decreasing(self.tpr_start(h), self.tpr_drop(h))
        fpr_anchor = self._bounded_monotone_decreasing(self.fpr_start(h), self.fpr_drop(h))
        fppi_anchor_norm = self._bounded_monotone_decreasing(self.fppi_start(h), self.fppi_drop(h))
        precision_anchor = torch.sigmoid(self.precision_head(h))

        return {
            "tau_anchor": tau_anchor,
            "tpr_anchor": tpr_anchor,
            "fpr_anchor": fpr_anchor,
            "fppi_anchor_norm": fppi_anchor_norm,
            "precision_anchor": precision_anchor,
        }
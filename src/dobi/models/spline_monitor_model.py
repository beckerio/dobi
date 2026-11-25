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


class TauSplineSingleHead(nn.Module):
    needs_gt_roi = False
    needs_tp_fp_roi = False

    def __init__(
        self,
        feat_channels: List[int],
        hidden_width: int = 1024,
        n_anchors: int = 8,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.n_anchors = int(n_anchors)
        in_dim = int(sum(feat_channels) * 2)  # GAP + GMP

        self.trunk = _make_mlp(in_dim, hidden_width, num_hidden_layers)

        # monotone decreasing in tau
        self.tpr_start = nn.Linear(hidden_width, 1)
        self.tpr_drop = nn.Linear(hidden_width, self.n_anchors - 1)

        self.fpr_start = nn.Linear(hidden_width, 1)
        self.fpr_drop = nn.Linear(hidden_width, self.n_anchors - 1)

        self.fppi_start = nn.Linear(hidden_width, 1)   # normalized [0,1], scale outside by u_max
        self.fppi_drop = nn.Linear(hidden_width, self.n_anchors - 1)

        # precision(tau): bounded but not forced monotone
        self.precision_head = nn.Linear(hidden_width, self.n_anchors)

        tau_anchor = torch.linspace(0.0, 1.0, self.n_anchors)
        self.register_buffer("tau_anchor", tau_anchor, persistent=True)

    @staticmethod
    def _bounded_monotone_decreasing(start_raw: torch.Tensor, drop_raw: torch.Tensor) -> torch.Tensor:
        """
        tau is ascending [0..1], so tpr/fpr/fppi should be non-increasing in tau.
        """
        start = torch.sigmoid(start_raw)  # [B,1] in (0,1)
        if drop_raw.shape[1] == 0:
            return start

        drops = F.softplus(drop_raw) + 1e-6
        drops = drops / (drops.sum(dim=1, keepdim=True) + 1e-8)
        drops = drops * start
        y = torch.cat([start, start - torch.cumsum(drops, dim=1)], dim=1)
        return y.clamp(0.0, 1.0)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        pooled = []
        for f in feats:
            gap = f.mean(dim=(2, 3))
            gmp = f.amax(dim=(2, 3))
            pooled.append(torch.cat([gap, gmp], dim=1))

        x = torch.cat(pooled, dim=1)
        h = self.trunk(x)

        tpr_anchor = self._bounded_monotone_decreasing(self.tpr_start(h), self.tpr_drop(h))
        fpr_anchor = self._bounded_monotone_decreasing(self.fpr_start(h), self.fpr_drop(h))
        fppi_anchor_norm = self._bounded_monotone_decreasing(self.fppi_start(h), self.fppi_drop(h))
        precision_anchor = torch.sigmoid(self.precision_head(h))

        return {
            "tau_anchor": self.tau_anchor.unsqueeze(0).expand(x.shape[0], -1),
            "tpr_anchor": tpr_anchor,
            "fpr_anchor": fpr_anchor,
            "fppi_anchor_norm": fppi_anchor_norm,
            "precision_anchor": precision_anchor,
        }

def forward_backbone(model_ultra, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Forward everything except detect head; return FPN features
    (the same ones the YOLO head would use).
    """
    layers = model_ultra.model  # nn.Sequential-like
    y = []
    for m in layers[:-1]:  # exclude head
        if hasattr(m, "f"):
            from_idxs = m.f if isinstance(m.f, list) else [m.f]
            inputs = [x if j == -1 else y[j] for j in from_idxs]
            x_in = inputs[0] if len(inputs) == 1 else inputs
        else:
            x_in = x
        x = m(x_in)
        y.append(x)

    head = layers[-1]
    return [y[j] for j in head.f]

def build_backbone_runner(ultra_model):
    layers = ultra_model.model
    body = layers[:-1]
    head = layers[-1]

    froms = []
    for m in body:
        if hasattr(m, "f"):
            f = m.f
            froms.append(f if isinstance(f, list) else [f])
        else:
            froms.append(None)

    head_from = head.f if isinstance(head.f, list) else [head.f]

    def run(x: torch.Tensor):
        y = []
        for m, f_idx in zip(body, froms):
            if f_idx is not None:
                inputs = [x if j == -1 else y[j] for j in f_idx]
                x_in = inputs[0] if len(inputs) == 1 else inputs
            else:
                x_in = x
            x = m(x_in)
            y.append(x)
        return [y[j] for j in head_from]

    return run
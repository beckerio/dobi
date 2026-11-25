# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Construct monitor heads from checkpoints and decode them into threshold-indexed curves.

Shared by evaluation and visualization so every head type is decoded exactly
the way it was trained:

* tau_spline_single  -> uniform anchors, linear resampling (decode_tau_spline_head, use_shared_anchor_grid=False)
* tau_shared_anchor  -> per-sample anchors, cubic Hermite   (decode_tau_spline_head, use_shared_anchor_grid=True)
* tau_behavior       -> global knots, linear or PCHIP       (TauBehaviorHead.decode)
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from dobi.models.shared_anchor_monitor_model import TauSplineSharedAnchorHead
from dobi.models.spline_monitor_model import TauSplineSingleHead
from dobi.models.tau_behavior_model import TauBehaviorHead
from dobi.models.pc1_behavior_model import PC1BehaviorHead
from dobi.utils.spline_helpers import decode_tau_spline_head

HEAD_TYPES = ("tau_spline_single", "tau_shared_anchor", "tau_behavior", "pc1_behavior")
LEGACY_HEAD_TYPES = ("tau_spline_single", "tau_shared_anchor")


def build_monitor_head(head_type: str, feat_channels: List[int], train_args: Dict) -> nn.Module:
    hidden = int(train_args.get("hidden", 1024))
    layers = int(train_args.get("mlp_layers", 2))

    if head_type == "tau_spline_single":
        return TauSplineSingleHead(
            feat_channels=feat_channels,
            hidden_width=hidden,
            n_anchors=int(train_args.get("n_anchors", 8)),
            num_hidden_layers=layers,
        )
    if head_type == "tau_shared_anchor":
        return TauSplineSharedAnchorHead(
            feat_channels=feat_channels,
            hidden_width=hidden,
            n_anchors=int(train_args.get("n_anchors", 8)),
            num_hidden_layers=layers,
            learn_shared_tau=bool(train_args.get("learn_shared_tau", False)),
        )
    if head_type == "tau_behavior":
        knots = train_args.get("knots")
        if knots is None:
            raise RuntimeError("tau_behavior checkpoint has no 'knots' in train_args.")
        return TauBehaviorHead(
            feat_channels=feat_channels,
            knots=knots,
            hidden_width=hidden,
            num_hidden_layers=layers,
            interp=str(train_args.get("interp", "pchip")),
            fppi_space=str(train_args.get("fppi_space", "log1p")),
            precision_envelope=bool(train_args.get("precision_envelope", False)),
        )
    if head_type == "pc1_behavior":
        return PC1BehaviorHead(feat_channels=feat_channels, pca=train_args["pca"], hidden_width=hidden,
                               num_hidden_layers=layers)
    raise ValueError(f"Unsupported head type '{head_type}', expected one of {HEAD_TYPES}")


def decode_monitor_output(
    head_type: str,
    monitor_head: nn.Module,
    head_out: Dict[str, torch.Tensor],
    tau_grid: torch.Tensor,
    u_max: float,
    enforce_monotonic: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    enforce_monotonic only affects legacy heads (cummax post-processing);
    tau_behavior is monotone by construction.

    Returns the common keys tpr_tau, recall_tau, fpr_tau, fppi_tau, precision_tau, f1_tau ([B,Q])."""
    if head_type in ("tau_behavior", "pc1_behavior"):
        return monitor_head.decode(head_out, tau_grid)
    if head_type in LEGACY_HEAD_TYPES:
        return decode_tau_spline_head(
            head_out=head_out,
            tau_grid=tau_grid,
            u_max=u_max,
            enforce_monotonic=enforce_monotonic,
            use_shared_anchor_grid=(head_type == "tau_shared_anchor"),
        )
    raise ValueError(f"Unsupported head type '{head_type}'")

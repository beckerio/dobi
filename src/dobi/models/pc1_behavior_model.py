# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
PC1 behavior baseline (appendix).

Behavior vector of a regime on the training tau grid:
    b = [TPR(tau), FPR(tau), log1p(FPPI(tau)), Precision(tau)]
A PCA is fitted on the TRAINING regimes only. The monitor (same GAP+GMP readout and
MLP trunk as TauBehaviorHead) predicts the standardised PC1 coefficient of the
regime that generated the image; behavior is reconstructed as
    b_hat = mean + c_hat * pc1
and evaluated on any query grid by linear interpolation. TPR, FPR and FPPI are
projected to non-increasing functions (running minimum over tau) and all
quantities are clipped to their valid ranges, because a single linear component
does not guarantee these constraints.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from dobi.models.spline_monitor_model import _make_mlp
from dobi.models.tau_behavior_model import pool_gap_gmp
from dobi.utils.tau_spline import linear_interp, safe_f1

PARTS = ("tpr", "fpr", "fppi", "precision")


def behavior_matrix(targets: Dict[str, np.ndarray]) -> np.ndarray:
    """[S, 4T] behavior vectors from build_tau_behavior_targets outputs."""
    return np.concatenate(
        [targets["tpr"], targets["fpr"], np.log1p(np.clip(targets["fppi"], 0, None)), targets["precision"]], axis=1
    ).astype(np.float64)


def fit_behavior_pca(targets: Dict[str, np.ndarray], tau_grid: np.ndarray) -> Dict:
    X = behavior_matrix(targets)
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    basis = Vt[0]
    # orient PC1 so that a larger coefficient means a lower best F1 (stronger degradation)
    coefs = (X - mean) @ basis
    if np.corrcoef(coefs, targets["f1"].max(axis=1))[0, 1] > 0:
        basis, coefs = -basis, -coefs
    var = S ** 2 / np.sum(S ** 2)
    recon = mean + np.outer(coefs, basis)
    return {
        "tau_grid": np.asarray(tau_grid, dtype=np.float64).tolist(),
        "mean": mean.tolist(),
        "basis": basis.tolist(),
        "coef_mean": float(coefs.mean()),
        "coef_std": float(coefs.std() + 1e-8),
        "train_coefs": coefs.tolist(),
        "explained_variance_ratio": var.tolist(),
        "train_recon_rmse": float(np.sqrt(np.mean((recon - X) ** 2))),
    }


def project_pc1(pca: Dict, targets: Dict[str, np.ndarray]) -> np.ndarray:
    X = behavior_matrix(targets)
    return (X - np.asarray(pca["mean"])) @ np.asarray(pca["basis"])


class PC1BehaviorHead(nn.Module):
    needs_gt_roi = False
    needs_tp_fp_roi = False

    def __init__(self, feat_channels: List[int], pca: Dict, hidden_width: int = 1024, num_hidden_layers: int = 2):
        super().__init__()
        self.pca = {k: v for k, v in pca.items()}
        self.register_buffer("grid", torch.tensor(pca["tau_grid"], dtype=torch.float32), persistent=True)
        self.register_buffer("mean", torch.tensor(pca["mean"], dtype=torch.float32), persistent=True)
        self.register_buffer("basis", torch.tensor(pca["basis"], dtype=torch.float32), persistent=True)
        self.coef_mean = float(pca["coef_mean"])
        self.coef_std = float(pca["coef_std"])
        self.trunk = _make_mlp(int(sum(feat_channels) * 2), hidden_width, num_hidden_layers)
        self.out = nn.Linear(hidden_width, 1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.trunk(pool_gap_gmp(feats)).float()
        return {"coef_norm": self.out(h)[:, 0]}

    def decode(self, head_out: Dict[str, torch.Tensor], tau_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        tau = tau_grid.float()
        with torch.autocast(device_type=tau.device.type, enabled=False):
            coef = head_out["coef_norm"].float() * self.coef_std + self.coef_mean
            vec = self.mean.unsqueeze(0) + coef.unsqueeze(1) * self.basis.unsqueeze(0)
            T = self.grid.numel()
            parts = {k: linear_interp(self.grid, vec[:, i * T:(i + 1) * T], tau) for i, k in enumerate(PARTS)}
            tpr = torch.cummin(parts["tpr"].clamp(0.0, 1.0), dim=1).values
            fpr = torch.cummin(parts["fpr"].clamp(0.0, 1.0), dim=1).values
            fppi_v = torch.cummin(parts["fppi"].clamp_min(0.0), dim=1).values
            precision = parts["precision"].clamp(0.0, 1.0)
            fppi = torch.expm1(fppi_v)
            f1 = safe_f1(precision, tpr)
        return {"tau": tau, "tpr_tau": tpr, "recall_tau": tpr, "fpr_tau": fpr, "fppi_tau": fppi,
                "fppi_space_tau": fppi_v, "precision_tau": precision, "f1_tau": f1, "coef": coef}

    def config(self) -> Dict:
        return {"pca": self.pca}

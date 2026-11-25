# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import torch
import torch.nn.functional as F

def _pchip_slopes(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [A], y: [B,A]
    h = x[1:] - x[:-1]                      # [A-1]
    delta = (y[:, 1:] - y[:, :-1]) / h.unsqueeze(0)

    B, A = y.shape
    m = torch.zeros_like(y)

    if A == 2:
        m[:, 0] = delta[:, 0]
        m[:, 1] = delta[:, 0]
        return m

    m[:, 0] = delta[:, 0]
    m[:, -1] = delta[:, -1]

    same_sign = (delta[:, :-1] * delta[:, 1:]) > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]

    num = (w1 + w2).unsqueeze(0)
    den = (w1.unsqueeze(0) / (delta[:, :-1] + eps)) + (w2.unsqueeze(0) / (delta[:, 1:] + eps))
    m_mid = num / (den + eps)
    m[:, 1:-1] = torch.where(same_sign, m_mid, torch.zeros_like(m_mid))
    return m


def _cubic_hermite_eval(xk: torch.Tensor, yk: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    # xk: [A], yk: [B,A], xq: [K]
    mk = _pchip_slopes(xk, yk)
    idx = torch.bucketize(xq, xk[1:-1], right=False)
    idx = idx.clamp(0, xk.numel() - 2)

    idx_b = idx.unsqueeze(0).expand(yk.shape[0], -1)

    x0 = xk[idx]
    x1 = xk[idx + 1]
    h = (x1 - x0).clamp_min(1e-8)
    t = (xq - x0) / h

    y0 = torch.gather(yk, 1, idx_b)
    y1 = torch.gather(yk, 1, idx_b + 1)
    m0 = torch.gather(mk, 1, idx_b)
    m1 = torch.gather(mk, 1, idx_b + 1)

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    yq = h00.unsqueeze(0) * y0 + h10.unsqueeze(0) * h.unsqueeze(0) * m0 \
       + h01.unsqueeze(0) * y1 + h11.unsqueeze(0) * h.unsqueeze(0) * m1
    return yq


def _linear_interp(xk: torch.Tensor, yk: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    idx = torch.bucketize(xq, xk[1:-1], right=False)
    idx = idx.clamp(0, xk.numel() - 2)

    idx_b = idx.unsqueeze(0).expand(yk.shape[0], -1)

    x0 = xk[idx]
    x1 = xk[idx + 1]
    y0 = torch.gather(yk, 1, idx_b)
    y1 = torch.gather(yk, 1, idx_b + 1)

    t = ((xq - x0) / (x1 - x0).clamp_min(1e-8)).unsqueeze(0)
    return y0 + t * (y1 - y0)


def render_tau_spline_outputs(head_out: dict, tau_grid: torch.Tensor, u_max: float) -> dict:
    xk = head_out["tau_anchor"][0].to(tau_grid.device)  # fixed across batch

    tpr = _cubic_hermite_eval(xk, head_out["tpr_anchor"], tau_grid).clamp(0.0, 1.0)
    fpr = _cubic_hermite_eval(xk, head_out["fpr_anchor"], tau_grid).clamp(0.0, 1.0)
    fppi_norm = _cubic_hermite_eval(xk, head_out["fppi_anchor_norm"], tau_grid).clamp(0.0, 1.0)
    precision = _linear_interp(xk, head_out["precision_anchor"], tau_grid).clamp(0.0, 1.0)

    fppi = fppi_norm * float(u_max)
    recall = tpr
    f1 = (2.0 * precision * recall) / torch.clamp(precision + recall, min=1e-8)

    return {
        "tau": tau_grid.unsqueeze(0).expand(tpr.shape[0], -1),
        "tpr": tpr,
        "fpr": fpr,
        "fppi": fppi,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def interp_uniform_anchors(y_anchor: torch.Tensor, n_out: int) -> torch.Tensor:
    """
    y_anchor: [B, A]
    returns:  [B, n_out]
    """
    return F.interpolate(
        y_anchor.unsqueeze(1),  # [B,1,A]
        size=n_out,
        mode="linear",
        align_corners=True,
    ).squeeze(1)

def make_nonincreasing(y: torch.Tensor) -> torch.Tensor:
    return torch.flip(
        torch.cummax(torch.flip(y, dims=[1]), dim=1).values,
        dims=[1],
    )

def decode_tau_spline_head(
    head_out,
    tau_grid,
    u_max,
    enforce_monotonic=False,
    use_shared_anchor_grid=True,
):
    k_tau = len(tau_grid)

    if use_shared_anchor_grid:
        xk = head_out["tau_anchor"].to(device=tau_grid.device, dtype=tau_grid.dtype)

        tpr_tau = _cubic_hermite_eval_batched(xk, head_out["tpr_anchor"], tau_grid).clamp(0.0, 1.0)
        fpr_tau = _cubic_hermite_eval_batched(xk, head_out["fpr_anchor"], tau_grid).clamp(0.0, 1.0)
        fppi_norm_tau = _cubic_hermite_eval_batched(xk, head_out["fppi_anchor_norm"], tau_grid).clamp(0.0, 1.0)
        precision_tau = _linear_interp_batched(xk, head_out["precision_anchor"], tau_grid).clamp(0.0, 1.0)
    else:
        tpr_tau = interp_uniform_anchors(head_out["tpr_anchor"], k_tau).clamp(0.0, 1.0)
        fpr_tau = interp_uniform_anchors(head_out["fpr_anchor"], k_tau).clamp(0.0, 1.0)
        fppi_norm_tau = interp_uniform_anchors(head_out["fppi_anchor_norm"], k_tau).clamp(0.0, 1.0)
        precision_tau = interp_uniform_anchors(head_out["precision_anchor"], k_tau).clamp(0.0, 1.0)

    if enforce_monotonic:
        tpr_tau = make_nonincreasing(tpr_tau)
        fpr_tau = make_nonincreasing(fpr_tau)
        fppi_norm_tau = make_nonincreasing(fppi_norm_tau)

    fppi_tau = fppi_norm_tau * float(u_max)
    recall_tau = tpr_tau
    f1_tau = 2 * precision_tau * recall_tau / (precision_tau + recall_tau + 1e-8)

    return {
        "tau": tau_grid,
        "tpr_tau": tpr_tau,
        "recall_tau": recall_tau,
        "fpr_tau": fpr_tau,
        "fppi_norm_tau": fppi_norm_tau,
        "fppi_tau": fppi_tau,
        "precision_tau": precision_tau,
        "f1_tau": f1_tau,
    }

def _cubic_hermite_eval_batched(xk: torch.Tensor, yk: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """
    xk: [A] or [B,A]
    yk: [B,A]
    xq: [K]
    returns: [B,K]
    """
    if xk.dim() == 1:
        return _cubic_hermite_eval(xk, yk, xq)

    outs = []
    for b in range(yk.shape[0]):
        outs.append(_cubic_hermite_eval(xk[b], yk[b:b+1], xq))
    return torch.cat(outs, dim=0)


def _linear_interp_batched(xk: torch.Tensor, yk: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """
    xk: [A] or [B,A]
    yk: [B,A]
    xq: [K]
    returns: [B,K]
    """
    if xk.dim() == 1:
        return _linear_interp(xk, yk, xq)

    outs = []
    for b in range(yk.shape[0]):
        outs.append(_linear_interp(xk[b], yk[b:b+1], xq))
    return torch.cat(outs, dim=0)
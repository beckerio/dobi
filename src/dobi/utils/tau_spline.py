# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Threshold-indexed interpolation and operating-point utilities.

All functions operate on a *global* set of threshold knots ``xk`` (shape [K],
strictly increasing) shared by every sample in the batch, control values
``yk`` (shape [B, K]) and query thresholds ``xq`` (shape [Q]).

Two interpolation schemes are provided:

* ``linear``: piecewise-linear interpolation (debugging / representation baseline)
* ``pchip`` : shape-preserving monotone cubic Hermite interpolation with
  Fritsch--Butland (harmonic-mean) interior slopes, i.e. the slope rule used by
  Fritsch--Carlson-type PCHIP schemes. Monotone control values give a monotone
  interpolant, local extrema get zero slope, so there is no overshoot.

Both reproduce the control values exactly at the knots and are differentiable.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

INTERP_METHODS = ("linear", "pchip")
OP_CRITERIA = ("f1", "youden", "fppi")


# ============================================================
# Interpolation
# ============================================================

def _segment_index(xk: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    # segment i covers [xk[i], xk[i+1]]; queries outside are extrapolated from the end segments
    idx = torch.bucketize(xq.contiguous(), xk[1:-1].contiguous(), right=False)
    return idx.clamp(0, xk.numel() - 2)


def linear_interp(xk: torch.Tensor, yk: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """xk: [K], yk: [B,K], xq: [Q] -> [B,Q]"""
    idx = _segment_index(xk, xq)
    x0 = xk[idx]
    h = (xk[idx + 1] - x0).clamp_min(1e-12)
    t = ((xq - x0) / h).unsqueeze(0)
    y0 = yk[:, idx]
    y1 = yk[:, idx + 1]
    return y0 + t * (y1 - y0)


def pchip_slopes(xk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:
    """
    Shape-preserving knot slopes, xk: [K], yk: [B,K] -> [B,K].

    Interior slopes use the weighted harmonic mean of the neighbouring secants
    (Fritsch & Butland 1984); they are zero where the secants change sign or one
    of them vanishes. End slopes use the one-sided secant. Together this keeps
    |m| <= 3 * min(|delta_left|, |delta_right|), the Fritsch--Carlson sufficient
    condition for monotonicity on every interval.
    """
    h = xk[1:] - xk[:-1]                                  # [K-1]
    delta = (yk[:, 1:] - yk[:, :-1]) / h.unsqueeze(0)     # [B,K-1]

    if yk.shape[1] == 2:
        return torch.cat([delta, delta], dim=1)

    d0 = delta[:, :-1]
    d1 = delta[:, 1:]
    w1 = (2.0 * h[1:] + h[:-1]).unsqueeze(0)
    w2 = (h[1:] + 2.0 * h[:-1]).unsqueeze(0)

    # m = (w1 + w2) / (w1/d0 + w2/d1) = (w1 + w2) d0 d1 / (w1 d1 + w2 d0)
    same_sign = (d0 * d1) > 0
    den = w1 * d1 + w2 * d0
    # double-where: never divide by a (near-)zero denominator, not even in the masked branch
    safe_den = torch.where(same_sign, den, torch.ones_like(den))
    m_mid = torch.where(same_sign, (w1 + w2) * d0 * d1 / safe_den, torch.zeros_like(den))

    return torch.cat([delta[:, :1], m_mid, delta[:, -1:]], dim=1)


def pchip_interp(xk: torch.Tensor, yk: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """xk: [K], yk: [B,K], xq: [Q] -> [B,Q]"""
    mk = pchip_slopes(xk, yk)
    idx = _segment_index(xk, xq)

    x0 = xk[idx]
    h = (xk[idx + 1] - x0).clamp_min(1e-12)
    t = ((xq - x0) / h).clamp(0.0, 1.0)  # constant extrapolation outside [xk[0], xk[-1]]

    y0 = yk[:, idx]
    y1 = yk[:, idx + 1]
    m0 = mk[:, idx]
    m1 = mk[:, idx + 1]

    t2 = t * t
    t3 = t2 * t
    h00 = (2 * t3 - 3 * t2 + 1).unsqueeze(0)
    h10 = (t3 - 2 * t2 + t).unsqueeze(0)
    h01 = (-2 * t3 + 3 * t2).unsqueeze(0)
    h11 = (t3 - t2).unsqueeze(0)
    hh = h.unsqueeze(0)

    return h00 * y0 + h10 * hh * m0 + h01 * y1 + h11 * hh * m1


def interpolate(xk: torch.Tensor, yk: torch.Tensor, xq: torch.Tensor, method: str = "pchip") -> torch.Tensor:
    if method == "linear":
        return linear_interp(xk, yk, xq)
    if method == "pchip":
        return pchip_interp(xk, yk, xq)
    raise ValueError(f"Unknown interpolation method '{method}', expected one of {INTERP_METHODS}")


# ============================================================
# Knot placement
# ============================================================

def uniform_knots(n_knots: int, tau_min: float = 0.0, tau_max: float = 1.0) -> np.ndarray:
    if n_knots < 2:
        raise ValueError("n_knots must be >= 2")
    return np.linspace(tau_min, tau_max, int(n_knots), dtype=np.float64)


def weighted_quantile_knots(
    values: np.ndarray,
    weights: np.ndarray,
    n_knots: int,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
) -> np.ndarray:
    """
    Knots at the empirical quantiles of a weighted score sample.

    The first and last knot are pinned to ``tau_min``/``tau_max`` so the knots
    always cover the full detector-score interval. Interior knots sit at the
    quantile levels k/(K-1), k = 1..K-2. Coinciding quantiles (heavy score mass
    at one value) are resolved by splitting the widest remaining gap until K
    distinct knots exist.
    """
    if n_knots < 2:
        raise ValueError("n_knots must be >= 2")

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = (weights > 0) & np.isfinite(values) & (values > tau_min) & (values < tau_max)
    values, weights = values[keep], weights[keep]

    knots = [float(tau_min), float(tau_max)]
    if values.size and n_knots > 2:
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        cdf = np.cumsum(weights) / weights.sum()
        levels = np.arange(1, n_knots - 1, dtype=np.float64) / (n_knots - 1)
        q = values[np.clip(np.searchsorted(cdf, levels, side="left"), 0, values.size - 1)]
        knots.extend(q.tolist())

    knots = np.unique(np.asarray(knots, dtype=np.float64))
    while knots.size < n_knots:
        gaps = np.diff(knots)
        i = int(np.argmax(gaps))
        knots = np.sort(np.r_[knots, 0.5 * (knots[i] + knots[i + 1])])
    return knots


# ============================================================
# F1 and operating points
# ============================================================

def safe_f1(precision, recall):
    """F1 = 2PR/(P+R) with F1 := 0 where P+R == 0. Works for torch tensors and numpy arrays."""
    if torch.is_tensor(precision):
        den = precision + recall
        pos = den > 0
        safe_den = torch.where(pos, den, torch.ones_like(den))
        return torch.where(pos, 2.0 * precision * recall / safe_den, torch.zeros_like(den))

    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    den = precision + recall
    out = np.zeros_like(den)
    pos = den > 0
    out[pos] = 2.0 * precision[pos] * recall[pos] / den[pos]
    return out


def operating_scores(curves: Dict[str, torch.Tensor], criterion: str = "f1", max_fppi: float = 0.1) -> torch.Tensor:
    """
    Per-threshold utility of a criterion, computed only from the given
    threshold-indexed curves ([B,Q] each). Infeasible thresholds get -inf.
    """
    if criterion == "f1":
        if "f1_tau" in curves:
            return curves["f1_tau"]
        return safe_f1(curves["precision_tau"], curves["tpr_tau"])
    if criterion == "youden":
        return curves["tpr_tau"] - curves["fpr_tau"]
    if criterion == "fppi":
        tpr = curves["tpr_tau"]
        feasible = curves["fppi_tau"] <= float(max_fppi)
        return torch.where(feasible, tpr, torch.full_like(tpr, -float("inf")))
    raise ValueError(f"Unknown criterion '{criterion}', expected one of {OP_CRITERIA}")


def select_operating_threshold(
    tau: torch.Tensor,
    curves: Dict[str, torch.Tensor],
    criterion: str = "f1",
    max_fppi: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hard operating point: tau* = argmax_tau utility(tau).

    tau: [Q] threshold grid, curves: dict of [B,Q] predicted functions.
    Returns (tau_star [B], index [B]). Uses only the predicted curves; if no
    threshold satisfies an FPPI constraint, the threshold with the lowest
    predicted FPPI is returned.
    """
    score = operating_scores(curves, criterion=criterion, max_fppi=max_fppi)
    if criterion == "fppi":
        none_ok = torch.isinf(score).all(dim=1)
        if none_ok.any():
            fallback = torch.argmin(curves["fppi_tau"], dim=1)
            idx = torch.where(none_ok, fallback, torch.argmax(score, dim=1))
            return tau[idx], idx
    idx = torch.argmax(score, dim=1)
    return tau[idx], idx


def soft_operating_threshold(tau: torch.Tensor, score: torch.Tensor, temperature: float = 0.02) -> torch.Tensor:
    """Differentiable soft-argmax: sum_tau softmax(score/T) * tau. score: [B,Q], tau: [Q] -> [B]"""
    w = F.softmax(score / max(float(temperature), 1e-8), dim=1)
    return (w * tau.unsqueeze(0)).sum(dim=1)


def uniform_grid_interp(xq: torch.Tensor, x_max: float, y: torch.Tensor) -> torch.Tensor:
    """
    Differentiable lookup of curves sampled on a uniform grid linspace(0, x_max, G).
    xq: [B,Q] query x values, y: [B,G] -> [B,Q]. Queries are clamped to [0, x_max].
    """
    G = y.shape[1]
    pos = (xq.clamp(0.0, float(x_max)) / float(x_max)) * (G - 1)
    i0 = pos.floor().long().clamp(0, G - 2)
    t = pos - i0.to(pos.dtype)
    y0 = torch.gather(y, 1, i0)
    y1 = torch.gather(y, 1, i0 + 1)
    return y0 + t * (y1 - y0)

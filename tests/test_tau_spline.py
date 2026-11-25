# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import numpy as np
import pytest
import torch

from dobi.utils.tau_spline import (
    interpolate,
    linear_interp,
    pchip_interp,
    safe_f1,
    select_operating_threshold,
    soft_operating_threshold,
    uniform_grid_interp,
    weighted_quantile_knots,
)

torch.manual_seed(0)
KNOTS = torch.tensor([0.0, 0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0], dtype=torch.float64)
DENSE = torch.linspace(0, 1, 2001, dtype=torch.float64)


def _random_monotone_decreasing(B, K):
    steps = torch.rand(B, K - 1, dtype=torch.float64)
    steps[:, ::3] = 0.0  # flat segments (plateaus)
    y = torch.cat([torch.zeros(B, 1, dtype=torch.float64), torch.cumsum(steps, 1)], 1)
    return 1.0 - y / y[:, -1:].clamp_min(1e-9)


@pytest.mark.parametrize("method", ["linear", "pchip"])
def test_exact_at_knots(method):
    y = torch.rand(5, len(KNOTS), dtype=torch.float64)
    out = interpolate(KNOTS, y, KNOTS, method)
    assert torch.allclose(out, y, atol=1e-12)


@pytest.mark.parametrize("method", ["linear", "pchip"])
def test_monotone_preserved(method):
    y = _random_monotone_decreasing(64, len(KNOTS))
    out = interpolate(KNOTS, y, DENSE, method)
    assert torch.all(out[:, 1:] - out[:, :-1] <= 1e-12)


def test_pchip_no_overshoot_on_non_monotone_data():
    y = torch.rand(64, len(KNOTS), dtype=torch.float64)  # e.g. precision controls in [0,1]
    out = pchip_interp(KNOTS, y, DENSE)
    lo = y.min(dim=1, keepdim=True).values
    hi = y.max(dim=1, keepdim=True).values
    assert torch.all(out >= lo - 1e-12) and torch.all(out <= hi + 1e-12)
    # stronger: stays within the two neighbouring control values on every interval
    idx = torch.bucketize(DENSE, KNOTS[1:-1])
    y0, y1 = y[:, idx], y[:, idx + 1]
    assert torch.all(out >= torch.minimum(y0, y1) - 1e-12) and torch.all(out <= torch.maximum(y0, y1) + 1e-12)


def test_stable_autograd_with_plateaus():
    y = _random_monotone_decreasing(16, len(KNOTS)).float().requires_grad_(True)
    out = pchip_interp(KNOTS.float(), y, DENSE.float())
    out.sum().backward()
    assert torch.isfinite(y.grad).all()
    assert torch.autograd.gradcheck(lambda v: pchip_interp(KNOTS, v, DENSE[::50]),
                                    (torch.rand(2, len(KNOTS), dtype=torch.float64).cumsum(1).requires_grad_(True),))


def test_linear_matches_numpy_interp():
    y = torch.rand(3, len(KNOTS), dtype=torch.float64)
    out = linear_interp(KNOTS, y, DENSE)
    ref = np.stack([np.interp(DENSE.numpy(), KNOTS.numpy(), y[b].numpy()) for b in range(3)])
    assert np.allclose(out.numpy(), ref)


def test_f1_zero_denominator_and_range():
    p = torch.tensor([0.0, 1.0, 0.5, 0.0])
    r = torch.tensor([0.0, 0.0, 0.5, 1.0])
    f1 = safe_f1(p, r)
    assert torch.allclose(f1, torch.tensor([0.0, 0.0, 0.5, 0.0]))
    p.requires_grad_(True)
    safe_f1(p, r).sum().backward()
    assert torch.isfinite(p.grad).all()
    assert np.allclose(safe_f1(np.array([0.0, 0.8]), np.array([0.0, 0.6])), [0.0, 2 * 0.48 / 1.4])


def test_synthetic_known_optimal_threshold():
    """TPR(t)=1-t^2, Precision(t)=t  =>  F1(t)=2t(1-t^2)/(t+1-t^2), maximised at t*."""
    tau = torch.linspace(0, 1, 100001, dtype=torch.float64)
    tpr = 1 - tau ** 2
    prec = tau
    curves = {"tpr_tau": tpr[None], "precision_tau": prec[None], "fpr_tau": (1 - tau)[None] ** 3,
              "fppi_tau": (10 * (1 - tau) ** 2)[None]}
    t_star, _ = select_operating_threshold(tau, curves, "f1")
    f = lambda t: 2 * t * (1 - t ** 2) / (t + 1 - t ** 2)
    grid = np.linspace(0, 1, 1000001)
    assert abs(float(t_star) - grid[np.argmax(f(grid))]) < 2e-5

    # same representation, other criteria, no retraining
    t_y, _ = select_operating_threshold(tau, curves, "youden")  # (1-t^2) - (1-t)^3 -> max at t = 2 - sqrt(2) ... check numerically
    yo = (1 - grid ** 2) - (1 - grid) ** 3
    assert abs(float(t_y) - grid[np.argmax(yo)]) < 2e-5
    t_f, _ = select_operating_threshold(tau, curves, "fppi", max_fppi=0.1)  # 10(1-t)^2 <= 0.1 -> t >= 0.9
    assert abs(float(t_f) - 0.9) < 1e-4


def test_fppi_constraint_infeasible_falls_back_to_min_fppi():
    tau = torch.linspace(0, 1, 11)
    curves = {"tpr_tau": torch.ones(1, 11), "fppi_tau": torch.full((1, 11), 5.0) - tau[None]}
    t, _ = select_operating_threshold(tau, curves, "fppi", max_fppi=0.1)
    assert float(t) == 1.0


def test_soft_threshold_approaches_hard_argmax():
    tau = torch.linspace(0, 1, 1001)
    f1 = torch.exp(-((tau - 0.31) / 0.1) ** 2)[None]
    soft = soft_operating_threshold(tau, f1, temperature=1e-3)
    assert abs(float(soft) - 0.31) < 2e-3
    f1.requires_grad_(True)
    soft_operating_threshold(tau, f1, temperature=0.02).sum().backward()
    assert torch.isfinite(f1.grad).all()


def test_uniform_grid_interp():
    g = torch.linspace(0, 2, 201)
    y = (g ** 2)[None]
    xq = torch.tensor([[0.0, 0.5, 1.234, 2.0, 3.0]])
    out = uniform_grid_interp(xq, 2.0, y)
    assert torch.allclose(out, torch.tensor([[0.0, 0.25, 1.234 ** 2, 4.0, 4.0]]), atol=1e-3)


def test_weighted_quantile_knots_with_point_mass():
    v = np.r_[np.full(1000, 0.001), np.linspace(0.1, 0.9, 50)]
    k = weighted_quantile_knots(v, np.ones_like(v), 8)
    assert len(k) == 8 and k[0] == 0.0 and k[-1] == 1.0 and np.all(np.diff(k) > 0)

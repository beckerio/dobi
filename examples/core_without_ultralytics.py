#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""DOBI core without Ultralytics: uses only Apache-2.0 modules of this repository (no detector, no data, CPU).

    python examples/core_without_ultralytics.py [--monitor weights/tau_pchip_s0.pt] [--seed 0]

1. Operating curves from detections (how the training targets are built): scores and true-positive labels of the
   detections of one image set -> threshold sweep -> recall, FPR, FPPI, precision, F1 over the threshold tau, and the
   non-deployable oracle thresholds. The detections here are synthetic.
2. Monitor head on pooled detector features -> predicted operating curves -> chosen thresholds. The input is the
   pooled feature vector of one image ([GAP, GMP] of the three YOLOv10-N feature levels, 2 x (64 + 128 + 256) = 896
   values). Here it is random, so the predicted numbers carry no meaning; the calls are the real ones. With --monitor,
   the head published with the GitHub release is loaded; without it, an untrained head with 16 uniform knots is built.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dobi.data.database.generate_curves import compute_operating_sweep_from_global_labels
from dobi.models.monitor_factory import build_monitor_head, decode_monitor_output
from dobi.utils.tau_spline import select_operating_threshold
from dobi.utils.tau_targets import dedupe_sweep, oracle_threshold, sweep_at_tau

FEAT_CHANNELS = [64, 128, 256]  # YOLOv10-N feature levels read by the monitor heads
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])


def table(rows, tau):
    print("   tau   recall      FPR     FPPI    prec.       F1")
    for i, t in enumerate(tau):
        print(f" {t:5.3f} " + " ".join(f"{float(rows[k][i]):8.4f}" for k in ("recall", "fpr", "fppi", "precision", "f1")))


def curves_from_detections(rng):
    n_imgs, n_gt = 200, 450
    n_tp, n_fp = 380, 2000
    scores = np.r_[rng.beta(5, 2, n_tp), rng.beta(1, 6, n_fp)].astype(np.float32)   # true positives score higher
    is_tp = np.r_[np.ones(n_tp, bool), np.zeros(n_fp, bool)]
    sweep = compute_operating_sweep_from_global_labels(scores, is_tp, num_gt=n_gt, num_imgs=n_imgs)
    s = sweep_at_tau(dedupe_sweep(sweep), TAUS)
    print(f"1. operating curves of {n_imgs} images ({n_gt} objects, {n_tp + n_fp} synthetic detections)")
    table(s, TAUS)
    print(f"   oracle threshold, best F1: tau = {oracle_threshold(sweep, 'f1'):.3f}; "
          f"FPPI <= 0.1: tau = {oracle_threshold(sweep, 'fppi', max_fppi=0.1):.3f}")


def pseudo_feature_maps(pooled: torch.Tensor):
    """Pooled [GAP_l, GMP_l] vector -> per level a [B, C, 1, 2] map whose spatial mean is GAP and max is GMP; the
    heads pool their input with mean and max, so they receive exactly the pooled vector."""
    out, o = [], 0
    for c in FEAT_CHANNELS:
        gap, gmp = pooled[:, o:o + c], pooled[:, o + c:o + 2 * c]
        o += 2 * c
        gmp = torch.maximum(gmp, gap)
        out.append(torch.stack([gmp, 2.0 * gap - gmp], dim=-1).unsqueeze(2))
    return out


def monitor_prediction(rng, monitor):
    if monitor:
        ckpt = torch.load(monitor, map_location="cpu", weights_only=True)
        train_args, tau_grid = ckpt["train_args"], torch.as_tensor(np.asarray(ckpt["tau_grid"], np.float32))
        head = build_monitor_head(train_args["head"], FEAT_CHANNELS, train_args)
        head.load_state_dict(ckpt["monitor_state_dict"])
        what = f"monitor head {monitor}"
    else:
        torch.manual_seed(0)
        train_args = {"head": "tau_behavior", "knots": np.linspace(0.0, 1.0, 16).tolist(), "interp": "pchip",
                      "fppi_space": "log1p", "hidden": 1024, "mlp_layers": 2}
        head = build_monitor_head("tau_behavior", FEAT_CHANNELS, train_args)
        tau_grid = torch.linspace(0.0, 1.0, 201)
        what = "untrained monitor head (16 uniform knots, PCHIP)"
    head.eval()
    pooled = torch.from_numpy(np.abs(rng.standard_normal((1, 2 * sum(FEAT_CHANNELS)))).astype(np.float32))
    with torch.no_grad():
        out = head(pseudo_feature_maps(pooled))
        curves = decode_monitor_output(train_args["head"], head, out, tau_grid, u_max=float(train_args.get("u_max", 2.0)))
    idx = [int(torch.argmin((tau_grid - t).abs())) for t in TAUS]
    rows = {k.replace("_tau", ""): curves[k][0, idx].numpy() for k in ("recall_tau", "fpr_tau", "fppi_tau", "precision_tau", "f1_tau")}
    print(f"\n2. predicted operating curves, {what}, on a random pooled feature vector")
    table(rows, tau_grid[idx].numpy())
    t_f1, _ = select_operating_threshold(tau_grid, curves, "f1")
    t_fppi, _ = select_operating_threshold(tau_grid, curves, "fppi", max_fppi=0.1)
    print(f"   chosen threshold, best predicted F1: tau = {float(t_f1[0]):.3f}; predicted FPPI <= 0.1: tau = {float(t_fppi[0]):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--monitor", default=None, help="monitor head from the GitHub release, e.g. weights/tau_pchip_s0.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    curves_from_detections(rng)
    monitor_prediction(rng, a.monitor)


if __name__ == "__main__":
    main()

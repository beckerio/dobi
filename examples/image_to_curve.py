#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""One image -> predicted operating behavior of the frozen detector (recall, FPR, FPPI, precision, F1 over the
confidence threshold tau), and the thresholds a deployment would pick from it.

    python examples/image_to_curve.py --image path/to/image.jpg \
        --detector weights/yolov10n.pt --monitor weights/tau_pchip_s0.pt [--plot curve.png]

Detector: the unchanged Ultralytics YOLOv10-N release (see weights/README.md).
Monitor:  one of the monitor heads published with the GitHub release (see weights/README.md).
The image is pre-processed exactly as in training (longest side 640 px, letterbox to 640x640) and the
detector is used unfused in fp32, as for the feature cache the monitors were trained on. CPU is fine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dobi.eval.evaluation_curve_monitor import MonitorPredictor, _load_monitor_head_from_ckpt, select_tau_from_curves
from dobi.models.monitor_factory import build_monitor_head
from dobi.models.spline_monitor_model import build_backbone_runner


def load(detector: str, monitor: str, device: torch.device):
    from ultralytics import YOLO  # AGPL-3.0 dependency

    ckpt = torch.load(monitor, map_location="cpu", weights_only=True)
    train_args = ckpt["train_args"]
    head_type = train_args["head"]
    imgsz = int(train_args.get("imgsz", 640))
    yolo = YOLO(detector)
    yolo.model.eval()
    yolo.to(device)
    for p in yolo.model.parameters():
        p.requires_grad_(False)
    backbone = build_backbone_runner(yolo.model)
    with torch.no_grad():
        feat_channels = [int(f.shape[1]) for f in backbone(torch.zeros(1, 3, imgsz, imgsz, device=device))]
    head = build_monitor_head(head_type, feat_channels, train_args)
    _load_monitor_head_from_ckpt(head, ckpt)
    head.to(device).eval()
    tau_grid = np.asarray(ckpt["tau_grid"], dtype=np.float64)
    return MonitorPredictor(backbone, head, head_type, tau_grid, float(train_args.get("u_max", 2.0)), imgsz, device), tau_grid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--detector", default="weights/yolov10n.pt")
    ap.add_argument("--monitor", default="weights/tau_pchip_s0.pt")
    ap.add_argument("--max_fppi", type=float, default=0.1, help="false-positive budget per image for the FPPI policy")
    ap.add_argument("--plot", default=None, help="optional output image of the predicted curves")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    predictor, tau_grid = load(a.detector, a.monitor, torch.device(a.device))
    _, curves = predictor(Path(a.image))

    tau_f1 = select_tau_from_curves(tau_grid, curves, criterion="f1")
    tau_fppi = select_tau_from_curves(tau_grid, curves, criterion="fppi", max_fppi=a.max_fppi)
    print(f"image: {a.image}")
    print(f"{'tau':>6} {'recall':>7} {'FPR':>7} {'FPPI':>7} {'prec.':>7} {'F1':>7}")
    for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        i = int(np.argmin(np.abs(tau_grid - t)))
        print(f"{tau_grid[i]:6.3f} {curves['recall_tau'][i]:7.4f} {curves['fpr_tau'][i]:7.4f} {curves['fppi_tau'][i]:7.4f} "
              f"{curves['precision_tau'][i]:7.4f} {curves['f1_tau'][i]:7.4f}")
    print(f"threshold with the highest predicted F1: tau = {tau_f1:.3f}")
    print(f"lowest threshold with predicted FPPI <= {a.max_fppi}: tau = {tau_fppi:.3f}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for k, lab in [("recall_tau", "recall"), ("precision_tau", "precision"), ("f1_tau", "F1"), ("fppi_tau", "FPPI")]:
            ax.plot(tau_grid, curves[k], label=lab)
        ax.axvline(tau_f1, color="0.5", ls="--", lw=0.8)
        ax.set_xlabel("confidence threshold tau")
        ax.set_ylim(0, max(1.0, float(np.max(curves["fppi_tau"]))))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print("plot written to", a.plot)


if __name__ == "__main__":
    main()

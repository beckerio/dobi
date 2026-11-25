# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
import sqlite3

import cv2
import numpy as np
import torch

import dobi.eval.evaluation_curve_monitor as ev
from dobi.models.tau_behavior_model import TauBehaviorHead
from dobi.utils.roc_db_helpers import _cummax, _interp_curve, _precision_envelope, _prepare_xy


# ---------------------------------------------------------------------------
# Frozen copy of the curve conversion of the original evaluation script
# (before the tau_behavior integration). Must stay byte-for-byte equivalent.
# ---------------------------------------------------------------------------
def _reference_geometric(fpr_tau, tpr_tau, fppi_tau, precision_tau, recall_tau, fpr_grid_np, fppi_grid_np, recall_grid_np):
    fpr_src, tpr_src = _prepare_xy(fpr_tau, tpr_tau, reduce="max")
    if tpr_src.size:
        tpr_src = _cummax(tpr_src)
    pred_roc = _interp_curve(fpr_src, tpr_src, fpr_grid_np, left=0.0,
                             right=float(tpr_src[-1]) if tpr_src.size else 0.0, reduce="max")
    pred_roc = np.clip(_cummax(pred_roc), 0.0, 1.0)
    fppi_src, tpr_froc_src = _prepare_xy(fppi_tau, tpr_tau, reduce="max")
    if tpr_froc_src.size:
        tpr_froc_src = _cummax(tpr_froc_src)
    pred_froc = _interp_curve(fppi_src, tpr_froc_src, fppi_grid_np, left=0.0,
                              right=float(tpr_froc_src[-1]) if tpr_froc_src.size else 0.0, reduce="max")
    pred_froc = np.clip(_cummax(pred_froc), 0.0, 1.0)
    recall_src, precision_src = _precision_envelope(recall_tau, precision_tau)
    pred_pr = _interp_curve(recall_src, precision_src, recall_grid_np,
                            left=float(precision_src[0]) if precision_src.size else 0.0,
                            right=float(precision_src[-1]) if precision_src.size else 0.0, reduce="max")
    return pred_roc, pred_froc, np.clip(pred_pr, 0.0, 1.0)


def _reference_metrics(gt_roc, gt_froc, gt_pr, pred_roc, pred_froc, pred_pr, fg, ug, rg):
    return {
        "auroc_gt": float(np.trapezoid(gt_roc, fg)), "auroc_pred": float(np.trapezoid(pred_roc, fg)),
        "afroc_pred": float(np.trapezoid(pred_froc, ug)), "auprc_pred": float(np.trapezoid(pred_pr, rg)),
        "mad_roc": float(np.mean(np.abs(pred_roc - gt_roc))), "mad_froc": float(np.mean(np.abs(pred_froc - gt_froc))),
        "mad_pr": float(np.mean(np.abs(pred_pr - gt_pr))),
    }


def _random_tau_curves(rng, n=75):
    tpr = np.sort(rng.uniform(0, 1, n))[::-1].astype(np.float32)
    fpr = np.sort(rng.uniform(0, 1, n))[::-1].astype(np.float32)
    fppi = np.sort(rng.uniform(0, 5, n))[::-1].astype(np.float32)
    prec = rng.uniform(0, 1, n).astype(np.float32)
    return {"tpr_tau": tpr, "recall_tau": tpr, "fpr_tau": fpr, "fppi_tau": fppi, "precision_tau": prec}


def test_curve_conversion_and_metrics_unchanged():
    rng = np.random.default_rng(0)
    fg, ug, rg = np.linspace(0, 1, 201, dtype=np.float32), np.linspace(0, 2, 201, dtype=np.float32), np.linspace(0, 1, 201, dtype=np.float32)
    for _ in range(20):
        c = _random_tau_curves(rng)
        new = ev.tau_curves_to_geometric(c, fg, ug, rg)
        ref = _reference_geometric(c["fpr_tau"], c["tpr_tau"], c["fppi_tau"], c["precision_tau"], c["recall_tau"], fg, ug, rg)
        for a, b in zip(new, ref):
            assert np.array_equal(a, b)
        gt = [rng.uniform(0, 1, 201).astype(np.float32) for _ in range(3)]
        m = ev.curve_metrics(*gt, *new, fg, ug, rg)
        r = _reference_metrics(*gt, *ref, fg, ug, rg)
        for k, v in r.items():
            assert m[k] == v, k


class _FakeBackbone:
    def __call__(self, x):
        return [torch.nn.functional.adaptive_avg_pool2d(x, s).repeat(1, c // 3 + 1, 1, 1)[:, :c] for s, c in ((4, 8), (2, 16))]


def test_adaptive_path_uses_no_ground_truth(tmp_path, monkeypatch):
    """The deployable path (image -> curves -> tau_adaptive) must not touch the database."""
    img = tmp_path / "img.jpg"
    cv2.imwrite(str(img), (np.random.default_rng(0).uniform(0, 255, (48, 64, 3))).astype(np.uint8))
    head = TauBehaviorHead([8, 16], np.linspace(0, 1, 8), hidden_width=16).eval()
    tau = np.linspace(0, 1, 101, dtype=np.float32)
    pred = ev.MonitorPredictor(_FakeBackbone(), head, "tau_behavior", tau, u_max=2.0, imgsz=64, device=torch.device("cpu"))

    def _forbidden(*a, **k):
        raise AssertionError("database access in adaptive inference path")

    monkeypatch.setattr(sqlite3, "connect", _forbidden)
    grids = [np.linspace(0, 1, 51, dtype=np.float32)] * 3
    _, roc, froc, pr, tau_hat, curves = ev.predict_curves_and_tau_for_image(img, pred, *grids, criterion="f1")
    assert 0.0 <= tau_hat <= 1.0
    assert tau_hat == float(tau[int(np.argmax(curves["f1_tau"]))])
    for crit in ("youden", "fppi"):  # other operating criteria from the same prediction, no retraining
        assert 0.0 <= ev.select_tau_from_curves(tau, curves, crit, max_fppi=0.5) <= 1.0


def test_threshold_policies_scored_on_identical_detections(synthetic_db):
    db, raw = synthetic_db
    conn = sqlite3.connect(db)
    sid = raw[("test", "gaussian_blur_3_5")]["subset_id"]
    sweeps = ev.RegimeSweeps(conn, 0, 0.5)
    oracle = ev.oracle_threshold(sweeps.raw(sid), "f1")
    res = {"tau_hat": 0.4}
    ev.add_threshold_policy_metrics(res, sid, sweeps, tau_pristine=0.3, tau_oracle=oracle)
    assert res["regret_adaptive"] >= -1e-12 and res["regret_pristine"] >= -1e-12
    assert np.isclose(res["gain_adaptive"], res["f1_adaptive"] - res["f1_pristine"])
    for p in ("pristine", "adaptive", "oracle"):
        assert res[f"tp_{p}"] + res[f"fn_{p}"] == raw[("test", "gaussian_blur_3_5")]["num_gt"]
    conn.close()


def test_mean_behavior_baseline_is_image_independent(synthetic_db):
    db, _ = synthetic_db
    tau = np.linspace(0, 1, 101, dtype=np.float32)
    p = ev.MeanBehaviorPredictor(db, 0, 0.5, tau, split="train")
    _, a = p("whatever.jpg")
    _, b = p("other.jpg")
    for k in a:
        assert np.array_equal(a[k], b[k])
    assert np.all(np.diff(a["tpr_tau"]) <= 1e-7)

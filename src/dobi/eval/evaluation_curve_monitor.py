#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Sequence, Tuple, List

import random

import cv2
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

from dobi.utils.roc_db_helpers import (
    build_cluster_targets,
    build_pr_targets,
    get_clusters_with_metrics,
    get_n_examples_for_cluster,
)
from dobi.utils.img_utils import letterbox_tensor
from dobi.models.spline_monitor_model import build_backbone_runner
from dobi.models.monitor_factory import HEAD_TYPES, LEGACY_HEAD_TYPES, build_monitor_head, decode_monitor_output
from dobi.utils.tau_spline import OP_CRITERIA, safe_f1, select_operating_threshold
from dobi.utils.tau_targets import (
    build_tau_behavior_targets,
    dedupe_sweep,
    detector_metrics_at_tau,
    load_raw_sweep,
    oracle_threshold,
    pristine_validation_threshold,
    sweep_at_tau,
)

from dobi.data.feature_cache import FeatureCache, check_cache_matches, pooled_to_pseudo_feats

TAU_CURVE_KEYS = ("tpr_tau", "recall_tau", "fpr_tau", "fppi_tau", "precision_tau", "f1_tau")


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ("module.", "monitor_head.", "model.")
    for _ in range(len(prefixes) + 2):
        before = tuple(state_dict.keys())
        for prefix in prefixes:
            state_dict = _strip_prefix_if_present(state_dict, prefix)
        after = tuple(state_dict.keys())
        if before == after:
            break
    return state_dict


def _load_monitor_head_from_ckpt(monitor_head, ckpt):
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dict.")

    state = None
    used_key = None

    for key in ("monitor_state_dict", "monitor_head_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            used_key = key
            break

    for key in ("model", "state_dict"):
        if state is None and key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            used_key = key
            break

    if state is None:
        raise RuntimeError(
            "Could not find monitor head weights in checkpoint. "
            "Tried: monitor_state_dict, monitor_head_state_dict, model, state_dict"
        )

    state = _normalize_state_dict_keys(state)
    missing, unexpected = monitor_head.load_state_dict(state, strict=False)

    print(f"Loaded monitor head from ckpt['{used_key}']")
    print("monitor head missing keys:", missing[:20])
    print("monitor head unexpected keys:", unexpected[:20])

    return missing, unexpected


def preprocess_for_monitor(img_path: Path, imgsz: int, device: torch.device):
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    h0, w0 = im.shape[:2]
    r = imgsz / max(h0, w0)

    if r != 1:
        w = min(math.ceil(w0 * r), imgsz)
        h = min(math.ceil(h0 * r), imgsz)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(im_rgb).permute(2, 0, 1).contiguous().float() / 255.0

    lb_out = letterbox_tensor(t)
    t_lb = lb_out[0] if isinstance(lb_out, (tuple, list)) else lb_out

    return im_rgb, t_lb.unsqueeze(0).to(device)


def _prepare_xy(x, y, reduce="last"):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    xs, ys = [], []
    i = 0
    n = x.shape[0]

    while i < n:
        j = i + 1
        while j < n and x[j] == x[i]:
            j += 1

        xs.append(x[i])
        if reduce == "max":
            ys.append(np.max(y[i:j]))
        elif reduce == "min":
            ys.append(np.min(y[i:j]))
        elif reduce == "mean":
            ys.append(np.mean(y[i:j]))
        else:
            ys.append(y[j - 1])

        i = j

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _interp_curve(x_src, y_src, x_dst, left=None, right=None, reduce="last"):
    x_src, y_src = _prepare_xy(x_src, y_src, reduce=reduce)
    x_dst = np.asarray(x_dst, dtype=np.float32)

    if x_src.size == 0:
        return np.zeros_like(x_dst, dtype=np.float32)

    if x_src.size == 1:
        return np.full_like(x_dst, float(y_src[0]), dtype=np.float32)

    if left is None:
        left = float(y_src[0])
    if right is None:
        right = float(y_src[-1])

    return np.interp(x_dst, x_src, y_src, left=left, right=right).astype(np.float32)


def _cummax(x):
    x = np.asarray(x, dtype=np.float32)
    return np.maximum.accumulate(x)


def _precision_envelope(recall, precision):
    recall, precision = _prepare_xy(recall, precision, reduce="max")
    if recall.size == 0:
        return recall, precision
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    precision = np.clip(precision, 0.0, 1.0)
    return recall, precision



def tau_curves_to_geometric(
    tau_curves: Dict[str, np.ndarray],
    fpr_grid_np: np.ndarray,
    fppi_grid_np: np.ndarray,
    recall_grid_np: np.ndarray,
):
    """
    Reparameterise predicted threshold-indexed functions into the ROC/FROC/PR
    evaluation format (unchanged from the original evaluation code).
    """
    fpr_tau = tau_curves["fpr_tau"]
    tpr_tau = tau_curves["tpr_tau"]
    fppi_tau = tau_curves["fppi_tau"]
    precision_tau = tau_curves["precision_tau"]
    recall_tau = tau_curves["recall_tau"]

    fpr_src, tpr_src = _prepare_xy(fpr_tau, tpr_tau, reduce="max")
    if tpr_src.size:
        tpr_src = _cummax(tpr_src)

    pred_roc = _interp_curve(
        fpr_src,
        tpr_src,
        fpr_grid_np,
        left=0.0,
        right=float(tpr_src[-1]) if tpr_src.size else 0.0,
        reduce="max",
    )
    pred_roc = np.clip(_cummax(pred_roc), 0.0, 1.0)

    fppi_src, tpr_froc_src = _prepare_xy(fppi_tau, tpr_tau, reduce="max")
    if tpr_froc_src.size:
        tpr_froc_src = _cummax(tpr_froc_src)

    pred_froc = _interp_curve(
        fppi_src,
        tpr_froc_src,
        fppi_grid_np,
        left=0.0,
        right=float(tpr_froc_src[-1]) if tpr_froc_src.size else 0.0,
        reduce="max",
    )
    pred_froc = np.clip(_cummax(pred_froc), 0.0, 1.0)

    recall_src, precision_src = _precision_envelope(recall_tau, precision_tau)
    pred_pr = _interp_curve(
        recall_src,
        precision_src,
        recall_grid_np,
        left=float(precision_src[0]) if precision_src.size else 0.0,
        right=float(precision_src[-1]) if precision_src.size else 0.0,
        reduce="max",
    )
    pred_pr = np.clip(pred_pr, 0.0, 1.0)

    return pred_roc, pred_froc, pred_pr


def select_tau_from_curves(
    tau_grid_np: np.ndarray,
    tau_curves: Dict[str, np.ndarray],
    criterion: str = "f1",
    max_fppi: float = 0.1,
) -> float:
    """Hard operating threshold derived only from predicted threshold-indexed curves (no ground truth)."""
    curves_t = {k: torch.as_tensor(np.asarray(v, dtype=np.float64)).unsqueeze(0) for k, v in tau_curves.items()}
    tau_t = torch.as_tensor(np.asarray(tau_grid_np, dtype=np.float64))
    tau_star, _ = select_operating_threshold(tau_t, curves_t, criterion=criterion, max_fppi=max_fppi)
    return float(tau_star[0])


class MonitorPredictor:
    """Image -> predicted threshold-indexed curves for a trained monitor head (any supported head type)."""

    def __init__(self, backbone_forward, monitor_head: nn.Module, head_type: str, tau_grid_np: np.ndarray,
                 u_max: float, imgsz: int, device: torch.device):
        self.backbone_forward = backbone_forward
        self.monitor_head = monitor_head
        self.head_type = head_type
        self.tau_grid_np = np.asarray(tau_grid_np, dtype=np.float32)
        self.u_max = float(u_max)
        self.imgsz = int(imgsz)
        self.device = device

    def __call__(self, img_path: Path):
        img_rgb, tensor = preprocess_for_monitor(img_path, self.imgsz, self.device)
        with torch.no_grad():
            feats = self.backbone_forward(tensor)
            head_out = self.monitor_head(feats)
            decoded = decode_monitor_output(
                head_type=self.head_type,
                monitor_head=self.monitor_head,
                head_out=head_out,
                tau_grid=torch.as_tensor(self.tau_grid_np, dtype=tensor.dtype, device=self.device),
                u_max=self.u_max,
            )
        curves = {k: decoded[k][0].detach().float().cpu().numpy() for k in TAU_CURVE_KEYS}
        return img_rgb, curves


class CachedMonitorPredictor(MonitorPredictor):
    """
    Same as MonitorPredictor, but the frozen-detector pooled features are read from a feature cache
    (dobi/data/feature_cache.py) keyed by the variant's rel_path instead of being recomputed from the image.
    """

    def __init__(self, cache: FeatureCache, images_root: Path, monitor_head: nn.Module, head_type: str,
                 tau_grid_np: np.ndarray, u_max: float, device: torch.device):
        super().__init__(None, monitor_head, head_type, tau_grid_np, u_max, 0, device)
        self.cache = cache
        self.images_root = Path(images_root)

    def __call__(self, img_path: Path):
        rel = Path(img_path).relative_to(self.images_root).as_posix()
        x = torch.from_numpy(self.cache.get([self.cache.row_for_relpath(rel)])).to(self.device)
        with torch.no_grad():
            head_out = self.monitor_head(pooled_to_pseudo_feats(x, self.cache.feat_channels))
            decoded = decode_monitor_output(
                head_type=self.head_type,
                monitor_head=self.monitor_head,
                head_out=head_out,
                tau_grid=torch.as_tensor(self.tau_grid_np, dtype=torch.float32, device=self.device),
                u_max=self.u_max,
            )
        curves = {k: decoded[k][0].detach().float().cpu().numpy() for k in TAU_CURVE_KEYS}
        return None, curves


class MeanBehaviorPredictor:
    """
    Image-independent baseline: the mean threshold-indexed behavior of the
    training regimes. Every input gets the same curves (and the same threshold).
    """

    def __init__(self, db_path: str, class_id: int, iou_thr: float, tau_grid_np: np.ndarray, split: str = "train",
                 exclude_regimes: Sequence[str] = ()):
        if split == "test":
            raise ValueError("mean_behavior must not be computed from test targets")
        self.tau_grid_np = np.asarray(tau_grid_np, dtype=np.float32)
        info, _, targets = build_tau_behavior_targets(
            db_path=db_path, class_id=class_id, iou_thr=iou_thr, split=split,
            n_tau_points=len(self.tau_grid_np),
            tau_min=float(self.tau_grid_np[0]), tau_max=float(self.tau_grid_np[-1]),
        )
        keep = [i for i, s in enumerate(info) if s["name"] not in set(exclude_regimes)]
        targets = {k: v[keep] for k, v in targets.items() if np.ndim(v) == 2 and v.shape[0] == len(info)}
        self.source = {"split": split, "regimes": [info[i]["name"] for i in keep]}
        tpr = targets["tpr"].mean(axis=0)
        precision = targets["precision"].mean(axis=0)
        self.curves = {
            "tpr_tau": tpr,
            "recall_tau": tpr,
            "fpr_tau": targets["fpr"].mean(axis=0),
            "fppi_tau": targets["fppi"].mean(axis=0),
            "precision_tau": precision,
            "f1_tau": safe_f1(precision, tpr).astype(np.float32),
        }

    def __call__(self, img_path: Path):
        return None, {k: v.copy() for k, v in self.curves.items()}


def predict_curves_and_tau_for_image(
    img_path: Path,
    predictor,
    fpr_grid_np: np.ndarray,
    fppi_grid_np: np.ndarray,
    recall_grid_np: np.ndarray,
    criterion: str = "f1",
    max_fppi: float = 0.1,
):
    img_rgb, tau_curves = predictor(img_path)
    tau_hat = select_tau_from_curves(predictor.tau_grid_np, tau_curves, criterion=criterion, max_fppi=max_fppi)
    pred_roc, pred_froc, pred_pr = tau_curves_to_geometric(tau_curves, fpr_grid_np, fppi_grid_np, recall_grid_np)
    return img_rgb, pred_roc, pred_froc, pred_pr, tau_hat, tau_curves


def evaluate_one_example(
    info: Tuple[str, str, int],
    gt_roc: np.ndarray,
    gt_froc: np.ndarray,
    gt_pr: np.ndarray,
    images_root: Path,
    predictor,
    fpr_grid_np: np.ndarray,
    fppi_grid_np: np.ndarray,
    recall_grid_np: np.ndarray,
    criterion: str = "f1",
    max_fppi: float = 0.1,
) -> Optional[Dict]:
    if info is None:
        return None

    rel_path, deg_name, severity = info
    img_path = images_root / rel_path

    try:
        _, pred_roc, pred_froc, pred_pr, tau_hat, tau_curves = predict_curves_and_tau_for_image(
            img_path=img_path,
            predictor=predictor,
            fpr_grid_np=fpr_grid_np,
            fppi_grid_np=fppi_grid_np,
            recall_grid_np=recall_grid_np,
            criterion=criterion,
            max_fppi=max_fppi,
        )
    except Exception as e:
        print(f"Prediction failed for {rel_path}: {e}")
        return None

    res = curve_metrics(gt_roc, gt_froc, gt_pr, pred_roc, pred_froc, pred_pr, fpr_grid_np, fppi_grid_np, recall_grid_np)
    res.update(rel_path=rel_path, degradation_name=deg_name, severity=severity, tau_hat=tau_hat)
    res["_tau_curves"] = tau_curves
    return res


def curve_metrics(gt_roc, gt_froc, gt_pr, pred_roc, pred_froc, pred_pr, fpr_grid_np, fppi_grid_np, recall_grid_np) -> Dict:
    auroc_gt = float(np.trapezoid(gt_roc, fpr_grid_np))
    afroc_gt = float(np.trapezoid(gt_froc, fppi_grid_np))
    auprc_gt = float(np.trapezoid(gt_pr, recall_grid_np))

    auroc_pred = float(np.trapezoid(pred_roc, fpr_grid_np))
    afroc_pred = float(np.trapezoid(pred_froc, fppi_grid_np))
    auprc_pred = float(np.trapezoid(pred_pr, recall_grid_np))

    mad_roc = float(np.mean(np.abs(pred_roc - gt_roc)))
    mad_froc = float(np.mean(np.abs(pred_froc - gt_froc)))
    mad_pr = float(np.mean(np.abs(pred_pr - gt_pr)))

    R = recall_grid_np.astype(np.float64)

    P_gt = np.clip(gt_pr.astype(np.float64), 0.0, 1.0)
    denom_gt = P_gt + R
    valid_gt = denom_gt > 0
    op_rec_gt = None
    op_prec_gt = None
    if np.any(valid_gt):
        f1_gt = np.zeros_like(denom_gt)
        f1_gt[valid_gt] = 2.0 * P_gt[valid_gt] * R[valid_gt] / denom_gt[valid_gt]
        idx_gt = int(np.argmax(f1_gt))
        op_rec_gt = float(R[idx_gt])
        op_prec_gt = float(P_gt[idx_gt])

    P_pred = np.clip(pred_pr.astype(np.float64), 0.0, 1.0)
    denom_pred = P_pred + R
    valid_pred = denom_pred > 0
    op_rec_pred = None
    op_prec_pred = None
    if np.any(valid_pred):
        f1_pred = np.zeros_like(denom_pred)
        f1_pred[valid_pred] = 2.0 * P_pred[valid_pred] * R[valid_pred] / denom_pred[valid_pred]
        idx_pred = int(np.argmax(f1_pred))
        op_rec_pred = float(R[idx_pred])
        op_prec_pred = float(P_pred[idx_pred])

    delta_op_rec = None if (op_rec_gt is None or op_rec_pred is None) else (op_rec_pred - op_rec_gt)
    delta_op_prec = None if (op_prec_gt is None or op_prec_pred is None) else (op_prec_pred - op_prec_gt)

    return {
        "auroc_gt": auroc_gt,
        "afroc_gt": afroc_gt,
        "auprc_gt": auprc_gt,
        "auroc_pred": auroc_pred,
        "afroc_pred": afroc_pred,
        "auprc_pred": auprc_pred,
        "delta_auroc": auroc_pred - auroc_gt,
        "delta_afroc": afroc_pred - afroc_gt,
        "delta_auprc": auprc_pred - auprc_gt,
        "mad_roc": mad_roc,
        "mad_froc": mad_froc,
        "mad_pr": mad_pr,
        "op_rec_gt": op_rec_gt,
        "op_prec_gt": op_prec_gt,
        "op_rec_pred": op_rec_pred,
        "op_prec_pred": op_prec_pred,
        "delta_op_rec": delta_op_rec,
        "delta_op_prec": delta_op_prec,
    }


def compute_cluster_op_tau(
    conn: sqlite3.Connection,
    cluster_id: int,
    class_id: int,
    iou_thr: float,
    mode: str,
    max_fppi: float,
) -> Optional[float]:
    cur = conn.cursor()

    cur.execute(
        """
        SELECT thr, fpr, tpr, fppi, precision
        FROM subset_operating_points
        WHERE subset_id = ? AND class_id = ? AND ABS(iou_thr - ?) < 1e-8
        ORDER BY idx ASC;
        """,
        (int(cluster_id), int(class_id), float(iou_thr)),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    thr = np.array([r[0] for r in rows], dtype=float)
    fpr = np.array([r[1] for r in rows], dtype=float)
    tpr = np.array([r[2] for r in rows], dtype=float)
    fppi = np.array([r[3] for r in rows], dtype=float)
    prec = np.array([r[4] for r in rows], dtype=float)

    if mode == "youden":
        youden = tpr - fpr
        if youden.size == 0:
            return None
        return float(thr[int(np.argmax(youden))])

    if mode == "fppi":
        valid = np.where(fppi <= max_fppi)[0]
        if valid.size == 0:
            return None
        idx = int(valid[np.argmax(tpr[valid])])
        return float(thr[idx])

    if mode == "f1":
        R = np.clip(tpr.astype(np.float64), 0.0, 1.0)
        P = np.clip(prec.astype(np.float64), 0.0, 1.0)
        denom = P + R
        valid = denom > 0
        if not np.any(valid):
            return None
        f1 = np.zeros_like(denom)
        f1[valid] = 2.0 * P[valid] * R[valid] / denom[valid]
        return float(thr[int(np.argmax(f1))])

    return None


def load_cluster_f1_curve(
    conn: sqlite3.Connection,
    cluster_id: int,
    class_id: int,
    iou_thr: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT thr, tpr, precision
        FROM subset_operating_points
        WHERE subset_id = ? AND class_id = ? AND ABS(iou_thr - ?) < 1e-8
        ORDER BY idx ASC;
        """,
        (int(cluster_id), int(class_id), float(iou_thr)),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    thr = np.array([r[0] for r in rows], dtype=float)
    tpr = np.array([r[1] for r in rows], dtype=float)
    prec = np.array([r[2] for r in rows], dtype=float)

    R = np.clip(tpr.astype(np.float64), 0.0, 1.0)
    P = np.clip(prec.astype(np.float64), 0.0, 1.0)
    denom = P + R
    valid = denom > 0
    if not np.any(valid):
        return None

    f1 = np.zeros_like(denom)
    f1[valid] = 2.0 * P[valid] * R[valid] / denom[valid]
    f1_star = float(np.max(f1))

    return thr, f1, f1_star


def _mean_of(key, rows, abs_value: bool = False):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return float("nan")
    vals = np.array(vals, dtype=float)
    if abs_value:
        vals = np.abs(vals)
    return float(np.mean(vals))




# ============================================================
# Fixed pristine threshold vs adaptive threshold vs oracle
# ============================================================

POLICIES = ("pristine", "adaptive", "oracle")
POLICY_METRICS = ("f1", "precision", "recall", "fppi", "tp", "fp", "fn")


class RegimeSweeps:
    """Cached deduplicated raw sweeps per regime (subset); used only for scoring selected thresholds."""

    def __init__(self, conn: sqlite3.Connection, class_id: int, iou_thr: float):
        self.conn = conn
        self.class_id = class_id
        self.iou_thr = iou_thr
        self._raw: Dict[int, Dict[str, np.ndarray]] = {}
        self._asc: Dict[int, Dict[str, np.ndarray]] = {}

    def raw(self, subset_id: int):
        if subset_id not in self._raw:
            self._raw[subset_id] = load_raw_sweep(self.conn, subset_id, self.class_id, self.iou_thr)
        return self._raw[subset_id]

    def asc(self, subset_id: int):
        if subset_id not in self._asc:
            self._asc[subset_id] = dedupe_sweep(self.raw(subset_id))
        return self._asc[subset_id]


def add_threshold_policy_metrics(res: Dict, subset_id: int, sweeps: RegimeSweeps, tau_pristine: float,
                                 tau_oracle: float) -> None:
    """Score the pristine, adaptive and oracle thresholds on the identical stored detections of the regime."""
    taus = {"pristine": tau_pristine, "adaptive": res["tau_hat"], "oracle": tau_oracle}
    asc = sweeps.asc(subset_id)
    for p, tau in taus.items():
        m = detector_metrics_at_tau(asc, tau)
        res[f"tau_{p}"] = float(tau)
        for k in POLICY_METRICS:
            res[f"{k}_{p}"] = m[k]
    res["err_tau_adaptive"] = res["tau_adaptive"] - res["tau_oracle"]
    res["err_tau_pristine"] = res["tau_pristine"] - res["tau_oracle"]
    res["regret_adaptive"] = res["f1_oracle"] - res["f1_adaptive"]
    res["regret_pristine"] = res["f1_oracle"] - res["f1_pristine"]
    res["gain_adaptive"] = res["f1_adaptive"] - res["f1_pristine"]


THRESHOLD_FIELDS = (
    ["tau_pristine", "tau_adaptive", "tau_oracle", "err_tau_adaptive", "err_tau_pristine",
     "regret_adaptive", "regret_pristine", "gain_adaptive"]
    + [f"{k}_{p}" for p in POLICIES for k in POLICY_METRICS]
)


def _stats(vals) -> Dict[str, float]:
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "n": int(v.size)}


def _threshold_errors(err) -> Dict[str, float]:
    e = np.asarray(err, dtype=np.float64)
    if e.size == 0:
        return {"signed_mean": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    return {"signed_mean": float(e.mean()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}


class RegimeCurveAccumulator:
    """Running mean/std of predicted tau-curves per regime plus a few example curves."""

    def __init__(self, n_examples: int = 16):
        self.n_examples = n_examples
        self.sum: Dict[str, np.ndarray] = {}
        self.sumsq: Dict[str, np.ndarray] = {}
        self.n = 0
        self.examples: List[Dict[str, np.ndarray]] = []
        self.example_paths: List[str] = []
        self.tau_adaptive: List[float] = []

    def add(self, curves: Dict[str, np.ndarray], rel_path: str, tau_hat: float):
        for k in TAU_CURVE_KEYS:
            v = np.asarray(curves[k], dtype=np.float64)
            self.sum[k] = self.sum.get(k, 0.0) + v
            self.sumsq[k] = self.sumsq.get(k, 0.0) + v * v
        self.n += 1
        self.tau_adaptive.append(float(tau_hat))
        if len(self.examples) < self.n_examples:
            self.examples.append({k: np.asarray(curves[k], dtype=np.float32) for k in TAU_CURVE_KEYS})
            self.example_paths.append(rel_path)

    def mean(self, k):
        return self.sum[k] / max(self.n, 1)

    def std(self, k):
        m = self.mean(k)
        return np.sqrt(np.clip(self.sumsq[k] / max(self.n, 1) - m * m, 0.0, None))


def summarize_threshold_eval(rows: List[Dict], accs: Dict[int, RegimeCurveAccumulator], tau_grid_np: np.ndarray,
                             sweeps: RegimeSweeps, tau_pristine: float, cid_to_name: Dict[int, str],
                             pristine_cluster_id: int, criterion: str, max_fppi: float):
    per_regime = []
    for cid in sorted({r["cluster_id"] for r in rows}):
        rr = [r for r in rows if r["cluster_id"] == cid]
        tau_oracle = rr[0]["tau_oracle"]
        acc = accs.get(cid)

        # regime-level adaptive threshold: argmax of the mean predicted utility over the regime's inputs
        tau_regime = None
        if acc is not None and acc.n > 0:
            mean_curves = {k: acc.mean(k) for k in TAU_CURVE_KEYS}
            mean_curves["f1_tau"] = safe_f1(mean_curves["precision_tau"], mean_curves["tpr_tau"])
            tau_regime = select_tau_from_curves(tau_grid_np, mean_curves, criterion=criterion, max_fppi=max_fppi)

        asc = sweeps.asc(cid)
        policy_metrics = {
            "pristine": detector_metrics_at_tau(asc, tau_pristine),
            "adaptive_regime": detector_metrics_at_tau(asc, tau_regime) if tau_regime is not None else None,
            "oracle": detector_metrics_at_tau(asc, tau_oracle),
            "adaptive_per_input_mean": {k: float(np.mean([r[f"{k}_adaptive"] for r in rr])) for k in POLICY_METRICS},
        }
        entry = {
            "cluster_id": int(cid),
            "cluster_name": cid_to_name.get(cid, str(cid)),
            "n": len(rr),
            "tau_pristine": float(tau_pristine),
            "tau_oracle": float(tau_oracle),
            "tau_adaptive": _stats([r["tau_adaptive"] for r in rr]),
            "tau_adaptive_regime": tau_regime,
            "tau_error_adaptive": _threshold_errors([r["err_tau_adaptive"] for r in rr]),
            "tau_error_pristine": float(tau_pristine - tau_oracle),
            "f1_oracle": policy_metrics["oracle"]["f1"],
            "f1_pristine": policy_metrics["pristine"]["f1"],
            "f1_adaptive": _stats([r["f1_adaptive"] for r in rr]),
            "regret_adaptive": _stats([r["regret_adaptive"] for r in rr]),
            "regret_pristine": float(policy_metrics["oracle"]["f1"] - policy_metrics["pristine"]["f1"]),
            "gain_adaptive": _stats([r["gain_adaptive"] for r in rr]),
            "gain_adaptive_regime": None if tau_regime is None else float(
                policy_metrics["adaptive_regime"]["f1"] - policy_metrics["pristine"]["f1"]),
            "detector_metrics": policy_metrics,
        }
        per_regime.append(entry)

    gains = [r["gain_adaptive"] for r in rows]
    regime_gain = [e["gain_adaptive"]["mean"] for e in per_regime]
    pristine_entry = next((e for e in per_regime if e["cluster_id"] == pristine_cluster_id), None)
    worst = per_regime[int(np.argmin(regime_gain))] if per_regime else None

    global_summary = {
        "n_inputs": len(rows),
        "n_regimes": len(per_regime),
        "tau_pristine": float(tau_pristine),
        "gain_adaptive": _stats(gains),
        "gain_adaptive_mean_over_regimes": float(np.mean(regime_gain)) if regime_gain else float("nan"),
        "gain_adaptive_worst_regime": None if worst is None else {
            "cluster_name": worst["cluster_name"], "gain": worst["gain_adaptive"]["mean"]},
        "gain_adaptive_regime_level": _stats([e["gain_adaptive_regime"] for e in per_regime]),
        "pristine_change": None if pristine_entry is None else pristine_entry["gain_adaptive"]["mean"],
        "fraction_inputs_improved": float(np.mean(np.asarray(gains) > 0)) if gains else float("nan"),
        "fraction_inputs_worse": float(np.mean(np.asarray(gains) < 0)) if gains else float("nan"),
        "regret_adaptive": _stats([r["regret_adaptive"] for r in rows]),
        "regret_pristine_per_input": _stats([r["regret_pristine"] for r in rows]),
        "regret_pristine_mean_over_regimes": float(np.mean([e["regret_pristine"] for e in per_regime])),
        "tau_error_adaptive": _threshold_errors([r["err_tau_adaptive"] for r in rows]),
        "tau_error_pristine": _threshold_errors([r["err_tau_pristine"] for r in rows]),
    }
    return global_summary, per_regime


def save_tau_curves_npz(path: Path, accs: Dict[int, RegimeCurveAccumulator], tau_grid_np: np.ndarray,
                        sweeps: RegimeSweeps, cid_to_name: Dict[int, str], tau_pristine: float,
                        per_regime: List[Dict]):
    cids = sorted(accs.keys())
    out = {
        "tau": np.asarray(tau_grid_np, dtype=np.float32),
        "cluster_ids": np.asarray(cids, dtype=np.int64),
        "cluster_names": np.asarray([cid_to_name.get(c, str(c)) for c in cids]),
        "tau_pristine": np.float32(tau_pristine),
    }
    by_cid = {e["cluster_id"]: e for e in per_regime}
    for gt_key, pred_key in (("tpr", "tpr_tau"), ("fpr", "fpr_tau"), ("fppi", "fppi_tau"),
                             ("precision", "precision_tau"), ("f1", "f1_tau")):
        gt = [sweep_at_tau(sweeps.asc(c), tau_grid_np)[gt_key] for c in cids]
        out[f"gt_{gt_key}"] = np.asarray(gt, dtype=np.float32)
        out[f"pred_mean_{gt_key}"] = np.asarray([accs[c].mean(pred_key) for c in cids], dtype=np.float32)
        out[f"pred_std_{gt_key}"] = np.asarray([accs[c].std(pred_key) for c in cids], dtype=np.float32)
        out[f"pred_examples_{gt_key}"] = np.asarray(
            [np.stack([e[pred_key] for e in accs[c].examples]) for c in cids], dtype=object)
    out["example_paths"] = np.asarray([np.asarray(accs[c].example_paths) for c in cids], dtype=object)
    out["tau_adaptive"] = np.asarray([np.asarray(accs[c].tau_adaptive, dtype=np.float32) for c in cids], dtype=object)
    out["tau_oracle"] = np.asarray([by_cid[c]["tau_oracle"] if c in by_cid else np.nan for c in cids], dtype=np.float32)
    out["tau_adaptive_regime"] = np.asarray(
        [by_cid[c]["tau_adaptive_regime"] if c in by_cid and by_cid[c]["tau_adaptive_regime"] is not None else np.nan
         for c in cids], dtype=np.float32)
    np.savez_compressed(path, **out)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tau-spline monitor: ROC/FROC/PR, tau_hat, and F1 regret."
    )
    parser.add_argument("--db", type=str, required=True, help="Path to SQLite DB file.")
    parser.add_argument("--images_root", type=str, required=True, help="Root folder of images.")
    parser.add_argument("--weights", type=str, default='weights/yolov10n.pt', help="YOLO weights (.pt).")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to trained checkpoint (not needed for mean_behavior).")
    parser.add_argument("--split", type=str, default="test", help="Which split to use.")
    parser.add_argument("--head_type", type=str, default=None, choices=list(HEAD_TYPES) + ["mean_behavior"],
                        help="Defaults to the head stored in the checkpoint. 'mean_behavior' = image-independent baseline.")
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--u_max", type=float, default=3.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--examples_per_cluster", type=int, default=0, help="0 = all.")
    parser.add_argument("--output_csv", type=str, default="curve_evaluation_results.csv")
    parser.add_argument("--op_mode", type=str, choices=["none", "youden", "f1", "fppi"], default="f1",
                        help="Operating criterion for tau_gt/tau_hat (tau_hat uses f1 when 'none').")
    parser.add_argument("--op_max_fppi", type=float, default=0.10)
    parser.add_argument("--eval_n_points", type=int, default=201)
    parser.add_argument("--n_dense_tau", type=int, default=1001,
                        help="Dense threshold grid for tau_behavior / mean_behavior (legacy heads keep their ckpt grid).")
    parser.add_argument("--threshold_eval", action="store_true",
                        help="Fixed pristine vs adaptive vs oracle threshold evaluation.")
    parser.add_argument("--pristine_split", type=str, default="val",
                        help="Split whose pristine subset calibrates the fixed threshold.")
    parser.add_argument("--skip_paired_pristine", action="store_true",
                        help="Do not re-evaluate the paired pristine image for every regime.")
    parser.add_argument("--baseline_split", type=str, default="train", help="Split for the mean_behavior baseline.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for example selection.")
    parser.add_argument("--feature_cache", type=str, default=None,
                        help="Use cached pooled frozen-detector features instead of decoding images.")
    parser.add_argument("--exclude_regimes", nargs="*", default=[],
                        help="mean_behavior only: regimes excluded from the baseline's training data (held-out runs).")

    args = parser.parse_args()
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    images_root = Path(args.images_root)
    split = args.split
    criterion = args.op_mode if args.op_mode in OP_CRITERIA else "f1"

    ckpt = {}
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args", {})

    head_type = args.head_type or train_args.get("head", "tau_spline_single")
    if head_type not in HEAD_TYPES and head_type != "mean_behavior":
        raise RuntimeError(f"Unsupported head_type: {head_type}")
    if head_type != "mean_behavior" and not ckpt:
        raise RuntimeError("--ckpt is required unless --head_type mean_behavior")

    u_max_train = float(train_args.get("u_max", args.u_max))

    if head_type in LEGACY_HEAD_TYPES:
        tau_grid = ckpt.get("tau_grid", train_args.get("tau_grid", None))
        if tau_grid is None:
            raise RuntimeError("No tau_grid found in checkpoint.")
        tau_grid_np = np.asarray(tau_grid, dtype=np.float32)
    else:
        tau_grid_np = np.linspace(0.0, 1.0, int(args.n_dense_tau), dtype=np.float32)

    eval_n_points = int(args.eval_n_points)

    clusters_info_targets, fpr_grid_np, fppi_grid_np, roc_targets_np, froc_targets_np = build_cluster_targets(
        db_path=args.db,
        class_id=args.class_id,
        iou_thr=args.iou_thr,
        u_max=u_max_train,
        n_roc_points=eval_n_points,
        n_froc_points=eval_n_points,
        split=split,
    )

    clusters_info_pr, recall_grid_np, pr_targets_np = build_pr_targets(
        db_path=args.db,
        class_id=args.class_id,
        iou_thr=args.iou_thr,
        n_pr_points=eval_n_points,
        split=split,
    )

    assert len(clusters_info_pr) == len(clusters_info_targets)
    for a, b in zip(clusters_info_pr, clusters_info_targets):
        assert a["cluster_id"] == b["cluster_id"]

    cid_to_index = {ci["cluster_id"]: ci["index"] for ci in clusters_info_targets}
    print(f"Found {len(clusters_info_targets)} clusters with ROC/FROC/PR metrics.")

    if head_type == "mean_behavior":
        predictor = MeanBehaviorPredictor(args.db, args.class_id, args.iou_thr, tau_grid_np, split=args.baseline_split,
                                          exclude_regimes=args.exclude_regimes)
    elif args.feature_cache:
        check_cache_matches(args.feature_cache, args.weights, args.imgsz)
        cache = FeatureCache(args.feature_cache, split)
        monitor_head = build_monitor_head(head_type, cache.feat_channels, train_args)
        _load_monitor_head_from_ckpt(monitor_head, ckpt)
        monitor_head.to(device).eval()
        predictor = CachedMonitorPredictor(cache, images_root, monitor_head, head_type, tau_grid_np, u_max_train, device)
    else:
        yolo = YOLO(args.weights)
        if isinstance(ckpt, dict) and "yolo_state_dict" in ckpt:
            print("Loading YOLO backbone weights from checkpoint.")
            yolo.model.load_state_dict(ckpt["yolo_state_dict"], strict=False)
        else:
            print("WARNING: No yolo_state_dict in checkpoint, using raw YOLO weights.")

        yolo.fuse()
        yolo.model.eval()
        yolo.to(device)

        backbone_forward = build_backbone_runner(yolo.model)

        dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        with torch.no_grad():
            feats_sample = backbone_forward(dummy)
        feat_channels = [int(f.shape[1]) for f in feats_sample]

        monitor_head = build_monitor_head(head_type, feat_channels, train_args)
        _load_monitor_head_from_ckpt(monitor_head, ckpt)
        monitor_head.to(device)
        monitor_head.eval()

        predictor = MonitorPredictor(backbone_forward, monitor_head, head_type, tau_grid_np, u_max_train,
                                     args.imgsz, device)

    conn = sqlite3.connect(args.db)
    clusters_db = get_clusters_with_metrics(
        conn,
        class_id=args.class_id,
        iou_thr=args.iou_thr,
        split=split,
    )
    cid_to_name = {cid: cname for cid, cname in clusters_db}

    pristine_cluster_id: Optional[int] = None
    for cid, cname in clusters_db:
        if "pristine" in cname.lower():
            pristine_cluster_id = cid
            break
    if pristine_cluster_id is None:
        raise RuntimeError("Could not find a pristine cluster.")

    pristine_cluster_index = cid_to_index[pristine_cluster_id]

    cid_to_tau_gt: Dict[int, Optional[float]] = {}
    cid_to_thr_f1: Dict[int, np.ndarray] = {}
    cid_to_f1_curve: Dict[int, np.ndarray] = {}
    cid_to_f1_star: Dict[int, float] = {}

    if args.op_mode != "none":
        print(f"Computing cluster-level tau_gt using op_mode='{args.op_mode}'")
        for cid, _ in clusters_db:
            cid_to_tau_gt[cid] = compute_cluster_op_tau(
                conn=conn,
                cluster_id=cid,
                class_id=args.class_id,
                iou_thr=args.iou_thr,
                mode=args.op_mode,
                max_fppi=args.op_max_fppi,
            )

            if args.op_mode == "f1":
                f1_data = load_cluster_f1_curve(
                    conn=conn,
                    cluster_id=cid,
                    class_id=args.class_id,
                    iou_thr=args.iou_thr,
                )
                if f1_data is not None:
                    thr_f1, f1_curve, f1_star = f1_data
                    cid_to_thr_f1[cid] = thr_f1
                    cid_to_f1_curve[cid] = f1_curve
                    cid_to_f1_star[cid] = f1_star

    # ---- fixed pristine threshold (calibrated on pristine validation data only) ----
    sweeps = RegimeSweeps(conn, args.class_id, args.iou_thr)
    tau_pristine = None
    cid_to_tau_oracle: Dict[int, float] = {}
    if args.threshold_eval:
        tau_pristine, pristine_info = pristine_validation_threshold(
            args.db, args.class_id, args.iou_thr, split=args.pristine_split,
            criterion=criterion, max_fppi=args.op_max_fppi,
        )
        print(f"Fixed pristine threshold from split='{args.pristine_split}' "
              f"(subset {pristine_info['subset_id']}): tau_pristine={tau_pristine:.4f}")
        for cid, _ in clusters_db:
            cid_to_tau_oracle[cid] = oracle_threshold(sweeps.raw(cid), criterion=criterion, max_fppi=args.op_max_fppi)

    all_results: List[Dict] = []
    threshold_rows: List[Dict] = []
    regime_accs: Dict[int, RegimeCurveAccumulator] = {}

    def _add_f1_regret(res: Dict, cid_for_curve: int):
        tau_hat = res.get("tau_hat", None)
        thr_f1 = cid_to_thr_f1.get(cid_for_curve, None)
        f1_curve = cid_to_f1_curve.get(cid_for_curve, None)
        f1_star = cid_to_f1_star.get(cid_for_curve, None)

        if tau_hat is None or thr_f1 is None or f1_curve is None or f1_star is None:
            res["f1_star"] = None
            res["f1_hat"] = None
            res["f1_regret"] = None
            return

        idx_hat = int(np.argmin(np.abs(thr_f1 - tau_hat)))
        f1_hat = float(f1_curve[idx_hat])
        res["f1_star"] = float(f1_star)
        res["f1_hat"] = f1_hat
        res["f1_regret"] = float(f1_star - f1_hat)

    def _finish(res: Dict, cid: int, cname: str, example_idx: int, variant: str):
        tau_gt = cid_to_tau_gt.get(cid, None)
        res.update(
            cluster_id=cid,
            cluster_name=cname,
            example_idx=example_idx,
            variant=variant,
            tau_gt=tau_gt,
            delta_tau=None if (tau_gt is None or res["tau_hat"] is None) else (res["tau_hat"] - tau_gt),
        )
        _add_f1_regret(res, cid)
        curves = res.pop("_tau_curves")
        if args.threshold_eval:
            add_threshold_policy_metrics(res, cid, sweeps, tau_pristine, cid_to_tau_oracle[cid])
            # each input is scored in its own regime; paired pristine re-evaluations are not double counted
            if variant == "degraded":
                threshold_rows.append(res)
                regime_accs.setdefault(cid, RegimeCurveAccumulator()).add(curves, res["rel_path"], res["tau_hat"])
        all_results.append(res)

    for cid, cname in clusters_db:
        if cid not in cid_to_index:
            print(f"Cluster {cid} not found in targets, skipping.")
            continue

        cluster_index = cid_to_index[cid]
        pristine_list, degraded_list = get_n_examples_for_cluster(conn, cid, args.examples_per_cluster, split)

        if not pristine_list or not degraded_list:
            print(f"Cluster {cid}: no pristine/degraded pairs found, skipping.")
            continue

        print(f"Evaluating cluster {cid} - {cname} with {len(pristine_list)} examples.")

        for example_idx, (pristine_info, degraded_info) in enumerate(zip(pristine_list, degraded_list)):
            if pristine_info is not None and not args.skip_paired_pristine:
                res_pr = evaluate_one_example(
                    info=pristine_info,
                    gt_roc=roc_targets_np[pristine_cluster_index],
                    gt_froc=froc_targets_np[pristine_cluster_index],
                    gt_pr=pr_targets_np[pristine_cluster_index],
                    images_root=images_root,
                    predictor=predictor,
                    fpr_grid_np=fpr_grid_np,
                    fppi_grid_np=fppi_grid_np,
                    recall_grid_np=recall_grid_np,
                    criterion=criterion,
                    max_fppi=args.op_max_fppi,
                )
                if res_pr is not None:
                    _finish(res_pr, pristine_cluster_id, cname, example_idx, "pristine")
                    # keep the original bookkeeping: the paired pristine row is listed under the current cluster
                    res_pr["cluster_id"] = cid

            if degraded_info is not None:
                res_deg = evaluate_one_example(
                    info=degraded_info,
                    gt_roc=roc_targets_np[cluster_index],
                    gt_froc=froc_targets_np[cluster_index],
                    gt_pr=pr_targets_np[cluster_index],
                    images_root=images_root,
                    predictor=predictor,
                    fpr_grid_np=fpr_grid_np,
                    fppi_grid_np=fppi_grid_np,
                    recall_grid_np=recall_grid_np,
                    criterion=criterion,
                    max_fppi=args.op_max_fppi,
                )
                if res_deg is not None:
                    _finish(res_deg, cid, cname, example_idx, "degraded")

    if not all_results:
        conn.close()
        print("No results collected; nothing to save.")
        return

    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "cluster_name",
        "variant",
        "example_idx",
        "rel_path",
        "degradation_name",
        "severity",
        "auroc_gt",
        "afroc_gt",
        "auprc_gt",
        "auroc_pred",
        "afroc_pred",
        "auprc_pred",
        "delta_auroc",
        "delta_afroc",
        "delta_auprc",
        "mad_roc",
        "mad_froc",
        "mad_pr",
        "op_rec_gt",
        "op_prec_gt",
        "op_rec_pred",
        "op_prec_pred",
        "delta_op_rec",
        "delta_op_prec",
        "tau_gt",
        "tau_hat",
        "delta_tau",
        "f1_star",
        "f1_hat",
        "f1_regret",
    ]
    if args.threshold_eval:
        fieldnames += THRESHOLD_FIELDS

    with output_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"Saved per-example evaluation metrics to {output_csv_path}")

    metric_keys = [
        ("mad_roc", False), ("mad_froc", False), ("mad_pr", False),
        ("delta_auroc", False), ("delta_afroc", False), ("delta_auprc", False),
        ("delta_tau", False), ("f1_regret", False), ("delta_op_rec", False), ("delta_op_prec", False),
    ]

    def _metrics_of(rows):
        out = {}
        for key, _ in metric_keys:
            out[key] = _mean_of(key, rows)
            if key.startswith("delta_") or key == "f1_regret":
                out[f"abs_{key}"] = _mean_of(key, rows, abs_value=True)
        return out

    global_metrics = _metrics_of(all_results)

    print("\nGlobal averages:")
    for k, v in global_metrics.items():
        print(f"  {k}: {v:.6f}")

    cluster_to_rows: Dict[Tuple[int, str], List[Dict]] = {}
    for r in all_results:
        key = (r["cluster_id"], r["cluster_name"])
        cluster_to_rows.setdefault(key, []).append(r)

    per_cluster = []
    for (cid, cname), rows in sorted(cluster_to_rows.items(), key=lambda x: x[0][0]):
        entry = {"cluster_id": int(cid), "cluster_name": str(cname)}
        entry.update(_metrics_of(rows))
        per_cluster.append(entry)

    pristine_rows = [r for r in all_results if r["cluster_id"] == pristine_cluster_id and r["variant"] == "degraded"]
    degraded_only = [r for r in all_results if r["cluster_id"] != pristine_cluster_id and r["variant"] == "degraded"]

    summary = {
        "ckpt": str(args.ckpt),
        "head_type": head_type,
        "split": args.split,
        "class_id": int(args.class_id),
        "iou_thr": float(args.iou_thr),
        "u_max": float(u_max_train),
        "imgsz": int(args.imgsz),
        "examples_per_cluster": int(args.examples_per_cluster),
        "op_mode": args.op_mode,
        "op_max_fppi": float(args.op_max_fppi),
        "n_tau_eval": int(len(tau_grid_np)),
        "seed": int(args.seed),
        "global_metrics": global_metrics,
        "pristine_metrics": _metrics_of(pristine_rows),
        "degraded_metrics": _metrics_of(degraded_only),
        "per_cluster": per_cluster,
    }

    summary_path = output_csv_path.with_name(output_csv_path.stem + "_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary JSON to {summary_path}")

    if args.threshold_eval and threshold_rows:
        global_thr, per_regime = summarize_threshold_eval(
            threshold_rows, regime_accs, tau_grid_np, sweeps, tau_pristine, cid_to_name,
            pristine_cluster_id, criterion, args.op_max_fppi,
        )
        thr_summary = {
            "ckpt": str(args.ckpt),
            "head_type": head_type,
            "split": args.split,
            "criterion": criterion,
            "pristine_split": args.pristine_split,
            "global": global_thr,
            "per_regime": per_regime,
        }
        thr_path = output_csv_path.with_name(output_csv_path.stem + "_threshold_summary.json")
        with thr_path.open("w") as f:
            json.dump(thr_summary, f, indent=2)

        regime_csv = output_csv_path.with_name(output_csv_path.stem + "_threshold_per_regime.csv")
        with regime_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cluster_id", "cluster_name", "n", "tau_pristine", "tau_oracle", "tau_adaptive_mean",
                        "tau_adaptive_regime", "tau_mae_adaptive", "f1_oracle", "f1_pristine", "f1_adaptive_mean",
                        "regret_pristine", "regret_adaptive_mean", "gain_adaptive_mean", "gain_adaptive_median",
                        "gain_adaptive_regime"])
            for e in per_regime:
                w.writerow([e["cluster_id"], e["cluster_name"], e["n"], e["tau_pristine"], e["tau_oracle"],
                            e["tau_adaptive"]["mean"], e["tau_adaptive_regime"], e["tau_error_adaptive"]["mae"],
                            e["f1_oracle"], e["f1_pristine"], e["f1_adaptive"]["mean"], e["regret_pristine"],
                            e["regret_adaptive"]["mean"], e["gain_adaptive"]["mean"], e["gain_adaptive"]["median"],
                            e["gain_adaptive_regime"]])

        npz_path = output_csv_path.with_name(output_csv_path.stem + "_tau_curves.npz")
        save_tau_curves_npz(npz_path, regime_accs, tau_grid_np, sweeps, cid_to_name, tau_pristine, per_regime)

        g = global_thr
        print("\nFixed pristine vs adaptive threshold:")
        print(f"  tau_pristine = {g['tau_pristine']:.4f}")
        print(f"  gain (mean/median) = {g['gain_adaptive']['mean']:+.4f} / {g['gain_adaptive']['median']:+.4f}")
        print(f"  worst regime gain  = {g['gain_adaptive_worst_regime']}")
        print(f"  pristine change    = {g['pristine_change']}")
        print(f"  regret adaptive    = {g['regret_adaptive']['mean']:.4f}, "
              f"regret pristine (regimes) = {g['regret_pristine_mean_over_regimes']:.4f}")
        print(f"  tau MAE adaptive / pristine = {g['tau_error_adaptive']['mae']:.4f} / {g['tau_error_pristine']['mae']:.4f}")
        print(f"Saved threshold evaluation to {thr_path}, {regime_csv}, {npz_path}")

    conn.close()


if __name__ == "__main__":
    main()

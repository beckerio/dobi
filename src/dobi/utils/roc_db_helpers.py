# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import cv2
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Sequence
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from dobi.utils.object_detection_helpers import pairwise_iou
from dobi.utils.colors import COLOR_CYCLE


# ============================================================
# Compatibility helpers
# ============================================================

def _compat_group_id(info: Dict) -> int:
    """
    Backward-compatible helper:
    accepts either {'subset_id': ...} or {'cluster_id': ...}.
    """
    if "subset_id" in info:
        return int(info["subset_id"])
    if "cluster_id" in info:
        return int(info["cluster_id"])
    raise KeyError("Expected 'subset_id' or 'cluster_id' in info dict.")


# ============================================================
# DB helpers: load subset curves and build targets
# Backward-compatible function names kept where useful.
# ============================================================

def build_youden_targets_tau_grid(
    db_path: str,
    clusters_info: List[Dict],
    class_id: int,
    iou_thr: float,
    n_tau_points: int = 50,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a global detector-score threshold grid tau_grid and per-subset
    Youden targets J(tau) = TPR(tau) - FPR(tau) on that grid.

    Backward-compatible with old clusters_info naming.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    tau_grid = np.linspace(tau_min, tau_max, n_tau_points, dtype=np.float32)
    K_clusters = len(clusters_info)
    youden_targets = np.zeros((K_clusters, n_tau_points), dtype=np.float32)

    for ci in clusters_info:
        sid = _compat_group_id(ci)
        cidx = int(ci["index"])

        cur.execute(
            """
            SELECT thr, fpr, tpr
            FROM subset_operating_points
            WHERE subset_id = ?
              AND class_id = ?
              AND ABS(iou_thr - ?) < 1e-8
            ORDER BY thr ASC;
            """,
            (sid, int(class_id), float(iou_thr)),
        )
        rows = cur.fetchall()
        if not rows:
            continue

        thr = np.array([r[0] for r in rows], dtype=np.float32)
        fpr = np.array([r[1] for r in rows], dtype=np.float32)
        tpr = np.array([r[2] for r in rows], dtype=np.float32)

        mask = np.isfinite(thr) & np.isfinite(fpr) & np.isfinite(tpr)
        thr = thr[mask]
        fpr = fpr[mask]
        tpr = tpr[mask]

        if thr.size < 2:
            continue

        order = np.argsort(thr)
        thr = thr[order]
        fpr = fpr[order]
        tpr = tpr[order]

        tpr_tau = np.interp(tau_grid, thr, tpr, left=tpr[0], right=tpr[-1]).astype(np.float32)
        fpr_tau = np.interp(tau_grid, thr, fpr, left=fpr[0], right=fpr[-1]).astype(np.float32)

        youden_targets[cidx] = tpr_tau - fpr_tau

    conn.close()
    return tau_grid, youden_targets


def build_f1_targets_tau_grid(
    db_path: str,
    clusters_info: List[Dict],
    class_id: int,
    iou_thr: float,
    n_tau_points: int = 50,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a global detector-score threshold grid tau_grid and per-subset
    F1 targets F1(tau) on that grid.

    Backward-compatible with old clusters_info naming.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    tau_grid = np.linspace(tau_min, tau_max, n_tau_points, dtype=np.float32)
    K_clusters = len(clusters_info)
    f1_targets = np.zeros((K_clusters, n_tau_points), dtype=np.float32)
    eps = 1e-8

    for ci in clusters_info:
        sid = _compat_group_id(ci)
        cidx = int(ci["index"])

        cur.execute(
            """
            SELECT thr, recall, precision
            FROM subset_operating_points
            WHERE subset_id = ?
              AND class_id = ?
              AND ABS(iou_thr - ?) < 1e-8
            ORDER BY thr ASC;
            """,
            (sid, int(class_id), float(iou_thr)),
        )
        rows = cur.fetchall()
        if not rows:
            continue

        thr = np.array([r[0] for r in rows], dtype=np.float32)
        recall = np.array([r[1] for r in rows], dtype=np.float32)
        prec = np.array([r[2] for r in rows], dtype=np.float32)

        mask = np.isfinite(thr) & np.isfinite(recall) & np.isfinite(prec)
        thr = thr[mask]
        recall = recall[mask]
        prec = prec[mask]

        if thr.size < 2:
            continue

        order = np.argsort(thr)
        thr = thr[order]
        recall = recall[order]
        prec = prec[order]

        recall_tau = np.interp(tau_grid, thr, recall, left=recall[0], right=recall[-1]).astype(np.float32)
        prec_tau = np.interp(tau_grid, thr, prec, left=prec[0], right=prec[-1]).astype(np.float32)

        f1_tau = 2.0 * prec_tau * recall_tau / (prec_tau + recall_tau + eps)
        f1_targets[cidx] = f1_tau.astype(np.float32)

    conn.close()
    return tau_grid, f1_targets


def get_clusters_with_metrics(
    conn: sqlite3.Connection,
    class_id: int,
    iou_thr: float,
    split: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """
    Backward-compatible name.
    Returns (subset_id, subset_name) pairs from the new DB scheme.
    """
    cur = conn.cursor()

    if split is None:
        cur.execute(
            """
            SELECT DISTINCT s.id, s.name
            FROM subsets s
            JOIN subset_eval_metrics m
              ON s.id = m.subset_id
            WHERE m.class_id = ?
              AND ABS(m.iou_thr - ?) < 1e-8
            ORDER BY s.id;
            """,
            (int(class_id), float(iou_thr)),
        )
    else:
        cur.execute(
            """
            SELECT DISTINCT s.id, s.name
            FROM subsets s
            JOIN subset_eval_metrics m
              ON s.id = m.subset_id
            WHERE m.class_id = ?
              AND ABS(m.iou_thr - ?) < 1e-8
              AND s.split = ?
            ORDER BY s.id;
            """,
            (int(class_id), float(iou_thr), split),
        )

    return [(int(row[0]), str(row[1])) for row in cur.fetchall()]


def load_cluster_curve_points(
    conn: sqlite3.Connection,
    cluster_id: int,
    class_id: int,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Backward-compatible name.
    cluster_id is interpreted as subset_id in the new DB.

    Returns:
        (fpr, tpr_roc, (fppi, tpr_froc))
    """
    subset_id = int(cluster_id)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT idx, thr, tpr, fpr, fppi
        FROM subset_operating_points
        WHERE subset_id = ?
          AND class_id = ?
          AND ABS(iou_thr - ?) < 1e-8
        ORDER BY idx ASC;
        """,
        (subset_id, int(class_id), float(iou_thr)),
    )
    rows = cur.fetchall()
    if not rows:
        raise ValueError(
            f"No curve points for subset_id={subset_id}, "
            f"class_id={class_id}, iou_thr={iou_thr}"
        )

    tpr = np.array([r[2] for r in rows], dtype=np.float32)
    fpr = np.array([r[3] for r in rows], dtype=np.float32)
    fppi = np.array([r[4] for r in rows], dtype=np.float32)

    order_roc = np.argsort(fpr)
    fpr_sorted = fpr[order_roc]
    tpr_roc = tpr[order_roc]

    order_froc = np.argsort(fppi)
    fppi_sorted = fppi[order_froc]
    tpr_froc = tpr[order_froc]

    return fpr_sorted, tpr_roc, (fppi_sorted, tpr_froc)


def build_cluster_targets(
    db_path: str,
    class_id: int,
    iou_thr: float,
    u_max: float,
    n_roc_points: int = 100,
    n_froc_points: int = 100,
    split: Optional[str] = None,
):
    """
    Backward-compatible name.
    Uses subsets + subset_operating_points from the new DB.

    Returns:
      clusters_info: List[dict] with keys:
         'cluster_id', 'subset_id', 'name', 'index'
      fpr_grid: np.ndarray [L]
      fppi_grid: np.ndarray [M]
      roc_targets: np.ndarray [K, L]
      froc_targets: np.ndarray [K, M]
    """
    conn = sqlite3.connect(db_path)
    clusters = get_clusters_with_metrics(
        conn,
        class_id=class_id,
        iou_thr=iou_thr,
        split=split,
    )
    if not clusters:
        conn.close()
        raise RuntimeError(
            f"No subsets with metrics for class_id={class_id}, "
            f"iou_thr={iou_thr}, split={split}"
        )

    fpr_grid = np.linspace(0.0, 1.0, n_roc_points, dtype=np.float32)
    fppi_grid = np.linspace(0.0, float(u_max), n_froc_points, dtype=np.float32)

    K = len(clusters)
    roc_targets = np.zeros((K, n_roc_points), dtype=np.float32)
    froc_targets = np.zeros((K, n_froc_points), dtype=np.float32)

    clusters_info = []
    for k, (sid, name) in enumerate(clusters):
        fpr, tpr_roc, (fppi, tpr_froc) = load_cluster_curve_points(
            conn,
            cluster_id=sid,
            class_id=class_id,
            iou_thr=iou_thr,
        )

        if fpr.size < 2:
            tpr_roc_interp = np.zeros_like(fpr_grid)
        else:
            tpr_roc_interp = np.interp(
                fpr_grid,
                fpr,
                tpr_roc,
                left=tpr_roc[0],
                right=tpr_roc[-1],
            ).astype(np.float32)

        mask_u = fppi <= float(u_max)
        if not np.any(mask_u):
            tpr_froc_interp = np.zeros_like(fppi_grid)
        else:
            fppi_u = fppi[mask_u]
            tpr_froc_u = tpr_froc[mask_u]
            if fppi_u.size < 2:
                tpr_froc_interp = np.zeros_like(fppi_grid)
            else:
                tpr_froc_interp = np.interp(
                    fppi_grid,
                    fppi_u,
                    tpr_froc_u,
                    left=tpr_froc_u[0],
                    right=tpr_froc_u[-1],
                ).astype(np.float32)

        roc_targets[k] = tpr_roc_interp
        froc_targets[k] = tpr_froc_interp

        clusters_info.append(
            {
                "cluster_id": sid,
                "subset_id": sid,
                "name": name,
                "index": k,
            }
        )

    conn.close()
    return clusters_info, fpr_grid, fppi_grid, roc_targets, froc_targets


def get_example_variants_for_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
) -> Tuple[Optional[Tuple[str, str]], Optional[Tuple[str, str]]]:
    """
    Backward-compatible name.
    cluster_id is interpreted as subset_id.

    Return (pristine_variant, degraded_variant), where each is (rel_path, deg_name).
    """
    subset_id = int(cluster_id)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT v.id, v.image_id, v.rel_path, v.deg_name, v.severity
        FROM subset_variants sv
        JOIN variants v ON sv.variant_id = v.id
        WHERE sv.subset_id = ?
        ORDER BY v.id;
        """,
        (subset_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return None, None

    pristine_in_subset = None
    degraded_in_subset = None

    for v_id, image_id, rel_path, deg_name, severity in rows:
        deg_norm = (deg_name or "").strip().lower()
        if deg_norm == "pristine" and pristine_in_subset is None:
            pristine_in_subset = (rel_path, deg_name, image_id)
        elif deg_norm != "pristine" and degraded_in_subset is None:
            degraded_in_subset = (rel_path, deg_name, image_id)

        if pristine_in_subset is not None and degraded_in_subset is not None:
            break

    pristine_variant: Optional[Tuple[str, str]] = None
    degraded_variant: Optional[Tuple[str, str]] = None

    if pristine_in_subset is not None:
        rel_path, deg_name, _ = pristine_in_subset
        pristine_variant = (rel_path, deg_name)

    if degraded_in_subset is not None:
        rel_path, deg_name, _ = degraded_in_subset
        degraded_variant = (rel_path, deg_name)

    if pristine_variant is None and degraded_in_subset is not None:
        _, _, degraded_image_id = degraded_in_subset
        cur.execute(
            """
            SELECT rel_path, deg_name
            FROM variants
            WHERE image_id = ?
              AND lower(trim(deg_name)) = 'pristine'
            ORDER BY id
            LIMIT 1;
            """,
            (degraded_image_id,),
        )
        row_p = cur.fetchone()
        if row_p:
            pristine_variant = (row_p[0], row_p[1])

    return pristine_variant, degraded_variant


def _compute_variant_curve_points(
    scores: np.ndarray,
    is_tp: np.ndarray,
    n_gt: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-variant ROC + FROC points from per-prediction scores and TP flags.

    Returns:
      fpr_points:  [K] in [0,1]
      tpr_roc:     [K]
      fppi_points: [K] FP per image
      tpr_froc:    [K]
    """
    scores = np.asarray(scores, dtype=np.float32)
    is_tp = np.asarray(is_tp, dtype=bool)

    if scores.size == 0 or n_gt == 0:
        fpr = np.array([0.0], dtype=np.float32)
        tpr = np.array([0.0], dtype=np.float32)
        fppi = np.array([0.0], dtype=np.float32)
        return fpr, tpr, fppi, tpr.copy()

    order = np.argsort(-scores)
    labels_sorted = is_tp[order].astype(np.float32)

    cum_tp = np.cumsum(labels_sorted)
    cum_fp = np.cumsum(1.0 - labels_sorted)

    n_gt_float = float(max(n_gt, 1))
    tpr = cum_tp / n_gt_float
    fp = cum_fp

    fp_max = float(fp[-1])
    if fp_max <= 0.0:
        fpr = np.zeros_like(fp, dtype=np.float32)
    else:
        fpr = (fp / fp_max).astype(np.float32)

    fppi = fp.astype(np.float32)

    fpr = np.concatenate([[0.0], fpr])
    tpr_roc = np.concatenate([[0.0], tpr])
    fppi = np.concatenate([[0.0], fppi])
    tpr_froc = np.concatenate([[0.0], tpr])

    return (
        fpr.astype(np.float32),
        tpr_roc.astype(np.float32),
        fppi.astype(np.float32),
        tpr_froc.astype(np.float32),
    )


def _get_variant_id_and_image_id(
    conn: sqlite3.Connection,
    rel_path: str,
    split: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Look up a variant's (id, image_id) from its rel_path.

    If split is not None, require the variant's image to have images.split = split.
    """
    cur = conn.cursor()

    if split is None:
        cur.execute(
            """
            SELECT id, image_id
            FROM variants
            WHERE rel_path = ?
            ORDER BY id
            LIMIT 1;
            """,
            (rel_path,),
        )
    else:
        cur.execute(
            """
            SELECT v.id, v.image_id
            FROM variants v
            JOIN images i ON i.id = v.image_id
            WHERE v.rel_path = ?
              AND i.split = ?
            ORDER BY v.id
            LIMIT 1;
            """,
            (rel_path, split),
        )

    row = cur.fetchone()
    if row is None:
        if split is None:
            raise ValueError(f"No variant found for rel_path='{rel_path}'")
        raise ValueError(f"No variant found for rel_path='{rel_path}' in split='{split}'")

    return int(row[0]), int(row[1])


def _boxes_iou_xywh(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """
    Compute IoU between two sets of boxes in (x, y, w, h) format.

    boxes1: [N, 4]
    boxes2: [M, 4]

    Returns:
      ious: [N, M]
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    b1_x1 = boxes1[:, 0]
    b1_y1 = boxes1[:, 1]
    b1_x2 = boxes1[:, 0] + boxes1[:, 2]
    b1_y2 = boxes1[:, 1] + boxes1[:, 3]

    b2_x1 = boxes2[:, 0]
    b2_y1 = boxes2[:, 1]
    b2_x2 = boxes2[:, 0] + boxes2[:, 2]
    b2_y2 = boxes2[:, 1] + boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = area1[:, None] + area2[None, :] - inter_area
    union_area = np.clip(union_area, a_min=1e-6, a_max=None)

    ious = inter_area / union_area
    return ious.astype(np.float32)


def build_variant_targets_for_relpath(
    conn: sqlite3.Connection,
    rel_path: str,
    class_id: int,
    iou_thr: float,
    u_max: float,
    fpr_grid: np.ndarray,
    fppi_grid: np.ndarray,
    split: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-image ROC/FROC target curves for a single variant, interpolated onto
    common grids fpr_grid and fppi_grid.
    """
    cur = conn.cursor()

    variant_id, image_id = _get_variant_id_and_image_id(conn, rel_path, split=split)

    cur.execute(
        """
        SELECT x, y, w, h
        FROM gt_objects
        WHERE image_id = ? AND class_id = ?;
        """,
        (image_id, class_id),
    )
    gt_rows = cur.fetchall()
    if not gt_rows:
        tpr_roc_interp = np.zeros_like(fpr_grid, dtype=np.float32)
        tpr_froc_interp = np.zeros_like(fppi_grid, dtype=np.float32)
        return tpr_roc_interp, tpr_froc_interp

    gt_boxes = np.array(gt_rows, dtype=np.float32)
    n_gt = gt_boxes.shape[0]

    cur.execute(
        """
        SELECT x, y, w, h, score, category_id
        FROM predictions
        WHERE variant_id = ?;
        """,
        (variant_id,),
    )
    pred_rows = cur.fetchall()
    if not pred_rows:
        tpr_roc_interp = np.zeros_like(fpr_grid, dtype=np.float32)
        tpr_froc_interp = np.zeros_like(fppi_grid, dtype=np.float32)
        return tpr_roc_interp, tpr_froc_interp

    pred_boxes_all = np.array([[r[0], r[1], r[2], r[3]] for r in pred_rows], dtype=np.float32)
    pred_scores_all = np.array([r[4] for r in pred_rows], dtype=np.float32)
    pred_cat_ids = np.array([r[5] for r in pred_rows], dtype=np.int32)

    mask_cls = pred_cat_ids == int(class_id)
    pred_boxes = pred_boxes_all[mask_cls]
    pred_scores = pred_scores_all[mask_cls]

    if pred_boxes.size == 0:
        tpr_roc_interp = np.zeros_like(fpr_grid, dtype=np.float32)
        tpr_froc_interp = np.zeros_like(fppi_grid, dtype=np.float32)
        return tpr_roc_interp, tpr_froc_interp

    ious = _boxes_iou_xywh(pred_boxes, gt_boxes)
    max_ious = ious.max(axis=1) if ious.size > 0 else np.zeros((pred_boxes.shape[0],), dtype=np.float32)
    is_tp = max_ious >= float(iou_thr)

    fpr_pts, tpr_roc_pts, fppi_pts, tpr_froc_pts = _compute_variant_curve_points(
        scores=pred_scores,
        is_tp=is_tp,
        n_gt=n_gt,
    )

    if fpr_pts.size < 2:
        tpr_roc_interp = np.zeros_like(fpr_grid, dtype=np.float32)
    else:
        tpr_roc_interp = np.interp(
            fpr_grid,
            fpr_pts,
            tpr_roc_pts,
            left=tpr_roc_pts[0],
            right=tpr_roc_pts[-1],
        ).astype(np.float32)

    mask_u = fppi_pts <= u_max
    if not np.any(mask_u) or fppi_pts[mask_u].size < 2:
        tpr_froc_interp = np.zeros_like(fppi_grid, dtype=np.float32)
    else:
        fppi_u = fppi_pts[mask_u]
        tpr_froc_u = tpr_froc_pts[mask_u]
        tpr_froc_interp = np.interp(
            fppi_grid,
            fppi_u,
            tpr_froc_u,
            left=tpr_froc_u[0],
            right=tpr_froc_u[-1],
        ).astype(np.float32)

    return tpr_roc_interp, tpr_froc_interp


def build_variant_pr_targets_for_relpath(
    conn: sqlite3.Connection,
    rel_path: str,
    class_id: int,
    iou_thr: float,
    recall_grid: np.ndarray,
    split: Optional[str] = None,
) -> np.ndarray:
    """
    Build a per-image PR target curve (precision vs recall) for a single variant,
    interpolated onto the common recall grid.
    """
    cur = conn.cursor()

    variant_id, image_id = _get_variant_id_and_image_id(conn, rel_path, split=split)

    cur.execute(
        """
        SELECT x, y, w, h
        FROM gt_objects
        WHERE image_id = ? AND class_id = ?;
        """,
        (image_id, class_id),
    )
    gt_rows = cur.fetchall()
    if not gt_rows:
        return np.zeros_like(recall_grid, dtype=np.float32)

    gt_boxes = np.array(gt_rows, dtype=np.float32)
    n_gt = gt_boxes.shape[0]

    cur.execute(
        """
        SELECT x, y, w, h, score, category_id
        FROM predictions
        WHERE variant_id = ?;
        """,
        (variant_id,),
    )
    pred_rows = cur.fetchall()
    if not pred_rows:
        return np.zeros_like(recall_grid, dtype=np.float32)

    pred_boxes_all = np.array([[r[0], r[1], r[2], r[3]] for r in pred_rows], dtype=np.float32)
    pred_scores_all = np.array([r[4] for r in pred_rows], dtype=np.float32)
    pred_cat_ids = np.array([r[5] for r in pred_rows], dtype=np.int32)

    mask_cls = pred_cat_ids == int(class_id)
    pred_boxes = pred_boxes_all[mask_cls]
    pred_scores = pred_scores_all[mask_cls]

    if pred_boxes.size == 0:
        return np.zeros_like(recall_grid, dtype=np.float32)

    ious = _boxes_iou_xywh(pred_boxes, gt_boxes)
    max_ious = ious.max(axis=1) if ious.size > 0 else np.zeros((pred_boxes.shape[0],), dtype=np.float32)
    is_tp = max_ious >= float(iou_thr)

    scores = np.asarray(pred_scores, dtype=np.float32)
    labels = np.asarray(is_tp, dtype=bool)
    order = np.argsort(-scores)
    labels_sorted = labels[order].astype(np.float32)

    if labels_sorted.size == 0 or n_gt <= 0:
        return np.zeros_like(recall_grid, dtype=np.float32)

    cum_tp = np.cumsum(labels_sorted)
    cum_fp = np.cumsum(1.0 - labels_sorted)

    n_gt_float = float(max(n_gt, 1))
    recall_pts = cum_tp / n_gt_float
    denom = cum_tp + cum_fp
    denom = np.where(denom <= 0, 1.0, denom)
    precision_pts = cum_tp / denom

    recall_pts = np.concatenate([[0.0], recall_pts])
    precision_pts = np.concatenate([[1.0], precision_pts])

    for i in range(len(precision_pts) - 2, -1, -1):
        precision_pts[i] = max(precision_pts[i], precision_pts[i + 1])

    if recall_pts.size < 2:
        return np.zeros_like(recall_grid, dtype=np.float32)

    precision_interp = np.interp(
        recall_grid,
        recall_pts,
        precision_pts,
        left=precision_pts[0],
        right=precision_pts[-1],
    ).astype(np.float32)

    return precision_interp


def get_n_examples_for_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    n: int,
    split: str,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Backward-compatible name.
    cluster_id is interpreted as subset_id.

    Return up to n pristine/degraded example pairs for this subset.
    Each list contains (rel_path, deg_name, severity).

    If n == 0, return all possible pairs.
    """
    import random

    subset_id = int(cluster_id)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT v.image_id, v.rel_path, v.deg_name, v.severity
        FROM subset_variants sv
        JOIN variants v ON sv.variant_id = v.id
        JOIN images i ON v.image_id = i.id
        WHERE sv.subset_id = ?
          AND i.split = ?
        ORDER BY v.image_id, v.severity DESC, v.id;
        """,
        (subset_id, split),
    )
    rows = cur.fetchall()
    if not rows:
        return [], []

    by_image: Dict[int, List[Tuple[str, str, int]]] = {}
    for image_id, rel_path, deg_name, severity in rows:
        by_image.setdefault(int(image_id), []).append((rel_path, deg_name, int(severity)))

    image_ids = list(by_image.keys())
    random.shuffle(image_ids)

    pristine_list: List[Tuple[str, str, int]] = []
    degraded_list: List[Tuple[str, str, int]] = []

    for image_id in image_ids:
        variants = by_image[image_id]
        non_pristine = [v for v in variants if (v[1] or "").strip().lower() != "pristine"]
        d_rel, d_deg, d_sev = random.choice(non_pristine or variants)

        cur.execute(
            """
            SELECT rel_path, deg_name, severity
            FROM variants
            WHERE image_id = ?
              AND lower(trim(deg_name)) = 'pristine'
            ORDER BY id
            LIMIT 1;
            """,
            (image_id,),
        )
        pristine = cur.fetchone()
        if pristine is None:
            continue

        pristine_list.append((pristine[0], pristine[1], int(pristine[2])))
        degraded_list.append((d_rel, d_deg, int(d_sev)))

        if n > 0 and len(pristine_list) >= n:
            break

    return pristine_list, degraded_list


def get_gt_boxes_for_relpath(
    conn: sqlite3.Connection,
    rel_path: str,
    class_id: int,
    assume_normalized: bool = False,
) -> np.ndarray:
    """
    Load ground-truth boxes for a given image (by variants.rel_path and class_id).

    Returns:
        boxes_xyxy: np.ndarray [N, 4] with (x1, y1, x2, y2) in pixel coords.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT g.x, g.y, g.w, g.h, i.width, i.height
        FROM variants AS v
        JOIN images AS i ON v.image_id = i.id
        JOIN gt_objects AS g ON g.image_id = i.id
        WHERE v.rel_path = ? AND g.class_id = ?
        """,
        (rel_path, class_id),
    )
    rows = cur.fetchall()
    if not rows:
        return np.zeros((0, 4), dtype=float)

    boxes = []
    for x, y, w, h, width, height in rows:
        if assume_normalized and width is not None and height is not None:
            x *= width
            w *= width
            y *= height
            h *= height

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes.append([x1, y1, x2, y2])

    return np.array(boxes, dtype=float)


def get_db_predictions_for_relpath(
    conn: sqlite3.Connection,
    rel_path: str,
    pred_class_id: int,
    conf_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predicted boxes + scores for a variant from the DB.

    Returns:
        boxes_xyxy: np.ndarray [N, 4] (x1, y1, x2, y2)
        scores:     np.ndarray [N]
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.x, p.y, p.w, p.h, p.score
        FROM variants AS v
        JOIN predictions AS p ON p.variant_id = v.id
        WHERE v.rel_path = ? AND p.category_id = ?
        """,
        (rel_path, pred_class_id),
    )
    rows = cur.fetchall()
    if not rows:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float)

    boxes = []
    scores = []
    for x, y, w, h, score in rows:
        if score < conf_thr:
            continue
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        boxes.append([x1, y1, x2, y2])
        scores.append(score)

    if not boxes:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float)

    return np.array(boxes, dtype=float), np.array(scores, dtype=float)


def draw_boxes_with_labels(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[Union[str, float]]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 6,
    font_scale: float = 0.7,
    float_decimals: int = 3,
):
    """
    Draws boxes + optional labels on img (in-place, BGR).
    Puts label inside the bbox and formats numeric labels with more decimals.
    """
    if boxes is None or len(boxes) == 0:
        return img

    h, w = img.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if labels is not None and i < len(labels):
            raw_label = labels[i]

            if isinstance(raw_label, (float, np.floating, int, np.integer)):
                label = f"{raw_label:.{float_decimals}f}"
            else:
                label = str(raw_label)

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1
            )

            tw = min(tw, max(0, w - x1))
            y_text_bottom = min(y2, y1 + th + baseline + 4)
            y_text_top = max(y1, y_text_bottom - th - baseline - 4)

            overlay = img.copy()

            cv2.rectangle(
                overlay,
                (x1, y_text_top),
                (x1 + tw, y_text_bottom),
                color,
                thickness=-1,
            )

            alpha = 0.7
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            cv2.putText(
                img,
                label,
                (x1, y_text_bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img


COLORS_RGB = [np.array(c, dtype=float) for c in COLOR_CYCLE]


def _to_bgr_uint8(rgb) -> tuple[int, int, int]:
    """Convert an RGB color (0-1 or 0-255) to uint8 BGR tuple."""
    rgb = np.array(rgb, dtype=float)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    r, g, b = rgb
    return int(b), int(g), int(r)


COLORS_BGR = [_to_bgr_uint8(c) for c in COLORS_RGB]


def select_best_tp_per_gt(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each GT box, keep at most one prediction:
      - IoU >= iou_thr with that GT
      - highest score among those predictions

    Inputs are expected in XYXY format.

    Returns:
      filtered_pred_boxes [K, 4], filtered_pred_scores [K]
    """
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)

    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    ious = pairwise_iou(gt_boxes, pred_boxes)

    keep_indices: list[int] = []
    for g in range(gt_boxes.shape[0]):
        mask = ious[g] >= float(iou_thr)
        if not np.any(mask):
            continue
        cand_idx = np.where(mask)[0]
        best_local_idx = cand_idx[np.argmax(pred_scores[cand_idx])]
        keep_indices.append(int(best_local_idx))

    keep_indices = sorted(set(keep_indices))
    if not keep_indices:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    return pred_boxes[keep_indices], pred_scores[keep_indices]


def get_object_detection_scores_from_boxes(
    conn: sqlite3.Connection,
    rel_path: str,
    class_id: int,
    iou_thr: float,
) -> tuple[Optional[list[float]], int, int]:
    """
    Compute per-GT detection scores tau_g for a given image and class.

    tau_g is defined as the maximum prediction score among predictions with
    IoU >= iou_thr to that GT box.

    Returns:
      tau_objects: list[float] or None
      num_gt: int
      num_gt_detected: int
    """
    gt_boxes = get_gt_boxes_for_relpath(conn, rel_path, class_id)
    if gt_boxes is None:
        gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)

    num_gt = gt_boxes.shape[0]
    if num_gt == 0:
        return None, 0, 0

    pred_boxes, pred_scores = get_db_predictions_for_relpath(
        conn=conn,
        rel_path=rel_path,
        pred_class_id=class_id,
        conf_thr=0.0,
    )
    if pred_boxes is None:
        pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.zeros((0,), dtype=np.float32)

    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)

    if pred_boxes.shape[0] == 0:
        return None, num_gt, 0

    ious = pairwise_iou(gt_boxes, pred_boxes)

    tau_objects: list[float] = []
    num_gt_detected = 0

    for g in range(num_gt):
        mask = ious[g] >= float(iou_thr)
        if not np.any(mask):
            continue
        best_score = float(pred_scores[mask].max())
        tau_objects.append(best_score)
        num_gt_detected += 1

    if num_gt_detected == 0:
        return None, num_gt, 0

    return tau_objects, num_gt, num_gt_detected


# ============================================================
# ROC with threshold coloring helper
# ============================================================

def plot_threshold_colored_roc_for_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    class_id: int,
    iou_thr: float,
    output_dir: Path,
    cluster_name: str,
    example_idx: Optional[int] = None,
) -> None:
    """
    Backward-compatible name.
    cluster_id is interpreted as subset_id.

    Plot the GT ROC curve for a subset, color-coding the line by threshold tau.
    """
    subset_id = int(cluster_id)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT thr, fpr, tpr
        FROM subset_operating_points
        WHERE subset_id = ?
          AND class_id = ?
          AND ABS(iou_thr - ?) < 1e-8
        ORDER BY thr DESC;
        """,
        (subset_id, int(class_id), float(iou_thr)),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"No ROC points with thresholds for subset {subset_id}, class {class_id}.")
        return

    thr = np.array([r[0] for r in rows], dtype=float)
    fpr = np.array([r[1] for r in rows], dtype=float)
    tpr = np.array([r[2] for r in rows], dtype=float)

    order = np.argsort(fpr)
    thr = thr[order]
    fpr = fpr[order]
    tpr = tpr[order]

    auc = float(np.trapezoid(tpr, fpr))

    points = np.stack([fpr, tpr], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=float(thr.min()), vmax=float(thr.max()))
    lc = LineCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidth=3.0,
    )
    lc.set_array(thr[:-1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(lc)

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="gray",
        label="Random baseline",
    )

    q_levels = [0.9, 0.7, 0.5, 0.3, 0.1]
    taus_to_show = np.quantile(thr, q_levels)
    for tau in taus_to_show:
        idx = int(np.argmin(np.abs(thr - tau)))
        ax.scatter(fpr[idx], tpr[idx], s=45, color="black", zorder=3)
        ax.text(
            fpr[idx] + 0.015,
            tpr[idx] - 0.04,
            f"τ={thr[idx]:.2f}",
            fontsize=9,
            color="black",
        )

    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Detector score threshold τ")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    title = f"Subset {subset_id} ROC with threshold coloring"
    if cluster_name:
        title = f"{cluster_name} – ROC with threshold coloring (AUC = {auc:.3f})"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if example_idx is not None:
        fname = f"{example_idx:02d}_roc_with_thresholds_subset_{subset_id}.png"
    else:
        fname = f"roc_with_thresholds_subset_{subset_id}.png"

    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_cluster_pr_points(
    conn: sqlite3.Connection,
    cluster_id: int,
    class_id: int,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible name.
    cluster_id is interpreted as subset_id.

    Returns (recall, precision) arrays for the subset.
    """
    subset_id = int(cluster_id)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT idx, recall, precision
        FROM subset_operating_points
        WHERE subset_id = ?
          AND class_id = ?
          AND ABS(iou_thr - ?) < 1e-8
        ORDER BY idx ASC;
        """,
        (subset_id, int(class_id), float(iou_thr)),
    )
    rows = cur.fetchall()
    if not rows:
        raise ValueError(
            f"No PR points for subset_id={subset_id}, "
            f"class_id={class_id}, iou_thr={iou_thr}"
        )

    recall = np.array([r[1] for r in rows], dtype=np.float32)
    precision = np.array([r[2] for r in rows], dtype=np.float32)

    mask = np.isfinite(recall) & np.isfinite(precision)
    recall = recall[mask]
    precision = precision[mask]

    if recall.size == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    uniq_r, idx_start = np.unique(recall, return_index=True)
    prec_uniq = np.zeros_like(uniq_r)
    for k in range(len(uniq_r)):
        start = idx_start[k]
        end = idx_start[k + 1] if k + 1 < len(uniq_r) else len(recall)
        prec_uniq[k] = np.max(precision[start:end])

    return uniq_r, prec_uniq


def build_pr_targets(
    db_path: str,
    class_id: int,
    iou_thr: float,
    n_pr_points: int = 100,
    split: Optional[str] = None,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Backward-compatible name.
    Build a common recall grid and per-subset precision targets on that grid.
    """
    conn = sqlite3.connect(db_path)
    clusters = get_clusters_with_metrics(
        conn,
        class_id=class_id,
        iou_thr=iou_thr,
        split=split,
    )
    if not clusters:
        conn.close()
        raise RuntimeError(
            f"No subsets with metrics for class_id={class_id}, "
            f"iou_thr={iou_thr}, split={split}"
        )

    recall_grid = np.linspace(0.0, 1.0, n_pr_points, dtype=np.float32)
    K = len(clusters)
    pr_targets = np.zeros((K, n_pr_points), dtype=np.float32)

    clusters_info = []
    for k, (sid, name) in enumerate(clusters):
        recall_raw, prec_raw = load_cluster_pr_points(
            conn,
            cluster_id=sid,
            class_id=class_id,
            iou_thr=iou_thr,
        )

        if recall_raw.size < 2:
            prec_interp = np.zeros_like(recall_grid)
        else:
            prec_interp = np.interp(
                recall_grid,
                recall_raw,
                prec_raw,
                left=prec_raw[0],
                right=prec_raw[-1],
            ).astype(np.float32)

        pr_targets[k] = prec_interp
        clusters_info.append(
            {
                "cluster_id": sid,
                "subset_id": sid,
                "name": name,
                "index": k,
            }
        )

    conn.close()
    return clusters_info, recall_grid, pr_targets


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


def build_tau_operating_targets(
    db_path,
    class_id,
    iou_thr,
    split="train",
    n_roc_points=201,
    n_froc_points=201,
    n_pr_points=201,
    n_tau_points=75,
    fppi_max=3.0,
    utility_beta=15.0,
):
    roc_grid = np.linspace(0.0, 1.0, int(n_roc_points), dtype=np.float32)
    froc_grid = np.linspace(0.0, float(fppi_max), int(n_froc_points), dtype=np.float32)
    pr_grid = np.linspace(0.0, 1.0, int(n_pr_points), dtype=np.float32)
    tau_grid = np.linspace(0.0, 1.0, int(n_tau_points), dtype=np.float32)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    subset_rows = cur.execute(
        """
        SELECT
            s.id AS subset_id,
            s.name AS subset_name,
            s.split AS subset_split,
            sv.variant_id AS variant_id
        FROM subsets s
        JOIN subset_variants sv
          ON sv.subset_id = s.id
        WHERE s.split = ?
        ORDER BY s.id, sv.variant_id
        """,
        (split,),
    ).fetchall()

    if not subset_rows:
        conn.close()
        raise ValueError(f"No subsets found for split='{split}' in DB: {db_path}")

    grouped = {}
    for row in subset_rows:
        sid = int(row["subset_id"])
        if sid not in grouped:
            grouped[sid] = {
                "subset_id": sid,
                "name": str(row["subset_name"]),
                "split": str(row["subset_split"]),
                "variant_ids": [],
            }
        grouped[sid]["variant_ids"].append(int(row["variant_id"]))

    subsets_info = []
    for subset_index, item in enumerate(grouped.values()):
        entry = dict(item)
        entry["subset_index"] = int(subset_index)
        subsets_info.append(entry)

    roc_targets = []
    froc_targets = []
    pr_targets = []

    tau_utilities = []
    tau_target_probs = []
    tau_opts = []

    tau_precisions = []
    tau_tprs = []
    tau_recalls = []
    tau_f1s = []
    tau_fprs = []
    tau_fppis = []
    tau_fppi_norms = []

    aurocs = []
    auc_frocs = []
    auprcs = []

    for s in subsets_info:
        subset_id = int(s["subset_id"])

        rows = cur.execute(
            """
            SELECT
                idx,
                thr,
                tpr,
                fpr,
                fppi,
                precision
            FROM subset_operating_points
            WHERE subset_id = ?
              AND class_id = ?
              AND ABS(iou_thr - ?) < 1e-8
            ORDER BY idx ASC
            """,
            (subset_id, int(class_id), float(iou_thr)),
        ).fetchall()

        if not rows:
            conn.close()
            raise ValueError(
                f"No subset_operating_points found for subset_id={subset_id}, "
                f"class_id={class_id}, iou_thr={iou_thr}"
            )

        thr = np.asarray([r["thr"] for r in rows], dtype=np.float32)
        tpr = np.asarray([r["tpr"] for r in rows], dtype=np.float32)
        fpr = np.asarray([r["fpr"] for r in rows], dtype=np.float32)
        fppi = np.asarray([r["fppi"] for r in rows], dtype=np.float32)
        precision = np.asarray([r["precision"] for r in rows], dtype=np.float32)

        thr = np.clip(thr, 0.0, 1.0)
        tpr = np.clip(tpr, 0.0, 1.0)
        fpr = np.clip(fpr, 0.0, 1.0)
        fppi = np.clip(fppi, 0.0, None)
        precision = np.clip(precision, 0.0, 1.0)
        recall = tpr.copy()

        f1 = (2.0 * precision * recall) / np.clip(precision + recall, 1e-8, None)
        f1 = np.clip(f1, 0.0, 1.0)

        thr_tau, tpr_tau_src = _prepare_xy(thr, tpr, reduce="last")
        _, fpr_tau_src = _prepare_xy(thr, fpr, reduce="last")
        _, fppi_tau_src = _prepare_xy(thr, fppi, reduce="last")
        _, precision_tau_src = _prepare_xy(thr, precision, reduce="last")
        _, recall_tau_src = _prepare_xy(thr, recall, reduce="last")
        _, f1_tau_src = _prepare_xy(thr, f1, reduce="last")

        tau_tpr = _interp_curve(
            thr_tau, tpr_tau_src, tau_grid,
            left=float(tpr_tau_src[0]) if tpr_tau_src.size else 0.0,
            right=float(tpr_tau_src[-1]) if tpr_tau_src.size else 0.0,
            reduce="last",
        )
        tau_fpr = _interp_curve(
            thr_tau, fpr_tau_src, tau_grid,
            left=float(fpr_tau_src[0]) if fpr_tau_src.size else 0.0,
            right=float(fpr_tau_src[-1]) if fpr_tau_src.size else 0.0,
            reduce="last",
        )
        tau_fppi = _interp_curve(
            thr_tau, fppi_tau_src, tau_grid,
            left=float(fppi_tau_src[0]) if fppi_tau_src.size else 0.0,
            right=float(fppi_tau_src[-1]) if fppi_tau_src.size else 0.0,
            reduce="last",
        )
        tau_precision = _interp_curve(
            thr_tau, precision_tau_src, tau_grid,
            left=float(precision_tau_src[0]) if precision_tau_src.size else 0.0,
            right=float(precision_tau_src[-1]) if precision_tau_src.size else 0.0,
            reduce="last",
        )
        tau_recall = _interp_curve(
            thr_tau, recall_tau_src, tau_grid,
            left=float(recall_tau_src[0]) if recall_tau_src.size else 0.0,
            right=float(recall_tau_src[-1]) if recall_tau_src.size else 0.0,
            reduce="last",
        )
        tau_f1 = _interp_curve(
            thr_tau, f1_tau_src, tau_grid,
            left=float(f1_tau_src[0]) if f1_tau_src.size else 0.0,
            right=float(f1_tau_src[-1]) if f1_tau_src.size else 0.0,
            reduce="last",
        )

        tau_tpr = np.clip(tau_tpr, 0.0, 1.0)
        tau_fpr = np.clip(tau_fpr, 0.0, 0.999999)
        tau_fppi = np.clip(tau_fppi, 0.0, float(fppi_max))
        tau_precision = np.clip(tau_precision, 0.0, 1.0)
        tau_recall = np.clip(tau_recall, 0.0, 1.0)
        tau_f1 = np.clip(tau_f1, 0.0, 1.0)

        tau_fppi_norm = np.clip(tau_fppi / max(float(fppi_max), 1e-8), 0.0, 1.0)

        tau_utility = tau_f1.copy()
        logits = float(utility_beta) * tau_utility
        logits = logits - np.max(logits)
        tau_prob = np.exp(logits)
        tau_prob = tau_prob / np.clip(np.sum(tau_prob), 1e-8, None)
        tau_opt = float(tau_grid[int(np.argmax(tau_utility))])

        fpr_roc, tpr_roc = _prepare_xy(fpr, tpr, reduce="max")
        if tpr_roc.size:
            tpr_roc = _cummax(tpr_roc)
        roc_tpr = _interp_curve(
            fpr_roc, tpr_roc, roc_grid,
            left=float(tpr_roc[0]) if tpr_roc.size else 0.0,
            right=float(tpr_roc[-1]) if tpr_roc.size else 0.0,
            reduce="max",
        )
        roc_tpr = np.clip(_cummax(roc_tpr), 0.0, 1.0)

        fppi_froc, tpr_froc = _prepare_xy(fppi, tpr, reduce="max")
        if tpr_froc.size:
            tpr_froc = _cummax(tpr_froc)
        froc_tpr = _interp_curve(
            fppi_froc, tpr_froc, froc_grid,
            left=float(tpr_froc[0]) if tpr_froc.size else 0.0,
            right=float(tpr_froc[-1]) if tpr_froc.size else 0.0,
            reduce="max",
        )
        froc_tpr = np.clip(_cummax(froc_tpr), 0.0, 1.0)

        recall_pr, precision_pr = _precision_envelope(recall, precision)
        pr_precision = _interp_curve(
            recall_pr, precision_pr, pr_grid,
            left=float(precision_pr[0]) if precision_pr.size else 0.0,
            right=float(precision_pr[-1]) if precision_pr.size else 0.0,
            reduce="max",
        )
        pr_precision = np.clip(pr_precision, 0.0, 1.0)

        auroc = float(np.trapezoid(roc_tpr, roc_grid))
        auc_froc = float(np.trapezoid(froc_tpr, froc_grid))
        auprc = float(np.trapezoid(pr_precision, pr_grid))

        roc_targets.append(roc_tpr)
        froc_targets.append(froc_tpr)
        pr_targets.append(pr_precision)

        tau_utilities.append(tau_utility)
        tau_target_probs.append(tau_prob)
        tau_opts.append(tau_opt)

        tau_precisions.append(tau_precision)
        tau_tprs.append(tau_tpr)
        tau_recalls.append(tau_recall)
        tau_f1s.append(tau_f1)
        tau_fprs.append(tau_fpr)
        tau_fppis.append(tau_fppi)
        tau_fppi_norms.append(tau_fppi_norm)

        aurocs.append(auroc)
        auc_frocs.append(auc_froc)
        auprcs.append(auprc)

    conn.close()

    targets_np = {
        "roc_fpr_grid": roc_grid.astype(np.float32),
        "froc_fppi_grid": froc_grid.astype(np.float32),
        "pr_recall_grid": pr_grid.astype(np.float32),
        "tau_grid": tau_grid.astype(np.float32),

        "roc_tpr": np.stack(roc_targets).astype(np.float32),
        "froc_tpr": np.stack(froc_targets).astype(np.float32),
        "pr_precision": np.stack(pr_targets).astype(np.float32),

        "tau_utility": np.stack(tau_utilities).astype(np.float32),
        "tau_target_probs": np.stack(tau_target_probs).astype(np.float32),
        "tau_opt": np.asarray(tau_opts, dtype=np.float32),

        "precision": np.stack(tau_precisions).astype(np.float32),
        "tpr": np.stack(tau_tprs).astype(np.float32),
        "recall": np.stack(tau_recalls).astype(np.float32),
        "f1": np.stack(tau_f1s).astype(np.float32),
        "fpr": np.stack(tau_fprs).astype(np.float32),
        "fppi": np.stack(tau_fppis).astype(np.float32),
        "fppi_norm": np.stack(tau_fppi_norms).astype(np.float32),

        "auroc": np.asarray(aurocs, dtype=np.float32),
        "auc_froc": np.asarray(auc_frocs, dtype=np.float32),
        "auprc": np.asarray(auprcs, dtype=np.float32),
    }

    return subsets_info, tau_grid.astype(np.float32), targets_np
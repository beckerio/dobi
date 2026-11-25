#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
from dobi.paths import P as _P  # machine-specific paths (configs/local_paths_<host>.yaml)
import argparse
import sqlite3
from typing import List, Tuple, Dict

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def pairwise_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1 + area2.T - inter
    union = np.clip(union, 1e-6, None)

    return (inter / union).astype(np.float32)

def ensure_eval_tables(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name IN ('subset_eval_metrics', 'subset_operating_points')
        """
    )
    found = {r[0] for r in cur.fetchall()}
    required = {"subset_eval_metrics", "subset_operating_points"}

    missing = required - found
    if missing:
        raise RuntimeError(
            "Missing required tables in DB: "
            + ", ".join(sorted(missing))
            + ". Create the schema first with your DB setup script."
        )

def greedy_match_labels_one_image_np(
    pred_xyxy: np.ndarray,
    pred_scores: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_thr: float = 0.5,
) -> Tuple[np.ndarray, int]:
    n_pred = pred_xyxy.shape[0]
    n_gt = gt_xyxy.shape[0]

    if n_pred == 0:
        return np.zeros((0,), dtype=bool), n_gt
    if n_gt == 0:
        return np.zeros((n_pred,), dtype=bool), 0

    order = np.argsort(-pred_scores)
    pred_xyxy_s = pred_xyxy[order]

    ious = pairwise_iou(pred_xyxy_s, gt_xyxy)
    gt_matched = np.zeros((n_gt,), dtype=bool)
    is_tp_sorted = np.zeros((n_pred,), dtype=bool)

    for i in range(n_pred):
        iou_row = ious[i]
        j = int(iou_row.argmax())
        best_iou = float(iou_row[j])
        if best_iou >= iou_thr and not gt_matched[j]:
            gt_matched[j] = True
            is_tp_sorted[i] = True

    is_tp = np.zeros((n_pred,), dtype=bool)
    is_tp[order] = is_tp_sorted
    return is_tp, n_gt


def compute_operating_sweep_from_global_labels(
    scores: np.ndarray,
    is_tp: np.ndarray,
    num_gt: int,
    num_imgs: int,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    num_gt = int(num_gt)
    num_imgs = int(num_imgs)

    if scores.size == 0:
        thr = np.array([1.0, 0.0], dtype=np.float64)
        tp = np.array([0, 0], dtype=np.int64)
        fp = np.array([0, 0], dtype=np.int64)
    else:
        order = np.argsort(-scores, kind="stable")
        s = scores[order].astype(np.float64)
        tp_det = is_tp[order].astype(np.int64)
        fp_det = 1 - tp_det

        cum_tp = np.cumsum(tp_det)
        cum_fp = np.cumsum(fp_det)

        group_end = np.where(np.r_[s[:-1] != s[1:], True])[0]

        thr = s[group_end]
        tp = cum_tp[group_end]
        fp = cum_fp[group_end]

        if thr[0] < 1.0:
            thr = np.r_[1.0, thr]
            tp = np.r_[0, tp]
            fp = np.r_[0, fp]

        if thr[-1] > 0.0:
            thr = np.r_[thr, 0.0]
            tp = np.r_[tp, cum_tp[-1]]
            fp = np.r_[fp, cum_fp[-1]]

    fn = np.maximum(num_gt - tp, 0)
    num_pred = tp + fp

    denom_gt = max(num_gt, 1)
    denom_imgs = max(num_imgs, 1)
    total_fp = int(fp[-1]) if fp.size > 0 else 0
    denom_fp = max(total_fp, 1)

    recall = tp / (denom_gt + eps)
    tpr = recall.copy()
    fpr = fp / (denom_fp + eps)
    fppi = fp / (denom_imgs + eps)

    precision = tp / np.maximum(num_pred, 1)
    precision = np.where(num_pred > 0, precision, 1.0)

    return {
        "thr": thr.astype(np.float64),
        "tp": tp.astype(np.int64),
        "fp": fp.astype(np.int64),
        "fn": fn.astype(np.int64),
        "num_pred": num_pred.astype(np.int64),
        "recall": recall.astype(np.float64),
        "tpr": tpr.astype(np.float64),
        "fpr": fpr.astype(np.float64),
        "fppi": fppi.astype(np.float64),
        "precision": precision.astype(np.float64),
    }


def _unique_max_curve(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    ux, idx_start = np.unique(x, return_index=True)
    uy = np.zeros_like(ux, dtype=np.float64)

    for k in range(len(ux)):
        start = idx_start[k]
        end = idx_start[k + 1] if k + 1 < len(ux) else len(x)
        uy[k] = np.max(y[start:end])

    return ux, uy


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0

    ux, uy = _unique_max_curve(x, y)
    if ux.size < 2:
        return 0.0

    return float(np.trapezoid(uy, ux))


def recall_at_fppi(fppi: np.ndarray, tpr: np.ndarray, target_fppi: float) -> float:
    if fppi.size == 0:
        return 0.0
    mask = fppi <= float(target_fppi)
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def pauc_roc(fpr: np.ndarray, tpr: np.ndarray, fpr_max: float = 0.1) -> float:
    if fpr.size < 2:
        return 0.0

    fpr_max = float(fpr_max)
    ux, uy = _unique_max_curve(fpr, tpr)
    if ux.size < 2:
        return 0.0

    if ux[0] > fpr_max:
        return 0.0

    if ux[-1] < fpr_max:
        ux = np.r_[ux, fpr_max]
        uy = np.r_[uy, uy[-1]]
    elif ux[-1] > fpr_max:
        y_at = np.interp(fpr_max, ux, uy)
        keep = ux <= fpr_max
        ux = np.r_[ux[keep], fpr_max]
        uy = np.r_[uy[keep], y_at]

    auc = trapezoid_auc(ux, uy)
    return float(auc / max(fpr_max, 1e-12))


def froc_auc_up_to(fppi: np.ndarray, tpr: np.ndarray, u_max: float = 3.0) -> float:
    if fppi.size == 0:
        return 0.0

    u_max = float(u_max)
    ux, uy = _unique_max_curve(fppi, tpr)
    if ux.size == 0:
        return 0.0

    if ux[0] > 0.0:
        ux = np.r_[0.0, ux]
        uy = np.r_[0.0, uy]

    if ux[-1] < u_max:
        ux = np.r_[ux, u_max]
        uy = np.r_[uy, uy[-1]]
    elif ux[-1] > u_max:
        y_at = np.interp(u_max, ux, uy)
        keep = ux <= u_max
        ux = np.r_[ux[keep], u_max]
        uy = np.r_[uy[keep], y_at]

    auc = trapezoid_auc(ux, uy)
    return float(auc / max(u_max, 1e-12))


def compute_best_f1(
    thr: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, float]:
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, eps)
    if f1.size == 0:
        return {
            "best_f1": 0.0,
            "best_f1_thr": 1.0,
            "best_f1_precision": 0.0,
            "best_f1_recall": 0.0,
        }

    best_idx = int(np.argmax(f1))
    return {
        "best_f1": float(f1[best_idx]),
        "best_f1_thr": float(thr[best_idx]),
        "best_f1_precision": float(precision[best_idx]),
        "best_f1_recall": float(recall[best_idx]),
    }


def compute_curves_for_subset(
    conn: sqlite3.Connection,
    subset_id: int,
    class_id: int,
    iou_thr: float = 0.5,
    froc_fppi_max: float = 3.0,
):
    cur = conn.cursor()

    cur.execute(
        """
        SELECT v.id, v.image_id
        FROM subset_variants sv
        JOIN variants v ON sv.variant_id = v.id
        WHERE sv.subset_id = ?;
        """,
        (subset_id,),
    )
    variant_rows = cur.fetchall()
    if not variant_rows:
        print(f"Subset {subset_id}, class {class_id}: no variants found, skipping.")
        return

    scores_all: List[float] = []
    is_tp_all: List[bool] = []
    num_gt_total = 0
    num_imgs = len(variant_rows)

    for variant_id, image_id in variant_rows:
        cur.execute(
            """
            SELECT x, y, w, h, score
            FROM predictions
            WHERE variant_id = ? AND category_id = ?;
            """,
            (variant_id, class_id),
        )
        preds = cur.fetchall()
        if preds:
            preds = np.array(preds, dtype=np.float32)
            pred_xywh = preds[:, :4]
            pred_xyxy = xywh_to_xyxy(pred_xywh)
            pred_scores = preds[:, 4]
        else:
            pred_xyxy = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)

        cur.execute(
            """
            SELECT x, y, w, h
            FROM gt_objects
            WHERE image_id = ? AND class_id = ?;
            """,
            (image_id, class_id),
        )
        gts = cur.fetchall()
        if gts:
            gt_xywh = np.array(gts, dtype=np.float32)
            gt_xyxy = xywh_to_xyxy(gt_xywh)
        else:
            gt_xyxy = np.zeros((0, 4), dtype=np.float32)

        is_tp, n_gt = greedy_match_labels_one_image_np(
            pred_xyxy=pred_xyxy,
            pred_scores=pred_scores,
            gt_xyxy=gt_xyxy,
            iou_thr=iou_thr,
        )
        num_gt_total += int(n_gt)

        if pred_scores.size > 0:
            scores_all.extend(pred_scores.tolist())
            is_tp_all.extend(is_tp.tolist())

    scores = np.asarray(scores_all, dtype=np.float64)
    is_tp = np.asarray(is_tp_all, dtype=bool)

    sweep = compute_operating_sweep_from_global_labels(
        scores=scores,
        is_tp=is_tp,
        num_gt=num_gt_total,
        num_imgs=num_imgs,
    )

    roc_auc = trapezoid_auc(sweep["fpr"], sweep["tpr"])
    roc_pauc_01 = pauc_roc(sweep["fpr"], sweep["tpr"], fpr_max=0.1)
    froc_auc = froc_auc_up_to(sweep["fppi"], sweep["tpr"], u_max=froc_fppi_max)
    auprc = trapezoid_auc(sweep["recall"], sweep["precision"])

    r_at_01 = recall_at_fppi(sweep["fppi"], sweep["tpr"], 0.1)
    r_at_05 = recall_at_fppi(sweep["fppi"], sweep["tpr"], 0.5)
    r_at_10 = recall_at_fppi(sweep["fppi"], sweep["tpr"], 1.0)

    best_f1_stats = compute_best_f1(
        thr=sweep["thr"],
        precision=sweep["precision"],
        recall=sweep["recall"],
    )

    cur.execute(
        """
        DELETE FROM subset_eval_metrics
        WHERE subset_id = ?
        AND class_id = ?
        AND iou_thr = ?
        AND froc_fppi_max = ?;
        """,
        (subset_id, class_id, float(iou_thr), float(froc_fppi_max)),
    )

    cur.execute(
        """
        DELETE FROM subset_operating_points
        WHERE subset_id = ? AND class_id = ? AND iou_thr = ?;
        """,
        (subset_id, class_id, float(iou_thr)),
    )

    cur.execute(
        """
        INSERT INTO subset_eval_metrics (
            subset_id, class_id, iou_thr, froc_fppi_max,
            num_gt, num_imgs,
            roc_auc, roc_pauc_fpr_0p1, froc_auc, auprc,
            recall_fppi_0p1, recall_fppi_0p5, recall_fppi_1p0,
            best_f1, best_f1_thr, best_f1_precision, best_f1_recall
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            subset_id,
            class_id,
            float(iou_thr),
            float(froc_fppi_max),
            int(num_gt_total),
            int(num_imgs),
            float(roc_auc),
            float(roc_pauc_01),
            float(froc_auc),
            float(auprc),
            float(r_at_01),
            float(r_at_05),
            float(r_at_10),
            best_f1_stats["best_f1"],
            best_f1_stats["best_f1_thr"],
            best_f1_stats["best_f1_precision"],
            best_f1_stats["best_f1_recall"],
        ),
    )

    thr = sweep["thr"]
    tp = sweep["tp"]
    fp = sweep["fp"]
    fn = sweep["fn"]
    num_pred = sweep["num_pred"]
    recall = sweep["recall"]
    tpr = sweep["tpr"]
    fpr = sweep["fpr"]
    fppi = sweep["fppi"]
    precision = sweep["precision"]

    rows_points = [
        (
            subset_id,
            class_id,
            float(iou_thr),
            int(i),
            float(thr[i]),
            int(tp[i]),
            int(fp[i]),
            int(fn[i]),
            int(num_pred[i]),
            float(recall[i]),
            float(tpr[i]),
            float(fpr[i]),
            float(fppi[i]),
            float(precision[i]),
        )
        for i in range(len(thr))
    ]

    cur.executemany(
        """
        INSERT INTO subset_operating_points (
            subset_id, class_id, iou_thr,
            idx, thr, tp, fp, fn, num_pred, recall, tpr, fpr, fppi, precision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows_points,
    )

    print(
        f"Subset {subset_id}, class {class_id}: "
        f"ROC_AUC={roc_auc:.4f}, "
        f"ROC_pAUC@FPR<=0.1={roc_pauc_01:.4f}, "
        f"FROC_AUC@FPPI<={froc_fppi_max:g}={froc_auc:.4f}, "
        f"AUPRC={auprc:.4f}, "
        f"best_F1={best_f1_stats['best_f1']:.4f} @ thr={best_f1_stats['best_f1_thr']:.4f}, "
        f"R@0.1FPPI={r_at_01:.4f}, R@0.5FPPI={r_at_05:.4f}, R@1.0FPPI={r_at_10:.4f}, "
        f"num_imgs={num_imgs}, num_gt={num_gt_total}, Ndet={len(scores_all)}"
    )


def main():
    base_path = _P('databases_dir')

    parser = argparse.ArgumentParser(
        description="Compute subset-wise operating-point sweeps and summary metrics from SQLite DB."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=f"{base_path}/INRIAPerson.db",
        help="Path to SQLite DB file.",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=0,
        help="Class id to evaluate.",
    )
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP matching.",
    )
    parser.add_argument(
        "--subset_id",
        type=int,
        default=None,
        help="If set, only evaluate this subset; otherwise all subsets.",
    )
    parser.add_argument(
        "--froc_fppi_max",
        type=float,
        default=3.0,
        help="Upper FPPI limit for normalized FROC AUC.",
    )

    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")
    ensure_eval_tables(conn)
    cur = conn.cursor()

    if args.subset_id is not None:
        subset_ids = [args.subset_id]
    else:
        cur.execute("SELECT id FROM subsets;")
        subset_ids = [r[0] for r in cur.fetchall()]

    if not subset_ids:
        print("No subsets found.")
        conn.close()
        return

    for sid in subset_ids:
        compute_curves_for_subset(
            conn,
            subset_id=sid,
            class_id=args.class_id,
            iou_thr=args.iou_thr,
            froc_fppi_max=args.froc_fppi_max,
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
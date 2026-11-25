#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
True per-image threshold execution.

For every test image variant i a policy selects a threshold tau_i; tau_i is applied ONLY to the
stored detections of image i (keep detection j iff score_j >= tau_i); TP_i/FP_i/FN_i are counted;
counts are aggregated over images FIRST and precision/recall/F1/FPPI computed SECOND.

Matching is the unchanged protocol of data/database/generate_curves.py:
greedy_match_labels_one_image_np matches all stored detections of an image once, in descending
score order, against the image's GT (IoU >= iou_thr, one GT per detection). Because the matching
decision of a detection only depends on higher-scoring detections, thresholding at any tau keeps a
score-ordered prefix and the TP labels of the kept detections are unchanged. This is exactly how the
stored regime-level sweeps (subset_operating_points) are produced, so a fixed tau applied to all
images of a regime reproduces the stored sweep point (sanity check A).

The regime-curve utility of the earlier evaluation (F1_w(tau_i) on the complete empirical regime
curve) is kept and computed alongside as "regime_curve_utility".
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from dobi.data.database.generate_curves import greedy_match_labels_one_image_np, xywh_to_xyxy


# ------------------------------------------------------------------
# core: per-image counts for per-image thresholds
# ------------------------------------------------------------------
@dataclass
class DetectionTable:
    """Flat storage of all stored detections of a set of image variants."""
    variant_ids: np.ndarray      # [N]
    n_gt: np.ndarray             # [N] number of GT objects of the variant's base image
    offsets: np.ndarray          # [N+1] detections of variant k are scores[offsets[k]:offsets[k+1]]
    scores: np.ndarray           # [M]
    is_tp: np.ndarray            # [M] bool, from greedy matching of ALL stored detections

    @property
    def n(self):
        return len(self.variant_ids)

    def det_owner(self) -> np.ndarray:
        return np.repeat(np.arange(self.n), np.diff(self.offsets))


def per_image_counts(table: DetectionTable, tau: np.ndarray):
    """TP_i, FP_i, FN_i (int64 arrays [N]) when detections of image i are kept iff score >= tau[i]."""
    tau = np.asarray(tau, dtype=np.float64)
    if tau.shape != (table.n,):
        raise ValueError("one threshold per image variant required")
    owner = table.det_owner()
    keep = table.scores >= tau[owner]
    tp = np.bincount(owner, weights=(keep & table.is_tp), minlength=table.n).astype(np.int64)
    fp = np.bincount(owner, weights=(keep & ~table.is_tp), minlength=table.n).astype(np.int64)
    fn = table.n_gt.astype(np.int64) - tp
    return tp, fp, fn


def metrics_from_counts(tp, fp, fn, n_images) -> Dict[str, float]:
    """Precision/recall/F1/FPPI from aggregated counts (zero-safe, same conventions as generate_curves)."""
    tp, fp, fn = int(tp), int(fp), int(fn)
    n_pred = tp + fp
    prec = tp / n_pred if n_pred > 0 else 1.0       # generate_curves: precision := 1 with no predictions
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
            "fppi": fp / max(int(n_images), 1), "n_images": int(n_images)}


# ------------------------------------------------------------------
# loading stored detections (identical queries/matching to generate_curves.compute_curves_for_subset)
# ------------------------------------------------------------------
def load_split_detections(db_path: str, split: str = "test", class_id: int = 0, iou_thr: float = 0.5):
    """
    All variants of the split's subsets with their regime, base image and matched stored detections.
    Returns (DetectionTable, meta dict of arrays: subset_id, regime, deg_name, severity, rel_path, image_id).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT v.id, v.image_id, v.rel_path, v.deg_name, v.severity, s.id, s.name
        FROM subset_variants sv
        JOIN subsets s ON s.id = sv.subset_id
        JOIN variants v ON v.id = sv.variant_id
        WHERE s.split = ?
        ORDER BY s.id, v.id
        """,
        (split,),
    ).fetchall()
    vids = np.asarray([r[0] for r in rows], dtype=np.int64)
    if len(np.unique(vids)) != len(vids):
        raise RuntimeError("a variant belongs to more than one subset of the split")
    meta = {
        "variant_id": vids,
        "image_id": np.asarray([r[1] for r in rows], dtype=np.int64),
        "rel_path": np.asarray([r[2] for r in rows]),
        "deg_name": np.asarray([r[3] for r in rows]),
        "severity": np.asarray([r[4] for r in rows], dtype=np.int64),
        "subset_id": np.asarray([r[5] for r in rows], dtype=np.int64),
        "regime": np.asarray([r[6] for r in rows]),
    }
    pos = {int(v): k for k, v in enumerate(vids)}

    preds: Dict[int, list] = {}
    for vid, x, y, w, h, s in conn.execute(
        """
        SELECT p.variant_id, p.x, p.y, p.w, p.h, p.score
        FROM predictions p
        JOIN subset_variants sv ON sv.variant_id = p.variant_id
        JOIN subsets s ON s.id = sv.subset_id
        WHERE s.split = ? AND p.category_id = ?
        """,
        (split, int(class_id)),
    ):
        preds.setdefault(int(vid), []).append((x, y, w, h, s))
    gts: Dict[int, list] = {}
    for img, x, y, w, h in conn.execute(
        """SELECT g.image_id, g.x, g.y, g.w, g.h FROM gt_objects g JOIN images i ON i.id = g.image_id
           WHERE i.split = ? AND g.class_id = ?""", (split, int(class_id))):
        gts.setdefault(int(img), []).append((x, y, w, h))
    conn.close()

    scores, is_tp, n_gt, counts = [], [], np.zeros(len(vids), np.int64), np.zeros(len(vids), np.int64)
    for k, vid in enumerate(vids):
        p = np.asarray(preds.get(int(vid), []), dtype=np.float32).reshape(-1, 5)
        g = np.asarray(gts.get(int(meta["image_id"][k]), []), dtype=np.float32).reshape(-1, 4)
        tp, ng = greedy_match_labels_one_image_np(xywh_to_xyxy(p[:, :4]), p[:, 4], xywh_to_xyxy(g), iou_thr)
        scores.append(p[:, 4].astype(np.float64))
        is_tp.append(tp.astype(bool))
        n_gt[k] = ng
        counts[k] = len(p)
    table = DetectionTable(variant_ids=vids, n_gt=n_gt, offsets=np.r_[0, np.cumsum(counts)],
                           scores=np.concatenate(scores) if scores else np.zeros(0),
                           is_tp=np.concatenate(is_tp) if is_tp else np.zeros(0, bool))
    assert len(pos) == table.n
    return table, meta


# ------------------------------------------------------------------
# aggregation
# ------------------------------------------------------------------
def aggregate(tp, fp, fn, groups: np.ndarray, group_names: Optional[List[str]] = None):
    """Per-group metrics from summed counts + micro (pooled) + macro (mean over groups)."""
    names = group_names if group_names is not None else list(dict.fromkeys(groups.tolist()))
    per = {}
    for gname in names:
        m = groups == gname
        per[gname] = metrics_from_counts(tp[m].sum(), fp[m].sum(), fn[m].sum(), m.sum())
    micro = metrics_from_counts(tp.sum(), fp.sum(), fn.sum(), len(tp))
    macro = {k: float(np.mean([per[g][k] for g in names])) for k in ("precision", "recall", "f1", "fppi")}
    return per, micro, macro


def per_image_oracle_tau(table: DetectionTable) -> np.ndarray:
    """
    EXPLORATORY, GT-dependent upper bound: per image, the threshold maximising the image's own F1
    (ties -> highest threshold; images without GT -> threshold above all detections, i.e. no FP).
    """
    taus = np.empty(table.n)
    for k in range(table.n):
        s = table.scores[table.offsets[k]:table.offsets[k + 1]]
        t = table.is_tp[table.offsets[k]:table.offsets[k + 1]]
        ng = int(table.n_gt[k])
        if len(s) == 0 or ng == 0:
            taus[k] = 1.0 + 1e-9 if len(s) else 1.0
            continue
        order = np.argsort(-s, kind="stable")
        ss, tt = s[order], t[order]
        ctp, cfp = np.cumsum(tt), np.cumsum(~tt)
        ends = np.r_[np.where(ss[:-1] != ss[1:])[0], len(ss) - 1]   # keep tied scores together
        f1 = 2 * ctp[ends] / (ctp[ends] + cfp[ends] + ng)
        best = int(np.argmax(f1))                                     # first max = highest threshold
        taus[k] = ss[ends[best]] if f1[best] > 0 else 1.0 + 1e-9
    return taus

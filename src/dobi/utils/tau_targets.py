# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Threshold-indexed (tau-indexed) targets built directly from the raw detector
threshold sweeps stored in ``subset_operating_points``.

Semantics
---------
``generate_curves.compute_operating_sweep_from_global_labels`` stores one row
per unique detector score ``thr``; the row holds the cumulative statistics of
all detections with ``score >= thr`` (TP/FP matching, TPR, FPR, FPPI and
precision definitions are untouched here). The statistics are therefore a
right-continuous step function of the threshold:

    stats(tau) = row with the smallest stored thr >= tau

which is exactly what applying threshold ``tau`` to the stored detections
yields. All targets in this module are obtained by evaluating that step
function, so a target value is always a real operating point of the detector.

Deduplication policy
--------------------
Rows are keyed by ``thr``. Several rows with the same ``thr`` (only possible
through the synthetic 1.0/0.0 end points) are collapsed to the row with the
most detections (largest ``num_pred``), i.e. the row that represents
``score >= thr``. Threshold plateaus (an interval of tau without any detection
score inside) are not points in the table; the step lookup returns the same
statistics for every tau in the plateau. Thresholds are never inferred by
inverting ROC/FROC/PR curves.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np

from dobi.utils.roc_db_helpers import _cummax, _interp_curve, _precision_envelope, _prepare_xy
from dobi.utils.tau_spline import safe_f1, uniform_knots, weighted_quantile_knots

SWEEP_COLUMNS = ("thr", "tp", "fp", "fn", "num_pred", "recall", "tpr", "fpr", "fppi", "precision")
BEHAVIOR_KEYS = ("tpr", "fpr", "fppi", "precision", "recall", "f1")


# ============================================================
# Raw sweeps
# ============================================================

def get_split_subsets(conn: sqlite3.Connection, split: str, class_id: int, iou_thr: float) -> List[Dict]:
    """Subsets of a split that have stored sweeps, ordered by id (same order as build_tau_operating_targets)."""
    rows = conn.execute(
        """
        SELECT s.id, s.name, s.split
        FROM subsets s
        WHERE s.split = ?
          AND EXISTS (
              SELECT 1 FROM subset_operating_points p
              WHERE p.subset_id = s.id AND p.class_id = ? AND ABS(p.iou_thr - ?) < 1e-8
          )
        ORDER BY s.id
        """,
        (split, int(class_id), float(iou_thr)),
    ).fetchall()
    return [
        {"subset_id": int(sid), "name": str(name), "split": str(sp), "subset_index": i}
        for i, (sid, name, sp) in enumerate(rows)
    ]


def load_raw_sweep(conn: sqlite3.Connection, subset_id: int, class_id: int, iou_thr: float) -> Dict[str, np.ndarray]:
    """Raw sweep rows in stored order (descending thr)."""
    rows = conn.execute(
        f"""
        SELECT {", ".join(SWEEP_COLUMNS)}
        FROM subset_operating_points
        WHERE subset_id = ? AND class_id = ? AND ABS(iou_thr - ?) < 1e-8
        ORDER BY idx ASC
        """,
        (int(subset_id), int(class_id), float(iou_thr)),
    ).fetchall()
    if not rows:
        raise ValueError(f"No subset_operating_points for subset_id={subset_id}, class_id={class_id}, iou_thr={iou_thr}")
    arr = np.asarray(rows, dtype=np.float64)
    return {c: arr[:, i] for i, c in enumerate(SWEEP_COLUMNS)}


def dedupe_sweep(sweep: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Ascending-thr sweep with unique thresholds (see module docstring for the policy)."""
    thr = sweep["thr"]
    # sort by thr ascending, ties by num_pred descending -> first row of each thr group is kept
    order = np.lexsort((-sweep["num_pred"], thr))
    thr_sorted = thr[order]
    first = np.r_[True, thr_sorted[1:] != thr_sorted[:-1]]
    keep = order[first]
    return {k: v[keep] for k, v in sweep.items()}


def sweep_at_tau(sweep_asc: Dict[str, np.ndarray], tau: np.ndarray) -> Dict[str, np.ndarray]:
    """Evaluate the step function stats(tau) = row(min{thr >= tau}) on a deduplicated ascending sweep."""
    tau = np.asarray(tau, dtype=np.float64)
    idx = np.searchsorted(sweep_asc["thr"], tau, side="left")
    idx = np.clip(idx, 0, sweep_asc["thr"].size - 1)
    out = {k: v[idx] for k, v in sweep_asc.items()}
    out["thr_row"] = out.pop("thr")
    out["tau"] = tau
    out["f1"] = safe_f1(out["precision"], out["recall"])
    return out


def detector_metrics_at_tau(sweep_asc: Dict[str, np.ndarray], tau: float) -> Dict[str, float]:
    """Actual detector metrics obtained by applying threshold tau to the stored detections."""
    s = sweep_at_tau(sweep_asc, np.asarray([tau]))
    return {
        "tau": float(tau),
        "precision": float(s["precision"][0]),
        "recall": float(s["recall"][0]),
        "f1": float(s["f1"][0]),
        "fppi": float(s["fppi"][0]),
        "fpr": float(s["fpr"][0]),
        "tp": int(s["tp"][0]),
        "fp": int(s["fp"][0]),
        "fn": int(s["fn"][0]),
    }


def oracle_threshold(sweep: Dict[str, np.ndarray], criterion: str = "f1", max_fppi: float = 0.1) -> Optional[float]:
    """
    Ground-truth optimal threshold over the raw stored thresholds (non-deployable).
    Rows are scanned in stored (descending thr) order, so ties resolve to the
    highest threshold, matching generate_curves.compute_best_f1 and
    evaluation_curve_monitor.compute_cluster_op_tau.
    """
    if criterion == "f1":
        score = safe_f1(sweep["precision"], sweep["recall"])
    elif criterion == "youden":
        score = sweep["tpr"] - sweep["fpr"]
    elif criterion == "fppi":
        ok = sweep["fppi"] <= float(max_fppi)
        if not np.any(ok):
            return None
        score = np.where(ok, sweep["tpr"], -np.inf)
    else:
        raise ValueError(f"Unknown criterion '{criterion}'")
    if score.size == 0:
        return None
    return float(sweep["thr"][int(np.argmax(score))])


def pristine_validation_threshold(
    db_path: str,
    class_id: int,
    iou_thr: float,
    split: str = "val",
    criterion: str = "f1",
    max_fppi: float = 0.1,
) -> Tuple[float, Dict]:
    """
    Fixed deployment threshold calibrated only on the pristine subset of the
    validation split: tau_pristine = argmax_tau F1_pristine_val(tau).
    """
    conn = sqlite3.connect(db_path)
    try:
        subsets = get_split_subsets(conn, split, class_id, iou_thr)
        pristine = [s for s in subsets if "pristine" in s["name"].lower()]
        if not pristine:
            raise RuntimeError(f"No pristine subset found in split '{split}'.")
        info = pristine[0]
        sweep = load_raw_sweep(conn, info["subset_id"], class_id, iou_thr)
    finally:
        conn.close()
    tau = oracle_threshold(sweep, criterion=criterion, max_fppi=max_fppi)
    if tau is None:
        raise RuntimeError("Could not determine pristine validation threshold.")
    return tau, info


# ============================================================
# Knots
# ============================================================

def training_score_distribution(
    conn: sqlite3.Connection,
    class_id: int,
    iou_thr: float,
    split: str = "train",
    source: str = "all",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pooled detector-score histogram of a split, recovered exactly from the
    stored sweeps: #detections with score == thr_k is num_pred(thr_k) minus
    num_pred of the next higher stored threshold. source='tp' counts only
    true-positive detections (same difference on the tp column).
    """
    if source not in ("all", "tp"):
        raise ValueError("source must be 'all' or 'tp'")
    col = "num_pred" if source == "all" else "tp"
    values, weights = [], []
    for info in get_split_subsets(conn, split, class_id, iou_thr):
        s = dedupe_sweep(load_raw_sweep(conn, info["subset_id"], class_id, iou_thr))
        n = s[col]
        count = n - np.r_[n[1:], 0.0]  # ascending thr: higher thr has fewer detections
        values.append(s["thr"])
        weights.append(np.clip(count, 0.0, None))
    if not values:
        raise RuntimeError(f"No sweeps found for split '{split}'.")
    return np.concatenate(values), np.concatenate(weights)


def build_threshold_knots(
    placement: str,
    n_knots: int,
    db_path: Optional[str] = None,
    class_id: int = 0,
    iou_thr: float = 0.5,
    split: str = "train",
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    quantile_source: str = "all",
) -> np.ndarray:
    """
    Global knots shared by all images and regimes; quantile knots only see the training split.
    quantile_source: 'all' = all detector scores (default), 'tp' = true-positive scores only.
    """
    if placement == "uniform":
        return uniform_knots(n_knots, tau_min, tau_max)
    if placement == "quantile":
        if db_path is None:
            raise ValueError("quantile knots require db_path")
        if split != "train":
            raise ValueError("quantile knots must be computed from the training split only")
        conn = sqlite3.connect(db_path)
        try:
            values, weights = training_score_distribution(conn, class_id, iou_thr, split=split, source=quantile_source)
        finally:
            conn.close()
        return weighted_quantile_knots(values, weights, n_knots, tau_min, tau_max)
    raise ValueError(f"Unknown knot placement '{placement}', expected 'uniform' or 'quantile'")


# ============================================================
# Targets
# ============================================================

def geometric_targets_from_sweep(
    sweep: Dict[str, np.ndarray],
    roc_grid: np.ndarray,
    froc_grid: np.ndarray,
    pr_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """ROC/FROC/PR reparameterisation, identical to build_tau_operating_targets."""
    fpr, tpr, fppi, precision = sweep["fpr"], sweep["tpr"], sweep["fppi"], sweep["precision"]
    recall = tpr.copy()

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
    return {"roc_tpr": roc_tpr, "froc_tpr": froc_tpr, "pr_precision": pr_precision}


def build_tau_behavior_targets(
    db_path: str,
    class_id: int,
    iou_thr: float,
    split: str = "train",
    n_tau_points: int = 201,
    fppi_max: float = 2.0,
    n_geom_points: int = 201,
    tau_min: float = 0.0,
    tau_max: float = 1.0,
    criterion: str = "f1",
    max_fppi: float = 0.1,
):
    """
    Aligned threshold-indexed targets per proxy regime (subset) of one split.

    Returns
      subsets_info: list of {subset_id, name, split, subset_index}
      tau_grid:     [T] float32
      targets:      dict of float32 arrays
        tpr, fpr, fppi, precision, recall, f1, precision_env : [S, T] (step-evaluated, no clipping of FPPI)
        tp, fp, fn                                            : [S, T]
        tau_opt [S], f1_opt [S]                               : oracle operating point on raw thresholds
        roc_fpr_grid, froc_fppi_grid, pr_recall_grid          : geometric grids
        roc_tpr, froc_tpr, pr_precision                        : [S, G] geometric targets
    """
    tau_grid = np.linspace(tau_min, tau_max, int(n_tau_points), dtype=np.float64)
    roc_grid = np.linspace(0.0, 1.0, int(n_geom_points), dtype=np.float32)
    froc_grid = np.linspace(0.0, float(fppi_max), int(n_geom_points), dtype=np.float32)
    pr_grid = np.linspace(0.0, 1.0, int(n_geom_points), dtype=np.float32)

    conn = sqlite3.connect(db_path)
    try:
        subsets_info = get_split_subsets(conn, split, class_id, iou_thr)
        if not subsets_info:
            raise ValueError(f"No subsets with sweeps for split='{split}' in DB: {db_path}")

        acc: Dict[str, List[np.ndarray]] = {}
        for info in subsets_info:
            raw = load_raw_sweep(conn, info["subset_id"], class_id, iou_thr)
            s = sweep_at_tau(dedupe_sweep(raw), tau_grid)
            s["precision_env"] = np.maximum.accumulate(s["precision"])  # non-decreasing in tau (PR envelope)
            for k in BEHAVIOR_KEYS + ("precision_env", "tp", "fp", "fn"):
                acc.setdefault(k, []).append(s[k])

            tau_opt = oracle_threshold(raw, criterion=criterion, max_fppi=max_fppi)
            acc.setdefault("tau_opt", []).append(np.nan if tau_opt is None else tau_opt)
            acc.setdefault("f1_opt", []).append(float(np.max(safe_f1(raw["precision"], raw["recall"]))))

            geom = geometric_targets_from_sweep(raw, roc_grid, froc_grid, pr_grid)
            for k, v in geom.items():
                acc.setdefault(k, []).append(v)
    finally:
        conn.close()

    targets = {k: np.asarray(v, dtype=np.float32) for k, v in acc.items()}
    targets["roc_fpr_grid"] = roc_grid
    targets["froc_fppi_grid"] = froc_grid
    targets["pr_recall_grid"] = pr_grid
    return subsets_info, tau_grid.astype(np.float32), targets

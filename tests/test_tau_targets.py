# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import sqlite3

import numpy as np

from dobi.utils.roc_db_helpers import build_tau_operating_targets
from dobi.utils.tau_spline import safe_f1
from dobi.utils.tau_targets import (
    build_tau_behavior_targets,
    build_threshold_knots,
    dedupe_sweep,
    detector_metrics_at_tau,
    load_raw_sweep,
    oracle_threshold,
    pristine_validation_threshold,
    sweep_at_tau,
)

from conftest import CLASS_ID, IOU_THR


def _brute_force(raw, tau):
    keep = raw["scores"] >= tau
    tp = int(np.sum(raw["is_tp"] & keep))
    fp = int(np.sum(~raw["is_tp"] & keep))
    prec = tp / (tp + fp) if tp + fp > 0 else 1.0
    rec = tp / raw["num_gt"]
    return tp, fp, prec, rec, fp / raw["num_imgs"]


def test_step_targets_equal_direct_thresholding(synthetic_db):
    db, raw = synthetic_db
    conn = sqlite3.connect(db)
    r = raw[("train", "gaussian_blur_3_5")]
    asc = dedupe_sweep(load_raw_sweep(conn, r["subset_id"], CLASS_ID, IOU_THR))
    taus = np.r_[0.0, np.unique(r["scores"]), np.linspace(0, 1, 37), 1.0]
    s = sweep_at_tau(asc, taus)
    for i, tau in enumerate(taus):
        tp, fp, prec, rec, fppi = _brute_force(r, tau)
        assert s["tp"][i] == tp and s["fp"][i] == fp
        assert np.isclose(s["precision"][i], prec) and np.isclose(s["recall"][i], rec, atol=1e-9)
        assert np.isclose(s["fppi"][i], fppi, atol=1e-9)
    conn.close()


def test_threshold_ordering_and_dedupe(synthetic_db):
    db, raw = synthetic_db
    conn = sqlite3.connect(db)
    sweep = load_raw_sweep(conn, raw[("train", "pristine")]["subset_id"], CLASS_ID, IOU_THR)
    # stored order: descending threshold
    assert np.all(np.diff(sweep["thr"]) <= 0)
    # inject a duplicate threshold row with fewer detections; dedupe must keep the 'score >= thr' row
    dup = {k: np.r_[v, v[3]] for k, v in sweep.items()}
    dup["num_pred"][-1] -= 1
    asc = dedupe_sweep(dup)
    assert np.all(np.diff(asc["thr"]) > 0)
    j = int(np.where(asc["thr"] == sweep["thr"][3])[0][0])
    assert asc["num_pred"][j] == sweep["num_pred"][3]
    conn.close()


def test_behavior_targets_monotone_and_aligned(synthetic_db):
    db, _ = synthetic_db
    info, tau, t = build_tau_behavior_targets(db, CLASS_ID, IOU_THR, split="train", n_tau_points=101)
    assert [i["name"] for i in info] == ["pristine", "gaussian_blur_3_5", "gaussian_noise_3_5"]
    assert np.all(np.diff(tau) > 0)
    for k in ("tpr", "fpr", "fppi"):
        assert np.all(np.diff(t[k], axis=1) <= 1e-7), k
    assert np.all((t["precision"] >= 0) & (t["precision"] <= 1))
    assert np.allclose(t["recall"], t["tpr"])
    assert np.allclose(t["f1"], safe_f1(t["precision"], t["recall"]), atol=1e-6)
    assert np.all(np.diff(t["precision_env"], axis=1) >= -1e-7)


def test_geometric_targets_identical_to_existing_builder(synthetic_db):
    db, _ = synthetic_db
    _, _, new = build_tau_behavior_targets(db, CLASS_ID, IOU_THR, split="val", fppi_max=2.0, n_geom_points=201)
    _, _, old = build_tau_operating_targets(db, CLASS_ID, IOU_THR, split="val", fppi_max=2.0)
    for k in ("roc_tpr", "froc_tpr", "pr_precision"):
        assert np.array_equal(new[k], old[k]), k


def test_oracle_threshold_is_brute_force_argmax(synthetic_db):
    db, raw = synthetic_db
    conn = sqlite3.connect(db)
    r = raw[("test", "gaussian_noise_3_5")]
    sweep = load_raw_sweep(conn, r["subset_id"], CLASS_ID, IOU_THR)
    tau_star = oracle_threshold(sweep, "f1")
    cands = np.unique(np.r_[r["scores"], 0.0, 1.0])
    f1 = [safe_f1(np.array([_brute_force(r, c)[2]]), np.array([_brute_force(r, c)[3]]))[0] for c in cands]
    assert np.isclose(detector_metrics_at_tau(dedupe_sweep(sweep), tau_star)["f1"], max(f1))
    conn.close()


def test_pristine_threshold_uses_validation_pristine_only(synthetic_db):
    db, raw = synthetic_db
    tau, info = pristine_validation_threshold(db, CLASS_ID, IOU_THR, split="val")
    assert info["split"] == "val" and info["name"] == "pristine"
    conn = sqlite3.connect(db)
    val_sweep = load_raw_sweep(conn, raw[("val", "pristine")]["subset_id"], CLASS_ID, IOU_THR)
    test_sweep = load_raw_sweep(conn, raw[("test", "pristine")]["subset_id"], CLASS_ID, IOU_THR)
    conn.close()
    assert tau == oracle_threshold(val_sweep, "f1")
    assert tau != oracle_threshold(test_sweep, "f1")


def test_quantile_knots_only_see_training_scores(synthetic_db, tmp_path):
    db, raw = synthetic_db
    k1 = build_threshold_knots("quantile", 16, db, CLASS_ID, IOU_THR)
    assert k1[0] == 0.0 and k1[-1] == 1.0 and len(k1) == 16 and np.all(np.diff(k1) > 0)
    # wiping all non-train sweeps must not change the knots
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM subset_operating_points WHERE subset_id IN (SELECT id FROM subsets WHERE split != 'train')")
    conn.commit()
    conn.close()
    assert np.array_equal(k1, build_threshold_knots("quantile", 16, db, CLASS_ID, IOU_THR))
    try:
        build_threshold_knots("quantile", 16, db, CLASS_ID, IOU_THR, split="test")
        raise AssertionError("quantile knots on a non-training split must be rejected")
    except ValueError:
        pass
    k_tp = build_threshold_knots("quantile", 8, db, CLASS_ID, IOU_THR, quantile_source="tp")
    assert len(k_tp) == 8 and np.all(np.diff(k_tp) > 0)
    assert np.allclose(build_threshold_knots("uniform", 5), [0, 0.25, 0.5, 0.75, 1.0])

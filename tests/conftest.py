# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import sqlite3

import numpy as np
import pytest

from dobi.data.database.generate_curves import compute_operating_sweep_from_global_labels
from dobi.data.database.generate_db import create_schema

CLASS_ID = 0
IOU_THR = 0.5


def make_detections(rng, n_tp, n_fp, tp_loc, fp_loc):
    """Synthetic detector output: TP scores centred at tp_loc, FP scores near fp_loc."""
    tp = np.clip(rng.beta(8 * tp_loc, 8 * (1 - tp_loc), n_tp), 0.001, 0.999)
    fp = np.clip(rng.beta(2 * fp_loc, 2 * (1 - fp_loc), n_fp), 0.001, 0.999)
    scores = np.round(np.r_[tp, fp], 3)  # rounding creates tied scores
    is_tp = np.r_[np.ones(n_tp, bool), np.zeros(n_fp, bool)]
    return scores, is_tp


def insert_subset(conn, sid, name, split, scores, is_tp, num_gt, num_imgs):
    conn.execute(
        "INSERT INTO subsets (id, name, split, params_json, created_at) VALUES (?, ?, ?, '{}', 'now')",
        (sid, name, split),
    )
    conn.execute("INSERT INTO images (id, base_name, split) VALUES (?, ?, ?)", (sid, f"img{sid}", split))
    conn.execute("INSERT INTO variants (id, image_id, deg_name, severity, rel_path) VALUES (?, ?, ?, 0, ?)",
                 (sid, sid, name, f"{name}/img{sid}.jpg"))
    conn.execute("INSERT INTO subset_variants (subset_id, variant_id) VALUES (?, ?)", (sid, sid))
    s = compute_operating_sweep_from_global_labels(scores, is_tp, num_gt, num_imgs)
    rows = [
        (sid, CLASS_ID, IOU_THR, i, float(s["thr"][i]), int(s["tp"][i]), int(s["fp"][i]), int(s["fn"][i]),
         int(s["num_pred"][i]), float(s["recall"][i]), float(s["tpr"][i]), float(s["fpr"][i]),
         float(s["fppi"][i]), float(s["precision"][i]))
        for i in range(len(s["thr"]))
    ]
    conn.executemany(
        """INSERT INTO subset_operating_points (subset_id, class_id, iou_thr, idx, thr, tp, fp, fn, num_pred,
           recall, tpr, fpr, fppi, precision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.execute(
        """INSERT INTO subset_eval_metrics (subset_id, class_id, iou_thr, froc_fppi_max, num_gt, num_imgs)
           VALUES (?, ?, ?, 3.0, ?, ?)""",
        (sid, CLASS_ID, IOU_THR, num_gt, num_imgs),
    )
    return s


@pytest.fixture
def synthetic_db(tmp_path):
    """
    Three splits x three regimes. Test-split detections are deliberately extreme
    so that any leakage of test data into knots / calibration is visible.
    """
    db = tmp_path / "synthetic.db"
    create_schema(str(db))
    rng = np.random.default_rng(0)
    conn = sqlite3.connect(db)
    raw = {}
    sid = 1
    regimes = [("pristine", 0.7, 0.2), ("gaussian_blur_3_5", 0.45, 0.2), ("gaussian_noise_3_5", 0.3, 0.3)]
    for split in ("train", "val", "test"):
        for name, tp_loc, fp_loc in regimes:
            if split == "test":
                tp_loc, fp_loc = min(tp_loc + 0.25, 0.95), 0.9
            n_gt = 400
            scores, is_tp = make_detections(rng, 300, 900, tp_loc, fp_loc)
            insert_subset(conn, sid, name, split, scores, is_tp, n_gt, 200)
            raw[(split, name)] = dict(subset_id=sid, scores=scores, is_tp=is_tp, num_gt=n_gt, num_imgs=200)
            sid += 1
    conn.commit()
    conn.close()
    return str(db), raw

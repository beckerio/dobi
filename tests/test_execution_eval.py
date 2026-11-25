# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Tests for true per-image threshold execution (dobi.eval.execution_eval)."""
import numpy as np
import pytest

from dobi.data.database.generate_curves import compute_operating_sweep_from_global_labels
from dobi.eval.execution_eval import (DetectionTable, aggregate, metrics_from_counts, per_image_counts,
                                      per_image_oracle_tau)
from dobi.utils.tau_targets import dedupe_sweep, detector_metrics_at_tau, oracle_threshold, sweep_at_tau


def two_images():
    # image 0: GT 2, detections 0.9 TP, 0.6 FP, 0.3 TP ; image 1: GT 1, detections 0.8 FP, 0.4 TP, 0.2 FP
    return DetectionTable(variant_ids=np.array([10, 11]), n_gt=np.array([2, 1]), offsets=np.array([0, 3, 6]),
                          scores=np.array([0.9, 0.6, 0.3, 0.8, 0.4, 0.2]),
                          is_tp=np.array([True, False, True, False, True, False]))


def test_each_tau_applies_only_to_its_image():
    t = two_images()
    tp, fp, fn = per_image_counts(t, np.array([0.5, 0.1]))
    assert tp.tolist() == [1, 1] and fp.tolist() == [1, 2] and fn.tolist() == [1, 0]
    # changing image 1's threshold leaves image 0 untouched
    tp2, fp2, fn2 = per_image_counts(t, np.array([0.5, 0.85]))
    assert (tp2[0], fp2[0], fn2[0]) == (tp[0], fp[0], fn[0])
    assert (tp2[1], fp2[1], fn2[1]) == (0, 0, 1)


def test_pooled_counts_aggregate_first():
    t = two_images()
    tp, fp, fn = per_image_counts(t, np.array([0.5, 0.35]))
    per, micro, macro = aggregate(tp, fp, fn, np.array(["a", "a"]))
    assert (micro["tp"], micro["fp"], micro["fn"]) == (2, 2, 1)
    assert micro["f1"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 1))
    # not the mean of per-image F1 values
    f1_img = [metrics_from_counts(tp[i], fp[i], fn[i], 1)["f1"] for i in range(2)]
    assert micro["f1"] != pytest.approx(np.mean(f1_img))
    assert macro["f1"] == pytest.approx(per["a"]["f1"])


def test_fixed_tau_reproduces_regime_sweep():
    rng = np.random.default_rng(0)
    n_img = 40
    counts = rng.integers(0, 8, n_img)
    scores = np.round(rng.uniform(0.001, 1, counts.sum()), 3)
    is_tp = rng.uniform(size=counts.sum()) < 0.4
    n_gt = np.array([is_tp[o:o + c].sum() + rng.integers(0, 3) for o, c in zip(np.r_[0, np.cumsum(counts)[:-1]], counts)])
    t = DetectionTable(np.arange(n_img), n_gt, np.r_[0, np.cumsum(counts)], scores, is_tp)
    sweep = dedupe_sweep({k: np.asarray(v, float) for k, v in
                          compute_operating_sweep_from_global_labels(scores, is_tp, n_gt.sum(), n_img).items()})
    for tau in (0.0, 0.05, 0.31, 0.5, 0.999, oracle_threshold(sweep, "f1")):
        tp, fp, fn = per_image_counts(t, np.full(n_img, tau))
        ref = detector_metrics_at_tau(sweep, tau)
        assert (tp.sum(), fp.sum(), fn.sum()) == (ref["tp"], ref["fp"], ref["fn"])
        assert metrics_from_counts(tp.sum(), fp.sum(), fn.sum(), n_img)["f1"] == pytest.approx(ref["f1"])


def test_execution_differs_from_regime_curve_utility():
    t = two_images()
    tau = np.array([0.95, 0.5])  # image 0 keeps nothing; image 1 keeps only its FP at 0.8
    tp, fp, fn = per_image_counts(t, tau)
    exec_f1 = metrics_from_counts(tp.sum(), fp.sum(), fn.sum(), 2)["f1"]
    sweep = dedupe_sweep({k: np.asarray(v, float) for k, v in compute_operating_sweep_from_global_labels(
        t.scores, t.is_tp, int(t.n_gt.sum()), 2).items()})
    curve_util = float(np.mean(sweep_at_tau(sweep, tau)["f1"]))
    assert abs(exec_f1 - curve_util) > 0.05


def test_empty_gt_and_no_detection_images():
    t = DetectionTable(np.array([0, 1, 2]), np.array([0, 2, 0]), np.array([0, 2, 2, 2]),
                       np.array([0.7, 0.2]), np.array([False, False]))
    tp, fp, fn = per_image_counts(t, np.array([0.5, 0.5, 0.5]))
    assert tp.tolist() == [0, 0, 0] and fp.tolist() == [1, 0, 0] and fn.tolist() == [0, 2, 0]
    m = metrics_from_counts(0, 0, 0, 1)
    assert m["precision"] == 1.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    taus = per_image_oracle_tau(t)
    assert taus[0] > 0.7 and taus[2] == 1.0


def test_threshold_policy_receives_no_ground_truth():
    """Adaptive thresholds come from predict_taus(ckpt, cache, variant_ids, ...) - GT is not an argument."""
    import inspect
    from dobi.eval.run_execution_eval import predict_taus
    params = set(inspect.signature(predict_taus).parameters)
    assert params == {"ckpt", "cache", "variant_ids", "tau_grid", "device", "batch"}


def test_one_threshold_per_image_required():
    with pytest.raises(ValueError):
        per_image_counts(two_images(), np.array([0.5]))

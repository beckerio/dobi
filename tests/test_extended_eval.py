# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Tests for the extended evaluation (dobi.eval.extended): protocol identities and output protection."""
import numpy as np
import pytest
import torch

from dobi.data.database.generate_curves import compute_operating_sweep_from_global_labels
from dobi.eval.execution_eval import DetectionTable, per_image_counts
from dobi.eval.extended import common as C
from dobi.eval.extended.exp_a_batch import sample_batches
from dobi.eval.extended.exp_b_equivalence import behavior_from_sweep, distance, pair_class, scales
from dobi.eval.extended.exp_d_consistency import reconstructed_precision
from dobi.utils.tau_spline import select_operating_threshold
from dobi.utils.tau_targets import dedupe_sweep, sweep_at_tau


def random_table(seed=0, n_img=60):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 9, n_img)
    scores = np.round(rng.uniform(0.001, 1, counts.sum()), 3)
    is_tp = rng.uniform(size=counts.sum()) < 0.4
    offs = np.r_[0, np.cumsum(counts)]
    n_gt = np.array([is_tp[offs[i]:offs[i + 1]].sum() + rng.integers(0, 3) for i in range(n_img)])
    return DetectionTable(np.arange(n_img), n_gt, offs, scores, is_tp)


def test_grid_counts_equal_per_image_execution():
    t = random_table()
    tp, fp = C.per_image_grid_counts(t)
    for q in (0, 1, 137, 500, 999, 1000):
        a, b, _ = per_image_counts(t, np.full(t.n, float(C.TAU[q])))
        assert (tp[:, q] == a).all() and (fp[:, q] == b).all()


def test_empirical_functions_reproduce_global_sweep():
    t = random_table(1)
    tp, fp = C.per_image_grid_counts(t)
    e = C.empirical_functions(tp.sum(0)[None], fp.sum(0)[None], np.array([t.n_gt.sum()]), np.array([t.n]))
    sw = dedupe_sweep({k: np.asarray(v, float) for k, v in
                       compute_operating_sweep_from_global_labels(t.scores, t.is_tp, int(t.n_gt.sum()), t.n).items()})
    ref = sweep_at_tau(sw, C.TAU.astype(np.float64))
    for k in ("tpr", "fppi", "precision", "fpr"):
        np.testing.assert_allclose(e[k][0], ref[k], atol=1e-6)


def monotone_curves(rng, n, q):
    tpr = np.sort(rng.uniform(0, 1, (n, q)), axis=1)[:, ::-1].astype(np.float32)
    fppi = (np.sort(rng.uniform(0, 3, (n, q)), axis=1)[:, ::-1] * rng.uniform(0.2, 1.0, (n, 1))).astype(np.float32)
    prec = np.sort(rng.uniform(0.2, 1, (n, q)), axis=1).astype(np.float32)
    return tpr, fppi, prec


@pytest.mark.parametrize("budget", [0.01, 0.1, 0.5, 5.0])
def test_fppi_policy_matches_select_operating_threshold(budget):
    rng = np.random.default_rng(2)
    tpr, fppi, _ = monotone_curves(rng, 40, len(C.TAU))
    fppi[0] = 9.0  # a row with no feasible threshold -> lowest predicted FPPI
    ours = C.fppi_constrained_tau(tpr, fppi, budget)
    ref, _ = select_operating_threshold(torch.as_tensor(C.TAU, dtype=torch.float64),
                                        {"tpr_tau": torch.as_tensor(tpr, dtype=torch.float64),
                                         "fppi_tau": torch.as_tensor(fppi, dtype=torch.float64)}, "fppi", budget)
    np.testing.assert_allclose(ours, ref.numpy(), atol=1e-7)


def test_f1_policy_matches_select_operating_threshold():
    rng = np.random.default_rng(3)
    tpr, _, prec = monotone_curves(rng, 30, len(C.TAU))
    ours = C.argmax_f1_tau(tpr, prec)
    ref, _ = select_operating_threshold(torch.as_tensor(C.TAU, dtype=torch.float64),
                                        {"tpr_tau": torch.as_tensor(tpr, dtype=torch.float64),
                                         "precision_tau": torch.as_tensor(prec, dtype=torch.float64)}, "f1")
    np.testing.assert_allclose(ours, ref.numpy(), atol=1e-7)


class _Ctx:
    def __init__(self):
        rng = np.random.default_rng(4)
        self.reg_names = ["pristine", "fog_3_5"]
        reg = ["pristine"] * 50 + ["fog_3_5"] * 150
        self.regimes = np.array(reg)
        img = np.r_[np.arange(50), np.repeat(np.arange(50), 3)]
        self.meta = {"image_id": img}


def test_batches_use_distinct_base_images_and_are_reproducible():
    ctx = _Ctx()
    b1 = sample_batches(ctx, (1, 8, 32), 50, seed=7)
    b2 = sample_batches(ctx, (1, 8, 32), 50, seed=7)
    for k, v in b1.items():
        assert (v == b2[k]).all()
        r = k[0]
        assert np.all(ctx.regimes[v] == r)
        for row in v:
            assert len(set(ctx.meta["image_id"][row])) == len(row)


def test_precision_and_fpr_identities_hold_for_empirical_sweeps():
    t = random_table(5, 200)
    sw = dedupe_sweep({k: np.asarray(v, float) for k, v in
                       compute_operating_sweep_from_global_labels(t.scores, t.is_tp, int(t.n_gt.sum()), t.n).items()})
    s = sweep_at_tau(sw, C.TAU.astype(np.float64))
    rho = t.n_gt.sum() / t.n
    np.testing.assert_allclose(reconstructed_precision(s["tpr"], s["fppi"], rho), s["precision"], atol=1e-6)
    np.testing.assert_allclose(s["fpr"] * s["fppi"][0], s["fppi"], atol=1e-6)


def test_behavior_distances():
    a = behavior_from_sweep(compute_operating_sweep_from_global_labels(*_sweep_inputs(6)))
    b = behavior_from_sweep(compute_operating_sweep_from_global_labels(*_sweep_inputs(7)))
    s = scales([a, b])
    for m in ("std_l2", "std_sup", "paper_rms", "geometric_mad", "decision"):
        assert distance(a, a, m, s) == pytest.approx(0.0, abs=1e-12)
        assert distance(a, b, m, s) == pytest.approx(distance(b, a, m, s))
        assert distance(a, b, m, s) >= 0
    assert pair_class("fog_s3", "fog_s5") == "same_degradation"
    assert pair_class("fog_s3", "snow_s3") == "within_family"
    assert pair_class("fog_s3", "gaussian_noise_s3") == "cross_family"
    assert pair_class("pristine", "brightness_s1") == "pristine_pair"


def _sweep_inputs(seed):
    t = random_table(seed, 120)
    return t.scores, t.is_tp, int(t.n_gt.sum()), t.n


def test_output_protection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        C.check_out_dir(tmp_path / 'results/main_study/paper_artifacts')
    run = tmp_path / C.OUT_ROOT / "run_x"
    run.mkdir(parents=True)
    assert C.check_out_dir(run) == run.resolve()
    f = C.new_file(run / "a.txt")
    f.write_text("x")
    with pytest.raises(FileExistsError):
        C.new_file(run / "a.txt")


def test_density_never_from_test():
    with pytest.raises(ValueError):
        C.train_gt_density(split="test")

# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Tests for the components added for the final paper experiments."""
import json

import numpy as np
import pytest
import torch

import dobi.eval.evaluation_curve_monitor as ev
from dobi.data.feature_cache import CachedFeatureDataset, FeatureCache, pool_feats, pooled_to_pseudo_feats
from dobi.models.monitor_factory import build_monitor_head, decode_monitor_output
from dobi.models.pc1_behavior_model import PC1BehaviorHead, fit_behavior_pca, project_pc1
from dobi.models.shared_anchor_monitor_model import TauSplineSharedAnchorHead
from dobi.models.spline_monitor_model import TauSplineSingleHead
from dobi.models.tau_behavior_model import TauBehaviorHead
from dobi.training.train_curve_monitor import exclude_subsets
from dobi.utils.tau_spline import linear_interp, pchip_interp
from dobi.utils.tau_targets import build_tau_behavior_targets

CH = [8, 16]


def _feats(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, c, 4, 5, generator=g) for c in CH]


def test_pseudo_feats_reproduce_pooled_features_exactly():
    f = _feats()
    x = pool_feats(f)
    assert torch.allclose(pool_feats(pooled_to_pseudo_feats(x, CH)), x, atol=1e-6)


@pytest.mark.parametrize("make", [
    lambda: TauBehaviorHead(CH, np.linspace(0, 1, 9), hidden_width=16),
    lambda: TauSplineSingleHead(CH, hidden_width=16, n_anchors=8),
    lambda: TauSplineSharedAnchorHead(CH, hidden_width=16, n_anchors=8, learn_shared_tau=True),
])
def test_heads_identical_on_cached_pooled_features(make):
    torch.manual_seed(0)
    head = make().eval()
    f = _feats()
    a = head(f)
    b = head(pooled_to_pseudo_feats(pool_feats(f), CH))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), k


def test_feature_cache_lookup(tmp_path):
    (tmp_path / "test").mkdir()
    x = np.random.default_rng(0).normal(size=(5, 2 * sum(CH))).astype(np.float32)
    np.save(tmp_path / "test" / "features.npy", x)
    np.savez(tmp_path / "test" / "index.npz", variant_ids=np.array([10, 11, 12, 13, 14]),
             rel_paths=np.array([f"fog/3/{i}.jpg" for i in range(5)]))
    (tmp_path / "meta.json").write_text(json.dumps({"feat_channels": CH, "weights": 'weights/yolov10n.pt', "imgsz": 640}))
    cache = FeatureCache(tmp_path, "test")
    assert np.array_equal(cache.get([cache.row_for_relpath("fog/3/3.jpg")])[0], x[3])

    class _Base:
        samples = [("p", 1, 0, 14), ("p", 0, 0, 10), ("p", 1, 0, 14)]

    ds = CachedFeatureDataset(_Base(), cache)
    assert np.allclose(ds[0]["feat"].numpy(), x[4]) and ds[1]["subset_index"] == 0


def test_seed_reproducibility_of_training_steps():
    def train(seed):
        torch.manual_seed(seed)
        head = TauBehaviorHead(CH, np.linspace(0, 1, 9), hidden_width=16)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        x = torch.randn(32, 2 * sum(CH), generator=torch.Generator().manual_seed(1))
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=8, shuffle=True)
        for (xb,) in loader:
            d = head.decode(head(pooled_to_pseudo_feats(xb, CH)), torch.linspace(0, 1, 51))
            loss = d["f1_tau"].mean() + d["fppi_tau"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return torch.cat([p.detach().flatten() for p in head.parameters()])

    assert torch.equal(train(3), train(3))
    assert not torch.equal(train(3), train(4))


def test_fppi_numerically_stable_for_extreme_logits():
    head = TauBehaviorHead(CH, np.linspace(0, 1, 9), hidden_width=16, fppi_space="log1p")
    with torch.no_grad():
        head.fppi_branch.bias.fill_(1e4)
    d = head.decode(head(_feats()), torch.linspace(0, 1, 101))
    assert torch.isfinite(d["fppi_tau"]).all() and float(d["fppi_tau"].max()) <= 1000.0 + 1e-3


def test_linear_and_pchip_agree_on_linear_data():
    xk = torch.tensor([0.0, 0.1, 0.35, 0.6, 1.0], dtype=torch.float64)
    yk = (0.9 - 0.7 * xk)[None]
    xq = torch.linspace(0, 1, 301, dtype=torch.float64)
    assert torch.allclose(linear_interp(xk, yk, xq), pchip_interp(xk, yk, xq), atol=1e-12)


def test_mean_behavior_data_boundaries(synthetic_db):
    db, _ = synthetic_db
    tau = np.linspace(0, 1, 51, dtype=np.float32)
    with pytest.raises(ValueError):
        ev.MeanBehaviorPredictor(db, 0, 0.5, tau, split="test")
    p = ev.MeanBehaviorPredictor(db, 0, 0.5, tau, split="train", exclude_regimes=["gaussian_noise_3_5"])
    assert p.source == {"split": "train", "regimes": ["pristine", "gaussian_blur_3_5"]}
    _, t = build_tau_behavior_targets(db, 0, 0.5, split="train", n_tau_points=51)[1:]
    assert np.allclose(p.curves["tpr_tau"], t["tpr"][:2].mean(axis=0))


def test_exclude_subsets_reindexes_targets(synthetic_db):
    db, _ = synthetic_db
    info, tau, t = build_tau_behavior_targets(db, 0, 0.5, split="train", n_tau_points=51)
    info2, t2 = exclude_subsets(info, t, ["gaussian_blur_3_5"])
    assert [s["name"] for s in info2] == ["pristine", "gaussian_noise_3_5"]
    assert [s["subset_index"] for s in info2] == [0, 1]
    assert np.array_equal(t2["tpr"][1], t["tpr"][2]) and np.array_equal(t2["roc_fpr_grid"], t["roc_fpr_grid"])


def test_pc1_head_is_valid_and_reconstructs_training_regimes(synthetic_db):
    db, _ = synthetic_db
    info, tau, t = build_tau_behavior_targets(db, 0, 0.5, split="train", n_tau_points=101)
    pca = fit_behavior_pca(t, tau)
    assert np.allclose(project_pc1(pca, t), pca["train_coefs"])
    head = PC1BehaviorHead(CH, pca, hidden_width=16)
    targs = {"head": "pc1_behavior", "hidden": 16, "mlp_layers": 2, **head.config()}
    clone = build_monitor_head("pc1_behavior", CH, targs)
    clone.load_state_dict(head.state_dict())
    out = {"coef_norm": torch.tensor((np.asarray(pca["train_coefs"]) - pca["coef_mean"]) / pca["coef_std"],
                                     dtype=torch.float32)}
    d = decode_monitor_output("pc1_behavior", clone, out, torch.as_tensor(tau), u_max=2.0)
    for k in ("tpr_tau", "fpr_tau", "fppi_tau"):
        assert torch.all(d[k][:, 1:] <= d[k][:, :-1] + 1e-7)
    assert torch.all((d["precision_tau"] >= 0) & (d["precision_tau"] <= 1))
    # with 3 training regimes PC1 is exact up to the residual of the other components
    err = np.abs(d["tpr_tau"].numpy() - t["tpr"]).mean()
    assert err < 0.1


def test_batched_cache_fetch_matches_per_item_loader():
    class _Cache:
        def row_for_variant(self, v):
            return v

        def get(self, rows):
            return np.asarray(rows, dtype=np.float32)[:, None] * np.ones((1, 4), dtype=np.float32)

    class _Base:
        samples = [("p", i % 3, 0, i) for i in range(37)]

    ds = CachedFeatureDataset(_Base(), _Cache())

    class _PerItem(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return ds[i]

    def batches(d):
        torch.manual_seed(0)
        return list(torch.utils.data.DataLoader(d, batch_size=8, shuffle=True, collate_fn=ds.collate_fn))

    for a, b in zip(batches(ds), batches(_PerItem())):
        assert torch.equal(a["feat"], b["feat"]) and torch.equal(a["subset_index"], b["subset_index"])

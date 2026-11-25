# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import numpy as np
import pytest
import torch

from dobi.models.monitor_factory import build_monitor_head, decode_monitor_output
from dobi.models.shared_anchor_monitor_model import TauSplineSharedAnchorHead
from dobi.models.spline_monitor_model import TauSplineSingleHead
from dobi.models.tau_behavior_model import TauBehaviorHead
from dobi.utils.spline_helpers import decode_tau_spline_head

CH = [16, 32, 64]
TAU = torch.linspace(0, 1, 501)


def feats(b=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, c, 5, 5, generator=g) * 3 for c in CH]


@pytest.mark.parametrize("interp", ["linear", "pchip"])
@pytest.mark.parametrize("fppi_space", ["log1p", "linear"])
@pytest.mark.parametrize("envelope", [False, True])
def test_constraints_hold_by_construction(interp, fppi_space, envelope):
    torch.manual_seed(1)
    knots = np.r_[0.0, np.sort(np.random.default_rng(0).uniform(0.01, 0.99, 14)), 1.0]
    head = TauBehaviorHead(CH, knots, hidden_width=32, interp=interp, fppi_space=fppi_space,
                           precision_envelope=envelope)
    for p in head.parameters():  # large random weights stress the parameterisation
        torch.nn.init.normal_(p, std=2.0)
    out = head(feats())
    d = head.decode(out, TAU)
    for k in ("tpr_tau", "fpr_tau", "fppi_tau"):
        assert torch.all(d[k][:, 1:] - d[k][:, :-1] <= 1e-6), k
    for k in ("tpr_tau", "fpr_tau", "precision_tau", "f1_tau"):
        assert torch.all((d[k] >= 0) & (d[k] <= 1)), k
    assert torch.all(d["fppi_tau"] >= 0)
    if envelope:
        assert torch.all(d["precision_tau"][:, 1:] - d["precision_tau"][:, :-1] >= -1e-6)
    # decode reproduces control values at the knots
    dk = head.decode(out, head.knots)
    assert torch.allclose(dk["tpr_tau"], out["tpr_ctrl"], atol=1e-5)


def test_decode_is_fp32_under_autocast():
    head = TauBehaviorHead(CH, np.linspace(0, 1, 16), hidden_width=32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        d = head.decode(head(feats()), TAU)
    assert d["tpr_tau"].dtype == torch.float32


def test_factory_roundtrip_tau_behavior():
    head = TauBehaviorHead(CH, np.r_[0, 0.05, 0.2, 0.5, 1.0], hidden_width=32, interp="linear", fppi_space="linear")
    train_args = {"head": "tau_behavior", "hidden": 32, "mlp_layers": 2, **head.config()}
    clone = build_monitor_head("tau_behavior", CH, train_args)
    clone.load_state_dict(head.state_dict())
    f = feats()
    a = decode_monitor_output("tau_behavior", head, head(f), TAU, u_max=2.0)
    b = decode_monitor_output("tau_behavior", clone, clone(f), TAU, u_max=2.0)
    for k in a:
        assert torch.equal(a[k], b[k])


@pytest.mark.parametrize("head_type", ["tau_spline_single", "tau_shared_anchor"])
def test_legacy_heads_still_load_and_decode_as_trained(head_type):
    """Old checkpoints (train_args + monitor_state_dict) load through the factory and decode like in training."""
    if head_type == "tau_spline_single":
        old = TauSplineSingleHead(CH, hidden_width=32, n_anchors=12)
        train_args = {"head": head_type, "hidden": 32, "n_anchors": 12, "mlp_layers": 2}
    else:
        old = TauSplineSharedAnchorHead(CH, hidden_width=32, n_anchors=12, learn_shared_tau=True)
        train_args = {"head": head_type, "hidden": 32, "n_anchors": 12, "mlp_layers": 2, "learn_shared_tau": True}
    ckpt = {"monitor_state_dict": old.state_dict(), "train_args": train_args}

    new = build_monitor_head(head_type, CH, ckpt["train_args"])
    missing, unexpected = new.load_state_dict(ckpt["monitor_state_dict"], strict=True)
    assert not missing and not unexpected

    tau = torch.linspace(0, 1, 75)
    f = feats()
    ref = decode_tau_spline_head(old(f), tau, 2.0, use_shared_anchor_grid=(head_type == "tau_shared_anchor"))
    got = decode_monitor_output(head_type, new, new(f), tau, u_max=2.0)
    for k in ("tpr_tau", "fpr_tau", "fppi_tau", "precision_tau", "f1_tau"):
        assert torch.allclose(ref[k], got[k]), k

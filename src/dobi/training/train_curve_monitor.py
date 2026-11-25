#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
from dobi.paths import P as _P  # machine-specific paths (configs/local_paths_<host>.yaml)
import json
import argparse
import sys
import gc
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO

from dobi.data.dataloader.roc_curves_data_loader import CurveMonitorDataset
from dobi.utils.roc_db_helpers import build_tau_operating_targets
from dobi.utils.files import next_run_dir
from dobi.models.spline_monitor_model import TauSplineSingleHead, forward_backbone, build_backbone_runner
from dobi.models.shared_anchor_monitor_model import TauSplineSharedAnchorHead 
from dobi.utils.spline_helpers import decode_tau_spline_head 
from dobi.models.tau_behavior_model import TauBehaviorHead
from dobi.training.tau_behavior_losses import compute_tau_behavior_losses
from dobi.utils.tau_targets import build_tau_behavior_targets, build_threshold_knots
from dobi.models.pc1_behavior_model import PC1BehaviorHead, fit_behavior_pca, project_pc1
from dobi.data.feature_cache import CachedFeatureDataset, FeatureCache, check_cache_matches, pooled_to_pseudo_feats


def exclude_subsets(subsets_info, targets_np, excluded):
    """Drop regimes (by subset name) from a split; targets are re-indexed consistently."""
    if not excluded:
        return subsets_info, targets_np
    keep = [i for i, s in enumerate(subsets_info) if s["name"] not in set(excluded)]
    S = len(subsets_info)
    targets_np = {k: (v[keep] if (np.ndim(v) >= 1 and v.shape[0] == S and not k.endswith("_grid")) else v)
                  for k, v in targets_np.items()}
    info = []
    for new_i, i in enumerate(keep):
        e = dict(subsets_info[i])
        e["subset_index"] = new_i
        e["index"] = new_i
        info.append(e)
    print(f"Excluded regimes {sorted(set(excluded))}: {S} -> {len(info)} subsets")
    return info, targets_np

def compute_losses(pred, targets, subset_idx, tau_grid, args, device):
    target_tpr = targets["tpr"][subset_idx]
    target_fpr = targets["fpr"][subset_idx]
    target_fppi = targets["fppi"][subset_idx]
    target_fppi_norm = targets["fppi_norm"][subset_idx]
    target_precision = targets["precision"][subset_idx]
    target_f1 = targets["f1"][subset_idx]

    loss_fpr = F.smooth_l1_loss(
        safe_log(pred["fpr_tau"], eps=1e-8),
        safe_log(target_fpr, eps=1e-8),
    )
    loss_fppi = F.smooth_l1_loss(
        safe_log1p(pred["fppi_tau"]),
        safe_log1p(target_fppi),
    )
    loss_tpr = F.smooth_l1_loss(pred["tpr_tau"], target_tpr)
    loss_precision = F.smooth_l1_loss(pred["precision_tau"], target_precision)

    if args.w_area > 0.0:
        pred_area_tpr = torch.trapz(pred["tpr_tau"], tau_grid, dim=1)
        pred_area_fpr = torch.trapz(pred["fpr_tau"], tau_grid, dim=1)
        pred_area_fppi = torch.trapz(pred["fppi_norm_tau"], tau_grid, dim=1)
        pred_area_precision = torch.trapz(pred["precision_tau"], tau_grid, dim=1)

        target_area_tpr = torch.trapz(target_tpr, tau_grid, dim=1)
        target_area_fpr = torch.trapz(target_fpr, tau_grid, dim=1)
        target_area_fppi = torch.trapz(target_fppi_norm, tau_grid, dim=1)
        target_area_precision = torch.trapz(target_precision, tau_grid, dim=1)

        loss_area = (
            F.l1_loss(pred_area_tpr, target_area_tpr)
            + F.l1_loss(pred_area_fpr, target_area_fpr)
            + F.l1_loss(pred_area_fppi, target_area_fppi)
            + F.l1_loss(pred_area_precision, target_area_precision)
        )
    else:
        loss_area = torch.zeros((), device=device)

    loss = (
        args.w_curve * (loss_tpr + loss_fpr + loss_fppi + loss_precision)
        + args.w_area * loss_area
    )

    return loss, loss_tpr, loss_fpr, loss_fppi, loss_precision, loss_area


def compute_legacy_losses(pred, targets, subset_idx, tau_grid, args, device):
    loss, loss_tpr, loss_fpr, loss_fppi, loss_precision, loss_area = compute_losses(
        pred, targets, subset_idx, tau_grid, args, device
    )
    return loss, {
        "tpr": loss_tpr,
        "fpr": loss_fpr,
        "fppi": loss_fppi,
        "precision": loss_precision,
        "area": loss_area,
    }

def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))

def safe_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x.clamp_min(0.0))

def build_loader(dataset, batch_size, num_workers, shuffle, device):
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 1

    return DataLoader(dataset, **loader_kwargs)

def ensure_f1_target(targets_dict):
    if "f1" in targets_dict:
        return targets_dict

    if "precision" not in targets_dict or "tpr" not in targets_dict:
        raise KeyError("train/val targets must contain either 'f1' or both 'precision' and 'tpr'.")

    precision = targets_dict["precision"]
    recall = targets_dict["tpr"]
    targets_dict["f1"] = (2.0 * precision * recall) / torch.clamp(precision + recall, min=1e-8)
    return targets_dict


def ensure_fppi_norm_target(targets_dict, u_max):
    if "fppi_norm" in targets_dict:
        return targets_dict

    if "fppi" not in targets_dict:
        raise KeyError("train/val targets must contain either 'fppi_norm' or 'fppi'.")

    denom = max(float(u_max), 1e-8)
    targets_dict["fppi_norm"] = torch.clamp(targets_dict["fppi"] / denom, 0.0, 1.0)
    return targets_dict


def build_tau_targets_compat(db_path, class_id, iou_thr, split, u_max):
    kwargs = dict(
        db_path=db_path,
        class_id=class_id,
        iou_thr=iou_thr,
        split=split,
    )

    try:
        sig = inspect.signature(build_tau_operating_targets)
        if "fppi_max" in sig.parameters:
            kwargs["fppi_max"] = u_max
    except (TypeError, ValueError):
        pass

    return build_tau_operating_targets(**kwargs)


def main():
    db_path = _P('databases_dir')
    img_path = _P('coco_mini_person_d', 'data/train2017')

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        description="Train tau-spline monitor head on tau-aligned operating targets."
    )

    parser.add_argument("--db", type=str, default=f"{db_path}/dobi.db", help="Path to SQLite DB file.")
    parser.add_argument(
        "--images_root",
        default=f"{img_path}/images",
        type=str,
        help="Root folder of images.",
    )
    parser.add_argument("--weights", type=str, default='weights/yolov10n.pt', help="YOLO weights (.pt).")
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="subfolders",
        choices=["subfolders", "skip", "overwrite"],
        help="subfolders | skip | overwrite",
    )

    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--u_max", type=float, default=2.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--variants_per_subset_train", type=int, default=5000)
    parser.add_argument("--variants_per_image_train", type=int, default=0)
    parser.add_argument("--variants_per_subset_val", type=int, default=0)
    parser.add_argument("--variants_per_image_val", type=int, default=18)
    parser.add_argument("--smart_sampling", action="store_true")

    parser.add_argument("--w_curve", type=float, default=1.0)
    parser.add_argument("--w_area", type=float, default=0.0, help="Optional tau-area regularizer weight.")
    parser.add_argument("--head_lr", type=float, default=1e-5)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--project", type=str, default=_P('data_root', 'runs'))
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--hidden", type=int, default=1024)

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--use_val", action="store_true")
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument(
        "--head",
        type=str,
        default="tau_spline_single",
        choices=["tau_spline_single", "tau_shared_anchor", "tau_behavior", "pc1_behavior"],
    )

    parser.add_argument(
        "--learn_shared_tau",
        action="store_true",
        help="Only used with --head tau_shared_anchor. If set, predict shared tau anchors per sample.",
    )

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--n_anchors", type=int, default=32)

    # ---- tau_behavior head (joint threshold-indexed representation) ----
    parser.add_argument("--interp", type=str, default="pchip", choices=["linear", "pchip"])
    parser.add_argument("--n_knots", type=int, default=16)
    parser.add_argument("--knot_placement", type=str, default="uniform", choices=["uniform", "quantile"])
    parser.add_argument("--knot_quantile_source", type=str, default="all", choices=["all", "tp"],
                        help="Training scores used for quantile knots: all detections or true positives only.")
    parser.add_argument("--fppi_space", type=str, default="log1p", choices=["log1p", "linear"])
    parser.add_argument("--precision_envelope", action="store_true", help="Ablation: monotone precision-envelope target.")
    parser.add_argument("--n_tau_points", type=int, default=201, help="Training tau grid size for tau_behavior.")
    parser.add_argument("--behavior_loss", type=str, default="smooth_l1", choices=["smooth_l1", "huber", "l1"])
    parser.add_argument("--loss_beta", type=float, default=0.05)
    parser.add_argument("--w_tpr", type=float, default=1.0)
    parser.add_argument("--w_fpr", type=float, default=1.0)
    parser.add_argument("--w_fppi", type=float, default=1.0)
    parser.add_argument("--w_prec", type=float, default=1.0)
    parser.add_argument("--w_roc", type=float, default=0.0, help="Aux: derived ROC loss.")
    parser.add_argument("--w_froc", type=float, default=0.0, help="Aux: derived FROC loss.")
    parser.add_argument("--w_pr", type=float, default=0.0, help="Aux: derived PR loss.")
    parser.add_argument("--w_smooth", type=float, default=0.0, help="Aux: control-value smoothness.")
    parser.add_argument("--w_tau", type=float, default=0.0, help="Aux: soft operating-threshold loss.")
    parser.add_argument("--tau_temperature", type=float, default=0.02, help="Softmax temperature of the soft threshold.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_cache", type=str, default=None,
                        help="Directory of pooled frozen-detector features (dobi/data/feature_cache.py). "
                             "If set, images are not decoded; the head receives the identical pooled features.")
    parser.add_argument("--exclude_regimes", nargs="*", default=[],
                        help="Subset names removed from training AND validation (held-out regime experiments).")

    args = parser.parse_args()
    is_behavior = args.head == "tau_behavior"
    is_pc1 = args.head == "pc1_behavior"
    uses_behavior_targets = is_behavior or is_pc1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("*" * 108)
    print("Train tau-spline monitor")
    print(f"torch version {torch.__version__}")
    print(f"python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print("*" * 108)
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("*" * 108)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # --------------------------------------------------
    # Save directory handling
    # --------------------------------------------------
    project_dir = Path(args.project)

    if args.save_strategy == "subfolders":
        save_dir = next_run_dir(project_dir / args.name / "exp")
    else:
        save_dir = project_dir / args.name

        if args.save_strategy == "skip":
            if save_dir.exists() and any(save_dir.glob("summary_*.json")):
                print(f"Found existing summary file(s) in {save_dir}. save_strategy='skip' -> skipping training.")
                return

        if args.save_strategy == "overwrite" and save_dir.exists():
            for p in save_dir.glob("summary_*.json"):
                p.unlink()

    save_dir.mkdir(parents=True, exist_ok=True)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / "last.pt"
    best = wdir / "best.pt"

    db_file = Path(args.db)
    if not db_file.is_file():
        print(f"ERROR: Database file not found: {db_file}")
        sys.exit(1)

    # --------------------------
    # Tau-space targets - TRAIN
    # --------------------------
    if uses_behavior_targets:
        train_subsets_info, tau_grid_np, train_targets_np = build_tau_behavior_targets(
            db_path=args.db,
            class_id=args.class_id,
            iou_thr=args.iou_thr,
            split=args.split,
            n_tau_points=args.n_tau_points,
            fppi_max=args.u_max,
        )
    else:
        train_subsets_info, tau_grid_np, train_targets_np = build_tau_targets_compat(
            db_path=args.db,
            class_id=args.class_id,
            iou_thr=args.iou_thr,
            split=args.split,
            u_max=args.u_max,
        )

    train_subsets_info, train_targets_np = exclude_subsets(train_subsets_info, train_targets_np, args.exclude_regimes)
    pca = None
    if is_pc1:
        pca = fit_behavior_pca(train_targets_np, tau_grid_np)
        train_targets_np["pc1_norm"] = ((np.asarray(pca["train_coefs"]) - pca["coef_mean"]) / pca["coef_std"]).astype(np.float32)
        print(f"PC1 explained variance (train regimes): {pca['explained_variance_ratio'][0]:.4f}, "
              f"train reconstruction RMSE {pca['train_recon_rmse']:.4f}")

    tau_grid = torch.from_numpy(tau_grid_np).float().to(device)
    train_targets = {k: torch.from_numpy(np.asarray(v)).float().to(device) for k, v in train_targets_np.items()}
    if not uses_behavior_targets:
        train_targets = ensure_fppi_norm_target(train_targets, args.u_max)
        train_targets = ensure_f1_target(train_targets)

    K_train = len(train_subsets_info)
    print(f"Found {K_train} subsets with tau-operating targets for split='{args.split}' (train).")

    # --------------------------
    # Dataset / DataLoader - TRAIN
    # --------------------------
    train_ds = CurveMonitorDataset(
        images_root=args.images_root,
        db_path=args.db,
        subsets_info=train_subsets_info,
        imgsz=args.imgsz,
        channels=3,
        split=args.split,
        class_id=args.class_id,
        variants_per_image=args.variants_per_image_train,
        variants_per_subset=args.variants_per_subset_train,
        smart_sampling=args.smart_sampling,
    )

    if args.feature_cache:
        check_cache_matches(args.feature_cache, args.weights, args.imgsz)
        train_ds = CachedFeatureDataset(train_ds, FeatureCache(args.feature_cache, args.split))
    train_loader = build_loader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=0 if args.feature_cache else args.num_workers,
        shuffle=True,
        device=device,
    )

    # --------------------------
    # Optional VAL targets / loader
    # --------------------------
    val_loader = None
    val_subsets_info = None
    val_targets = None

    if args.use_val:
        if uses_behavior_targets:
            val_subsets_info, tau_grid_val_np, val_targets_np = build_tau_behavior_targets(
                db_path=args.db,
                class_id=args.class_id,
                iou_thr=args.iou_thr,
                split="val",
                n_tau_points=args.n_tau_points,
                fppi_max=args.u_max,
            )
        else:
            val_subsets_info, tau_grid_val_np, val_targets_np = build_tau_targets_compat(
                db_path=args.db,
                class_id=args.class_id,
                iou_thr=args.iou_thr,
                split="val",
                u_max=args.u_max,
            )

        assert np.allclose(tau_grid_val_np, tau_grid_np), "tau_grid mismatch train/val"
        val_subsets_info, val_targets_np = exclude_subsets(val_subsets_info, val_targets_np, args.exclude_regimes)
        if is_pc1:
            val_targets_np["pc1_norm"] = ((project_pc1(pca, val_targets_np) - pca["coef_mean"]) / pca["coef_std"]).astype(np.float32)

        val_targets = {k: torch.from_numpy(np.asarray(v)).float().to(device) for k, v in val_targets_np.items()}
        if not uses_behavior_targets:
            val_targets = ensure_fppi_norm_target(val_targets, args.u_max)
            val_targets = ensure_f1_target(val_targets)

        K_val = len(val_subsets_info)
        print(f"Found {K_val} subsets with tau-operating targets for split='val'.")

        val_ds = CurveMonitorDataset(
            images_root=args.images_root,
            db_path=args.db,
            subsets_info=val_subsets_info,
            imgsz=args.imgsz,
            channels=3,
            split="val",
            class_id=args.class_id,
            variants_per_image=args.variants_per_image_val,
            variants_per_subset=args.variants_per_subset_val,
            smart_sampling=args.smart_sampling,
        )

        if args.feature_cache:
            val_ds = CachedFeatureDataset(val_ds, FeatureCache(args.feature_cache, "val"))
        val_loader = build_loader(
            dataset=val_ds,
            batch_size=args.batch_size,
            num_workers=0 if args.feature_cache else args.num_workers,
            shuffle=False,
            device=device,
        )

    # --------------------------
    # YOLO backbone + monitor head
    # --------------------------
    yolo = None
    if not args.feature_cache:
        yolo = YOLO(args.weights)
        yolo.to(device)
        yolo.model.eval()

    resume_path = None
    if args.resume is not None:
        resume_path = Path(args.resume)
    else:
        candidate = wdir / "last.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")

    if args.feature_cache:
        feat_channels = FeatureCache(args.feature_cache, args.split).feat_channels
        backbone_forward = None
    else:
        sample = next(iter(train_loader))
        imgs_sample = sample["img"].to(device)
        with torch.no_grad():
            feats_sample = forward_backbone(yolo.model, imgs_sample)
        feat_channels = [int(f.shape[1]) for f in feats_sample]

        for p in yolo.model.parameters():
            p.requires_grad_(False)

        backbone_forward = build_backbone_runner(yolo.model)

    if args.head == "tau_spline_single":
        monitor_head = TauSplineSingleHead(
            feat_channels=feat_channels,
            hidden_width=args.hidden,
            n_anchors=args.n_anchors,
            num_hidden_layers=args.mlp_layers,
        ).to(device)

    elif args.head == "tau_shared_anchor":
        monitor_head = TauSplineSharedAnchorHead(
            feat_channels=feat_channels,
            hidden_width=args.hidden,
            n_anchors=args.n_anchors,
            num_hidden_layers=args.mlp_layers,
            learn_shared_tau=args.learn_shared_tau,
        ).to(device)

    elif args.head == "tau_behavior":
        knots_np = build_threshold_knots(
            placement=args.knot_placement,
            n_knots=args.n_knots,
            db_path=args.db,
            class_id=args.class_id,
            iou_thr=args.iou_thr,
            split="train",
            quantile_source=args.knot_quantile_source,
        )
        print(f"Threshold knots ({args.knot_placement}, K={len(knots_np)}): {np.round(knots_np, 4).tolist()}")
        monitor_head = TauBehaviorHead(
            feat_channels=feat_channels,
            knots=knots_np,
            hidden_width=args.hidden,
            num_hidden_layers=args.mlp_layers,
            interp=args.interp,
            fppi_space=args.fppi_space,
            precision_envelope=args.precision_envelope,
        ).to(device)

    elif is_pc1:
        monitor_head = PC1BehaviorHead(
            feat_channels=feat_channels,
            pca=pca,
            hidden_width=args.hidden,
            num_hidden_layers=args.mlp_layers,
        ).to(device)

    else:
        raise ValueError(f"Unsupported head: {args.head}")

    def forward_and_loss(imgs, subset_idx, targets):
        with torch.no_grad():
            if args.feature_cache:
                feats = pooled_to_pseudo_feats(imgs, feat_channels)
            else:
                feats = backbone_forward(imgs)
        head_out = monitor_head(feats)
        if is_pc1:
            loss = F.smooth_l1_loss(head_out["coef_norm"].float(), targets["pc1_norm"][subset_idx], beta=args.loss_beta)
            return loss, {"pc1": loss}
        if is_behavior:
            pred = monitor_head.decode(head_out, tau_grid)
            return compute_tau_behavior_losses(
                pred, head_out, targets, subset_idx, tau_grid, args, fppi_space=args.fppi_space
            )
        pred = decode_tau_spline_head(
            head_out=head_out,
            tau_grid=tau_grid,
            u_max=args.u_max,
            use_shared_anchor_grid=(args.head == "tau_shared_anchor"),
        )
        return compute_legacy_losses(pred, targets, subset_idx, tau_grid, args, device)

    optimizer = torch.optim.AdamW(
        monitor_head.parameters(),
        lr=args.head_lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.head_lr * 0.1,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_loss = float("inf")
    best_epoch = -1
    loss_history = []
    epochs_no_improve = 0
    prev_train_loss = None
    prev_val_loss = None
    start_epoch = 0

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

        if "yolo_state_dict" in ckpt and yolo is not None:
            yolo.model.load_state_dict(ckpt["yolo_state_dict"])
        if "monitor_state_dict" in ckpt:
            monitor_head.load_state_dict(ckpt["monitor_state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        if "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

        if yolo is not None:
            yolo.to(device)
        monitor_head.to(device)

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        loss_history = ckpt.get("loss_history", loss_history)
        epochs_no_improve = int(ckpt.get("epochs_no_improve", epochs_no_improve))

        if loss_history:
            prev_train_loss = loss_history[-1].get("train_loss")
            prev_val_loss = loss_history[-1].get("val_loss")

        print(f"Resuming from epoch {start_epoch}")

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_enabled = (device.type == "cuda")

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(start_epoch, args.epochs):
        if yolo is not None:
            yolo.model.eval()
        monitor_head.train()

        running_loss_train = 0.0
        running_parts_train = {}

        pbar = tqdm(train_loader, total=len(train_loader), mininterval=10)
        for batch in pbar:
            imgs = batch["feat" if args.feature_cache else "img"].to(device, non_blocking=True)
            subset_idx = batch["subset_index"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(autocast_device, enabled=autocast_enabled):
                loss, parts = forward_and_loss(imgs, subset_idx, train_targets)

            if not torch.isfinite(loss):
                print("Non-finite loss, skipping batch")
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(monitor_head.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

            running_loss_train += loss.item()
            for k, v in parts.items():
                running_parts_train[k] = running_parts_train.get(k, 0.0) + float(v.item())

            pbar.set_description(
                f"Epoch {epoch+1}/{args.epochs} "
                f"TrainLoss={loss.item():.6f} ("
                + ", ".join(f"{k}={float(v.item()):.6f}" for k, v in parts.items())
                + ")"
            )

        den_train = max(1, len(train_loader))
        avg_loss_train = running_loss_train / den_train
        avg_parts_train = {k: v / den_train for k, v in running_parts_train.items()}

        if scheduler is not None:
            scheduler.step()

        # ---- VALIDATION ----
        did_validate = False
        if args.use_val and val_loader is not None and (
            (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs
        ):
            did_validate = True
            if yolo is not None:
                yolo.model.eval()
            monitor_head.eval()

            running_loss_val = 0.0
            running_parts_val = {}

            with torch.inference_mode(), torch.amp.autocast(autocast_device, enabled=autocast_enabled):
                for batch in val_loader:
                    imgs = batch["feat" if args.feature_cache else "img"].to(device, non_blocking=True)
                    subset_idx = batch["subset_index"].to(device, non_blocking=True).long()

                    loss, parts = forward_and_loss(imgs, subset_idx, val_targets)

                    running_loss_val += loss.item()
                    for k, v in parts.items():
                        running_parts_val[k] = running_parts_val.get(k, 0.0) + float(v.item())

            den_val = max(1, len(val_loader))
            avg_loss_val = running_loss_val / den_val
            avg_parts_val = {k: v / den_val for k, v in running_parts_val.items()}

            print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss_train:.6f} - val loss: {avg_loss_val:.6f}")
        else:
            avg_loss_val = None
            avg_parts_val = None

            print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss_train:.6f}")

        # ---- Update loss history ----
        train_drop = None if prev_train_loss is None else prev_train_loss - avg_loss_train
        prev_train_loss = avg_loss_train

        if did_validate and avg_loss_val is not None:
            val_drop = None if prev_val_loss is None else prev_val_loss - avg_loss_val
            prev_val_loss = avg_loss_val
        else:
            val_drop = None

        entry = {
            "epoch": int(epoch + 1),
            "train_loss": float(avg_loss_train),
            "train_loss_drop_from_prev": None if train_drop is None else float(train_drop),
        }
        for k, v in avg_parts_train.items():
            entry[f"train_loss_{k}"] = float(v)

        if args.use_val:
            entry["val_loss"] = None if avg_loss_val is None else float(avg_loss_val)
            entry["val_loss_drop_from_prev"] = None if val_drop is None else float(val_drop)

            if avg_loss_val is not None:
                for k, v in avg_parts_val.items():
                    entry[f"val_loss_{k}"] = float(v)

        loss_history.append(entry)

        # ---- Best metric / early stopping ----
        if args.use_val:
            metric_available = did_validate and avg_loss_val is not None
            current_metric = avg_loss_val if metric_available else None
        else:
            metric_available = True
            current_metric = avg_loss_train

        improved = False
        if metric_available:
            improved = current_metric < best_loss - 1e-8
            if improved:
                best_loss = current_metric
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        train_args = vars(args).copy()
        train_args["tau_grid"] = tau_grid_np.tolist()
        train_args["tau_hidden"] = int(args.hidden)
        train_args["tau_utility_type"] = "F1"
        train_args["mlp_layers"] = int(args.mlp_layers)
        if is_behavior or is_pc1:
            train_args.update(monitor_head.config())

        ckpt = {
            "epoch": epoch,
            # with --feature_cache the frozen detector is not loaded; evaluation then uses the raw --weights
            **({"yolo_state_dict": yolo.model.state_dict()} if yolo is not None else {}),
            "monitor_state_dict": monitor_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs_no_improve": epochs_no_improve,
            "loss_history": loss_history,
            "train_args": train_args,
            "subsets_info": train_subsets_info,
            "tau_grid": tau_grid_np,
        }

        torch.save(ckpt, last)
        if improved:
            torch.save(ckpt, best)

        if args.use_val and did_validate and epochs_no_improve >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch+1} "
                f"(no improvement for {args.patience} validation checks)."
            )
            break

    print("Training finished.")
    print(f"Best avg loss: {best_loss:.6f}")
    print(f"Checkpoints: {best}, {last}")

    num_subsets = len(val_subsets_info) if args.use_val and val_subsets_info is not None else len(train_subsets_info)

    summary = {
        "args": vars(args),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "num_subsets": int(num_subsets),
        "save_dir": str(save_dir),
        "best_weights": str(best),
        "last_weights": str(last),
    }

    summary_json_path = save_dir / f"summary_(best_loss_{best_loss:.6f}).json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    loss_history_path = save_dir / "loss_history.json"
    with open(loss_history_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Loss history saved to: {loss_history_path}")


if __name__ == "__main__":
    main()
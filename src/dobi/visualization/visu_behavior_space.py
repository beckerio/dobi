#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Data-space sampling analysis: do the constructed proxy regimes induce
structured variation in the detector's behavior space?

Uses only stored detector predictions / sweeps (no monitor model):

  behavior_curves_<split>.png      empirical F1/TPR/FPPI/Precision vs tau for every proxy regime
  geometric_curves_<split>.png     empirical ROC / FROC / PR for every proxy regime
  behavior_distance_<split>.png    distance of each regime's tau-behavior to pristine
  behavior_embedding.png           PCA of regime-level tau-behavior vectors of all splits
  severity_distance.png            (--per_severity) behavior distance vs. severity 1..5 per degradation,
                                   computed from the raw predictions with the unchanged matching/sweep code
  example_images.png               (--images_root) one image under pristine and every degradation
  behavior_space.json              all numbers behind the figures

Behavior vector of a regime: [TPR(tau), Precision(tau), F1(tau), log1p(FPPI(tau))/log1p(fppi_cap)]
on a uniform tau grid; distances are root-mean-square differences.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dobi.data.database.generate_curves import (
    compute_operating_sweep_from_global_labels,
    greedy_match_labels_one_image_np,
    xywh_to_xyxy,
)
from dobi.utils.curve_plotting import COLORS, FONT_SIZE
from dobi.utils.tau_targets import build_tau_behavior_targets, dedupe_sweep, sweep_at_tau

SPLIT_MARKERS = {"train": "o", "val": "s", "test": "^"}
FAMILIES = {
    "blur": ("defocus_blur", "gaussian_blur", "glass_blur", "motion_blur"),
    "noise": ("gaussian_noise", "impulse_noise", "shot_noise", "speckle_noise"),
    "weather": ("fog", "frost", "snow", "spatter"),
    "digital/photometric": ("brightness", "contrast", "saturate", "jpeg_compression", "pixelate"),
    "pristine": ("pristine",),
}
FAMILY_COLORS = {"pristine": "0.15", "blur": COLORS[2], "noise": COLORS[0], "weather": COLORS[3],
                 "digital/photometric": COLORS[5]}


def family_of(name: str) -> str:
    for fam, members in FAMILIES.items():
        if any(name.startswith(m) for m in members):
            return fam
    return "digital/photometric"


def behavior_vector(t: Dict[str, np.ndarray], i: int, fppi_cap: float) -> np.ndarray:
    fppi = np.log1p(np.minimum(t["fppi"][i], fppi_cap)) / np.log1p(fppi_cap)
    return np.concatenate([t["tpr"][i], t["precision"][i], t["f1"][i], fppi])


def rms(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def short(name: str) -> str:
    return name.replace("_3_5", "").replace("_", " ")


# ------------------------------------------------------------------
def plot_split_curves(info, tau, t, split, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    for i, s in enumerate(info):
        fam = family_of(s["name"])
        kw = dict(color=FAMILY_COLORS[fam], lw=2.5 if fam == "pristine" else 1.2, alpha=1.0 if fam == "pristine" else 0.75)
        axes[0].plot(tau, t["f1"][i], **kw)
        axes[1].plot(tau, t["tpr"][i], **kw)
        axes[2].plot(tau, t["fppi"][i], **kw)
        axes[3].plot(tau, t["precision"][i], **kw)
    for ax, lab in zip(axes, ("F1", "TPR", "FPPI", "Precision")):
        ax.set_xlabel("detector threshold τ")
        ax.set_title(lab, fontsize=FONT_SIZE)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25, lw=0.5)
    axes[2].set_yscale("symlog", linthresh=0.1)
    for fam, col in FAMILY_COLORS.items():
        axes[0].plot([], [], color=col, lw=2, label=fam)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Empirical threshold-indexed behavior of all proxy regimes ({split})", fontsize=FONT_SIZE)
    fig.savefig(out / f"behavior_curves_{split}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for i, s in enumerate(info):
        fam = family_of(s["name"])
        kw = dict(color=FAMILY_COLORS[fam], lw=2.5 if fam == "pristine" else 1.2, alpha=1.0 if fam == "pristine" else 0.75)
        axes[0].plot(t["roc_fpr_grid"], t["roc_tpr"][i], **kw)
        axes[1].plot(t["froc_fppi_grid"], t["froc_tpr"][i], **kw)
        axes[2].plot(t["pr_recall_grid"], t["pr_precision"][i], **kw)
    for ax, (xl, yl, ti) in zip(axes, (("FPR", "TPR", "ROC"), ("FPPI", "TPR", "FROC"), ("Recall", "Precision", "PR"))):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ti, fontsize=FONT_SIZE)
        ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(out / f"geometric_curves_{split}.png", dpi=180)
    plt.close(fig)


def plot_distance(info, dists, split, out: Path):
    order = np.argsort(dists)
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(order) + 1.2), constrained_layout=True)
    y = np.arange(len(order))
    ax.barh(y, [dists[i] for i in order], color=[FAMILY_COLORS[family_of(info[i]["name"])] for i in order], height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([short(info[i]["name"]) for i in order])
    ax.set_xlabel("RMS distance to pristine behavior")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for fam, col in FAMILY_COLORS.items():
        ax.barh([], [], color=col, label=fam)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / f"behavior_distance_{split}.png", dpi=180)
    plt.close(fig)


def fit_pca_train_only(points: List[Dict]):
    """PCA fitted on TRAINING-split regime vectors only; other splits are only transformed."""
    X = np.stack([p["vec"] for p in points if p["split"] == "train"])
    mean = X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = (S ** 2) / np.sum(S ** 2)
    return {"mean": mean[0], "components": Vt[:2], "explained_variance_ratio": var}


def pca_transform(pca, vec):
    return (np.atleast_2d(vec) - pca["mean"]) @ pca["components"].T


def plot_embedding(points: List[Dict], out: Path, pca, severity=None):
    var = pca["explained_variance_ratio"]
    ncol = 2 if severity else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.5 * ncol, 5.2), constrained_layout=True, squeeze=False)
    ax = axes[0, 0]
    for p in points:
        z = pca_transform(pca, p["vec"])[0]
        p["pc"] = z
        ax.scatter(z[0], z[1], marker=SPLIT_MARKERS.get(p["split"], "o"), s=70,
                   color=FAMILY_COLORS[family_of(p["name"])], edgecolor="white", linewidth=1.2)
        if p["split"] == "test":
            ax.annotate(short(p["name"]), z, fontsize=7, xytext=(4, 3), textcoords="offset points")
    for fam, col in FAMILY_COLORS.items():
        ax.scatter([], [], color=col, label=fam)
    for sp, m in SPLIT_MARKERS.items():
        ax.scatter([], [], marker=m, color="0.5", label=sp)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for a in axes[0]:
        a.set_xlabel(f"PC1 ({100 * var[0]:.1f}% of train variance)")
        a.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)")
        a.grid(alpha=0.25, lw=0.5)
    ax.set_title("Proxy regimes (severity 3-5), PCA fitted on train regimes", fontsize=FONT_SIZE)
    if severity:
        ax2 = axes[0, 1]
        base = pca_transform(pca, severity["pristine_vec"])[0]
        ax2.scatter(base[0], base[1], color=FAMILY_COLORS["pristine"], s=90, zorder=3, label="pristine")
        for d, r in severity["degradations"].items():
            Z = np.vstack([base] + [pca_transform(pca, v)[0] for v in r["vecs"]])
            col = FAMILY_COLORS[family_of(d)]
            ax2.plot(Z[:, 0], Z[:, 1], "-o", color=col, ms=3.5, lw=1.2, alpha=0.85)
            ax2.annotate(d.replace("_", " "), Z[-1], fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax2.set_title("Severity trajectories 0 -> 1..5 (test split, projected)", fontsize=FONT_SIZE)
    fig.savefig(out / "behavior_space_pca.png", dpi=180)
    fig.savefig(out / "behavior_space_pca.pdf")
    plt.close(fig)


# ------------------------------------------------------------------
def sweep_for_variants(conn, split: str, deg_name: str, severity: int, class_id: int, iou_thr: float):
    """Operating sweep of all variants (deg_name, severity) of a split, with the unchanged generate_curves code."""
    rows = conn.execute(
        """
        SELECT v.id, v.image_id FROM variants v JOIN images i ON i.id = v.image_id
        WHERE i.split = ? AND v.deg_name = ? AND v.severity = ?
        """,
        (split, deg_name, int(severity)),
    ).fetchall()
    if not rows:
        return None
    vid_to_img = {int(v): int(i) for v, i in rows}

    preds: Dict[int, list] = {}
    for vid, x, y, w, h, s in conn.execute(
        """
        SELECT p.variant_id, p.x, p.y, p.w, p.h, p.score FROM predictions p
        JOIN variants v ON v.id = p.variant_id JOIN images i ON i.id = v.image_id
        WHERE i.split = ? AND v.deg_name = ? AND v.severity = ? AND p.category_id = ?
        """,
        (split, deg_name, int(severity), int(class_id)),
    ):
        preds.setdefault(int(vid), []).append((x, y, w, h, s))

    gts: Dict[int, list] = {}
    for img, x, y, w, h in conn.execute(
        """
        SELECT g.image_id, g.x, g.y, g.w, g.h FROM gt_objects g JOIN images i ON i.id = g.image_id
        WHERE i.split = ? AND g.class_id = ?
        """,
        (split, int(class_id)),
    ):
        gts.setdefault(int(img), []).append((x, y, w, h))

    scores, is_tp, num_gt = [], [], 0
    for vid, img in vid_to_img.items():
        p = np.asarray(preds.get(vid, []), dtype=np.float32).reshape(-1, 5)
        g = np.asarray(gts.get(img, []), dtype=np.float32).reshape(-1, 4)
        tp, n_gt = greedy_match_labels_one_image_np(xywh_to_xyxy(p[:, :4]), p[:, 4], xywh_to_xyxy(g), iou_thr)
        num_gt += int(n_gt)
        scores.append(p[:, 4])
        is_tp.append(tp)
    sweep = compute_operating_sweep_from_global_labels(
        np.concatenate(scores).astype(np.float64), np.concatenate(is_tp).astype(bool), num_gt, len(vid_to_img))
    return sweep


def severity_analysis(db: str, split: str, class_id: int, iou_thr: float, tau: np.ndarray, fppi_cap: float, out: Path):
    conn = sqlite3.connect(db)
    degs = [r[0] for r in conn.execute(
        "SELECT DISTINCT deg_name FROM variants WHERE deg_name != 'pristine' ORDER BY deg_name").fetchall()]
    sevs = [r[0] for r in conn.execute(
        "SELECT DISTINCT severity FROM variants WHERE severity > 0 ORDER BY severity").fetchall()]

    def vec(sweep):
        s = sweep_at_tau(dedupe_sweep({k: np.asarray(v, dtype=np.float64) for k, v in sweep.items()}), tau)
        t = {k: s[k][None] for k in ("tpr", "precision", "f1", "fppi")}
        return behavior_vector(t, 0, fppi_cap), float(np.max(s["f1"]))

    base_vec, base_f1 = vec(sweep_for_variants(conn, split, "pristine", 0, class_id, iou_thr))
    result = {"pristine_best_f1": base_f1, "pristine_vec": base_vec, "degradations": {}}
    for d in degs:
        dist, f1, vecs = [], [], []
        for s in sevs:
            sw = sweep_for_variants(conn, split, d, s, class_id, iou_thr)
            if sw is None:
                dist.append(np.nan)
                f1.append(np.nan)
                continue
            v, bf = vec(sw)
            dist.append(rms(v, base_vec))
            f1.append(bf)
            vecs.append(v)
        result["degradations"][d] = {"severity": sevs, "distance": dist, "best_f1": f1, "vecs": vecs}
        print(f"  {d}: distance {np.round(dist, 3).tolist()}")
    conn.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for d, r in result["degradations"].items():
        col = FAMILY_COLORS[family_of(d)]
        axes[0].plot(r["severity"], r["distance"], "-o", color=col, lw=1.4, ms=4, alpha=0.85)
        axes[1].plot(r["severity"], r["best_f1"], "-o", color=col, lw=1.4, ms=4, alpha=0.85)
        axes[0].annotate(d.replace("_", " "), (r["severity"][-1], r["distance"][-1]), fontsize=7,
                         xytext=(4, 0), textcoords="offset points")
    axes[1].axhline(base_f1, color="0.15", ls="--", lw=1.2, label="pristine")
    axes[0].set_ylabel("RMS behavior distance to pristine")
    axes[1].set_ylabel("best F1 of the regime")
    for ax in axes:
        ax.set_xlabel("severity")
        ax.set_xticks(sevs)
        ax.grid(alpha=0.25, lw=0.5)
    for fam, col in FAMILY_COLORS.items():
        if fam != "pristine":
            axes[1].plot([], [], color=col, label=fam)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "distance_to_pristine_by_severity.png", dpi=180)
    fig.savefig(out / "distance_to_pristine_by_severity.pdf")
    plt.close(fig)
    return result


def example_images(db: str, images_root: Path, split: str, severity: int, out: Path, image_id: int = None):
    conn = sqlite3.connect(db)
    if image_id is None:
        image_id = conn.execute(
            """SELECT g.image_id FROM gt_objects g JOIN images i ON i.id = g.image_id
               WHERE i.split = ? GROUP BY g.image_id HAVING COUNT(*) BETWEEN 2 AND 4 ORDER BY g.image_id LIMIT 1""",
            (split,)).fetchone()[0]
    rows = conn.execute(
        """SELECT deg_name, rel_path FROM variants WHERE image_id = ?
           AND (deg_name = 'pristine' OR severity = ?) ORDER BY deg_name = 'pristine' DESC, deg_name""",
        (int(image_id), int(severity))).fetchall()
    conn.close()
    n = len(rows)
    cols = 6
    nr = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(2.6 * cols, 2.2 * nr), constrained_layout=True, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (deg, rel) in zip(axes.ravel(), rows):
        p = images_root / rel
        if not p.exists():
            p = images_root.parent.parent / f"{split}2017" / "images" / rel
        im = cv2.imread(str(p))
        if im is not None:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(deg.replace("_", " "), fontsize=8)
    fig.savefig(out / "example_images.png", dpi=160)
    plt.close(fig)


def severity_images(db: str, images_root: Path, split: str, degradations, out: Path, image_id: int = None):
    """Severity progression 0..5 of one test image for a few degradations (Figure 1)."""
    conn = sqlite3.connect(db)
    if image_id is None:
        image_id = conn.execute(
            """SELECT g.image_id FROM gt_objects g JOIN images i ON i.id = g.image_id
               WHERE i.split = ? GROUP BY g.image_id HAVING COUNT(*) BETWEEN 2 AND 4 ORDER BY g.image_id LIMIT 1""",
            (split,)).fetchone()[0]
    pristine = conn.execute("SELECT rel_path FROM variants WHERE image_id = ? AND deg_name = 'pristine'",
                            (int(image_id),)).fetchone()[0]
    rows = {d: [pristine] + [conn.execute(
        "SELECT rel_path FROM variants WHERE image_id = ? AND deg_name = ? AND severity = ?",
        (int(image_id), d, s)).fetchone()[0] for s in range(1, 6)] for d in degradations}
    conn.close()
    fig, axes = plt.subplots(len(degradations), 6, figsize=(13, 2.0 * len(degradations)), constrained_layout=True,
                             squeeze=False)
    for r, d in enumerate(degradations):
        for c, rel in enumerate(rows[d]):
            ax = axes[r, c]
            ax.axis("off")
            im = cv2.imread(str(images_root / rel))
            if im is not None:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if r == 0:
                ax.set_title("pristine" if c == 0 else f"severity {c}", fontsize=9)
        axes[r, 0].text(-0.05, 0.5, d.replace("_", " "), transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9)
    fig.savefig(out / "severity_progression_images.png", dpi=160)
    fig.savefig(out / "severity_progression_images.pdf")
    plt.close(fig)
    return {"image_id": int(image_id)}


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Proxy regimes in detector-behavior space.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--class_id", type=int, default=0)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--n_tau_points", type=int, default=201)
    ap.add_argument("--fppi_cap", type=float, default=10.0, help="FPPI cap for the behavior vector (log1p-scaled).")
    ap.add_argument("--u_max", type=float, default=2.0, help="FROC range for the geometric plot.")
    ap.add_argument("--per_severity", action="store_true", help="Also sweep each (degradation, severity) of --severity_split.")
    ap.add_argument("--severity_split", default="test")
    ap.add_argument("--images_root", type=str, default=None, help="If set, render example images (split=test).")
    ap.add_argument("--example_severity", type=int, default=5)
    ap.add_argument("--severity_degradations", nargs="*", default=["gaussian_noise", "defocus_blur", "fog", "contrast"])
    ap.add_argument("--output_dir", default="./behavior_space")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    points, summary = [], {"splits": {}}
    tau = None
    for split in args.splits:
        info, tau, t = build_tau_behavior_targets(args.db, args.class_id, args.iou_thr, split=split,
                                                  n_tau_points=args.n_tau_points, fppi_max=args.u_max)
        pidx = next(i for i, s in enumerate(info) if "pristine" in s["name"].lower())
        vecs = [behavior_vector(t, i, args.fppi_cap) for i in range(len(info))]
        dists = [rms(v, vecs[pidx]) for v in vecs]
        plot_split_curves(info, tau, t, split, out)
        plot_distance(info, dists, split, out)
        summary["splits"][split] = [
            {"name": s["name"], "subset_id": s["subset_id"], "distance_to_pristine": d,
             "best_f1": float(t["f1_opt"][i]), "tau_opt": float(t["tau_opt"][i])}
            for i, (s, d) in enumerate(zip(info, dists))
        ]
        points += [{"name": s["name"], "split": split, "vec": v} for s, v in zip(info, vecs)]

    severity = None
    if args.per_severity:
        print("Per-severity sweeps from raw predictions ...")
        severity = severity_analysis(args.db, args.severity_split, args.class_id, args.iou_thr, tau, args.fppi_cap, out)

    pca = fit_pca_train_only(points)
    plot_embedding(points, out, pca, severity)
    summary["pca"] = {
        "fitted_on": "train regimes only (val/test transformed)",
        "n_train_regimes": int(sum(p["split"] == "train" for p in points)),
        "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
    }

    # per-regime table (distance to own-split pristine, PCA coordinates)
    dist_by = {(sp, e["name"]): e for sp, lst in summary["splits"].items() for e in lst}
    with open(out / "behavior_space_per_regime.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "regime", "family", "distance_to_pristine", "best_f1", "tau_opt", "pc1", "pc2"])
        for p in points:
            e = dist_by[(p["split"], p["name"])]
            w.writerow([p["split"], p["name"], family_of(p["name"]), e["distance_to_pristine"], e["best_f1"],
                        e["tau_opt"], float(p["pc"][0]), float(p["pc"][1])])

    # consistency of the same regime across splits vs distances between different regimes
    by_name: Dict[str, Dict[str, np.ndarray]] = {}
    for p in points:
        by_name.setdefault(p["name"], {})[p["split"]] = p["vec"]
    with open(out / "cross_split_regime_stability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "split_pair", "rms_distance"])
        same = []
        for n, v in by_name.items():
            for a_ in v:
                for b_ in v:
                    if a_ < b_:
                        d = rms(v[a_], v[b_])
                        same.append(d)
                        w.writerow([n, f"{a_}-{b_}", d])
    test_vecs = {p["name"]: p["vec"] for p in points if p["split"] == "test"}
    names = sorted(test_vecs)
    between = [rms(test_vecs[a_], test_vecs[b_]) for i, a_ in enumerate(names) for b_ in names[i + 1:]]
    summary["cross_split_stability"] = {
        "same_regime_across_splits_rms": {"mean": float(np.mean(same)), "max": float(np.max(same))},
        "between_regimes_test_rms": {"mean": float(np.mean(between)), "min": float(np.min(between)),
                                     "max": float(np.max(between))},
    }
    if severity:
        summary["severity"] = {
            "split": args.severity_split,
            "pristine_best_f1": severity["pristine_best_f1"],
            "degradations": {d: {k: r[k] for k in ("severity", "distance", "best_f1")}
                             for d, r in severity["degradations"].items()},
        }
        mono = {d: bool(np.all(np.diff(r["distance"]) >= -1e-9)) for d, r in severity["degradations"].items()}
        summary["severity"]["distance_monotone_in_severity"] = mono

    if args.images_root:
        example_images(args.db, Path(args.images_root), "test", args.example_severity, out)
        severity_images(args.db, Path(args.images_root), "test", args.severity_degradations, out)

    with open(out / "behavior_space_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved behavior-space analysis to {out}")


if __name__ == "__main__":
    main()

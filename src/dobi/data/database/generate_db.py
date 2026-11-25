#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
import argparse
import glob
import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple

import cv2

# Supported image formats

IMG_FORMATS = {
    "jpeg",
    "jpg",
    "png",
}


# ---------- DB schema (variant-aware, with split on images) ----------

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def create_schema(db_path: str):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_db(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        -- Base images (logical scene)
        CREATE TABLE IF NOT EXISTS images (
            id         INTEGER PRIMARY KEY,
            base_name  TEXT NOT NULL,          -- e.g. '000000000123'
            width      INTEGER,
            height     INTEGER,
            split      TEXT NOT NULL,          -- 'train', 'val', 'test', ...
            UNIQUE (base_name, split)
        );

        -- Concrete variant on disk (pristine or degraded)
        CREATE TABLE IF NOT EXISTS variants (
            id          INTEGER PRIMARY KEY,
            image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            deg_name    TEXT NOT NULL,         -- 'pristine', 'gaussian_blur', ...
            severity    INTEGER NOT NULL,      -- 0 for pristine, 1..5 etc.
            rel_path    TEXT NOT NULL,         -- e.g. 'pristine/000000000123.jpg'
            UNIQUE (image_id, deg_name, severity)
        );

        -- Ground-truth objects (shared by all variants of one image)
        CREATE TABLE IF NOT EXISTS gt_objects (
            id        INTEGER PRIMARY KEY,
            image_id  INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            class_id  INTEGER NOT NULL,
            x         REAL NOT NULL,
            y         REAL NOT NULL,
            w         REAL NOT NULL,
            h         REAL NOT NULL
        );

        -- Predictions per variant
        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY,
            variant_id        INTEGER NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
            category_id       INTEGER NOT NULL,
            x                 REAL NOT NULL,
            y                 REAL NOT NULL,
            w                 REAL NOT NULL,
            h                 REAL NOT NULL,
            score             REAL NOT NULL,
            max_iou           REAL,
            matched_gt_id     INTEGER
        );

        -- Subset definitions
        CREATE TABLE IF NOT EXISTS subsets (
            id          INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            split       TEXT NOT NULL,
            params_json TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        -- Subset membership (variants)
        CREATE TABLE IF NOT EXISTS subset_variants (
            subset_id   INTEGER NOT NULL REFERENCES subsets(id) ON DELETE CASCADE,
            variant_id  INTEGER NOT NULL REFERENCES variants(id) ON DELETE CASCADE,
            PRIMARY KEY (subset_id, variant_id)
        );

        -- Summary metrics per subset / class / IoU
        CREATE TABLE IF NOT EXISTS subset_eval_metrics (
            subset_id          INTEGER NOT NULL REFERENCES subsets(id) ON DELETE CASCADE,
            class_id           INTEGER NOT NULL,
            iou_thr            REAL NOT NULL,
            froc_fppi_max      REAL NOT NULL,

            num_gt             INTEGER NOT NULL,
            num_imgs           INTEGER NOT NULL,

            roc_auc            REAL,
            roc_pauc_fpr_0p1   REAL,
            froc_auc           REAL,
            auprc              REAL,

            recall_fppi_0p1    REAL,
            recall_fppi_0p5    REAL,
            recall_fppi_1p0    REAL,

            best_f1            REAL,
            best_f1_thr        REAL,
            best_f1_precision  REAL,
            best_f1_recall     REAL,

            PRIMARY KEY (subset_id, class_id, iou_thr, froc_fppi_max)
        );

        -- Full threshold sweep (tau-indexed operating behavior)
        CREATE TABLE IF NOT EXISTS subset_operating_points (
            subset_id   INTEGER NOT NULL REFERENCES subsets(id) ON DELETE CASCADE,
            class_id    INTEGER NOT NULL,
            iou_thr     REAL NOT NULL,
            idx         INTEGER NOT NULL,   -- order along descending threshold sweep
            thr         REAL NOT NULL,      -- detector threshold tau

            tp          INTEGER NOT NULL,
            fp          INTEGER NOT NULL,
            fn          INTEGER NOT NULL,
            num_pred    INTEGER NOT NULL,

            recall      REAL NOT NULL,      -- same as TPR here
            tpr         REAL NOT NULL,
            fpr         REAL NOT NULL,      -- normalized detection FPR from your thesis
            fppi        REAL NOT NULL,
            precision   REAL NOT NULL,

            PRIMARY KEY (subset_id, class_id, iou_thr, idx)
        );

        CREATE INDEX IF NOT EXISTS idx_gt_image
        ON gt_objects(image_id);

        CREATE INDEX IF NOT EXISTS idx_var_image
        ON variants(image_id);

        CREATE INDEX IF NOT EXISTS idx_pred_variant
        ON predictions(variant_id);

        CREATE INDEX IF NOT EXISTS idx_subset_variants_subset
        ON subset_variants(subset_id);

        CREATE INDEX IF NOT EXISTS idx_subset_operating_points_lookup
        ON subset_operating_points(subset_id, class_id, iou_thr, thr);

        CREATE INDEX IF NOT EXISTS idx_subset_operating_points_idx
        ON subset_operating_points(subset_id, class_id, iou_thr, idx);
        """
    )

    conn.commit()
    conn.close()


# ---------- helpers for image + label handling ----------

def get_image_files_recursive(root: str) -> List[str]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image root not found: {root}")
    files = []
    for f in glob.glob(str(root / "**" / "*.*"), recursive=True):
        if f.rsplit(".", 1)[-1].lower() in IMG_FORMATS:
            files.append(os.path.abspath(f))
    files.sort()
    return files


def get_prediction_files_recursive(root: str) -> List[str]:
    """
    Recursively collect all .txt prediction files under root.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Predictions root not found: {root}")
    files = []
    for f in glob.glob(str(root / "**" / "*.txt"), recursive=True):
        files.append(os.path.abspath(f))
    files.sort()
    return files


def get_image_size(path: str) -> Tuple[int, int]:
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = im.shape[:2]
    return w, h


def load_yolo_labels_for_image(label_path: Path, img_w: int, img_h: int):
    """
    Load YOLO labels for one base image and return a list of dicts:
      {class_id, bbox=[x,y,w,h]} in pixels.
    """
    if not label_path.is_file():
        return []

    gt_objects = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])

            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h

            x_min = x_c - w / 2.0
            y_min = y_c - h / 2.0
            gt_objects.append(
                {
                    "class_id": cls_id,
                    "bbox": [x_min, y_min, w, h],
                }
            )
    return gt_objects


def load_yolo_predictions_for_variant(pred_path: Path, img_w: int, img_h: int):
    """
    Load YOLO-format predictions for one variant and return a list of dicts:
      {class_id, bbox=[x,y,w,h], score} in pixels.

    Format per line:
      class_id x_center y_center width height score
    with all coordinates normalized to [0,1].
    """
    if not pred_path.is_file():
        return []

    preds = []
    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            cls_id = int(parts[0])
            x_c, y_c, w, h, score = map(float, parts[1:6])

            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h

            x_min = x_c - w / 2.0
            y_min = y_c - h / 2.0

            preds.append(
                {
                    "class_id": cls_id,
                    "bbox": [x_min, y_min, w, h],
                    "score": score,
                }
            )
    return preds


def parse_variant_from_path(rel_path: Path):
    """
    Infer:
      - deg_name
      - severity
      - base_name (stem of file)

    from a path relative to images_root or predictions_root.
    """
    parts = rel_path.parts
    fname = rel_path.name
    base_name = Path(fname).stem

    if len(parts) >= 2 and parts[0] == "pristine":
        deg_name = "pristine"
        severity = 0
        return deg_name, severity, base_name

    if len(parts) >= 2 and parts[1].startswith("deg_"):
        deg_name = parts[0]
        try:
            severity = int(parts[1].split("deg_")[1])
        except ValueError:
            severity = 0
        return deg_name, severity, base_name

    # Fallback
    deg_name = parts[0]
    severity = 0
    return deg_name, severity, base_name


# ---------- main ingest function (images + GT) ----------


def ingest_images_and_labels(
    db_path: str,
    images_root: str,
    labels_root: str,
    split: str,
):
    """
    Fill DB with:
      - images (base images, tagged with split)
      - variants (all deg/severity variants)
      - gt_objects (YOLO GT per base image)

    """
    db_path = Path(db_path)
    images_root = Path(images_root)
    labels_root = Path(labels_root)

    create_schema(db_path)

    conn = connect_db(db_path)
    cur = conn.cursor()

    # Cache to avoid inserting GT multiple times for same base image
    seen_base_names = set()
    base_name_to_image_id: Dict[str, int] = {}

    all_files = get_image_files_recursive(str(images_root))
    print(f"[{split}] Found {len(all_files)} image files under {images_root}")

    for full_path in all_files:
        full_path = Path(full_path)
        rel_path = full_path.relative_to(images_root)  # e.g. 'pristine/000.jpg'

        deg_name, severity, base_name = parse_variant_from_path(rel_path)

        # 1) ensure base image row exists for this split
        if base_name not in base_name_to_image_id:
            img_w, img_h = get_image_size(str(full_path))

            cur.execute(
                """
                INSERT OR IGNORE INTO images (base_name, width, height, split)
                VALUES (?, ?, ?, ?)
                """,
                (base_name, img_w, img_h, split),
            )

            cur.execute(
                "SELECT id FROM images WHERE base_name = ? AND split = ?",
                (base_name, split),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    f"Failed to fetch image row for base_name={base_name}, split={split}"
                )
            image_id = row[0]
            base_name_to_image_id[base_name] = image_id
        else:
            image_id = base_name_to_image_id[base_name]

        # 2) insert variant row
        rel_path_str = rel_path.as_posix()
        cur.execute(
            """
            INSERT OR IGNORE INTO variants (image_id, deg_name, severity, rel_path)
            VALUES (?, ?, ?, ?)
            """,
            (image_id, deg_name, severity, rel_path_str),
        )

        # 3) insert GT objects once per base image (per split)
        if base_name not in seen_base_names:
            seen_base_names.add(base_name)

            cur.execute(
                "SELECT width, height FROM images WHERE id = ?",
                (image_id,),
            )
            img_w, img_h = cur.fetchone()

            # Make GT ingestion idempotent on reruns
            cur.execute("DELETE FROM gt_objects WHERE image_id = ?", (image_id,))

            label_path = labels_root / f"{base_name}.txt"
            gt_objects = load_yolo_labels_for_image(
                label_path, img_w=img_w, img_h=img_h
            )

            for g in gt_objects:
                x, y, w, h = g["bbox"]
                cur.execute(
                    """
                    INSERT INTO gt_objects (image_id, class_id, x, y, w, h)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (image_id, g["class_id"], x, y, w, h),
                )

    conn.commit()
    conn.close()
    print(f"[{split}] Ingest (images + GT) finished. DB saved to {db_path}")


# ---------- ingest predictions (YOLO-format, per variant) ----------


def ingest_predictions(
    db_path: str,
    predictions_root: str,
    split: str,
):
    """
    Fill DB with predictions from YOLO-format .txt files stored per variant.

    predictions_root mirrors the image structure for that split.
    """
    db_path = Path(db_path)
    predictions_root = Path(predictions_root)

    if not predictions_root.is_dir():
        print(f"[{split}] Predictions root {predictions_root} does not exist; skipping.")
        return

    conn = connect_db(db_path)
    cur = conn.cursor()

    all_pred_files = get_prediction_files_recursive(str(predictions_root))
    print(f"[{split}] Found {len(all_pred_files)} prediction files under {predictions_root}")

    # cache: (base_name, deg_name, severity) -> (variant_id, img_w, img_h)
    variant_cache: Dict[Tuple[str, str, int], Tuple[int, int, int]] = {}

    for full_path in all_pred_files:
        full_path = Path(full_path)
        rel_path = full_path.relative_to(predictions_root)

        deg_name, severity, base_name = parse_variant_from_path(rel_path)

        key = (base_name, deg_name, severity)
        if key not in variant_cache:
            # Find variant_id and image size via DB, enforcing split
            cur.execute(
                """
                SELECT v.id, i.width, i.height
                FROM variants v
                JOIN images i ON v.image_id = i.id
                WHERE i.base_name = ?
                  AND i.split     = ?
                  AND v.deg_name  = ?
                  AND v.severity  = ?
                """,
                (base_name, split, deg_name, severity),
            )
            row = cur.fetchone()
            if row is None:
                print(
                    f"[{split}] Warning: no variant found for prediction file {rel_path} "
                    f"(base_name={base_name}, deg={deg_name}, sev={severity})"
                )
                continue
            variant_id, img_w, img_h = row
            variant_cache[key] = (variant_id, img_w, img_h)
        else:
            variant_id, img_w, img_h = variant_cache[key]

        preds = load_yolo_predictions_for_variant(
            full_path, img_w=img_w, img_h=img_h
        )

        # Avoid duplicate predictions for same variant if re-running:
        cur.execute("DELETE FROM predictions WHERE variant_id = ?", (variant_id,))

        for p in preds:
            x, y, w, h = p["bbox"]
            cur.execute(
                """
                INSERT INTO predictions (
                    variant_id, category_id, x, y, w, h, score, max_iou, matched_gt_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (variant_id, p["class_id"], x, y, w, h, p["score"]),
            )

    conn.commit()
    conn.close()
    print(f"[{split}] Ingest (predictions) finished. DB updated at {db_path}")


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ingest train/val/test images + YOLO labels + YOLO predictions into a "
            "single SQLite DB with variants (deg/sev) and a split column on images."
        )
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite DB file.",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of dataset.",
    )

    # Train split
    parser.add_argument(
        "--train_root",
        type=str,
        default="/train2017",
        help="Train: root directory containing images, labels and predictions.",
    )

    # Val split
    parser.add_argument(
        "--val_root",
        type=str,
        default="/val2017",
        help="Val: root directory containing images, labels and predictions.",
    )

    # Test split
    parser.add_argument(
        "--test_root",
        type=str,
        default="/test2017",
        help="Test: root directory containing images, labels and predictions.",
    )

    args = parser.parse_args()

    db_path = args.db
    create_schema(db_path)  # ensure schema exists once

    splits = ["train", "val", "test"]
    active_splits = [s for s in splits if bool(getattr(args, f"{s}_root", None))]

    if not active_splits:
        print("No splits provided (need at least images_root and labels_root for a split). Nothing to do.")
        raise SystemExit(0)

    print("Adding splits:", ", ".join(active_splits))

    dataset_root = Path(args.dataset_root)

    for split in active_splits:
        print(f"\n=== Adding {split} split ===")

        split_root = getattr(args, f"{split}_root").lstrip("/")

        images_root = dataset_root / split_root / "images"
        labels_root = dataset_root / split_root / "labels"
        preds_root = dataset_root / split_root / "predictions"

        ingest_images_and_labels(
            db_path=db_path,
            images_root=str(images_root),
            labels_root=str(labels_root),
            split=split,
        )

        ingest_predictions(
            db_path=db_path,
            predictions_root=str(preds_root),
            split=split,
        )
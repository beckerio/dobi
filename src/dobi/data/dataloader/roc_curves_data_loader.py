# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
# SPDX-FileCopyrightText: Ultralytics Inc.
import math
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dobi.utils.img_utils import letterbox_tensor

# (rel_path, image_id, variant_id, deg_name, severity)
RowWithDeg = Tuple[str, int, int, str, int]
RowWithSplit = Tuple[str, str]


def _get_table_names(cur: sqlite3.Cursor) -> set[str]:
    rows = cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view')
        """
    ).fetchall()
    return {str(r[0]) for r in rows}


def _get_table_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(r[1]) for r in rows}


def _resolve_curve_table_names(cur: sqlite3.Cursor) -> Tuple[str, str]:
    """
    Supports both schemas:

    Old:
      - subset_roc_froc_metrics
      - subset_roc_froc_points

    Newer:
      - subset_eval_metrics
      - subset_operating_points
    """
    tables = _get_table_names(cur)

    if "subset_roc_froc_metrics" in tables:
        metrics_table = "subset_roc_froc_metrics"
    elif "subset_eval_metrics" in tables:
        metrics_table = "subset_eval_metrics"
    else:
        raise RuntimeError(
            "Could not find subset metrics table. Expected one of: "
            "'subset_roc_froc_metrics', 'subset_eval_metrics'."
        )

    if "subset_roc_froc_points" in tables:
        points_table = "subset_roc_froc_points"
    elif "subset_operating_points" in tables:
        points_table = "subset_operating_points"
    else:
        raise RuntimeError(
            "Could not find subset points table. Expected one of: "
            "'subset_roc_froc_points', 'subset_operating_points'."
        )

    return metrics_table, points_table


def _split_subdir_candidates(split: Optional[str]) -> List[Path]:
    if split is None:
        return [Path("")]

    split_l = str(split).lower()
    mapping = {
        "train": [Path("train2017/images"), Path("train/images"), Path("train")],
        "val": [Path("val2017/images"), Path("val/images"), Path("val")],
        "valid": [Path("val2017/images"), Path("val/images"), Path("val")],
        "validation": [Path("val2017/images"), Path("val/images"), Path("val")],
        "test": [Path("test2017/images"), Path("test/images"), Path("test")],
    }
    return mapping.get(split_l, [Path(split_l)])


def _resolve_image_path(images_root: Path, rel_path: str, split: Optional[str]) -> Path:
    """
    Supports both styles:

    1) images_root = /.../data
       -> /.../data/train2017/images/<rel_path>
       -> /.../data/val2017/images/<rel_path>

    2) images_root = /.../data/train2017/images
       -> train stays there
       -> val resolves to sibling /.../data/val2017/images/<rel_path>
    """
    rel = Path(rel_path)
    if rel.is_absolute():
        return rel

    candidates: List[Path] = [images_root / rel]

    for subdir in _split_subdir_candidates(split):
        candidates.append(images_root / subdir / rel)

    if images_root.name == "images" and images_root.parent.name.endswith("2017"):
        dataset_root = images_root.parent.parent
        for subdir in _split_subdir_candidates(split):
            candidates.append(dataset_root / subdir / rel)

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def _resolve_subset_id(
    db_path: str,
    subset_name: str,
    split: Optional[str] = None,
) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if split is None:
        cur.execute(
            """
            SELECT id
            FROM subsets
            WHERE name = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (subset_name,),
        )
    else:
        cur.execute(
            """
            SELECT id
            FROM subsets
            WHERE name = ?
              AND split = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (subset_name, split),
        )

    row = cur.fetchone()
    conn.close()

    if row is None:
        if split is None:
            raise ValueError(f"No subset named '{subset_name}' in {db_path}")
        raise ValueError(
            f"No subset named '{subset_name}' with split='{split}' in {db_path}"
        )

    return int(row[0])


def get_variant_records_for_subset(
    db_path: str,
    subset_name: str,
    split: Optional[str] = None,
) -> Tuple[int, List[RowWithSplit]]:
    """
    Return (subset_id, [(rel_path, image_split), ...]) for the given subset_name.
    """
    subset_id = _resolve_subset_id(db_path=db_path, subset_name=subset_name, split=split)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if split is None:
        cur.execute(
            """
            SELECT v.rel_path, i.split
            FROM variants v
            JOIN subset_variants sv
              ON sv.variant_id = v.id
            JOIN images i
              ON i.id = v.image_id
            WHERE sv.subset_id = ?
            ORDER BY v.id
            """,
            (subset_id,),
        )
    else:
        cur.execute(
            """
            SELECT v.rel_path, i.split
            FROM variants v
            JOIN subset_variants sv
              ON sv.variant_id = v.id
            JOIN images i
              ON i.id = v.image_id
            WHERE sv.subset_id = ?
              AND i.split = ?
            ORDER BY v.id
            """,
            (subset_id, split),
        )

    rows = [(str(r[0]), str(r[1])) for r in cur.fetchall()]
    conn.close()

    return subset_id, rows


def get_variant_paths_for_subset(
    db_path: str,
    subset_name: str,
    split: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """
    Return (subset_id, list_of_rel_paths) for the given subset_name.
    """
    subset_id, rows = get_variant_records_for_subset(
        db_path=db_path,
        subset_name=subset_name,
        split=split,
    )
    rel_paths = [rel_path for rel_path, _ in rows]
    return subset_id, rel_paths


# ============================================================
# Smart sampling helpers
# ============================================================

def _order_rows_for_image_round_robin(
    rows_img: List[RowWithDeg],
    rng: np.random.Generator,
) -> List[RowWithDeg]:
    if not rows_img:
        return []

    rows_img = list(rows_img)
    rng.shuffle(rows_img)

    by_deg: Dict[str, List[RowWithDeg]] = defaultdict(list)
    for r in rows_img:
        by_deg[r[3]].append(r)

    for deg_name in by_deg:
        rng.shuffle(by_deg[deg_name])

    deg_names = list(by_deg.keys())
    idx_per_deg: Dict[str, int] = {deg: 0 for deg in deg_names}

    ordered: List[RowWithDeg] = []
    while True:
        active_degs = [deg for deg in deg_names if idx_per_deg[deg] < len(by_deg[deg])]
        if not active_degs:
            break

        rng.shuffle(active_degs)
        for deg in active_degs:
            i = idx_per_deg[deg]
            ordered.append(by_deg[deg][i])
            idx_per_deg[deg] = i + 1

    return ordered


def smart_sample_rows_per_subset(
    rows: List[RowWithDeg],
    variants_per_subset: Optional[int],
    variants_per_image: Optional[int],
    seed: int,
) -> List[RowWithDeg]:
    if not rows:
        return []

    variants_per_image = None if variants_per_image is None or variants_per_image <= 0 else variants_per_image
    variants_per_subset = None if variants_per_subset is None or variants_per_subset <= 0 else variants_per_subset

    rng = np.random.default_rng(seed)

    by_image: Dict[int, List[RowWithDeg]] = defaultdict(list)
    for r in rows:
        by_image[r[1]].append(r)

    image_ids = list(by_image.keys())
    rng.shuffle(image_ids)

    sampled_rows: List[RowWithDeg] = []

    for img_id in image_ids:
        ordered_img_rows = _order_rows_for_image_round_robin(by_image[img_id], rng)

        if variants_per_image is not None:
            ordered_img_rows = ordered_img_rows[:variants_per_image]

        sampled_rows.extend(ordered_img_rows)

    if variants_per_subset is not None:
        sampled_rows = sampled_rows[:variants_per_subset]

    return sampled_rows


# ============================================================
# Dataset: multi-subset images + curve targets
# ============================================================

class CurveMonitorDataset(Dataset):
    """
    Samples image variants from all subsets that have curve targets.

    Expected `subsets_info` format:
      [
        {
          "subset_id": int,
          "subset_name": str,
          "split": str,
        },
        ...
      ]

    Returns per sample:
      {
        "img": Tensor [C, imgsz, imgsz],
        "subset_index": int,
        "gt_boxes": Tensor [N, 4] in letterboxed coords (x1,y1,x2,y2),
        "pred_boxes": Tensor [M, 4] in letterboxed coords (x1,y1,x2,y2),
        "pred_scores": Tensor [M],
        "pred_max_iou": Tensor [M],
      }
    """

    def __init__(
        self,
        images_root: str,
        db_path: str,
        subsets_info: List[Dict],
        imgsz: int = 640,
        channels: int = 3,
        split: Optional[str] = None,
        class_id: int = 0,
        smart_sampling: bool = False,
        variants_per_subset: Optional[int] = None,
        variants_per_image: Optional[int] = None,
        smart_sample_seed: int = 0,
    ):
        super().__init__()
        self.images_root = Path(images_root)
        self.imgsz = imgsz
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.split = split
        self.class_id = class_id
        self.db_path = db_path
        self.subsets_info = subsets_info

        # (full_img_path, subset_index, image_id, variant_id)
        self.samples: List[Tuple[str, int, int, int]] = []

        self.variants_per_subset = variants_per_subset
        self.smart_sampling = smart_sampling
        self.variants_per_image = variants_per_image
        self.smart_sample_seed = smart_sample_seed

        self._conn: Optional[sqlite3.Connection] = None

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        for subset_index, subset_info in enumerate(subsets_info):
            subset_id = int(subset_info["subset_id"])

            if self.split is None:
                cur.execute(
                    """
                    SELECT v.rel_path, v.image_id, v.id, v.deg_name, v.severity, i.split
                    FROM subset_variants sv
                    JOIN variants v
                      ON sv.variant_id = v.id
                    JOIN images i
                      ON i.id = v.image_id
                    WHERE sv.subset_id = ?
                    ORDER BY v.id
                    """,
                    (subset_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT v.rel_path, v.image_id, v.id, v.deg_name, v.severity, i.split
                    FROM subset_variants sv
                    JOIN variants v
                      ON sv.variant_id = v.id
                    JOIN images i
                      ON i.id = v.image_id
                    WHERE sv.subset_id = ?
                      AND i.split = ?
                    ORDER BY v.id
                    """,
                    (subset_id, self.split),
                )

            raw_rows = cur.fetchall()

            rows: List[RowWithDeg] = [
                (
                    str(rel_path),
                    int(image_id),
                    int(variant_id),
                    str(deg_name),
                    int(severity),
                )
                for rel_path, image_id, variant_id, deg_name, severity, _img_split in raw_rows
            ]

            split_by_variant: Dict[int, str] = {
                int(variant_id): str(img_split)
                for _rel_path, _image_id, variant_id, _deg_name, _severity, img_split in raw_rows
            }

            if self.smart_sampling:
                rows = smart_sample_rows_per_subset(
                    rows,
                    variants_per_subset=self.variants_per_subset,
                    variants_per_image=self.variants_per_image,
                    seed=self.smart_sample_seed + subset_id,
                )
            else:
                if self.variants_per_subset is not None and self.variants_per_subset > 0:
                    rows = rows[:self.variants_per_subset]

            for rel_path, image_id, variant_id, _, _ in rows:
                img_split = split_by_variant[int(variant_id)]
                full_path = str(_resolve_image_path(self.images_root, rel_path, img_split))
                self.samples.append(
                    (full_path, subset_index, int(image_id), int(variant_id))
                )

        conn.close()

        self.ni = len(self.samples)
        if self.ni == 0:
            raise RuntimeError(
                f"No variants found for any subset in dataset "
                f"(split={self.split}, root={self.images_root})."
            )

        print(
            f"CurveMonitorDataset :: split={self.split}, "
            f"{len(subsets_info)} subsets, {self.ni} images "
            f"(root={self.images_root})"
        )

    # ------------------------------------------------------------------
    # SQLite connection handling
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def __del__(self):
        try:
            if hasattr(self, "_conn") and self._conn is not None:
                self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_channel_dim(im: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if im is None:
            return None
        return im[..., None] if im.ndim == 2 else im

    @staticmethod
    def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
        try:
            file_bytes = np.fromfile(filename, np.uint8)
        except Exception:
            return None

        if filename.lower().endswith((".tiff", ".tif")):
            success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
            if not success:
                return None

            if len(frames) == 1:
                im = frames[0]
            else:
                im = np.stack(frames, axis=2)

            return CurveMonitorDataset._ensure_channel_dim(im)

        im = cv2.imdecode(file_bytes, flags)
        return CurveMonitorDataset._ensure_channel_dim(im)

    def _load_and_resize(self, idx: int):
        path = self.samples[idx][0]

        im = self.imread(path, flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Image not found: {path}")

        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)

        if r != 1:
            w = min(math.ceil(w0 * r), self.imgsz)
            h = min(math.ceil(h0 * r), self.imgsz)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            im = self._ensure_channel_dim(im)
        else:
            h, w = h0, w0

        return im, (h0, w0), (h, w)

    @staticmethod
    def _to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(img_uint8)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )

    @staticmethod
    def _apply_resize_and_letterbox_to_xyxy(
        boxes: torch.Tensor,
        w0: int,
        h0: int,
        w: int,
        h: int,
        ratio,
        pad,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes

        scale_w = w / w0
        scale_h = h / h0
        boxes[:, [0, 2]] *= scale_w
        boxes[:, [1, 3]] *= scale_h

        rw, rh = ratio if isinstance(ratio, (tuple, list)) else (ratio, ratio)
        dw, dh = pad if isinstance(pad, (tuple, list)) else (pad, pad)

        boxes[:, [0, 2]] *= rw
        boxes[:, [1, 3]] *= rh
        boxes[:, [0, 2]] += dw
        boxes[:, [1, 3]] += dh

        return boxes

    def __len__(self):
        return self.ni

    # ------------------------------------------------------------------
    # GT boxes
    # ------------------------------------------------------------------
    def _get_gt_boxes_from_db(self, image_id: int) -> np.ndarray:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT x, y, w, h
            FROM gt_objects
            WHERE image_id = ? AND class_id = ?
            """,
            (image_id, self.class_id),
        )
        rows = cur.fetchall()

        if not rows:
            return np.zeros((0, 4), dtype=np.float32)

        return np.asarray(rows, dtype=np.float32)  # xywh

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def _get_predictions_from_db(self, variant_id: int) -> Dict[str, np.ndarray]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT x, y, w, h, score, max_iou
            FROM predictions
            WHERE variant_id = ? AND category_id = ?
            """,
            (variant_id, self.class_id),
        )
        rows = cur.fetchall()

        if not rows:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "max_iou": np.zeros((0,), dtype=np.float32),
            }

        boxes = []
        scores = []
        max_iou = []

        for x, y, w, h, s, miou in rows:
            x1 = float(x)
            y1 = float(y)
            x2 = x1 + float(w)
            y2 = y1 + float(h)
            boxes.append([x1, y1, x2, y2])
            scores.append(float(s))
            max_iou.append(0.0 if miou is None else float(miou))

        return {
            "boxes": np.asarray(boxes, dtype=np.float32),
            "scores": np.asarray(scores, dtype=np.float32),
            "max_iou": np.asarray(max_iou, dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        t0 = time.time()
        im, (h0, w0), (h, w) = self._load_and_resize(index)
        t1 = time.time()

        if self.channels == 3 and im.ndim == 3 and im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        t = self._to_tensor(im.copy())
        lb_out = letterbox_tensor(t)
        t2 = time.time()

        if isinstance(lb_out, (tuple, list)):
            if len(lb_out) == 3:
                t_lb, ratio, pad = lb_out
            else:
                t_lb = lb_out[0]
                ratio = (1.0, 1.0)
                pad = (0.0, 0.0)
        else:
            t_lb = lb_out
            ratio = (1.0, 1.0)
            pad = (0.0, 0.0)

        _, subset_index, image_id, variant_id = self.samples[index]

        # GT boxes: xywh -> xyxy -> resized -> letterboxed
        gt_boxes_orig = self._get_gt_boxes_from_db(image_id)
        t3 = time.time()

        if gt_boxes_orig.shape[0] > 0:
            gt_boxes = torch.from_numpy(gt_boxes_orig).float()
            gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
            gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]
            gt_boxes = self._apply_resize_and_letterbox_to_xyxy(
                gt_boxes, w0=w0, h0=h0, w=w, h=h, ratio=ratio, pad=pad
            )
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)

        # Pred boxes: xyxy -> resized -> letterboxed
        pred_info = self._get_predictions_from_db(variant_id)
        t4 = time.time()

        pred_boxes_orig = pred_info["boxes"]
        pred_scores = pred_info["scores"]
        pred_max_iou = pred_info["max_iou"]

        if pred_boxes_orig.shape[0] > 0:
            pred_boxes = torch.from_numpy(pred_boxes_orig).float()
            pred_boxes = self._apply_resize_and_letterbox_to_xyxy(
                pred_boxes, w0=w0, h0=h0, w=w, h=h, ratio=ratio, pad=pad
            )
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)

        pred_scores = torch.from_numpy(pred_scores).float()
        pred_max_iou = torch.from_numpy(pred_max_iou).float()

        # Debug if needed:
        # if index % 100 == 0:
        #     print(
        #         f"idx={index} load+resize={t1-t0:.3f}s, "
        #         f"letterbox={t2-t1:.3f}s, "
        #         f"gt_db={t3-t2:.3f}s, pred_db={t4-t3:.3f}s"
        #     )

        return {
            "img": t_lb.to(torch.float32),
            "subset_index": int(subset_index),
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "pred_max_iou": pred_max_iou,
        }

    # ------------------------------------------------------------------
    # collate_fn
    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: List[Dict]):
        return {
            "img": torch.stack([b["img"] for b in batch], 0),
            "subset_index": torch.tensor(
                [b["subset_index"] for b in batch], dtype=torch.long
            ),
            "gt_boxes": [b["gt_boxes"] for b in batch],
            "pred_boxes": [b["pred_boxes"] for b in batch],
            "pred_scores": [b["pred_scores"] for b in batch],
            "pred_max_iou": [b["pred_max_iou"] for b in batch],
        }


class SubsetVariantDataset(Dataset):
    """
    Dataset that loads only the variants listed in a given subset.

    Returns:
      {
        "img": Tensor [C, H, W],
        "rel_path": str
      }
    """

    def __init__(
        self,
        dataset_root: str,
        db_path: str,
        subset_name: str,
        imgsz: int = 640,
        channels: int = 3,
        max_images: Optional[int] = None,
        split: Optional[str] = None,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.imgsz = imgsz
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.split = split

        _, rel_records = get_variant_records_for_subset(
            db_path=db_path,
            subset_name=subset_name,
            split=self.split,
        )

        if not rel_records:
            raise RuntimeError(
                f"Subset '{subset_name}' has no variants (split={self.split})."
            )

        if max_images is not None:
            rel_records = rel_records[:max_images]

        self.rel_paths: List[Path] = [Path(p) for p, _ in rel_records]
        self.im_files: List[str] = [
            str(_resolve_image_path(self.dataset_root, rel_path, img_split))
            for rel_path, img_split in rel_records
        ]
        self.ni = len(self.im_files)

        self.ims: List[Optional[np.ndarray]] = [None] * self.ni
        self.im_hw0: List[Optional[Tuple[int, int]]] = [None] * self.ni
        self.im_hw: List[Optional[Tuple[int, int]]] = [None] * self.ni

        print(
            f"SubsetVariantDataset :: split={self.split}, subset='{subset_name}' "
            f"-> {self.ni} images (root={self.dataset_root})"
        )

    @staticmethod
    def _ensure_channel_dim(im: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if im is None:
            return None
        return im[..., None] if im.ndim == 2 else im

    @staticmethod
    def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
        try:
            file_bytes = np.fromfile(filename, np.uint8)
        except Exception:
            return None

        if filename.lower().endswith((".tiff", ".tif")):
            success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
            if not success:
                return None

            if len(frames) == 1:
                im = frames[0]
            else:
                im = np.stack(frames, axis=2)

            return SubsetVariantDataset._ensure_channel_dim(im)

        im = cv2.imdecode(file_bytes, flags)
        return SubsetVariantDataset._ensure_channel_dim(im)

    def _load_and_resize(self, idx: int):
        path = self.im_files[idx]
        im = self.ims[idx]

        if im is None:
            im = self.imread(path, flags=self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"Image not found: {path}")

            h0, w0 = im.shape[:2]
            r = self.imgsz / max(h0, w0)

            if r != 1:
                w = min(math.ceil(w0 * r), self.imgsz)
                h = min(math.ceil(h0 * r), self.imgsz)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                im = self._ensure_channel_dim(im)

            self.ims[idx] = im
            self.im_hw0[idx] = (h0, w0)
            self.im_hw[idx] = im.shape[:2]

        return self.ims[idx], self.im_hw0[idx], self.im_hw[idx]

    @staticmethod
    def _to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(img_uint8)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )

    def __len__(self):
        return self.ni

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        img, _, _ = self._load_and_resize(index)

        if self.channels == 3 and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t = self._to_tensor(img.copy())
        lb_out = letterbox_tensor(t)

        if isinstance(lb_out, (tuple, list)):
            t_lb = lb_out[0]
        else:
            t_lb = lb_out

        return {
            "img": t_lb.to(torch.float32),
            "rel_path": str(self.rel_paths[index]),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor | str]]):
        return {
            "img": torch.stack([b["img"] for b in batch], 0),
            "rel_path": [b["rel_path"] for b in batch],
        }


def load_subset_roc_froc(
    db_path: str,
    subset_name: str,
    class_id: int = 0,
    iou_thr: float = 0.5,
    split: Optional[str] = None,
):
    """
    Load ROC/FROC summary metrics + full curve for a given subset from the DB.
    Supports both the old and newer DB schemas.
    """
    subset_id = _resolve_subset_id(
        db_path=db_path,
        subset_name=subset_name,
        split=split,
    )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    metrics_table, points_table = _resolve_curve_table_names(cur)

    if metrics_table == "subset_roc_froc_metrics":
        cur.execute(
            """
            SELECT
                num_gt,
                num_imgs,
                roc_auc,
                roc_pauc_fpr_0p1,
                froc_auc_fppi_3,
                recall_fppi_0p1,
                recall_fppi_0p5,
                recall_fppi_1p0
            FROM subset_roc_froc_metrics
            WHERE subset_id = ? AND class_id = ? AND ABS(iou_thr - ?) < 1e-8
            """,
            (subset_id, class_id, float(iou_thr)),
        )
    else:
        metrics_cols = _get_table_columns(cur, metrics_table)

        if "froc_auc_fppi_3" in metrics_cols:
            froc_auc_expr = "froc_auc_fppi_3"
        elif "froc_auc" in metrics_cols:
            froc_auc_expr = "froc_auc"
        else:
            conn.close()
            raise RuntimeError(
                f"Table '{metrics_table}' is missing a FROC AUC column."
            )

        order_sql = ""
        if "froc_fppi_max" in metrics_cols:
            order_sql = "ORDER BY ABS(froc_fppi_max - 3.0) ASC LIMIT 1"
        else:
            order_sql = "LIMIT 1"

        cur.execute(
            f"""
            SELECT
                num_gt,
                num_imgs,
                roc_auc,
                roc_pauc_fpr_0p1,
                {froc_auc_expr},
                recall_fppi_0p1,
                recall_fppi_0p5,
                recall_fppi_1p0
            FROM {metrics_table}
            WHERE subset_id = ? AND class_id = ? AND ABS(iou_thr - ?) < 1e-8
            {order_sql}
            """,
            (subset_id, class_id, float(iou_thr)),
        )

    mrow = cur.fetchone()
    if mrow is None:
        conn.close()
        raise ValueError(
            f"No subset metrics found for "
            f"subset_id={subset_id}, class_id={class_id}, iou_thr={iou_thr}."
        )

    (
        num_gt,
        num_imgs,
        roc_auc,
        roc_pauc_fpr_0p1,
        froc_auc_fppi_3,
        recall_fppi_0p1,
        recall_fppi_0p5,
        recall_fppi_1p0,
    ) = mrow

    cur.execute(
        f"""
        SELECT idx, thr, tpr, fpr, fppi, precision
        FROM {points_table}
        WHERE subset_id = ? AND class_id = ? AND ABS(iou_thr - ?) < 1e-8
        ORDER BY idx ASC
        """,
        (subset_id, class_id, float(iou_thr)),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise ValueError(
            f"No subset operating points found for "
            f"subset_id={subset_id}, class_id={class_id}, iou_thr={iou_thr}."
        )

    idxs = np.asarray([r[0] for r in rows], dtype=int)
    thr = np.asarray([r[1] for r in rows], dtype=float)
    tpr = np.asarray([r[2] for r in rows], dtype=float)
    fpr = np.asarray([r[3] for r in rows], dtype=float)
    fppi = np.asarray([r[4] for r in rows], dtype=float)
    precision = np.asarray([r[5] for r in rows], dtype=float)

    order = np.argsort(idxs)
    thr = thr[order]
    tpr = tpr[order]
    fpr = fpr[order]
    fppi = fppi[order]
    precision = precision[order]

    return {
        "subset_id": subset_id,
        "class_id": class_id,
        "iou_thr": float(iou_thr),
        "num_gt": int(num_gt),
        "num_imgs": int(num_imgs),
        "roc_auc": float(roc_auc),
        "roc_pauc_fpr_0p1": float(roc_pauc_fpr_0p1),
        "froc_auc_fppi_3": float(froc_auc_fppi_3),
        "recall_fppi_0p1": float(recall_fppi_0p1),
        "recall_fppi_0p5": float(recall_fppi_0p5),
        "recall_fppi_1p0": float(recall_fppi_1p0),
        "thr": thr,
        "tpr": tpr,
        "fpr": fpr,
        "fppi": fppi,
        "precision": precision,
    }
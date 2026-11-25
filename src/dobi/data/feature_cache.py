#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Cache of pooled frozen-detector features.

The detector is frozen and the monitor input pipeline is deterministic (no
augmentation): image -> resize/letterbox (CurveMonitorDataset / preprocess_for_monitor)
-> frozen YOLO body (build_backbone_runner) -> per FPN level global average and
global max pooling. The monitor heads only ever see these pooled vectors, so they
can be computed once per image variant and reused by training and evaluation.

Heads are not modified: a pooled vector is turned back into exact "pseudo feature
maps" of spatial size 1x2 whose mean equals the GAP and whose max equals the GMP
(see ``pooled_to_pseudo_feats``), and the unchanged ``head(feats)`` is called.

Layout of a cache directory:
    meta.json                     weights, imgsz, feat_channels, pooling, dtype
    <split>/features.npy          float32 [N, 2*sum(C)]   ([gap_l, gmp_l] per level, level order)
    <split>/index.npz             variant_ids [N], rel_paths [N]

Build:
    python src/dobi/data/feature_cache.py --db <dobi.db> \
        --images_root <.../train2017/images> --weights yolov10n.pt --out <cache_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

POOLING = "gap_gmp_per_level"


# ------------------------------------------------------------------
# pooled <-> pseudo feature maps
# ------------------------------------------------------------------
def pool_feats(feats: Sequence[torch.Tensor]) -> torch.Tensor:
    """Same pooling as the monitor heads: concat over levels of [GAP_l, GMP_l]."""
    return torch.cat([torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1) for f in feats], dim=1)


def pooled_to_pseudo_feats(x: torch.Tensor, feat_channels: Sequence[int]) -> List[torch.Tensor]:
    """
    Inverse view of ``pool_feats``: per level a [B, C, 1, 2] map with values (gmp, 2*gap - gmp).
    Its spatial mean is gap and its spatial max is gmp (since gmp >= gap), so any head that pools
    with mean/amax over (2, 3) receives exactly the cached pooled features.
    """
    out, o = [], 0
    for c in feat_channels:
        gap = x[:, o:o + c]
        gmp = x[:, o + c:o + 2 * c]
        o += 2 * c
        gmp = torch.maximum(gmp, gap)  # guard against rounding
        out.append(torch.stack([gmp, 2.0 * gap - gmp], dim=-1).unsqueeze(2))
    return out


# ------------------------------------------------------------------
# cache access
# ------------------------------------------------------------------
class FeatureCache:
    def __init__(self, cache_dir: str | Path, split: str):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.split = split
        d = self.dir / split
        self.features = np.load(d / "features.npy", mmap_mode="r")
        idx = np.load(d / "index.npz", allow_pickle=False)
        self.variant_ids = idx["variant_ids"]
        self.rel_paths = idx["rel_paths"]
        self._row_by_vid = {int(v): i for i, v in enumerate(self.variant_ids)}
        self._row_by_rel = {str(r): i for i, r in enumerate(self.rel_paths)}

    @property
    def feat_channels(self) -> List[int]:
        return [int(c) for c in self.meta["feat_channels"]]

    def row_for_variant(self, vid: int) -> int:
        return self._row_by_vid[int(vid)]

    def row_for_relpath(self, rel_path: str) -> int:
        return self._row_by_rel[str(rel_path)]

    def get(self, rows) -> np.ndarray:
        return np.asarray(self.features[rows], dtype=np.float32)


def check_cache_matches(cache_dir: str | Path, weights: str, imgsz: int) -> Dict:
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    if Path(meta["weights"]).name != Path(weights).name or int(meta["imgsz"]) != int(imgsz):
        raise RuntimeError(f"Feature cache {cache_dir} was built for weights={meta['weights']} imgsz={meta['imgsz']}, "
                           f"requested weights={weights} imgsz={imgsz}")
    return meta


class CachedFeatureDataset(Dataset):
    """Wraps the sample list of a CurveMonitorDataset; returns cached pooled features instead of images."""

    def __init__(self, base_dataset, cache: FeatureCache):
        self.samples = base_dataset.samples  # (full_img_path, subset_index, image_id, variant_id)
        rows = np.asarray([cache.row_for_variant(s[3]) for s in self.samples], dtype=np.int64)
        self.feats = torch.from_numpy(cache.get(np.sort(np.unique(rows))))
        remap = {r: i for i, r in enumerate(np.sort(np.unique(rows)))}
        self.rows = torch.as_tensor([remap[r] for r in rows], dtype=torch.long)
        self.subset_index = torch.as_tensor([s[1] for s in self.samples], dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return {"feat": self.feats[self.rows[i]], "subset_index": int(self.subset_index[i])}

    def __getitems__(self, idxs):
        """Batched fetch used by DataLoader (same sampler order, already collated)."""
        idx = torch.as_tensor(idxs, dtype=torch.long)
        return {"feat": self.feats[self.rows[idx]], "subset_index": self.subset_index[idx]}

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch, dict):  # produced by __getitems__
            return batch
        return {
            "feat": torch.stack([b["feat"] for b in batch], 0),
            "subset_index": torch.tensor([b["subset_index"] for b in batch], dtype=torch.long),
        }


# ------------------------------------------------------------------
# build
# ------------------------------------------------------------------
class _VariantImages(Dataset):
    def __init__(self, items, images_root: Path, imgsz: int):
        from dobi.data.dataloader.roc_curves_data_loader import CurveMonitorDataset, _resolve_image_path
        from dobi.utils.img_utils import letterbox_tensor
        import cv2
        import math
        self.items = items
        self.images_root = images_root
        self.imgsz = imgsz
        self._resolve = _resolve_image_path
        self._imread = CurveMonitorDataset.imread
        self._to_tensor = CurveMonitorDataset._to_tensor
        self._letterbox = letterbox_tensor
        self._cv2 = cv2
        self._math = math

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, rel, split = self.items[i]
        cv2, math = self._cv2, self._math
        path = self._resolve(self.images_root, rel, split)
        im = self._imread(str(path), flags=cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        h0, w0 = im.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:  # identical to CurveMonitorDataset._load_and_resize and preprocess_for_monitor
            im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
                            interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        t = self._letterbox(self._to_tensor(im.copy()))[0]
        return t.float(), i


def split_variants(conn: sqlite3.Connection, split: str):
    rows = conn.execute(
        """
        SELECT DISTINCT v.id, v.rel_path, i.split
        FROM subset_variants sv
        JOIN subsets s ON s.id = sv.subset_id
        JOIN variants v ON v.id = sv.variant_id
        JOIN images i ON i.id = v.image_id
        WHERE s.split = ?
        ORDER BY v.id
        """,
        (split,),
    ).fetchall()
    return [(int(a), str(b), str(c)) for a, b, c in rows]


def _sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build(args):
    from ultralytics import YOLO
    from dobi.models.spline_monitor_model import build_backbone_runner

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(args.weights)
    yolo.to(device)
    yolo.model.eval()
    for p in yolo.model.parameters():
        p.requires_grad_(False)
    run = build_backbone_runner(yolo.model)
    with torch.no_grad():
        feat_channels = [int(f.shape[1]) for f in run(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))]

    meta = {
        "weights": str(args.weights),
        "weights_sha256": _sha256(Path(yolo.ckpt_path) if getattr(yolo, "ckpt_path", None) else Path(args.weights)),
        "imgsz": int(args.imgsz),
        "feat_channels": feat_channels,
        "pooling": POOLING,
        "dtype": "float32",
        "precision": "fp32, no autocast, unfused model (as in training)",
        "db": str(args.db),
        "images_root": str(args.images_root),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    conn = sqlite3.connect(args.db)
    for split in args.splits:
        items = split_variants(conn, split)
        d = out / split
        d.mkdir(exist_ok=True)
        if (d / "features.npy").exists() and not args.overwrite:
            print(f"{split}: exists, skipping")
            continue
        D = 2 * sum(feat_channels)
        feats = np.lib.format.open_memmap(d / "features.npy.tmp.npy", mode="w+", dtype=np.float32, shape=(len(items), D))
        loader = DataLoader(_VariantImages(items, Path(args.images_root), args.imgsz), batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False, pin_memory=True)
        t0 = time.time()
        with torch.no_grad():
            for imgs, idx in tqdm(loader, desc=f"cache {split} ({len(items)})", mininterval=30):
                pooled = pool_feats(run(imgs.to(device, non_blocking=True))).float().cpu().numpy()
                feats[idx.numpy()] = pooled
        feats.flush()
        del feats
        (d / "features.npy.tmp.npy").rename(d / "features.npy")
        np.savez(d / "index.npz", variant_ids=np.asarray([a for a, _, _ in items], dtype=np.int64),
                 rel_paths=np.asarray([b for _, b, _ in items]))
        print(f"{split}: {len(items)} variants in {time.time() - t0:.0f}s")
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Cache pooled frozen-detector features for all subset variants.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--weights", default='weights/yolov10n.pt')
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    build(ap.parse_args())


if __name__ == "__main__":
    main()

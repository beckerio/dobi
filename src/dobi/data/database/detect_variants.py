#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Run the frozen detector on every image variant of one split and write its detections in the format that
generate_db.py reads.

    python -m dobi.data.database.detect_variants --weights weights/yolov10n.pt --split_root <COCOMiniPerson-D>/data/test2017
    python -m dobi.data.database.generate_db --db dobi.db --dataset_root <COCOMiniPerson-D>/data

Input:  <split_root>/images/<type>/deg_<s>/<base>.png and <split_root>/images/pristine/<base>.png
Output: <split_root>/predictions/<type>/deg_<s>/YOLO_format/<base>.txt (pristine: predictions/pristine/YOLO_format/),
        one line per detection: "class x_center y_center width height score", coordinates normalized to [0, 1],
        all values with six decimals. Existing files are skipped (resumable), never overwritten.
        <out>/detect_protocol.json records the settings and the numerical state.

Protocol of the main study (all settings are explicit here):
  * each image is read with OpenCV and resized so that its long side is 640 px (bilinear, aspect ratio kept),
    converted to RGB and to a float32 tensor in [0, 1], and letterboxed to 640 x 640 (padding value 114/255);
  * one image per forward pass, fp32 (half=False, no autocast), no test-time augmentation;
  * confidence floor 0.001, IoU 0.7 for NMS (not used by the NMS-free YOLOv10 head), at most 300 detections,
    all 80 COCO classes, class-aware NMS;
  * numerical state (PyTorch defaults at the time of the study): FP32 matrix multiplication with precision
    "highest", matmul TF32 off, cuDNN TF32 on, cudnn.benchmark off, no deterministic algorithms;
  * boxes are mapped back to the resized image (letterbox padding removed, clipped to the image) and normalized
    by its width and height.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dobi.utils.img_utils import letterbox_tensor, load_image

PROTOCOL = {"imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300, "classes": None, "agnostic_nms": False,
            "augment": False, "half": False, "batch": 1, "pad_value": 114}
NUMERIC_STATE = {"float32_matmul_precision": "highest", "matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                 "cudnn_benchmark": False, "cudnn_deterministic": False, "deterministic_algorithms": False,
                 "autocast": False}
IMG_EXT = (".png", ".jpg", ".jpeg")


def set_numeric_state(state: dict) -> dict:
    torch.set_float32_matmul_precision(state["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = state["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.use_deterministic_algorithms(state["deterministic_algorithms"])
    return check_numeric_state(state)


def check_numeric_state(state: dict) -> dict:
    now = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "cudnn_deterministic": torch.backends.cudnn.deterministic,
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "autocast": torch.is_autocast_enabled()}
    assert now == state, f"numerical state differs: {now} != {state}"
    return now


def image_variants(images_root: Path):
    """Relative paths <type>/deg_<s>/<file> and pristine/<file>, sorted."""
    out = []
    for p in sorted(images_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            rel = p.relative_to(images_root)
            if rel.parts[0] == "pristine" or (len(rel.parts) == 3 and rel.parts[1].startswith("deg_")):
                out.append(rel)
    return out


def prediction_path(pred_root: Path, rel: Path) -> Path:
    return pred_root / rel.parent / "YOLO_format" / (rel.stem + ".txt")


def preprocess(path: Path, imgsz: int, pad_value: int):
    """Returns the letterboxed 1x3xHxW float32 tensor and the (height, width) of the resized image."""
    im_bgr, _, _ = load_image(str(path), imgsz)
    h, w = im_bgr.shape[:2]
    rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div(255)
    x, _, _ = letterbox_tensor(x, new_shape=(imgsz, imgsz), pad_value=pad_value)
    return x.unsqueeze(0), (h, w)


def to_resized_image(xyxy: torch.Tensor, net_hw, img_hw) -> torch.Tensor:
    """Map xyxy boxes from the letterboxed network input back to the resized image: remove the (centered) padding,
    divide by the letterbox gain, clip to the image."""
    gain = min(net_hw[0] / img_hw[0], net_hw[1] / img_hw[1])
    pad_x = (net_hw[1] - img_hw[1] * gain) / 2
    pad_y = (net_hw[0] - img_hw[0] * gain) / 2
    b = xyxy.clone()
    b[:, [0, 2]] -= pad_x
    b[:, [1, 3]] -= pad_y
    b[:, :4] /= gain
    b[:, [0, 2]] = b[:, [0, 2]].clamp(0, img_hw[1])
    b[:, [1, 3]] = b[:, [1, 3]].clamp(0, img_hw[0])
    return b


def yolo_lines(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, img_hw) -> list[str]:
    h, w = float(img_hw[0]), float(img_hw[1])
    lines = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        bw, bh = x2 - x1, y2 - y1
        xc, yc = (x1 + bw / 2.0) / w, (y1 + bh / 2.0) / h
        lines.append(f"{int(k)} {xc:04f} {yc:04f} {bw / w:04f} {bh / h:04f} {c:04f}")
    return lines


def detect(model, path: Path, device: str, proto: dict = PROTOCOL) -> list[str]:
    x, img_hw = preprocess(path, proto["imgsz"], proto["pad_value"])
    r = model.predict(x.to(device), imgsz=proto["imgsz"], conf=proto["conf"], iou=proto["iou"], max_det=proto["max_det"],
                      classes=proto["classes"], agnostic_nms=proto["agnostic_nms"], augment=proto["augment"],
                      half=proto["half"], batch=proto["batch"], device=device, verbose=False)[0]
    b = to_resized_image(r.boxes.xyxy, x.shape[2:], img_hw)
    return yolo_lines(b.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy(), img_hw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, help="detector weights, e.g. weights/yolov10n.pt (see weights/README.md)")
    ap.add_argument("--split_root", required=True, help="<COCOMiniPerson-D>/data/<split>2017 (contains images/)")
    ap.add_argument("--out", default=None, help="predictions root (default: <split_root>/predictions)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--list", default=None, help="optional text file with relative image paths (one per line) to restrict to")
    a = ap.parse_args()
    from ultralytics import YOLO

    state = set_numeric_state(NUMERIC_STATE)
    split_root = Path(a.split_root)
    out = Path(a.out) if a.out else split_root / "predictions"
    rels = image_variants(split_root / "images")
    if a.list:
        keep = {l.strip() for l in Path(a.list).read_text().splitlines() if l.strip()}
        rels = [r for r in rels if r.as_posix() in keep]
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    t0, n_done = time.time(), 0
    for i, rel in enumerate(rels):
        dst = prediction_path(out, rel)
        if dst.exists():
            continue
        lines = detect(model, split_root / "images" / rel, a.device)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".txt.tmp")
        tmp.write_text("".join(l + "\n" for l in lines))
        tmp.rename(dst)
        n_done += 1
        if n_done % 1000 == 0:
            print(f"{i + 1}/{len(rels)} images, {n_done / (time.time() - t0):.1f} img/s", flush=True)
    check_numeric_state(state)
    import ultralytics
    (out / "detect_protocol.json").write_text(json.dumps(
        {"protocol": PROTOCOL, "numeric_state": state, "weights": Path(a.weights).name, "device": a.device,
         "device_name": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else platform.processor(),
         "images": len(rels), "written": n_done, "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        indent=1))
    print(f"done: {n_done} new prediction files ({len(rels)} images) -> {out}")


if __name__ == "__main__":
    main()

# Data

This repository contains **no images and no annotations**. The paper uses two datasets:

- **COCOMiniPerson-D**: a person-only, synthetically degraded subset of COCO-2017. It is built from the official COCO images with the recipe below.
- **BDD100K**: used by reference only (natural-condition study). No BDD100K data is redistributed.

## 1. Download COCO-2017

| File | URL |
|---|---|
| train2017 images | http://images.cocodataset.org/zips/train2017.zip |
| val2017 images | http://images.cocodataset.org/zips/val2017.zip |
| train/val annotations | http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| COCO-minitrain annotations (`instances_minitrain2017.json`, 25,000 train2017 images) | https://github.com/giddyyupp/coco-minitrain |

The COCO annotations are licensed under CC BY 4.0. The images are subject to the Flickr terms of use (see https://cocodataset.org/#termsofuse). The `instances_minitrain2017.json` we used has SHA-256 `5682389c3bb0a356912771811f7836659cc6de5a588413752922ccb94a6df1a8`. The official `instances_val2017.json` has SHA-256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`.

## 2. Base images and splits

**The image list is authoritative.** [`datasets/cocominiperson_d/image_list.csv`](datasets/cocominiperson_d/image_list.csv) lists all 10,356 base images. It has the columns `coco_image_id, coco_file_name, coco_split, dobi_split`. It contains no pixels or labels.

| DOBI split | Source | Base images |
|---|---|---|
| train | COCO-minitrain (train2017 images) | 7,836 |
| val | COCO-minitrain (train2017 images) | 870 |
| test | COCO val2017 | 1,650 |

The splits share no base image. The CSV can be re-derived from the annotations as follows; we checked that these rules reproduce it exactly.

1. **Person filter.** It is applied to every image of `instances_minitrain2017.json` (train/val pool) and of `instances_val2017.json` (test pool). An image is kept if it has at least one `person` annotation and **every** `person` box (crowd boxes included) has a height ≥ 50 px in original image coordinates. This keeps 8,706 of the 25,000 minitrain images and 1,750 of the 5,000 val2017 images.
2. **Test exclusions.** 100 val2017 images were excluded when the dataset was built, which leaves 1,650. Their ids are listed in [`datasets/cocominiperson_d/val2017_excluded_ids.txt`](datasets/cocominiperson_d/val2017_excluded_ids.txt).
3. **Train/val split.** The 8,706 filtered minitrain images are sorted by image id. The 870 lowest ids (the first 10%, rounded down) form `val`, and the remaining 7,836 form `train`. The split is deterministic and has no random component.

Ground truth is the COCO `person` boxes of these images, converted to normalized YOLO format (class 0). No other class is used.

## 3. Pristine images

The following steps are applied to each base image, in this order.

1. Decode the original COCO JPEG with PIL and convert it to BGR. Re-encode it as JPEG with OpenCV defaults (`cv2.imwrite`, quality 95).
2. Read that JPEG with `cv2.imread` (3-channel BGR). Resize it so that the **longest side is 640 px**, using `cv2.INTER_LINEAR`, with `r = 640 / max(h, w)` and new size `(min(ceil(w·r), 640), min(ceil(h·r), 640))`. Images whose longest side is already 640 px are left unchanged; smaller images are upscaled.
3. Save the result as a PNG. This is the **pristine** variant.

## 4. Degradations

There are **17 corruption types** from Hendrycks & Dietterich (ICLR 2019), each at **severities 1–5**, with the original parameter tables:

- noise: `gaussian_noise`, `shot_noise`, `impulse_noise`, `speckle_noise`;
- blur: `gaussian_blur`, `defocus_blur`, `glass_blur`, `motion_blur`;
- weather: `fog`, `frost`, `snow`, `spatter`;
- color: `brightness`, `contrast`, `saturate`;
- digital: `pixelate`, `jpeg_compression`.

Each corruption is applied to the pristine PNG (RGB, `ToTensor` → PIL → corruption → `uint8`) and the result is stored as PNG.

- **Per base image:** 1 pristine + 17 × 5 degraded = **86 distinct images**; 890,616 in total.
- **Database index:** the reference database also indexes a `severity 0` alias of the pristine image under each type, which gives 103 entries per base image and **1,066,668** indexed variants (train 807,108 / val 89,610 / test 169,950).
- **Variants used in the paper:** 52 per base image (pristine + 17 types × severities 3, 4, 5), i.e. 407,472 / 45,240 / 85,800 variants for train / val / test.

**Regimes.** A name such as `fog_3_5` denotes the regime that pools severities **3, 4 and 5 (inclusive)** of one type. Together with `pristine`, this gives **18 regimes** per split. The held-out experiments (leave one degradation type out: Gaussian noise, glass blur, fog) remove one type's regime from training and validation. They keep the same image splits.

## 5. Regenerating the images

`src/dobi/degradation/generator.py` contains the corruption functions with explicit parameters and an explicit random state:

```python
import numpy as np
from dobi.degradation import generator as G

rgb = G.load_rgb("pristine/000000056752.png")          # step 3 output
for typ in ["contrast", "fog", "glass_blur", "spatter"]:  # the 17 corruption names of the dataset
    for sev in range(1, 6):
        seed = G.seed_of("COCOMiniPerson-D", typ, 56752, sev)
        out = G.apply_dataset_degradation(rgb, typ, sev, seed)   # uint8 RGB
        G.save_png_x(f"{typ}/deg_{sev}/000000056752.png", out)
```

- `apply_dataset_degradation` takes the dataset names. Internally `spatter` severities 1–3 are the type `spatter_water` and 4–5 `spatter_mud` (one function, one parameter table).
- `glass_blur` has integer parameters and is defined for integer severities only; it follows the ImageNet-C / imagecorruptions implementation exactly (bitwise equal for the same seed).
- Folder layout per split: `data/<split>2017/images/pristine/<id>.png`, `data/<split>2017/images/<type>/deg_<s>/<id>.png`, and the ground truth `data/<split>2017/labels/<id>.txt` (YOLO format, class 0).
- `src/dobi/data/database/detect_variants.py` runs the frozen detector on every image and writes `data/<split>2017/predictions/<type>/deg_<s>/YOLO_format/<id>.txt` (see the README, "Train and evaluate the main study").
- `src/dobi/data/database/generate_db.py` indexes images, labels and detections into SQLite.
- `generate_subsets.py` builds the 18 regimes from `configs/regime_subsets.json`, e.g. `{"deg_name": "fog", "severity_min": 3, "severity_max": 5}`.

## 6. What is bitwise reproducible

| Part | Status |
|---|---|
| Base-image list and splits | **Exact.** Fully determined by the CSV; the rules in §2 reproduce it. |
| Pristine PNGs (§3) | **Bitwise identical** in our checks: 36 images across all splits, including 12 that needed resizing. Tested with OpenCV 4.13.0 and Pillow 12.0.0. Other JPEG/PNG codec versions may differ. |
| Deterministic types: `brightness`, `contrast`, `saturate`, `gaussian_blur`, `defocus_blur`, `pixelate`, `jpeg_compression` | **Bitwise identical** to the stored images: 3,500/3,500 (7 types × 100 train images × severities 1–5), plus 108 spot checks across all splits. |
| Stochastic types: noise ×4, `fog`, `frost`, `snow`, `spatter`, `motion_blur` (random angle), `glass_blur` | **Not bitwise reproducible.** The original run drew from the global NumPy random state without a seed, in parallel worker processes. The public generator reproduces the **distribution**, not the pixels: mean PSNR against pristine agrees within 0.5 dB in 44 of 45 type × severity cells; spatter severity 1 differs by 1.5 dB because only a few droplets appear. For `glass_blur` (6 test images × severities 3–5) the mean PSNR of regenerated and stored images agrees within 0.03 dB. |

Results in the paper refer to the stored images. A regenerated COCOMiniPerson-D is therefore identical for the image list, the pristine images and the 7 deterministic types. For the stochastic types it is a statistically equivalent re-draw, so numbers on those regimes may differ slightly.

## 7. BDD100K (reference only)

The natural-condition study uses BDD100K (Yu et al., CVPR 2020) and its 2018 detection labels: `bdd100k_labels_images_train.json` and `bdd100k_labels_images_val.json`. Obtain the dataset from the official site (https://bdd-data.berkeley.edu/) under its license terms. This repository contains no BDD100K images or labels, and no BDD100K images are shown in the paper.

# Third-party notices

## Code adapted from third parties

| File | Origin | License | What |
|---|---|---|---|
| `src/dobi/utils/img_utils.py` | Ultralytics, `ultralytics/utils/patches.py` (`imread`) and `ultralytics/data/base.py` (`BaseDataset.load_image`), https://github.com/ultralytics/ultralytics | AGPL-3.0 | `imread` copied; `load_image` adapted (resize of the long side to the input size). The file is distributed under AGPL-3.0-or-later. |
| `src/dobi/data/dataloader/roc_curves_data_loader.py` | Ultralytics, `ultralytics/utils/patches.py` and `ultralytics/data/base.py` | AGPL-3.0 | `imread` and the image cache / resize methods (`_load_and_resize`, attributes `ims`, `im_hw0`, `im_hw`) adapted from `BaseDataset`. The file is distributed under AGPL-3.0-or-later. |
| `src/dobi/degradation/generator.py` | ImageNet-C corruptions, Hendrycks & Dietterich (https://github.com/hendrycks/robustness); size-generalized variant (https://github.com/asharakeh/probdet, https://github.com/bethgelab/imagecorruptions) | Apache-2.0 | corruption functions, parameter tables and helpers, modified (see `NOTICE` and the file header) |

## Data loaded at run time

| Data | Source | License |
|---|---|---|
| Frost textures `frost1.png` … `frost6.jpg` | Python package `imagecorruptions` 1.1.2 (checked by SHA-256 in `generator.py`) | Apache-2.0 |

## Runtime dependencies

| Package | License | Used for |
|---|---|---|
| ultralytics | AGPL-3.0 | YOLO detector, feature extraction, detector inference (AGPL part of this repository only) |
| torch, torchvision | BSD-3-Clause | models, training |
| numpy, scipy, pandas, scikit-learn, scikit-image | BSD-3-Clause | numerics, statistics |
| matplotlib | Matplotlib license (PSF-based) | figures |
| opencv-python | Apache-2.0 | image I/O and resizing |
| Pillow | MIT-CMU (HPND) | image I/O |
| PyYAML | MIT | configuration |
| tqdm | MPL-2.0 and MIT | progress bars |
| Wand (Python binding) | MIT | motion blur and snow in the degradation generator |
| ImageMagick (system library used by Wand) | ImageMagick License | motion blur and snow in the degradation generator |
| imagecorruptions | Apache-2.0 | frost textures |
| pytest | MIT | tests |

## Weights and datasets (not included)

| Item | Source | License / terms |
|---|---|---|
| YOLOv10-N, YOLOv10-M, YOLO11-M detector weights | Ultralytics releases (see `weights/README.md`) | AGPL-3.0 (Ultralytics) |
| COCO-2017 images and annotations | https://cocodataset.org | annotations CC BY 4.0; images Flickr terms of use |
| BDD100K | https://bdd-data.berkeley.edu | BDD100K license (research use) |

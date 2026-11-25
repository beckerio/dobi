# Weights

No weights are stored in this repository.

## Detectors (Ultralytics releases, AGPL-3.0)

The frozen detectors are the unchanged official Ultralytics weights. Download them into `weights/`:

| File | URL | SHA-256 |
|---|---|---|
| `yolov10n.pt` (main study) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt | `11287ed0735678e7ba1ac2a9b3098c049155b3fde123992e724c1264bcc16b6f` |
| `yolov10m.pt` (medium-detector study) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt | `6dc78f7a88591cec1e8716b8f5c7e3aefa9206684f025d202be34439ccb329a0` |
| `yolo11m.pt` (medium-detector study) | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |

```bash
mkdir -p weights && cd weights
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt
sha256sum yolov10n.pt   # must match the table
```

## Monitor heads of the main study (GitHub release)

The 12 monitor heads of the main study (YOLOv10-N, COCOMiniPerson-D) are attached to the GitHub release
(https://github.com/beckerio/dobi/releases). Each file holds the head weights (`monitor_state_dict`), its training
arguments (`train_args`, data locations as placeholders), the threshold grid (`tau_grid`) and the training regimes
(`subsets_info`); optimizer state and loss history were removed. The head weights are bitwise identical to the
checkpoints used for the paper. Load with `torch.load(path, map_location="cpu", weights_only=True)`.

| File | Head | SHA-256 |
|---|---|---|
| `tau_pchip_s0.pt` | PCHIP-16 (DOBI), seed 0 | `ce586a2d6a5be3e624cd28651a161e1a0709defc99c811945032a8a0f573b64d` |
| `tau_pchip_s1.pt` | PCHIP-16 (DOBI), seed 1 | `d63fd6604655c42d397d63faa7d79568d68e916ce83eabd681631ec004bfb31e` |
| `tau_pchip_s2.pt` | PCHIP-16 (DOBI), seed 2 | `08128db3fc49776bc2b9cad85ad6b1f121a4390f7ce740b1e19d1639ff10a3e1` |
| `tau_linear_s0.pt` | linear interpolation, seed 0 | `2f0d0b992058e34ee62c7c68e2f08e8d31f4d3c8c4288e7f763ee15acf3e33aa` |
| `tau_linear_s1.pt` | linear interpolation, seed 1 | `4d22fa49babd0939380d21a1c7753e307c8106e4e7fa1b368d4178a9bb618522` |
| `tau_linear_s2.pt` | linear interpolation, seed 2 | `fde09c363677f90ce40af687d72e96ff55a092d12d13720b5c5612de55d9777d` |
| `pc1_behavior_s0.pt` | PC1 behavior baseline, seed 0 | `90147fa54cebabda6a653a4d196d83ee37b38843cdca8812a189b1d8bc9c44ff` |
| `pc1_behavior_s1.pt` | PC1 behavior baseline, seed 1 | `c3da4f91308a9be793455e678271e8cf4d9b5e3c30aa30b90a0a21e2b5e30342` |
| `pc1_behavior_s2.pt` | PC1 behavior baseline, seed 2 | `f9b03e20b02776e5784a0766312416a82e9ef1fca33ca68da1639594d055b9d4` |
| `lodo_gaussian_noise_pchip_s0.pt` | PCHIP-16, gaussian noise held out in training | `cd1d1166765e2cf883efc9804fb6817974d7a594bcfabbe189866f64e0781c45` |
| `lodo_glass_blur_pchip_s0.pt` | PCHIP-16, glass blur held out in training | `5ea0a8a42c7930b94e438bcdeaa62e20fe818de7750ae7357cd8ed82a6c76723` |
| `lodo_fog_pchip_s0.pt` | PCHIP-16, fog held out in training | `d928980c96e3ad88110969418e082d66cc0222327b6ddf3565148749ffe0d1b3` |

The monitor heads of the follow-up and continuous-probing studies will be published together with their code.

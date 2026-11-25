# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
# SPDX-FileCopyrightText: Dan Hendrycks, Thomas Dietterich
# DOBI degradation generator (the generator validated for the continuous-probing experiments of the paper).
"""Image degradations at continuous strength.

The corruption functions, their parameter tables and the helpers disk/plasma_fractal/clipped_zoom/MotionImage are
modified copies of the ImageNet-C corruption code by Dan Hendrycks and Thomas Dietterich
(https://github.com/hendrycks/robustness, Apache-2.0) in the variant generalised to arbitrary image sizes
(as in https://github.com/asharakeh/probdet by Ali Harakeh and Steven Waslander, Apache-2.0, and
https://github.com/bethgelab/imagecorruptions, Apache-2.0). See NOTICE.

Modifications by the DOBI authors (Apache-2.0, section 4(b)):
  * the parameter vector c is passed explicitly instead of c = TABLE[severity-1];
  * every random draw uses the passed numpy RandomState `rs` instead of the global np.random
    (skimage random_noise gets an integer seed drawn from rs);
  * disk(): for radius > 8 the grid is arange(-ceil(r), ceil(r)+1) (identical to the original for integer r;
    for non-integer r the original grid would not be centred on 0).
  * glass_blur added (from imagecorruptions, random state explicit; integer strengths only);
  * apply_one/apply_condition/apply_dataset_degradation, the continuous parameter c(s) (param) and the per-image
    seeding (seed_of) are new.
Input/output path (identical to the generator that produced COCOMiniPerson-D):
  uint8 RGB -> ToTensor -> tensor_to_pil -> fn -> pil_or_np_to_tensor -> (t*255).clip(0,255).astype(uint8).
Frost textures are read from the installed `imagecorruptions` package (files frost1-6, SHA-256 in FROST_SHA256).
"""
from __future__ import annotations

import ctypes
import hashlib
import math
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import skimage as sk
import torch
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from torchvision import transforms
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

FROST_SHA256 = {
    "frost1.png": "ff9f907860bd7a835d459e32f9d588062b7f61ee267343cc7222b56753a14755",
    "frost2.png": "fe211a89b336999c207a852ce05818d4545d0b57c5beadd824b4cc9d9a9b6137",
    "frost3.png": "2d0d50b4a9bb213f38b024ef7768731bb83cc08d2f26b5766bbc167cdfa0e504",
    "frost4.jpg": "3f8b91ca1a9fa7167b09e773da53f5ae60d0a1fd88f02a783f6e328a72887f6e",
    "frost5.jpg": "5fc6a19df4a429ba68abdcc8f8a4278d4c9f81c9ccafd2c92ab0c8cf8992ebd2",
    "frost6.jpg": "1f92b2f48408748085b68dd81d816ef239f42cef3029c25d041fcb6760fb4f25",
}


def _frost_dir() -> str:
    """Frost textures of the `imagecorruptions` package (pip install imagecorruptions==1.1.2), checked by SHA-256."""
    import importlib.util
    spec = importlib.util.find_spec("imagecorruptions")
    if spec is None or spec.origin is None:
        raise RuntimeError("the frost degradation needs the textures of the package imagecorruptions (pip install imagecorruptions)")
    d = Path(spec.origin).parent / "frost"
    for name, sha in FROST_SHA256.items():
        if hashlib.sha256((d / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"unexpected frost texture {d / name} (SHA-256 differs)")
    return str(d)


FROST_DIR = None  # resolved on first use by frost()

# ---------------------------------------------------------------- tables (copied, hendrycks.py)
TABLE = {
    "gaussian_noise": [.08, .12, 0.18, 0.26, 0.38],
    "shot_noise": [60, 25, 12, 5, 3],
    "impulse_noise": [.03, .06, .09, 0.17, 0.27],
    "speckle_noise": [.15, .2, 0.35, 0.45, 0.6],
    "gaussian_blur": [1, 2, 3, 4, 6],
    "glass_blur": [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)],  # sigma, max_delta, iterations
    "defocus_blur": [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)],
    "motion_blur": [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)],
    "fog": [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)],
    "frost": [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)],
    "snow": [(0.1, 0.3, 3, 0.5, 10, 4, 0.8), (0.2, 0.3, 2, 0.5, 12, 4, 0.7), (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
             (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65), (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)],
    "spatter": [(0.65, 0.3, 4, 0.69, 0.6, 0), (0.65, 0.3, 3, 0.68, 0.6, 0), (0.65, 0.3, 2, 0.68, 0.5, 0),
                (0.65, 0.3, 1, 0.65, 1.5, 1), (0.67, 0.4, 1, 0.65, 1.5, 1)],
    "contrast": [0.4, .3, .2, .1, .05],
    "brightness": [.1, .2, .3, .4, .5],
    "saturate": [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)],
    "jpeg_compression": [25, 18, 15, 10, 7],
    "pixelate": [0.6, 0.5, 0.4, 0.3, 0.25],
}
# type -> (hendrycks function, range lo, range hi)   (PROTOCOL_FINAL §2.1)
TYPES = {
    "brightness": ("brightness", 1, 5), "contrast": ("contrast", 1, 5), "defocus_blur": ("defocus_blur", 1, 5),
    "fog": ("fog", 1, 5), "frost": ("frost", 1, 5), "gaussian_blur": ("gaussian_blur", 1, 5),
    "gaussian_noise": ("gaussian_noise", 1, 5), "impulse_noise": ("impulse_noise", 1, 5),
    "glass_blur": ("glass_blur", 1, 5),  # integer strengths only (integer parameters)
    "jpeg_compression": ("jpeg_compression", 1, 5), "motion_blur": ("motion_blur", 1, 5),
    "pixelate": ("pixelate", 1, 5), "saturate": ("saturate", 3, 5), "shot_noise": ("shot_noise", 1, 5),
    "snow": ("snow", 1, 5), "spatter_water": ("spatter", 1, 3), "spatter_mud": ("spatter", 4, 5),
    "speckle_noise": ("speckle_noise", 1, 5),
}
DETERMINISTIC = ["contrast", "brightness", "saturate", "gaussian_blur", "defocus_blur", "pixelate", "jpeg_compression"]
GROUP = {"fog": 0, "frost": 0, "snow": 0, "spatter_water": 0, "spatter_mud": 0, "brightness": 1, "contrast": 1,
         "saturate": 1, "defocus_blur": 2, "gaussian_blur": 2, "glass_blur": 2, "motion_blur": 2, "gaussian_noise": 3, "shot_noise": 3,
         "impulse_noise": 3, "speckle_noise": 3, "pixelate": 4, "jpeg_compression": 4}


def canonical_order(types):
    return sorted(types, key=lambda t: (GROUP[t], t))


def param(fn: str, s: float):
    """c(s): table value at integer s, componentwise linear interpolation between floor/ceil otherwise
    (shot_noise: log-linear; jpeg quality rounded inside the function)."""
    tab = TABLE[fn]
    if fn == "glass_blur" and abs(s - round(s)) >= 1e-12:
        raise ValueError("glass_blur has integer parameters and is defined for integer strengths 1-5 only")
    if abs(s - round(s)) < 1e-12:
        return tab[int(round(s)) - 1]
    i = int(math.floor(s))
    a = s - i
    lo, hi = np.asarray(tab[i - 1], np.float64), np.asarray(tab[i], np.float64)
    if fn == "shot_noise":
        v = math.exp((1 - a) * math.log(float(lo)) + a * math.log(float(hi)))
        return v
    v = (1 - a) * lo + a * hi
    return float(v) if v.ndim == 0 else tuple(float(x) for x in v)


# ---------------------------------------------------------------- helpers (copied)
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        R = int(math.ceil(radius))
        L = np.arange(-R, R + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def plasma_fractal(rs, mapsize=256, wibbledecay=3):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * rs.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize_:stepsize, 0:mapsize_:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[0], img.shape[1]
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    side = (w - cw) // 2
    img = scizoom(img[top:top + ch, side:side + cw], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    trim_side = (img.shape[1] - w) // 2
    return img[trim_top:trim_top + h, trim_side:trim_side + w]


# ---------------------------------------------------------------- corruptions (copied; c and rs explicit)
def gaussian_noise(x, c, rs):
    x = np.array(x) / 255.
    return np.clip(x + rs.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, c, rs):
    x = np.array(x) / 255.
    return np.clip(rs.poisson(x * c) / float(c), 0, 1) * 255


def impulse_noise(x, c, rs):
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c, rng=int(rs.randint(2 ** 31 - 1)))
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, c, rs):
    x = np.array(x) / 255.
    return np.clip(x + x * rs.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, c, rs):
    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, c, rs):
    """ImageNet-C glass blur (Hendrycks & Dietterich), size-generalised as in imagecorruptions; c = (sigma, max_delta,
    iterations). The local pixel exchange keeps the assignment semantics of the reference code exactly (with numpy views
    the right-hand side is evaluated before the first assignment, so x[h, w] takes the value of x[h', w'] and
    x[h', w'] keeps it), and draws the offsets from `rs` in the same order as the reference draws from np.random."""
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)
    x_shape = np.array(x).shape
    for i in range(c[2]):
        for h in range(x_shape[0] - c[1], c[1], -1):
            for w in range(x_shape[1] - c[1], c[1], -1):
                dx, dy = rs.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, c, rs):
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])
    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255


def motion_blur(x, c, rs):
    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())
    x.motion_blur(radius=c[0], sigma=c[1], angle=rs.uniform(-45, 45))
    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    if len(x.shape) != 2:
        return np.clip(x[..., [2, 1, 0]], 0, 255)
    return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def fog(x, c, rs):
    x = np.array(x) / 255.
    max_val = x.max()
    fractal = cv2.resize(plasma_fractal(rs, wibbledecay=c[1]), (x.shape[1], x.shape[0]))
    x += c[0] * fractal[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, c, rs):
    global FROST_DIR
    FROST_DIR = FROST_DIR or _frost_dir()
    idx = rs.randint(5)
    filename = [f"{FROST_DIR}/frost1.png", f"{FROST_DIR}/frost2.png", f"{FROST_DIR}/frost3.png",
                f"{FROST_DIR}/frost4.jpg", f"{FROST_DIR}/frost5.jpg", f"{FROST_DIR}/frost6.jpg"][idx]
    frost_im = cv2.imread(filename)
    frost_im = cv2.resize(frost_im, x.size)
    frost_im = frost_im[..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost_im, 0, 255)


def snow(x, c, rs):
    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = rs.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())
    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=rs.uniform(-135, -45))
    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, c, rs):
    x = np.array(x, dtype=np.float32) / 255.
    liquid_layer = rs.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]), 238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    m = np.where(liquid_layer > c[3], 1, 0)
    m = gaussian(m.astype(np.float32), sigma=c[4])
    m[m < 0.8] = 0
    color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]), 42 / 255. * np.ones_like(x[..., :1]),
                            20 / 255. * np.ones_like(x[..., :1])), axis=2)
    color *= m[..., np.newaxis]
    x *= (1 - m[..., np.newaxis])
    return np.clip(x + color, 0, 1) * 255


def contrast(x, c, rs):
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, c, rs):
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def saturate(x, c, rs):
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, c, rs):
    output = BytesIO()
    x.save(output, 'JPEG', quality=int(round(c)))
    x = PILImage.open(output)
    return np.array(x)


def pixelate(x, c, rs):
    h_max, w_max = x.size[1], x.size[0]
    x = x.resize((int(w_max * c), int(h_max * c)), PILImage.BOX)
    x = x.resize((w_max, h_max), PILImage.BOX)
    return np.array(x)


FUNCS = {f.__name__: f for f in (gaussian_noise, shot_noise, impulse_noise, speckle_noise, gaussian_blur, glass_blur, defocus_blur,
                                  motion_blur, fog, frost, snow, spatter, contrast, brightness, saturate,
                                  jpeg_compression, pixelate)}


# ---------------------------------------------------------------- conversion between the tensor path and PIL/numpy
def tensor_to_pil(image: torch.Tensor):
    """CHW (or HWC) float tensor in [0, 1] -> 8-bit PIL image (values rounded half to even)."""
    t = image.detach().cpu()
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    a = np.rint(t.clamp(0, 1).numpy() * 255.0)
    return PILImage.fromarray(a.astype(np.uint8))


def pil_or_np_to_tensor(img) -> torch.Tensor:
    """PIL image or HW/HWC array with values in [0, 255] -> CHW float32 tensor in [0, 1]."""
    a = np.asarray(img).astype(np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., np.newaxis]
    return torch.from_numpy(a).permute(2, 0, 1)


_TT = transforms.ToTensor()


def seed_of(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:4], "little")


def apply_one(rgb_u8: np.ndarray, typ: str, s: float, seed: int) -> np.ndarray:
    """One degradation at continuous strength s, exactly the original tensor path; returns uint8 RGB."""
    fn = TYPES[typ][0]
    c = param(fn, s)
    rs = np.random.RandomState(seed)
    t = _TT(rgb_u8)
    out = pil_or_np_to_tensor(FUNCS[fn](tensor_to_pil(t.clone()), c, rs))
    a = out.numpy().transpose(1, 2, 0)
    a = (a * 255).clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_RGBA2RGB)
    return a


# corruption names of COCOMiniPerson-D (17 types, severities 1-5) -> generator type
def dataset_type(name: str, severity: int) -> str:
    """'spatter' severities 1-3 are the water variant, 4-5 the mud variant (one function, one parameter table)."""
    if name == "spatter":
        return "spatter_water" if severity <= 3 else "spatter_mud"
    if name not in TYPES:
        raise KeyError(name)
    return name


def apply_dataset_degradation(rgb_u8: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    """One COCOMiniPerson-D variant: corruption `name` (e.g. 'fog', 'glass_blur', 'spatter') at integer severity 1-5."""
    return apply_one(rgb_u8, dataset_type(name, int(severity)), float(int(severity)), seed)


def apply_condition(rgb_u8: np.ndarray, cond: dict, image_id: int) -> np.ndarray:
    """cond: {'id', 'steps': [[type, s], ...]} (steps already in canonical order). pristine: no steps."""
    x = rgb_u8
    for k, (typ, s) in enumerate(cond["steps"]):
        x = apply_one(x, typ, float(s), seed_of("MF1", cond["id"], image_id, k))
    return x


def load_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def save_png_x(path: str, rgb_u8: np.ndarray):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    assert ok
    with open(path, "xb") as f:
        f.write(buf.tobytes())

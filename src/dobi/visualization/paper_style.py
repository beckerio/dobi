# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Common publication style for all DOBI paper figures.

Every quantitative paper figure is produced by scripts/make_paper_figures.py and uses this module,
so terminology, colours, line styles and sizes are identical across figures. Critical distinctions
are encoded by line style / marker as well as colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# column widths of the two-column article layout (inches)
COL_W = 3.35
FULL_W = 7.0

# terminology (identical in figures, tables and text)
NAME = {
    "gt": "Empirical (regime)",
    "fixed_pristine": "Fixed pristine",
    "mean_behavior": "Mean behavior",
    "adaptive_linear": "Linear",
    "adaptive_pchip": "Adaptive PCHIP",
    "pc1_behavior": "PC1 behavior",
    "regime_id_validation": "Regime-ID",
    "pchip_regime_mean_tau": "PCHIP regime mean",
    "regime_oracle": "Regime oracle",
}

# colour + line style + marker per role (validated categorical palette, see dataviz check)
STYLE = {
    "gt": dict(color="#1f1f1f", ls="-", lw=1.9, marker=None),
    "adaptive_pchip": dict(color="#009682", ls="-", lw=1.4, marker="o"),
    "adaptive_linear": dict(color="#4664aa", ls="--", lw=1.1, marker="^"),
    "mean_behavior": dict(color="#a07a1e", ls="-.", lw=1.1, marker="D"),
    "pc1_behavior": dict(color="#7b52ae", ls=(0, (3, 1, 1, 1)), lw=1.1, marker="v"),
    "fixed_pristine": dict(color="#a01e28", ls="--", lw=1.2, marker="s"),
    "regime_oracle": dict(color="#dca01e", ls=":", lw=1.6, marker="*"),
    "regime_id_validation": dict(color="#6e6e6e", ls=(0, (1, 1)), lw=1.1, marker="x"),
    "pchip_regime_mean_tau": dict(color="#50aae6", ls="-", lw=1.0, marker="P"),
}

REGIME_LABEL = {
    "pristine": "Pristine", "brightness_3_5": "Brightness", "contrast_3_5": "Contrast",
    "defocus_blur_3_5": "Defocus blur", "fog_3_5": "Fog", "frost_3_5": "Frost", "gaussian_blur_3_5": "Gaussian blur",
    "gaussian_noise_3_5": "Gaussian noise", "glass_blur_3_5": "Glass blur", "impulse_noise_3_5": "Impulse noise",
    "jpeg_compression_3_5": "JPEG", "motion_blur_3_5": "Motion blur", "pixelate_3_5": "Pixelate",
    "saturate_3_5": "Saturate", "shot_noise_3_5": "Shot noise", "snow_3_5": "Snow", "spatter_3_5": "Spatter",
    "speckle_noise_3_5": "Speckle noise",
}


def use_paper_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
    })


def line_kw(role, **over):
    s = dict(STYLE[role])
    s.pop("marker", None)
    s.update(over)
    return s


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def save(fig, out: Path, name: str, png_dpi: int = 300):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=png_dpi)
    plt.close(fig)

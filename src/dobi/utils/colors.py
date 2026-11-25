# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Colour cycle used by the DOBI plots (RGB tuples in [0, 1])."""
from __future__ import annotations

import itertools

RED = (160 / 255, 30 / 255, 40 / 255)
GREEN = (0, 150 / 255, 130 / 255)
BLUE = (70 / 255, 100 / 255, 170 / 255)
CYAN = (80 / 255, 170 / 255, 230 / 255)
MAGENTA = (160 / 255, 0, 120 / 255)
ORANGE = (220 / 255, 160 / 255, 30 / 255)
YELLOW = (250 / 255, 230 / 255, 20 / 255)
LIGHT_GREEN = (130 / 255, 190 / 255, 60 / 255)
BROWN = (160 / 255, 130 / 255, 50 / 255)

# order of the default colour cycle
COLOR_CYCLE = (RED, GREEN, BLUE, CYAN, MAGENTA, ORANGE, YELLOW, LIGHT_GREEN)


def cycle_colors():
    """Endless iterator over COLOR_CYCLE."""
    return itertools.cycle(COLOR_CYCLE)

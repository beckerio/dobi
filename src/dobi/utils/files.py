# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Small file-system helpers."""
from __future__ import annotations

import re
from pathlib import Path


def next_run_dir(base) -> Path:
    """Return `base` if it does not exist yet, otherwise the first free sibling `<base>N`.

    N is one more than the largest number already used by a sibling `<base>N` (starting at 2),
    e.g. runs/exp -> runs/exp2 -> runs/exp3. The directory itself is not created.
    """
    base = Path(base)
    if not base.exists():
        return base
    pattern = re.compile(re.escape(base.name) + r"(\d+)$")
    used = [int(m.group(1)) for p in base.parent.iterdir() if (m := pattern.match(p.name))]
    return base.parent / f"{base.name}{max(used) + 1 if used else 2}"

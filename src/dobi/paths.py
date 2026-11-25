# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Data locations for DOBI.

Code asks for locations by key:  from dobi.paths import P;  P("dobi_db"), P("tau_behavior_models", "feature_cache").

Resolution order for a key:
  1. a YAML file with a top-level `paths:` mapping, named by $DOBI_PATHS, else configs/paths.yaml in the repository
     (copy configs/paths_TEMPLATE.yaml; the file is not tracked);
  2. otherwise the default location below $DOBI_DATA_ROOT (default: <repository>/data), see DEFAULTS.
`repo_root` is always the repository root. Looking up a key never fails; a missing file only fails when it is opened.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# key -> location relative to the data root
DEFAULTS = {
    "data_root": "",
    "dobi_db": "dobi.db",
    "databases_dir": "",
    "coco_mini_person_d": "COCOMiniPerson-D",
    "tau_behavior_models": "TauBehavior",
    "rq5_data_exp1": "rq5_exp1",
    "rq5_data_exp2": "rq5_exp2",
    "bdd100k_root": "bdd100k",
}


def data_root() -> str:
    return os.path.expanduser(os.environ.get("DOBI_DATA_ROOT") or str(REPO_ROOT / "data"))


@lru_cache(maxsize=1)
def _config() -> dict:
    f = os.environ.get("DOBI_PATHS") or str(REPO_ROOT / "configs" / "paths.yaml")
    if not Path(f).is_file():
        return {}
    import yaml
    with open(f) as fh:
        cfg = yaml.safe_load(fh) or {}
    return {k: v for k, v in (cfg.get("paths") or {}).items() if v not in (None, "")}


def P(key: str, *parts: str) -> str:
    """Path for `key` (optionally joined with sub-parts), as str."""
    if key == "repo_root":
        base = str(REPO_ROOT)
    elif key in _config():
        base = os.path.expanduser(str(_config()[key]))
    elif key in DEFAULTS:
        base = os.path.join(data_root(), DEFAULTS[key]) if DEFAULTS[key] else data_root()
    else:
        raise KeyError(f"unknown path key '{key}' (known: repo_root, {', '.join(sorted(DEFAULTS))})")
    return os.path.join(base, *parts) if parts else base


_PLACEHOLDER = re.compile(r"^\$\{([a-z0-9_]+)\}(.*)$")


def resolve(obj):
    """Replace placeholders "${key}" / "${key}/sub/path" in strings (recursively in dicts/lists) by P(key)+sub.
    Used for configs/*.json."""
    if isinstance(obj, dict):
        return {k: resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve(v) for v in obj]
    if isinstance(obj, str):
        m = _PLACEHOLDER.match(obj)
        if m:
            return P(m.group(1)) + m.group(2)
    return obj

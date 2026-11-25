#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Extended DOBI evaluation (experiments A-D). Evaluation only: no training, no change to existing artifacts.

    PYTHONPATH=src python src/dobi/eval/extended/run_extended_eval.py --out paper_artifacts_extended_eval/<run_id> --experiments A B C D

Every experiment refuses to overwrite its own outputs; the output directory must lie below
paper_artifacts_extended_eval/.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from dobi.eval.extended import common as C

MODULES = {"A": "dobi.eval.extended.exp_a_batch", "B": "dobi.eval.extended.exp_b_equivalence",
           "C": "dobi.eval.extended.exp_c_fppi", "D": "dobi.eval.extended.exp_d_consistency"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments", nargs="+", default=list(MODULES), choices=list(MODULES))
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    out = C.check_out_dir(Path(a.out))
    if not out.exists():
        raise SystemExit(f"run directory {out} does not exist (create it with a unique run id first)")
    for e in a.experiments:
        log_path = C.new_file(out / f"logs/experiment_{e}.log")
        with open(log_path, "w") as lf:
            def log(msg, _lf=lf):
                print(msg, flush=True)
                _lf.write(str(msg) + "\n")
                _lf.flush()
            t0 = time.time()
            log(f"command: {C.run_command_line()}")
            log(f"experiment {e} started {time.strftime('%Y-%m-%dT%H:%M:%S')}")
            importlib.import_module(MODULES[e]).run(out, workers=a.workers, log=log)
            log(f"experiment {e} finished in {time.time() - t0:.0f}s")
        with open(out / "reproducibility/commands.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  experiment {e}: PYTHONPATH=src {C.run_command_line()}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Run the training and/or evaluation stage of a paper config (configs/paper_*.json).

Commands are built exactly like training/train_curve_monitor_sweeps.py and
eval/evaluation_curve_monitor_sweeps.py (global settings merged with run settings,
booleans as flags, lists as nargs), but every run gets its own log file and runs can
be executed in parallel. Every command is appended to the reproducibility command log.

    python scripts/run_paper_experiments.py --config configs/paper_main.json --stage train --jobs 3
    python scripts/run_paper_experiments.py --config configs/paper_main.json --stage eval --jobs 3
"""
from __future__ import annotations
from dobi.paths import P as _P  # machine-specific paths (configs/local_paths_<host>.yaml)
from dobi.paths import resolve as _resolve

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(_P("repo_root"))
TRAIN_SCRIPT = "src/dobi/training/train_curve_monitor.py"
EVAL_SCRIPT = "src/dobi/eval/evaluation_curve_monitor.py"


def load_runs(config: Path):
    data = _resolve(json.loads(config.read_text()))   # ${key} placeholders -> configs/local_paths_<hostname>.yaml
    g = data.get("global", {})
    return [{**g, **r} for r in data["runs"]]


def to_flags(cfg: dict, skip=()):
    out = []
    for k, v in cfg.items():
        if k in skip or k.startswith("eval_") or k.startswith("_"):
            continue
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            if v:
                out.append(f"--{k}")
                out.extend(str(x) for x in v)
        else:
            out.extend([f"--{k}", str(v)])
    return out


def train_cmd(cfg):
    if cfg.get("head") == "mean_behavior":
        return None
    return [sys.executable, TRAIN_SCRIPT] + to_flags(cfg)


def ckpt_path(cfg):
    return Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"


def eval_cmd(cfg, out_root: Path, split: str):
    out_csv = out_root / cfg["name"] / "curve_evaluation_results.csv"
    cmd = [sys.executable, EVAL_SCRIPT,
           "--db", cfg["db"], "--images_root", cfg.get("eval_images_root", cfg["images_root"]),
           "--weights", cfg["weights"], "--class_id", str(cfg.get("class_id", 0)), "--iou_thr", str(cfg.get("iou_thr", 0.5)),
           "--u_max", str(cfg.get("u_max", 2.0)), "--imgsz", str(cfg.get("imgsz", 640)), "--split", split,
           "--examples_per_cluster", "0", "--op_mode", "f1", "--eval_n_points", "201", "--output_csv", str(out_csv)]
    if cfg.get("head") == "mean_behavior":
        cmd += ["--head_type", "mean_behavior"]
    else:
        cmd += ["--ckpt", str(ckpt_path(cfg))]
    cmd += to_flags(cfg.get("eval_args", {}))
    return cmd, out_csv


def run(cmd, log: Path, cmdlog: Path, tag: str):
    log.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(shlex.quote(str(c)) for c in cmd).replace(sys.executable, "$PY")
    with open(cmdlog, "a") as f:
        f.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {tag} (log: {log.relative_to(ROOT) if log.is_relative_to(ROOT) else log})\n{line}\n")
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"# command: {line}\n# cwd: {ROOT}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**__import__('os').environ, "PYTHONPATH": "src", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}).returncode
    print(f"[{tag}] rc={rc} {time.time() - t0:.0f}s -> {log}", flush=True)
    return tag, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None, help="Run names to include.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_root", default='runs/main_study/eval')
    ap.add_argument("--log_root", default='runs/main_study/logs')
    ap.add_argument("--cmdlog", default='runs/main_study/commands.sh')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = [r for r in load_runs(Path(args.config)) if args.only is None or r["name"] in args.only]
    cmdlog = ROOT / args.cmdlog
    results = []

    def stage(kind):
        jobs = []
        for cfg in runs:
            if kind == "train":
                cmd = train_cmd(cfg)
                if cmd is None:
                    continue
                if (Path(cfg["project"]) / cfg["name"]).glob("summary_*.json") and any((Path(cfg["project"]) / cfg["name"]).glob("summary_*.json")):
                    print(f"[train {cfg['name']}] already finished, skipping")
                    continue
                log = ROOT / args.log_root / f"train_{cfg['name']}.log"
            else:
                cmd, out_csv = eval_cmd(cfg, ROOT / args.out_root, args.split)
                if out_csv.with_name(out_csv.stem + "_threshold_summary.json").exists():
                    print(f"[eval {cfg['name']}] already evaluated, skipping")
                    continue
                log = ROOT / args.log_root / f"eval_{cfg['name']}.log"
            if args.dry_run:
                print(" ".join(shlex.quote(str(c)) for c in cmd))
                continue
            jobs.append((cmd, log, f"{kind} {cfg['name']} (config {args.config}, seed {cfg.get('seed')})"))
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            results.extend(ex.map(lambda j: run(j[0], j[1], cmdlog, j[2]), jobs))

    if args.stage in ("train", "both"):
        stage("train")
    if args.stage in ("eval", "both"):
        stage("eval")
    failed = [t for t, rc in results if rc != 0]
    print(f"done: {len(results)} jobs, failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

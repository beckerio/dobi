#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""Regenerate the paper tables and figures from the result files in results/ and compare the tables with reference/.

    python make_all_tables_figures.py [--out outputs]

Tables: outputs/tables/*.tex (plus .csv/.md for the main-study tables). Every generated table that the paper uses
is compared byte for byte with reference/tables/. Four paper tables were written by hand from result files and have
no generator; they are listed with their source (see TABLE_SOURCES below).
Figures: outputs/figures/. Figures whose inputs are not part of this repository (curve exports *.npz, see README)
or that show dataset images are listed as skipped.
Exit code 0 if every generated table equals its reference and no step failed for another reason than a missing
input that is not part of this repository; 1 otherwise. The last line reads PASSED or FAILED.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = "results"

# tables of the paper without a generator: transcribed by hand from these result files
TABLE_SOURCES = {
    "tab_fppi_policy.tex": "results/main_study/paper_artifacts_extended_eval/run_20261011_182510_f7b65d/tables/expC_summary.csv (budget 0.1)",
    "tabS_medium_curves.tex": "results/medium_detector/func_summary.csv",
    "tabS_medium_execution.tex": "results/medium_detector/exe.csv",
    "tabS_medium_fppi.tex": "results/medium_detector/expc.csv (rows *_valcal)",
}
# tables written inline in the paper source (no .tex file): content and source
INLINE_TABLES = {
    "S:tab:pc_weighting": "results/medium_detector/A1_pca_by_representation_regimes.csv",
    "S:tab:new_batch": "results/main_study/paper_artifacts_extended_eval/run_20261011_182510_f7b65d/tables/expA_macro_summary.csv",
    "S:tab:supp_fppi_budgets": "results/main_study/paper_artifacts_extended_eval/run_20261011_182510_f7b65d/tables/expC_summary.csv, expC_bootstrap_differences.csv",
    "S:tab:cont_n": "results/rq5_continuous/phase0_median_noise.json (median_noise)",
    "S:tab:regime_names": "static (regime display names)",
    "S:tab:cont_sets": "static (description of the condition sets)",
}
# input of the main-study table S:tab:pc1 that no script in this repository writes
NO_PRODUCER = {"results/main_study/paper_artifacts/aggregate/pc1_baseline_summary.json":
               "PCA baseline summary (explained variance, reconstruction RMSE); producer not part of the repository"}


EXT = f"{R}/main_study/paper_artifacts_extended_eval/run_20261011_182510_f7b65d"
# inputs that are not part of this version of the repository; steps that need them are skipped, not failed
NOT_INCLUDED = {"B": f"{EXT}/metrics/expB_matrices_and_curves.npz"}


def run(cmd, log):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
    p = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, env=env, capture_output=True, text=True)
    log.append({"cmd": " ".join(map(str, cmd)), "returncode": p.returncode, "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]})
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = (ROOT / a.out).resolve()
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    log = []

    # ---------------------------------------------------------------- tables
    run(["experiments/main_study/scripts/make_paper_tables.py", "--paper_results", f"{R}/main_study/paper_artifacts",
         "--execution_results", f"{R}/main_study/paper_artifacts_execution", "--out", tab], log)
    run(["scripts/tables/make_followup_tables.py", f"{R}/rq4_followups", tab], log)
    run(["scripts/tables/make_continuous_tables.py", f"{R}/rq5_continuous", tab, fig / "main"], log)
    run(["scripts/tables/make_natural_tables.py", f"{R}/natural_bdd", tab], log)

    # ---------------------------------------------------------------- figures
    run(["experiments/main_study/scripts/make_paper_figures.py", "--paper_results", f"{R}/main_study/paper_artifacts",
         "--execution_results", f"{R}/main_study/paper_artifacts_execution", "--out", fig, "--all"], log)
    skipped = []
    if log[-1]["returncode"] == 3:   # make_paper_figures.py: some figures need the curve exports (*.npz), not included
        line = [l for l in log[-1]["stdout"].splitlines() if "| skipped:" in l][-1]
        skipped += [f"{n.strip()} (needs results/main_study/paper_artifacts/curve_exports/*.npz)" for n in line.split("| skipped:")[1].split(",")]
        log[-1]["skipped"] = True
    for part in ("A", "B", "C", "D"):
        if part in NOT_INCLUDED and not (ROOT / NOT_INCLUDED[part]).is_file():
            skipped.append(f"extended evaluation {part} figures (needs {NOT_INCLUDED[part]})")
            continue
        run(["experiments/main_study/scripts/make_extended_eval_figures.py", "--run", EXT,
             "--only", part, "--out", fig / "extended"], log)
    (fig / "main").mkdir(parents=True, exist_ok=True)
    run(["scripts/figures/fig3_regime_distance_agreement/make_fig3_regime_distance_agreement_main.py", "main",
         fig / "main" / "figB_regime_distance_agreement_main.pdf"], log)

    # ---------------------------------------------------------------- compare tables
    res = {"equal": [], "different": [], "not_generated": [], "hand_written": {}, "inline": INLINE_TABLES, "no_producer": NO_PRODUCER,
           "skipped_figures": skipped}
    for ref in sorted((ROOT / "reference/tables").glob("*.tex")):
        if ref.name in TABLE_SOURCES:
            res["hand_written"][ref.name] = TABLE_SOURCES[ref.name]
        elif not (tab / ref.name).is_file():
            res["not_generated"].append(ref.name)
        elif filecmp.cmp(ref, tab / ref.name, shallow=False):
            res["equal"].append(ref.name)
        else:
            res["different"].append(ref.name)
    res["figures"] = sorted(str(p.relative_to(fig)) for p in fig.rglob("*") if p.suffix in (".pdf", ".png"))
    res["commands"] = log
    (out / "make_all_report.json").write_text(json.dumps(res, indent=1))
    failed = [c["cmd"] for c in log if c["returncode"] != 0 and not c.get("skipped")]
    n_gen = len(res["equal"]) + len(res["different"]) + len(res["not_generated"])
    print(f"tables: {len(res['equal'])}/{n_gen} identical to reference/tables "
          f"({len(res['hand_written'])} hand-written tables have no generator; sources in {out / 'make_all_report.json'})")
    for k in ("different", "not_generated"):
        if res[k]:
            print(f"  {k}: {', '.join(res[k])}")
    print(f"figures: {len(res['figures'])} files written to {fig}")
    for x in skipped:
        print(f"  skipped, input not included: {x}")
    for c in failed:
        print(f"  FAILED step: {c}")
    ok = not (res["different"] or res["not_generated"] or failed)
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

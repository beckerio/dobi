#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Evaluation-only: true per-image threshold execution for all test image variants (no retraining).

    PYTHONPATH=src python src/dobi/eval/run_execution_eval.py --out paper_artifacts_execution

Policies: fixed_pristine, mean_behavior, adaptive_linear (3 seeds), adaptive_pchip (3 seeds),
pc1_behavior (3 seeds), regime_id_validation, pchip_regime_mean_tau (3 seeds), regime_oracle,
per_image_oracle (exploratory only).
"""
from __future__ import annotations
from dobi.paths import P as _P  # machine-specific paths (configs/local_paths_<host>.yaml)

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dobi.data.feature_cache import FeatureCache, pooled_to_pseudo_feats
from dobi.eval.aggregate_paper_results import jsonable, write_table
from dobi.eval.evaluation_curve_monitor import MeanBehaviorPredictor, _load_monitor_head_from_ckpt
from dobi.eval.execution_eval import aggregate, load_split_detections, metrics_from_counts, per_image_counts, per_image_oracle_tau
from dobi.models.monitor_factory import build_monitor_head, decode_monitor_output
from dobi.utils.curve_plotting import COLORS, FONT_SIZE, PRED_COLOR, PRIMARY_COLOR
from dobi.utils.tau_spline import select_operating_threshold
from dobi.utils.tau_targets import (dedupe_sweep, detector_metrics_at_tau, get_split_subsets, load_raw_sweep,
                                    oracle_threshold, pristine_validation_threshold, sweep_at_tau)

LEARNED = {"adaptive_linear": "tau_linear_s{}", "adaptive_pchip": "tau_pchip_s{}", "pc1_behavior": "pc1_behavior_s{}"}
SEEDS = (0, 1, 2)
MAIN_POLICIES = ["fixed_pristine", "mean_behavior", "adaptive_linear", "adaptive_pchip", "pc1_behavior",
                 "pchip_regime_mean_tau", "regime_id_validation", "regime_oracle"]
LABEL = {"fixed_pristine": "Fixed pristine τ", "mean_behavior": "Mean behavior", "adaptive_linear": "Adaptive linear",
         "adaptive_pchip": "Adaptive PCHIP", "pc1_behavior": "PC1 behavior", "pchip_regime_mean_tau": "PCHIP regime-mean τ",
         "regime_id_validation": "Regime-ID (validation τ)", "regime_oracle": "Regime oracle (test)",
         "per_image_oracle": "Per-image oracle (exploratory)"}
COLOR = {"fixed_pristine": COLORS[0], "mean_behavior": COLORS[2], "adaptive_linear": COLORS[3], "adaptive_pchip": PRED_COLOR,
         "pc1_behavior": COLORS[4], "pchip_regime_mean_tau": COLORS[7], "regime_id_validation": "0.45",
         "regime_oracle": PRIMARY_COLOR}
STRONG = ["glass_blur_3_5", "pixelate_3_5", "gaussian_noise_3_5", "impulse_noise_3_5", "shot_noise_3_5",
          "jpeg_compression_3_5"]


def short(n):
    return n.replace("_3_5", "").replace("_", " ")


# ------------------------------------------------------------------
def predict_taus(ckpt, cache: FeatureCache, variant_ids, tau_grid, device, batch=2048):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ta = ck["train_args"]
    head = build_monitor_head(ta["head"], cache.feat_channels, ta)
    _load_monitor_head_from_ckpt(head, ck)
    head.to(device).eval()
    rows = np.asarray([cache.row_for_variant(v) for v in variant_ids])
    tg = torch.as_tensor(tau_grid, dtype=torch.float32, device=device)
    out = np.empty(len(rows))
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            x = torch.from_numpy(cache.get(rows[s:s + batch])).to(device)
            d = decode_monitor_output(ta["head"], head, head(pooled_to_pseudo_feats(x, cache.feat_channels)), tg, ta.get("u_max", 2.0))
            t, _ = select_operating_threshold(tg.double(), {k: d[k].double() for k in ("f1_tau", "tpr_tau", "precision_tau")}, "f1")
            out[s:s + batch] = t.cpu().numpy()
    return out


def regime_curve_utility(tau, regimes, sweeps_asc):
    """Old metric: F1 of the complete empirical regime curve at each image's threshold."""
    f1 = np.empty(len(tau))
    for r, asc in sweeps_asc.items():
        m = regimes == r
        f1[m] = sweep_at_tau(asc, tau[m])["f1"]
    return f1


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_P('dobi_db'))
    ap.add_argument("--feature_cache", default=_P('tau_behavior_models', 'feature_cache'))
    ap.add_argument("--project", default=_P('tau_behavior_models', 'paper'))
    ap.add_argument("--old_eval_root", default='results/main_study/results_legacy/results/paper/eval')
    ap.add_argument("--old_artifacts", default='results/main_study/paper_artifacts')
    ap.add_argument("--out", default='results/main_study/paper_artifacts_execution')
    ap.add_argument("--n_dense_tau", type=int, default=1001)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=12345)
    a = ap.parse_args()
    out = Path(a.out)
    for d in ("tables", "per_image", "figures", "diagnostics", "reproducibility"):
        (out / d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---------------- detections ----------------
    table, meta = load_split_detections(a.db, "test", 0, 0.5)
    regimes = meta["regime"]
    reg_names = [s["name"] for s in sorted(get_split_subsets(__import__("sqlite3").connect(a.db), "test", 0, 0.5),
                                           key=lambda s: s["subset_id"])]
    print(f"loaded {table.n} variants, {len(table.scores)} detections in {time.time() - t0:.0f}s")

    import sqlite3
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    test_subsets = {s["name"]: s["subset_id"] for s in get_split_subsets(conn, "test", 0, 0.5)}
    val_subsets = {s["name"]: s["subset_id"] for s in get_split_subsets(conn, "val", 0, 0.5)}
    raw = {r: load_raw_sweep(conn, sid, 0, 0.5) for r, sid in test_subsets.items()}
    asc = {r: dedupe_sweep(v) for r, v in raw.items()}
    val_raw = {r: load_raw_sweep(conn, sid, 0, 0.5) for r, sid in val_subsets.items()}
    conn.close()

    # ---------------- policies (thresholds) ----------------
    tau_grid = np.linspace(0, 1, a.n_dense_tau, dtype=np.float32)
    tau_pristine, _ = pristine_validation_threshold(a.db, 0, 0.5, "val")
    mb = MeanBehaviorPredictor(a.db, 0, 0.5, tau_grid, split="train")
    curves = {k: torch.as_tensor(v, dtype=torch.float64)[None] for k, v in mb.curves.items()}
    tau_mb = float(select_operating_threshold(torch.as_tensor(tau_grid, dtype=torch.float64), curves, "f1")[0][0])
    tau_oracle = {r: oracle_threshold(raw[r], "f1") for r in reg_names}
    tau_regval = {r: oracle_threshold(val_raw[r], "f1") for r in reg_names}

    policies = {  # name -> {seed: tau array}
        "fixed_pristine": {None: np.full(table.n, tau_pristine)},
        "mean_behavior": {None: np.full(table.n, tau_mb)},
        "regime_oracle": {None: np.array([tau_oracle[r] for r in regimes])},
        "regime_id_validation": {None: np.array([tau_regval[r] for r in regimes])},
    }
    cache = FeatureCache(a.feature_cache, "test")
    for pol, pat in LEARNED.items():
        policies[pol] = {}
        for s in SEEDS:
            ck = Path(a.project) / pat.format(s) / "weights/best.pt"
            policies[pol][s] = predict_taus(ck, cache, table.variant_ids, tau_grid, device)
            print(f"{pol} seed {s}: tau range [{policies[pol][s].min():.3f}, {policies[pol][s].max():.3f}]")
    policies["pchip_regime_mean_tau"] = {}
    for s in SEEDS:
        t = policies["adaptive_pchip"][s]
        rm = {r: float(t[regimes == r].mean()) for r in reg_names}
        policies["pchip_regime_mean_tau"][s] = np.array([rm[r] for r in regimes])
    policies["per_image_oracle"] = {None: per_image_oracle_tau(table)}

    # ---------------- sanity checks ----------------
    checks = {}
    checks["score_floor"] = {"min_stored_score": float(table.scores.min()), "max_stored_score": float(table.scores.max())}
    learned_all = np.concatenate([v for p in LEARNED for v in policies[p].values()] + [policies["fixed_pristine"][None],
                                  policies["mean_behavior"][None], policies["regime_id_validation"][None]])
    checks["score_floor"].update(min_selected_tau=float(learned_all.min()), max_selected_tau=float(learned_all.max()),
                                 n_selected_tau_below_floor=int((learned_all < table.scores.min()).sum()))
    if checks["score_floor"]["n_selected_tau_below_floor"] > 0:
        raise SystemExit("selected thresholds below the stored score floor - results would be invalid")
    rep_err = []
    for r in reg_names:
        m = regimes == r
        for t in (tau_pristine, tau_oracle[r], 0.05, 0.5, tau_mb):
            tp, fp, fn = per_image_counts(table, np.full(table.n, t))
            ref = detector_metrics_at_tau(asc[r], t)
            rep_err.append(abs(tp[m].sum() - ref["tp"]) + abs(fp[m].sum() - ref["fp"]) + abs(fn[m].sum() - ref["fn"]))
    checks["A_fixed_tau_reproduces_stored_sweep"] = {"max_abs_count_difference": int(max(rep_err)),
                                                     "n_checks": len(rep_err), "passed": max(rep_err) == 0}
    tp, fp, fn = per_image_counts(table, policies["regime_oracle"][None])
    per, _, _ = aggregate(tp, fp, fn, regimes, reg_names)
    oerr = max(abs(per[r]["f1"] - detector_metrics_at_tau(asc[r], tau_oracle[r])["f1"]) for r in reg_names)
    checks["B_regime_oracle_reproduces_stored_oracle_point"] = {"max_abs_f1_difference": float(oerr), "passed": oerr < 1e-12}
    checks["C_counts_integer"] = {"passed": all(x.dtype.kind == "i" for x in (tp, fp, fn))}
    # D: perturb one image's threshold and confirm only that image's counts change
    tau_test = policies["adaptive_pchip"][0].copy()
    tpa, fpa, fna = per_image_counts(table, tau_test)
    k = int(np.argmax(np.diff(table.offsets)))
    tau_test[k] = 0.0
    tpb, fpb, fnb = per_image_counts(table, tau_test)
    changed = np.where((tpa != tpb) | (fpa != fpb) | (fna != fnb))[0]
    checks["D_image_association"] = {"perturbed_image": k, "changed_images": changed.tolist(),
                                     "passed": set(changed.tolist()) <= {k}}
    checks["E_no_test_leakage"] = {"passed": True, "note": "adaptive thresholds are computed from cached pooled frozen-detector "
                                   "features only (predict_taus); test GT/sweeps are read only by the oracle and for scoring"}
    checks["F_same_raw_detections"] = {"passed": True, "note": "all policies use the same DetectionTable (scores, is_tp, n_gt)"}
    checks["G_identical_matching"] = {"passed": True, "note": "is_tp computed once per image with generate_curves.greedy_match_labels_one_image_np (IoU 0.5, class 0)"}
    # consistency with the earlier evaluation: recomputed per-image taus vs old eval CSV
    cons = {}
    for pol, pat in LEARNED.items():
        diffs = []
        for s in SEEDS:
            old = pd.read_csv(Path(a.old_eval_root) / pat.format(s) / "curve_evaluation_results.csv")
            old = old[old["variant"] == "degraded"]
            pos = {rp: i for i, rp in enumerate(meta["rel_path"])}
            idx = np.array([pos[rp] for rp in old["rel_path"]])
            diffs.append(np.abs(policies[pol][s][idx] - old["tau_adaptive"].values))
        d = np.concatenate(diffs)
        cons[pol] = {"n_rows_compared": int(len(d)), "fraction_identical": float(np.mean(d < 1e-6)),
                     "max_abs_difference": float(d.max())}
    checks["tau_consistency_with_previous_evaluation"] = cons
    print(json.dumps(jsonable(checks), indent=1))
    if not (checks["A_fixed_tau_reproduces_stored_sweep"]["passed"] and checks["B_regime_oracle_reproduces_stored_oracle_point"]["passed"]
            and checks["D_image_association"]["passed"]):
        raise SystemExit("sanity check failed")

    # ---------------- execution + regime-curve utility ----------------
    counts = {}
    results = []   # rows: policy, seed, regime|ALL, metrics (execution), regime_curve_f1
    for pol, seeds in policies.items():
        for s, tau in seeds.items():
            tp, fp, fn = per_image_counts(table, tau)
            counts[(pol, s)] = (tp, fp, fn)
            per, micro, macro = aggregate(tp, fp, fn, regimes, reg_names)
            rcu = regime_curve_utility(tau, regimes, asc)
            for r in reg_names:
                m = regimes == r
                results.append({"policy": pol, "seed": s, "regime": r, **per[r], "tau_mean": float(tau[m].mean()),
                                "regime_curve_f1": float(rcu[m].mean())})
            results.append({"policy": pol, "seed": s, "regime": "ALL_micro", **micro, "tau_mean": float(tau.mean()),
                            "regime_curve_f1": float(rcu.mean())})
            results.append({"policy": pol, "seed": s, "regime": "ALL_macro", **macro, "tau_mean": float(tau.mean()),
                            "regime_curve_f1": float(np.mean([rcu[regimes == r].mean() for r in reg_names]))})
    R = pd.DataFrame(results)
    R["seed"] = R["seed"].astype("Int64")
    base = R[R.policy == "fixed_pristine"].set_index("regime")
    orc = R[R.policy == "regime_oracle"].set_index("regime")
    R["execution_gain"] = R["f1"] - R["regime"].map(base["f1"])
    R["execution_regret"] = R["regime"].map(orc["f1"]) - R["f1"]
    R["regime_curve_gain"] = R["regime_curve_f1"] - R["regime"].map(base["regime_curve_f1"])
    R["regime_curve_regret"] = R["regime"].map(orc["regime_curve_f1"]) - R["regime_curve_f1"]
    R.to_csv(out / "per_image" / "policy_seed_regime_results_long.csv", index=False)

    def seed_mean(df):
        num = [c for c in df.columns if c not in ("policy", "seed", "regime") and pd.api.types.is_numeric_dtype(df[c])]
        g = df.groupby(["policy", "regime"], sort=False)[num]
        m, sd = g.mean(), g.std(ddof=1).fillna(0.0)
        m.columns = [f"{c}" for c in m.columns]
        sd.columns = [f"{c}_seed_std" for c in sd.columns]
        return pd.concat([m, sd], axis=1).reset_index()

    S = seed_mean(R)
    order = {p: i for i, p in enumerate(MAIN_POLICIES + ["per_image_oracle"])}
    S["_o"] = S["policy"].map(order)
    S = S.sort_values(["_o"]).drop(columns="_o")

    # overall table
    ov = []
    for pol in MAIN_POLICIES + ["per_image_oracle"]:
        mi = S[(S.policy == pol) & (S.regime == "ALL_micro")].iloc[0]
        ma = S[(S.policy == pol) & (S.regime == "ALL_macro")].iloc[0]
        pr = R[(R.policy == pol) & ~R.regime.str.startswith("ALL")]
        gains = pr.groupby("regime")["execution_gain"].mean()
        ov.append({"policy": pol, "n_seeds": int(R[R.policy == pol]["seed"].nunique()) or 1,
                   "deployable": pol in ("fixed_pristine", "mean_behavior", "adaptive_linear", "adaptive_pchip", "pc1_behavior"),
                   "micro_precision": mi["precision"], "micro_recall": mi["recall"], "micro_f1": mi["f1"],
                   "micro_f1_seed_std": mi["f1_seed_std"], "micro_fppi": mi["fppi"],
                   "micro_tp": mi["tp"], "micro_fp": mi["fp"], "micro_fn": mi["fn"],
                   "macro_regime_precision": ma["precision"], "macro_regime_recall": ma["recall"], "macro_regime_f1": ma["f1"],
                   "macro_regime_f1_seed_std": ma["f1_seed_std"], "macro_regime_fppi": ma["fppi"],
                   "micro_execution_gain": mi["f1"] - S[(S.policy == "fixed_pristine") & (S.regime == "ALL_micro")]["f1"].iloc[0],
                   "macro_execution_gain": float(gains.mean()), "median_regime_execution_gain": float(gains.median()),
                   "std_regime_execution_gain": float(gains.std(ddof=0)),
                   "n_regimes_improved": int((gains > 1e-9).sum()), "n_regimes_degraded": int((gains < -1e-9).sum()),
                   "worst_regime": gains.idxmin(), "worst_regime_gain": float(gains.min()),
                   "best_regime": gains.idxmax(), "best_regime_gain": float(gains.max()),
                   "macro_execution_regret": float(pr.groupby("regime")["execution_regret"].mean().mean()),
                   "macro_regime_curve_f1": ma["regime_curve_f1"],
                   "macro_regime_curve_regret": float(pr.groupby("regime")["regime_curve_regret"].mean().mean())})
    OV = pd.DataFrame(ov)
    write_table(OV, out / "tables/per_image_execution_overall",
                "True per-image execution on all 85,800 test image variants (1,650 base images x 18 regimes incl. 3 severities); "
                "learned policies: mean over 3 seeds; per_image_oracle is an exploratory GT-dependent upper bound")

    # per-regime table (wide)
    pr = S[~S.regime.str.startswith("ALL")]
    cols = ["policy", "regime", "tau_mean", "tp", "fp", "fn", "precision", "recall", "f1", "f1_seed_std", "fppi",
            "execution_gain", "execution_regret", "regime_curve_f1", "regime_curve_gain", "regime_curve_regret"]
    write_table(pr[cols], out / "tables/per_image_execution_per_regime",
                "True per-image execution per regime (learned: mean over 3 seeds); regime_curve_* = previous regime-curve utility "
                "on the same image variants")

    # ---------------- comparison with old evaluation ----------------
    old_pr = pd.read_csv(Path(a.old_artifacts) / "aggregate/threshold_policy_per_regime.csv")
    old = old_pr.groupby(["policy", "regime"])[["f1", "gain_vs_pristine", "oracle_regret"]].mean()
    cmp_rows = []
    for pol in ["fixed_pristine", "mean_behavior", "adaptive_linear", "adaptive_pchip", "pc1_behavior"]:
        old_pol = "oracle" if pol == "regime_oracle" else pol
        for r in reg_names + ["MACRO"]:
            if r == "MACRO":
                n = pr[pr.policy == pol]
                o = old.loc[old_pol] if old_pol in old.index.get_level_values(0) else None
                cmp_rows.append({"policy": pol, "regime": "MACRO (18 regimes)",
                                 "old_f1": float(o["f1"].mean()), "new_f1": float(n["f1"].mean()),
                                 "old_gain": float(o["gain_vs_pristine"].mean()), "new_gain": float(n["execution_gain"].mean()),
                                 "old_regret": float(o["oracle_regret"].mean()), "new_regret": float(n["execution_regret"].mean()),
                                 "regime_curve_f1_all_variants": float(n["regime_curve_f1"].mean())})
                continue
            n = pr[(pr.policy == pol) & (pr.regime == r)].iloc[0]
            o = old.loc[(old_pol, r)]
            cmp_rows.append({"policy": pol, "regime": r, "old_f1": float(o["f1"]), "new_f1": float(n["f1"]),
                             "old_gain": float(o["gain_vs_pristine"]), "new_gain": float(n["execution_gain"]),
                             "old_regret": float(o["oracle_regret"]), "new_regret": float(n["execution_regret"]),
                             "regime_curve_f1_all_variants": float(n["regime_curve_f1"])})
    CMP = pd.DataFrame(cmp_rows)
    CMP["f1_difference_new_minus_old"] = CMP["new_f1"] - CMP["old_f1"]
    write_table(CMP, out / "tables/evaluation_mode_comparison",
                "old = regime-curve utility from paper_artifacts (1,650 sampled variants per regime, F1 of the regime curve at tau_i, "
                "averaged over images); new = true per-image execution (all variants, counts aggregated first); "
                "regime_curve_f1_all_variants = old metric recomputed on the new variant set")

    # ---------------- count changes (PCHIP vs fixed pristine) ----------------
    cc = []
    fx = pr[pr.policy == "fixed_pristine"].set_index("regime")
    for pol in ("adaptive_pchip", "mean_behavior", "adaptive_linear", "pc1_behavior"):
        ad = pr[pr.policy == pol].set_index("regime")
        for r in reg_names:
            cc.append({"policy": pol, "regime": r, "tau_fixed": fx.loc[r, "tau_mean"], "tau_policy_mean": ad.loc[r, "tau_mean"],
                       "delta_tp": ad.loc[r, "tp"] - fx.loc[r, "tp"], "delta_fp": ad.loc[r, "fp"] - fx.loc[r, "fp"],
                       "delta_fn": ad.loc[r, "fn"] - fx.loc[r, "fn"],
                       "delta_precision": ad.loc[r, "precision"] - fx.loc[r, "precision"],
                       "delta_recall": ad.loc[r, "recall"] - fx.loc[r, "recall"], "delta_f1": ad.loc[r, "f1"] - fx.loc[r, "f1"],
                       "delta_fppi": ad.loc[r, "fppi"] - fx.loc[r, "fppi"],
                       "fp_added_per_tp_gained": (ad.loc[r, "fp"] - fx.loc[r, "fp"]) / (ad.loc[r, "tp"] - fx.loc[r, "tp"])
                       if abs(ad.loc[r, "tp"] - fx.loc[r, "tp"]) > 0.5 else float("nan")})
    CC = pd.DataFrame(cc)
    write_table(CC[CC.policy == "adaptive_pchip"].sort_values("delta_f1").drop(columns="policy"),
                out / "tables/adaptation_count_changes",
                "Adaptive PCHIP minus fixed pristine per regime (counts summed over all variants of the regime; mean over 3 seeds), "
                "sorted by Delta F1")
    CC.to_csv(out / "diagnostics/count_changes_all_policies.csv", index=False)

    # ---------------- regime-ID baseline table ----------------
    rid = []
    for r in reg_names:
        row = {"regime": r, "tau_regime_validation": tau_regval[r], "tau_regime_oracle_test": tau_oracle[r]}
        for pol in ("mean_behavior", "regime_id_validation", "adaptive_pchip", "pchip_regime_mean_tau", "regime_oracle"):
            row[f"f1_{pol}"] = float(pr[(pr.policy == pol) & (pr.regime == r)]["f1"].iloc[0])
        rid.append(row)
    pd.DataFrame(rid).to_csv(out / "tables/regime_id_baseline.csv", index=False)

    # ---------------- per-image export (seed 0 for learned) ----------------
    pi = pd.DataFrame({"variant_id": table.variant_ids, "base_image_id": meta["image_id"], "rel_path": meta["rel_path"],
                       "regime": regimes, "degradation": meta["deg_name"], "severity": meta["severity"], "n_gt": table.n_gt,
                       "n_detections": np.diff(table.offsets)})
    short_name = {"fixed_pristine": "fixed", "mean_behavior": "mean", "adaptive_linear": "linear", "adaptive_pchip": "pchip",
                  "pc1_behavior": "pc1", "pchip_regime_mean_tau": "pchip_regmean", "regime_id_validation": "regid",
                  "regime_oracle": "oracle", "per_image_oracle": "imgoracle"}
    for pol, sn in short_name.items():
        s = 0 if 0 in policies[pol] else None
        pi[f"tau_{sn}"] = policies[pol][s]
        tp, fp, fn = counts[(pol, s)]
        pi[f"TP_{sn}"], pi[f"FP_{sn}"], pi[f"FN_{sn}"] = tp, fp, fn
    for sn in ("mean", "linear", "pchip", "pc1"):
        for c in ("TP", "FP", "FN"):
            pi[f"delta_{c}_{sn}_vs_fixed"] = pi[f"{c}_{sn}"] - pi[f"{c}_fixed"]
    pi.to_csv(out / "per_image/per_image_policy_results.csv", index=False)

    # ---------------- tau distributions ----------------
    td = []
    for pol in ("adaptive_linear", "adaptive_pchip", "pc1_behavior"):
        for s in SEEDS:
            t = policies[pol][s]
            for r in reg_names:
                x = t[regimes == r]
                q = np.percentile(x, [5, 25, 50, 75, 95])
                td.append({"policy": pol, "seed": s, "regime": r, "mean": x.mean(), "median": q[2], "std": x.std(),
                           "p05": q[0], "p25": q[1], "p75": q[3], "p95": q[4], "iqr": q[3] - q[1], "min": x.min(), "max": x.max(),
                           "tau_regime_oracle": tau_oracle[r], "tau_regime_validation": tau_regval[r]})
    TD = pd.DataFrame(td)
    TD.to_csv(out / "tables/tau_distribution_by_regime.csv", index=False)
    TD.to_csv(out / "diagnostics/tau_distribution_by_regime.csv", index=False)

    # ---------------- within-regime adaptation ----------------
    wr, wra = [], []
    t_img_oracle = policies["per_image_oracle"][None]
    from scipy.stats import spearmanr
    for s in SEEDS:
        t = policies["adaptive_pchip"][s]
        tp_a, fp_a, fn_a = counts[("adaptive_pchip", s)]
        tp_m, fp_m, fn_m = counts[("pchip_regime_mean_tau", s)]
        for r in reg_names:
            m = np.where(regimes == r)[0]
            x = t[m]
            has_gt = table.n_gt[m] > 0
            rho = spearmanr(x[has_gt], t_img_oracle[m][has_gt]).correlation if has_gt.sum() > 10 else float("nan")
            # baseline detector confidence statistics per image
            top = np.array([table.scores[table.offsets[k]:table.offsets[k + 1]].max() if table.offsets[k + 1] > table.offsets[k] else 0.0 for k in m])
            ndet25 = np.array([(table.scores[table.offsets[k]:table.offsets[k + 1]] >= 0.25).sum() for k in m])
            qs = np.quantile(x, [0.25, 0.5, 0.75])
            qi = np.digitize(x, qs)
            wr.append({"seed": s, "regime": r, "std_tau": x.std(), "iqr_tau": qs[2] - qs[0], "min_tau": x.min(), "max_tau": x.max(),
                       "spearman_tau_vs_per_image_oracle_tau": rho,
                       "f1_per_image_tau": metrics_from_counts(tp_a[m].sum(), fp_a[m].sum(), fn_a[m].sum(), len(m))["f1"],
                       "f1_regime_mean_tau": metrics_from_counts(tp_m[m].sum(), fp_m[m].sum(), fn_m[m].sum(), len(m))["f1"]})
            if s == 0:
                for qq in range(4):
                    mm = m[qi == qq]
                    if len(mm) == 0:
                        continue
                    wra.append({"regime": r, "tau_quartile": qq + 1, "n_images": len(mm), "tau_mean": float(t[mm].mean()),
                                "mean_n_gt": float(table.n_gt[mm].mean()), "mean_top_score": float(top[qi == qq].mean()),
                                "mean_n_det_ge_0.25": float(ndet25[qi == qq].mean()),
                                "mean_TP_pchip": float(tp_a[mm].mean()), "mean_FP_pchip": float(fp_a[mm].mean()),
                                "mean_FN_pchip": float(fn_a[mm].mean()),
                                "mean_severity": float(meta["severity"][mm].mean()),
                                "recall_at_fixed_pristine": metrics_from_counts(*[c[mm].sum() for c in counts[("fixed_pristine", None)]], len(mm))["recall"],
                                "per_image_oracle_tau_median": float(np.median(t_img_oracle[mm][table.n_gt[mm] > 0])) if (table.n_gt[mm] > 0).any() else float("nan")})
    WR = pd.DataFrame(wr)
    WR["f1_gain_per_image_over_regime_mean"] = WR["f1_per_image_tau"] - WR["f1_regime_mean_tau"]
    WR.to_csv(out / "tables/within_regime_adaptation.csv", index=False)
    pd.DataFrame(wra).to_csv(out / "diagnostics/within_regime_analysis.csv", index=False)

    # ---------------- fog ----------------
    fog = []
    for pol in MAIN_POLICIES:
        e = pr[(pr.policy == pol) & (pr.regime == "fog_3_5")].iloc[0]
        fog.append({"policy": pol, "tau_mean": e["tau_mean"], "tp": e["tp"], "fp": e["fp"], "fn": e["fn"], "precision": e["precision"],
                    "recall": e["recall"], "f1": e["f1"], "fppi": e["fppi"], "execution_gain": e["execution_gain"],
                    "execution_regret": e["execution_regret"], "old_regime_curve_gain": float(old.loc[("oracle" if pol == "regime_oracle" else pol, "fog_3_5")]["gain_vs_pristine"]) if ("oracle" if pol == "regime_oracle" else pol, "fog_3_5") in old.index else float("nan")})
    FOG = pd.DataFrame(fog)
    FOG.to_csv(out / "diagnostics/fog_execution_comparison.csv", index=False)

    # ---------------- strong shift ----------------
    ss = []
    for r in STRONG:
        for pol in ("fixed_pristine", "mean_behavior", "adaptive_pchip", "regime_oracle"):
            e = pr[(pr.policy == pol) & (pr.regime == r)].iloc[0]
            ss.append({"regime": r, "policy": pol, "tau_mean": e["tau_mean"], "tp": e["tp"], "fp": e["fp"], "fn": e["fn"],
                       "precision": e["precision"], "recall": e["recall"], "f1": e["f1"], "fppi": e["fppi"]})
    pd.DataFrame(ss).to_csv(out / "diagnostics/strong_shift_analysis.csv", index=False)

    # ---------------- bootstrap over base images ----------------
    rng = np.random.default_rng(a.boot_seed)
    bases, binv = np.unique(meta["image_id"], return_inverse=True)
    rid_idx = np.array([reg_names.index(r) for r in regimes])
    nb, nr = len(bases), len(reg_names)

    def cell_counts(pol, s):
        tp, fp, fn = counts[(pol, s)]
        arr = np.zeros((3, nb, nr))
        for c, v in enumerate((tp, fp, fn)):
            np.add.at(arr[c], (binv, rid_idx), v)
        return arr

    nimg = np.zeros((nb, nr))
    np.add.at(nimg, (binv, rid_idx), 1)
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=a.n_boot).astype(np.float64)

    def boot_f1(arr):
        tp = W @ arr[0]
        fp = W @ arr[1]
        fn = W @ arr[2]
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 1.0)
        rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), 0.0)
        f1_reg = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        T, Fp, Fn = tp.sum(1), fp.sum(1), fn.sum(1)
        P = np.where(T + Fp > 0, T / np.maximum(T + Fp, 1e-12), 1.0)
        Rr = np.where(T + Fn > 0, T / np.maximum(T + Fn, 1e-12), 0.0)
        return {"micro_f1": 2 * P * Rr / np.maximum(P + Rr, 1e-12), "macro_regime_f1": f1_reg.mean(1)}

    bf = {}
    for pol, seeds in policies.items():
        if pol == "per_image_oracle":
            continue
        vals = [boot_f1(cell_counts(pol, s)) for s in seeds]
        bf[pol] = {k: np.mean([v[k] for v in vals], axis=0) for k in ("micro_f1", "macro_regime_f1")}
    pairs = [("adaptive_pchip", "fixed_pristine"), ("adaptive_pchip", "mean_behavior"), ("adaptive_pchip", "pc1_behavior"),
             ("adaptive_pchip", "adaptive_linear"), ("adaptive_pchip", "pchip_regime_mean_tau"),
             ("adaptive_pchip", "regime_id_validation"), ("mean_behavior", "fixed_pristine"), ("regime_oracle", "adaptive_pchip"),
             ("regime_id_validation", "mean_behavior")]
    point = {(p, k): float(OV[OV.policy == p][("micro_f1" if k == "micro_f1" else "macro_regime_f1")].iloc[0])
             for p in policies if p != "per_image_oracle" for k in ("micro_f1", "macro_regime_f1")}
    boot = {"method": "paired percentile bootstrap: resample the 1,650 test base images with replacement; every resampled base "
                      "image contributes all of its variants (pristine + all degraded variants of all regimes); TP/FP/FN are re-summed "
                      "per regime and pooled, then F1 recomputed; learned policies: F1 averaged over the 3 seeds within each replicate",
            "n_resamples": a.n_boot, "seed": a.boot_seed, "n_base_images": int(nb), "differences": {}, "policies": {}}
    for p in bf:
        for k in ("micro_f1", "macro_regime_f1"):
            v = bf[p][k]
            boot["policies"].setdefault(p, {})[k] = {"estimate": point[(p, k)], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    for x, y in pairs:
        for k in ("micro_f1", "macro_regime_f1"):
            d = bf[x][k] - bf[y][k]
            boot["differences"][f"{k}: {x} - {y}"] = {"estimate": point[(x, k)] - point[(y, k)],
                                                      "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                      "fraction_of_replicates_positive": float(np.mean(d > 0))}
    (out / "diagnostics/bootstrap_execution.json").write_text(json.dumps(jsonable(boot), indent=2))

    # ---------------- figures ----------------
    def bar_by_regime(metric, fname, ylabel, pols):
        st = R[(R.policy.isin(pols)) & ~R.regime.str.startswith("ALL")].groupby(["policy", "regime"])[metric].agg(["mean", "std"]).reset_index()
        ordr = st[st.policy == "adaptive_pchip"].sort_values("mean")["regime"].tolist()
        x = np.arange(len(ordr))
        w = 0.8 / len(pols)
        fig, ax = plt.subplots(figsize=(14, 4.6), constrained_layout=True)
        for i, p in enumerate(pols):
            sdf = st[st.policy == p].set_index("regime").reindex(ordr)
            ax.bar(x + (i - (len(pols) - 1) / 2) * w, sdf["mean"], w * 0.9, yerr=sdf["std"].fillna(0), capsize=1.5,
                   error_kw=dict(lw=0.7), color=COLOR[p], label=LABEL[p])
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([short(r) for r in ordr], rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.set_title("True per-image execution (counts aggregated per regime; learned: mean ± sd over 3 seeds)", fontsize=FONT_SIZE)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"figures/{fname}.{ext}", dpi=200)
        plt.close(fig)

    bar_by_regime("execution_gain", "execution_gain_by_regime", "F1 gain over fixed pristine τ",
                  ["mean_behavior", "pc1_behavior", "adaptive_linear", "adaptive_pchip", "regime_id_validation", "regime_oracle"])
    bar_by_regime("execution_regret", "execution_regret_by_regime", "execution regret vs regime oracle",
                  ["fixed_pristine", "mean_behavior", "pc1_behavior", "adaptive_pchip", "pchip_regime_mean_tau", "regime_id_validation"])

    c = CC[CC.policy == "adaptive_pchip"].set_index("regime").reindex(
        pr[pr.policy == "adaptive_pchip"].sort_values("execution_gain")["regime"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True, sharey=True)
    y = np.arange(len(c))
    for ax, (col, lab, colr) in zip(axes, (("delta_recall", "Δ recall", PRED_COLOR), ("delta_precision", "Δ precision", COLORS[0]),
                                           ("delta_f1", "Δ F1", PRIMARY_COLOR))):
        ax.barh(y, c[col], color=colr)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel(lab)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([short(r) for r in c.index])
    fig.suptitle("Adaptive PCHIP vs fixed pristine τ: how the F1 change arises (true per-image execution)", fontsize=FONT_SIZE)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figures/precision_recall_change_by_regime.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    pos0 = np.arange(len(reg_names))
    for i, (pol, colr) in enumerate((("adaptive_linear", COLORS[3]), ("adaptive_pchip", PRED_COLOR), ("pc1_behavior", COLORS[4]))):
        data = [policies[pol][0][regimes == r] for r in reg_names]
        bp = ax.boxplot(data, positions=pos0 + (i - 1) * 0.27, widths=0.22, patch_artist=True, showfliers=False,
                        whis=(5, 95), medianprops=dict(color="black", lw=1))
        for b in bp["boxes"]:
            b.set_facecolor(colr)
            b.set_alpha(0.7)
        ax.plot([], [], "s", color=colr, label=f"{LABEL[pol]} (seed 0)")
    ax.plot(pos0, [tau_oracle[r] for r in reg_names], "*", color=PRIMARY_COLOR, ms=11, label="regime oracle τ (test)")
    ax.plot(pos0, [tau_regval[r] for r in reg_names], "x", color="0.2", ms=8, label="regime-ID τ (validation)")
    ax.axhline(tau_pristine, color=COLORS[0], ls="--", lw=1.2, label="fixed pristine τ")
    ax.set_xticks(pos0)
    ax.set_xticklabels([short(r) for r in reg_names], rotation=45, ha="right")
    ax.set_ylabel("selected threshold τ_i (box 25-75 %, whiskers 5-95 %)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figures/tau_distribution_by_regime.{ext}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    F = FOG.set_index("policy")
    xs = np.arange(len(F))
    axes[0].bar(xs - 0.2, F["tp"], 0.4, color=PRED_COLOR, label="TP")
    axes[0].bar(xs + 0.2, F["fp"], 0.4, color=COLORS[0], label="FP")
    lo = min(F["tp"].min(), F["fp"].min()) * 0.9
    axes[0].set_ylim(lo, max(F["tp"].max(), F["fp"].max()) * 1.03)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([LABEL[p] for p in F.index], rotation=35, ha="right", fontsize=8)
    axes[0].legend(frameon=False)
    axes[0].set_title("Fog: TP and FP summed over 4,950 variants", fontsize=FONT_SIZE)
    axes[1].bar(xs, F["f1"], color=[COLOR[p] for p in F.index])
    axes[1].set_ylim(F["f1"].min() - 0.005, F["f1"].max() + 0.003)
    axes[1].axhline(F.loc["fixed_pristine", "f1"], color=COLORS[0], ls="--", lw=1)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([LABEL[p] for p in F.index], rotation=35, ha="right", fontsize=8)
    axes[1].set_title("Fog: F1 (true per-image execution)", fontsize=FONT_SIZE)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figures/fog_execution_counts.{ext}", dpi=200)
    plt.close(fig)

    # ---------------- machine-readable results ----------------
    res = {
        "evaluation_type": "true_per_image_threshold_execution",
        "n_image_variants": int(table.n), "n_base_images": int(nb), "n_regimes": len(reg_names),
        "n_stored_detections": int(len(table.scores)),
        "sanity_checks": checks,
        "thresholds": {"tau_pristine_validation": tau_pristine, "tau_mean_behavior": tau_mb,
                       "tau_regime_oracle_test": tau_oracle, "tau_regime_validation": tau_regval},
        "policies": {p: OV[OV.policy == p].iloc[0].to_dict() for p in OV.policy},
        "overall_micro": {p: {k: OV[OV.policy == p].iloc[0][f"micro_{k}"] for k in ("precision", "recall", "f1", "fppi", "tp", "fp", "fn")} for p in OV.policy},
        "overall_macro": {p: {k: OV[OV.policy == p].iloc[0][f"macro_regime_{k}"] for k in ("precision", "recall", "f1", "fppi")} for p in OV.policy},
        "per_regime": {p: {r: pr[(pr.policy == p) & (pr.regime == r)].iloc[0][["tau_mean", "tp", "fp", "fn", "precision", "recall", "f1", "fppi",
                                                                                 "execution_gain", "execution_regret", "regime_curve_f1",
                                                                                 "regime_curve_gain", "regime_curve_regret"]].to_dict()
                           for r in reg_names} for p in MAIN_POLICIES},
        "bootstrap": boot,
        "fog": FOG.to_dict(orient="records"),
        "within_regime_adaptation": {
            "per_seed_regime": WR.to_dict(orient="records"),
            "mean_f1_gain_per_image_over_regime_mean_tau": float(WR["f1_gain_per_image_over_regime_mean"].mean()),
            "mean_spearman_tau_vs_per_image_oracle": float(WR["spearman_tau_vs_per_image_oracle_tau"].mean())},
        "comparison_to_old_evaluation": CMP.to_dict(orient="records"),
        "runtime_seconds": time.time() - t0,
    }
    (out / "execution_results.json").write_text(json.dumps(jsonable(res), indent=2, allow_nan=False))
    print(OV[["policy", "micro_f1", "macro_regime_f1", "macro_execution_gain", "macro_execution_regret", "n_regimes_improved",
              "n_regimes_degraded"]].round(4).to_string())
    print(json.dumps(jsonable(boot["differences"]), indent=1))


if __name__ == "__main__":
    main()

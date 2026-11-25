# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Experiment C - FPPI-constrained operating policy under true per-image threshold execution.

Objective: maximise recall subject to FPPI <= B in every operating regime.
Each policy selects one threshold per test image; the threshold is applied only to that image's stored
detections; TP/FP/FN are summed per regime first (unchanged execution protocol).

Budgets B are fixed a priori (configs/expC_budgets.json) and are not tuned on test data.
Policies (selection data in brackets):
  fixed_pristine_val     max TPR s.t. FPPI <= B on the pristine validation sweep           [val]
  fixed_pooled_val       same on the pooled validation detections of all 18 regimes        [val]
  fixed_worst_case_val   smallest threshold that satisfies B in every validation regime    [val]
  mean_behavior          image-independent mean training behavior                           [train]
  adaptive_linear/pchip, pc1_behavior   per-image predicted TPR(tau), FPPI(tau)            [image features only]
  regime_id_validation   validation sweep of the true regime (non-deployable)              [val + regime label]
  regime_oracle          test sweep of the true regime (diagnostic upper bound)            [TEST labels]
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from dobi.data.database.generate_curves import compute_operating_sweep_from_global_labels
from dobi.data.feature_cache import FeatureCache
from dobi.eval.execution_eval import per_image_counts
from dobi.eval.extended import common as C
from dobi.utils.tau_targets import oracle_threshold

BUDGETS = (0.05, 0.1, 0.2, 0.5)
TOLERANCE = 0.10          # sensitivity: violation only if FPPI > (1 + 10 %) B
N_BOOT, BOOT_SEED = 2000, 12345
FIXED = ("fixed_pristine_val", "fixed_pooled_val", "fixed_worst_case_val")
CALIBRATED = ("mean_behavior_valcal", "adaptive_linear_valcal", "adaptive_pchip_valcal", "pc1_behavior_valcal")
POLICIES = FIXED + ("mean_behavior", "adaptive_linear", "adaptive_pchip", "pc1_behavior") + CALIBRATED + ("regime_id_validation", "regime_oracle")
DEPLOYABLE = FIXED + ("mean_behavior", "adaptive_linear", "adaptive_pchip", "pc1_behavior") + CALIBRATED
KAPPA_GRID = np.round(np.arange(0.20, 1.0001, 0.01), 2)


def calibrate_kappa(val, tpr, fppi, budget):
    """
    Validation calibration of a safety factor kappa (no test data): the largest kappa on KAPPA_GRID such that the
    per-image policy run with the internal budget kappa*B keeps the observed FPPI <= B in every validation regime
    (the criterion of fixed_worst_case_val). Violations are monotone in kappa, so a bisection over the grid is exact.
    """
    def ok(kappa):
        tau = C.fppi_constrained_tau(tpr, fppi, kappa * budget)
        tp, fp, fn = per_image_counts(val.table, tau)
        return max(m["fppi"] for m in regime_metrics(tp, fp, fn, val.regimes, val.reg_names).values()) <= budget
    if ok(KAPPA_GRID[-1]):
        return float(KAPPA_GRID[-1])
    lo, hi = 0, len(KAPPA_GRID) - 1          # invariant: hi fails; lo unknown
    if not ok(KAPPA_GRID[lo]):
        return float(KAPPA_GRID[lo])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(KAPPA_GRID[mid]) else (lo, mid)
    return float(KAPPA_GRID[lo])


def pooled_sweep(ctx: C.SplitContext):
    t = ctx.table
    return compute_operating_sweep_from_global_labels(t.scores, t.is_tp, int(t.n_gt.sum()), int(t.n))


def regime_metrics(tp, fp, fn, regimes, reg_names):
    rows = {}
    for r in reg_names:
        m = regimes == r
        T, Fp, Fn, n = tp[m].sum(), fp[m].sum(), fn[m].sum(), m.sum()
        rows[r] = {"tp": int(T), "fp": int(Fp), "fn": int(Fn), "n_images": int(n),
                   "recall": T / max(T + Fn, 1), "fppi": Fp / max(n, 1), "precision": T / (T + Fp) if T + Fp > 0 else 1.0}
    return rows


def run(out: Path, workers: int = 20, log=print, test=None, val=None):
    t0 = time.time()
    C.write_json({"budgets": BUDGETS, "tolerance": TOLERANCE, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                  "rationale": "a-priori operating requirements spanning strict (0.05) to permissive (0.5) false-positive "
                               "budgets; the validation FPPI of the F1-optimal pristine threshold is logged for context only; "
                               "no test data are used to choose budgets or thresholds (except regime_oracle)"},
                 out / "configs/expC_budgets.json")
    test = test or C.load_split_context("test")
    val = val or C.load_split_context("val")
    regs = test.reg_names
    assert regs == val.reg_names
    pooled_val = pooled_sweep(val)
    from dobi.utils.tau_targets import pristine_validation_threshold
    tau_pf1, _ = pristine_validation_threshold(C.DB, C.CLASS_ID, C.IOU_THR, "val")
    tp_, fp_, fn_ = per_image_counts(val.table, np.full(val.table.n, tau_pf1))
    ctx_info = {"tau_pristine_f1_val": tau_pf1, "pooled_val_fppi_at_tau_pristine_f1": fp_.sum() / val.table.n,
                "val_regime_fppi_range_at_tau_pristine_f1": [min(m["fppi"] for m in regime_metrics(tp_, fp_, fn_, val.regimes, regs).values()),
                                                             max(m["fppi"] for m in regime_metrics(tp_, fp_, fn_, val.regimes, regs).values())]}
    log(f"C: context {ctx_info}")

    # ---- thresholds ----
    mb = C.mean_behavior_functions()
    cache = FeatureCache(C.FEATURE_CACHE, "test")
    vcache = FeatureCache(C.FEATURE_CACHE, "val")
    preds, vpreds = {}, {}
    for pol in C.LEARNED:
        for s in C.SEEDS:
            F = C.predict_functions(C.checkpoint_path(pol, s), cache, test.table.variant_ids)
            preds[(pol, s)] = {"tpr": F["tpr"], "fppi": F["fppi"]}
            del F
            F = C.predict_functions(C.checkpoint_path(pol, s), vcache, val.table.variant_ids)
            vpreds[(pol, s)] = {"tpr": F["tpr"], "fppi": F["fppi"]}
            del F
    log(f"C: predictions loaded ({time.time() - t0:.0f}s)")

    N = test.table.n
    taus, pred_fppi, infeasible = {}, {}, {}
    thr_rows, kappas = [], []
    for B in BUDGETS:
        tw = [oracle_threshold(val.raw[r], "fppi", B) for r in regs]
        taus[(B, "fixed_pristine_val", None)] = np.full(N, oracle_threshold(val.raw["pristine"], "fppi", B))
        taus[(B, "fixed_pooled_val", None)] = np.full(N, oracle_threshold(pooled_val, "fppi", B))
        taus[(B, "fixed_worst_case_val", None)] = np.full(N, max(tw))
        tmb = float(C.fppi_constrained_tau(mb["tpr"][None], mb["fppi"][None], B)[0])
        taus[(B, "mean_behavior", None)] = np.full(N, tmb)
        q = int(np.argmin(np.abs(C.TAU - tmb)))
        pred_fppi[(B, "mean_behavior", None)] = np.full(N, mb["fppi"][q])
        rid = {r: oracle_threshold(val.raw[r], "fppi", B) for r in regs}
        orc = {r: oracle_threshold(test.raw[r], "fppi", B) for r in regs}
        taus[(B, "regime_id_validation", None)] = np.array([rid[r] for r in test.regimes])
        taus[(B, "regime_oracle", None)] = np.array([orc[r] for r in test.regimes])
        for (pol, s), P in preds.items():
            t = C.fppi_constrained_tau(P["tpr"], P["fppi"], B)
            idx = np.rint(t * (len(C.TAU) - 1)).astype(int)
            taus[(B, pol, s)] = t
            pred_fppi[(B, pol, s)] = P["fppi"][np.arange(N), idx]
            infeasible[(B, pol, s)] = ~(P["fppi"] <= B).any(axis=1)
        # validation-calibrated safety factor (val labels only)
        k_mb = calibrate_kappa(val, np.repeat(mb["tpr"][None], val.table.n, 0), np.repeat(mb["fppi"][None], val.table.n, 0), B)
        taus[(B, "mean_behavior_valcal", None)] = np.full(N, float(C.fppi_constrained_tau(mb["tpr"][None], mb["fppi"][None], k_mb * B)[0]))
        kappas.append({"budget": B, "policy": "mean_behavior_valcal", "seed": -1, "kappa": k_mb})
        for (pol, s), P in preds.items():
            k = calibrate_kappa(val, vpreds[(pol, s)]["tpr"], vpreds[(pol, s)]["fppi"], B)
            t = C.fppi_constrained_tau(P["tpr"], P["fppi"], k * B)
            idx = np.rint(t * (len(C.TAU) - 1)).astype(int)
            taus[(B, f"{pol}_valcal", s)] = t
            pred_fppi[(B, f"{pol}_valcal", s)] = P["fppi"][np.arange(N), idx]
            kappas.append({"budget": B, "policy": f"{pol}_valcal", "seed": s, "kappa": k})
        log(f"C: budget {B} calibrated ({time.time() - t0:.0f}s)")
        for r in regs:
            thr_rows.append({"budget": B, "regime": r, "tau_regime_id_validation": rid[r], "tau_regime_oracle_test": orc[r],
                             "tau_fixed_pristine_val": taus[(B, "fixed_pristine_val", None)][0],
                             "tau_fixed_pooled_val": taus[(B, "fixed_pooled_val", None)][0],
                             "tau_fixed_worst_case_val": taus[(B, "fixed_worst_case_val", None)][0], "tau_mean_behavior": tmb})
    pd.DataFrame(thr_rows).to_csv(C.new_file(out / "metrics/expC_reference_thresholds.csv"), index=False)
    C.write_table(pd.DataFrame(kappas), out / "tables/expC_validation_kappa",
                  "Experiment C: validation-calibrated safety factor kappa (internal budget kappa*B) per policy, seed and budget")

    # ---- execution ----
    rows, counts = [], {}
    for (B, pol, s), tau in taus.items():
        tp, fp, fn = per_image_counts(test.table, tau)
        counts[(B, pol, s)] = (tp, fp, fn)
        rm = regime_metrics(tp, fp, fn, test.regimes, regs)
        for r in regs:
            m = test.regimes == r
            e = {"budget": B, "policy": pol, "seed": -1 if s is None else s, "regime": r, **rm[r],
                 "tau_mean": float(tau[m].mean()), "tau_median": float(np.median(tau[m])),
                 "tau_p05": float(np.percentile(tau[m], 5)), "tau_p95": float(np.percentile(tau[m], 95)),
                 "violation": rm[r]["fppi"] > B, "violation_tol10": rm[r]["fppi"] > (1 + TOLERANCE) * B,
                 "excess_fppi": max(0.0, rm[r]["fppi"] - B)}
            if (B, pol, s) in pred_fppi:
                e["predicted_fppi_mean"] = float(pred_fppi[(B, pol, s)][m].mean())
                e["predicted_violation"] = e["predicted_fppi_mean"] > B
            if (B, pol, s) in infeasible:
                e["fraction_images_no_feasible_tau"] = float(infeasible[(B, pol, s)][m].mean())
            rows.append(e)
        T, Fp, Fn = tp.sum(), fp.sum(), fn.sum()
        rows.append({"budget": B, "policy": pol, "seed": -1 if s is None else s, "regime": "ALL_micro", "tp": int(T), "fp": int(Fp),
                     "fn": int(Fn), "n_images": N, "recall": T / (T + Fn), "fppi": Fp / N, "precision": T / max(T + Fp, 1),
                     "tau_mean": float(tau.mean()), "violation": Fp / N > B, "violation_tol10": Fp / N > (1 + TOLERANCE) * B,
                     "excess_fppi": max(0.0, Fp / N - B)})
    R = pd.DataFrame(rows)
    R.to_csv(C.new_file(out / "metrics/expC_per_seed_regime.csv"), index=False)

    # ---- summaries ----
    pr = R[R.regime != "ALL_micro"]
    per_seed = pr.groupby(["budget", "policy", "seed"]).agg(
        macro_recall=("recall", "mean"), macro_fppi=("fppi", "mean"), n_violations=("violation", "sum"),
        n_violations_tol10=("violation_tol10", "sum"), macro_excess_fppi=("excess_fppi", "mean"), max_fppi=("fppi", "max"),
        min_recall=("recall", "min"), n_predicted_violations=("predicted_violation", lambda x: x.fillna(False).astype(bool).sum())).reset_index()
    mic = R[R.regime == "ALL_micro"].set_index(["budget", "policy", "seed"])[["recall", "fppi"]].rename(
        columns={"recall": "micro_recall", "fppi": "micro_fppi"})
    per_seed = per_seed.join(mic, on=["budget", "policy", "seed"])
    per_seed["max_fppi_over_budget"] = per_seed["max_fppi"] / per_seed["budget"]
    per_seed.to_csv(C.new_file(out / "metrics/expC_summary_per_seed.csv"), index=False)
    num = [c for c in per_seed.columns if c not in ("budget", "policy", "seed")]
    S = per_seed.groupby(["budget", "policy"], sort=False)[num].mean()
    Ssd = per_seed.groupby(["budget", "policy"], sort=False)[["macro_recall", "n_violations"]].std(ddof=1).fillna(0).add_suffix("_seed_sd")
    S = S.join(Ssd).reset_index()
    S["order"] = S["policy"].map({p: i for i, p in enumerate(POLICIES)})
    S = S.sort_values(["budget", "order"]).drop(columns="order")
    S.insert(2, "deployable", S["policy"].isin(DEPLOYABLE))

    # ---- bootstrap over base images ----
    bases, binv = np.unique(test.meta["image_id"], return_inverse=True)
    ridx = np.array([regs.index(r) for r in test.regimes])
    nb, nr = len(bases), len(regs)
    W = C.bootstrap_weights(nb, N_BOOT, BOOT_SEED)
    n_img_cell = C.cell_sums(np.ones(N), binv, ridx, nb, nr)
    I = W @ n_img_cell

    def boot(B, pol):
        seeds = [s for (bb, p, s) in counts if bb == B and p == pol]
        rec, vio, exc, fppi = [], [], [], []
        for s in seeds:
            tp, fp, fn = counts[(B, pol, s)]
            T = W @ C.cell_sums(tp, binv, ridx, nb, nr)
            Fp = W @ C.cell_sums(fp, binv, ridx, nb, nr)
            G = W @ C.cell_sums(tp + fn, binv, ridx, nb, nr)
            r_ = T / np.maximum(G, 1)
            f_ = Fp / np.maximum(I, 1)
            rec.append(r_.mean(1))
            vio.append((f_ > B).sum(1))
            exc.append(np.maximum(f_ - B, 0).mean(1))
            fppi.append(f_)
        return {"macro_recall": np.mean(rec, 0), "n_violations": np.mean(vio, 0), "macro_excess_fppi": np.mean(exc, 0),
                "p_violation_regime": np.mean([(f > B).mean(0) for f in fppi], 0)}

    bt, diffs, pviol = {}, [], []
    for B in BUDGETS:
        for pol in POLICIES:
            bt[(B, pol)] = boot(B, pol)
            for k, r in enumerate(regs):
                pviol.append({"budget": B, "policy": pol, "regime": r, "bootstrap_probability_fppi_above_budget": bt[(B, pol)]["p_violation_regime"][k]})
        for pol in POLICIES:
            for k in ("macro_recall", "n_violations", "macro_excess_fppi"):
                v = bt[(B, pol)][k]
                S.loc[(S.budget == B) & (S.policy == pol), f"{k}_ci_lo"] = np.percentile(v, 2.5)
                S.loc[(S.budget == B) & (S.policy == pol), f"{k}_ci_hi"] = np.percentile(v, 97.5)
        for x in ("adaptive_pchip", "adaptive_pchip_valcal"):
            for y in ("fixed_pristine_val", "fixed_pooled_val", "fixed_worst_case_val", "mean_behavior", "mean_behavior_valcal",
                      "adaptive_linear", "pc1_behavior", "adaptive_linear_valcal", "pc1_behavior_valcal", "regime_id_validation", "regime_oracle"):
                if x == y:
                    continue
                for k in ("macro_recall", "n_violations", "macro_excess_fppi"):
                    d = bt[(B, x)][k] - bt[(B, y)][k]
                    pt = S[(S.budget == B) & (S.policy == x)][k].iloc[0] - S[(S.budget == B) & (S.policy == y)][k].iloc[0]
                    diffs.append({"budget": B, "comparison": f"{x} - {y}", "metric": k, "estimate": pt,
                                  "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "fraction_positive": float(np.mean(d > 0))})
    pd.DataFrame(pviol).to_csv(C.new_file(out / "metrics/expC_bootstrap_violation_probability.csv"), index=False)
    D = pd.DataFrame(diffs)
    C.write_table(D, out / "tables/expC_bootstrap_differences",
                  f"Experiment C: paired bootstrap over test base images ({N_BOOT} resamples, seed {BOOT_SEED}); learned: seed mean inside replicate")
    cols = ["budget", "policy", "deployable", "macro_recall", "macro_recall_ci_lo", "macro_recall_ci_hi", "macro_recall_seed_sd", "micro_recall",
            "micro_fppi", "macro_fppi", "max_fppi_over_budget", "n_violations", "n_violations_ci_lo", "n_violations_ci_hi",
            "n_violations_tol10", "n_predicted_violations", "macro_excess_fppi", "min_recall"]
    C.write_table(S[cols], out / "tables/expC_summary",
                  "Experiment C: recall under an FPPI budget B per regime (true per-image execution, 18 test regimes); n_violations = regimes "
                  "with observed FPPI > B (tol10: > 1.1 B); n_predicted_violations = regimes whose mean predicted FPPI at the selected "
                  "thresholds exceeds B; learned: mean over 3 seeds; CI: paired bootstrap over base images")

    # predicted vs observed FPPI (learned, seed mean)
    pv = pr[pr.policy.isin(list(C.LEARNED) + ["mean_behavior"] + list(CALIBRATED))].groupby(["budget", "policy", "regime"])[
        ["predicted_fppi_mean", "fppi", "recall", "tau_mean", "fraction_images_no_feasible_tau"]].mean().reset_index()
    pv["observed_minus_predicted_fppi"] = pv["fppi"] - pv["predicted_fppi_mean"]
    C.write_table(pv, out / "tables/expC_predicted_vs_observed_fppi",
                  "Experiment C: predicted FPPI (mean of per-image predicted FPPI at the selected thresholds) vs observed FPPI after execution")
    # per-regime table for B = 0.1
    w = pr[pr.budget == 0.1].groupby(["regime", "policy"])[["recall", "fppi"]].mean().unstack("policy")
    w.columns = [f"{m}_{p}" for m, p in w.columns]
    keep = [f"{m}_{p}" for p in ("fixed_worst_case_val", "fixed_pooled_val", "adaptive_pchip", "adaptive_pchip_valcal", "regime_id_validation", "regime_oracle")
            for m in ("recall", "fppi")]
    C.write_table(w[keep].reset_index(), out / "tables/expC_per_regime_B0p1",
                  "Experiment C, B = 0.1: per-regime recall and observed FPPI (learned: seed mean)")

    # leave-one-regime-out: macro recall difference PCHIP - fixed baselines
    lo = []
    for B in BUDGETS:
        for x in ("adaptive_pchip", "adaptive_pchip_valcal"):
            a = pr[(pr.budget == B) & (pr.policy == x)].groupby("regime")["recall"].mean()
            for y in FIXED + ("mean_behavior",):
                b = pr[(pr.budget == B) & (pr.policy == y)].groupby("regime")["recall"].mean()
                d = a - b
                vals = [d.drop(r).mean() for r in d.index]
                av = pr[(pr.budget == B) & (pr.policy == x)].groupby("regime")["violation"].mean()
                bv = pr[(pr.budget == B) & (pr.policy == y)].groupby("regime")["violation"].mean()
                lo.append({"budget": B, "comparison": f"{x} - {y}", "macro_recall_difference": d.mean(),
                           "loro_min": min(vals), "loro_max": max(vals),
                           "violations_policy": av.sum(), "violations_baseline": bv.sum()})
    C.write_table(pd.DataFrame(lo), out / "tables/expC_leave_one_regime_out",
                  "Experiment C: leave-one-regime-out range of the macro recall difference")
    C.write_json({"context": ctx_info, "kappas": kappas, "summary": S.to_dict(orient="records"), "bootstrap_differences": D.to_dict(orient="records")},
                 out / "metrics/expC_results.json")
    log(S[["budget", "policy", "macro_recall", "n_violations_tol10", "micro_fppi", "max_fppi_over_budget", "n_violations", "n_predicted_violations"]].round(4).to_string())
    log(f"C finished in {time.time() - t0:.0f}s")

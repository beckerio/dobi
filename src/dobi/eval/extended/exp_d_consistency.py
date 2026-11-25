# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Experiment D - internal consistency of the independently predicted behavior components (diagnostic only).

Identities that hold exactly for empirical regime behavior (generate_curves definitions):
  (I1) precision(tau) = rho * TPR(tau) / (rho * TPR(tau) + FPPI(tau)),  rho = GT objects per image
  (I2) FPPI(tau)      = FPR(tau) * FPPI(0)                             (FPR = FP(tau) / FP(0))
The monitor predicts TPR, FPR, FPPI and precision with separate outputs; nothing enforces (I1)/(I2).
rho is a deployment constant estimated on the TRAINING split (validation value reported); the test
density is used only for the exact-identity check of the empirical curves.

Threshold domains: (a) tau in [0, 0.8]; (b) 'active' thresholds where the predicted expected number of
detections per image rho*TPR + FPPI >= 0.05 (prediction-only mask, no ground truth).
(I2) is evaluated twice: anchored at tau = 0 (eF_tau08; dominated by the boundary jump of the empirical FPR/FPPI
just above the score floor 0.001, which uniform knots cannot resolve below the first interior knot 1/15) and
anchored at the first interior knot tau0 = 1/15, FPPI(tau) = FPR(tau) FPPI(tau0) / FPR(tau0) for tau in [tau0, 0.8]
(eF_knot1; internal consistency where the representation has resolution).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dobi.data.feature_cache import FeatureCache
from dobi.eval.execution_eval import per_image_counts
from dobi.eval.extended import common as C
from dobi.utils.tau_spline import safe_f1
from dobi.utils.tau_targets import oracle_threshold, sweep_at_tau

TAU_MAX = 0.8
ACTIVE_MIN = 0.05
Q_MAX = int(round(TAU_MAX * (C.N_DENSE_TAU - 1)))
Q_KNOT1 = int(round((C.N_DENSE_TAU - 1) / 15))     # first interior uniform knot (K = 16)


def reconstructed_precision(tpr, fppi, rho):
    num = rho * tpr
    den = num + fppi
    return np.where(den > 1e-9, num / np.maximum(den, 1e-12), 1.0)


def consistency_errors(F, rho, chunk=8192):
    """Per-row errors over the two threshold domains. F: dict [N, Q]."""
    N = len(F["tpr"])
    res = {k: np.empty(N) for k in ("eP_tau08", "eP_active", "eF_tau08", "eF_knot1", "active_fraction")}
    for s in range(0, N, chunk):
        tpr, fpr, fppi, prec = (F[k][s:s + chunk, :Q_MAX + 1].astype(np.float64) for k in C.FUNC_KEYS)
        prec_rec = reconstructed_precision(tpr, fppi, rho)
        d = np.abs(prec - prec_rec)
        act = (rho * tpr + fppi) >= ACTIVE_MIN
        res["eP_tau08"][s:s + chunk] = d.mean(1)
        res["eP_active"][s:s + chunk] = np.where(act.any(1), (d * act).sum(1) / np.maximum(act.sum(1), 1), np.nan)
        res["active_fraction"][s:s + chunk] = act.mean(1)
        fppi0 = F["fppi"][s:s + chunk, :1].astype(np.float64)
        res["eF_tau08"][s:s + chunk] = np.abs(np.log1p(fppi) - np.log1p(fpr * fppi0)).mean(1)
        q0 = Q_KNOT1
        ratio = fppi[:, q0:q0 + 1] / np.maximum(fpr[:, q0:q0 + 1], 1e-9)
        res["eF_knot1"][s:s + chunk] = np.abs(np.log1p(fppi[:, q0:]) - np.log1p(fpr[:, q0:] * ratio)).mean(1)
    return res


def accuracy_errors(F, emp, ridx, rho, chunk=8192):
    """tau-domain errors of each row against the empirical behavior of its regime, tau in [0, 0.8]."""
    N = len(F["tpr"])
    out = {k: np.empty(N) for k in ("err_precision_direct", "err_precision_reconstructed", "err_tpr", "err_log1p_fppi")}
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        r = ridx[sl]
        tpr, fppi, prec = (F[k][sl, :Q_MAX + 1].astype(np.float64) for k in ("tpr", "fppi", "precision"))
        e_tpr, e_fppi, e_prec = emp["tpr"][r, :Q_MAX + 1], emp["fppi"][r, :Q_MAX + 1], emp["precision"][r, :Q_MAX + 1]
        out["err_precision_direct"][sl] = np.abs(prec - e_prec).mean(1)
        out["err_precision_reconstructed"][sl] = np.abs(reconstructed_precision(tpr, fppi, rho) - e_prec).mean(1)
        out["err_tpr"][sl] = np.abs(tpr - e_tpr).mean(1)
        out["err_log1p_fppi"][sl] = np.abs(np.log1p(fppi) - np.log1p(e_fppi)).mean(1)
    return out


def run(out: Path, workers: int = 20, log=print, test=None):
    t0 = time.time()
    test = test or C.load_split_context("test")
    regs = test.reg_names
    ridx = np.array([regs.index(r) for r in test.regimes])
    dens = {s: C.train_gt_density(split=s) for s in ("train", "val")}
    rho = dens["train"]["rho"]
    n_gt_test = int(test.table.n_gt[test.regimes == "pristine"].sum())
    rho_test = n_gt_test / int((test.regimes == "pristine").sum())       # identity check only, never a monitor input
    C.write_json({"rho_train": dens["train"], "rho_val": dens["val"], "rho_test_identity_check_only": rho_test,
                  "tau_max": TAU_MAX, "active_min_expected_detections": ACTIVE_MIN}, out / "configs/expD_settings.json")

    # empirical regime behavior on the dense grid
    emp = {k: np.stack([sweep_at_tau(test.asc[r], C.TAU.astype(np.float64))[k] for r in regs]) for k in ("tpr", "fpr", "fppi", "precision")}
    tau_star = np.array([oracle_threshold(test.raw[r], "f1") for r in regs])
    f1_star = np.array([float(sweep_at_tau(test.asc[r], np.array([t]))["f1"][0]) for r, t in zip(regs, tau_star)])
    floor = []
    for rh, lab in ((rho, "rho_train"), (rho_test, "rho_test (identity check)")):
        e = consistency_errors({k: v.astype(np.float32) for k, v in emp.items()}, rh)
        for k, r in enumerate(regs):
            floor.append({"density": lab, "regime": r, "eP_tau08": e["eP_tau08"][k], "eP_active": e["eP_active"][k], "eF_tau08": e["eF_tau08"][k],
                          "eF_knot1": e["eF_knot1"][k]})
    FL = pd.DataFrame(floor)
    FL.to_csv(C.new_file(out / "metrics/expD_empirical_floor_per_regime.csv"), index=False)
    log(f"D: empirical identity check (rho_test): max eP {FL[FL.density.str.startswith('rho_test')].eP_tau08.max():.2e}, "
        f"max eF {FL.eF_tau08.max():.2e}; density-mismatch floor (rho_train) mean eP {FL[FL.density == 'rho_train'].eP_tau08.mean():.4f}")

    cache = FeatureCache(C.FEATURE_CACHE, "test")
    paper_mad = {}
    for pol, pat in C.LEARNED.items():
        for s in C.SEEDS:
            old = pd.read_csv(Path('results/main_study/results_legacy/results/paper/eval') / pat.format(s) / "curve_evaluation_results.csv")
            old = old[old["variant"] == "degraded"]
            paper_mad[(pol, s)] = old.set_index("rel_path")[["mad_roc", "mad_froc", "mad_pr"]]
    per_img, derived = [], []
    for pol in C.LEARNED:
        for s in C.SEEDS:
            F = C.predict_functions(C.checkpoint_path(pol, s), cache, test.table.variant_ids)
            ce = consistency_errors(F, rho)
            ae = accuracy_errors(F, emp, ridx, rho)
            tau_hat = C.argmax_f1_tau(F["tpr"], F["precision"])
            prec_rec = np.concatenate([reconstructed_precision(F["tpr"][a:a + 8192].astype(np.float64), F["fppi"][a:a + 8192].astype(np.float64), rho)
                                       for a in range(0, test.table.n, 8192)])
            tau_rec = C.argmax_f1_tau(F["tpr"], prec_rec)
            f1_rc = np.empty(test.table.n)
            for k, r in enumerate(regs):
                m = ridx == k
                f1_rc[m] = sweep_at_tau(test.asc[r], tau_hat[m].astype(np.float64))["f1"]
            df = pd.DataFrame({"model": pol, "seed": s, "variant_id": test.table.variant_ids, "rel_path": test.meta["rel_path"],
                               "base_image_id": test.meta["image_id"], "regime": test.regimes, **ce, **ae, "tau_hat": tau_hat,
                               "abs_tau_error": np.abs(tau_hat - tau_star[ridx]), "regret_rcu": f1_star[ridx] - f1_rc,
                               "tau_hat_reconstructed_precision": tau_rec})
            pm = paper_mad[(pol, s)]
            df = df.join(pm, on="rel_path")
            per_img.append(df)
            # exploratory diagnostic: operating point from (TPR, reconstructed precision) under true execution
            for lab, t in (("direct_precision", tau_hat), ("reconstructed_precision", tau_rec)):
                tp, fp, fn = per_image_counts(test.table, t)
                f1r = []
                for k, r in enumerate(regs):
                    m = ridx == k
                    T, Fp, Fn = tp[m].sum(), fp[m].sum(), fn[m].sum()
                    P = T / (T + Fp) if T + Fp else 1.0
                    Rr = T / max(T + Fn, 1)
                    f1r.append(2 * P * Rr / (P + Rr) if P + Rr else 0.0)
                derived.append({"model": pol, "seed": s, "operating_point_from": lab, "macro_execution_f1": float(np.mean(f1r)),
                                **{f"f1_{r}": v for r, v in zip(regs, f1r)}})
            log(f"D: {pol} s{s} done ({time.time() - t0:.0f}s)")
            del F, prec_rec
    P = pd.concat(per_img, ignore_index=True)
    P.to_parquet(C.new_file(out / "metrics/expD_per_image.parquet"), index=False)
    DV = pd.DataFrame(derived)
    DV.to_csv(C.new_file(out / "metrics/expD_reconstructed_precision_operating_point.csv"), index=False)
    summarize(out, P, FL, DV, regs, log)
    log(f"D finished in {time.time() - t0:.0f}s")


def summarize(out, P, FL, DV, regs, log=print):
    E = ["eP_tau08", "eP_active", "eF_tau08", "eF_knot1", "err_precision_direct", "err_precision_reconstructed", "err_tpr", "err_log1p_fppi",
         "abs_tau_error", "regret_rcu"]
    reg = P.groupby(["model", "seed", "regime"])[E + ["mad_pr"]].mean().reset_index()
    reg.to_csv(C.new_file(out / "metrics/expD_per_seed_regime.csv"), index=False)
    macro = reg.groupby(["model", "seed"])[E + ["mad_pr"]].mean().reset_index()
    rows = []
    for m, d in macro.groupby("model", sort=False):
        e = {"model": m}
        for k in E:
            e[k] = d[k].mean()
            e[f"{k}_seed_sd"] = d[k].std(ddof=1)
        rows.append(e)
    fl = FL[FL.density == "rho_train"][["eP_tau08", "eP_active", "eF_tau08", "eF_knot1"]].mean()
    rows.append({"model": "empirical regime curves (rho_train floor)", **fl.to_dict()})
    S = pd.DataFrame(rows)
    C.write_table(S[["model", "eP_tau08", "eP_tau08_seed_sd", "eP_active", "eF_tau08", "eF_knot1", "eF_knot1_seed_sd", "err_precision_direct",
                     "err_precision_reconstructed", "err_tpr", "err_log1p_fppi"]], out / "tables/expD_consistency_summary",
                  "Experiment D: macro (18 regimes) internal-consistency errors of the predicted components (mean |precision - "
                  "rho TPR/(rho TPR + FPPI)| over tau in [0, 0.8]; FPR-FPPI identity anchored at tau = 0 (eF_tau08, boundary-dominated) and at "
                  "the first interior knot 1/15 (eF_knot1, tau in [1/15, 0.8])) and accuracy of direct vs "
                  "reconstructed precision against the empirical regime precision; learned: mean over 3 seeds")

    # relationships
    rel = []
    for m in C.LEARNED:
        d = P[P.model == m]
        for x, y in (("eP_tau08", "mad_pr"), ("eP_tau08", "abs_tau_error"), ("eP_tau08", "regret_rcu"), ("eF_tau08", "mad_pr"),
                     ("eF_knot1", "mad_pr"), ("eF_knot1", "abs_tau_error"),
                     ("eP_tau08", "err_precision_direct")):
            dd = d[[x, y, "regime"]].dropna()
            within = [spearmanr(g[x], g[y]).correlation for _, g in dd.groupby("regime") if len(g) > 20]
            rr = reg[reg.model == m].groupby("regime")[[x, y]].mean()
            rel.append({"model": m, "x": x, "y": y, "spearman_pooled_images": spearmanr(dd[x], dd[y]).correlation,
                        "spearman_within_regime_mean": float(np.nanmean(within)), "spearman_across_18_regimes": spearmanr(rr[x], rr[y]).correlation,
                        "n_images": len(dd)})
    # regime level: consistency vs execution regret of the paper
    ex = pd.read_csv('results/main_study/paper_artifacts_execution/tables/per_image_execution_per_regime.csv')
    for m in C.LEARNED:
        rr = reg[reg.model == m].groupby("regime")["eP_tau08"].mean()
        er = ex[ex.policy == m].set_index("regime")["execution_regret"].reindex(rr.index)
        rel.append({"model": m, "x": "eP_tau08", "y": "execution_regret (paper_artifacts_execution)",
                    "spearman_across_18_regimes": spearmanr(rr, er).correlation, "n_images": np.nan})
    C.write_table(pd.DataFrame(rel), out / "tables/expD_relationships",
                  "Experiment D: Spearman correlations between consistency error and curve / operating-point errors (pooled images, "
                  "mean within-regime, and across the 18 regime means); mad_pr from the paper's per-image evaluation sample")

    # model comparison: paired bootstrap over base images on eP (seed mean per image)
    comp = []
    for metric in ("eP_tau08", "eF_knot1"):
        pv = P.groupby(["model", "base_image_id"])[metric].mean().unstack("model")
        rng = np.random.default_rng(12345)
        idx = rng.integers(0, len(pv), size=(2000, len(pv)))
        for a, b in (("adaptive_pchip", "adaptive_linear"), ("adaptive_pchip", "pc1_behavior"), ("adaptive_linear", "pc1_behavior")):
            d = (pv[a] - pv[b]).values
            bs = d[idx].mean(1)
            comp.append({"comparison": f"{a} - {b}", "metric": f"{metric} (mean over the variants of a base image, seeds averaged)",
                         "estimate": d.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
    C.write_table(pd.DataFrame(comp), out / "tables/expD_model_comparison",
                  "Experiment D: paired bootstrap (2,000 resamples of base images, seed 12345) of consistency-error differences")
    # per-regime pchip
    pr = reg[reg.model.isin(["adaptive_pchip", "adaptive_linear"])].groupby(["regime", "model"])[["eP_tau08", "eF_tau08", "eF_knot1", "err_precision_direct",
                                                                                                   "err_precision_reconstructed", "mad_pr"]].mean().unstack("model")
    pr.columns = [f"{a}_{b.replace('adaptive_', '')}" for a, b in pr.columns]
    pr = pr.join(FL[FL.density == "rho_train"].set_index("regime")[["eP_tau08"]].rename(columns={"eP_tau08": "eP_empirical_floor"}))
    C.write_table(pr.reset_index(), out / "tables/expD_per_regime", "Experiment D: per-regime consistency and precision errors (seed mean)")
    dv = DV.groupby(["model", "operating_point_from"])["macro_execution_f1"].agg(["mean", "std"]).reset_index()
    C.write_table(dv, out / "tables/expD_reconstructed_precision_operating_point",
                  "Experiment D (exploratory, no training): macro execution F1 when the F1-optimal threshold is derived from predicted TPR "
                  "and the directly predicted precision vs. the precision reconstructed from TPR, FPPI and rho_train")
    log(S.round(4).to_string())
    log(dv.round(4).to_string())

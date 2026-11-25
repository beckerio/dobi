# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Experiment B - functional equivalence of data conditions in the detector's behavior space.

Conditions
  regime level   : the 18 proxy regimes (stored sweeps of the train / val / test subsets; paper protocol)
  severity level : pristine + 17 degradations x severities 1..5 (85 + 1), sweeps recomputed from the stored
                   detections with the unchanged matching/sweep code (visu_behavior_space.sweep_for_variants);
                   monitor predictions exist for severities 3-5 (cached test variants)
Behavior of a condition: TPR, FPR, FPPI, precision (and F1) as functions of tau on a 201-point grid in [0, 1].

Distance metrics (sensitivity analysis)
  std_l2        primary. sqrt(mean_c mean_tau ((f_c^a - f_c^b) / s_c)^2), c in {TPR, precision, FPR, log(1+FPPI)};
                s_c = RMS (over tau) of the across-regime std of component c on the 18 TRAINING regimes
  std_sup       max_c max_tau |f_c^a - f_c^b| / s_c
  paper_rms     RMS of [TPR, precision, F1, log1p(min(FPPI, 2))/log1p(2)] (behavior vector of the paper's Fig. 2)
  geometric_mad mean of the ROC / FROC / PR MADs between the two conditions' curves (paper curve metric)
  decision      symmetric F1 cross-regret: max(F1_a(t_a) - F1_a(t_b), F1_b(t_b) - F1_b(t_a)), t = F1-optimal threshold
Equivalence tolerance eps (per metric): 95th percentile over conditions of the distance between the SAME condition
measured on the training and on the validation split (sampling variability); no test data are used for eps or s_c.
Two conditions are called approximately equivalent iff their distance is <= eps on the training split AND on the
test split (replication on disjoint images). This is approximate similarity under a tolerance, not exact equality.
"""
from __future__ import annotations

import sqlite3
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dobi.data.feature_cache import FeatureCache
from dobi.eval.evaluation_curve_monitor import tau_curves_to_geometric
from dobi.eval.extended import common as C
from dobi.utils.tau_spline import safe_f1
from dobi.utils.tau_targets import dedupe_sweep, geometric_targets_from_sweep, load_raw_sweep, get_split_subsets, oracle_threshold, sweep_at_tau
from dobi.visualization.visu_behavior_space import FAMILIES, family_of, sweep_for_variants

TAU201 = np.linspace(0.0, 1.0, 201)
GRID = np.linspace(0.0, 1.0, 201, dtype=np.float32)
FGRID = np.linspace(0.0, C.U_MAX, 201, dtype=np.float32)
METRICS = ("std_l2", "std_sup", "paper_rms", "geometric_mad", "decision")
SPLITS = ("train", "val", "test")
N_PERM, PERM_SEED = 10000, 20261011
DEGS = sorted(m for f, ms in FAMILIES.items() if f != "pristine" for m in ms)


# ------------------------------------------------------------------
def behavior_from_sweep(raw):
    """raw: sweep dict in stored or generate_curves order -> behavior record."""
    raw = {k: np.asarray(v, np.float64) for k, v in raw.items()}
    asc = dedupe_sweep(raw)
    s = sweep_at_tau(asc, TAU201)
    geo = geometric_targets_from_sweep(raw, GRID, FGRID, GRID)
    t_opt = oracle_threshold(raw, "f1")
    return {"tpr": s["tpr"], "fpr": s["fpr"], "fppi": s["fppi"], "precision": s["precision"], "f1": s["f1"],
            "roc": geo["roc_tpr"], "froc": geo["froc_tpr"], "pr": geo["pr_precision"], "tau_opt": t_opt,
            "f1_opt": float(sweep_at_tau(asc, np.array([t_opt]))["f1"][0]), "asc": asc}


def behavior_from_predicted(F):
    """F: dict of averaged predicted functions on the 1001 grid (1-D) -> behavior record (same fields)."""
    f = {k: np.asarray(F[k], np.float64)[::5] for k in C.FUNC_KEYS}
    f1 = safe_f1(f["precision"], f["tpr"])
    c = {"tpr_tau": F["tpr"], "recall_tau": F["tpr"], "fpr_tau": F["fpr"], "fppi_tau": F["fppi"], "precision_tau": F["precision"]}
    roc, froc, pr = tau_curves_to_geometric(c, GRID, FGRID, GRID)
    f1d = safe_f1(np.asarray(F["precision"], np.float64), np.asarray(F["tpr"], np.float64))
    q = int(np.argmax(f1d))
    asc = {"thr": C.TAU.astype(np.float64), "tpr": np.asarray(F["tpr"], np.float64), "recall": np.asarray(F["tpr"], np.float64),
           "precision": np.asarray(F["precision"], np.float64), "fppi": np.asarray(F["fppi"], np.float64), "fpr": np.asarray(F["fpr"], np.float64)}
    return {**f, "f1": f1, "roc": roc, "froc": froc, "pr": pr, "tau_opt": float(C.TAU[q]), "f1_opt": float(f1d[q]), "asc": asc}


def f1_at(rec, t):
    return float(sweep_at_tau(rec["asc"], np.array([t]))["f1"][0]) if "num_pred" in rec["asc"] else \
        float(safe_f1(rec["asc"]["precision"], rec["asc"]["tpr"])[min(np.searchsorted(rec["asc"]["thr"], t), len(rec["asc"]["thr"]) - 1)])


def scales(train_regimes):
    comps = {"tpr": [], "precision": [], "fpr": [], "lfppi": []}
    for b in train_regimes:
        comps["tpr"].append(b["tpr"])
        comps["precision"].append(b["precision"])
        comps["fpr"].append(b["fpr"])
        comps["lfppi"].append(np.log1p(b["fppi"]))
    return {k: float(np.sqrt(np.mean(np.var(np.stack(v), axis=0)))) for k, v in comps.items()}


def comp(b):
    return {"tpr": b["tpr"], "precision": b["precision"], "fpr": b["fpr"], "lfppi": np.log1p(b["fppi"])}


def distance(a, b, metric, s):
    if metric in ("std_l2", "std_sup"):
        ca, cb = comp(a), comp(b)
        z = np.stack([(ca[k] - cb[k]) / s[k] for k in ca])
        return float(np.sqrt(np.mean(z ** 2))) if metric == "std_l2" else float(np.max(np.abs(z)))
    if metric == "paper_rms":
        def v(x):
            return np.concatenate([x["tpr"], x["precision"], x["f1"], np.log1p(np.minimum(x["fppi"], 2.0)) / np.log1p(2.0)])
        return float(np.sqrt(np.mean((v(a) - v(b)) ** 2)))
    if metric == "geometric_mad":
        return float(np.mean([np.mean(np.abs(a[k] - b[k])) for k in ("roc", "froc", "pr")]))
    if metric == "decision":
        return float(max(a["f1_opt"] - f1_at(a, b["tau_opt"]), b["f1_opt"] - f1_at(b, a["tau_opt"])))
    raise ValueError(metric)


def dmatrix(recs, names, metric, s):
    n = len(names)
    D = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        D[i, j] = D[j, i] = distance(recs[names[i]], recs[names[j]], metric, s)
    return D


def deg_of(name):
    if name == "pristine":
        return "pristine"
    return name.rsplit("_s", 1)[0] if "_s" in name and name.rsplit("_s", 1)[1].isdigit() else name.replace("_3_5", "")


def pair_class(a, b):
    if "pristine" in (a, b):
        return "pristine_pair"
    if deg_of(a) == deg_of(b):
        return "same_degradation"
    return "within_family" if family_of(deg_of(a)) == family_of(deg_of(b)) else "cross_family"


# ------------------------------------------------------------------
def _sweep_job(args):
    split, deg, sev = args
    conn = sqlite3.connect(f"file:{C.DB}?mode=ro", uri=True)
    try:
        return args, sweep_for_variants(conn, split, deg, sev, C.CLASS_ID, C.IOU_THR)
    finally:
        conn.close()


def load_conditions(log, workers=8):
    """Regime-level and severity-level behavior records for all splits."""
    from multiprocessing import get_context
    reg, sev = {sp: {} for sp in SPLITS}, {sp: {} for sp in SPLITS}
    conn = sqlite3.connect(f"file:{C.DB}?mode=ro", uri=True)
    for sp in SPLITS:
        for s in get_split_subsets(conn, sp, C.CLASS_ID, C.IOU_THR):
            reg[sp][s["name"]] = behavior_from_sweep(load_raw_sweep(conn, s["subset_id"], C.CLASS_ID, C.IOU_THR))
        sev[sp]["pristine"] = reg[sp]["pristine"]
    conn.close()
    jobs = [(sp, d, k) for sp in SPLITS for d in DEGS for k in range(1, 6)]
    with get_context("fork").Pool(workers) as pool:
        for n, ((sp, d, k), sw) in enumerate(pool.imap_unordered(_sweep_job, jobs)):
            if sw is not None:
                sev[sp][f"{d}_s{k}"] = behavior_from_sweep(sw)
            if n % 40 == 0:
                log(f"B: severity sweeps {n + 1}/{len(jobs)}")
    return reg, sev


def ordered(names):
    fam_order = ["pristine", "noise", "blur", "weather", "digital/photometric"]
    return sorted(names, key=lambda n: (fam_order.index(family_of(deg_of(n))), deg_of(n), n))


# ------------------------------------------------------------------
def run(out: Path, workers: int = 20, log=print):
    t0 = time.time()
    reg, sev = load_conditions(log)
    log(f"B: conditions loaded ({time.time() - t0:.0f}s): regime {len(reg['test'])}, severity {len(sev['test'])}")
    s = scales(list(reg["train"].values()))
    C.write_json({"scales_from_training_regimes": s, "metrics": METRICS, "tau_grid": "201 points in [0,1]",
                  "tolerance": "95th percentile of same-condition train-vs-val distance", "n_perm": N_PERM, "perm_seed": PERM_SEED},
                 out / "configs/expB_settings.json")

    results = {"scales": s, "levels": {}}
    long_rows, eq_rows, nn_rows, cls_rows = [], [], [], []
    mats = {}
    for level, data in (("regime", reg), ("severity", sev)):
        names = ordered([n for n in data["test"] if all(n in data[sp] for sp in SPLITS)])
        lv = {"conditions": names, "tolerance": {}, "classes": {}}
        for metric in METRICS:
            Dsp = {sp: dmatrix(data[sp], names, metric, s) for sp in SPLITS}
            mats[(level, metric)] = Dsp
            same = np.array([distance(data["train"][n], data["val"][n], metric, s) for n in names])
            eps = float(np.percentile(same, 95))
            lv["tolerance"][metric] = {"eps": eps, "same_condition_train_val_median": float(np.median(same)),
                                       "same_condition_train_val_max": float(same.max())}
            iu = np.triu_indices(len(names), 1)
            for i, j in zip(*iu):
                long_rows.append({"level": level, "metric": metric, "a": names[i], "b": names[j], "class": pair_class(names[i], names[j]),
                                  **{f"d_{sp}": Dsp[sp][i, j] for sp in SPLITS}, "eps": eps})
            # class separation on test
            cls = np.array([pair_class(names[i], names[j]) for i, j in zip(*iu)])
            dt = Dsp["test"][iu]
            for c in ("pristine_pair", "same_degradation", "within_family", "cross_family"):
                if (cls == c).any():
                    cls_rows.append({"level": level, "metric": metric, "class": c, "n_pairs": int((cls == c).sum()),
                                     "median": float(np.median(dt[cls == c])), "p10": float(np.percentile(dt[cls == c], 10)),
                                     "p90": float(np.percentile(dt[cls == c], 90)),
                                     "fraction_within_eps_test": float(np.mean(dt[cls == c] <= eps)),
                                     "fraction_within_eps_train_and_test": float(np.mean((dt[cls == c] <= eps) & (Dsp["train"][iu][cls == c] <= eps)))})
            w, x = dt[cls == "within_family"], dt[cls == "cross_family"]
            auc = float(np.mean(w[:, None] < x[None, :]) + 0.5 * np.mean(w[:, None] == x[None, :]))
            lv["classes"][metric] = {"auroc_within_vs_cross_family": auc}
            # nearest neighbour among other degradation types
            fam = np.array([family_of(deg_of(n)) for n in names])
            degs = np.array([deg_of(n) for n in names])
            Dt = Dsp["test"].copy()
            nn = []
            for i in range(len(names)):
                mask = degs != degs[i]
                if names[i] == "pristine":
                    continue
                j = np.where(mask)[0][np.argmin(Dt[i, mask])]
                nn.append((i, j))
                nn_rows.append({"level": level, "metric": metric, "condition": names[i], "family": fam[i], "nearest_other_degradation": names[j],
                                "nn_family": fam[j], "same_family": fam[i] == fam[j], "d_test": Dt[i, j],
                                "d_train": Dsp["train"][i, j], "within_eps_train_and_test": bool(Dt[i, j] <= eps and Dsp["train"][i, j] <= eps)})
            obs = np.mean([fam[i] == fam[j] for i, j in nn])
            # permutation null: shuffle family labels across degradation types
            rng = np.random.default_rng(PERM_SEED)
            udeg = sorted(set(degs) - {"pristine"})
            dfam = {d: family_of(d) for d in udeg}
            labels = np.array([dfam[d] for d in udeg])
            null = []
            for _ in range(N_PERM):
                perm = dict(zip(udeg, rng.permutation(labels)))
                pf = np.array([perm.get(d, "pristine") for d in degs])
                null.append(np.mean([pf[i] == pf[j] for i, j in nn]))
            null = np.array(null)
            lv["classes"][metric].update({"nn_same_family_fraction": float(obs), "nn_same_family_null_mean": float(null.mean()),
                                          "nn_permutation_p_value": float((np.sum(null >= obs) + 1) / (N_PERM + 1))})
        results["levels"][level] = lv
        log(f"B: {level} distances done ({time.time() - t0:.0f}s)")
    L = pd.DataFrame(long_rows)
    L.to_csv(C.new_file(out / "metrics/expB_pairwise_distances.csv.gz"), index=False, compression="gzip")
    pd.DataFrame(nn_rows).to_csv(C.new_file(out / "metrics/expB_nearest_neighbours.csv"), index=False)
    C.write_table(pd.DataFrame(cls_rows), out / "tables/expB_distance_by_pair_class",
                  "Experiment B: test-split behavior distances by pair class; eps = 95th percentile of same-condition train-vs-val distance")

    # metric dependence: rank agreement of distance matrices (test) and of equivalent-pair sets
    md = []
    for level in ("regime", "severity"):
        for m1, m2 in combinations(METRICS, 2):
            a = mats[(level, m1)]["test"]
            b = mats[(level, m2)]["test"]
            iu = np.triu_indices(len(a), 1)
            e1 = L[(L.level == level) & (L.metric == m1)]
            e2 = L[(L.level == level) & (L.metric == m2)]
            s1 = set(map(tuple, e1[(e1.d_test <= e1.eps) & (e1.d_train <= e1.eps) & (e1["class"] == "cross_family")][["a", "b"]].values))
            s2 = set(map(tuple, e2[(e2.d_test <= e2.eps) & (e2.d_train <= e2.eps) & (e2["class"] == "cross_family")][["a", "b"]].values))
            md.append({"level": level, "metric_1": m1, "metric_2": m2, "spearman_test_distances": spearmanr(a[iu], b[iu]).correlation,
                       "n_equivalent_cross_family_1": len(s1), "n_equivalent_cross_family_2": len(s2),
                       "jaccard_equivalent_sets": len(s1 & s2) / max(len(s1 | s2), 1)})
    C.write_table(pd.DataFrame(md), out / "tables/expB_metric_dependence",
                  "Experiment B: agreement between distance metrics (Spearman of test pairwise distances; Jaccard of replicated cross-family "
                  "equivalent pairs)")

    # replicated cross-family equivalences per metric
    for level in ("regime", "severity"):
        for metric in METRICS:
            e = L[(L.level == level) & (L.metric == metric) & (L["class"] == "cross_family")]
            ok = e[(e.d_test <= e.eps) & (e.d_train <= e.eps)].sort_values("d_test")
            for _, r in ok.iterrows():
                eq_rows.append({"level": level, "metric": metric, "a": r.a, "b": r.b, "family_a": family_of(deg_of(r.a)),
                                "family_b": family_of(deg_of(r.b)), "d_train": r.d_train, "d_val": r.d_val, "d_test": r.d_test, "eps": r.eps})
    EQ = pd.DataFrame(eq_rows, columns=["level", "metric", "a", "b", "family_a", "family_b", "d_train", "d_val", "d_test", "eps"])
    C.write_table(EQ, out / "tables/expB_cross_family_equivalent_pairs",
                  "Experiment B: cross-family condition pairs whose behavior distance is within the tolerance eps on the training AND "
                  "the test split (approximate functional equivalence)")
    # cross-family nearest-neighbour table (primary metric, severity level)
    NN = pd.DataFrame(nn_rows)
    nnt = NN[(NN.metric == "std_l2") & (NN.level == "severity") & (~NN.same_family)].sort_values("d_test")
    C.write_table(nnt, out / "tables/expB_cross_family_nearest_neighbours",
                  "Experiment B (std_l2, severity level): conditions whose nearest neighbour among other degradation types belongs to a "
                  "different family")

    # ---- ground truth vs predicted distances ----
    cache = FeatureCache(C.FEATURE_CACHE, "test")
    test = C.load_split_context("test")
    sevs, degn = test.meta["severity"], test.meta["deg_name"]
    groups = {"regime": {r: test.regimes == r for r in test.reg_names},
              "severity": {**{"pristine": test.regimes == "pristine"},
                           **{f"{d}_s{k}": (degn == d) & (sevs == k) & (test.regimes != "pristine") for d in DEGS for k in (3, 4, 5)}}}
    pred = {m: {lv: {} for lv in groups} for m in C.LEARNED}
    for pol in C.LEARNED:
        acc = {lv: {n: {k: [] for k in C.FUNC_KEYS} for n in g} for lv, g in groups.items()}
        for sd in C.SEEDS:
            F = C.predict_functions(C.checkpoint_path(pol, sd), cache, test.table.variant_ids)
            for lv, g in groups.items():
                for n, m in g.items():
                    for k in C.FUNC_KEYS:
                        acc[lv][n][k].append(F[k][m].mean(0))
            del F
        for lv, g in groups.items():
            for n in g:
                pred[pol][lv][n] = behavior_from_predicted({k: np.mean(acc[lv][n][k], 0) for k in C.FUNC_KEYS})
    gp = []
    for level, data in (("regime", reg), ("severity", sev)):
        names = ordered([n for n in groups[level] if n in data["test"]])
        iu = np.triu_indices(len(names), 1)
        degs = np.array([deg_of(n) for n in names])
        for metric in METRICS:
            Dg = dmatrix(data["test"], names, metric, s)
            rng = np.random.default_rng(PERM_SEED)
            for pol in C.LEARNED:
                Dp = dmatrix(pred[pol][level], names, metric, s)
                rho = spearmanr(Dg[iu], Dp[iu]).correlation
                null = []
                for _ in range(2000):
                    p = rng.permutation(len(names))
                    null.append(spearmanr(Dg[iu], Dp[p][:, p][iu]).correlation)
                top1, top3 = [], []
                for i in range(len(names)):
                    if names[i] == "pristine":
                        continue
                    mask = degs != degs[i]
                    cand = np.where(mask)[0]
                    g_ord = cand[np.argsort(Dg[i, cand])]
                    p_ord = cand[np.argsort(Dp[i, cand])]
                    top1.append(g_ord[0] == p_ord[0])
                    top3.append(len(set(g_ord[:3]) & set(p_ord[:3])) / 3)
                gp.append({"level": level, "metric": metric, "model": pol, "n_conditions": len(names), "spearman_gt_vs_pred": rho,
                           "mantel_p_value": float((np.sum(np.array(null) >= rho) + 1) / 2001),
                           "pearson_gt_vs_pred": float(np.corrcoef(Dg[iu], Dp[iu])[0, 1]),
                           "nn_top1_agreement": float(np.mean(top1)), "nn_top3_overlap": float(np.mean(top3)),
                           "median_abs_distance_error": float(np.median(np.abs(Dg[iu] - Dp[iu]))),
                           "median_gt_distance": float(np.median(Dg[iu]))})
                if pol == "adaptive_pchip":
                    mats[(level, metric, "pred_pchip")] = (names, Dg, Dp)
    GP = pd.DataFrame(gp)
    C.write_table(GP, out / "tables/expB_gt_vs_predicted_distances",
                  "Experiment B: agreement of ground-truth (test) and predicted behavior distances; predicted condition behavior = mean "
                  "per-image prediction over the condition's test variants and the 3 seeds; Mantel p from 2,000 label permutations")

    # ---- offline coverage diagnostic for the existing held-out runs (no training) ----
    gen = pd.read_csv('results/main_study/paper_artifacts/aggregate/generalization_results.csv')
    cov = []
    for h in ("fog_3_5", "gaussian_noise_3_5", "glass_blur_3_5"):
        others = [r for r in reg["train"] if r != h]
        d = {r: distance(reg["train"][h], reg["train"][r], "std_l2", s) for r in others}
        nn = min(d, key=d.get)
        row = {"held_out_regime": h, "nearest_training_regime": nn, "gt_distance_to_nearest_training_regime_std_l2": d[nn],
               "eps_regime_std_l2": results["levels"]["regime"]["tolerance"]["std_l2"]["eps"]}
        for model in ("held_out_pchip", "held_out_mean_behavior", "in_distribution_pchip"):
            g = gen[(gen.model == model) & (gen.held_out_regime == h)]
            row[f"{model}_oracle_regret_rcu"] = float(g.oracle_regret.mean()) if len(g) else np.nan
            row[f"{model}_mad_pr"] = float(g.mad_pr.mean()) if len(g) and "mad_pr" in g else np.nan
        cov.append(row)
    C.write_table(pd.DataFrame(cov), out / "tables/expE_offline_coverage_diagnostic",
                  "Offline diagnostic (existing held-out runs, no training): ground-truth behavior distance of each held-out regime to its "
                  "nearest training regime vs. held-out regret (regime-curve utility, ‡). Uses ground-truth curves: NOT a deployable "
                  "coverage estimator; n = 3")

    # ---- save matrices and condition curves for figures ----
    save = {}
    for (level, metric, *rest), v in mats.items():
        if rest:
            names, Dg, Dp = v
            save[f"{level}__{metric}__gt_test"] = Dg
            save[f"{level}__{metric}__pred_pchip"] = Dp
            save[f"{level}__{metric}__pred_names"] = np.array(names)
        else:
            for sp, D in v.items():
                save[f"{level}__{metric}__{sp}"] = D
    for level, data in (("regime", reg), ("severity", sev)):
        names = ordered([n for n in data["test"] if all(n in data[sp] for sp in SPLITS)])
        save[f"{level}__names"] = np.array(names)
        for k in ("tpr", "precision", "fppi", "f1"):
            save[f"{level}__test_{k}"] = np.stack([data["test"][n][k] for n in names])
            save[f"{level}__train_{k}"] = np.stack([data["train"][n][k] for n in names])
        save[f"{level}__test_tau_opt"] = np.array([data["test"][n]["tau_opt"] for n in names])
    for k in ("tpr", "precision", "fppi", "f1"):
        nm = list(pred["adaptive_pchip"]["severity"])
        save[f"severity__pred_pchip_names"] = np.array(nm)
        save[f"severity__pred_pchip_{k}"] = np.stack([pred["adaptive_pchip"]["severity"][n][k] for n in nm])
    np.savez_compressed(C.new_file(out / "metrics/expB_matrices_and_curves.npz"), **save)
    results["gt_vs_predicted"] = GP.to_dict(orient="records")
    results["metric_dependence"] = md
    results["n_replicated_cross_family_equivalent_pairs"] = EQ.groupby(["level", "metric"]).size().to_dict() if len(EQ) else {}
    results["offline_coverage"] = cov
    C.write_json(results, out / "metrics/expB_results.json")
    log(pd.DataFrame(cls_rows).round(4).to_string())
    log(GP.round(3).to_string())
    log(f"B finished in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------
def shifted_equivalence(out: Path, log=print):
    """
    Post-analysis on the saved pairwise distances: equivalences between two near-pristine conditions are trivial.
    A condition counts as functionally SHIFTED if its distance to pristine exceeds eps on the training AND the test
    split; cross-family equivalences are then counted only between two shifted conditions.
    """
    L = pd.read_csv(out / "metrics/expB_pairwise_distances.csv.gz")
    rows, pairs = [], []
    for (level, metric), d in L.groupby(["level", "metric"]):
        eps = d.eps.iloc[0]
        pp = d[(d.a == "pristine") | (d.b == "pristine")]
        dist_p = {(r.b if r.a == "pristine" else r.a): (r.d_train, r.d_test) for r in pp.itertuples()}
        shifted = {n for n, (tr, te) in dist_p.items() if tr > eps and te > eps}
        x = d[d.a.isin(shifted) & d.b.isin(shifted)]
        cf = x[x["class"] == "cross_family"]
        eq = cf[(cf.d_train <= eps) & (cf.d_test <= eps)]
        rows.append({"level": level, "metric": metric, "eps": eps, "n_conditions": len(dist_p) + 1, "n_shifted": len(shifted),
                     "n_cross_family_pairs_shifted": len(cf), "n_equivalent_shifted": len(eq),
                     "fraction_equivalent_shifted": len(eq) / max(len(cf), 1),
                     "fraction_equivalent_within_family_shifted": float(np.mean((x[x["class"] == "within_family"].d_train <= eps) &
                                                                                (x[x["class"] == "within_family"].d_test <= eps)))
                     if (x["class"] == "within_family").any() else np.nan})
        for r in eq.sort_values("d_test").itertuples():
            pairs.append({"level": level, "metric": metric, "a": r.a, "b": r.b, "family_a": family_of(deg_of(r.a)), "family_b": family_of(deg_of(r.b)),
                          "d_train": r.d_train, "d_test": r.d_test, "eps": eps, "a_dist_to_pristine_test": dist_p[r.a][1],
                          "b_dist_to_pristine_test": dist_p[r.b][1]})
    C.write_table(pd.DataFrame(rows), out / "tables/expB_shifted_equivalence",
                  "Experiment B: replicated cross-family equivalences restricted to conditions that are themselves shifted from pristine "
                  "(distance to pristine > eps on train and test)")
    C.write_table(pd.DataFrame(pairs), out / "tables/expB_shifted_equivalent_pairs",
                  "Experiment B: cross-family equivalent pairs (d <= eps on train and test) between two shifted conditions")
    log(pd.DataFrame(rows).round(3).to_string())


def severity_compression(out: Path, log=print):
    """GT vs predicted (PCHIP) severity-level distances by pair class, from the saved matrices."""
    Z = np.load(out / "metrics/expB_matrices_and_curves.npz")
    rows = []
    for metric in METRICS:
        names = list(Z[f"severity__{metric}__pred_names"])
        g, p = Z[f"severity__{metric}__gt_test"], Z[f"severity__{metric}__pred_pchip"]
        iu = np.triu_indices(len(names), 1)
        cls = np.array([pair_class(names[i], names[j]) for i, j in zip(*iu)])
        for c in ("same_degradation", "within_family", "cross_family", "pristine_pair", "all_but_same_degradation"):
            m = cls != "same_degradation" if c == "all_but_same_degradation" else cls == c
            rows.append({"metric": metric, "pair_class": c, "n_pairs": int(m.sum()), "median_gt": float(np.median(g[iu][m])),
                         "median_pred_pchip": float(np.median(p[iu][m])), "spearman": spearmanr(g[iu][m], p[iu][m]).correlation})
    C.write_table(pd.DataFrame(rows), out / "tables/expB_severity_compression",
                  "Experiment B: ground-truth vs predicted (PCHIP) distances between severity-level test conditions by pair class; the "
                  "monitor was trained on regime-level targets shared by severities 3-5")
    log(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    import sys
    run = C.check_out_dir(Path(sys.argv[1]))
    if len(sys.argv) > 2 and sys.argv[2] == "severity_compression":
        severity_compression(run)
    else:
        shifted_equivalence(run)

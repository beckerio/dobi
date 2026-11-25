# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Experiment A - single-image vs. batch behavior inference.

For each test regime, batch size b and repetition j, b DISTINCT base images are drawn without replacement
(one randomly chosen variant per base image, so variants of one original image never appear twice in a batch).
The per-image predicted threshold functions (TPR, FPR, FPPI, precision) of the batch are averaged and
evaluated exactly like a single-image prediction: ROC/FROC/PR MAD against the regime target with the
unchanged conversion, and the F1-optimal operating threshold of the averaged function.

References (not monitors):
  mean_behavior        image-independent prior (constant; batch size irrelevant)
  labelled_batch       the empirical behavior of the same b images computed from their own detections AND
                       ground truth (needs labels; shows how many labelled images give the same accuracy)
  all_images           the averaged prediction over all test variants of the regime (b -> full regime)
Operating-point regret is the regime-curve utility (F1 of the regime curve at the batch threshold), marked ‡.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from dobi.data.feature_cache import FeatureCache
from dobi.eval.extended import common as C
from dobi.utils.tau_spline import safe_f1
from dobi.utils.tau_targets import oracle_threshold, sweep_at_tau

BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def sample_batches(ctx: C.SplitContext, batch_sizes, n_rep: int, seed: int):
    """{(regime, b): int array [n_rep, b]} of variant row indices with distinct base images per batch."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in ctx.reg_names:
        rows = np.where(ctx.regimes == r)[0]
        bases = ctx.meta["image_id"][rows]
        order = np.argsort(bases, kind="stable")
        ub, start = np.unique(bases[order], return_index=True)
        groups = np.split(rows[order], start[1:])
        for b in batch_sizes:
            sel = np.empty((n_rep, b), np.int64)
            for j in range(n_rep):
                pick = rng.choice(len(ub), size=b, replace=False)
                sel[j] = [g[rng.integers(len(g))] for g in (groups[k] for k in pick)]
            out[(r, b)] = sel
    return out


def evaluate_batches(funcs_batched, ridx, ctx, tau_oracle, f1_oracle, workers):
    """funcs_batched: dict key -> [M, Q] averaged functions; ridx [M]."""
    mads = C.geometric_mads(funcs_batched, ridx, ctx.targets, workers=workers)
    tau_hat = C.argmax_f1_tau(funcs_batched["tpr"], funcs_batched["precision"])
    f1_rc = np.empty(len(ridx))
    for k, r in enumerate(ctx.reg_names):
        m = ridx == k
        f1_rc[m] = sweep_at_tau(ctx.asc[r], tau_hat[m].astype(np.float64))["f1"]
    to = np.array([tau_oracle[r] for r in ctx.reg_names])[ridx]
    fo = np.array([f1_oracle[r] for r in ctx.reg_names])[ridx]
    return pd.DataFrame({"mad_roc": mads[:, 0], "mad_froc": mads[:, 1], "mad_pr": mads[:, 2], "tau_hat": tau_hat,
                         "abs_tau_error": np.abs(tau_hat - to), "regret_rcu": fo - f1_rc})


def run(out: Path, n_rep: int = 500, seed: int = 20261011, workers: int = 20, ctx=None, log=print):
    t0 = time.time()
    ctx = ctx or C.load_split_context("test", with_targets=True)
    reg_names = ctx.reg_names
    tau_oracle = {r: oracle_threshold(ctx.raw[r], "f1") for r in reg_names}
    f1_oracle = {r: float(sweep_at_tau(ctx.asc[r], np.array([tau_oracle[r]]))["f1"][0]) for r in reg_names}
    batches = sample_batches(ctx, BATCH_SIZES, n_rep, seed)
    np.savez_compressed(C.new_file(out / "metrics/expA_batch_indices.npz"),
                        **{f"{r}__b{b}": v for (r, b), v in batches.items()},
                        variant_ids=ctx.table.variant_ids)
    keys = list(batches)
    sel_all = [batches[k] for k in keys]
    ridx_all = np.concatenate([np.full(len(batches[k]), reg_names.index(k[0])) for k in keys])
    b_all = np.concatenate([np.full(len(batches[k]), k[1]) for k in keys])
    rep_all = np.concatenate([np.arange(len(batches[k])) for k in keys])
    meta_cols = pd.DataFrame({"regime": np.array(reg_names)[ridx_all], "batch_size": b_all, "rep": rep_all})

    frames = []
    spread = []   # between-batch variability of the averaged predicted functions

    def averaged(F):
        return {k: np.concatenate([F[k][s].mean(axis=1) for s in sel_all]) for k in C.FUNC_KEYS}

    cache = FeatureCache(C.FEATURE_CACHE, "test")
    all_img_rows = []
    for pol in C.LEARNED:
        for s in C.SEEDS:
            F = C.predict_functions(C.checkpoint_path(pol, s), cache, ctx.table.variant_ids)
            A = averaged(F)
            df = evaluate_batches(A, ridx_all, ctx, tau_oracle, f1_oracle, workers)
            frames.append(pd.concat([meta_cols, df], axis=1).assign(model=pol, seed=s))
            for (r, b), sl in zip(keys, np.split(np.arange(len(ridx_all)), np.cumsum([len(batches[k]) for k in keys])[:-1])):
                spread.append({"model": pol, "seed": s, "regime": r, "batch_size": b,
                               **{f"sd_{k}": float(A[k][sl].std(axis=0).mean()) for k in C.FUNC_KEYS},
                               "sd_tau_hat": float(df["tau_hat"].values[sl].std())})
            # asymptote: average over all variants of the regime
            full = {k: np.stack([F[k][ctx.regimes == r].mean(axis=0) for r in reg_names]) for k in C.FUNC_KEYS}
            d = evaluate_batches(full, np.arange(len(reg_names)), ctx, tau_oracle, f1_oracle, 1)
            all_img_rows.append(d.assign(model=pol, seed=s, regime=reg_names))
            log(f"A: {pol} s{s} done ({time.time() - t0:.0f}s)")
            del F, A

    # labelled empirical batches (ground truth of the batch images)
    tp, fp = C.per_image_grid_counts(ctx.table)
    E = {k: [] for k in C.FUNC_KEYS}
    for sl in sel_all:
        e = C.empirical_functions(tp[sl].sum(axis=1), fp[sl].sum(axis=1), ctx.table.n_gt[sl].sum(axis=1),
                                  np.full(len(sl), sl.shape[1]))
        for k in C.FUNC_KEYS:
            E[k].append(e[k].astype(np.float32))
    E = {k: np.concatenate(v) for k, v in E.items()}
    df = evaluate_batches(E, ridx_all, ctx, tau_oracle, f1_oracle, workers)
    frames.append(pd.concat([meta_cols, df], axis=1).assign(model="labelled_batch", seed=-1))
    for (r, b), sl in zip(keys, np.split(np.arange(len(ridx_all)), np.cumsum([len(batches[k]) for k in keys])[:-1])):
        spread.append({"model": "labelled_batch", "seed": -1, "regime": r, "batch_size": b,
                       **{f"sd_{k}": float(E[k][sl].std(axis=0).mean()) for k in C.FUNC_KEYS},
                       "sd_tau_hat": float(df["tau_hat"].values[sl].std())})
    # conversion floor: the full regime empirical function through the same tau-grid conversion
    full_e = C.empirical_functions(np.stack([tp[ctx.regimes == r].sum(0) for r in reg_names]),
                                   np.stack([fp[ctx.regimes == r].sum(0) for r in reg_names]),
                                   np.array([ctx.table.n_gt[ctx.regimes == r].sum() for r in reg_names]),
                                   np.array([(ctx.regimes == r).sum() for r in reg_names]))
    floor = evaluate_batches({k: v.astype(np.float32) for k, v in full_e.items()}, np.arange(len(reg_names)), ctx,
                             tau_oracle, f1_oracle, 1).assign(model="labelled_all_images", seed=-1, regime=reg_names)
    del tp, fp, E

    # image-independent prior
    mb = C.mean_behavior_functions()
    M = {k: np.repeat(mb[k][None].astype(np.float32), len(reg_names), axis=0) for k in C.FUNC_KEYS}
    dmb = evaluate_batches(M, np.arange(len(reg_names)), ctx, tau_oracle, f1_oracle, 1).assign(model="mean_behavior", seed=-1,
                                                                                               regime=reg_names)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(C.new_file(out / "metrics/expA_batch_raw.csv.gz"), index=False, compression="gzip")
    asym = pd.concat(all_img_rows + [floor, dmb], ignore_index=True)
    asym.to_csv(C.new_file(out / "metrics/expA_reference_per_regime.csv"), index=False)
    SP = pd.DataFrame(spread)
    SP.to_csv(C.new_file(out / "metrics/expA_prediction_spread.csv"), index=False)
    summarize(out, raw, asym, SP, n_rep, seed, log)
    log(f"A finished in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------
METRICS = ("mad_roc", "mad_froc", "mad_pr", "abs_tau_error", "regret_rcu")


def summarize(out: Path, raw: pd.DataFrame, asym: pd.DataFrame, SP: pd.DataFrame, n_rep, seed, log=print):
    # per (model, seed, regime, b): mean over repetitions + between-batch spread
    g = raw.groupby(["model", "seed", "regime", "batch_size"])
    per_regime = g[list(METRICS) + ["tau_hat"]].mean().add_suffix("_mean")
    per_regime = per_regime.join(g[list(METRICS) + ["tau_hat"]].std(ddof=1).add_suffix("_sd_between_batches"))
    per_regime = per_regime.join(g[["mad_pr"]].quantile(0.95).rename(columns={"mad_pr": "mad_pr_p95"}))
    per_regime = per_regime.reset_index()
    per_regime.to_csv(C.new_file(out / "metrics/expA_per_seed_regime.csv"), index=False)

    # macro over regimes per seed, then seed mean/sd
    macro = per_regime.groupby(["model", "seed", "batch_size"])[[f"{m}_mean" for m in METRICS] + ["tau_hat_sd_between_batches",
                                                                                                    "mad_pr_sd_between_batches"]].mean().reset_index()
    macro.to_csv(C.new_file(out / "metrics/expA_macro_per_seed.csv"), index=False)
    rows = []
    for (model, b), d in macro.groupby(["model", "batch_size"], sort=False):
        e = {"model": model, "batch_size": int(b), "n_seeds": len(d)}
        for m in METRICS:
            e[m] = d[f"{m}_mean"].mean()
            e[f"{m}_seed_sd"] = d[f"{m}_mean"].std(ddof=1) if len(d) > 1 else 0.0
        e["sd_tau_hat_between_batches"] = d["tau_hat_sd_between_batches"].mean()
        e["sd_mad_pr_between_batches"] = d["mad_pr_sd_between_batches"].mean()
        rows.append(e)
    # constant / asymptotic references (batch_size = inf)
    for (model), d in asym.groupby("model", sort=False):
        ds = d.groupby("seed")[list(METRICS)].mean()
        e = {"model": f"{model} (all images)" if model in C.LEARNED else model, "batch_size": -1, "n_seeds": len(ds)}
        for m in METRICS:
            e[m] = ds[m].mean()
            e[f"{m}_seed_sd"] = ds[m].std(ddof=1) if len(ds) > 1 else 0.0
        rows.append(e)
    S = pd.DataFrame(rows)
    C.write_table(S, out / "tables/expA_macro_summary",
                  f"Experiment A: macro (18 regimes) curve error and operating-point error vs. batch size (distinct base images per batch; "
                  f"{n_rep} batches per regime and size; sampling seed {seed}); learned: mean over 3 seeds; batch_size -1 = constant or "
                  f"full-regime reference; regret_rcu = regime-curve utility (‡)")

    # relative reduction b=1 -> b=32 and fraction of the gap to the full-regime asymptote
    red = []
    for pol in list(C.LEARNED) + ["labelled_batch"]:
        s1 = S[(S.model == pol) & (S.batch_size == 1)].iloc[0]
        s32 = S[(S.model == pol) & (S.batch_size == 32)].iloc[0]
        a = S[S.model == (f"{pol} (all images)" if pol in C.LEARNED else "labelled_all_images")].iloc[0]
        for m in ("mad_roc", "mad_froc", "mad_pr", "abs_tau_error"):
            red.append({"model": pol, "metric": m, "b1": s1[m], "b32": s32[m], "all_images": a[m],
                        "relative_reduction_b1_to_b32": 1 - s32[m] / s1[m],
                        "fraction_of_reducible_error_removed": (s1[m] - s32[m]) / (s1[m] - a[m]) if s1[m] > a[m] else np.nan})
    C.write_table(pd.DataFrame(red), out / "tables/expA_error_reduction",
                  "Experiment A: error at b=1, b=32 and with all images of the regime; reducible error = b1 minus all-images error")

    # labelled images needed to match the unlabeled monitor (per metric, PCHIP seed mean)
    eq = []
    lab = S[S.model == "labelled_batch"].set_index("batch_size")
    for pol in C.LEARNED:
        for b in (1, 8, 32):
            r = S[(S.model == pol) & (S.batch_size == b)].iloc[0]
            for m in ("mad_roc", "mad_froc", "mad_pr", "abs_tau_error"):
                better = [bb for bb in lab.index if bb > 0 and lab.loc[bb, m] <= r[m]]
                eq.append({"model": pol, "monitor_batch_size": b, "metric": m, "monitor_error": r[m],
                           "smallest_labelled_batch_with_lower_or_equal_error": min(better) if better else ">32"})
    C.write_table(pd.DataFrame(eq), out / "tables/expA_labelled_equivalent",
                  "Experiment A: smallest labelled batch (empirical behavior of b labelled images) whose error is <= the monitor's error")

    # per-regime table PCHIP (seed mean) at b in {1, 8, 32} vs full regime
    pr = per_regime[per_regime.model == "adaptive_pchip"].groupby(["regime", "batch_size"])[["mad_pr_mean", "abs_tau_error_mean",
                                                                                               "tau_hat_sd_between_batches"]].mean().reset_index()
    wide = pr.pivot(index="regime", columns="batch_size")
    wide.columns = [f"{a.replace('_mean', '')}_b{b}" for a, b in wide.columns]
    asym_p = asym[asym.model == "adaptive_pchip"].groupby("regime")[["mad_pr", "abs_tau_error"]].mean().add_suffix("_all_images")
    lab32 = per_regime[(per_regime.model == "labelled_batch") & (per_regime.batch_size == 32)].set_index("regime")[["mad_pr_mean"]].rename(
        columns={"mad_pr_mean": "mad_pr_labelled_b32"})
    W = wide.join(asym_p).join(lab32).reset_index()
    keep = ["regime", "mad_pr_b1", "mad_pr_b8", "mad_pr_b32", "mad_pr_all_images", "mad_pr_labelled_b32", "abs_tau_error_b1",
            "abs_tau_error_b32", "abs_tau_error_all_images", "tau_hat_sd_between_batches_b1", "tau_hat_sd_between_batches_b32"]
    C.write_table(W[keep], out / "tables/expA_per_regime_pchip",
                  "Experiment A, adaptive PCHIP (seed mean): per-regime PR MAD and |tau_hat - tau*| for batch sizes 1/8/32 and all images")

    # leave-one-regime-out sensitivity of the macro PR MAD at b = 1 and 32
    lo = []
    for pol in list(C.LEARNED) + ["labelled_batch"]:
        for b in (1, 32):
            d = per_regime[(per_regime.model == pol) & (per_regime.batch_size == b)].groupby("regime")["mad_pr_mean"].mean()
            vals = [d.drop(r).mean() for r in d.index]
            lo.append({"model": pol, "batch_size": b, "macro_mad_pr": d.mean(), "loro_min": min(vals), "loro_max": max(vals),
                       "most_influential_regime": d.index[int(np.argmax(np.abs(np.array(vals) - d.mean())))]})
    C.write_table(pd.DataFrame(lo), out / "tables/expA_leave_one_regime_out",
                  "Experiment A: leave-one-regime-out range of the macro PR MAD")
    log(S[S.model.isin(["adaptive_pchip", "labelled_batch", "mean_behavior", "adaptive_pchip (all images)"])][
        ["model", "batch_size", "mad_roc", "mad_froc", "mad_pr", "abs_tau_error", "regret_rcu"]].round(4).to_string())


def resummarize(out: Path, log=print):
    """Rebuild all summary tables of experiment A from the saved raw metrics (no model evaluation)."""
    raw = pd.read_csv(out / "metrics/expA_batch_raw.csv.gz")
    asym = pd.read_csv(out / "metrics/expA_reference_per_regime.csv")
    SP = pd.read_csv(out / "metrics/expA_prediction_spread.csv")
    summarize(out, raw, asym, SP, int(raw.rep.max()) + 1, 20261011, log)


if __name__ == "__main__":
    import sys
    resummarize(C.check_out_dir(Path(sys.argv[1])))

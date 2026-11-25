#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The DOBI authors
"""
Aggregate the final paper evaluations (outputs of evaluation_curve_monitor.py --threshold_eval)
into paper_artifacts/: per-seed / per-regime CSVs, policy comparison, paired bootstrap
confidence intervals (resampling base images), curve exports, tables (CSV/MD/LaTeX) and
the main operational figures.

All curve metrics are the existing definitions computed by evaluation_curve_monitor.py
(MAD ROC/FROC/PR, Delta AUROC/AFROC/AUPRC); nothing is recomputed here except
averages, differences and bootstrap resamples of those per-image values.

    python src/dobi/eval/aggregate_paper_results.py --eval_root results/paper/eval --out paper_artifacts
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dobi.utils.curve_plotting import COLORS, FONT_SIZE, PRED_COLOR, PRIMARY_COLOR

SEEDS = (0, 1, 2)
MAIN_MODELS = {
    "adaptive_linear": [f"tau_linear_s{s}" for s in SEEDS],
    "adaptive_pchip": [f"tau_pchip_s{s}" for s in SEEDS],
}
BASELINE_RUN = "mean_behavior"
APPENDIX_MODELS = {"pc1_behavior": [f"pc1_behavior_s{s}" for s in SEEDS]}
ABLATIONS = [  # run, interpolation, K, knot source, aux tau loss
    ("tau_linear_s0", "linear", 16, "uniform", False),
    ("tau_pchip_s0", "pchip", 16, "uniform", False),
    ("abl_pchip_K8_uniform", "pchip", 8, "uniform", False),
    ("abl_pchip_K32_uniform", "pchip", 32, "uniform", False),
    ("abl_pchip_K16_quantile_all", "pchip", 16, "quantile (all scores)", False),
    ("abl_pchip_K16_quantile_tp", "pchip", 16, "quantile (TP scores)", False),
    ("abl_pchip_K16_uniform_auxtau", "pchip", 16, "uniform", True),
]
LODO = {"glass_blur_3_5": "glass_blur", "gaussian_noise_3_5": "gaussian_noise", "fog_3_5": "fog"}
POLICY_ORDER = ["fixed_pristine", "mean_behavior", "adaptive_linear", "adaptive_pchip", "oracle"]
POLICY_LABEL = {"fixed_pristine": "Fixed pristine τ", "mean_behavior": "Mean behavior",
                "adaptive_linear": "Adaptive (linear)", "adaptive_pchip": "Adaptive (PCHIP)", "oracle": "Oracle",
                "pc1_behavior": "PC1 behavior (appendix)"}
POLICY_COLOR = {"fixed_pristine": COLORS[0], "mean_behavior": COLORS[2], "adaptive_linear": COLORS[3],
                "adaptive_pchip": PRED_COLOR, "oracle": PRIMARY_COLOR, "pc1_behavior": COLORS[4]}
SELECTED_REGIMES = ["pristine", "gaussian_noise_3_5", "glass_blur_3_5", "pixelate_3_5", "motion_blur_3_5", "fog_3_5"]
CURVE_KEYS = ["mad_roc", "mad_froc", "mad_pr", "delta_auroc", "delta_afroc", "delta_auprc"]
POL_METRICS = ["f1", "precision", "recall", "fppi", "tp", "fp", "fn"]


def jsonable(x):
    """Recursively replace NaN/inf by None and numpy types by Python types."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def short(name: str) -> str:
    return name.replace("_3_5", "").replace("_", " ")


# ------------------------------------------------------------------
# loading
# ------------------------------------------------------------------
class Run:
    def __init__(self, root: Path, name: str):
        self.name = name
        p = root / name / "curve_evaluation_results"
        self.prefix = p
        self.ok = Path(f"{p}_threshold_summary.json").exists()
        if not self.ok:
            return
        self.rows = pd.read_csv(f"{p}.csv")
        self.rows = self.rows[self.rows["variant"] == "degraded"].copy()
        self.rows["base_image"] = self.rows["rel_path"].map(lambda r: Path(str(r)).stem)
        self.summary = json.loads(Path(f"{p}_summary.json").read_text())
        self.thr = json.loads(Path(f"{p}_threshold_summary.json").read_text())
        self.npz = np.load(f"{p}_tau_curves.npz", allow_pickle=True)


def seed_of(name: str):
    return int(name.rsplit("_s", 1)[1]) if "_s" in name and name.rsplit("_s", 1)[1].isdigit() else None


# ------------------------------------------------------------------
# curve fit
# ------------------------------------------------------------------
def curve_fit_tables(runs: Dict[str, Run], groups: Dict[str, List[str]]):
    per_seed, per_regime = [], []
    for model, names in groups.items():
        for n in names:
            r = runs.get(n)
            if r is None or not r.ok:
                continue
            pc = r.summary["per_cluster"]
            macro = {k: float(np.mean([e[k] for e in pc])) for k in CURVE_KEYS}
            macro.update({f"abs_{k}": float(np.mean([e[f"abs_{k}"] for e in pc])) for k in CURVE_KEYS if k.startswith("delta")})
            micro = r.summary["global_metrics"]
            prist = next(e for e in pc if e["cluster_name"] == "pristine")
            row = {"model": model, "run": n, "seed": seed_of(n), "n_regimes": len(pc), "n_images": int(len(r.rows))}
            row.update({f"macro_{k}": v for k, v in macro.items()})
            row.update({f"micro_{k}": micro[k] for k in CURVE_KEYS})
            row.update({f"micro_abs_{k}": micro[f"abs_{k}"] for k in CURVE_KEYS if k.startswith("delta")})
            row.update({f"pristine_{k}": prist[k] for k in ("mad_roc", "mad_froc", "mad_pr")})
            per_seed.append(row)
            for e in pc:
                per_regime.append({"model": model, "run": n, "seed": seed_of(n), "regime": e["cluster_name"],
                                   **{k: e[k] for k in CURVE_KEYS},
                                   **{f"abs_{k}": e[f"abs_{k}"] for k in CURVE_KEYS if k.startswith("delta")}})
    ps = pd.DataFrame(per_seed)
    summ = []
    for model, g in ps.groupby("model", sort=False):
        e = {"model": model, "n_seeds": len(g)}
        for c in [c for c in ps.columns if c.startswith(("macro_", "micro_", "pristine_"))]:
            e[f"{c}_mean"] = float(g[c].mean())
            e[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        summ.append(e)
    return ps, pd.DataFrame(per_regime), pd.DataFrame(summ)


# ------------------------------------------------------------------
# threshold policies
# ------------------------------------------------------------------
def policy_rows(run: Run, policy_col: str, policy: str, model_seed):
    """Per-regime aggregation of one policy's columns ('pristine' | 'adaptive' | 'oracle') of an eval run."""
    out = []
    for (cid, cname), g in run.rows.groupby(["cluster_id", "cluster_name"]):
        e = {"policy": policy, "run": run.name, "seed": model_seed, "regime": cname, "n_images": len(g)}
        for m in POL_METRICS:
            e[m] = float(g[f"{m}_{policy_col}"].mean())
        tau = g[f"tau_{policy_col}"].astype(float)
        err = tau - g["tau_oracle"].astype(float)
        e.update(tau_mean=float(tau.mean()), tau_std_over_images=float(tau.std(ddof=0)),
                 tau_signed_error=float(err.mean()), tau_abs_error=float(err.abs().mean()),
                 tau_rmse=float(np.sqrt((err ** 2).mean())), tau_oracle=float(g["tau_oracle"].iloc[0]),
                 f1_pristine=float(g["f1_pristine"].mean()), f1_oracle=float(g["f1_oracle"].mean()))
        e["gain_vs_pristine"] = e["f1"] - e["f1_pristine"]
        e["oracle_regret"] = e["f1_oracle"] - e["f1"]
        e["frac_images_improved"] = float(((g[f"f1_{policy_col}"] - g["f1_pristine"]) > 1e-12).mean())
        out.append(e)
    return out


def threshold_tables(runs: Dict[str, Run], groups: Dict[str, List[str]]):
    ref = runs[BASELINE_RUN]
    rows = policy_rows(ref, "pristine", "fixed_pristine", None) + policy_rows(ref, "oracle", "oracle", None)
    rows += policy_rows(ref, "adaptive", "mean_behavior", None)
    for model, names in groups.items():
        for n in names:
            if n in runs and runs[n].ok:
                rows += policy_rows(runs[n], "adaptive", model, seed_of(n))
    pr = pd.DataFrame(rows)
    per_seed = []
    for (pol, run, seed), g in pr.groupby(["policy", "run", "seed"], dropna=False, sort=False):
        prist = g[g["regime"] == "pristine"]
        e = {"policy": pol, "run": run, "seed": None if pd.isna(seed) else int(seed), "n_regimes": len(g)}
        for c in POL_METRICS + ["gain_vs_pristine", "oracle_regret", "tau_abs_error", "tau_signed_error"]:
            e[f"{c}_macro"] = float(g[c].mean())
        e["tau_rmse_macro"] = float(np.sqrt((g["tau_rmse"] ** 2).mean()))
        e["gain_median_over_regimes"] = float(g["gain_vs_pristine"].median())
        e["gain_std_over_regimes"] = float(g["gain_vs_pristine"].std(ddof=0))
        worst = g.loc[g["gain_vs_pristine"].idxmin()]
        e["gain_worst_regime"] = float(worst["gain_vs_pristine"])
        e["worst_regime"] = worst["regime"]
        e["gain_pristine_regime"] = float(prist["gain_vs_pristine"].iloc[0]) if len(prist) else None
        e["n_regimes_positive_gain"] = int((g["gain_vs_pristine"] > 1e-12).sum())
        e["n_regimes_negative_gain"] = int((g["gain_vs_pristine"] < -1e-12).sum())
        per_seed.append(e)
    ps = pd.DataFrame(per_seed)
    summ = []
    for pol, g in ps.groupby("policy", sort=False):
        e = {"policy": pol, "n_seeds": len(g)}
        for c in [c for c in ps.columns if c.endswith(("_macro", "_over_regimes", "_regime")) or c.startswith("n_regimes_")]:
            if not pd.api.types.is_numeric_dtype(g[c]):
                continue
            e[f"{c}_mean"] = float(g[c].mean())
            e[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        e["worst_regimes"] = sorted(set(g["worst_regime"]))
        summ.append(e)
    order = {p: i for i, p in enumerate(POLICY_ORDER + ["pc1_behavior"])}
    summ = sorted(summ, key=lambda e: order.get(e["policy"], 99))
    return pr, ps, pd.DataFrame(summ)


# ------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------
def _per_image_frame(runs, names, value_fn):
    """Average a per-row quantity over seeds; rows keyed by (cluster_id, base_image)."""
    frames = []
    for n in names:
        r = runs[n].rows.set_index(["cluster_id", "base_image"])
        frames.append(value_fn(r).rename(n))
    return pd.concat(frames, axis=1).mean(axis=1)


def bootstrap(runs: Dict[str, Run], groups, n_boot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    ref = runs[BASELINE_RUN].rows.set_index(["cluster_id", "base_image"])
    quantities = {}
    for model, names in list(groups.items()) + [("mean_behavior", [BASELINE_RUN])]:
        names = [n for n in names if n in runs and runs[n].ok]
        quantities[f"gain_vs_pristine[{model}]"] = _per_image_frame(runs, names, lambda r: r["f1_adaptive"] - r["f1_pristine"])
        quantities[f"oracle_regret[{model}]"] = _per_image_frame(runs, names, lambda r: r["f1_oracle"] - r["f1_adaptive"])
        for k in ("mad_roc", "mad_froc", "mad_pr"):
            quantities[f"{k}[{model}]"] = _per_image_frame(runs, names, lambda r, k=k: r[k])
    quantities["oracle_regret[fixed_pristine]"] = ref["f1_oracle"] - ref["f1_pristine"]
    diffs = {}
    for a, b in (("adaptive_pchip", "adaptive_linear"), ("adaptive_pchip", "mean_behavior"),
                 ("adaptive_linear", "mean_behavior")):
        for k in ("oracle_regret", "gain_vs_pristine", "mad_roc", "mad_froc", "mad_pr"):
            if f"{k}[{a}]" in quantities and f"{k}[{b}]" in quantities:
                diffs[f"{k}[{a}] - {k}[{b}]"] = quantities[f"{k}[{a}]"] - quantities[f"{k}[{b}]"]
    for m in ("adaptive_pchip", "adaptive_linear", "mean_behavior"):
        diffs[f"oracle_regret[{m}] - oracle_regret[fixed_pristine]"] = quantities[f"oracle_regret[{m}]"] - quantities["oracle_regret[fixed_pristine]"]
    all_q = {**quantities, **diffs}

    base_ids = sorted(set(ref.index.get_level_values("base_image")))
    pos = {b: i for i, b in enumerate(base_ids)}
    W = rng.multinomial(len(base_ids), np.full(len(base_ids), 1.0 / len(base_ids)), size=n_boot).astype(np.float64)
    out = {}
    for name, s in all_q.items():
        s = s.dropna()
        idx = np.asarray([pos[b] for b in s.index.get_level_values("base_image")])
        sums = np.bincount(idx, weights=s.values, minlength=len(base_ids))
        cnts = np.bincount(idx, minlength=len(base_ids)).astype(np.float64)
        boot = (W @ sums) / (W @ cnts)
        out[name] = {"estimate": float(s.mean()), "ci95_low": float(np.percentile(boot, 2.5)),
                     "ci95_high": float(np.percentile(boot, 97.5)), "n_rows": int(len(s))}
    return {"method": "paired percentile bootstrap over test base images (all regimes of a resampled base image "
                      "are kept together); per-row values are first averaged over the seeds of a model",
            "n_resamples": n_boot, "seed": seed, "n_base_images": len(base_ids), "results": out}


# ------------------------------------------------------------------
# exports
# ------------------------------------------------------------------
def export_curves(runs: Dict[str, Run], groups, out: Path):
    sel_dir = out / "curve_exports" / "selected_regimes"
    sel_dir.mkdir(parents=True, exist_ok=True)
    for model, names in groups.items():
        for n in names:
            r = runs.get(n)
            if r is None or not r.ok:
                continue
            z = r.npz
            seed = seed_of(n)
            per = {e["cluster_id"]: e for e in r.thr["per_regime"]}
            d = out / "curve_exports" / model / (f"seed{seed}" if seed is not None else "deterministic")
            d.mkdir(parents=True, exist_ok=True)
            for i, cid in enumerate(z["cluster_ids"]):
                name = str(z["cluster_names"][i])
                e = per[int(cid)]
                arrays = {"tau": z["tau"]}
                for k in ("tpr", "fpr", "fppi", "precision", "f1"):
                    arrays[f"{k}_gt"] = z[f"gt_{k}"][i]
                    arrays[f"{k}_pred"] = z[f"pred_mean_{k}"][i]
                    arrays[f"{k}_pred_std"] = z[f"pred_std_{k}"][i]
                arrays["recall_gt"], arrays["recall_pred"] = arrays["tpr_gt"], arrays["tpr_pred"]
                meta = dict(regime_id=f"{int(cid)}_{name}", degradation=name.replace("_3_5", ""),
                            severity="0" if name == "pristine" else "3-5", split="test", num_images=int(e["n"]),
                            tau_pristine=float(e["tau_pristine"]), tau_adaptive=float(e["tau_adaptive"]["mean"]),
                            tau_adaptive_regime_aggregate=float(e["tau_adaptive_regime"]),
                            tau_oracle=float(e["tau_oracle"]), seed=-1 if seed is None else seed, model_name=model,
                            pred_note="pred = mean over the regime's test images of the per-image predicted tau-functions")
                np.savez_compressed(d / f"{int(cid)}_{name}.npz", **arrays, **{k: np.asarray(v) for k, v in meta.items()})
                if name in SELECTED_REGIMES and seed in (0, None):
                    df = pd.DataFrame({k: v for k, v in arrays.items()})
                    with open(sel_dir / f"{model}_{name}.csv", "w") as f:
                        f.write("# " + json.dumps(meta) + "\n")
                        df.to_csv(f, index=False)


def write_table(df: pd.DataFrame, stem: Path, caption: str, fmt="{:.4f}"):
    df.to_csv(f"{stem}.csv", index=False)

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "–" if (v is None or np.isnan(v)) else fmt.format(v)
        return str(v)

    cols = list(df.columns)
    with open(f"{stem}.md", "w") as f:
        f.write(f"<!-- {caption} -->\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(cell(r[c]) for c in cols) + " |\n")
    with open(f"{stem}.tex", "w") as f:
        f.write("% " + caption + "\n\\begin{tabular}{" + "l" * 1 + "r" * (len(cols) - 1) + "}\n\\toprule\n")
        f.write(" & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(" & ".join(cell(r[c]).replace("_", "\\_").replace("τ", "$\\tau$") for c in cols) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


# ------------------------------------------------------------------
# figures
# ------------------------------------------------------------------
def fig_f1_vs_tau(runs, out: Path, model_runs: Dict[str, str], regimes=SELECTED_REGIMES):
    main = runs[model_runs["adaptive_pchip"]]
    z = main.npz
    names = [str(n) for n in z["cluster_names"]]
    regimes = [r for r in regimes if r in names]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for ax, reg in zip(axes.ravel(), regimes):
        i = names.index(reg)
        tau = z["tau"]
        ax.plot(tau, z["gt_f1"][i], color=PRIMARY_COLOR, lw=3.0, label="empirical F1(τ)")
        m, s = z["pred_mean_f1"][i], z["pred_std_f1"][i]
        ax.fill_between(tau, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=PRED_COLOR, alpha=0.18, lw=0)
        ax.plot(tau, m, color=PRED_COLOR, lw=2.0, label="predicted F1(τ), PCHIP (mean ± sd over images)")
        if "adaptive_linear" in model_runs:
            zl = runs[model_runs["adaptive_linear"]].npz
            ax.plot(tau, zl["pred_mean_f1"][i], color=COLORS[3], lw=1.3, ls="--", label="predicted F1(τ), linear")
        per = {e["cluster_name"]: e for e in main.thr["per_regime"]}[reg]
        ax.axvline(per["tau_pristine"], color=COLORS[0], ls="--", lw=1.8, label="fixed pristine τ")
        ax.axvline(per["tau_adaptive"]["mean"], color=PRED_COLOR, ls="-", lw=1.8, label="adaptive τ (mean over images)")
        ax.axvline(per["tau_oracle"], color=PRIMARY_COLOR, ls=":", lw=2.2, label="oracle τ")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(short(reg), fontsize=FONT_SIZE)
        ax.set_xlabel("detector confidence threshold τ")
        ax.set_ylabel("F1")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower center")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig3_f1_vs_tau.{ext}", dpi=200)
    plt.close(fig)


def fig_gain_by_regime(pr: pd.DataFrame, out: Path, policies=("mean_behavior", "adaptive_linear", "adaptive_pchip"),
                       fname="fig4_gain_by_regime"):
    g = pr[pr["policy"].isin(policies)]
    stats = g.groupby(["policy", "regime"])["gain_vs_pristine"].agg(["mean", "std"]).reset_index()
    order = (stats[stats["policy"] == policies[-1]].sort_values("mean")["regime"].tolist())
    x = np.arange(len(order))
    w = 0.8 / len(policies)
    fig, ax = plt.subplots(figsize=(13, 4.4), constrained_layout=True)
    for k, pol in enumerate(policies):
        s = stats[stats["policy"] == pol].set_index("regime").reindex(order)
        ax.bar(x + (k - (len(policies) - 1) / 2) * w, s["mean"], width=w * 0.9, color=POLICY_COLOR[pol],
               yerr=s["std"].fillna(0), capsize=2, error_kw=dict(lw=0.8), label=POLICY_LABEL[pol])
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short(r) for r in order], rotation=45, ha="right")
    ax.set_ylabel("F1(τ_selected) − F1(τ_pristine)")
    ax.set_title("Operational gain over the fixed pristine threshold per test regime "
                 "(learned models: mean ± sd over 3 seeds)", fontsize=FONT_SIZE)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{fname}.{ext}", dpi=200)
    plt.close(fig)


def fig_regret_and_scatter(pr: pd.DataFrame, runs, model_runs, out: Path):
    pols = ["fixed_pristine", "mean_behavior", "adaptive_linear", "adaptive_pchip"]
    stats = pr[pr["policy"].isin(pols)].groupby(["policy", "regime"])["oracle_regret"].mean().unstack(0)
    stats = stats.sort_values("fixed_pristine")
    y = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(stats) + 1.5), constrained_layout=True)
    for pol, mk in zip(pols, ("s", "D", "^", "o")):
        ax.plot(stats[pol], y, mk, color=POLICY_COLOR[pol], ms=7, label=POLICY_LABEL[pol])
    ax.set_yticks(y)
    ax.set_yticklabels([short(r) for r in stats.index])
    ax.set_xlabel("oracle regret  F1(τ_oracle) − F1(τ_selected)")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"regret_by_regime.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    for model, mk in (("adaptive_pchip", "o"), ("adaptive_linear", "^")):
        g = pr[pr["policy"] == model].groupby("regime")[["tau_oracle", "tau_mean"]].mean()
        ax.plot(g["tau_oracle"], g["tau_mean"], mk, color=POLICY_COLOR[model], ms=7, label=POLICY_LABEL[model])
    mb = pr[pr["policy"] == "mean_behavior"].groupby("regime")[["tau_oracle", "tau_mean"]].mean()
    ax.plot(mb["tau_oracle"], mb["tau_mean"], "D", color=POLICY_COLOR["mean_behavior"], ms=6, label="Mean behavior")
    tp = pr[pr["policy"] == "fixed_pristine"]["tau_mean"].iloc[0]
    ax.axhline(tp, color=COLORS[0], ls="--", lw=1.5, label="fixed pristine τ")
    hi = max(0.35, float(pr["tau_oracle"].max()) + 0.05)
    ax.plot([0, hi], [0, hi], color="0.4", lw=1, label="selected = oracle")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("oracle τ (test regime)")
    ax.set_ylabel("selected τ (mean over images and seeds)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"adaptive_vs_oracle_tau.{ext}", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default='results/main_study/results_legacy/results/paper/eval')
    ap.add_argument("--out", default='results/main_study/paper_artifacts')
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_seed", type=int, default=12345)
    args = ap.parse_args()
    root, out = Path(args.eval_root), Path(args.out)
    for d in ("aggregate", "tables", "figures/main", "figures/appendix"):
        (out / d).mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    runs = {n: Run(root, n) for n in names}
    status = {n: r.ok for n, r in runs.items()}
    groups = {m: [n for n in ns if runs.get(n) is not None and runs[n].ok] for m, ns in MAIN_MODELS.items()}
    app_groups = {m: [n for n in ns if runs.get(n) is not None and runs[n].ok] for m, ns in APPENDIX_MODELS.items()}

    # ---- curve fit
    cf_seed, cf_reg, cf_sum = curve_fit_tables(runs, {**{"mean_behavior": [BASELINE_RUN]}, **groups, **app_groups})
    cf_seed.to_csv(out / "aggregate/curve_fit_per_seed.csv", index=False)
    cf_reg.to_csv(out / "aggregate/curve_fit_per_regime.csv", index=False)
    cf_sum.to_csv(out / "aggregate/curve_fit_summary.csv", index=False)

    # ---- threshold policies
    pr, ps, psum = threshold_tables(runs, {**groups, **app_groups})
    pr.to_csv(out / "aggregate/threshold_policy_per_regime.csv", index=False)
    ps.to_csv(out / "aggregate/threshold_policy_per_seed.csv", index=False)
    psum.to_csv(out / "aggregate/threshold_policy_summary.csv", index=False)

    # ---- bootstrap
    boot = bootstrap(runs, groups, n_boot=args.n_boot, seed=args.boot_seed)
    (out / "aggregate/bootstrap_confidence_intervals.json").write_text(json.dumps(jsonable(boot), indent=2))

    # ---- tables
    ta = []
    for _, e in cf_sum.iterrows():
        if e["model"] not in ("mean_behavior", "adaptive_linear", "adaptive_pchip"):
            continue
        ta.append({"model": POLICY_LABEL[e["model"]], "n_seeds": int(e["n_seeds"]),
                   "roc_mad_mean": e["macro_mad_roc_mean"], "roc_mad_std": e["macro_mad_roc_std"],
                   "froc_mad_mean": e["macro_mad_froc_mean"], "froc_mad_std": e["macro_mad_froc_std"],
                   "pr_mad_mean": e["macro_mad_pr_mean"], "pr_mad_std": e["macro_mad_pr_std"],
                   "auroc_abs_error": e["macro_abs_delta_auroc_mean"], "afroc_abs_error": e["macro_abs_delta_afroc_mean"],
                   "auprc_abs_error": e["macro_abs_delta_auprc_mean"]})
    write_table(pd.DataFrame(ta), out / "tables/table_curve_fit",
                "Table A: curve fit on the test split (macro average over 18 regimes; mean/std over seeds; "
                "FROC domain [0, 2] FPPI for all models)")

    tb = []
    for _, e in psum.iterrows():
        if e["policy"] not in POLICY_ORDER:
            continue
        tb.append({"policy": POLICY_LABEL[e["policy"]], "n_seeds": int(e["n_seeds"]),
                   "f1_mean": e["f1_macro_mean"], "f1_std": e["f1_macro_std"],
                   "precision_mean": e["precision_macro_mean"], "recall_mean": e["recall_macro_mean"],
                   "fppi_mean": e["fppi_macro_mean"], "gain_vs_pristine": e["gain_vs_pristine_macro_mean"],
                   "oracle_regret": e["oracle_regret_macro_mean"], "tau_abs_error": e["tau_abs_error_macro_mean"]})
    write_table(pd.DataFrame(tb), out / "tables/table_operational",
                "Table B: operational threshold performance on the test split (macro over 18 regimes; "
                "std = over seeds; identical stored detections for all policies)")

    tc = []
    for run, interp, K, src, aux in ABLATIONS:
        if run not in runs or not runs[run].ok:
            continue
        cs = cf_seed[cf_seed["run"] == run]
        if cs.empty:
            cs = curve_fit_tables(runs, {"abl": [run]})[0]
        thr = policy_rows(runs[run], "adaptive", "ablation", seed_of(run))
        tc.append({"run": run, "interpolation": interp, "num_knots": K, "knot_source": src, "aux_tau_loss": aux,
                   "roc_mad": float(cs["macro_mad_roc"].iloc[0]), "froc_mad": float(cs["macro_mad_froc"].iloc[0]),
                   "pr_mad": float(cs["macro_mad_pr"].iloc[0]),
                   "gain_vs_pristine": float(np.mean([e["gain_vs_pristine"] for e in thr])),
                   "oracle_regret": float(np.mean([e["oracle_regret"] for e in thr])),
                   "tau_abs_error": float(np.mean([e["tau_abs_error"] for e in thr]))})
    tc_df = pd.DataFrame(tc)
    tc_df.to_csv(out / "aggregate/ablation_results.csv", index=False)
    if not tc_df.empty:
        write_table(tc_df.drop(columns=["run"]), out / "tables/table_ablations",
                    "Table C: focused single-seed ablations (seed 0, test split, macro over regimes)")

    # ---- generalization (leave one degradation type out)
    gen = []
    for reg, short_name in LODO.items():
        for run, kind in ((f"lodo_{short_name}_pchip_s0", "held_out_pchip"),
                          (f"lodo_{short_name}_mean_behavior", "held_out_mean_behavior")):
            if run not in runs or not runs[run].ok:
                continue
            r = runs[run]
            pc = {e["cluster_name"]: e for e in r.summary["per_cluster"]}[reg]
            th = [e for e in policy_rows(r, "adaptive", kind, 0) if e["regime"] == reg][0]
            oth = [e for e in policy_rows(r, "adaptive", kind, 0) if e["regime"] != reg]
            gen.append({"held_out_regime": reg, "model": kind, "run": run, "mad_roc": pc["mad_roc"],
                        "mad_froc": pc["mad_froc"], "mad_pr": pc["mad_pr"], "gain_vs_pristine": th["gain_vs_pristine"],
                        "oracle_regret": th["oracle_regret"], "tau_abs_error": th["tau_abs_error"],
                        "tau_mean": th["tau_mean"], "tau_oracle": th["tau_oracle"],
                        "regret_other_regimes_mean": float(np.mean([e["oracle_regret"] for e in oth]))})
        # in-distribution reference: main PCHIP seeds on the same regime
        for n in groups.get("adaptive_pchip", []):
            pc = {e["cluster_name"]: e for e in runs[n].summary["per_cluster"]}[reg]
            th = [e for e in policy_rows(runs[n], "adaptive", "in_dist", seed_of(n)) if e["regime"] == reg][0]
            gen.append({"held_out_regime": reg, "model": "in_distribution_pchip", "run": n, "mad_roc": pc["mad_roc"],
                        "mad_froc": pc["mad_froc"], "mad_pr": pc["mad_pr"], "gain_vs_pristine": th["gain_vs_pristine"],
                        "oracle_regret": th["oracle_regret"], "tau_abs_error": th["tau_abs_error"],
                        "tau_mean": th["tau_mean"], "tau_oracle": th["tau_oracle"], "regret_other_regimes_mean": None})
    pd.DataFrame(gen).to_csv(out / "aggregate/generalization_results.csv", index=False)

    # ---- curve exports
    export_curves(runs, {**groups, "mean_behavior": [BASELINE_RUN], **app_groups}, out)

    # ---- figures
    seed0 = {m: ns[0] for m, ns in groups.items() if ns}
    if "adaptive_pchip" in seed0:
        fig_f1_vs_tau(runs, out / "figures/main", seed0)
    fig_gain_by_regime(pr, out / "figures/main")
    if app_groups.get("pc1_behavior"):
        fig_gain_by_regime(pr, out / "figures/appendix", ("mean_behavior", "pc1_behavior", "adaptive_pchip"),
                           "gain_by_regime_with_pc1")
    fig_regret_and_scatter(pr, runs, seed0, out / "figures/appendix")

    (out / "aggregate/run_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps({"runs_ok": [n for n, ok in status.items() if ok],
                      "runs_missing_outputs": [n for n, ok in status.items() if not ok]}, indent=1))


if __name__ == "__main__":
    main()

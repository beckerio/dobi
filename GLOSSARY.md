# Paper terms and names in the code

| Paper | Code / configs |
|---|---|
| DOBI, detector operating behavior inference | the whole package `dobi` |
| fixed (frozen) detector | YOLOv10-N, `weights/yolov10n.pt`; used unchanged through Ultralytics |
| data-space sampling / probing | degraded image sets from `dobi.degradation.generator` (`apply_dataset_degradation`), indexed as regime subsets in the database |
| regime (proxy regime *w*) | one subset of the database, e.g. `fog_3_5` = fog, severities 3–5; `pristine`; 18 per split, defined in `configs/regime_subsets.json` (`generate_subsets.py`) |
| operating curve, threshold response, behavior *B(P_w; τ)* | per-regime threshold sweep: `generate_curves.compute_operating_sweep_from_global_labels`, table `subset_operating_points`; read with `tau_targets.load_raw_sweep`, `dedupe_sweep`, `sweep_at_tau` |
| behavior target | `tau_targets.build_tau_behavior_targets` (TPR, FPR, FPPI, precision at the knots) |
| stored detections | `predictions` table of the database, written by `detect_variants.py` → `generate_db.py` |
| multi-scale detector features | three YOLOv10-N feature levels (64, 128, 256 channels), pooled as [GAP, GMP] per level: `feature_cache.pool_feats`; cache: `dobi.data.feature_cache` |
| monitor (monitor head, monitor readout) | `dobi.models` heads built by `monitor_factory.build_monitor_head(train_args["head"], ...)`; released files `tau_*_s*.pt`, `pc1_behavior_s*.pt` |
| predicted behavior *B̂(x; τ)* | output of `monitor_factory.decode_monitor_output`: `recall_tau`, `tpr_tau`, `fpr_tau`, `fppi_tau`, `precision_tau`, `f1_tau` on the grid `tau_grid` |
| functional decoder (PCHIP, K = 16 knots) | head `tau_behavior` (`TauBehaviorHead`) with `interp: "pchip"`, `n_knots: 16`, `knot_placement: "uniform"`, `fppi_space: "log1p"`; runs `tau_pchip_s0..2`; interpolation in `utils/tau_spline.py` |
| linear decoder | the same head with `interp: "linear"`; runs `tau_linear_s0..2` |
| knot ablations | runs `abl_pchip_K8_uniform`, `abl_pchip_K32_uniform`, `abl_pchip_K16_quantile_*` (`configs/paper_ablations.json`) |
| PC1 behavior baseline | head `pc1_behavior` (`PC1BehaviorHead`); runs `pc1_behavior_s0..2` |
| legacy spline heads (appendix) | heads `tau_spline_single`, `tau_shared_anchor` (`configs/paper_appendix.json`) |
| mean-behavior prior (image-independent) | `mean_behavior` (run without training; `MeanBehaviorPredictor` in `evaluation_curve_monitor.py`) |
| held-out degradation type (leave one type out) | runs `lodo_gaussian_noise_pchip_s0`, `lodo_glass_blur_pchip_s0`, `lodo_fog_pchip_s0` and their `lodo_*_mean_behavior` controls |
| mean absolute deviation (MAD) of ROC / FROC / PR curves | `mad_roc`, `mad_froc`, `mad_pr` (macro over regimes: `macro_mad_*`) in `evaluation_curve_monitor.py`, `aggregate_paper_results.py` |
| operating point, operating policy | threshold chosen from the predicted curves: `tau_spline.select_operating_threshold(tau, curves, criterion)` |
| best-F1 operating point | criterion `"f1"` |
| FPPI budget (FPPI-constrained operation) | criterion `"fppi"` with `max_fppi` (budget 0.1 in the main text); evaluation in `eval/extended/exp_c_fppi.py` |
| validation-calibrated FPPI operation | policies `*_valcal` in `exp_c_fppi.py` (safety factor `kappa` calibrated on validation, `calibrate_kappa`) |
| operating-point execution (thresholds applied per image) | `dobi.eval.run_execution_eval`, `dobi.eval.execution_eval`; results `results/main_study/paper_artifacts_execution/` |
| pristine calibration | policy `fixed_pristine` (`tau_targets.pristine_validation_threshold`) |
| adaptive (DOBI) operating points | policies `adaptive_pchip`, `adaptive_linear`, `pc1_behavior` |
| lookup that knows the true regime (non-deployable) | policy `regime_id_validation` (validation threshold of the true regime) |
| regime oracle | policy `regime_oracle` (best test threshold per regime); per image: `per_image_oracle` (exploratory) |
| macro-regime F1 | `overall_macro` in `execution_summary.json` (mean over the 18 regimes) |
| functional distance vs. empirical regime distance | `eval/extended/exp_b_equivalence.py` (`behavior_from_sweep`, `distance`); main Fig. 3: `scripts/figures/fig3_regime_distance_agreement/` |
| single image vs. batch of images | `eval/extended/exp_a_batch.py` |
| consistency of the predicted components | `eval/extended/exp_d_consistency.py` |
| regime-supervised curve lookups: hard lookup, soft lookup; template lookup (follow-up and continuous-probing studies) | arms `hard_lookup`, `soft_lookup`, `soft_lookup_formula` in `results/rq4_followups/` and `results/rq5_continuous/` and their table generators in `scripts/tables/`; the pipelines of these studies are not part of this version |
| continuous and compositional probing | `results/rq5_continuous/`, `scripts/tables/make_continuous_tables.py` |
| natural conditions (BDD100K) | `results/natural_bdd/`, `scripts/tables/make_natural_tables.py` |

# UNCERTAFACE results (real skulls, SYNTHETIC faces)

> Every number below is read from saved JSON produced by the code in this repository. Skulls are real (MUG500+, CC BY 4.0); the face attached to each is SYNTHETIC (an assumed soft-tissue rule). Errors are measured against that synthetic face, so they say nothing about real faces, forensic validity or identification. All distances/uncertainties are in millimetres.

## Dataset
- version 0.2.0, config sha256 `378bc9ab1e2d`
- independent base geometries: **60**; observations: **480**
- bases per split: {'calibration': 6, 'test': 6, 'train': 42, 'validation': 6}; observations per split: {'calibration': 48, 'test': 48, 'train': 336, 'validation': 48}
- all integrity checks passed: **True**
- measured missingness (mean area fraction): extreme 0.77, mild 0.15, moderate 0.38, severe 0.60

## Irreducible-error floor (generator property)
- complete noise-free skull, face latents redrawn: floor vertex RMSE = **1.34 mm** (oracle positional std: mean 1.25, range 0.33-3.60 mm)

## Reference baseline (no skull evidence)
- mean training face for every sample, test split (mm): vertex_mae 3.17, vertex_rmse 8.11, chamfer 5.11, hausdorff 21.20, point_to_surface 4.61, surface_error 4.71

## Main model `face_v3_pn` on the test split (n = 48 observations)
### Reconstruction (mm)
| metric | mean | std |
|---|---|---|
| vertex_mae | 2.34 | 0.52 |
| vertex_rmse | 5.86 | 1.28 |
| chamfer | 3.26 | 0.73 |
| hausdorff | 16.86 | 4.68 |
| point_to_surface | 2.62 | 0.85 |
| surface_error | 2.61 | 0.83 |

Vertex metrics use known correspondence; Chamfer/Hausdorff treat vertices as point sets; point-to-surface/surface error use distances to the other mesh's triangles.

### By missingness severity (vertex RMSE, mm)
| severity | rmse | chamfer | surface_error |
|---|---|---|---|
| mild | 5.21 | 2.97 | 2.28 |
| moderate | 5.53 | 3.18 | 2.50 |
| severe | 6.02 | 3.21 | 2.57 |
| extreme | 6.70 | 3.68 | 3.12 |

### Does uncertainty respond to missingness?
- spearman_missing_vs_error = 0.455; spearman_missing_vs_pred_std = 0.547; within_sample_spearman_err_vs_std_mean = 0.525; pooled_spearman_err_vs_std = 0.534

### Prediction intervals: raw vs conformal-calibrated (test targets never used for fitting)
- raw CRPS = 1.725 mm, NLL = 2.495, standardised-residual mean/std = 0.006/0.798, raw calibration error (mean |gap|) = 0.082

| nominal | raw_pointwise | raw_width_mm | global_pointwise | per_coordinate_pointwise | per_group_pointwise | global_width_mm | raw_simultaneous | subject_max_simultaneous |
|---|---|---|---|---|---|---|---|---|
| 50% | 0.663 | 2.524 | 0.509 | 0.503 | 0.511 | 1.780 | 0.000 | 0.938 |
| 80% | 0.891 | 4.796 | 0.813 | 0.821 | 0.809 | 3.737 | 0.000 | 1.000 |
| 90% | 0.948 | 6.156 | 0.910 | 0.919 | 0.904 | 5.175 | 0.000 | 1.000 |
| 95% | 0.975 | 7.335 | 0.959 | 0.962 | 0.956 | 6.535 | 0.146 | 1.000 |

Pointwise = fraction of coordinates inside their interval (marginal). Simultaneous = fraction of test samples whose EVERY vertex/axis is inside; only `subject_max` calibration targets it, and with few calibration bases the highest levels become conservative or infinite (calibration observations: 48).

### Sparsification and risk-coverage
- sparsification_error_mm = 0.860; normalised_se = 0.374; aurc_mm = 3.515; aurc_oracle_mm = 2.655; aurc_random_mm = 4.953
- risk (mean error, mm) at retained coverage: 1: 4.95, 0.9: 4.53, 0.75: 4.07, 0.5: 3.49, 0.25: 2.82, 0.1: 2.61
- best-of-10 (oracle-selected, NOT a single-prediction result): single 6.00 mm, mean sampled 9.27 mm, best-of-N 9.09 mm

### Test-set breakdown by severity
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| extreme | 12 | 6.84 | 7.19 | 0.94 |
| mild | 12 | 5.34 | 6.49 | 0.95 |
| moderate | 12 | 5.58 | 6.46 | 0.95 |
| severe | 12 | 6.14 | 6.86 | 0.95 |

### Test-set breakdown by pattern
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| cap | 10 | 6.52 | 6.76 | 0.92 |
| lateral | 8 | 5.46 | 6.51 | 0.96 |
| patchy | 6 | 5.35 | 6.55 | 0.97 |
| region | 12 | 5.89 | 6.89 | 0.95 |
| upper | 12 | 6.31 | 6.86 | 0.95 |

### Test-set breakdown by n_points
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 1024 | 10 | 5.23 | 6.65 | 0.97 |
| 2048 | 17 | 6.19 | 6.85 | 0.96 |
| 4096 | 21 | 6.18 | 6.72 | 0.93 |

### Test-set breakdown by sampler
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| area | 19 | 6.11 | 6.75 | 0.94 |
| fps | 29 | 5.93 | 6.75 | 0.95 |

### Test-set breakdown by noise_mm
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 0-0.5 | 17 | 6.29 | 6.69 | 0.94 |
| 0.5-1 | 16 | 5.52 | 6.64 | 0.97 |
| 1-1.51 | 15 | 6.16 | 6.93 | 0.94 |

### Controlled sweep: noise
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 0.0 | 6 | 4.86 | 6.49 | 0.96 |
| 0.5 | 6 | 5.10 | 6.51 | 0.94 |
| 1.0 | 6 | 4.98 | 6.42 | 0.95 |
| 2.0 | 6 | 5.55 | 6.54 | 0.94 |
| 3.0 | 6 | 5.86 | 6.61 | 0.93 |

### Controlled sweep: density
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 256 | 6 | 6.02 | 6.65 | 0.92 |
| 512 | 6 | 5.72 | 6.71 | 0.95 |
| 1024 | 6 | 5.71 | 6.56 | 0.94 |
| 2048 | 6 | 5.68 | 6.50 | 0.94 |
| 4096 | 6 | 5.62 | 6.41 | 0.94 |

### Controlled sweep: sampler
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| area | 6 | 5.39 | 6.75 | 0.96 |
| fps | 6 | 5.19 | 6.63 | 0.95 |

### Controlled sweep: deformation
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 0.0 | 6 | 5.45 | 6.43 | 0.94 |
| 0.03 | 6 | 6.21 | 6.48 | 0.94 |
| 0.06 | 6 | 8.99 | 6.61 | 0.80 |
| 0.1 | 6 | 12.73 | 6.76 | 0.60 |

### Controlled sweep: missingness
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| mild-patchy | 6 | 5.20 | 6.42 | 0.96 |
| mild-cap | 6 | 5.67 | 6.33 | 0.94 |
| mild-upper | 6 | 5.24 | 6.51 | 0.96 |
| mild-lateral | 6 | 5.66 | 6.31 | 0.92 |
| mild-region | 6 | 5.83 | 6.44 | 0.90 |
| moderate-patchy | 6 | 5.20 | 6.57 | 0.96 |
| moderate-cap | 6 | 5.97 | 6.54 | 0.94 |
| moderate-upper | 6 | 5.51 | 6.66 | 0.97 |
| moderate-lateral | 6 | 5.66 | 6.22 | 0.94 |
| moderate-region | 6 | 5.55 | 6.54 | 0.94 |
| severe-patchy | 6 | 5.25 | 6.61 | 0.96 |
| severe-cap | 6 | 5.92 | 6.74 | 0.95 |
| severe-upper | 6 | 5.88 | 7.15 | 0.97 |
| severe-lateral | 6 | 5.51 | 6.64 | 0.96 |
| severe-region | 6 | 6.23 | 6.99 | 0.96 |
| extreme-patchy | 6 | 5.33 | 6.82 | 0.96 |
| extreme-cap | 6 | 6.60 | 7.42 | 0.96 |
| extreme-upper | 6 | 6.43 | 7.23 | 0.95 |
| extreme-lateral | 6 | 7.05 | 7.51 | 0.94 |
| extreme-region | 6 | 6.66 | 7.51 | 0.94 |

### Which face regions are most certain (test split)
| region | mean_halfwidth_mm | mean_error_mm |
|---|---|---|
| forehead | 4.03 | 3.34 |
| eyes | 5.00 | 4.21 |
| nose | 5.45 | 4.74 |
| cheeks | 5.84 | 4.86 |
| mouth | 6.74 | 6.60 |
| chin and jaw | 6.76 | 7.25 |

- rank correlation of region certainty vs real error: 1.00 over 6 regions; per-vertex Spearman 0.53
- error in the 25% most / least certain points: 2.82 / 7.60 mm (all points 4.95 mm)
- shown per-coordinate half-widths contain 90.7% of true coordinates (nominal 90%)
- plausible faces (K=4, smoothing 12.0 mm, an assumption): mean sample RMSE 8.37 mm vs mean prediction 5.86 mm; best-of-K (oracle-selected) 7.47 mm

Figures: `reports/results/face_v3_pn/*.png`

## Ablations
| variant | status | params | epochs_run | best_epoch | val_rmse_mm | test_vertex_rmse_mm | test_chamfer_mm | test_surface_error_mm | test_hausdorff_mm | rmse_mild | rmse_moderate | rmse_severe | rmse_extreme | crps_mm | nll | ece_raw | cov90_raw | cov90_global | half_width90_raw | half_width90_global | sim95_subject_max | sparsification_norm_se | aurc_mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | completed | 787190 | 60 | 59 | 8.783 | 6.791 | 4.064 | 3.560 | 19.543 | 6.929 | 6.793 | 6.785 | 6.659 | 2.040 | 2.676 | 0.102 | 0.964 | 0.895 | 7.825 | 5.610 | 1.000 | 0.501 | 4.513 |
| full_seed1 | completed | 787190 | 60 | 59 | 10.439 | 7.925 | 4.971 | 4.546 | 20.808 | 8.144 | 7.889 | 7.846 | 7.822 | 2.296 | 2.755 | 0.109 | 0.975 | 0.904 | 8.867 | 6.259 | 1.000 | 0.497 | 5.372 |
| full_seed2 | completed | 787190 | 60 | 58 | 8.656 | 6.534 | 3.987 | 3.496 | 19.124 | 6.583 | 6.517 | 6.547 | 6.491 | 1.970 | 2.656 | 0.112 | 0.970 | 0.899 | 7.819 | 5.399 | 1.000 | 0.530 | 4.445 |
| deterministic | completed | 786803 | 60 | 59 | 9.372 | 6.979 | 4.258 | 3.795 | 19.470 | 7.324 | 7.007 | 6.973 | 6.613 |  |  |  |  |  |  |  |  |  |  |
| no_surface | completed | 787190 | 60 | 59 | 8.854 | 6.854 | 4.056 | 3.554 | 20.095 | 6.948 | 6.730 | 7.136 | 6.600 | 2.041 | 2.672 | 0.103 | 0.963 | 0.895 | 7.829 | 5.607 | 1.000 | 0.500 | 4.511 |
| no_missing_info | completed | 787190 | 60 | 59 | 9.072 | 7.068 | 4.325 | 3.853 | 19.697 | 7.623 | 6.921 | 6.965 | 6.763 | 2.111 | 2.700 | 0.104 | 0.966 | 0.900 | 8.117 | 5.909 | 1.000 | 0.532 | 4.843 |
| encoder_pointnet | completed | 619062 | 60 | 51 | 8.084 | 5.910 | 3.411 | 2.788 | 17.294 | 5.221 | 5.656 | 6.086 | 6.676 | 1.763 | 2.537 | 0.111 | 0.972 | 0.919 | 6.983 | 5.145 | 1.000 | 0.388 | 3.535 |
| points_2048 | completed | 787190 | 60 | 58 | 9.093 | 6.962 | 4.300 | 3.831 | 20.038 | 7.180 | 6.789 | 6.943 | 6.936 | 2.079 | 2.685 | 0.111 | 0.974 | 0.897 | 8.144 | 5.689 | 1.000 | 0.516 | 4.748 |
| points_4096 | completed | 787190 | 60 | 59 | 9.010 | 6.755 | 4.141 | 3.657 | 19.605 | 7.158 | 6.655 | 6.890 | 6.317 | 2.033 | 2.672 | 0.112 | 0.976 | 0.891 | 8.048 | 5.494 | 1.000 | 0.527 | 4.644 |

## Deep ensemble `ensemble3` (3 members)
- single members' RMSE (mm): {'abl_full': 6.961998462677002, 'abl_full_seed1': 8.055062294006348, 'abl_full_seed2': 6.668401718139648}; ensemble mean-prediction RMSE: 6.80 mm

| kind | mean_std_mm | cov90_raw | crps_mm | pooled_spearman_err_vs_std | normalised_sparsification_error | aurc_mm |
|---|---|---|---|---|---|---|
| aleatoric | 4.987 | 0.977 | 2.041 | 0.480 | 0.495 | 4.578 |
| epistemic | 0.766 | 0.229 | 2.386 | 0.266 | 0.656 | 5.004 |
| combined | 5.093 | 0.980 | 2.047 | 0.477 | 0.505 | 4.604 |
- epistemic share of total variance: 0.048


## Findings (written by hand from the tables above; every number is in a results file)

**Scope.** 60 real MUG500+ skulls, synthetic faces from an assumed soft-tissue rule, split by skull 42/6/6/6 (train/validation/calibration/test).
The test set is 6 skulls (48 partial views), so every number is rough. Errors are against the synthetic face, not a real one.

**Headline (`face_v3_pn`, PointNet encoder).** Test vertex RMSE 5.86 mm against 8.11 mm for the no-skull mean-face baseline and a 1.34 mm
generator floor. The PointNet++ model (`face_v3`) scored 6.08 mm. Seed-to-seed spread of the identical model is about 1.4 mm (ablation
table), so that 0.2 mm gap is not established. The PointNet choice followed validation error (7.16 vs 7.69 mm) and a 60-epoch ablation.

**History, including what got worse.** Results across versions are NOT directly comparable (different face definition, skull count, test skulls).

| run | what changed | test skulls | model RMSE (mm) | mean-face baseline (mm) | raw 90% cov | calibrated 90% cov | pooled Spearman err vs std |
|---|---|---|---|---|---|---|---|
| `real_main` | first real-skull run, face v1, 41 skulls | 4 (32 views) | 5.24 | 8.42 | 0.970 | 0.918 | 0.33 |
| `face_v2` | facial relief, 41 skulls | 4 (32 views) | 5.29 | 8.10 | 0.980 | 0.919 | 0.28 |
| `face_v3` | wider face, 60 skulls, PointNet++ | 6 (48 views) | 6.08 | 8.11 | 0.964 | 0.910 | 0.55 |
| `face_v3_pn` | PointNet encoder | 6 (48 views) | 5.86 | 8.11 | 0.948 | 0.910 | 0.53 |

Absolute error rose from about 5.3 mm to 5.9-6.1 mm and the gain over the mean-face baseline fell from about 37% to about 28%. The wider face
covers harder regions and the test skulls differ, so this is not proof the model got worse, but it is NOT evidence that v3 is "more precise".
What did improve is how well predicted uncertainty tracks error (Spearman 0.28-0.33 to 0.53-0.55).

**Uncertainty (supported).** Raw 90% intervals cover 94.8% of points (too wide); split conformal calibration gives 91.0% (nominal 90%). At 50%
the calibrated coverage is 51%. Across the face, the mean error map and the mean predicted-uncertainty map correlate at 0.92. Removing the
most uncertain points lowers error (risk at 100%/50%/10% retained: 4.95/3.49/2.61 mm; AURC 3.52 mm against 4.95 random and 2.65 oracle).
Predicted uncertainty also rises with missing bone (extreme patterns 6.8-7.5 mm std against 6.3-6.6 for mild).

**Stress tests (supported, with a failure).**
- Density 256-4096 points and area vs farthest-point sampling: error moves by under 0.5 mm.
- Coordinate noise 0 to 3 mm: error 4.9 to 5.9 mm; coverage 96% to 93%.
- Missing patterns: error rises with severity, worst for lateral and region occlusion at extreme (about 6.7-7.1 mm).
- **Deformation is a failure.** Joint affine deformation of 6% and 10% raises error to 9.0 and 12.7 mm while predicted uncertainty stays about
  6.6-6.8 mm, and coverage collapses to 80% and 60%. The model's uncertainty does not warn about inputs unlike its training skulls.
  The training data contains only small deformations; this model should not be trusted on unusual skulls.

**Not supported / negative results.**
- Best-of-10 sampled predictions (9.09 mm) are worse than the single mean prediction (6.00 mm); sampling from the large predicted spread
  adds error. Best-of-N does not help here and is reported only as a separate, oracle-selected number.
- Three seeds of PointNet++ give 6.5 to 7.9 mm; an ensemble of them gives 6.80 mm. Epistemic uncertainty is about 5% of total variance, so the
  ensemble adds little over the single-model uncertainty.
- Removing NLL, surface regularisation, missing-amount input, or changing the number of points changed test error by less than the seed spread,
  so no effect is demonstrated for those components. Only the simpler encoder stood out.

**Domain gap.** The skulls are real; the faces are not. A model trained to reproduce an assumed soft-tissue rule can match that rule without
saying anything about real faces. Validation on real paired skull-face scans is future work and is not done here.

# UNCERTAFACE results (SYNTHETIC data only)

> Every number below is read from saved JSON produced by the code in this repository. Results are for a procedurally generated synthetic skull-face task; they say nothing about real human anatomy, forensic validity or identification. All distances/uncertainties are in millimetres.

## Dataset
- version 0.1.0, config sha256 `03db455a747e`
- independent base geometries: **300**; observations: **1200**
- bases per split: {'calibration': 30, 'test': 30, 'train': 210, 'validation': 30}; observations per split: {'calibration': 120, 'test': 120, 'train': 840, 'validation': 120}
- all integrity checks passed: **True**
- measured missingness (mean area fraction): extreme 0.77, mild 0.15, moderate 0.38, severe 0.60

## Irreducible-error floor (generator property)
- complete noise-free skull, face latents redrawn: floor vertex RMSE = **1.62 mm** (oracle positional std: mean 1.56, range 0.80-4.07 mm)

## Main model `probabilistic_scratch` on the test split (n = 120 observations)
### Reconstruction (mm)
| metric | mean | std |
|---|---|---|
| vertex_mae | 1.46 | 0.52 |
| vertex_rmse | 3.19 | 1.12 |
| chamfer | 2.88 | 0.93 |
| hausdorff | 5.46 | 1.55 |
| point_to_surface | 1.98 | 0.92 |
| surface_error | 2.00 | 0.93 |

Vertex metrics use known correspondence; Chamfer/Hausdorff treat vertices as point sets; point-to-surface/surface error use distances to the other mesh's triangles.

### By missingness severity (vertex RMSE, mm)
| severity | rmse | chamfer | surface_error |
|---|---|---|---|
| mild | 3.05 | 2.80 | 1.82 |
| moderate | 3.03 | 2.74 | 1.85 |
| severe | 3.20 | 2.87 | 2.04 |
| extreme | 3.50 | 3.12 | 2.30 |

### Does uncertainty respond to missingness?
- spearman_missing_vs_error = 0.090; spearman_missing_vs_pred_std = 0.585; within_sample_spearman_err_vs_std_mean = 0.382; pooled_spearman_err_vs_std = 0.282

### Prediction intervals: raw vs conformal-calibrated (test targets never used for fitting)
- raw CRPS = 1.120 mm, NLL = 2.162, standardised-residual mean/std = -0.060/0.698, raw calibration error (mean |gap|) = 0.108

| nominal | raw_pointwise | raw_width_mm | global_pointwise | per_coordinate_pointwise | per_group_pointwise | global_width_mm | raw_simultaneous | subject_max_simultaneous |
|---|---|---|---|---|---|---|---|---|
| 50% | 0.691 | 1.890 | 0.537 | 0.539 | 0.537 | 1.304 | 0.008 | 0.833 |
| 80% | 0.930 | 3.591 | 0.820 | 0.822 | 0.819 | 2.559 | 0.158 | 0.958 |
| 90% | 0.972 | 4.609 | 0.918 | 0.916 | 0.919 | 3.427 | 0.525 | 0.975 |
| 95% | 0.988 | 5.492 | 0.961 | 0.961 | 0.962 | 4.224 | 0.692 | 1.000 |

Pointwise = fraction of coordinates inside their interval (marginal). Simultaneous = fraction of test samples whose EVERY vertex/axis is inside; only `subject_max` calibration targets it, and with few calibration bases the highest levels become conservative or infinite (calibration observations: 120).

### Sparsification and risk-coverage
- sparsification_error_mm = 0.740; normalised_se = 0.627; aurc_mm = 2.584; aurc_oracle_mm = 1.845; aurc_random_mm = 3.024
- risk (mean error, mm) at retained coverage: 1: 3.02, 0.9: 3.01, 0.75: 2.83, 0.5: 2.60, 0.25: 2.33, 0.1: 2.13
- best-of-10 (oracle-selected, NOT a single-prediction result): single 3.38 mm, mean sampled 6.04 mm, best-of-N 5.87 mm

### Test-set breakdown by severity
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| extreme | 30 | 3.78 | 5.45 | 0.97 |
| mild | 30 | 3.14 | 4.65 | 0.98 |
| moderate | 30 | 3.16 | 4.76 | 0.97 |
| severe | 30 | 3.41 | 4.98 | 0.97 |

### Test-set breakdown by pattern
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| cap | 27 | 3.43 | 5.06 | 0.97 |
| lateral | 23 | 3.16 | 4.81 | 0.98 |
| patchy | 23 | 3.03 | 4.80 | 0.98 |
| region | 22 | 3.74 | 5.10 | 0.96 |
| upper | 25 | 3.50 | 5.03 | 0.97 |

### Test-set breakdown by n_points
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 1024 | 41 | 3.30 | 4.95 | 0.98 |
| 2048 | 43 | 3.33 | 4.95 | 0.97 |
| 4096 | 36 | 3.53 | 4.98 | 0.97 |

### Test-set breakdown by sampler
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| area | 62 | 3.17 | 5.00 | 0.98 |
| fps | 58 | 3.60 | 4.92 | 0.96 |

### Test-set breakdown by noise_mm
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 0-0.5 | 41 | 3.19 | 4.95 | 0.99 |
| 0.5-1 | 32 | 3.36 | 4.86 | 0.95 |
| 1-1.51 | 47 | 3.56 | 5.04 | 0.97 |

### Controlled sweep: noise
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 0.0 | 30 | 3.02 | 4.73 | 0.98 |
| 0.5 | 30 | 3.07 | 4.74 | 0.98 |
| 1.0 | 30 | 3.09 | 4.76 | 0.98 |
| 2.0 | 30 | 3.57 | 4.84 | 0.96 |
| 3.0 | 30 | 4.02 | 4.98 | 0.92 |

### Controlled sweep: density
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 256 | 30 | 3.17 | 4.79 | 0.98 |
| 512 | 30 | 3.17 | 4.77 | 0.98 |
| 1024 | 30 | 3.12 | 4.76 | 0.99 |
| 2048 | 30 | 3.23 | 4.75 | 0.98 |
| 4096 | 30 | 3.16 | 4.75 | 0.98 |

### Controlled sweep: sampler
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| area | 30 | 3.26 | 4.76 | 0.97 |
| fps | 30 | 3.23 | 4.74 | 0.97 |

### Controlled sweep: deformation
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| 0.0 | 30 | 3.12 | 4.89 | 0.99 |
| 0.03 | 30 | 3.92 | 4.89 | 0.96 |
| 0.06 | 30 | 5.63 | 4.88 | 0.86 |
| 0.1 | 30 | 8.24 | 4.90 | 0.70 |

### Controlled sweep: missingness
| group | n | rmse_mm | mean_pos_std_mm | cov90_raw |
|---|---|---|---|---|
| mild-patchy | 30 | 3.02 | 4.59 | 0.98 |
| mild-cap | 30 | 2.90 | 4.59 | 0.99 |
| mild-upper | 30 | 3.06 | 4.62 | 0.98 |
| mild-lateral | 30 | 2.96 | 4.55 | 0.98 |
| mild-region | 30 | 3.10 | 4.62 | 0.98 |
| moderate-patchy | 30 | 3.12 | 4.64 | 0.98 |
| moderate-cap | 30 | 3.08 | 4.73 | 0.98 |
| moderate-upper | 30 | 3.24 | 4.75 | 0.98 |
| moderate-lateral | 30 | 2.96 | 4.57 | 0.98 |
| moderate-region | 30 | 3.07 | 4.77 | 0.98 |
| severe-patchy | 30 | 3.15 | 4.81 | 0.98 |
| severe-cap | 30 | 3.18 | 5.07 | 0.99 |
| severe-upper | 30 | 3.26 | 5.15 | 0.98 |
| severe-lateral | 30 | 3.28 | 4.95 | 0.98 |
| severe-region | 30 | 3.29 | 4.98 | 0.99 |
| extreme-patchy | 30 | 3.22 | 4.96 | 0.98 |
| extreme-cap | 30 | 3.43 | 5.38 | 0.98 |
| extreme-upper | 30 | 3.91 | 5.74 | 0.97 |
| extreme-lateral | 30 | 3.39 | 5.70 | 1.00 |
| extreme-region | 30 | 3.74 | 5.24 | 0.97 |

Figures: `reports/results/probabilistic_scratch/*.png`

## Ablations
not run

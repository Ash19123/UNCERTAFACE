# UNCERTAFACE methodology

Uncertainty-aware 3D facial reconstruction from incomplete cranial geometry. **All experiments are on synthetic data.**
Claims are scoped to the synthetic task; see "Limitations" and "What can and cannot be claimed".

## 1. Task
Input: a partial, noisy, irregularly sampled skull point cloud `X` (1024-4096 points) with a measured missing area fraction `rho`.
Output: for each of `V = 570` template face vertices a diagonal Gaussian `N(mu_{v,k}, sigma_{v,k}^2)` over the 3 axes, in mm.
The system never claims to identify an individual or produce a uniquely correct face.

## 2. Data
See `docs/GENERATOR.md`. Splits are made over **base geometries before observations are attached** (70/10/10/10 of 300 =
210/30/30/30; 4 observations each, severities cycled). Roles: train = fit weights and normaliser; validation = checkpoint selection
and weight choices; calibration = conformal fitting only; test = final reporting only. `dataset_report.json` records counts,
measured missingness, bounding boxes, unit plausibility, duplicate statistics and leakage checks recomputed from disk.

## 3. Preprocessing
One isotropic similarity `x_n = (x_mm - c)/s` fitted on the training bases (`c` = mean vertex, `s` = RMS radius, scalar).
Mapping back: `mu_mm = s mu_n + c`, `sigma_mm = s sigma_n`, `logvar_mm = logvar_n + 2 ln s`. Scalar `s` keeps a diagonal covariance
diagonal; anisotropic scaling is not used. Clouds are subsampled or zero-padded to a fixed size with a real-point mask. Padded
points never influence the encoder (tested: identical outputs with 0 or 200 garbage padding rows).
Augmentation: joint rotation (<=5 deg), translation (<=3 mm), isotropic scale (+-2%) applied to input and target; input-only jitter, dropout,
density change and extra cap occlusion (which updates the missing fraction approximately, assuming point share ~ area share).

## 4. Model
Encoder (PointNet baseline or PointNet++-style: FPS centroids, multi-scale kNN grouping with k=16/32, two levels + global) -> local tokens
(64 x 256) + global vector -> aggregation (Fourier positional encoding of token positions; context from global feature and, optionally, the
missing fraction; local-density feature at the input) -> template-conditioned decoder (one query per face vertex = Fourier-encoded template
position + learned per-vertex embedding + context; cross-attention to tokens, optional self-attention) -> Gaussian head:
`mu = template + delta`, `logvar = lo + (hi - lo) sigmoid(raw)` with bounds set in mm (predicted std in [0.1, 25] mm). ~0.76 M parameters.
Surface representation: fixed-topology template mesh with predicted vertex displacements (a face **patch**, not a watertight head).

## 5. Loss
`L = l1 L_rec + l2 L_NLL + l3 L_surface + l4 L_reg` with `L_rec` in {MSE, MAE, Chamfer}; coordinate-wise Gaussian NLL
`0.5 [logvar + (y-mu)^2/exp(logvar)]` (diagonal covariance; optional beta-NLL); `L_surface` = Laplacian + edge-length + normal-consistency
terms measured against the **target's** local geometry (so real detail is not smoothed away); `L_reg` = log-variance spatial smoothness.
Coordinates are evaluated in 10 mm units so terms have comparable scale. The NLL weight ramps from 0 over the first epochs.

## 6. Training
AdamW, linear warm-up + cosine decay, gradient clipping, optional fp16, atomic best/latest checkpoints with RNG state, automatic resume,
early stopping, seeds. Monitors recorded per epoch: non-finite loss/gradients, exploding gradients, uncertainty collapse/inflation
(fraction of predicted stds at the bounds), possible overfitting. A falling training loss is never taken as convergence.
Lessons recorded in the repo history: an early probabilistic run was mis-selected at epoch 0 by a 15-epoch patience; the deterministic model
shows a long initial plateau at template-level error before improving.

## 7. Uncertainty
Aleatoric: heteroscedastic variance from the NLL. Epistemic: **deep ensembles** (independent seeds); MC dropout is not used.
`var_aleatoric = mean_m sigma_m^2`, `var_epistemic = var_m(mu_m)`, combined = sum. Positional std = sqrt(sum over axes of variance)
(root-sum-square, not the std of the 3D distance). The three are always reported separately.

## 8. Calibration
Split conformal on the calibration split: score `|y - mu|/sigma` per coordinate, quantile `ceil((n+1)p)/n`. Scopes: global,
per-coordinate, per-region (k-means of the template), and subject-level max-score (simultaneous whole-surface coverage; needs enough bases).
Pointwise marginal coverage is **not** simultaneous coverage and is reported separately. Test targets are never used.

## 9. Evaluation
Reconstruction: vertex MAE/RMSE (correspondence), Chamfer and Hausdorff (point sets), point-to-surface and symmetric surface error (triangles).
Uncertainty: nominal vs observed coverage (50/80/90/95%), calibration error `mean_p |c(p) - p|`, Gaussian CRPS
`sigma [z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi)]`, NLL, sparsification error (uncertainty vs oracle vs random ranking, normalised), risk-coverage (AURC),
best-of-N (reported separately; it uses ground truth to select). Stress tests: severity/pattern/noise/density/sampler breakdowns of the stored test
set, and controlled single-factor sweeps regenerated on the test bases, including joint affine deformations.
The **irreducible floor** of the synthetic task is computed from the generator (`src/evaluation/oracle.py`) and quoted next to model errors.

## 10. Ablations
`python -m src.evaluation.ablation run|table`: full, two extra seeds of full (seed-noise estimate and ensemble members), deterministic (no NLL),
no surface regularisation, no missingness information, PointNet encoder, 2048 and 4096 points. All from scratch under one shared protocol with
validation-RMSE checkpoint selection. Differences smaller than the seed-to-seed gap are not claimed as effects.

## 11. What can and cannot be claimed
Can (once the corresponding result files exist): the pipeline runs and is tested; on this synthetic task, how error, predicted uncertainty, interval
coverage and sparsification behave; relative effects of the ablated components within this world.
Cannot: anything about real skulls or faces, forensic validity, identification, or calibration under real distribution shift. The synthetic
skull-to-face rule is invented here; the domain gap to real anatomy (soft-tissue structure, age/sex/ancestry/BMI variation, scan artefacts,
pose, damage patterns) is unquantified, and conformal guarantees do not transfer across it.

## 12. Reproducibility
`pip install -r requirements.txt`; `python -m src.data.build_dataset`; `python -m src.training.trainer --set ...`;
`python -m src.evaluation.run_evaluation --run <run>`; `python -m src.evaluation.report --main <run>`. Every run stores config, seed,
hardware, dataset version/config hash, normaliser, checkpoints and history in `experiments/runs/<name>/`. GPU kernels are not bit-deterministic;
CPU runs of the same seed reproduced exactly in tests.

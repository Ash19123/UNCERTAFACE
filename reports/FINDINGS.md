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

# Using UNCERTAFACE on a broken skull

Scenario: a skull is found with parts missing. You want the most likely face, several plausible faces, and to know which face regions the
model is sure about. Everything here is a SYNTHETIC-face research prototype: the faces follow an assumed soft-tissue rule, not real anatomy.
Not a forensic or identification tool.

```bash
python scripts/face_from_any_skull.py --skull found_skull.stl --out out/found_skull
```

Writes, in `out/found_skull/`:

| file | what it is |
|---|---|
| `face.ply` | most likely face; vertex colour = predicted uncertainty |
| `face_sample_1..4.ply` | four plausible faces (`--samples K` for other counts) |
| `skull_aligned.ply` | your skull, cleaned and aligned to the reference frame |
| `result.json` | missing fraction used, alignment residual, and `regions_most_to_least_certain` (mean +- half-width in mm per region) |

## How each part works
- **Alignment.** The mesh is cleaned and aligned to a reference skull (`real_skulls.prepare_case`); the tool fails if the residual or scale
  is implausible. Alignment of broken skulls was checked on three real skulls with known cuts (cap, one side, back): residual 3.2-3.8 mm.
- **Missing fraction.** `--missing-fraction auto` (default) estimates it as the share of reference-skull points with no input surface within
  8 mm, clipped to the trained range 0.10-0.85. On the three cut skulls it was within 0.03-0.07 of the true area fraction and always a little
  high (cap 0.20 true / 0.27 estimated; side 0.35 / 0.38; back 0.39 / 0.46). Pass a number to override it.
- **Most likely face.** The model's mean prediction.
- **Certainty.** Per coordinate, the calibrated 90% half-width = `q90 * sigma` from split conformal calibration (global scope, fitted on the
  calibration skulls only). For display one number per point is used: `q90 * sqrt(mean over axes of sigma^2)`. On the 6 held-out test skulls
  (48 views) these half-widths contained 90.7% of true face coordinates (target 90%).
- **Regions.** Six bands (forehead, eyes, nose, cheeks, mouth, chin and jaw) defined on the synthetic template (`src/uncertainty/plausible_faces.py`).
  The bands are rough, not anatomical landmarks. On the test skulls the regions' certainty ranking matched their real-error ranking
  (rank correlation 1.00 over 6 regions); the 25% most certain points had 2.8 mm mean error, the 25% least certain 7.6 mm
  (`reports/results/face_v3_pn/zone_certainty.json`, from `scripts/zone_certainty_check.py`).
- **Plausible faces.** Draws from the calibrated per-vertex Gaussian, with the noise smoothed over the face (12 mm Gaussian kernel) so the surface
  stays smooth. The smoothing length is an ASSUMPTION, not learned or validated. The samples show how much the skull leaves open; they are not
  better guesses: on the test skulls a sample's error was 8.4 mm against 5.9 mm for the most likely face (best of 4, chosen using the true
  face, 7.5 mm). Do not read a sample as a candidate real face.

## Limits
The model was trained on healthy adult CT skulls (MUG500+) with a synthetic face. It has no defence against skulls unlike its training set:
under 6-10% affine deformation the error rose to 9-13 mm while predicted uncertainty barely moved (`reports/FINDINGS.md`). Children, damaged,
non-human or heavily deformed skulls may align badly or be extrapolated, and the uncertainty will then understate the real error.

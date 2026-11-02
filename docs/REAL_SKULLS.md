# Real skulls: MUG500+ with a synthetic face

## What is real and what is not
| Component | Status |
|---|---|
| Skull geometry (`dataset v0.2.0`) | **REAL**: CT-derived, anonymised human skull meshes from MUG500+ |
| Face (the prediction target) | **SYNTHETIC**: a procedurally generated soft-tissue layer with assumed thicknesses |
| Incomplete observations (missing regions, noise, density) | simulated from the real skulls |
| Skull-to-face relationship | **invented by this project's face model**, not measured |

Because the face is synthetic, a model trained here learns *this project's face rule*, not real craniofacial anatomy.
Nothing here is forensic validation, and the system does not identify individuals.

## Source and licence
- Li J. et al., **MUG500+ Repository**, Figshare 2021, DOI [10.6084/m9.figshare.9616319](https://doi.org/10.6084/m9.figshare.9616319) (version 20), licence **CC BY 4.0**.
- Li J. et al., *MUG500+: Database of 500 high-resolution healthy human skulls and 29 craniotomy skulls and implants*, Data in Brief 2021, DOI 10.1016/j.dib.2021.107524.
- 500 healthy skulls segmented from head CT scans (Medical University of Graz). Please cite both references if you use the data.
- Only the cleaned meshes (`A####_clear.stl`) of the first N cases are fetched, via HTTP range requests from inside the zips,
  by `scripts/fetch_mug500.py`. The 210 GB of CT volumes are not downloaded.
- Handling: the raw and processed meshes live in `external_data/` and the built dataset in `UNCERTAFACE_DATA_REAL/`; both are
  **git-ignored and not redistributed by this repository**. Downloaded files are only parsed (binary STL via NumPy), never executed.

## Processing (`src/data/real_skulls.py`, `scripts/prepare_mug500.py`)
1. Parse the binary STL; vertex-clustering decimation to ~70k vertices; drop small connected components (CT speckle).
2. Rigid + uniform-scale trimmed ICP onto a reference skull (case A0001, rotated 180 degrees about z so +y is anterior); two
   initial poses are tried and the better residual is kept. A fit is **rejected** if the residual exceeds 8 mm or the fitted scale is
   outside 0.8-1.25 (free-scale ICP can shrink a corrupt mesh onto the reference, so the residual alone is not enough).
3. The skull is stored in the aligned frame in **its own millimetres** (real size variation is kept); the fitted scale is stored
   and used only to place the anatomical face grid. Neck/spine remnants beyond 30 mm of the reference are cropped (none were
   present in `_clear` meshes so far).
Observed so far (first cases): alignment residuals about 2-4 mm; fitted scales 0.93-1.00.

## The synthetic face (`src/data/face_from_skull.py`)
In the reference frame the front-most skull surface y(x, z) is rasterised (vertex splats), holes are filled, grey-scale closing
(22 mm radius) bridges the orbits, nasal opening and the gap between the teeth (soft tissue spans them), a 13 mm dilation keeps the
envelope at or above the bone, and Gaussian smoothing (4 mm) removes bony detail. The envelope sits on average ~7 mm above the bone
(including the bridged openings), and thickness is measured from it. Without the dilation 29% of skull vertices poked through the face (max 37 mm); now penetration is ~0 and is checked in every dataset report (`face_clears_bone`). The envelope is sampled on a fixed oval 5 mm grid (603 vertices), so vertex `v` is the same grid location for
every skull. `face = base + t * n` with `n` the base-grid vertex normal and thickness
`t = max(1, s * t0 * (1 + a g) + nose + lips + cheeks + u)`. `t0` blends assumed regional anchors (forehead 4.5, glabella 5, nose 3.5,
orbit 6, cheek 12, lip 8, mouth 10, chin 8 mm; anchor positions were set by eye on the reference skull). The latents
`s, a, g, nose, lips, cheeks, u` are drawn independently of the bone, so part of the face cannot be inferred from the skull.

## Limits
- The face is skull + normal offset, not real facial anatomy; the envelope is a frontal heightfield (frontal patch only).
- The thickness anchors are assumed, not clinical measurements; soft-tissue variation with age/sex/ancestry/BMI is not modelled.
- MUG500+ is a single clinical centre's CT population; it is not representative of all populations, and the first N cases are used,
  not a random sample.
- With a few dozen real skulls the split sets are small (e.g. 6 test skulls), so confidence intervals on every metric are wide.
  Joint non-rigid warp and left-right mirror augmentation (train only) create shape variants of the *training* skulls; they are not
  independent people and never cross splits.
- Left-right mirroring assumes the face grid and skull are approximately symmetric.


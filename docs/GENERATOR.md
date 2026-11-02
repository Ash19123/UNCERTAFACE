# Synthetic generator: model and assumptions

**Everything generated here is synthetic.** The generator (`src/data/generator.py`) is a simplified geometric research
prototype. It is **not** an anatomical simulation, and nothing it produces is real human, forensic, or clinical data.

## Asset audit
No real paired skull-face dataset was available in the execution environment, and none was downloaded, scraped or
redistributed. Parametric face models with restrictive licences (e.g. statistical head models) were not used. The default
and only data source is the procedural generator below.

| Asset | Source | Licence | Real/synthetic | Paired correspondence |
|---|---|---|---|---|
| `UNCERTAFACE_DATA` | `src/data/generator.py` (this repo) | repository licence (see `LICENSE`) | procedural, synthetic | exact, by construction |

## Geometry (units: mm; +x right, +y anterior, +z superior; origin at the cranial centre)
**Skull.** A deformed ellipsoid on a fixed icosphere topology (2562 vertices). Vertex `i` has the same unit direction in every
subject, so vertex `i` is in correspondence across all subjects. Radial factor
`1 + sum_i amp_i * bump_i(d)` with Gaussian bumps in geodesic angle on the sphere for: forehead, occiput, brow ridge,
zygoma, chin, jaw width, plus a nasal-aperture indentation, plus a smooth random field with a controllable
left-right antisymmetric part (`asymmetry`). Semi-axes (width/length/height) are drawn per subject.

**Face.** The frontal subset of skull vertices (|x| <= 0.8, anterior, below the forehead top), 570 vertices, shared topology.
`face_v = skull_v + n_v * t_v` where `n_v` is the skull vertex normal and the soft-tissue thickness is

`t = max(t_min, s * t0(d) * (1 + a * g(d)) + nose(d) + lips(d) + cheeks(d) + u(d))`

`t0` blends **assumed** anchor thicknesses (forehead 4.5, nasal bridge 3.0, nose 3.5, cheeks 10, mouth 10, chin 8 mm,
background 6 mm). These are prototype values, not clinical measurements.

**Irreducible ambiguity by design.** `s, a, g, nose, lips, cheeks, u` are drawn **independently of the bone parameters**.
Part of the face therefore cannot be inferred from the skull even with a complete noise-free skull. `src/evaluation/oracle.py`
measures this floor (redraw only these latents, hold the skull fixed). It is a property of this generator only.

## Bounds
All parameters are drawn uniformly from the ranges in `configs/dataset.yaml` (section `generator`). A candidate base is
redrawn if its skull is within 1.0 mm per-vertex RMS of an already accepted skull (a 2 mm rule was infeasible for 300
subjects because variation is dominated by three axis lengths). Parameter-space and geometric duplicate statistics are in
`dataset_report.json`.

## Observations (incomplete inputs)
Visibility is a function on the continuous skull surface (`src/data/pointcloud.py`): patchy Voronoi cells, spherical cap,
upper-only, lateral-only, or whole anatomical regions. The threshold is chosen from a dense area-uniform sample to hit a
random target inside the severity bin, then the **measured** removed area fraction is re-estimated on an independent dense
cloud (with Monte-Carlo standard error); targets are redrawn until the measurement is inside the bin. Bins: mild 10-20%,
moderate 30-45%, severe 50-70%, extreme 70-85%. Noise: isotropic Gaussian (0-1.5 mm), 0-2% outliers (5 mm scale), optional
density tilt, area-uniform or farthest-point sampling, 1024/2048/4096 points.

## Known limitations of the synthetic world
- Skull and face are smooth star-shaped surfaces; no sutures, teeth, orbits, nasal cavity, ears, eyes or hair.
- The face is the skull pushed along its normals; real soft tissue is not a normal offset.
- The skull-to-face rule (and its irreducible part) is whatever this generator encodes. A model can only learn this rule.
- Only 300 independent base geometries; the calibration split has 30 bases.
- Real-data distribution shift (age, sex, ancestry, BMI, scanner artefacts, pose) is absent.

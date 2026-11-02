# UNCERTAFACE — Phase 1: Research and Feasibility Audit

Status: design only. No code has been run, no results exist. Everything below is a proposal or a hypothesis unless marked "audited".

## 1. Formal task definition

**Notation.** Units are millimetres (mm) in a canonical skull-fixed frame unless stated otherwise.

- Complete cranial surface `S ⊂ R^3` (synthetic, known).
- Observation `X = {x_i}_{i=1..N}`, `x_i ∈ R^3`, obtained from `S` by an observation operator `O_φ`:
  `X = O_φ(S) = { T(p_j) + ε_j : p_j ∈ S, m(p_j)=1 }`, where `m: S → {0,1}` is a visibility mask (missingness), `ε_j ~ N(0, s²I)` is coordinate noise, and sampling density `N` varies. `φ = (m, s, N, sampler)` are the recorded observation parameters.
- Missingness fraction is **measured**: `ρ = area(S ∩ {m=0}) / area(S)`, computed on the mesh, not assumed.
- Target facial surface represented as a fixed-topology template mesh with `V` vertices, `Y ∈ R^{V×3}`, where vertex `v` has a known semantic correspondence across all subjects (template-index correspondence).
- Model: `p_θ(Y | X, M)` factorised as independent per-vertex, per-axis Gaussians:
  `y_{v,k} ~ N(μ_{v,k}(X,M;θ), σ²_{v,k}(X,M;θ))`, `k ∈ {x,y,z}`.

**Per-vertex NLL (coordinate-wise, diagonal covariance):**
`L_NLL = (1/3V) Σ_v Σ_k 0.5 [ log σ²_{v,k} + (y_{v,k} − μ_{v,k})² / σ²_{v,k} ]`
Parameterised as `s = log σ²` clamped to `[s_min, s_max]`, with bounds chosen in mm² (not normalised units) so the clamp is physically meaningful.

**Positional uncertainty from a diagonal Gaussian.** Total positional variance `σ²_pos,v = Σ_k σ²_{v,k}`; positional std `σ_pos,v = sqrt(σ²_pos,v)` (root-sum-square). The 3D error `‖y_v − μ_v‖` follows a scaled chi distribution with 3 d.o.f. only if the axes are equal-variance; in general it is a generalised chi-square, so 3D radii are computed by Monte Carlo or via per-axis intervals, and this is documented. Normal/tangent-direction uncertainty is a possible later refinement.

**Back to millimetres.** Normalisation is an isotropic similarity transform `x_n = (x − c)/s` with a single scalar `s` (mm). Then `μ_mm = s·μ_n + c`, `σ_mm = s·σ_n`, `σ²_mm = s²·σ²_n`. Isotropic scale is deliberate: anisotropic scaling would make per-axis variances non-interchangeable. The transform is stored per sample and unit-tested.

**Epistemic / combined (later experiment).** With `M` ensemble members: `μ̄ = mean μ_m`; aleatoric `= mean σ²_m`; epistemic `= var_m(μ_m)`; combined `= aleatoric + epistemic` (law of total variance). Reported separately and never mislabelled.

**Calibration (split conformal).** On the calibration split compute a nonconformity score, e.g. `r = |y − μ| / σ` per coordinate. The `⌈(n+1)(1−α)⌉/n` empirical quantile `q_{1−α}` gives interval `μ ± q σ`. Scope must be declared: **global per-coordinate** first, optionally region-specific. The guarantee is *marginal pointwise* coverage under exchangeability of calibration scores; it is **not** simultaneous whole-surface coverage. Because vertices within a subject are strongly dependent, exchangeability should be applied at the subject level (e.g. subject-level max or sampled-vertex scores) to avoid overstating the effective sample size — this will be examined, not assumed.

## 2. Architecture proposal

Primary surface representation: **fixed-topology template mesh with predicted per-vertex offsets** (option A). Rationale: paired synthetic data gives exact vertex correspondence; training is stable; per-vertex μ/σ is direct; point-to-surface and regularisers (Laplacian, edge length, normals) are well defined. Option B (point-based + separate surface reconstruction) is deferred and only attempted after A is complete. The output is a mesh with the template's topology; it is not a claim of anatomical watertightness beyond what the template provides.

```
Partial skull points X (N×3) + visibility/density feature
  → Encoder (PointNet++-style: FPS + ball-query set abstraction, 2–3 levels, multi-scale grouping)
  → Local feats F_l (M×C) and global feat g (C_g) via max+mean pooling
  → Feature aggregation: concat[g, attention-pooled F_l, positional encoding, missingness summary]
  → Template-conditioned decoder: per template vertex v, query = [template xyz_v, PE(xyz_v), z],
    cross-attention to local tokens F_l, then MLP
  → Head: Δ_v ∈ R^3 (μ_v = template_v + Δ_v) and s_v ∈ R^3 (log-variance, clamped)
```

Choices: begin with a PointNet baseline (simple, fast), then the hierarchical encoder as the recommended model. A lightweight transformer is optional. Target ≲ 2–5M parameters for the 6 GB RTX 3050 laptop GPU (audited below). Missingness enters as (a) a per-point flag/local-density feature and (b) an optional global missing-fraction scalar; the "no missingness info" ablation removes both. Deterministic baseline = same network, μ head only, MSE/Chamfer loss.

Total loss: `L = λ1 L_rec (MSE/MAE/Chamfer) + λ2 L_NLL + λ3 L_surf (Laplacian, edge-length, normal consistency) + λ4 L_reg (weight decay / variance regulariser)`. Weights are chosen on the validation split only. Known risk: jointly optimising μ and σ via NLL can under-train μ; mitigation is a warm-up with MSE only, or β-NLL, both treated as ablations.

## 3. Synthetic-data strategy (scientifically defensible version)

**Honest framing:** this is a *simplified geometric research prototype*, not an anatomical simulation. The skull is not a real skull and the face is not derived from real soft-tissue statistics.

Planned generator:
1. **Cranium:** a parametric closed surface (deformed ellipsoid/superellipsoid base on an icosphere) with bounded parameters: length, width, height, forehead slope, occipital bulge, zygomatic width, a mandible-like jaw lobe with controllable angle/width, a nasal-aperture-like indentation, and brow ridge. Smooth low-frequency random fields (spherical-harmonic-like, bounded amplitude) add shape variation; a bounded left-right asymmetry term is added. Regions (frontal, parietal, occipital, temporal-L/R, facial-skeleton, jaw) are labelled by position, to drive region-specific occlusion.
2. **Face:** a template mesh in fixed topology, generated by displacing each facial-region skull vertex along its surface normal by a **soft-tissue thickness field** `t(v) = t0(region) · (1 + a·g(v))`, with region baselines `t0` as *documented, assumed values* (not clinical measurements) and `g` a bounded smooth random field. To avoid the "plain inward offset" criticism, the face additionally receives (i) independent facial-feature parameters not recoverable from bone (nose-tip projection, lip fullness, cheek fat), which act as true irreducible ambiguity, and (ii) smoothing so the face is not a copy of skull shape. This makes the aleatoric uncertainty meaningful by construction, and the amount of unobservable variance is a known ground truth we can compare uncertainty to.
3. **Correspondence:** shared topology gives exact vertex-wise skull-to-face correspondence.
4. **Bounds and diversity:** every parameter has a documented range in `configs/dataset.yaml`. Duplicate/near-duplicate detection (parameter-space and Chamfer-based) is part of the dataset report.
5. **Splits by base identity** (70/10/10/10) before any observation variants; reports base geometry count and observation count separately.
6. **Missingness severity bins** computed from mesh area: mild 10–20, moderate 30–45, severe 50–70, extreme 70–85 percent. Generation is by accept/reject on the measured value. Patterns: random point removal, contiguous region occlusion, partial upper-cranial, partial lateral.
7. **Assets audit (audited at runtime, Phase 2 start):** I will check for locally available parametric models/assets and record source/licence/real-vs-synthetic/correspondence. I have not downloaded anything. Candidates such as FLAME or Basel Face Model are licence-restricted and would be used only if the user obtains them; no real skull–face paired set is known to be openly redistributable. Default path is the procedural generator.

## 4. Dependencies

Core: `torch`, `numpy`, `scipy`, `pyyaml`, `trimesh`, `matplotlib`, `plotly`, `pytest`, `tqdm`, `tensorboard`.
Optional: `open3d` (visualisation, FPS alternative), `pyvista`. I plan own PyTorch implementations of FPS/ball-query/Chamfer so that no compiled CUDA extension is needed. `torch_geometric` is not needed. Point-to-surface distance uses `trimesh`/`scipy.spatial.cKDTree`.

## 5. Execution plan (Colab and local)

- **Colab:** `pip install -r requirements.txt` on the default runtime (no CUDA extensions, so no build step); mount Drive for `data/` and `checkpoints/`; device auto-detect (CUDA else CPU); checkpoint-resume into Drive so disconnects are recoverable. Notebooks call into `src/` and contain no core logic.
- **Local audit of this machine (audited just now):** Windows 11, Python **3.14.3**, NVIDIA RTX 3050 Laptop **6 GB**. Installed: numpy, scipy, plotly, matplotlib, pyyaml, pytest. **Not installed:** torch, trimesh, open3d, pyvista, tensorboard. PyTorch wheels for Python 3.14 may not exist yet; I have not verified this. Options: install Python 3.11/3.12 in a venv for local work, or use Colab. This needs your decision.

## 6. Feasibility assessment

| Experiment | Feasible on 6 GB GPU / Colab | Notes |
|---|---|---|
| Generator + dataset (≈300–1000 base identities, several observations each) | Yes, CPU | Mesh size ~2–5k verts |
| Deterministic and Gaussian-NLL models, 1k–4k points | Yes | batch 8–16, AMP |
| Ablations (loss terms, missingness input, encoder, point count) | Yes but slow if serial | Many short runs, fixed seeds |
| Deep ensemble (e.g. 5 members) | Yes, 5× cost | Separate experiment, after baseline |
| Conformal calibration, CRPS, sparsification, risk–coverage | Yes | Cheap post-hoc |
| Region-specific / subject-level conformal | Yes | Needs enough calibration subjects (~10% of base geometries; with 500 bases that is only 50) — a real limitation |
| Real-data validation | No | No data |

Risks: small calibration subject count; NLL instability/uncertainty collapse; the model may learn the generator's soft-tissue rule exactly, giving trivially low error. These are why monitoring for collapsed/inflated σ is built in, and why results will be reported as they come out, including failures.

## 7. What synthetic data can and cannot support

**Can support (as demonstrated-on-synthetic only, once actually run):**
- The pipeline is correct, reproducible and unit-tested.
- Whether predicted σ tracks actual error when the data-generating process is known, including how much of the error is genuinely irreducible by construction.
- Whether conformal calibration repairs coverage under an exchangeable synthetic distribution.
- Relative effects of missingness, noise, density, loss terms and encoder choices *within this synthetic world*.

**Cannot support:**
- Any statement about accuracy on real human skulls or faces, forensic validity, or identification.
- That the learned skull→face mapping resembles real craniofacial anatomy (the mapping is whatever the generator encodes, plus assumed thickness values).
- Calibration under real-world distribution shift (the domain gap): real anatomy has structure, ancestry/age/sex/BMI variation, scan artefacts and correlated soft-tissue variation absent here. Synthetic calibration would likely not transfer, and this must be re-done on real held-out data.
- That "plausible face" means "this person's face". The system never establishes identity.

## 8. Proposed next steps (awaiting your go-ahead)

1. Decide the Python/PyTorch environment (venv with Python 3.11/3.12 locally vs Colab-first).
2. Phase 2, module 1: `src/data/mesh_utils.py` + `configs/dataset.yaml` + tests (icosphere, area, normals, validation, sampling), then `generator.py`.

Per your instructions, I will implement one module at a time and wait for execution and validation before building dependent modules.

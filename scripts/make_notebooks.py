"""Generate the nine notebooks in notebooks/ (library code lives in src/; notebooks inspect, plot and check).

  python scripts/make_notebooks.py          # write the .ipynb files (no outputs)
  python scripts/make_notebooks.py --run    # also execute them in place so they ship with outputs

The notebooks describe the REAL-SKULL experiment (60 MUG500+ skulls, SYNTHETIC faces, run ``face_v3_pn``) and need the built dataset and the
trained run (see README). Notebook 02 builds a tiny PROCEDURAL synthetic dataset in a temporary folder to show the generator on its own.
They are regenerated, not hand-edited.
"""
import json
import subprocess
import sys
from pathlib import Path
from inspect import cleandoc
from textwrap import dedent

SETUP = dedent("""\
    import sys, pathlib, os, json
    ROOT = pathlib.Path.cwd()
    while not (ROOT / 'src').exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
    import numpy as np, matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
    REAL, RUN, RES = 'UNCERTAFACE_DATA_REAL', 'experiments/runs/face_v3_pn', 'reports/results/face_v3_pn'
    if not pathlib.Path(REAL).exists():
        raise SystemExit('Build the real-skull dataset first (README: fetch_mug500, prepare_mug500, build_dataset).')""")


def nb(cells):
    out = []
    for kind, text in cells:
        c = {"cell_type": kind, "metadata": {}, "source": cleandoc(text).splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


M, C = "markdown", "code"
NOTEBOOKS = {
    "01_environment_setup": [
        (M, """# 01 Environment setup
        Checks Python, PyTorch, hardware and that the repo imports. Skulls are real (MUG500+, CC BY 4.0); faces are **synthetic**.

        On Colab: `!git clone https://github.com/Ash19123/UNCERTAFACE && cd UNCERTAFACE && pip install -r requirements.txt`, restart, run this notebook."""),
        (C, SETUP),
        (C, """import platform, torch, numpy, scipy, trimesh, yaml
        print('python', platform.python_version(), '|', platform.platform())
        print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
        print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| trimesh', trimesh.__version__)"""),
        (C, """from src.data import mesh_utils, generator, pointcloud, preprocessing, dataset, real_skulls
        from src.models.uncertaface import UncertaFace
        from src.uncertainty import calibration, ensemble, plausible_faces
        print('all modules import')"""),
        (C, """rep = json.load(open(REAL + '/dataset_report.json'))
        print('provenance:', rep['provenance'])
        print('real_skull =', rep['real_skull'], '| face_synthetic =', rep['face_synthetic'], '| units =', rep['units'])""")],
    "02_synthetic_dataset": [
        (M, """# 02 Procedural synthetic generator
        The original data source of the project: a simplified geometric skull and face (not anatomy; see `docs/GENERATOR.md`). A tiny build in a
        temporary folder shows what it produces. The real-skull experiment replaces the procedural skull by real scans but keeps this pipeline."""),
        (C, SETUP),
        (C, """import tempfile, yaml
        from src.data import build_dataset as bd, mesh_utils as mu
        cfg = yaml.safe_load(open('configs/dataset.yaml'))
        tmp = tempfile.mkdtemp()
        rep = bd.build_dataset(cfg, tmp, n_base=8, obs_per_base=2, source='procedural', log=lambda *a: None)
        print(json.dumps(rep['counts'], indent=1)); print('all checks passed:', rep['all_checks_passed'], '| synthetic:', rep['synthetic'])"""),
        (C, """%matplotlib inline
        from pathlib import Path
        sv, sf = mu.load_mesh(str(Path(tmp) / 'raw/generated/cranial_meshes/B0000.ply'))
        fv, ff = mu.load_mesh(str(Path(tmp) / 'raw/generated/facial_meshes/B0000.ply'))
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        for a, (i, j, t) in zip(ax, [(0, 2, 'front (x, z)'), (1, 2, 'side (y, z)'), (0, 1, 'top (x, y)')]):
            a.scatter(sv[:, i], sv[:, j], s=1, c='0.6', label='skull'); a.scatter(fv[:, i], fv[:, j], s=1, c='tab:orange', label='face')
            a.set_title(t); a.set_aspect('equal'); a.set_xlabel('mm')
        ax[0].legend(markerscale=6); plt.tight_layout(); plt.show()"""),
        (C, """print(open('docs/GENERATOR.md', encoding='utf-8').read()[:1800])""")],
    "03_data_inspection": [
        (M, """# 03 Data inspection (real skulls, synthetic faces)
        Sample counts, missingness distributions, point density, bounding boxes, unit verification, duplicate and leakage checks. The checks come from
        `validate_dataset`, which recomputes them from files on disk; the plots and the leakage check below are recomputed here too."""),
        (C, SETUP),
        (C, """from src.data import dataset as ds
        rep = json.load(open(REAL + '/dataset_report.json'))
        c = rep['counts']
        print('independent base geometries:', c['independent_base_geometries'], '| observations:', c['observations'])
        print('bases per split       :', c['base_per_split']); print('observations per split:', c['observations_per_split'])
        print('all integrity checks passed:', rep['all_checks_passed'])
        for k, v in rep['checks'].items(): print(f"  {k:32s}", v if not isinstance(v, dict) else v.get('passed'))"""),
        (C, """recs = {s: ds.read_records(REAL, s) for s in ['train', 'validation', 'calibration', 'test']}
        bases = {s: {r['base_id'] for r in rr} for s, rr in recs.items()}
        pairs = [(a, b) for a in bases for b in bases if a < b]
        print('split leakage (shared base geometries between splits):', {f'{a}/{b}': len(bases[a] & bases[b]) for a, b in pairs})
        assert all(len(bases[a] & bases[b]) == 0 for a, b in pairs)"""),
        (C, """allr = [r for rr in recs.values() for r in rr]
        sev = ['mild', 'moderate', 'severe', 'extreme']
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for s in sev:
            ax[0].hist([r['observation']['measured_missing_fraction'] for r in allr if r['observation']['severity'] == s], bins=20, alpha=.7, label=s)
        ax[0].set_xlabel('measured missing fraction (area)'); ax[0].set_title('missingness by severity'); ax[0].legend()
        npts = [r['observation']['n_points'] for r in allr]; u, k = np.unique(npts, return_counts=True)
        ax[1].bar([str(x) for x in u], k, color='0.3'); ax[1].set_title('points per observation'); ax[1].set_xlabel('points')
        ax[2].hist([r['observation']['noise_sigma_mm'] for r in allr], bins=20, color='0.3'); ax[2].set_title('coordinate noise sigma'); ax[2].set_xlabel('mm')
        plt.tight_layout(); plt.show()
        print('measured missingness (mean area fraction):', {k: round(v['measured_mean'], 3) for k, v in rep['missingness'].items()})
        print('occlusion patterns:', rep['patterns'])"""),
        (C, """bb = rep['bounding_box_mm']
        print('skull extent (mm)  min', np.round(bb['skull_extent_min'], 0), ' max', np.round(bb['skull_extent_max'], 0))
        print('face  extent (mm)  min', np.round(bb['face_extent_min'], 0), ' max', np.round(bb['face_extent_max'], 0))
        print('units plausible:', rep['checks']['units_plausible_mm'], '| face never inside bone:', rep['checks']['face_clears_bone'], rep['face_bone_penetration'])
        print('duplicates: min pairwise skull Chamfer (mm) =', round(rep['duplicates']['min_pairwise_skull_chamfer_mm'], 2),
              '| exact duplicates absent:', rep['checks']['no_exact_duplicate_skulls'], '| near duplicates absent:', rep['checks']['no_near_duplicate_skulls'])"""),
        (C, """from IPython.display import Image, display
        for f in ['missingness_density.png', 'example_sample.png']: display(Image(REAL + '/visualizations/' + f))""")],
    "04_preprocessing": [
        (M, """# 04 Preprocessing
        Train-fitted isotropic normaliser and its inverse (so uncertainty converts back to millimetres), fixed-size clouds with masks, kNN, sampling and augmentation."""),
        (C, SETUP),
        (C, """from src.visualization.results import load_run
        from src.data import dataset as ds, preprocessing as pp
        model, norm, cfg, root = load_run(RUN)
        print(norm)
        x = np.random.default_rng(0).normal(size=(2000, 3)) * 60
        print('round-trip max error (mm):', float(abs(norm.inverse(norm.forward(x)) - x).max()))
        print('normalised std 0.02 ->', float(norm.std_to_mm(0.02)), 'mm (scale =', float(norm.scale), 'mm per unit)')"""),
        (C, """d = ds.UncertafaceDataset(root, 'test', norm, 1024, 8, cache=False)
        it = d[0]
        print({k: tuple(v.shape) for k, v in it.items() if hasattr(v, 'shape')})
        print('real points:', int(it['mask'].sum()), '/', len(it['mask']), '| missing fraction:', round(float(it['missing_fraction']), 3), '| severity record:', d.records[0]['observation']['severity'])"""),
        (C, """%matplotlib inline
        from src.data import mesh_utils as mu
        sv, sf = mu.load_mesh(str(pathlib.Path(root) / ('raw/generated/cranial_meshes/B%04d.ply' % d.records[0]['base_id'])))
        rng = np.random.default_rng(0)
        area = mu.sample_surface(sv, sf, 2048, rng)
        idx = mu.farthest_point_sampling(sv, 2048) if hasattr(mu, 'farthest_point_sampling') else None
        fig, ax = plt.subplots(1, 2 if idx is not None else 1, figsize=(9, 4))
        ax = np.atleast_1d(ax)
        ax[0].scatter(area[:, 0], area[:, 2], s=1, c='0.3'); ax[0].set_title('area-weighted surface sampling (2048)'); ax[0].set_aspect('equal')
        if idx is not None: ax[1].scatter(sv[idx, 0], sv[idx, 2], s=1, c='0.3'); ax[1].set_title('farthest-point sampling (2048)'); ax[1].set_aspect('equal')
        plt.tight_layout(); plt.show()"""),
        (C, """from src.data import augmentation as aug
        print([n for n in dir(aug) if not n.startswith('_')][:12])
        print('augmentations are applied in the training loader only (rotation, translation, jitter, dropout, density change, small warp, mirror).')""")],
    "05_model_architecture": [
        (M, """# 05 Model architecture
        Masked PointNet encoder (the PointNet++ variant is the other option) -> feature aggregation (local, global, position, missingness) ->
        template-conditioned decoder with cross/self-attention -> Gaussian head with bounded log-variance. All outputs convert to millimetres."""),
        (C, SETUP),
        (C, """import torch
        from src.visualization.results import load_run
        from src.data import dataset as ds
        model, norm, cfg, root = load_run(RUN)
        print('encoder:', cfg['model']['encoder']['name'], '| parameters:', model.num_parameters(), '| probabilistic:', cfg['model']['probabilistic'])
        print('template (face vertices):', tuple(model.decoder.template.shape))"""),
        (C, """d = ds.UncertafaceDataset(root, 'test', norm, cfg['data']['n_points'], cfg['data']['knn_k'], cache=False)
        b = next(iter(ds.make_loader(d, 4, False)))
        model.eval()
        with torch.no_grad(): out = model.forward_batch(b)
        print({k: tuple(v.shape) for k, v in out.items() if v is not None})
        from src.uncertainty.estimation import gaussian_to_mm
        phys = gaussian_to_mm(out['mean'], out['logvar'], norm)
        s = phys['std_mm']
        print('predicted per-axis std (mm): min %.2f mean %.2f max %.2f' % (float(s.min()), float(s.mean()), float(s.max())))
        assert torch.isfinite(s).all() and (s > 0).all(); print('uncertainty is positive and finite')"""),
        (C, """tpl = norm.inverse(model.decoder.template.cpu().numpy())
        fig, ax = plt.subplots(1, 2, figsize=(8, 4.2))
        ax[0].scatter(tpl[:, 0], tpl[:, 2], s=2, c=tpl[:, 1], cmap='Greys'); ax[0].set_title('face template, front (colour = depth)'); ax[0].set_aspect('equal')
        ax[1].scatter(tpl[:, 1], tpl[:, 2], s=2, c='0.3'); ax[1].set_title('side profile'); ax[1].set_aspect('equal')
        plt.tight_layout(); plt.show()""")],
    "06_training": [
        (M, """# 06 Training
        Training is run from the command line so it is resumable and logged (`scripts/run_face_v3_pointnet.ps1`). This notebook inspects the finished runs:
        configuration, hardware, duration, health warnings (NaN, exploding gradients, collapsing or inflating uncertainty) and learning curves. A falling
        training loss alone is not treated as convergence: validation error and calibration are checked too."""),
        (C, SETUP),
        (C, """# summary of both finished runs
        for name in ['face_v3', 'face_v3_pn']:
            e = json.load(open(f'experiments/runs/{name}/experiment.json'))
            print(name, {k: e[k] for k in ['status', 'epochs_run', 'best_epoch', 'monitor', 'best_monitor', 'duration_s', 'model_params']})
            print('   hardware:', e['hardware'], '| seed:', e.get('seed'), '| warnings:', len(e['all_warnings']))"""),
        (C, """fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for name, c in [('face_v3 (PointNet++)', 'tab:blue'), ('face_v3_pn (PointNet)', 'tab:orange')]:
            h = json.load(open(f"experiments/runs/{name.split()[0]}/history.json")); ep = [x['epoch'] for x in h]
            ax[0].plot(ep, [x['train_loss'] for x in h], c=c, label=name); ax[1].plot(ep, [x['val_rmse_mm'] for x in h], c=c); ax[2].plot(ep, [x['val_pos_std_mm'] for x in h], c=c)
        ax[0].set_title('training loss'); ax[1].set_title('validation vertex RMSE (mm)'); ax[2].set_title('validation mean predicted positional std (mm)')
        for a in ax: a.set_xlabel('epoch')
        ax[0].legend(); plt.tight_layout(); plt.show()"""),
        (C, """from IPython.display import Image, display
        display(Image(RES + '/training_curves.png'))"""),
        (M, "The 60-epoch ablation runs (`abl_*`) use the same protocol; their histories are in `experiments/runs/abl_*/` and summarised in notebook 08.")],
    "07_uncertainty_calibration": [
        (M, """# 07 Uncertainty calibration
        Split conformal calibration, fitted on the **calibration split only** (6 skulls, 48 views). The test split is used for reporting only.
        Scopes: global, per-coordinate, per-region, subject-level (simultaneous). Pointwise coverage is never reported as simultaneous coverage."""),
        (C, SETUP),
        (C, """r = json.load(open(RES + '/results.json'))
        cal = json.load(open(RES + '/calibrator_global.json'))
        print('global conformal scale factors q (multiply the predicted std):', {k: round(v, 3) for k, v in cal['q'].items()}, '| n scores:', cal['n_scores'])
        print(f"{'level':>6} {'raw cov':>9} {'calibrated':>11} {'raw halfwidth mm':>17} {'cal halfwidth mm':>17} {'simult (raw)':>13}")
        for l in ['50', '80', '90', '95']:
            raw, g = r['calibration']['raw'][l], r['calibration']['global']['test'][l]
            print(f"{l:>6} {raw['pointwise']:9.3f} {g['pointwise']:11.3f} {raw['mean_half_width_mm']:17.2f} {g['mean_half_width_mm']:17.2f} {raw['simultaneous']:13.3f}")"""),
        (C, """lv = [50, 80, 90, 95]
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.plot([50, 95], [50, 95], 'k--', lw=1, label='ideal')
        ax.plot(lv, [100 * r['calibration']['raw'][str(l)]['pointwise'] for l in lv], 'o-', c='tab:red', label='raw model')
        ax.plot(lv, [100 * r['calibration']['global']['test'][str(l)]['pointwise'] for l in lv], 'o-', c='tab:blue', label='conformal (global)')
        ax.set_xlabel('nominal coverage (%)'); ax.set_ylabel('observed coverage (%)'); ax.legend(); plt.tight_layout(); plt.show()"""),
        (C, """u = r['uncertainty_raw']
        print('raw CRPS (mm): %.3f | NLL: %.3f | standardised residual mean/std: %.3f / %.3f' % (u['crps_mm'], u['nll'], u['z_mean'], u['z_std']))
        print('calibration error (mean |observed - nominal|):', round(u['coverage']['ece_coverage'], 3))
        print('subject-level simultaneous coverage, subject_max scope:', {l: round(r['calibration']['subject_max']['test'][l]['simultaneous'], 3) for l in ['50', '80', '90', '95']})
        print('note: with few calibration skulls the highest simultaneous levels become conservative or infinite.')"""),
        (C, """from IPython.display import Image, display
        display(Image(RES + '/reliability.png'))""")],
    "08_evaluation": [
        (M, """# 08 Evaluation
        Reconstruction metrics (vertex, point-set, surface), CRPS, coverage, sparsification, risk-coverage, stress tests, ablations, the ensemble, and
        bootstrap intervals over test skulls. The test split is for final reporting only. All numbers are read from saved results and are errors against a
        **synthetic** face."""),
        (C, SETUP),
        (C, """r = json.load(open(RES + '/results.json')); b = json.load(open('reports/baselines.json')); o = json.load(open('reports/oracle_floor.json'))
        print('test views:', r['n_test'])
        for k, v in r['reconstruction'].items(): print(f"  {k:18s} {v['mean']:.2f} +- {v['std']:.2f} mm")
        print('mean-face baseline vertex RMSE: %.2f mm | oracle floor: %.2f mm' % (b['mean_template']['test']['vertex_rmse']['mean'], o['floor_rmse_mm']))
        sev = r['reconstruction_by_severity']
        print('by severity:', {k: round(v['vertex_rmse']['mean'], 2) for k, v in sev.items()})"""),
        (C, """ci = json.load(open(RES + '/bootstrap_ci.json'))
        print(f"{ci['n_test_skulls']} test skulls, bootstrap over skulls ({ci['n_boot']} draws), 95% intervals:")
        for k in ['model_rmse_mm', 'baseline_rmse_mm', 'coverage90_calibrated']:
            print(f"  {k:24s} {ci[k]['estimate']:.3f}  [{ci[k]['ci95'][0]:.3f}, {ci[k]['ci95'][1]:.3f}]")
        g = ci['relative_gain_over_baseline']
        print(f"  gain over mean-face baseline {g['estimate']*100:.1f}%  [{g['ci95'][0]*100:.1f}%, {g['ci95'][1]*100:.1f}%]; model better in {g['share_of_draws_model_better']*100:.2f}% of draws")
        print(ci['note'])"""),
        (C, """fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for name in ['model_rmse_mm', 'baseline_rmse_mm']: ax[0].plot(range(1, 7), ci[name]['per_skull'], 'o-', label=name.replace('_rmse_mm', ''))
        ax[0].set_xlabel('test skull'); ax[0].set_ylabel('vertex RMSE (mm)'); ax[0].legend(); ax[0].set_title('per-skull error')
        ax[1].bar(list(sev), [v['vertex_rmse']['mean'] for v in sev.values()], color='0.3'); ax[1].set_title('vertex RMSE by missingness (mm)')
        plt.tight_layout(); plt.show()"""),
        (C, """sp = r['sparsification']; print({k: round(v, 3) for k, v in sp.items()})
        print('risk (mean error mm) at retained coverage:', {k: round(v, 2) for k, v in r['risk_coverage'].items()})
        print('best-of-N (oracle-selected, NOT a single prediction):', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r['best_of_n'].items() if k != 'note'})
        from IPython.display import Image, display
        for f in ['sparsification.png', 'error_vs_uncertainty.png', 'spatial_maps.png']: display(Image(RES + '/' + f))"""),
        (C, """sw = r['stress_controlled']
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
        for a, f in zip(ax, ['noise', 'density', 'sampler', 'deformation']):
            rows = sw[f]; a.bar([str(x['group']) for x in rows], [x['rmse_mm'] for x in rows], color='0.3')
            a2 = a.twinx(); a2.plot([str(x['group']) for x in rows], [100 * x['cov90_raw'] for x in rows], 'o-', c='tab:red'); a2.set_ylim(40, 100)
            a.set_title(f); a.set_ylabel('RMSE (mm)')
        plt.tight_layout(); plt.show()
        print('deformation sweep: error rises but predicted std barely moves ->', [(x['group'], round(x['rmse_mm'], 1), round(x['mean_pos_std_mm'], 1)) for x in sw['deformation']])"""),
        (C, """import csv
        rows = list(csv.DictReader(open('reports/ablation_table.csv')))
        print(f"{'variant':18s} {'test RMSE':>10} {'cov90 raw':>10} {'cov90 cal':>10}")
        f3 = lambda v: f"{float(v):.3f}" if v not in ('', None) else 'n/a'
        for x in rows: print(f"{x['variant']:18s} {float(x['test_vertex_rmse_mm']):10.2f} {f3(x['cov90_raw']):>10} {f3(x['cov90_global']):>10}")
        full = [float(x['test_vertex_rmse_mm']) for x in rows if x['variant'].startswith('full')]
        print('seed-to-seed spread of the identical model: %.2f - %.2f mm; differences smaller than this are noise' % (min(full), max(full)))"""),
        (C, """e = json.load(open('reports/results/ensemble3/results.json'))
        print('single members (mm):', {k: round(v, 2) for k, v in e['single_member_rmse_mm'].items()}, '| ensemble:', round(e['ensemble_reconstruction']['vertex_rmse']['mean'], 2))
        print('epistemic share of total variance: %.3f' % e['epistemic_share_of_total_variance'])
        for k, v in e['uncertainty'].items(): print(f"  {k:10s} mean std {v['mean_std_mm']:.2f} mm | raw cov90 {v['cov90_raw']:.3f} | err-vs-std Spearman {v['pooled_spearman_err_vs_std']:.3f}")""")],
    "09_visualization": [
        (M, """# 09 Visualisation
        The broken-skull workflow on held-out test skulls: the hidden bone, the most likely face, where it is certain, four plausible faces, and the
        regions ranked from most to least certain. Millimetre scales are labelled. The interactive version is the demo page. Faces are synthetic."""),
        (C, SETUP),
        (C, """from src.visualization.results import load_run, predict_sample, pick_by_severity, reconstruction_figure
        from src.data import dataset as ds
        model, norm, cfg, root = load_run(RUN)
        d = ds.UncertafaceDataset(root, 'test', norm, cfg['data']['n_points'], cfg['data']['knn_k'], cache=False)
        idx = pick_by_severity(d); preds = {s: predict_sample(model, norm, d, i) for s, i in idx.items()}
        summ = reconstruction_figure(preds, 'reports/reconstruction_test.png', 'face_v3_pn, held-out test skull')
        print({s: {k: round(v, 2) for k, v in x.items() if isinstance(v, float)} for s, x in summ.items()})
        from IPython.display import Image, display; display(Image('reports/reconstruction_test.png'))"""),
        (C, """%matplotlib inline
        from src.uncertainty import plausible_faces as pf
        from src.uncertainty.calibration import ConformalCalibrator
        from src.evaluation.predict import collect
        o = collect(model, norm, d, 'cpu')
        i = idx['severe']
        q90 = float(ConformalCalibrator.load(RES + '/calibrator_global.json').scale(0.9, o['mean'].shape))
        tpl = norm.inverse(model.decoder.template.cpu().numpy())
        W = pf.smoothing_matrix(tpl[:, [0, 2]], 12.0)
        samples = pf.sample_faces(o['mean'][i], o['std'][i], W, q90, 4, seed=1)
        hw = q90 * np.sqrt((o['std'][i] ** 2).mean(-1))
        fig, ax = plt.subplots(1, 6, figsize=(17, 3.6))
        def face(a, v, c, title, cmap='Greys', vmin=None, vmax=None):
            s = a.scatter(v[:, 0], v[:, 2], s=7, c=c, cmap=cmap, vmin=vmin, vmax=vmax); a.set_aspect('equal'); a.set_title(title, fontsize=9); a.axis('off'); return s
        face(ax[0], o['gt'][i], o['gt'][i][:, 1], 'synthetic truth (depth)')
        face(ax[1], o['mean'][i], o['mean'][i][:, 1], 'most likely face')
        s = face(ax[2], o['mean'][i], hw, 'certain to within (mm, 90%)', 'Oranges', 0, float(np.quantile(hw, .98))); plt.colorbar(s, ax=ax[2], fraction=.046)
        for k in range(3): face(ax[3 + k], samples[k], samples[k][:, 1], f'plausible face {"ABC"[k]}')
        plt.tight_layout(); plt.show()"""),
        (C, """zone = pf.zone_ids(tpl[:, 0], tpl[:, 2])
        spread = np.linalg.norm(samples.std(0), axis=1)
        rows = pf.zone_table(zone, hw, spread)
        err = np.linalg.norm(o['mean'][i] - o['gt'][i], axis=1)
        print(f"{'region':14s} {'+- half-width mm':>17} {'faces differ mm':>16} {'actual error mm':>16}")
        for r_ in rows: print(f"{r_['zone']:14s} {r_['halfwidth_mm']:17.1f} {r_['sample_spread_mm']:16.1f} {err[zone == pf.ZONES.index(r_['zone'])].mean():16.1f}")"""),
        (C, """z = json.load(open(RES + '/zone_certainty.json'))
        print('over all', z['n_test'], 'test views: region rank correlation (certainty vs real error) =', round(z['zone_rank_spearman_halfwidth_vs_error'], 2))
        print('error in the 25%% most / least certain points: %.2f / %.2f mm' % (z['error_in_25pct_most_certain_mm'], z['error_in_25pct_least_certain_mm']))
        print('shown half-widths contain %.1f%% of true coordinates (target 90)' % (100 * z['coverage_per_coordinate_inside_halfwidth']))
        print('samples: mean sample RMSE %.2f mm vs mean prediction %.2f mm; best-of-%d (oracle) %.2f mm' % (z['samples']['mean_sample_rmse_mm'], z['samples']['mean_prediction_rmse_mm'], z['k'], z['samples']['best_of_k_rmse_mm']))
        print('The plausible faces illustrate the size of the uncertainty. They are not better guesses than the most likely face.')""")],
}

if __name__ == "__main__":
    out = Path(__file__).parents[1] / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (out / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print(f"wrote {len(NOTEBOOKS)} notebooks to {out}")
    if "--run" in sys.argv:
        for name in NOTEBOOKS:
            p = out / f"{name}.ipynb"
            r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                "--ExecutePreprocessor.timeout=900", str(p)], capture_output=True, text=True)
            print(name, "ok" if r.returncode == 0 else "FAILED\n" + r.stderr[-1500:])

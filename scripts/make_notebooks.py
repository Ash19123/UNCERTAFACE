"""Generate the nine thin notebooks in notebooks/ (library code lives in src/, not in notebooks).

  python scripts/make_notebooks.py

Notebooks are small drivers with markdown explanations. They are regenerated, not hand-edited.
"""
import json
from pathlib import Path

SETUP = ("import sys, pathlib\nROOT = pathlib.Path.cwd()\nwhile not (ROOT / 'src').exists() and ROOT != ROOT.parent:\n"
         "    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nimport os; os.chdir(ROOT)")


def nb(cells):
    out = []
    for kind, text in cells:
        c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


M, C = "markdown", "code"
NOTEBOOKS = {
    "01_environment_setup": [
        (M, "# 01 Environment setup\nChecks Python, PyTorch, CUDA and the repo imports. **All data in this project is synthetic.**\n\n"
            "On Colab: `!git clone https://github.com/Ash19123/UNCERTAFACE && cd UNCERTAFACE && pip install -r requirements.txt` then restart and run this notebook."),
        (C, SETUP),
        (C, "import torch, numpy, scipy, trimesh, yaml\nprint('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),\n"
            "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\nprint('numpy', numpy.__version__, 'scipy', scipy.__version__)"),
        (C, "from src.data import mesh_utils, generator, pointcloud, preprocessing, dataset\nfrom src.models.uncertaface import UncertaFace\nprint('imports ok')")],
    "02_synthetic_dataset": [
        (M, "# 02 Synthetic dataset\nGenerates the paired synthetic skull/face data from `configs/dataset.yaml` (deterministic given the seed). "
            "The generator is a simplified geometric model, **not anatomy**; see `docs/GENERATOR.md`.\nA small build is used here; the full build is `python -m src.data.build_dataset`."),
        (C, SETUP),
        (C, "import yaml, json\nfrom src.data import build_dataset as bd\ncfg = yaml.safe_load(open('configs/dataset.yaml'))\n"
            "rep = bd.build_dataset(cfg, 'UNCERTAFACE_DATA_demo', n_base=10, obs_per_base=2)\nprint(json.dumps(rep['counts'], indent=1)); print('all checks passed:', rep['all_checks_passed'])")],
    "03_data_inspection": [
        (M, "# 03 Data inspection\nSample counts, missingness distributions, point density, bounding boxes, unit verification, duplicate and split-leakage checks "
            "(all recomputed from files on disk by `validate_dataset`). Uses the full dataset if present."),
        (C, SETUP),
        (C, "import json, yaml, pathlib\nroot = pathlib.Path('UNCERTAFACE_DATA')\nrep = json.load(open(root / 'dataset_report.json'))\n"
            "print(json.dumps(rep['counts'], indent=1))\nprint('missingness by severity:'); [print(' ', k, {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items() if a != 'bin'}) for k, v in rep['missingness'].items()]\n"
            "print('point counts:', rep['point_density']['counts']); print('bbox (mm):', rep['bounding_box_mm'])\nprint('duplicates:', rep['duplicates'])\n"
            "print('checks:'); [print(' ', k, v if not isinstance(v, dict) else v['passed']) for k, v in rep['checks'].items()]"),
        (C, "from IPython.display import Image, display\nfor f in ['missingness_density.png', 'example_sample.png']:\n    display(Image(str(root / 'visualizations' / f)))")],
    "04_preprocessing": [
        (M, "# 04 Preprocessing\nTrain-fitted isotropic normaliser, inverse transform, uncertainty conversion to millimetres, fixed-size clouds with masks, kNN."),
        (C, SETUP),
        (C, "import numpy as np\nfrom src.data import preprocessing as pp\nnorm = pp.Normalizer.load('UNCERTAFACE_DATA/metadata/normalizer.json')\nprint(norm)\n"
            "x = np.random.default_rng(0).normal(size=(5, 3)) * 50\nprint('roundtrip max error (mm):', abs(norm.inverse(norm.forward(x)) - x).max())\n"
            "print('std 0.02 normalised ->', float(norm.std_to_mm(0.02)), 'mm')"),
        (C, "from src.data import dataset as ds\nd = ds.UncertafaceDataset('UNCERTAFACE_DATA', 'validation', norm, 1024)\nit = d[0]\n"
            "print({k: tuple(v.shape) for k, v in it.items() if hasattr(v, 'shape')}); print('real points:', int(it['mask'].sum()), 'missing fraction:', float(it['missing_fraction']))")],
    "05_model_architecture": [
        (M, "# 05 Model architecture\nMasked PointNet / PointNet++ encoder -> aggregation -> template-conditioned decoder -> Gaussian head (bounded log-variance)."),
        (C, SETUP),
        (C, "import yaml, torch\nfrom src.data import dataset as ds, preprocessing as pp\nfrom src.models.uncertaface import UncertaFace\n"
            "R = 'UNCERTAFACE_DATA'; norm = pp.Normalizer.load(R + '/metadata/normalizer.json')\ntpl = ds.fit_template(R, norm, pp.read_split(R, 'train')['base_ids'])\n"
            "cfg = yaml.safe_load(open('configs/model.yaml'))['model']\nm = UncertaFace(cfg, tpl, norm.scale)\nprint('parameters:', m.num_parameters())\n"
            "b = next(iter(ds.make_loader(ds.UncertafaceDataset(R, 'validation', norm), 4, False)))\nout = m.forward_batch(b)\n"
            "print({k: tuple(v.shape) for k, v in out.items()})"),
        (C, "from src.uncertainty.estimation import gaussian_to_mm\nphys = gaussian_to_mm(out['mean'], out['logvar'], norm)\n"
            "print('initial predicted per-axis std (mm):', float(phys['std_mm'].mean()))")],
    "06_training": [
        (M, "# 06 Training\nTrains via the CLI so runs are resumable and logged. The commands used for the reported runs:\n\n"
            "```\npython -m src.training.trainer --set run.name=probabilistic_scratch optim.epochs=150 early_stopping.patience=40 early_stopping.monitor=val_nll_mm\n"
            "python -m src.evaluation.ablation run --epochs 60\n```\nThis notebook only inspects a finished run's history and warnings."),
        (C, SETUP),
        (C, "import json\nrun = 'experiments/runs/probabilistic_scratch'\nexp = json.load(open(run + '/experiment.json'))\n"
            "print({k: exp[k] for k in ['status', 'epochs_run', 'best_epoch', 'monitor', 'best_monitor', 'duration_s', 'model_params']})\nprint('hardware:', exp['hardware'])\nprint('warnings:', exp['all_warnings'][:5])"),
        (C, "from src.visualization import uncertainty_viewer as uv\nuv.training_curves(json.load(open(run + '/history.json')), 'reports/training_curves_main.png')\n"
            "from IPython.display import Image; Image('reports/training_curves_main.png')")],
    "07_uncertainty_calibration": [
        (M, "# 07 Uncertainty calibration\nSplit conformal calibration on the **calibration split only**; scopes: global, per-coordinate, per-region, subject-level (simultaneous). "
            "Pointwise coverage is never reported as simultaneous coverage."),
        (C, SETUP),
        (C, "import json\nr = json.load(open('reports/results/probabilistic_scratch/results.json'))\n"
            "for lvl in ['50', '80', '90', '95']:\n    print(lvl, 'raw', round(r['calibration']['raw'][lvl]['pointwise'], 3),\n"
            "          'global', round(r['calibration']['global']['test'][lvl]['pointwise'], 3),\n"
            "          'subject_max simultaneous', round(r['calibration']['subject_max']['test'][lvl]['simultaneous'], 3))"),
        (C, "from IPython.display import Image; Image('reports/results/probabilistic_scratch/reliability.png')")],
    "08_evaluation": [
        (M, "# 08 Evaluation\nReconstruction metrics (vertex / point-set / surface), CRPS, coverage, sparsification, risk-coverage, stress tests. "
            "Produced by `python -m src.evaluation.run_evaluation --run experiments/runs/<run>`; test split is for final reporting only."),
        (C, SETUP),
        (C, "import json\nr = json.load(open('reports/results/probabilistic_scratch/results.json'))\n"
            "print({k: round(v['mean'], 3) for k, v in r['reconstruction'].items()})\nprint(r['sparsification'])\nprint(r['risk_coverage'])\nprint(r['best_of_n'])"),
        (C, "from IPython.display import Image, display\nfor f in ['sparsification.png', 'error_vs_uncertainty.png', 'spatial_maps.png', 'stress_sweeps.png']:\n"
            "    display(Image('reports/results/probabilistic_scratch/' + f))")],
    "09_visualization": [
        (M, "# 09 Visualisation\nValidation reconstructions per missingness level, and the interactive progressive-reveal 'face being built' animation (real model predictions)."),
        (C, SETUP),
        (C, "from src.visualization.results import load_run, predict_sample, pick_by_severity, reconstruction_figure\nfrom src.data import dataset as ds\n"
            "model, norm, cfg, root = load_run('experiments/runs/probabilistic_scratch')\nd = ds.UncertafaceDataset(root, 'validation', norm, cfg['data']['n_points'], cfg['data']['knn_k'])\n"
            "preds = {s: predict_sample(model, norm, d, i) for s, i in pick_by_severity(d).items()}\n"
            "print(reconstruction_figure(preds, 'reports/reconstruction_validation.png', 'probabilistic_scratch'))"),
        (C, "from IPython.display import Image; Image('reports/reconstruction_validation.png')"),
        (C, "from src.visualization.build_animation import build\nbuild('experiments/runs/probabilistic_scratch', 'reports/face_build.html')\nprint('open reports/face_build.html in a browser')")],
}

if __name__ == "__main__":
    out = Path(__file__).parents[1] / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (out / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1))
    print(f"wrote {len(NOTEBOOKS)} notebooks to {out}")

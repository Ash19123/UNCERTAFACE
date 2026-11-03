# Face v2 (3 mm grid, facial relief): rebuild the real-skull dataset, train, floor/baseline, evaluate (test split, once).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$py = ".\.venv\Scripts\python.exe"
& $py -m src.data.build_dataset --config configs/dataset_mug500.yaml --out UNCERTAFACE_DATA_REAL
& $py -m src.training.trainer --set run.name=face_v2 optim.epochs=120 optim.batch_size=8 early_stopping.patience=50 early_stopping.monitor=val_nll_mm augmentation.warp_mm=3.0 augmentation.mirror_prob=0.5
& $py -m src.evaluation.oracle
& $py -m src.evaluation.baselines
& $py -m src.evaluation.run_evaluation --run experiments/runs/face_v2 --no-sweeps

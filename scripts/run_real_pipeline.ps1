# Real-skull pipeline: prepare skulls -> build dataset -> train -> floor/baseline -> evaluate (test split, once).
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$py = ".\.venv\Scripts\python.exe"
& $py scripts\prepare_mug500.py
& $py -m src.data.build_dataset --config configs/dataset_mug500.yaml --out UNCERTAFACE_DATA_REAL
& $py -m src.training.trainer --set run.name=real_main optim.epochs=150 early_stopping.patience=60 early_stopping.monitor=val_nll_mm augmentation.warp_mm=3.0 augmentation.mirror_prob=0.5
& $py -m src.evaluation.oracle
& $py -m src.evaluation.baselines
& $py -m src.evaluation.run_evaluation --run experiments/runs/real_main --no-sweeps

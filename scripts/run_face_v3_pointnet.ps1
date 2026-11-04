# face_v3_pn: same protocol as face_v3 but with the plain PointNet encoder (it led the 60-epoch ablation).
# Oracle/baselines are unchanged (data identical), so only train + evaluate.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$py = ".\.venv\Scripts\python.exe"
& $py -m src.training.trainer --set run.name=face_v3_pn model.encoder.name=pointnet optim.epochs=150 optim.batch_size=8 early_stopping.patience=50 early_stopping.monitor=val_nll_mm augmentation.warp_mm=3.0 augmentation.mirror_prob=0.5
& $py -m src.evaluation.run_evaluation --run experiments/runs/face_v3_pn --no-sweeps

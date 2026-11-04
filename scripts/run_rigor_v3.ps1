# Ablations (9 variants, 60 epochs, shared protocol), 3-member deep ensemble, then stress sweeps on face_v3.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
$py = ".\.venv\Scripts\python.exe"
& $py -m src.evaluation.ablation run --epochs 60
& $py -m src.evaluation.ablation table
& $py -m src.evaluation.run_ensemble --runs experiments/runs/abl_full experiments/runs/abl_full_seed1 experiments/runs/abl_full_seed2 --name ensemble3
& $py -m src.evaluation.run_evaluation --run experiments/runs/face_v3

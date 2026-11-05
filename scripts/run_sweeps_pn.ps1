# Full evaluation (with controlled stress sweeps) for the headline model; settings from configs/evaluation.yaml.
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "."
& .\.venv\Scripts\python.exe -m src.evaluation.run_evaluation --run experiments/runs/face_v3_pn

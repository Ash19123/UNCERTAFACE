# Waits for the main training run to finish, then computes the irreducible floor and the final test-set
# evaluation of the main run. (The test split is evaluated once, for the model fixed in advance.)
param([string]$Main = "probabilistic_scratch")
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "run.name=$Main\b" }) { Start-Sleep -Seconds 30 }
& $py -m src.evaluation.oracle
& $py -m src.evaluation.run_evaluation --run "experiments/runs/$Main"

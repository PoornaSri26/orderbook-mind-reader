#!/usr/bin/env pwsh
# PowerShell stand-in for `make reproduce`.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"
$py = if ($env:PY) { $env:PY } else { "python" }
$stages = if ($args.Count -gt 0) { $args } else { @("dataset", "baselines", "model", "interp") }
$map = @{
    "data"      = @("scripts/download_bookticker.py", "--days", "7", "--end", "2024-03-30")
    "dataset"   = @("scripts/build_dataset.py")
    "baselines" = @("scripts/run_baselines.py")
    "model"     = @("scripts/run_model.py")
    "interp"    = @("scripts/run_interp.py")
}
foreach ($s in $stages) {
    Write-Host "== stage: $s"
    & $py $map[$s]
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

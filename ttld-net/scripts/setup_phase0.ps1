# Phase 0 setup for Windows dev machine (no GPU / MMCV)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$Root\ttld-net")) { $Root = Split-Path -Parent $PSScriptRoot }
Set-Location "$Root\ttld-net"

Write-Host "=== TTLD-Net Phase 0 Setup (Windows dev) ===" -ForegroundColor Cyan

Write-Host "--- T0.2 Installing core dependencies ---"
pip install -r requirements.txt
pip install pytest setuptools

Write-Host "--- T0.5 Dataset verification ---"
python scripts/verify_dataset.py

Write-Host "--- Phase 0 gate (dev mode — MMCV/CUDA optional) ---"
python test_env.py --dev | Tee-Object -FilePath logs/env_check.txt
Write-Host "Log saved to ttld-net/logs/env_check.txt"
Write-Host ""
Write-Host "NOTE: MMCV + CUDA must be verified on GPU server:" -ForegroundColor Yellow
Write-Host "  bash scripts/setup_phase0.sh" -ForegroundColor Yellow

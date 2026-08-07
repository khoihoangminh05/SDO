# Pack TTLD-Net code for Google Colab (no images).
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File ttld-net\scripts\pack_colab.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$OutZip = Join-Path $RepoRoot "ttld_colab_code.zip"
$PyScript = Join-Path $PSScriptRoot "pack_colab.py"

Write-Host "Repo: $RepoRoot"
Write-Host "Out : $OutZip"

python $PyScript --repo $RepoRoot --out $OutZip
Write-Host "Upload to Google Drive: My Drive/SDO_train/ttld_colab_code.zip"

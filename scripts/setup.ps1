<#
.SYNOPSIS
    Create the ARC Director environment and verify it before the launch button.

.PARAMETER CudaWheel
    PyTorch wheel channel. cu128 supports current NVIDIA GPUs; use cpu on a
    machine without CUDA when you only want tests and smoke runs.
#>
[CmdletBinding()]
param([string]$CudaWheel = "cu128")

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo ".venv"
$pythonExe = Join-Path $venv "Scripts\python.exe"

function Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }

if (-not (Test-Path $pythonExe)) {
    Step "Creating Python 3.11 environment at $venv"
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.11 $venv
    } else {
        py -3.11 -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create Python 3.11 environment." }
}

Step "Installing CUDA-enabled PyTorch ($CudaWheel)"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install --python $pythonExe "torch>=2.1" --index-url "https://download.pytorch.org/whl/$CudaWheel"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
    uv pip install --python $pythonExe -e "$repo[dev]"
} else {
    & $pythonExe -m pip install "torch>=2.1" --index-url "https://download.pytorch.org/whl/$CudaWheel"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
    & $pythonExe -m pip install -e "$repo[dev]"
}
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }

Step "Checking PyTorch and the GPU"
& $pythonExe -c @"
import torch
print('torch          ', torch.__version__)
print('cuda available ', torch.cuda.is_available())
print('device         ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
if '$CudaWheel' != 'cpu':
    assert torch.cuda.is_available(), 'CUDA PyTorch was installed but no CUDA GPU is available'
"@
if ($LASTEXITCODE -ne 0) { throw "PyTorch verification failed." }

Step "Running the correctness suite"
$testTemp = Join-Path $repo ".test-tmp"
New-Item -ItemType Directory -Force $testTemp | Out-Null
$oldTemp = $env:TEMP
$oldTmp = $env:TMP
try {
    $env:TEMP = $testTemp
    $env:TMP = $testTemp
    & $pythonExe -m pytest -q -p no:cacheprovider
    $testsPassed = $LASTEXITCODE -eq 0
} finally {
    $env:TEMP = $oldTemp
    $env:TMP = $oldTmp
}
if (-not $testsPassed) { throw "Tests failed." }

Step "Ready"
Write-Host @"
Open Run and Debug in VS Code and choose:
  ARC Director - START (generated programs -> ARC-1 + ARC-2)
The live dashboard opens at http://127.0.0.1:8321/.
"@

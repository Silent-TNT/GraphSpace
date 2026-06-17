$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Venv = Join-Path $Root ".venv-v5-cuda"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $Venv
}

& $Python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
& $Python -m pip install numpy

& $Python -c @"
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the new environment")
print("gpu", torch.cuda.get_device_name(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
"@

& $Python (Join-Path $Root "scripts\train_v5\train.py") `
    --smoke-test `
    --device cuda `
    --output-dir (Join-Path $Root "outputs\v5_minimal_cuda_smoke")

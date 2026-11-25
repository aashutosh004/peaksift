# Activate PeakSift's project-local environment in the current PowerShell session.
#   . .\scripts\env.ps1
# Everything (venv, ffmpeg, model caches, app data) lives inside this project
# folder, so deleting the folder removes it all.
$Root = Split-Path -Parent $PSScriptRoot
$Local = Join-Path $Root ".local"
$Cache = Join-Path $Local "cache"

$env:PEAKSIFT_HOME = Join-Path $Local "data"
$env:PEAKSIFT_NO_KEYRING = "1"
$env:HF_HOME = Join-Path $Cache "huggingface"
$env:TORCH_HOME = Join-Path $Cache "torch"
$env:XDG_CACHE_HOME = $Cache
$env:UV_CACHE_DIR = Join-Path $Cache "uv"
$env:PIP_CACHE_DIR = Join-Path $Cache "pip"
$env:UV_PYTHON_DOWNLOADS = "never"

$tools = @((Join-Path $Local "tools\ffmpeg\bin"), (Join-Path $Root ".venv\Scripts"))
foreach ($t in $tools) { if ((Test-Path $t) -and -not ($env:PATH -split ';' -contains $t)) { $env:PATH = "$t;$env:PATH" } }

Write-Host "PeakSift project environment active ($Root)"

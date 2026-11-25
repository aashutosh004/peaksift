# PeakSift setup for Windows. Run it from Setup-PeakSift.cmd (double-click), or:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Everything goes inside this folder:
#   .venv\           Python packages
#   .local\tools\    ffmpeg
#   .local\cache\    downloaded models and package caches
#   .local\data\     your videos, clips, settings and API keys
# Delete the folder and PeakSift is gone. Nothing is installed system-wide.
#
# Options:
#   -Cpu        don't install NVIDIA GPU support even if a GPU is found
#   -SkipFfmpeg use the ffmpeg already on PATH

param([switch]$Cpu, [switch]$SkipFfmpeg)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Local = Join-Path $Root ".local"
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host ""; Write-Host $text -ForegroundColor Red; exit 1 }

# Creating .local first switches PeakSift into project-local mode (see
# backend/peaksift/portable.py) and gives the caches below somewhere to go.
New-Item -ItemType Directory -Force (Join-Path $Local "tools"), (Join-Path $Local "cache"), (Join-Path $Local "data") | Out-Null
. (Join-Path $PSScriptRoot "env.ps1") | Out-Null

# --------------------------------------------------------------------------
Step "Looking for Python 3.11 or 3.12"
$python = $null
foreach ($candidate in @(@("py", "-3.12"), @("py", "-3.11"), @("python"), @("python3"))) {
    $exe = $candidate[0]
    $rest = @($candidate | Select-Object -Skip 1)
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    try { $version = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null } catch { continue }
    if ($version -in @("3.11", "3.12")) { $python = $candidate; Write-Host "Found Python $version ($($candidate -join ' '))"; break }
}
if (-not $python) {
    Fail @"
PeakSift needs Python 3.11 or 3.12 (3.13 isn't supported yet: MediaPipe has no
packages for it). Install Python 3.12 from https://www.python.org/downloads/
(tick "Add python.exe to PATH"), or run:  winget install Python.Python.3.12
Then run this setup again.
"@
}

# --------------------------------------------------------------------------
if (-not (Test-Path $Py)) {
    Step "Creating the project environment (.venv)"
    $exe = $python[0]; $rest = @($python | Select-Object -Skip 1)
    & $exe @rest -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create .venv." }
}

Step "Installing uv (fast installer) into .venv"
& $Py -m pip install --quiet --upgrade pip uv
if ($LASTEXITCODE -ne 0) { Fail "Could not install uv into .venv." }
$Uv = Join-Path $Venv "Scripts\uv.exe"

$extras = "dev"
$hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasNvidia -and -not $Cpu) { $extras = "gpu,dev"; Write-Host "NVIDIA GPU found: installing GPU support." }
Step "Installing PeakSift and its packages (first time: a few minutes)"
# Native packages are pinned to versions verified with Smart App Control on;
# see the comment at the top of scripts\windows-constraints.txt.
$constraints = Join-Path $PSScriptRoot "windows-constraints.txt"
& $Uv pip install --python $Py -e ".[$extras]" -c $constraints
if ($LASTEXITCODE -ne 0) { Fail "Package install failed. Scroll up for the error." }

Step "Checking that Windows lets the native libraries load"
# Smart App Control (Windows 11) judges each unsigned library the first time
# it loads, using a cloud reputation check. That verdict is occasionally
# "block" for a file it allows elsewhere (the same file loaded fine in another
# copy), and the verdict sticks to that copy. A freshly reinstalled copy gets
# checked again, so blocked packages are reinstalled and re-checked twice.
$packagesFor = @{
    "ctranslate2" = @("ctranslate2"); "cv2" = @("opencv-python", "opencv-contrib-python");
    "av" = @("av"); "onnxruntime" = @("onnxruntime"); "numpy" = @("numpy")
}
$probe = "import importlib, sys`nbad = []`nfor m in ('ctranslate2', 'cv2', 'av', 'onnxruntime', 'numpy'):`n    try: importlib.import_module(m)`n    except Exception as e: bad.append(m + ' | ' + str(e))`nprint('\n'.join(bad)); sys.exit(1 if bad else 0)"
for ($attempt = 0; $attempt -le 2; $attempt++) {
    $failures = @(& $Py -c $probe | Where-Object { $_ })
    if ($LASTEXITCODE -eq 0) { break }
    $blockedOnly = -not ($failures | Where-Object { $_ -notmatch "Application Control" })
    if (-not $blockedOnly -or $attempt -eq 2) { break }
    $reinstall = @($failures | ForEach-Object { $packagesFor[($_ -split ' \| ')[0]] } | Select-Object -Unique)
    Write-Host "Windows blocked: $($reinstall -join ', '). Reinstalling for a fresh check..." -ForegroundColor Yellow
    & $Uv pip install --python $Py --reinstall --no-deps -c $constraints @reinstall | Out-Null
}
if ($LASTEXITCODE -ne 0) {
    $failures | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    if ("$failures" -match "Application Control") {
        Fail @"
Windows Smart App Control keeps blocking a library PeakSift needs (above).
Run this setup again later (the check is repeated each time), and if it
persists, open an issue on PeakSift's GitHub page with the lines above.
"@
    }
    Fail "A required library failed to load (above). See docs\TROUBLESHOOTING.md."
}
Write-Host "All native libraries load."

# --------------------------------------------------------------------------
function Test-Ffmpeg($exe) {
    try { $filters = & $exe -hide_banner -filters 2>$null } catch { return $false }
    return [bool]($filters | Select-String -Pattern '\sass\s' -Quiet)
}

$ffmpegDir = Join-Path $Local "tools\ffmpeg"
$ffmpegExe = Join-Path $ffmpegDir "bin\ffmpeg.exe"
if ($SkipFfmpeg) {
    Step "Skipping ffmpeg download (-SkipFfmpeg)"
} elseif (Test-Path $ffmpegExe) {
    Step "ffmpeg already in .local\tools"
} else {
    Step "Downloading ffmpeg into .local\tools (about 260 MB, once)"
    $headers = @{ "User-Agent" = "peaksift-setup" }
    $release = Invoke-RestMethod -Headers $headers "https://api.github.com/repos/GyanD/codexffmpeg/releases/latest"
    $asset = $release.assets | Where-Object { $_.name -like "*-full_build.zip" } | Select-Object -First 1
    if (-not $asset) { Fail "Could not find an ffmpeg download. Install ffmpeg yourself and re-run with -SkipFfmpeg." }
    $zip = Join-Path $Local "cache\$($asset.name)"
    $ProgressPreference = "SilentlyContinue"   # the progress bar makes downloads 10x slower
    Invoke-WebRequest -Headers $headers -Uri $asset.browser_download_url -OutFile $zip
    $unpack = Join-Path $Local "cache\ffmpeg-unpack"
    if (Test-Path $unpack) { Remove-Item -Recurse -Force $unpack }
    Expand-Archive -Path $zip -DestinationPath $unpack
    $inner = Get-ChildItem $unpack -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Force $ffmpegDir | Out-Null
    Copy-Item -Recurse -Force (Join-Path $inner.FullName "bin") $ffmpegDir
    Remove-Item -Recurse -Force $unpack, $zip
    Write-Host "ffmpeg $($release.tag_name) installed."
}
$ffmpegToCheck = if (Test-Path $ffmpegExe) { $ffmpegExe } else { "ffmpeg" }
if (-not (Test-Ffmpeg $ffmpegToCheck)) {
    Write-Host "Warning: this ffmpeg can't burn captions (no 'ass' filter). See docs\TROUBLESHOOTING.md." -ForegroundColor Yellow
}

# --------------------------------------------------------------------------
Step "Checking everything"
& $Py -m peaksift init | Out-Null
& $Py -m peaksift doctor

Write-Host ""
Write-Host "PeakSift is ready." -ForegroundColor Green
Write-Host "  Web app:      double-click Start-PeakSift.cmd  (opens http://localhost:8000)"
Write-Host "  Command line: .\peaksift clip `"<video link or file>`""
Write-Host "  Optional:     install Ollama (ollama.com) and run  ollama pull qwen3.5:9b"
Write-Host "                for AI-written titles, then pick it in Settings."

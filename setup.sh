#!/usr/bin/env bash
# PeakSift setup for macOS and Linux:  ./setup.sh   (add --cpu to skip GPU support)
#
# Everything goes inside this folder:
#   .venv/           Python packages
#   .local/cache/    downloaded models and package caches
#   .local/data/     your videos, clips, settings and API keys
# Delete the folder and PeakSift is gone. ffmpeg is the one thing that comes
# from your system (Homebrew / apt), because it needs system libraries.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
CPU=0
for arg in "$@"; do [ "$arg" = "--cpu" ] && CPU=1; done

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$1"; exit 1; }

# Creating .local switches PeakSift into project-local mode.
mkdir -p .local/cache .local/data
# shellcheck source=scripts/env.sh
source scripts/env.sh

step "Looking for Python 3.11 or 3.12"
PYTHON=""
for candidate in python3.12 python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$version" = "3.12" ] || [ "$version" = "3.11" ]; then PYTHON="$candidate"; break; fi
  fi
done
[ -n "$PYTHON" ] || fail "PeakSift needs Python 3.11 or 3.12 (3.13 isn't supported yet).
  macOS:  brew install python@3.12
  Ubuntu: sudo apt install python3.12 python3.12-venv
Then run ./setup.sh again."
echo "Found $PYTHON ($version)"

if [ ! -x .venv/bin/python ]; then
  step "Creating the project environment (.venv)"
  "$PYTHON" -m venv .venv
fi

step "Installing uv (fast installer) into .venv"
.venv/bin/python -m pip install --quiet --upgrade pip uv

EXTRAS="dev"
if [ "$CPU" = 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  EXTRAS="gpu,dev"; echo "NVIDIA GPU found: installing GPU support."
fi
step "Installing PeakSift and its packages (first time: a few minutes)"
.venv/bin/uv pip install --python .venv/bin/python -e ".[$EXTRAS]"

step "Checking ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  fail "ffmpeg is not installed.
  macOS:  brew install ffmpeg-full && brew link --force --overwrite ffmpeg-full
          (the plain 'ffmpeg' formula can't burn captions)
  Ubuntu: sudo apt install ffmpeg
Then run ./setup.sh again."
fi
filters="$(ffmpeg -hide_banner -filters 2>/dev/null || true)"
if ! printf '%s\n' "$filters" | grep -qE '[[:space:]]ass[[:space:]]'; then
  printf '\033[33mWarning: this ffmpeg has no "ass" filter, so it cannot burn captions.\nSee docs/TROUBLESHOOTING.md.\033[0m\n'
fi

step "Checking everything"
.venv/bin/python -m peaksift init >/dev/null
.venv/bin/python -m peaksift doctor || true

printf '\n\033[32mPeakSift is ready.\033[0m\n'
echo "  Web app:      ./start.sh   (opens http://localhost:8000)"
echo "  Command line: ./peaksift.sh clip \"<video link or file>\""
echo "  Optional:     install Ollama (ollama.com), run  ollama pull qwen3.5:9b"
echo "                for AI-written titles, then pick it in Settings."

# Activate PeakSift's project-local environment in the current shell:
#   source scripts/env.sh
# Everything (venv, ffmpeg, model caches, app data) lives inside this project
# folder, so deleting the folder removes it all.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
LOCAL="$ROOT/.local"; CACHE="$LOCAL/cache"
export PEAKSIFT_HOME="$LOCAL/data" PEAKSIFT_NO_KEYRING=1 HF_HOME="$CACHE/huggingface" \
  TORCH_HOME="$CACHE/torch" XDG_CACHE_HOME="$CACHE" UV_CACHE_DIR="$CACHE/uv" \
  PIP_CACHE_DIR="$CACHE/pip" UV_PYTHON_DOWNLOADS=never
for t in "$LOCAL/tools/ffmpeg/bin" "$ROOT/.venv/Scripts" "$ROOT/.venv/bin"; do
  [ -d "$t" ] && case ":$PATH:" in *":$t:"*) ;; *) PATH="$t:$PATH";; esac
done
export PATH

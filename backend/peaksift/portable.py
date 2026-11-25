"""Project-local ("portable") mode.

When PeakSift runs from a source checkout that has a ``.local/`` directory next
to ``pyproject.toml``, everything it would otherwise scatter across the system
is redirected into that directory instead:

    .local/
      data/            PeakSift home: media, work, exports, models, db, config
      tools/ffmpeg/    portable ffmpeg + ffprobe (bin/ is put first on PATH)
      cache/           HuggingFace, torch, yt-dlp, uv and pip caches

Secrets are kept in ``.local/data/config.json`` rather than the OS keyring, so
deleting the project folder removes every trace of it from the machine.

This runs once, at package import, before anything that reads these variables
(huggingface_hub, yt-dlp, the keyring) gets imported. Explicitly set
environment variables always win, so a user can still point any one of them
elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The checkout root, when running from source with portable mode enabled.
PROJECT_ROOT: Path | None = None


def _find_project_root() -> Path | None:
    # backend/peaksift/portable.py -> parents[2] is the checkout root.
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / ".local").is_dir():
        return candidate
    return None


def local_dir() -> Path | None:
    return PROJECT_ROOT / ".local" if PROJECT_ROOT else None


def apply() -> None:
    """Point caches, data and tools at ``.local/`` if this is a portable checkout."""
    global PROJECT_ROOT
    if os.environ.get("PEAKSIFT_PORTABLE", "").lower() in ("0", "false", "no"):
        return

    root = _find_project_root()
    if root is None:
        return
    PROJECT_ROOT = root
    local = root / ".local"
    cache = local / "cache"

    defaults = {
        "PEAKSIFT_HOME": local / "data",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "XDG_CACHE_HOME": cache,
        "UV_CACHE_DIR": cache / "uv",
        "PIP_CACHE_DIR": cache / "pip",
        "MPLCONFIGDIR": cache / "matplotlib",
        "YOLO_CONFIG_DIR": cache / "ultralytics",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))

    # Windows without Developer Mode can't create symlinks; the HF cache works
    # without them, so silence the warning it prints on every model load.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # MediaPipe/TFLite print several INFO/WARNING lines per model load to stderr.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Keep secrets out of the OS credential store; see config._keyring.
    os.environ.setdefault("PEAKSIFT_NO_KEYRING", "1")

    tool_dirs = [local / "tools" / "ffmpeg" / "bin"]
    existing = os.environ.get("PATH", "")
    parts = existing.split(os.pathsep) if existing else []
    for tool_dir in reversed(tool_dirs):
        if tool_dir.is_dir() and str(tool_dir) not in parts:
            parts.insert(0, str(tool_dir))
    os.environ["PATH"] = os.pathsep.join(parts)

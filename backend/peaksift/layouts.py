"""Stacked-layout settings: where the facecam and the gameplay are, and how to stack them.

The gaming layout splits the 9:16 frame into a facecam panel and a gameplay
panel. Three things decide how it looks, in priority order:

1. **A layout saved for this job** (``work/<job>/layout.json``) — set with the
   picker after a run, to fix one video.
2. **A preset saved for the channel** (``<home>/layouts.json``) — set once for a
   streamer whose overlay never moves; every later VOD and live stream of that
   channel reuses it.
3. **Auto-detection** — the facecam found by the visual signal pass, with the
   gameplay taken as the centre of the frame.

Boxes are normalised ``(x, y, w, h)`` in the source frame, so a preset made on a
1080p VOD works on the 720p live recording of the same channel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from . import paths

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]
Position = Literal["top", "bottom"]

MIN_SPLIT = 0.25
MAX_SPLIT = 0.70


@dataclass
class StackLayout:
    facecam: Box
    #: None means "auto": a full-height centre band of the frame.
    gameplay: Box | None = None
    #: Share of the output height given to the facecam panel.
    split: float = 0.35
    facecam_position: Position = "top"
    #: How the boxes were chosen: "manual" (picker), "preset" or "auto".
    origin: str = "auto"

    def __post_init__(self) -> None:
        self.facecam = clamp_box(self.facecam)
        if self.gameplay is not None:
            self.gameplay = clamp_box(self.gameplay)
        self.split = max(MIN_SPLIT, min(MAX_SPLIT, float(self.split)))
        if self.facecam_position not in ("top", "bottom"):
            self.facecam_position = "top"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["facecam"] = list(self.facecam)
        data["gameplay"] = list(self.gameplay) if self.gameplay else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, origin: str | None = None) -> StackLayout:
        gameplay = data.get("gameplay")
        return cls(
            facecam=tuple(data["facecam"]),  # type: ignore[arg-type]
            gameplay=tuple(gameplay) if gameplay else None,  # type: ignore[arg-type]
            split=float(data.get("split", 0.35)),
            facecam_position=data.get("facecam_position", "top"),
            origin=origin or data.get("origin", "manual"),
        )


def clamp_box(box: Any) -> Box:
    x, y, w, h = (float(v) for v in box)
    w = max(0.02, min(1.0, w))
    h = max(0.02, min(1.0, h))
    x = max(0.0, min(1.0 - w, x))
    y = max(0.0, min(1.0 - h, y))
    return (x, y, w, h)


# --------------------------------------------------------------------------
# Keys and storage
# --------------------------------------------------------------------------


def preset_key(platform: str | None, channel: str | None) -> str | None:
    """``twitch:somechannel`` style key, or None when the source has no channel."""
    if not platform or not channel:
        return None
    return f"{platform.strip().lower()}:{channel.strip().lower()}"


def presets_path() -> Path:
    return paths.root() / "layouts.json"


def job_layout_path(job_id: str) -> Path:
    return paths.job_work_dir(job_id) / "layout.json"


def _read_presets() -> dict[str, Any]:
    path = presets_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("presets", {})
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return {}


def load_preset(key: str | None) -> StackLayout | None:
    if not key:
        return None
    data = _read_presets().get(key)
    return StackLayout.from_dict(data, origin="preset") if data else None


def save_preset(key: str, layout: StackLayout) -> Path:
    presets = _read_presets()
    presets[key] = {**layout.to_dict(), "origin": "manual"}
    path = presets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"presets": presets}, indent=2), encoding="utf-8")
    return path


def list_presets() -> dict[str, StackLayout]:
    return {k: StackLayout.from_dict(v, origin="preset") for k, v in _read_presets().items()}


def delete_preset(key: str) -> bool:
    presets = _read_presets()
    if presets.pop(key, None) is None:
        return False
    presets_path().write_text(json.dumps({"presets": presets}, indent=2), encoding="utf-8")
    return True


def load_job_layout(job_id: str) -> StackLayout | None:
    path = job_layout_path(job_id)
    if not path.exists():
        return None
    try:
        return StackLayout.from_dict(json.loads(path.read_text(encoding="utf-8")), origin="manual")
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        log.warning("Ignoring unreadable %s: %s", path, exc)
        return None


def save_job_layout(job_id: str, layout: StackLayout) -> Path:
    path = job_layout_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**layout.to_dict(), "origin": "manual"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def resolve(
    *,
    job_id: str | None,
    platform: str | None,
    channel: str | None,
    timeline_meta: dict[str, Any] | None,
    split: float = 0.35,
    facecam_position: Position = "top",
) -> StackLayout | None:
    """The stacked layout to use: job override, channel preset, or auto-detected."""
    if job_id:
        manual = load_job_layout(job_id)
        if manual is not None:
            return manual
    preset = load_preset(preset_key(platform, channel))
    if preset is not None:
        return preset
    detected = (timeline_meta or {}).get("layout") or {}
    if detected.get("kind") == "facecam" and detected.get("facecam"):
        return StackLayout(
            facecam=tuple(detected["facecam"]),  # type: ignore[arg-type]
            split=split,
            facecam_position=facecam_position,
            origin="auto",
        )
    return None

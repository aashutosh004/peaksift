"""Facecam/gameplay layout picker: frames, context, and saving the user's boxes."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .. import layouts, paths
from ..config import load as load_settings
from ..db import store
from ..pipeline import ffmpeg

router = APIRouter(tags=["layouts"])

PICKER_HTML = Path(__file__).resolve().parent.parent / "picker" / "picker.html"

#: Set whenever a layout is saved; the `peaksift layout` command waits on it.
saved_event = threading.Event()


class LayoutIn(BaseModel):
    facecam: list[float] = Field(min_length=4, max_length=4)
    gameplay: list[float] | None = Field(default=None, min_length=4, max_length=4)
    split: float = Field(default=0.35, ge=0.25, le=0.70)
    facecam_position: str = "top"
    save_for_channel: bool = True


# --------------------------------------------------------------------------
# Media lookup (finished files and live recordings)
# --------------------------------------------------------------------------


def media_at(job_id: str, t: float) -> tuple[Path, float, float]:
    """``(file, time inside that file, total duration)`` for a job's source at ``t``."""
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    source = store.get_source(job.source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    if source.type != "live":
        return (
            Path(source.path),
            max(0.0, min(t, max(0.0, source.duration_s - 0.5))),
            source.duration_s,
        )

    from ..live.recorder import SegmentRecorder

    segments = SegmentRecorder("", Path(source.path).parent / "segments", origin_epoch=0).segments()
    if not segments:
        raise HTTPException(status_code=404, detail="No recorded video for this live job yet.")
    total = segments[-1].end_s
    chosen = min(segments, key=lambda s: 0 if s.start_s <= t < s.end_s else abs(s.start_s - t))
    return chosen.path, max(0.0, min(t - chosen.start_s, chosen.duration_s - 0.2)), total


def extract_frame(media: Path, t: float, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(media),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return destination


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.get("/layout-picker", include_in_schema=False)
async def picker_page() -> FileResponse:
    return FileResponse(PICKER_HTML, media_type="text/html")


@router.get("/api/layouts/jobs/{job_id}/context")
async def picker_context(job_id: str) -> dict:
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    source = await asyncio.to_thread(store.get_source, job.source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    media, _, duration = await asyncio.to_thread(media_at, job_id, 0.0)
    info = await asyncio.to_thread(ffmpeg.probe, media)

    meta_path = paths.job_work_dir(job_id) / "timeline.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    settings = load_settings()
    current = layouts.resolve(
        job_id=job_id,
        platform=source.platform,
        channel=source.channel,
        timeline_meta=meta,
        split=settings.export.split_ratio,
        facecam_position=settings.export.facecam_position,
    )
    detected = (meta.get("layout") or {}).get("facecam")
    clips = await asyncio.to_thread(store.list_clips, job_id)
    peak = next((c.peak_s for c in clips if c.peak_s is not None), None)
    return {
        "job_id": job_id,
        "title": source.title,
        "duration_s": duration,
        "width": info.width,
        "height": info.height,
        "preset_key": layouts.preset_key(source.platform, source.channel),
        "layout": current.to_dict() if current else None,
        "detected_facecam": detected,
        "suggested_t": peak if peak is not None else duration / 3,
        "defaults": {
            "split": settings.export.split_ratio,
            "facecam_position": settings.export.facecam_position,
        },
    }


@router.get("/api/layouts/jobs/{job_id}/frame")
async def picker_frame(job_id: str, t: float = Query(0.0, ge=0.0)) -> Response:
    media, local_t, _ = await asyncio.to_thread(media_at, job_id, t)
    target = paths.job_work_dir(job_id) / "picker" / f"frame_{int(t * 10)}.jpg"
    try:
        await asyncio.to_thread(extract_frame, media, local_t, target)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail="Could not extract a frame.") from exc
    return FileResponse(target, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.put("/api/layouts/jobs/{job_id}")
async def save_layout(job_id: str, payload: LayoutIn) -> dict:
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    source = await asyncio.to_thread(store.get_source, job.source_id)
    layout = layouts.StackLayout(
        facecam=tuple(payload.facecam),  # type: ignore[arg-type]
        gameplay=tuple(payload.gameplay) if payload.gameplay else None,  # type: ignore[arg-type]
        split=payload.split,
        facecam_position="bottom" if payload.facecam_position == "bottom" else "top",
        origin="manual",
    )
    saved = {"job": str(layouts.save_job_layout(job_id, layout))}
    key = layouts.preset_key(source.platform, source.channel) if source else None
    if payload.save_for_channel and key:
        layouts.save_preset(key, layout)
        saved["channel"] = key
    saved_event.set()
    return {"ok": True, "saved": saved, "layout": layout.to_dict()}

"""Endpoints behind the built-in web UI.

The REST API already covers sources, jobs and clips one call at a time. The UI
needs a few things on top:

* **Start from a link in one step** — download (which can take minutes for a
  long VOD) happens in a background task the page polls, then the job is queued.
  A link that is live right now starts a live session instead.
* **Live sessions** — start, watch, snapshot and stop ``peaksift live`` from the
  browser. Sessions run in background threads inside the server process.
* **Re-render** a job's clips after a layout change, and read its report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import paths, sources
from ..config import load as load_settings
from ..db import store
from ..db.models import Job, new_id
from ..jobs.queue import queue
from .jobs import _apply_overrides
from .schemas import JobSettingsIn

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ui", tags=["ui"])


# --------------------------------------------------------------------------
# Download-then-clip tasks
# --------------------------------------------------------------------------


@dataclass
class _Task:
    id: str
    url: str
    status: str = "checking"  # checking | downloading | queued | failed
    job_id: str | None = None
    live_id: str | None = None
    error: str = ""
    hint: str = ""
    progress: float = 0.0
    created: float = field(default_factory=time.time)


_tasks: dict[str, _Task] = {}


class StartIn(BaseModel):
    url: str
    settings: JobSettingsIn = Field(default_factory=JobSettingsIn)
    #: Live-only options, used when the link turns out to be live.
    chat_url: str | None = None
    backfill: bool = True
    max_hours: float | None = Field(default=None, gt=0)
    clips_per_hour: int | None = Field(default=None, ge=1, le=20)


@router.post("/start")
async def start(payload: StartIn) -> dict:
    """Clip a link: finished videos are downloaded and queued; live links go live."""
    url = payload.url.strip()
    if not sources.is_url(url) and url.lower() != "obs":
        raise HTTPException(status_code=400, detail="That is not a link.")
    task = _Task(id=uuid.uuid4().hex[:12], url=url)
    _tasks[task.id] = task
    threading.Thread(target=_run_task, args=(task, payload), daemon=True).start()
    return {"task_id": task.id}


@router.get("/tasks/{task_id}")
async def task_status(task_id: str) -> dict:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task.__dict__


def _run_task(task: _Task, payload: StartIn) -> None:
    from ..pipeline import ingest

    settings = _apply_overrides(load_settings(), payload.settings)
    try:
        if task.url.lower() == "obs" or _is_live(task.url, settings):
            task.live_id = _start_live(task.url, settings, payload)
            task.status = "live"
            return

        task.status = "downloading"

        def progress(fraction: float) -> None:
            task.progress = round(fraction, 3)

        source = ingest.ingest_url(task.url, settings.ingest, on_progress=progress)
        store.create_source(source)
        job = store.create_job(
            Job(
                id=new_id(),
                source_id=source.id,
                provider=settings.active_provider,
                settings=settings.model_dump(mode="json"),
            )
        )
        queue.notify()
        task.job_id = job.id
        task.status = "queued"
    except Exception as exc:  # surfaced to the page, not raised
        log.warning("Start task for %s failed: %s", task.url, exc)
        task.status = "failed"
        task.error = str(exc).split("\n\n")[0]
        task.hint = getattr(exc, "hint", "")


def _is_live(url: str, settings) -> bool:
    if not sources.supports_live(sources.platform_of(url)):
        return False
    try:
        return sources.probe(url, cookies_from_browser=settings.ingest.cookies_from_browser).is_live
    except Exception:
        return False


# --------------------------------------------------------------------------
# Live sessions
# --------------------------------------------------------------------------


@dataclass
class _Live:
    id: str
    url: str
    session: Any
    thread: threading.Thread | None = None
    error: str = ""
    finished: bool = False
    started: float = field(default_factory=time.time)


_live: dict[str, _Live] = {}


class LiveIn(BaseModel):
    url: str
    settings: JobSettingsIn = Field(default_factory=JobSettingsIn)
    chat_url: str | None = None
    backfill: bool = True
    max_hours: float | None = Field(default=None, gt=0)
    clips_per_hour: int | None = Field(default=None, ge=1, le=20)


def _start_live(url: str, settings, options: StartIn | LiveIn) -> str:
    from ..live.session import LiveSession

    if options.clips_per_hour:
        settings.live.clips_per_hour = options.clips_per_hour
        settings.selection.per_bucket = options.clips_per_hour
    session = LiveSession(
        url,
        settings,
        backfill=options.backfill,
        max_hours=options.max_hours,
        chat_url=options.chat_url or None,
        keyboard=False,
    )
    entry = _Live(id=uuid.uuid4().hex[:12], url=url, session=session)

    def run() -> None:
        try:
            session.run()
        except Exception as exc:
            log.warning("Live session %s ended with an error: %s", entry.id, exc)
            entry.error = str(exc)
        finally:
            entry.finished = True

    entry.thread = threading.Thread(target=run, name=f"live-{entry.id}", daemon=True)
    _live[entry.id] = entry
    entry.thread.start()
    return entry.id


@router.post("/live")
async def live_start(payload: LiveIn) -> dict:
    url = payload.url.strip()
    if url.lower() != "obs" and not url.lower().startswith("rtmp://") and not sources.is_url(url):
        raise HTTPException(status_code=400, detail="Give a live link, or 'obs'.")
    settings = _apply_overrides(load_settings(), payload.settings)
    live_id = await asyncio.to_thread(_start_live, url, settings, payload)
    return {"live_id": live_id}


def _live_out(entry: _Live) -> dict:
    session = entry.session
    status = session.status
    job = session.job
    return {
        "id": entry.id,
        "url": entry.url,
        "obs": session.obs,
        "title": session.source.title if session.source else "",
        "platform": session.source.platform if session.source else "",
        "job_id": job.id if job else None,
        "running": not entry.finished,
        "error": entry.error,
        "recorded_s": status.recorded_s,
        "processed_s": status.processed_s,
        "backfill_total_s": status.backfill_total_s,
        "backfill_done_s": status.backfill_done_s,
        "clips": status.clips,
        "chat_messages": status.chat_messages,
        "events": status.events[-60:],
        "started": entry.started,
    }


@router.get("/live")
async def live_list() -> list[dict]:
    return [_live_out(e) for e in sorted(_live.values(), key=lambda e: -e.started)]


@router.get("/live/{live_id}")
async def live_status(live_id: str) -> dict:
    entry = _live.get(live_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Live session not found.")
    return _live_out(entry)


class SnapshotIn(BaseModel):
    seconds: float = Field(default=30, ge=5, le=180)


@router.post("/live/{live_id}/snapshot")
async def live_snapshot(live_id: str, payload: SnapshotIn) -> dict:
    entry = _live.get(live_id)
    if entry is None or entry.finished:
        raise HTTPException(status_code=409, detail="That live session is not running.")
    entry.session.request_snapshot(payload.seconds)
    return {"ok": True}


@router.post("/live/{live_id}/stop")
async def live_stop(live_id: str) -> dict:
    entry = _live.get(live_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Live session not found.")
    entry.session.stop()
    return {"ok": True}


# --------------------------------------------------------------------------
# Re-render and report
# --------------------------------------------------------------------------


class RerenderIn(BaseModel):
    ranks: list[int] | None = None
    settings: JobSettingsIn = Field(default_factory=JobSettingsIn)


_rerendering: set[str] = set()


@router.post("/jobs/{job_id}/rerender")
async def rerender(job_id: str, payload: RerenderIn) -> dict:
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in ("queued", "running") or job_id in _rerendering:
        raise HTTPException(status_code=409, detail="This job is still running.")
    from ..jobs.queue import _job_settings

    # Start from the settings the job was made with, then apply any changes.
    settings = _apply_overrides(_job_settings(job), payload.settings)

    def run() -> None:
        from ..rerender import rerender_job

        _rerendering.add(job_id)
        try:
            store.update_job(
                job_id, status="running", current_stage="rerender", progress=0.0, error=None
            )
            rerender_job(job_id, settings, ranks=payload.ranks)
            store.update_job(job_id, status="done", progress=1.0, error=None)
        except Exception as exc:
            log.warning("Re-render of %s failed: %s", job_id, exc)
            # The clips are still there; only this re-render didn't happen.
            store.update_job(job_id, status="done", error=f"Re-render failed: {exc}")
        finally:
            _rerendering.discard(job_id)

    threading.Thread(target=run, name=f"rerender-{job_id}", daemon=True).start()
    return {"ok": True}


@router.get("/jobs/{job_id}/report")
async def report(job_id: str) -> dict:
    path = paths.exports_dir() / job_id / "clips.json"
    if not path.exists():
        return {"clips": []}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Moments strip
# --------------------------------------------------------------------------

#: Points in the activity curve sent to the browser, whatever the video length.
MOMENT_POINTS = 480


@router.get("/jobs/{job_id}/moments")
async def moments(job_id: str) -> dict:
    """The whole video's moment score, downsampled, with where each clip sits.

    The score is recomputed from the saved signals (milliseconds), so it needs
    no extra storage and always matches the job's profile and chat delay.
    """
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    def build() -> dict:
        import numpy as np

        from ..jobs.queue import _job_settings
        from ..signals.fusion import fuse
        from ..signals.timeline import Timeline

        source = store.get_source(job.source_id)
        path = paths.job_work_dir(job_id) / "timeline.npz"
        points: list[float] = []
        length = 0
        if path.exists():
            timeline = Timeline.load(path)
            length = timeline.length
            if length:
                settings = _job_settings(job)
                score = fuse(
                    timeline,
                    profile=timeline.meta.get("profile") or settings.detection.profile,
                    weights=settings.detection.weights or None,
                    chat_lag_s=timeline.meta.get("chat_lag_s"),
                ).score
                buckets = np.array_split(np.maximum(score, 0), min(MOMENT_POINTS, length))
                peaks = np.array([float(b.max()) if b.size else 0.0 for b in buckets])
                top = float(np.percentile(peaks, 98)) or 1.0
                points = [round(min(1.0, p / top), 3) for p in peaks]
        clips = sorted(store.list_clips(job_id), key=lambda c: c.rank)
        duration = (
            (source.duration_s if source else 0)
            or length
            or (max((c.end_s for c in clips), default=0.0))
        )
        return {
            "duration_s": duration,
            "points": points,
            "clips": [
                {
                    "id": c.id,
                    "rank": c.rank,
                    "start_s": c.start_s,
                    "end_s": c.end_s,
                    "peak_s": c.peak_s,
                    "title": c.title,
                }
                for c in clips
            ],
        }

    return await asyncio.to_thread(build)

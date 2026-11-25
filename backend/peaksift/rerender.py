"""Re-render a finished job's clips — e.g. after changing the facecam/gameplay layout.

Nothing is re-detected: the same clips (same start/end, titles, captions) are
cropped and exported again from the original media, so a layout fix costs only
render time and never loses quality the way re-cropping a finished 9:16 file
would.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from . import paths
from .config import Settings
from .db import store
from .db.models import Clip, Export, new_id
from .pipeline.report import write_report
from .pipeline.transcript import Transcript
from .signals.timeline import Timeline

log = logging.getLogger(__name__)


class RerenderError(RuntimeError):
    pass


def _remove(path: Path, *, attempts: int = 20, wait_s: float = 0.25) -> None:
    """Delete an old export, waiting briefly if something still has it open.

    Windows refuses to delete a file another program is reading — a browser
    tab still streaming the clip, or a video player. A tab that was just told
    to let go needs a moment; anything longer gets a clear message.
    """
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(wait_s)
    raise RerenderError(
        f"{path.name} is open in another program (a video player or a browser tab). "
        "Close it, then re-render again."
    )


def rerender_job(
    job_id: str,
    settings: Settings,
    *,
    ranks: list[int] | None = None,
    on_progress: Callable[[int, int, Clip], None] | None = None,
) -> list[str]:
    """Re-export a job's clips (all, or the given ranks). Returns the new file paths."""
    job = store.get_job(job_id)
    if job is None:
        raise RerenderError(f"Job {job_id} not found.")
    source = store.get_source(job.source_id)
    if source is None:
        raise RerenderError(f"Source for job {job_id} not found.")
    clips = sorted(store.list_clips(job_id), key=lambda c: c.rank)
    if ranks:
        clips = [c for c in clips if c.rank in set(ranks)]
    if not clips:
        raise RerenderError("No clips to re-render.")

    work = paths.job_work_dir(job_id)
    transcript = (
        Transcript.load(work / "transcript.json")
        if (work / "transcript.json").exists()
        else Transcript()
    )
    timeline = (
        Timeline.load(work / "timeline.npz") if (work / "timeline.npz").exists() else Timeline()
    )

    # Old files go away first so a changed title can't leave a stale duplicate.
    for clip in clips:
        for old in store.delete_exports(clip.id):
            for suffix in (".srt", ".vtt"):
                _remove(Path(old).with_suffix(suffix))
            _remove(Path(old))
        (work / "crops" / f"{clip.id}.json").unlink(missing_ok=True)

    if source.type == "live":
        files = _rerender_live(job, source, settings, clips, transcript, timeline, on_progress)
    else:
        files = _rerender_file(job, source, settings, clips, transcript, timeline, on_progress)

    all_clips = store.list_clips(job_id)
    write_report(
        paths.exports_dir() / job_id,
        source=source,
        clips=all_clips,
        view=settings.selection.view,
        meta=timeline.meta,
    )
    return files


def _rerender_file(job, source, settings, clips, transcript, timeline, on_progress) -> list[str]:
    from .pipeline.runner import PipelineRunner

    runner = PipelineRunner(job, source, settings=settings)
    runner.timeline = timeline if timeline.length else None
    crop_paths = runner._stage_reframe(clips, transcript)
    runner._stage_export(clips, transcript, crop_paths)
    files: list[str] = []
    for index, clip in enumerate(clips, start=1):
        for export in store.list_exports(clip.id):
            files.append(export.path)
        if on_progress:
            on_progress(index, len(clips), clip)
    store.update_job(job.id, status="done", progress=1.0)
    return files


def _rerender_live(job, source, settings, clips, transcript, timeline, on_progress) -> list[str]:
    from .live.recorder import SegmentRecorder
    from .live.session import LiveSession

    session = LiveSession(source.url or "obs", settings)
    session.job, session.source = job, source
    session.session_dir = Path(source.path).parent
    session.work_dir = paths.job_work_dir(job.id)
    session.export_dir = paths.exports_dir() / job.id
    session.timeline = timeline
    session.recorder = SegmentRecorder("", session.session_dir / "segments", origin_epoch=0)
    files: list[str] = []
    for index, clip in enumerate(clips, start=1):
        path = session._render(clip, transcript)
        if path:
            files.append(path)
            store.create_export(
                Export(
                    id=new_id(),
                    clip_id=clip.id,
                    path=path,
                    ratio=settings.export.ratio,
                    style=settings.export.caption_style,
                    size_bytes=Path(path).stat().st_size,
                )
            )
            store.update_clip(clip.id, status="exported")
        if on_progress:
            on_progress(index, len(clips), clip)
    return files

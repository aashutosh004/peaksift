"""Pipeline orchestration.

Runs a job stage by stage, writing each stage's artifacts into
``work/{job_id}/``. Because every stage's output is a file on disk, a retry
skips everything already done and resumes at the stage that failed — which
matters when stage two took eleven minutes and stage four hit a rate limit.

Progress is reported through a callback rather than written directly, so the
same runner serves the CLI (a progress bar) and the web API (an SSE stream).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..config import Settings
from ..config import load as load_settings
from ..db import store
from ..db.models import Clip, Export, Job, Source, new_id, utcnow
from ..db.models import Transcript as TranscriptRow
from ..providers import build_provider, detection_config
from ..signals import build_timeline
from ..signals.timeline import Timeline
from . import (
    Stage,
    captions,
    export,
    ffmpeg,
    highlights,
    prepare,
    signal_highlights,
    transcribe,
)
from .clip_words import words_for_clip
from .prepare import Silence
from .reframe.croppath import CropPath
from .transcript import Transcript

log = logging.getLogger(__name__)

#: Weight of each stage in the overall progress bar. Rough proportions of wall
#: time on a mid-range machine — transcription and export dominate.
STAGE_WEIGHTS: dict[Stage, float] = {
    Stage.PREPARE: 0.05,
    Stage.TRANSCRIBE: 0.28,
    Stage.SIGNALS: 0.17,
    Stage.HIGHLIGHTS: 0.08,
    Stage.REFRAME: 0.17,
    Stage.CAPTIONS: 0.02,
    Stage.EXPORT: 0.23,
}


class JobCancelled(RuntimeError):
    """The job was cancelled by the user."""


class PipelineError(RuntimeError):
    """A stage failed in a way the user should see."""

    def __init__(self, message: str, *, stage: Stage) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass
class ProgressEvent:
    stage: Stage
    #: Progress within the stage, 0..1.
    stage_progress: float
    #: Progress across the whole job, 0..1.
    overall: float
    message: str = ""


ProgressHandler = Callable[[ProgressEvent], None]


class JobWorkspace:
    """Paths for one job's intermediate artifacts."""

    def __init__(self, job_id: str) -> None:
        self.root = paths.job_work_dir(job_id)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.json"

    @property
    def silences(self) -> Path:
        return self.root / "silences.json"

    @property
    def thumbnails(self) -> Path:
        return self.root / "thumbnails"

    def crop_path(self, clip_id: str) -> Path:
        return self.root / "crops" / f"{clip_id}.json"

    @property
    def captions_dir(self) -> Path:
        return self.root / "captions"

    @property
    def timeline(self) -> Path:
        return self.root / "timeline.npz"


class PipelineRunner:
    """Executes one job."""

    def __init__(
        self,
        job: Job,
        source: Source,
        *,
        settings: Settings | None = None,
        on_progress: ProgressHandler | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.job = job
        self.source = source
        self.settings = settings or load_settings()
        self.on_progress = on_progress
        self._is_cancelled = is_cancelled or (lambda: False)
        self.workspace = JobWorkspace(job.id)
        self._completed_weight = 0.0
        self.timeline: Timeline | None = None

    # -- progress ----------------------------------------------------------

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise JobCancelled("Job cancelled.")

    def _emit(
        self,
        stage: Stage,
        stage_progress: float,
        message: str = "",
        *,
        overall: float | None = None,
    ) -> None:
        if overall is None:
            overall = self._completed_weight + STAGE_WEIGHTS[stage] * stage_progress
        overall = min(1.0, overall)
        store.update_job(self.job.id, current_stage=stage.value, progress=round(overall, 4))
        if self.on_progress:
            self.on_progress(
                ProgressEvent(
                    stage=stage,
                    stage_progress=stage_progress,
                    overall=overall,
                    message=message or stage.label,
                )
            )

    def _finish_stage(self, stage: Stage) -> None:
        """Mark a stage complete.

        ``overall`` is passed explicitly rather than derived. Deriving it after
        incrementing the accumulated weight counts the stage twice, so the bar
        jumped past the true total and then snapped backwards when the next
        stage reported 0 — visibly, at every stage boundary.
        """
        self._completed_weight += STAGE_WEIGHTS[stage]
        self._emit(stage, 1.0, overall=self._completed_weight)

    def _stage_progress(self, stage: Stage) -> Callable[[float], None]:
        def report(fraction: float) -> None:
            self._emit(stage, max(0.0, min(1.0, fraction)))

        return report

    # -- entry point -------------------------------------------------------

    async def run(self) -> list[Clip]:
        """Run the full pipeline and return the exported clips."""
        from ..jobs.owner import claim

        with claim(self.job.id):
            return await self._run()

    async def _run(self) -> list[Clip]:
        store.update_job(self.job.id, status="running", started_at=utcnow(), error=None)

        try:
            audio = self._stage_prepare()
            transcript = self._stage_transcribe(audio)
            silences = self._load_or_detect_silences(audio)
            if self.multimodal:
                self.timeline = self._stage_signals(audio, transcript)
            else:
                self._skip_stage(Stage.SIGNALS)
            clips = await self._stage_highlights(transcript, silences)
            crop_paths = self._stage_reframe(clips, transcript)
            self._stage_captions(clips, transcript)
            self._stage_export(clips, transcript, crop_paths)
            self._write_report(clips)
        except JobCancelled:
            store.update_job(self.job.id, status="cancelled", finished_at=utcnow(), progress=0.0)
            raise
        except Exception as exc:
            log.exception("Job %s failed.", self.job.id)
            store.update_job(self.job.id, status="failed", error=str(exc), finished_at=utcnow())
            raise

        store.update_job(self.job.id, status="done", progress=1.0, finished_at=utcnow())
        return clips

    @property
    def multimodal(self) -> bool:
        return self.settings.detection.mode == "multimodal"

    @property
    def source_dir(self) -> Path:
        return Path(self.source.path).parent

    def _skip_stage(self, stage: Stage) -> None:
        self._finish_stage(stage)

    # -- stages ------------------------------------------------------------

    def _stage_prepare(self) -> Path:
        stage = Stage.PREPARE
        self._check_cancelled()
        source_path = Path(self.source.path)

        if self.workspace.audio.exists() and self.workspace.audio.stat().st_size > 0:
            log.info("Reusing existing audio for job %s.", self.job.id)
        else:
            self._emit(stage, 0.0, "Extracting audio")
            prepare.extract_audio(
                source_path,
                self.workspace.audio,
                duration_s=self.source.duration_s,
                on_progress=self._stage_progress(stage),
            )

        if self.source.has_video and not self.workspace.thumbnails.exists():
            prepare.generate_thumbnails(source_path, self.workspace.thumbnails)

        self._finish_stage(stage)
        return self.workspace.audio

    def _stage_transcribe(self, audio: Path) -> Transcript:
        stage = Stage.TRANSCRIBE
        self._check_cancelled()

        if self.workspace.transcript.exists():
            log.info("Reusing existing transcript for job %s.", self.job.id)
            transcript = Transcript.load(self.workspace.transcript)
            self._finish_stage(stage)
            return transcript

        self._emit(stage, 0.0, "Transcribing")
        try:
            transcript = transcribe.transcribe(
                audio,
                self.settings.whisper,
                duration_s=self.source.duration_s,
                on_progress=self._stage_progress(stage),
                cancelled=self._is_cancelled,
            )
        except transcribe.NoSpeechError:
            if not self.multimodal:
                raise
            # Gameplay with no commentary: clip from audio, video and chat alone.
            log.info("No speech found; continuing with non-speech signals only.")
            transcript = Transcript(model=self.settings.whisper.model, source="whisper")
        finally:
            transcribe.release_models()

        if self.settings.whisper.diarization:
            from ..config import HF_TOKEN_KEY, get_secret

            self._emit(stage, 0.95, "Identifying speakers")
            transcribe.diarize(audio, transcript, hf_token=get_secret(HF_TOKEN_KEY, self.settings))

        transcript.save(self.workspace.transcript)
        store.upsert_transcript(
            TranscriptRow(
                job_id=self.job.id,
                json_path=str(self.workspace.transcript),
                language=transcript.language,
                model=transcript.model,
                has_diarization=transcript.has_diarization,
                word_count=len(transcript.words),
                source=transcript.source,
            )
        )

        self._finish_stage(stage)
        return transcript

    def _load_or_detect_silences(self, audio: Path) -> list[Silence]:
        if self.workspace.silences.exists():
            raw = json.loads(self.workspace.silences.read_text(encoding="utf-8"))
            return [Silence(**item) for item in raw]

        silences = prepare.detect_silences(audio)
        self.workspace.silences.write_text(
            json.dumps([{"start": s.start, "end": s.end} for s in silences]),
            encoding="utf-8",
        )
        return silences

    def _stage_signals(self, audio: Path, transcript: Transcript) -> Timeline:
        stage = Stage.SIGNALS
        self._check_cancelled()

        if self.workspace.timeline.exists():
            log.info("Reusing signal timeline for job %s.", self.job.id)
            timeline = Timeline.load(self.workspace.timeline)
            self._finish_stage(stage)
            return timeline

        det = self.settings.detection
        self._emit(stage, 0.0, stage.label)
        timeline = build_timeline(
            video=Path(self.source.path) if self.source.has_video else None,
            audio=audio,
            transcript=transcript,
            duration_s=self.source.duration_s,
            source_dir=self.source_dir,
            sample_fps=det.sample_fps,
            face_fps=det.face_fps,
            visual_mode=det.visual_mode,
            faces=det.faces,
            on_progress=self._stage_progress(stage),
            cancelled=self._is_cancelled,
        )
        self._check_cancelled()
        timeline.save(self.workspace.timeline)
        layout = (timeline.meta.get("layout") or {}).get("kind", "?")
        log.info(
            "Signals ready: %s (layout %s, %d chat messages).",
            ", ".join(timeline.names()),
            layout,
            int(timeline.meta.get("chat_messages", 0)),
        )
        self._finish_stage(stage)
        return timeline

    async def _stage_highlights(
        self, transcript: Transcript, silences: list[Silence]
    ) -> list[Clip]:
        stage = Stage.HIGHLIGHTS
        self._check_cancelled()

        existing = store.list_clips(self.job.id)
        if existing:
            log.info("Reusing %d existing clips for job %s.", len(existing), self.job.id)
            self._finish_stage(stage)
            return existing

        if self.multimodal and self.timeline is not None:
            self._emit(stage, 0.0, "Ranking moments")
            provider = await self._rerank_provider()
            clips = await signal_highlights.detect(
                self.timeline,
                transcript,
                self.settings,
                job_id=self.job.id,
                silences=silences,
                provider=provider,
                detection_config=detection_config(self.settings),
                source_dir=self.source_dir,
                duration_s=self.source.duration_s,
                on_progress=self._stage_progress(stage),
            )
            if not clips:
                raise PipelineError("No highlight moments were found in this source.", stage=stage)
            # Profile and chat lag were decided during detection; keep them.
            self.timeline.save(self.workspace.timeline)
            store.replace_clips(self.job.id, clips)
            self._finish_stage(stage)
            return clips

        provider_name = self.job.provider or self.settings.active_provider
        provider = build_provider(provider_name, self.settings)
        config = detection_config(self.settings)

        self._emit(stage, 0.0, f"Finding highlights with {provider.name}")
        clips = await highlights.detect(
            transcript,
            provider,
            config,
            job_id=self.job.id,
            silences=silences,
            on_progress=self._stage_progress(stage),
        )

        store.replace_clips(self.job.id, clips)
        self._finish_stage(stage)
        return clips

    async def _rerank_provider(self):
        """The LLM to judge candidates with, or None to rank on signals alone."""
        mode = self.settings.detection.llm_rerank
        if mode == "off":
            return None
        name = self.job.provider or self.settings.active_provider
        try:
            provider = build_provider(name, self.settings)
        except Exception as exc:
            if mode == "on":
                raise
            log.info("No LLM for re-ranking (%s); ranking on signals alone.", exc)
            return None
        if mode == "auto":
            try:
                status = await provider.health_check()
            except Exception as exc:
                log.info("LLM %s unreachable (%s); ranking on signals alone.", name, exc)
                return None
            if not status.available:
                log.info("LLM %s unavailable (%s); ranking on signals alone.", name, status.detail)
                return None
        return provider

    def _stack_layout(self):
        """Facecam/gameplay layout: job override, channel preset, or auto-detected."""
        from .. import layouts

        return layouts.resolve(
            job_id=self.job.id,
            platform=self.source.platform,
            channel=self.source.channel,
            timeline_meta=self.timeline.meta if self.timeline else None,
            split=self.settings.export.split_ratio,
            facecam_position=self.settings.export.facecam_position,
        )

    def _stage_reframe(self, clips: list[Clip], transcript: Transcript) -> dict[str, CropPath]:
        stage = Stage.REFRAME
        self._check_cancelled()
        source_path = Path(self.source.path)

        crop_paths: dict[str, CropPath] = {}

        if not self.source.has_video:
            # Audio-only sources render as captions on a solid background, so
            # there is nothing to reframe.
            self._finish_stage(stage)
            return crop_paths

        from .framing import crop_for_clip, layout_choice

        stack = self._stack_layout()
        log.info(
            "Export layout: %s%s",
            layout_choice(self.settings, stack, self.timeline),
            f" (facecam/gameplay from {stack.origin})" if stack else "",
        )
        info = ffmpeg.probe(source_path)

        for index, clip in enumerate(clips):
            self._check_cancelled()
            cached = self.workspace.crop_path(clip.id)
            if cached.exists():
                crop_paths[clip.id] = CropPath.load(cached)
            else:
                crop_paths[clip.id] = crop_for_clip(
                    source_path,
                    clip.start_s,
                    clip.end_s,
                    settings=self.settings,
                    stack=stack,
                    transcript=transcript,
                    width=info.width,
                    height=info.height,
                    timeline=self.timeline,
                )
                crop_paths[clip.id].save(cached)

            self._emit(stage, (index + 1) / len(clips), f"Reframing clip {index + 1}")

        self._finish_stage(stage)
        return crop_paths

    def _stage_captions(self, clips: list[Clip], transcript: Transcript) -> None:
        stage = Stage.CAPTIONS
        self._check_cancelled()
        # Caption files are written during export, where the output dimensions
        # are known. This stage validates the style so a typo fails fast rather
        # than after the reframe work is already done.
        captions.get_style(self.settings.export.caption_style)
        self._finish_stage(stage)

    def _stage_export(
        self,
        clips: list[Clip],
        transcript: Transcript,
        crop_paths: dict[str, CropPath],
    ) -> None:
        stage = Stage.EXPORT
        self._check_cancelled()

        style = captions.get_style(self.settings.export.caption_style)
        ratio = self.settings.export.ratio
        source_path = Path(self.source.path)
        destination_dir = paths.exports_dir() / self.job.id
        destination_dir.mkdir(parents=True, exist_ok=True)

        for index, clip in enumerate(clips):
            self._check_cancelled()

            crop_path = crop_paths.get(clip.id) or self._fallback_crop_path(clip, ratio)
            words, _origin = words_for_clip(
                clip,
                transcript,
                self.settings,
                work_dir=self.workspace.root,
                media=source_path,
                fetch=True,
            )
            name = export.output_filename(clip.title or f"clip-{clip.rank}", ratio)
            destination = destination_dir / f"{clip.rank:02d}_{name}"

            request = export.ExportRequest(
                source=source_path,
                destination=destination,
                start_s=clip.start_s,
                end_s=clip.end_s,
                crop_path=crop_path,
                words=words,
                style=style,
                ratio=ratio,
            )

            def clip_progress(fraction: float, i: int = index) -> None:
                self._emit(stage, (i + fraction) / len(clips), f"Exporting clip {i + 1}")

            export.export_clip(
                request,
                work_dir=self.workspace.captions_dir,
                settings=self.settings.export,
                on_progress=clip_progress,
                cancelled=self._is_cancelled,
            )

            store.create_export(
                Export(
                    id=new_id(),
                    clip_id=clip.id,
                    path=str(destination),
                    ratio=ratio,
                    style=style.key,
                    size_bytes=destination.stat().st_size,
                )
            )
            store.update_clip(clip.id, status="exported")

        self._finish_stage(stage)

    def _write_report(self, clips: list[Clip]) -> None:
        """Write clips.json + clips.md beside the exports: what, where, and why."""
        from .report import write_report

        write_report(
            paths.exports_dir() / self.job.id,
            source=self.source,
            clips=clips,
            view=self.settings.selection.view if self.multimodal else "overall",
            meta=(self.timeline.meta if self.timeline else {}),
        )

    def _fallback_crop_path(self, clip: Clip, ratio: str) -> CropPath:
        """Centre crop for sources with no reframe data (audio-only, or a failure)."""
        from .reframe.croppath import centre_crop

        width, height = export.ratio_dimensions(ratio)
        info = ffmpeg.probe(Path(self.source.path))
        return centre_crop(
            info.width or width,
            info.height or height,
            clip.end_s - clip.start_s,
            aspect_w=9 if ratio == "9:16" else (1 if ratio == "1:1" else 16),
            aspect_h=16 if ratio == "9:16" else (1 if ratio == "1:1" else 9),
        )


async def run_job(
    job_id: str,
    *,
    settings: Settings | None = None,
    on_progress: ProgressHandler | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Clip]:
    """Load a job and run it to completion."""
    job = store.get_job(job_id)
    if job is None:
        raise PipelineError(f"Job {job_id} not found.", stage=Stage.PREPARE)

    source = store.get_source(job.source_id)
    if source is None:
        raise PipelineError(f"Source {job.source_id} not found.", stage=Stage.PREPARE)

    runner = PipelineRunner(
        job, source, settings=settings, on_progress=on_progress, is_cancelled=is_cancelled
    )
    # The pipeline is mostly blocking work (ffmpeg, Whisper, MediaPipe) with one
    # async stage. Running it in a worker thread keeps the web server's event
    # loop responsive while a job is going.
    return await asyncio.get_running_loop().run_in_executor(None, _run_sync, runner)


def _run_sync(runner: PipelineRunner) -> list[Clip]:
    return asyncio.run(runner.run())

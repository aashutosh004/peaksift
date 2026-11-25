"""Live session — clip a stream while it is happening.

    recorder (60 s segments) ─┐
    backfill (10 min blocks) ─┼─► process chunk ─► timeline + transcript ─► fuse
    chat recorder (jsonl) ────┘    (audio, whisper, visual)                   │
                                                                         candidates
                                                     final once the reaction is over
                                                                              ▼
                                                              render clip from segments

* **Session time** is seconds since the stream started (when the platform
  reports it) so live segments, backfilled blocks and chat all share one clock.
* **Live first**: whenever new live segments are waiting they are processed
  before any backfill block, so clips keep flowing at the live edge while the
  earlier hours fill in.
* **Finalization** is coverage-based: a candidate is final once every second
  from its start to ``finalize_after_s`` past its end has been measured. The
  same rule works for the live edge and for backfilled hours.
* When the stream ends (or the user stops), a final pass re-ranks everything
  with the chosen view (by default, best moments per hour) and renders any
  clip that was not already rendered live.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import paths
from ..config import Settings
from ..db import store
from ..db.models import Clip, Export, Job, Source, new_id, utcnow
from ..jobs.owner import claim
from ..pipeline import captions, export, prepare, signal_highlights, transcribe
from ..pipeline.clip_words import words_for_clip
from ..pipeline.prepare import Silence
from ..pipeline.report import hms, write_report
from ..pipeline.transcript import Segment as TxSegment
from ..pipeline.transcript import Transcript, Word
from ..providers import build_provider, detection_config
from ..signals import add_chunk_visual, finalize_layout, refresh_derived
from ..signals.audio import audio_features
from ..signals.candidates import Candidate, build_candidates
from ..signals.chat import read_jsonl
from ..signals.fusion import FusionResult, fuse
from ..signals.timeline import Timeline
from ..sources import LinkInfo, kick_channel, probe, twitch_channel
from .recorder import Segment, SegmentRecorder, concat_segments

log = logging.getLogger(__name__)

BACKFILL_BLOCK_S = 600
POLL_S = 4.0
#: Never backfill further than this (DVR windows are limited anyway).
MAX_BACKFILL_S = 12 * 3600


class LiveError(RuntimeError):
    pass


OBS_DEFAULT_PORT = 1935


def is_obs_url(url: str) -> bool:
    lowered = url.strip().lower()
    return lowered == "obs" or lowered.startswith("rtmp://")


def obs_listen_url(url: str, port: int) -> str:
    """The RTMP address ffmpeg listens on (localhost only, never the network)."""
    if url.strip().lower().startswith("rtmp://"):
        return url.strip()
    return f"rtmp://127.0.0.1:{port}/live/stream"


@dataclass
class Chunk:
    path: Path
    start_s: float
    duration_s: float
    kind: str  # "live" | "backfill"


@dataclass
class _Emitted:
    clip: Clip
    file: str = ""
    #: Requested by the user (snapshot): always kept in the final set.
    manual: bool = False
    segments_needed: tuple[float, float] = (0.0, 0.0)


@dataclass
class LiveStatus:
    recorded_s: float = 0.0
    processed_s: float = 0.0
    backfill_total_s: float = 0.0
    backfill_done_s: float = 0.0
    clips: int = 0
    chat_messages: int = 0
    message: str = ""
    events: list[str] = field(default_factory=list)


class LiveSession:
    def __init__(
        self,
        url: str,
        settings: Settings,
        *,
        backfill: bool | None = None,
        max_hours: float | None = None,
        on_status: Callable[[LiveStatus], None] | None = None,
        chat_url: str | None = None,
        obs_port: int = OBS_DEFAULT_PORT,
        keyboard: bool = True,
    ) -> None:
        #: Read s/l/q keys from the console. Off when run from the web UI.
        self.keyboard = keyboard
        self.url = url
        self.obs = is_obs_url(url)
        self.obs_port = obs_port
        #: Where chat comes from: the stream itself, or (for OBS) a channel URL.
        self.chat_url = chat_url or (None if self.obs else url)
        self._snapshot_requests: list[float] = []
        self._pending_snapshots: list[tuple[float, float]] = []
        self._snapshot_lock = threading.Lock()
        self.settings = settings
        self.backfill = settings.live.backfill if backfill is None else backfill
        self.max_s = max_hours * 3600 if max_hours else None
        self.on_status = on_status or (lambda _s: None)
        self.status = LiveStatus()
        self._stop = threading.Event()

        self.info: LinkInfo | None = None
        self.source: Source | None = None
        self.job: Job | None = None
        self._owner: contextlib.AbstractContextManager | None = None
        self.origin_epoch = time.time()
        self.join_offset_s = 0.0

        self.timeline = Timeline()
        self.chunk_transcripts: list[tuple[float, Transcript]] = []
        self.silences: list[Silence] = []
        self.processed: set[str] = set()
        self.backfill_queue: list[Chunk] = []
        self.backfill_lock = threading.Lock()
        self.emitted: list[_Emitted] = []
        self._exported: set[str] = set()
        self.recorder: SegmentRecorder | None = None
        self.chat_recorder = None
        self.provider = None

    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------

    def stop(self) -> None:
        """Ask the session to wrap up (Ctrl+C). The final pass still runs."""
        self._stop.set()

    def _start(self) -> None:
        cookies = self.settings.ingest.cookies_from_browser
        if self.obs:
            self._start_obs()
            return
        try:
            self.info = probe(self.url, cookies_from_browser=cookies)
        except Exception as exc:
            message = str(exc).replace("ERROR: ", "").strip()
            raise LiveError(f"Could not open this stream: {message}") from exc
        if not self.info.is_live:
            raise LiveError(
                f"This link is not live right now (status: {self.info.live_status}). "
                "Use `peaksift clip <url>` for finished videos and past broadcasts."
            )
        if self.info.platform not in ("youtube", "twitch", "kick"):
            raise LiveError(
                f"Live clipping supports YouTube, Twitch and Kick; this is {self.info.platform}."
            )

        now = time.time()
        started = self.info.started_at
        if started and 0 < now - started < 48 * 3600:
            self.origin_epoch = float(started)
            self.join_offset_s = now - self.origin_epoch
        else:
            self.origin_epoch = now
            self.join_offset_s = 0.0

        self._create_records(
            self.info.title or "Live stream", self.info.channel, self.info.platform
        )
        self._event(
            f"Live: {self.source.title} ({self.info.platform}) — joined "
            f"{hms(self.join_offset_s)} into the stream."
        )

        self.recorder = SegmentRecorder(
            self.url,
            self.session_dir / "segments",
            origin_epoch=self.origin_epoch,
            segment_s=self.settings.live.segment_s,
            max_height=self.settings.live.max_height,
            cookies_from_browser=cookies,
            is_still_live=self._still_live,
        )
        self.recorder.start()
        self._start_chat()
        if self.backfill and self.join_offset_s > 60:
            threading.Thread(target=self._backfill_worker, name="backfill", daemon=True).start()

        self.provider = self._make_provider()

    def _start_obs(self) -> None:
        """Local OBS input: ffmpeg listens on localhost and OBS streams into it."""
        listen = obs_listen_url(self.url, self.obs_port)
        platform, channel = "obs", "obs"
        if self.chat_url:
            from ..sources import platform_of

            platform = platform_of(self.chat_url)
            channel = twitch_channel(self.chat_url) or kick_channel(self.chat_url) or "obs"
        self.info = LinkInfo(
            url=listen,
            platform=platform,
            live_status="is_live",
            title="OBS stream",
            channel=channel,
        )
        self.origin_epoch = time.time()
        self.join_offset_s = 0.0
        self.backfill = False
        self._create_records("OBS stream", channel, platform)
        server, _, key = listen.rpartition("/")
        self._event(
            "Waiting for OBS. In OBS: Settings > Stream > Service: Custom, "
            f"Server: {server}, Stream Key: {key}. Then press Start Streaming."
        )
        self.recorder = SegmentRecorder(
            listen,
            self.session_dir / "segments",
            origin_epoch=self.origin_epoch,
            segment_s=self.settings.live.segment_s,
            listen=True,
        )
        self.recorder.start()
        if self.chat_url:
            self._start_chat()
        self.provider = self._make_provider()

    def _create_records(self, title: str, channel: str, platform: str) -> None:
        source_id = new_id()
        self.session_dir = paths.source_media_dir(source_id)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.source = Source(
            id=source_id,
            type="live",
            url=self.url,
            path=str(self.session_dir / "recording"),
            title=title,
            channel=channel,
            platform=platform,
            live_status="is_live",
            duration_s=0.0,
        )
        store.create_source(self.source)
        self.job = store.create_job(
            Job(
                id=new_id(),
                source_id=source_id,
                provider=self.settings.active_provider,
                settings=self.settings.model_dump(mode="json"),
                status="running",
                current_stage="live",
                started_at=utcnow(),
            )
        )
        self.work_dir = paths.job_work_dir(self.job.id)
        self.export_dir = paths.exports_dir() / self.job.id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        # Tell a web app started later that this job is running here, not crashed.
        self._owner = claim(self.job.id)
        self._owner.__enter__()
        (self.work_dir / "live_session.json").write_text(
            json.dumps({"origin_epoch": self.origin_epoch, "url": self.url}), encoding="utf-8"
        )

    def _still_live(self) -> bool:
        if self.obs:
            return False
        try:
            return probe(
                self.url, cookies_from_browser=self.settings.ingest.cookies_from_browser
            ).is_live
        except Exception:
            return False

    def _start_chat(self) -> None:
        assert self.info is not None
        from ..sources import platform_of

        chat_url = self.chat_url or self.url
        chat_platform = platform_of(chat_url)
        chat_path = self.session_dir / "chat.jsonl"
        try:
            if chat_platform == "twitch" and (channel := twitch_channel(chat_url)):
                from ..sources.twitch import TwitchLiveChat

                self.chat_recorder = TwitchLiveChat(
                    channel, chat_path, origin_epoch=self.origin_epoch
                )
            elif chat_platform == "kick" and (channel := kick_channel(chat_url)):
                from ..sources.kick import KickLiveChat

                self.chat_recorder = KickLiveChat(
                    channel, chat_path, origin_epoch=self.origin_epoch
                )
            elif chat_platform == "youtube":
                from ..sources.youtube import YouTubeLiveChat

                self.chat_recorder = YouTubeLiveChat(
                    (self.info.webpage_url if not self.obs else "") or chat_url,
                    chat_path,
                    origin_epoch=self.origin_epoch,
                    work_dir=self.session_dir / "ytchat",
                )
            if self.chat_recorder:
                self.chat_recorder.start()
                self._event("Recording live chat.")
        except Exception as exc:
            log.warning("Live chat unavailable: %s", exc)

    def _make_provider(self):
        mode = self.settings.detection.llm_rerank
        if mode == "off":
            return None
        try:
            provider = build_provider(self.settings.active_provider, self.settings)
            if mode == "auto":
                status = asyncio.run(provider.health_check())
                if not status.available:
                    return None
            return provider
        except Exception as exc:
            log.info("Live re-ranking without LLM (%s).", exc)
            return None

    # ----------------------------------------------------------------------
    # Backfill
    # ----------------------------------------------------------------------

    def _backfill_worker(self) -> None:
        assert self.info is not None
        end = min(self.join_offset_s, MAX_BACKFILL_S)
        start = max(0.0, self.join_offset_s - MAX_BACKFILL_S)
        target_url, live_from_start = self._backfill_target()
        if target_url is None:
            self._event("Backfill not available for this platform/stream; clipping from now on.")
            return
        self.status.backfill_total_s = end - start
        self._event(f"Backfilling the first {hms(end - start)} of the stream in the background.")

        if self.info.platform == "twitch":
            from ..sources import twitch_vod_id
            from ..sources.twitch import fetch_vod_chat

            vod = twitch_vod_id(target_url)
            if vod and self.settings.ingest.fetch_chat:
                count = fetch_vod_chat(
                    vod, self.session_dir / "chat_backfill.jsonl", duration_s=end
                )
                if count:
                    self._event(f"Fetched {count} chat messages from before we joined.")

        block_dir = self.session_dir / "backfill"
        block_dir.mkdir(parents=True, exist_ok=True)
        t = start
        failures = 0
        while t < end and not self._stop.is_set():
            b = min(end, t + BACKFILL_BLOCK_S)
            dest = block_dir / f"block_{int(t):06d}.mp4"
            try:
                self._download_block(target_url, t, b, dest, live_from_start)
                with self.backfill_lock:
                    self.backfill_queue.append(Chunk(dest, t, b - t, "backfill"))
                failures = 0
                t = b
            except Exception as exc:
                failures += 1
                log.warning("Backfill block %s-%s failed: %s", hms(t), hms(b), exc)
                if failures >= 3:
                    self._event(
                        "Backfill stopped after repeated failures; live clipping continues."
                    )
                    return
                time.sleep(5)

    def _backfill_target(self) -> tuple[str | None, bool]:
        assert self.info is not None
        if self.info.platform == "youtube":
            return self.info.webpage_url or self.url, True
        if self.info.platform == "twitch":
            from ..sources.twitch import current_vod_id

            channel = twitch_channel(self.url)
            vod = current_vod_id(channel) if channel else None
            if vod:
                return f"https://www.twitch.tv/videos/{vod}", False
        return None, False

    def _download_block(
        self, url: str, a: float, b: float, dest: Path, live_from_start: bool
    ) -> None:
        import yt_dlp
        from yt_dlp.utils import download_range_func

        height = self.settings.live.max_height
        options: dict = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "format": f"best[height<={height}][vcodec!=none][acodec!=none]/best",
            "outtmpl": str(dest.with_suffix(".%(ext)s")),
            "download_ranges": download_range_func(None, [(a, b)]),
            "merge_output_format": "mp4",
        }
        if live_from_start:
            options["live_from_start"] = True
        if self.settings.ingest.cookies_from_browser:
            options["cookiesfrombrowser"] = (self.settings.ingest.cookies_from_browser,)
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
        produced = sorted(dest.parent.glob(dest.stem + ".*"), key=lambda p: p.stat().st_size)
        produced = [p for p in produced if p.suffix in (".mp4", ".ts", ".mkv", ".webm")]
        if not produced:
            raise RuntimeError("no file produced")
        if produced[-1] != dest:
            produced[-1].replace(dest)

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def run(self) -> list[Clip]:
        try:
            return self._run()
        finally:
            if self._owner is not None:
                self._owner.__exit__(None, None, None)
                self._owner = None

    def _run(self) -> list[Clip]:
        self._start()
        assert self.recorder is not None
        keys = _KeyReader(enabled=self.keyboard)
        try:
            while True:
                self._poll_snapshot_triggers(keys)
                did_work = self._process_pending_live()
                if not did_work:
                    did_work = self._process_one_backfill()
                if did_work:
                    self._rescore(final=False)
                self._handle_snapshots()
                self._report()

                if self.max_s and self.status.recorded_s >= self.max_s:
                    self._event("Reached the --max-hours limit.")
                    self._stop.set()
                if self._stop.is_set() or self.recorder.ended.is_set():
                    break
                if not did_work:
                    self._sleep_polling(keys)
        except KeyboardInterrupt:
            self._event("Stopping — finishing up the clips so far.")
        finally:
            self._shutdown_inputs()

        # Drain whatever finished recording before we stopped. Snapshot requests
        # that arrived during the last iteration are collected first, or a
        # request made just before the stream ended would be silently lost.
        while self._process_pending_live():
            pass
        self._poll_snapshot_triggers(keys)
        if self._pending_snapshots:
            self._rescore(final=True)
            self._handle_snapshots(force=True)
        if not self._stop.is_set():
            while self._process_one_backfill():
                pass
        return self._finish()

    def _shutdown_inputs(self) -> None:
        if self.recorder:
            self.recorder.stop()
        if self.chat_recorder:
            with contextlib.suppress(Exception):
                self.chat_recorder.stop()

    def _process_pending_live(self) -> bool:
        assert self.recorder is not None
        segments = self.recorder.segments()
        if segments:
            self.status.recorded_s = max(s.end_s for s in segments) - (
                self.join_offset_s if self.join_offset_s else 0
            )
        pending = [s for s in segments if str(s.path) not in self.processed]
        if not pending:
            return False
        for segment in pending:
            self._process_chunk(Chunk(segment.path, segment.start_s, segment.duration_s, "live"))
            self.processed.add(str(segment.path))
        return True

    def _process_one_backfill(self) -> bool:
        with self.backfill_lock:
            if not self.backfill_queue:
                return False
            chunk = self.backfill_queue.pop(0)
        self._process_chunk(chunk)
        self.status.backfill_done_s += chunk.duration_s
        return True

    def _process_chunk(self, chunk: Chunk) -> None:
        wav = self.work_dir / "chunks" / (chunk.path.stem + f"_{int(chunk.start_s)}.wav")
        try:
            prepare.extract_audio(chunk.path, wav)
        except Exception as exc:
            log.warning("Skipping unreadable chunk %s: %s", chunk.path.name, exc)
            return
        offset = int(round(chunk.start_s))
        self.timeline.put_all(offset, audio_features(wav))

        try:
            tr = transcribe.transcribe(wav, self.settings.whisper)
        except transcribe.NoSpeechError:
            tr = Transcript()
        except Exception as exc:
            log.warning("Transcription failed for %s: %s", chunk.path.name, exc)
            tr = Transcript()
        self.chunk_transcripts.append((chunk.start_s, tr))

        self.silences.extend(
            Silence(start=s.start + chunk.start_s, end=s.end + chunk.start_s)
            for s in prepare.detect_silences(wav)
        )
        det = self.settings.detection
        try:
            add_chunk_visual(
                self.timeline,
                chunk.path,
                offset_s=chunk.start_s,
                duration_s=chunk.duration_s,
                sample_fps=det.sample_fps,
                face_fps=det.face_fps,
                faces=det.faces,
            )
        except Exception as exc:
            log.warning("Visual signals failed for %s: %s", chunk.path.name, exc)
        wav.unlink(missing_ok=True)
        self.status.processed_s += chunk.duration_s

    # ----------------------------------------------------------------------
    # Scoring and clip emission
    # ----------------------------------------------------------------------

    def transcript(self) -> Transcript:
        """All chunk transcripts merged onto session time, in order."""
        merged = Transcript(model=self.settings.whisper.model)
        for offset, tr in sorted(self.chunk_transcripts, key=lambda item: item[0]):
            base = len(merged.words)
            last_end = merged.words[-1].end if merged.words else -1.0
            kept = [w for w in tr.words if w.start + offset > last_end]
            dropped = len(tr.words) - len(kept)
            for w in kept:
                merged.words.append(Word(w.text, w.start + offset, w.end + offset, w.speaker))
            for seg in tr.segments:
                first = base + max(0, seg.first_word - dropped)
                last = base + max(0, seg.last_word - dropped)
                if seg.start + offset <= last_end:
                    continue
                merged.segments.append(
                    TxSegment(
                        seg.text,
                        seg.start + offset,
                        seg.end + offset,
                        min(first, len(merged.words) - 1),
                        min(last, len(merged.words) - 1),
                    )
                )
        languages = [tr.language for _, tr in self.chunk_transcripts if tr.language]
        if languages:
            merged.language = max(set(languages), key=languages.count)
        return merged

    def _chat(self):
        chat = read_jsonl(self.session_dir / "chat.jsonl")
        chat += read_jsonl(self.session_dir / "chat_backfill.jsonl")
        self.status.chat_messages = len(chat)
        return chat

    def _rescore(self, *, final: bool) -> None:
        if self.timeline.length == 0:
            return
        transcript = self.transcript()
        refresh_derived(self.timeline, transcript=transcript, source_dir=None, chat=self._chat())
        finalize_layout(self.timeline)
        self.timeline.save(self.work_dir / "timeline.npz")
        transcript.save(self.work_dir / "transcript.json")
        if final:
            return

        det = self.settings.detection
        fusion = fuse(
            self.timeline,
            profile=det.profile,
            weights=det.weights or None,
            chat_lag_s=det.chat_lag_s,
        )
        hours = max(1.0, self.timeline.duration_s / 3600)
        per_hour = self.settings.live.clips_per_hour
        candidates = build_candidates(
            fusion,
            transcript=transcript,
            silences=self.silences,
            max_candidates=int(math.ceil(hours * per_hour * 2)) + 4,
            min_duration_s=self.settings.clips.min_duration_s,
            max_duration_s=self.settings.clips.max_duration_s,
        )
        covered = self.timeline.covered("loudness")
        wait = int(self.settings.live.finalize_after_s)

        def is_final(c: Candidate) -> bool:
            a, b = int(c.start_s), int(math.ceil(c.end_s)) + wait
            return b <= len(covered) and bool(covered[a:b].all())

        by_hour: dict[int, list[Candidate]] = {}
        for c in candidates:
            by_hour.setdefault(int(c.peak_s // 3600), []).append(c)
        for members in by_hour.values():
            members.sort(key=lambda c: c.raw_score, reverse=True)
            for c in members[:per_hour]:
                if is_final(c) and not self._already_emitted(c):
                    self._emit(c, transcript, fusion.profile, fusion.chat_lag_s)

    def _already_emitted(self, c: Candidate) -> bool:
        for e in self.emitted:
            inter = min(c.end_s, e.clip.end_s) - max(c.start_s, e.clip.start_s)
            if inter > 0 and inter / min(c.duration_s, e.clip.duration_s) > 0.5:
                return True
        return False

    def _emit(self, c: Candidate, transcript: Transcript, profile: str, chat_lag: float) -> None:
        assert self.job is not None
        verdict = None
        if self.provider is not None:
            try:
                judged = asyncio.run(
                    signal_highlights._rerank(
                        [c],
                        transcript,
                        self._chat(),
                        self.provider,
                        detection_config(self.settings),
                        chat_lag,
                        lambda _f: None,
                    )
                )
                verdict = judged.get(0)
                if verdict:
                    c.score = int(round(0.6 * c.score + 0.4 * verdict["score"]))
            except Exception as exc:
                log.debug("Live re-rank failed: %s", exc)
        clip = signal_highlights._to_clip(
            c,
            verdict,
            transcript,
            self.job.id,
            hms(c.peak_s // 3600 * 3600),
            FusionResult(score=np.zeros(0, dtype=np.float32), profile=profile, chat_lag_s=chat_lag),
        )
        entry = _Emitted(clip=clip)
        self.emitted.append(entry)
        self.status.clips = len(self.emitted)
        self._event(f"Clip #{len(self.emitted)} at {hms(c.peak_s)}: {clip.title} ({c.reason})")
        if self.settings.live.render_while_live:
            entry.file = self._render(clip, transcript) or ""
        self._save_clips()

    def _save_clips(self) -> None:
        assert self.job is not None
        clips = [e.clip for e in self.emitted]
        for rank, clip in enumerate(sorted(clips, key=lambda c: c.score, reverse=True), start=1):
            clip.rank = rank
        store.replace_clips(self.job.id, clips)
        for e in self.emitted:
            if e.file and Path(e.file).exists():
                store.create_export(
                    Export(
                        id=new_id(),
                        clip_id=e.clip.id,
                        path=e.file,
                        ratio=self.settings.export.ratio,
                        style=self.settings.export.caption_style,
                        size_bytes=Path(e.file).stat().st_size,
                    )
                )
                store.update_clip(e.clip.id, status="exported")

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------

    def _media_for(self, start_s: float, end_s: float) -> list[Chunk]:
        chunks: list[Chunk] = []
        if self.recorder:
            for s in self.recorder.segments():
                if s.end_s > start_s and s.start_s < end_s:
                    chunks.append(Chunk(s.path, s.start_s, s.duration_s, "live"))
        if not chunks:
            for path in sorted((self.session_dir / "backfill").glob("block_*.mp4")):
                a = float(path.stem.split("_")[1])
                if a < end_s and a + BACKFILL_BLOCK_S > start_s:
                    chunks.append(Chunk(path, a, BACKFILL_BLOCK_S, "backfill"))
        return sorted(chunks, key=lambda c: c.start_s)

    def _render(self, clip: Clip, transcript: Transcript) -> str | None:
        chunks = self._media_for(clip.start_s, clip.end_s)
        if not chunks:
            log.warning("No recorded media covers clip at %s.", hms(clip.start_s))
            return None
        base = chunks[0].start_s
        tmp = self.work_dir / "render" / f"{clip.id}.ts"
        try:
            if len(chunks) == 1:
                media = chunks[0].path
            else:
                media = concat_segments(
                    [
                        Segment(i, ch.path, ch.start_s, ch.start_s + ch.duration_s)
                        for i, ch in enumerate(chunks)
                    ],
                    tmp.with_suffix(".mp4"),
                )
            local_start = max(0.0, clip.start_s - base)
            local_end = clip.end_s - base
            session_words, _origin = words_for_clip(
                clip,
                transcript,
                self.settings,
                work_dir=self.work_dir,
                media=media,
                media_offset_s=base,
                fetch=True,
            )
            words = [Word(w.text, w.start - base, w.end - base, w.speaker) for w in session_words]
            crop = self._crop_for(media, local_start, local_end, timeline_start_s=clip.start_s)
            ratio = self.settings.export.ratio
            destination = self.export_dir / (
                f"live_{int(clip.peak_s or clip.start_s):06d}_"
                + export.output_filename(clip.title or "clip", ratio)
            )
            export.export_clip(
                export.ExportRequest(
                    source=media,
                    destination=destination,
                    start_s=local_start,
                    end_s=local_end,
                    crop_path=crop,
                    words=words,
                    style=captions.get_style(self.settings.export.caption_style),
                    ratio=ratio,
                ),
                work_dir=self.work_dir / "captions",
                settings=self.settings.export,
            )
            return str(destination)
        except Exception as exc:
            log.warning("Rendering clip at %s failed: %s", hms(clip.start_s), exc)
            return None
        finally:
            for leftover in (tmp.with_suffix(".mp4"), tmp.with_suffix(".txt")):
                leftover.unlink(missing_ok=True)

    def _crop_for(
        self, media: Path, start_s: float, end_s: float, timeline_start_s: float | None = None
    ):
        from .. import layouts
        from ..pipeline.framing import crop_for_clip

        assert self.job is not None and self.source is not None
        stack = layouts.resolve(
            job_id=self.job.id,
            platform=self.source.platform,
            channel=self.source.channel,
            timeline_meta=self.timeline.meta,
            split=self.settings.export.split_ratio,
            facecam_position=self.settings.export.facecam_position,
        )
        return crop_for_clip(
            media,
            start_s,
            end_s,
            settings=self.settings,
            stack=stack,
            timeline=self.timeline,
            timeline_start_s=timeline_start_s,
        )

    # ----------------------------------------------------------------------
    # Instant snapshots
    # ----------------------------------------------------------------------

    def request_snapshot(self, seconds: float = 30.0) -> None:
        """Clip the last ``seconds`` of the stream (thread-safe)."""
        with self._snapshot_lock:
            self._snapshot_requests.append(max(5.0, min(180.0, float(seconds))))

    def _live_edge(self) -> float:
        return time.time() - self.origin_epoch

    def _poll_snapshot_triggers(self, keys: _KeyReader) -> None:
        for key in keys.read():
            seconds = SNAPSHOT_KEYS.get(key.lower())
            if seconds:
                self.request_snapshot(seconds)
        trigger = self.work_dir / SNAPSHOT_TRIGGER
        if trigger.exists():
            try:
                for line in trigger.read_text(encoding="utf-8").split():
                    self.request_snapshot(float(line))
            except (OSError, ValueError):
                self.request_snapshot(30.0)
            trigger.unlink(missing_ok=True)
        with self._snapshot_lock:
            requests, self._snapshot_requests = self._snapshot_requests, []
        for seconds in requests:
            end = self._live_edge()
            self._pending_snapshots.append((max(0.0, end - seconds), end))
            self._event(f"Snapshot requested: the last {seconds:.0f}s (until {hms(end)}).")

    def _sleep_polling(self, keys: _KeyReader) -> None:
        deadline = time.time() + POLL_S
        while time.time() < deadline and not self._stop.is_set():
            if keys.pending():
                return
            time.sleep(0.2)

    def _handle_snapshots(self, force: bool = False) -> None:
        """Render snapshots once the recording covers them (or now, if forced)."""
        if not self._pending_snapshots or self.recorder is None:
            return
        segments = self.recorder.segments()
        recorded_until = max((seg.end_s for seg in segments), default=0.0)
        waiting: list[tuple[float, float]] = []
        for start, end in self._pending_snapshots:
            if recorded_until < end and not force:
                waiting.append((start, end))
                continue
            end = min(end, recorded_until)
            if end - start < 3:
                self._event("Snapshot skipped: nothing recorded in that window yet.")
                continue
            self._emit_snapshot(start, end)
        self._pending_snapshots = waiting

    def _emit_snapshot(self, start: float, end: float) -> None:
        assert self.job is not None
        transcript = self.transcript()
        clip = Clip(
            id=new_id(),
            job_id=self.job.id,
            start_s=round(start, 3),
            end_s=round(end, 3),
            title=f"Snapshot at {hms(end)}",
            score=100,
            reason="manual snapshot",
            peak_s=end,
            group_label="Snapshots",
            signals={"manual": True},
        )
        if transcript.words:
            first, last = transcript.indices_in_range(start, end)
            clip.start_word, clip.end_word = first, max(first, last - 1)
        entry = _Emitted(clip=clip, manual=True)
        self.emitted.append(entry)
        self.status.clips = len(self.emitted)
        entry.file = self._render(clip, transcript) or ""
        self._event(
            f"Snapshot saved: {Path(entry.file).name}" if entry.file else "Snapshot render failed."
        )
        self._save_clips()

    # ----------------------------------------------------------------------
    # Wrap-up
    # ----------------------------------------------------------------------

    def _finish(self) -> list[Clip]:
        assert self.job is not None and self.source is not None
        self._event("Stream finished — running the final ranking pass.")
        self._rescore(final=True)
        transcript = self.transcript()
        duration = self.timeline.duration_s
        store.update_source(self.source.id, duration_s=duration, live_status="was_live")
        self.source.duration_s = duration

        settings = self.settings.model_copy(deep=True)
        if settings.selection.view == "overall" and duration > 3600:
            # Long streams read best as "top moments of each hour".
            settings.selection.view = "per_hour"
        clips: list[Clip] = []
        if self.timeline.length:
            clips = asyncio.run(
                signal_highlights.detect(
                    self.timeline,
                    transcript,
                    settings,
                    job_id=self.job.id,
                    silences=self.silences,
                    provider=self.provider,
                    detection_config=detection_config(settings),
                    source_dir=self.session_dir,
                    duration_s=duration,
                )
            )

        files: dict[str, str] = {}
        final: list[Clip] = []
        for clip in clips:
            match = next(
                (
                    e
                    for e in self.emitted
                    if e.file and not e.manual and _overlap(clip, e.clip) > 0.5
                ),
                None,
            )
            if match is not None:
                match.clip.rank, match.clip.group_label = clip.rank, clip.group_label
                match.clip.score = max(match.clip.score, clip.score)
                final.append(match.clip)
                files[match.clip.id] = match.file
            else:
                path = self._render(clip, transcript)
                final.append(clip)
                if path:
                    files[clip.id] = path

        # Snapshots the user asked for are always kept, after the ranked clips.
        next_rank = max((c.rank for c in final), default=0)
        for e in self.emitted:
            if e.manual and e.clip not in final:
                next_rank += 1
                e.clip.rank = next_rank
                final.append(e.clip)
                if e.file:
                    files[e.clip.id] = e.file

        final.sort(key=lambda c: c.rank)
        store.replace_clips(self.job.id, final)
        for clip in final:
            if clip.id in files:
                store.create_export(
                    Export(
                        id=new_id(),
                        clip_id=clip.id,
                        path=files[clip.id],
                        ratio=settings.export.ratio,
                        style=settings.export.caption_style,
                        size_bytes=Path(files[clip.id]).stat().st_size,
                    )
                )
                store.update_clip(clip.id, status="exported")
        # Live-rendered clips that didn't make the final cut stay on disk in an
        # "extras" folder rather than being deleted.
        kept = set(files.values())
        extras = self.export_dir / "extras"
        for e in self.emitted:
            if e.file and e.file not in kept and Path(e.file).exists():
                extras.mkdir(exist_ok=True)
                shutil.move(e.file, extras / Path(e.file).name)

        write_report(
            self.export_dir,
            source=self.source,
            clips=final,
            view=settings.selection.view,
            meta=self.timeline.meta,
            files=files,
        )
        store.update_job(self.job.id, status="done", progress=1.0, finished_at=utcnow())
        self._event(f"Done: {len(final)} clips in {self.export_dir}")
        return final

    # ----------------------------------------------------------------------

    def _event(self, message: str) -> None:
        log.info(message)
        self.status.events.append(message)
        self.status.message = message
        self.on_status(self.status)

    def _report(self) -> None:
        self.on_status(self.status)


SNAPSHOT_TRIGGER = "snapshot.request"
#: Keys pressed in the live terminal: s = last 30 s, l = last 60 s, q = stop.
SNAPSHOT_KEYS = {"s": 30.0, "l": 60.0, "1": 15.0, "3": 30.0, "6": 60.0}


class _KeyReader:
    """Non-blocking single-key reads from the console (Windows), if attached."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._buffer: list[str] = []
        self._msvcrt = None
        if not enabled:
            return
        try:
            import msvcrt

            self._msvcrt = msvcrt if sys.stdin is not None and sys.stdin.isatty() else None
        except ImportError:
            self._msvcrt = None

    def pending(self) -> bool:
        if self._msvcrt is None:
            return False
        while self._msvcrt.kbhit():
            char = self._msvcrt.getwch()
            if char == "q":
                raise KeyboardInterrupt
            self._buffer.append(char)
        return bool(self._buffer)

    def read(self) -> list[str]:
        self.pending()
        keys, self._buffer = self._buffer, []
        return keys


def request_snapshot_for_running_session(seconds: float = 30.0) -> Path | None:
    """Ask the newest running live session (another process) for a snapshot."""
    for job in store.list_jobs(limit=20, status="running"):
        if job.current_stage == "live":
            trigger = paths.job_work_dir(job.id) / SNAPSHOT_TRIGGER
            with trigger.open("a", encoding="utf-8") as handle:
                handle.write(f"{seconds}\n")
            return trigger
    return None


def _overlap(a: Clip, b: Clip) -> float:
    inter = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    if inter <= 0:
        return 0.0
    return inter / max(1e-6, min(a.duration_s, b.duration_s))


def coverage_summary(timeline: Timeline) -> str:
    covered = timeline.covered("loudness")
    if not len(covered):
        return "nothing measured yet"
    return f"{int(np.sum(covered))}s measured of {len(covered)}s"

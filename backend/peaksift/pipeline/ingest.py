"""Ingestion — YouTube URLs and local file uploads become source records.

Both paths converge on a validated :class:`~peaksift.db.models.Source` with a
file on disk and probed metadata.

A note on YouTube: as of 2026 most anonymous downloads hit a bot check, and
proof-of-origin tokens no longer clear it reliably — browser cookies do. So
:class:`IngestError` distinguishes that specific failure and tells the user how
to fix it, rather than surfacing a raw yt-dlp traceback.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from .. import paths
from ..config import IngestSettings
from ..db.models import Source, new_id
from . import ffmpeg

log = logging.getLogger(__name__)

#: Container and audio formats we accept for upload.
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
ACCEPTED_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com")

#: yt-dlp error fragments that mean "YouTube wants proof you're a human".
_BOT_CHECK_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you are not a bot",
    "this content isn't available",
    "player response",
)


class IngestError(RuntimeError):
    """Ingestion failed in a way worth explaining to the user."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n\n{self.hint}" if self.hint else base


def is_youtube_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _YOUTUBE_HOSTS


def slugify(text: str, *, max_length: int = 60) -> str:
    """Turn a title into a filesystem- and URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length] or "clip"


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------


def ingest_youtube(
    url: str,
    settings: IngestSettings | None = None,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Source:
    """Download a YouTube video. Kept for API compatibility; see :func:`ingest_url`."""
    return ingest_url(url, settings, on_progress=on_progress)


def ingest_url(
    url: str,
    settings: IngestSettings | None = None,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Source:
    """Download a finished video from any yt-dlp-supported site.

    YouTube, Twitch and Kick VODs also bring their chat replay; YouTube adds
    its "most replayed" heatmap and timestamps from top comments. TikTok,
    Facebook, Instagram and other sites bring just the video. All extras are
    written next to the media (``chat.jsonl``, ``heatmap.json``,
    ``comments.json``) where the signal engine picks them up.

    Preconditions:
        url points at content the user owns or has the rights to process.
    """
    import yt_dlp

    from ..sources import platform_of

    settings = settings or IngestSettings()
    platform = platform_of(url)
    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    def hook(status: dict) -> None:
        if not on_progress or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes")
        if total and done:
            on_progress(min(1.0, done / total))

    options: dict = {
        "format": settings.ytdlp_format,
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "retries": 3,
        "fragment_retries": 3,
    }
    options["noplaylist"] = True
    if settings.cookies_from_browser:
        # yt-dlp expects a tuple; only the browser name is required.
        options["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    sub_langs: list[str] = []
    if settings.prefer_youtube_captions and platform == "youtube":
        options["writeautomaticsub"] = True
        sub_langs.append("en.*")
    if settings.fetch_chat and platform == "youtube":
        # Chat replay arrives as the "live_chat" pseudo-subtitle.
        options["writesubtitles"] = True
        sub_langs.append("live_chat")
    if sub_langs:
        options["subtitleslangs"] = sub_langs
        options["subtitlesformat"] = "json3/json/best"
    if settings.fetch_comments and platform == "youtube" and settings.max_comments > 0:
        options["getcomments"] = True
        options["extractor_args"] = {
            "youtube": {
                "max_comments": [str(settings.max_comments), "all", "0", "0"],
                "comment_sort": ["top"],
            }
        }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            metadata = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise _translate_ytdlp_error(exc, settings) from exc
    except Exception as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(f"Could not download {url}: {exc}") from exc

    metadata = metadata or {}
    if metadata.get("live_status") == "is_live" or metadata.get("is_live"):
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(
            "This stream is live right now.",
            hint="Use `peaksift live <url>` to clip it as it happens.",
        )

    downloaded = _find_downloaded_file(target_dir)
    if downloaded is None:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError("yt-dlp reported success but produced no media file.")

    try:
        info = _probe_and_validate(downloaded)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    duration = info.duration_s or float(metadata.get("duration") or 0.0)
    _save_extras(platform, url, metadata, target_dir, duration, settings)

    return Source(
        id=source_id,
        type="youtube" if platform == "youtube" else "url",
        platform=platform,
        live_status=str(metadata.get("live_status") or ""),
        url=url,
        path=str(downloaded),
        filename=downloaded.name,
        title=(metadata or {}).get("title") or downloaded.stem,
        channel=(metadata or {}).get("uploader") or (metadata or {}).get("channel"),
        duration_s=info.duration_s or float((metadata or {}).get("duration") or 0.0),
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


def _save_extras(
    platform: str,
    url: str,
    metadata: dict,
    target_dir: Path,
    duration_s: float,
    settings: IngestSettings,
) -> None:
    """Write chat replay, heatmap and comment timestamps beside the media.

    Every extra is optional: a failure is logged and ingest still succeeds.
    """
    from ..sources import twitch_vod_id
    from ..sources import youtube as yt

    chat_path = target_dir / "chat.jsonl"
    try:
        if platform == "youtube":
            for live_chat in target_dir.glob("*.live_chat.json"):
                count = yt.convert_live_chat_file(live_chat, chat_path)
                live_chat.unlink(missing_ok=True)
                log.info("Saved %d chat replay messages.", count)
            if yt.save_heatmap(metadata, target_dir / "heatmap.json"):
                log.info("Saved YouTube most-replayed heatmap.")
            if settings.fetch_comments:
                stamps = yt.save_comment_timestamps(metadata, target_dir / "comments.json")
                if stamps:
                    log.info("Found %d comment timestamps.", stamps)
        elif platform == "twitch" and settings.fetch_chat:
            raw_id = str(metadata.get("id", "")).lstrip("v")
            vod = twitch_vod_id(metadata.get("webpage_url") or url) or (
                raw_id if raw_id.isdigit() else None
            )
            if vod:
                from ..sources import twitch

                count = twitch.fetch_vod_chat(vod, chat_path, duration_s=duration_s)
                log.info("Saved %d Twitch chat replay messages.", count)
    except Exception as exc:  # extras must never fail an ingest
        log.warning("Could not save platform extras for %s: %s", url, exc)


def _translate_ytdlp_error(exc: Exception, settings: IngestSettings) -> IngestError:
    """Turn a yt-dlp failure into something the user can act on."""
    message = str(exc).lower()

    if any(marker in message for marker in _BOT_CHECK_MARKERS):
        if settings.cookies_from_browser:
            hint = (
                f"Cookies are already being read from {settings.cookies_from_browser}, but "
                "YouTube still refused. Make sure you are signed in to YouTube in that "
                "browser and that the browser is fully closed — it locks its cookie "
                "database while running."
            )
        else:
            hint = (
                "YouTube is asking for proof you're not a bot. Set a browser to pull "
                "cookies from — in Settings, or via config.json's "
                '`ingest.cookies_from_browser` (e.g. "chrome", "firefox", "edge"). '
                "You must be signed in to YouTube in that browser, and it must be closed "
                "while PeakSift downloads."
            )
        return IngestError("YouTube blocked this download with a bot check.", hint=hint)

    if "private video" in message or "members-only" in message:
        return IngestError(
            "This video is private or members-only.",
            hint="PeakSift only downloads content you can access and have the rights to use.",
        )
    if "unavailable" in message or "removed" in message:
        return IngestError("This video is unavailable or has been removed.")
    if "drm" in message:
        return IngestError(
            "This content is DRM-protected.",
            hint="PeakSift does not and will not circumvent DRM.",
        )

    return IngestError(
        "yt-dlp could not download this video.",
        hint=(
            "YouTube changes frequently and yt-dlp is updated often. Try "
            "`peaksift update-ytdlp` to pull the latest version.\n\n"
            f"Original error: {exc}"
        ),
    )


def _find_downloaded_file(directory: Path) -> Path | None:
    """Return the largest media file in a directory, ignoring sidecars."""
    candidates = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


def ingest_file(path: Path, *, move: bool = False, title: str | None = None) -> Source:
    """Register a local file as a source, copying it into PeakSift's media store.

    Copying rather than referencing in place means a job stays reproducible even
    if the user moves or deletes the original.

    Preconditions:
        path exists and is a readable media file.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"{path} does not exist.")
    if not path.is_file():
        raise IngestError(f"{path} is not a file.")

    suffix = path.suffix.lower()
    if suffix not in ACCEPTED_SUFFIXES:
        accepted = ", ".join(sorted(ACCEPTED_SUFFIXES))
        raise IngestError(
            f"{suffix or 'This file'} is not a supported format.",
            hint=f"Accepted formats: {accepted}",
        )

    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"source{suffix}"

    try:
        if move:
            shutil.move(str(path), target)
        else:
            shutil.copy2(path, target)
    except OSError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(f"Could not store {path.name}: {exc}") from exc

    try:
        info = _probe_and_validate(target)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise

    return Source(
        id=source_id,
        type="upload",
        path=str(target),
        filename=path.name,
        title=title or info.title or path.stem,
        duration_s=info.duration_s,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


def _probe_and_validate(path: Path) -> ffmpeg.MediaInfo:
    """Probe a media file and reject anything the pipeline can't process."""
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegError as exc:
        raise IngestError(f"{path.name} could not be read as media.", hint=str(exc)) from exc

    if not info.has_audio:
        raise IngestError(
            f"{path.name} has no audio track.",
            hint=(
                "PeakSift finds clips by transcribing speech, so a file with no audio "
                "has nothing to work from."
            ),
        )
    if info.duration_s <= 0:
        raise IngestError(
            f"{path.name} reports zero duration.",
            hint="The file may be corrupt or still being written.",
        )

    return info

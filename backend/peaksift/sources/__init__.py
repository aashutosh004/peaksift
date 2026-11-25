"""Platform adapters: which site a link belongs to, and what it can give us.

Scope (deliberate):

* **YouTube, Twitch, Kick** — finished videos/VODs *and* live streams, with
  chat (live chat while live, chat replay for VODs where the platform keeps it).
* **TikTok, Facebook, Instagram** and anything else yt-dlp supports — uploaded
  videos only. No live support and no chat; clips come from the audio, video
  and transcript signals alone.

Everything here degrades gracefully: a chat source that fails (rate limits,
API changes, Cloudflare) logs a warning and the clip pipeline carries on
without that signal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

log = logging.getLogger(__name__)

LIVE_PLATFORMS = ("youtube", "twitch", "kick")
UPLOAD_ONLY_PLATFORMS = ("tiktok", "facebook", "instagram")

_HOSTS: dict[str, tuple[str, ...]] = {
    "youtube": ("youtube.com", "youtu.be", "youtube-nocookie.com"),
    "twitch": ("twitch.tv",),
    "kick": ("kick.com",),
    "tiktok": ("tiktok.com",),
    "facebook": ("facebook.com", "fb.watch", "fb.com"),
    "instagram": ("instagram.com",),
}


def platform_of(url: str) -> str:
    """Return the platform key for a URL, ``other`` for any other site."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return "other"
    for platform, suffixes in _HOSTS.items():
        if any(host == s or host.endswith("." + s) for s in suffixes):
            return platform
    return "other"


def is_url(text: str) -> bool:
    return bool(re.match(r"^https?://", text.strip(), re.IGNORECASE))


def supports_live(platform: str) -> bool:
    return platform in LIVE_PLATFORMS


@dataclass
class LinkInfo:
    """What a link points at, from a metadata-only yt-dlp probe."""

    url: str
    platform: str
    #: ``is_live``, ``is_upcoming``, ``was_live``, ``not_live``, ``post_live``.
    live_status: str
    title: str = ""
    channel: str = ""
    video_id: str = ""
    duration_s: float = 0.0
    #: Unix time the live stream started, when the platform reports it.
    started_at: float | None = None
    webpage_url: str = ""

    @property
    def is_live(self) -> bool:
        return self.live_status == "is_live"


def probe(url: str, *, cookies_from_browser: str = "") -> LinkInfo:
    """Ask yt-dlp what the link is without downloading anything."""
    import yt_dlp

    options: dict = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False, process=False) or {}
    if info.get("_type") == "url" and info.get("url") and info.get("url") != url:
        # Channel pages resolve to the current live video, e.g. youtube.com/@x/live.
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(info["url"], download=False, process=False) or info

    status = info.get("live_status") or ("is_live" if info.get("is_live") else "not_live")
    return LinkInfo(
        url=url,
        platform=platform_of(info.get("webpage_url") or url),
        live_status=status,
        title=info.get("title") or "",
        channel=info.get("uploader") or info.get("channel") or "",
        video_id=str(info.get("id") or ""),
        duration_s=float(info.get("duration") or 0.0),
        started_at=info.get("release_timestamp") or info.get("timestamp"),
        webpage_url=info.get("webpage_url") or url,
    )


def twitch_channel(url: str) -> str | None:
    """``twitch.tv/<channel>`` -> channel login; None for VOD/clip URLs."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts or parts[0] in ("videos", "clip", "directory", "settings"):
        return None
    return parts[0].lower()


def twitch_vod_id(url: str) -> str | None:
    match = re.search(r"/videos/(\d+)", url)
    return match.group(1) if match else None


def kick_channel(url: str) -> str | None:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts or parts[0] in ("video", "categories"):
        return None
    return parts[0].lower()

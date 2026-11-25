"""YouTube extras: chat (replay and live), most-replayed heatmap, comment timestamps.

Chat comes from yt-dlp's ``live_chat`` pseudo-subtitle, which works for both
a finished stream's chat replay and a stream that is live right now (yt-dlp
keeps appending JSON lines while the stream runs). Parsing it here keeps us on
one well-maintained dependency instead of a dedicated chat library.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..signals.chat import ChatMessage, write_jsonl

log = logging.getLogger(__name__)

_TIMESTAMP = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")
_MONEY = re.compile(r"[\d.,]+")


# --------------------------------------------------------------------------
# live_chat.json parsing
# --------------------------------------------------------------------------


def parse_live_chat_line(line: str, *, origin_epoch: float | None = None) -> list[ChatMessage]:
    """Parse one JSON line of yt-dlp's ``live_chat`` output.

    Replay lines carry ``videoOffsetTimeMsec`` (stream time). Live lines may
    only carry ``timestampUsec`` (wall time); ``origin_epoch`` converts those to
    stream time.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return []
    replay = data.get("replayChatItemAction") or {}
    actions = replay.get("actions") or ([data] if "addChatItemAction" in data else [])
    offset_ms = replay.get("videoOffsetTimeMsec")

    messages: list[ChatMessage] = []
    for action in actions:
        item = (action.get("addChatItemAction") or {}).get("item") or {}
        for kind_key, renderer in item.items():
            t = _message_time(renderer, offset_ms, origin_epoch)
            if t is None:
                continue
            author = ((renderer.get("authorName") or {}).get("simpleText")) or ""
            text = _runs_text(renderer.get("message") or {})
            if kind_key == "liveChatTextMessageRenderer":
                messages.append(ChatMessage(t=t, text=text, author=author))
            elif kind_key in ("liveChatPaidMessageRenderer", "liveChatPaidStickerRenderer"):
                amount_text = (renderer.get("purchaseAmountText") or {}).get("simpleText", "")
                messages.append(
                    ChatMessage(
                        t=t,
                        text=text,
                        author=author,
                        kind="superchat",
                        amount=_parse_amount(amount_text),
                    )
                )
            elif kind_key in (
                "liveChatMembershipItemRenderer",
                "liveChatSponsorshipsGiftPurchaseAnnouncementRenderer",
            ):
                messages.append(ChatMessage(t=t, text=text, author=author, kind="sub"))
    return messages


def _message_time(renderer: dict, offset_ms: Any, origin_epoch: float | None) -> float | None:
    if offset_ms is not None:
        try:
            return max(0.0, int(offset_ms) / 1000.0)
        except (TypeError, ValueError):
            pass
    usec = renderer.get("timestampUsec")
    if usec is not None and origin_epoch is not None:
        try:
            return max(0.0, int(usec) / 1_000_000 - origin_epoch)
        except (TypeError, ValueError):
            return None
    return None


def _runs_text(message: dict) -> str:
    if "simpleText" in message:
        return message["simpleText"]
    parts: list[str] = []
    for run in message.get("runs") or []:
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            emoji = run["emoji"]
            shortcut = (emoji.get("shortcuts") or [""])[0]
            parts.append(emoji.get("emojiId") if len(emoji.get("emojiId", "")) <= 2 else shortcut)
    return "".join(p for p in parts if p)


def _parse_amount(text: str) -> float:
    match = _MONEY.search(text or "")
    if not match:
        return 0.0
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return 0.0


def convert_live_chat_file(source: Path, destination: Path) -> int:
    """Convert a finished ``*.live_chat.json`` (replay) into ``chat.jsonl``."""
    messages: list[ChatMessage] = []
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            messages.extend(parse_live_chat_line(line))
    messages.sort(key=lambda m: m.t)
    return write_jsonl(destination, messages)


# --------------------------------------------------------------------------
# Heatmap and comments
# --------------------------------------------------------------------------


def save_heatmap(metadata: dict, destination: Path) -> bool:
    heatmap = metadata.get("heatmap") or []
    if not heatmap:
        return False
    destination.write_text(json.dumps(heatmap), encoding="utf-8")
    return True


def comment_timestamps(comments: list[dict], duration_s: float) -> list[float]:
    """Pull ``1:23:45`` / ``12:34`` style timestamps out of comment text."""
    stamps: list[float] = []
    for comment in comments or []:
        text = comment.get("text") or ""
        likes = int(comment.get("like_count") or 0)
        for h, m, s in _TIMESTAMP.findall(text):
            seconds = int(h or 0) * 3600 + int(m) * 60 + int(s)
            if 0 < seconds <= duration_s:
                # A liked timestamp is agreement from many viewers.
                stamps.extend([float(seconds)] * min(5, 1 + likes // 25))
    return stamps


def save_comment_timestamps(metadata: dict, destination: Path) -> int:
    stamps = comment_timestamps(
        metadata.get("comments") or [], float(metadata.get("duration") or 0)
    )
    if stamps:
        destination.write_text(json.dumps(stamps), encoding="utf-8")
    return len(stamps)


# --------------------------------------------------------------------------
# Live chat recorder
# --------------------------------------------------------------------------


class YouTubeLiveChat:
    """Follow a live stream's chat via a yt-dlp subprocess, appending to chat.jsonl."""

    def __init__(self, url: str, out_path: Path, *, origin_epoch: float, work_dir: Path):
        self.url = url
        self.out_path = out_path
        self.origin_epoch = origin_epoch
        self.work_dir = work_dir
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--skip-download",
            "--write-subs",
            "--sub-langs",
            "live_chat",
            "--no-warnings",
            "--quiet",
            "-o",
            str(self.work_dir / "yt"),
            self.url,
        ]
        self._proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._tail, name="yt-live-chat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def _tail(self) -> None:
        position = 0
        while not self._stop.is_set():
            files = list(self.work_dir.glob("*.live_chat.json*"))
            if not files:
                time.sleep(2)
                continue
            path = max(files, key=lambda p: p.stat().st_size)
            try:
                with path.open(encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    batch: list[ChatMessage] = []
                    while True:
                        line = handle.readline()
                        if not line or not line.endswith("\n"):
                            break
                        position = handle.tell()
                        batch.extend(parse_live_chat_line(line, origin_epoch=self.origin_epoch))
                if batch:
                    write_jsonl(self.out_path, batch, append=True)
            except OSError as exc:
                log.debug("Live chat tail: %s", exc)
            time.sleep(2)

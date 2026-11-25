"""Kick live chat over Kick's public Pusher websocket.

Kick has no official public chat API. The website reads chat from a Pusher
channel named ``chatrooms.<id>.v2``; the chatroom id comes from the channel
endpoint, which sits behind Cloudflare and sometimes refuses non-browser
clients. Every failure here is soft: a warning, and clipping continues without
chat.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from ..signals.chat import ChatMessage, write_jsonl

log = logging.getLogger(__name__)

PUSHER_URL = (
    "wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679?protocol=7&client=js&version=8.4.0-rc2"
    "&flash=false"
)
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)


class ChatUnavailable(RuntimeError):
    pass


def chatroom_id(channel: str) -> int:
    import httpx

    try:
        response = httpx.get(
            f"https://kick.com/api/v2/channels/{channel}",
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=15,
            follow_redirects=True,
        )
        response.raise_for_status()
        return int(response.json()["chatroom"]["id"])
    except Exception as exc:
        raise ChatUnavailable(f"Kick chatroom lookup failed for {channel}: {exc}") from exc


def parse_pusher_message(raw: str, t: float) -> ChatMessage | None:
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return None
    event = envelope.get("event", "")
    if (
        not event.endswith("ChatMessageEvent")
        and not event.endswith("SubscriptionEvent")
        and not event.endswith("GiftedSubscriptionsEvent")
    ):
        return None
    try:
        data = json.loads(envelope.get("data") or "{}")
    except json.JSONDecodeError:
        return None
    if event.endswith("ChatMessageEvent"):
        sender = (data.get("sender") or {}).get("username", "")
        return ChatMessage(t=t, text=data.get("content", ""), author=sender)
    if event.endswith("GiftedSubscriptionsEvent"):
        gifted = data.get("gifted_usernames") or []
        return ChatMessage(
            t=t,
            text="",
            author=data.get("gifter_username", ""),
            kind="gift",
            amount=float(len(gifted) or 1),
        )
    return ChatMessage(t=t, text="", author=data.get("username", ""), kind="sub")


class KickLiveChat:
    def __init__(self, channel: str, out_path: Path, *, origin_epoch: float):
        self.channel = channel
        self.out_path = out_path
        self.origin_epoch = origin_epoch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="kick-chat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            room = chatroom_id(self.channel)
        except ChatUnavailable as exc:
            log.warning("%s — continuing without Kick chat.", exc)
            return
        from websockets.sync.client import connect

        backoff = 2.0
        while not self._stop.is_set():
            try:
                with connect(PUSHER_URL, open_timeout=20) as ws:
                    ws.send(
                        json.dumps(
                            {
                                "event": "pusher:subscribe",
                                "data": {"auth": "", "channel": f"chatrooms.{room}.v2"},
                            }
                        )
                    )
                    batch: list[ChatMessage] = []
                    last_flush = time.time()
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv(timeout=1.0)
                        except TimeoutError:
                            raw = None
                        if raw:
                            if '"pusher:ping"' in raw:
                                ws.send(json.dumps({"event": "pusher:pong", "data": {}}))
                            else:
                                message = parse_pusher_message(raw, time.time() - self.origin_epoch)
                                if message:
                                    batch.append(message)
                        if batch and time.time() - last_flush > 1.0:
                            write_jsonl(self.out_path, batch, append=True)
                            batch, last_flush = [], time.time()
                backoff = 2.0
            except Exception as exc:
                log.warning("Kick chat disconnected (%s); reconnecting.", exc)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

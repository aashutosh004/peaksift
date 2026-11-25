"""Twitch chat: live via anonymous IRC, VOD replay via Twitch's public GQL API.

Live IRC is the officially supported, stable path: an anonymous ``justinfan``
login can read any public channel. Tags give us bits, subs, gifts and raids.

VOD chat replay uses the same GQL endpoint the Twitch website uses. It is not
an official API and Twitch changes it from time to time, so failures are
reported as a warning and the pipeline continues without chat.
"""

from __future__ import annotations

import logging
import random
import socket
import threading
import time
from pathlib import Path

from ..signals.chat import ChatMessage, write_jsonl

log = logging.getLogger(__name__)

IRC_HOST = "irc.chat.twitch.tv"
IRC_PORT = 6667

GQL_URL = "https://gql.twitch.tv/gql"
#: The web client's public client id, used by the Twitch site itself.
GQL_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
COMMENTS_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"


# --------------------------------------------------------------------------
# IRC parsing
# --------------------------------------------------------------------------


def parse_irc_line(line: str, t: float) -> ChatMessage | None:
    """Turn one IRC line into a ChatMessage, or None for non-chat lines."""
    tags: dict[str, str] = {}
    rest = line
    if rest.startswith("@"):
        raw_tags, _, rest = rest.partition(" ")
        for pair in raw_tags[1:].split(";"):
            key, _, value = pair.partition("=")
            tags[key] = value.replace("\\s", " ")
    # rest: ":nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :message text"
    head, _, trailing = rest.partition(" :")
    words = head.split()
    if len(words) < 2:
        return None
    command = words[1]
    if command == "PRIVMSG":
        author = tags.get("display-name") or words[0].lstrip(":").split("!")[0]
        bits = tags.get("bits")
        if bits and bits.isdigit():
            return ChatMessage(
                t=t, text=trailing, author=author, kind="bits", amount=int(bits) / 100.0
            )
        return ChatMessage(t=t, text=trailing, author=author)
    if command == "USERNOTICE":
        msg_id = tags.get("msg-id", "")
        author = tags.get("display-name") or tags.get("login", "")
        if msg_id in ("sub", "resub"):
            return ChatMessage(t=t, text=tags.get("system-msg", ""), author=author, kind="sub")
        if msg_id in ("subgift", "submysterygift", "giftpaidupgrade"):
            count = tags.get("msg-param-mass-gift-count") or "1"
            return ChatMessage(
                t=t,
                text="",
                author=author,
                kind="gift",
                amount=float(count) if count.isdigit() else 1.0,
            )
        if msg_id == "raid":
            viewers = tags.get("msg-param-viewerCount", "0")
            return ChatMessage(
                t=t,
                text="",
                author=author,
                kind="raid",
                amount=float(viewers) if viewers.isdigit() else 0.0,
            )
    return None


class TwitchLiveChat:
    """Anonymous read-only IRC client appending to chat.jsonl in a thread."""

    def __init__(self, channel: str, out_path: Path, *, origin_epoch: float):
        self.channel = channel.lower().lstrip("#")
        self.out_path = out_path
        self.origin_epoch = origin_epoch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="twitch-irc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                self._session()
                backoff = 2.0
            except OSError as exc:
                log.warning("Twitch chat disconnected (%s); reconnecting.", exc)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    def _session(self) -> None:
        with socket.create_connection((IRC_HOST, IRC_PORT), timeout=30) as sock:
            sock.settimeout(1.0)
            nick = f"justinfan{random.randint(10000, 99999)}"
            for command in (
                "CAP REQ :twitch.tv/tags twitch.tv/commands",
                "PASS SCHMOOPIIE",
                f"NICK {nick}",
                f"JOIN #{self.channel}",
            ):
                sock.sendall((command + "\r\n").encode())
            buffer = ""
            batch: list[ChatMessage] = []
            last_flush = time.time()
            while not self._stop.is_set():
                try:
                    data = sock.recv(65536)
                    if not data:
                        raise OSError("connection closed")
                    buffer += data.decode("utf-8", "replace")
                except TimeoutError:  # socket.timeout is an alias since 3.10
                    pass
                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)
                    if line.startswith("PING"):
                        sock.sendall(line.replace("PING", "PONG", 1).encode() + b"\r\n")
                        continue
                    message = parse_irc_line(line, time.time() - self.origin_epoch)
                    if message is not None:
                        batch.append(message)
                if batch and time.time() - last_flush > 1.0:
                    write_jsonl(self.out_path, batch, append=True)
                    batch = []
                    last_flush = time.time()


# --------------------------------------------------------------------------
# VOD replay
# --------------------------------------------------------------------------


def fetch_vod_chat(
    vod_id: str, destination: Path, *, duration_s: float, on_progress=None, max_requests: int = 6000
) -> int:
    """Download a VOD's chat replay. Returns the number of messages (0 on failure)."""
    import httpx

    headers = {"Client-Id": GQL_CLIENT_ID, "Content-Type": "text/plain;charset=UTF-8"}
    messages: list[ChatMessage] = []
    offset = 0.0
    seen: set[str] = set()
    requests = 0
    try:
        with httpx.Client(timeout=20, headers=headers) as client:
            while offset <= duration_s + 1 and requests < max_requests:
                body = [
                    {
                        "operationName": "VideoCommentsByOffsetOrCursor",
                        "variables": {"videoID": vod_id, "contentOffsetSeconds": int(offset)},
                        "extensions": {
                            "persistedQuery": {"version": 1, "sha256Hash": COMMENTS_HASH}
                        },
                    }
                ]
                response = client.post(GQL_URL, json=body)
                requests += 1
                response.raise_for_status()
                payload = response.json()
                video = ((payload[0] or {}).get("data") or {}).get("video") if payload else None
                comments = (video or {}).get("comments") or {}
                edges = comments.get("edges") or []
                if not edges:
                    if (payload[0] or {}).get("errors"):
                        raise RuntimeError(payload[0]["errors"][0].get("message", "GQL error"))
                    offset += 30
                    continue
                newest = offset
                for edge in edges:
                    node = edge.get("node") or {}
                    node_id = node.get("id") or edge.get("cursor") or ""
                    if node_id in seen:
                        continue
                    seen.add(node_id)
                    t = float(node.get("contentOffsetSeconds") or 0)
                    newest = max(newest, t)
                    fragments = ((node.get("message") or {}).get("fragments")) or []
                    text = "".join(f.get("text", "") for f in fragments)
                    author = ((node.get("commenter") or {}).get("displayName")) or ""
                    messages.append(ChatMessage(t=t, text=text, author=author))
                offset = newest + 1 if newest > offset else offset + 5
                if on_progress and duration_s:
                    on_progress(min(1.0, offset / duration_s))
    except Exception as exc:
        log.warning("Twitch chat replay unavailable for VOD %s: %s", vod_id, exc)
        if not messages:
            return 0
    messages.sort(key=lambda m: m.t)
    return write_jsonl(destination, messages)


def current_vod_id(channel: str) -> str | None:
    """The in-progress archive VOD of a live channel, used for backfill."""
    import httpx

    query = (
        '{user(login:"' + channel + '")'
        "{videos(first:1,type:ARCHIVE,sort:TIME){edges{node{id status}}}}}"
    )
    try:
        response = httpx.post(
            GQL_URL, json={"query": query}, headers={"Client-Id": GQL_CLIENT_ID}, timeout=15
        )
        response.raise_for_status()
        edges = (((response.json().get("data") or {}).get("user") or {}).get("videos") or {}).get(
            "edges"
        ) or []
        if edges and (edges[0]["node"].get("status") or "").upper() == "RECORDING":
            return str(edges[0]["node"]["id"])
    except Exception as exc:
        log.info("Could not look up the live VOD for %s: %s", channel, exc)
    return None

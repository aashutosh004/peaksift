"""Chat signals — what the audience thought, per second.

``chat_rate``    messages per second.
``chat_unique``  distinct chatters per second (resists one spammer).
``chat_laugh``   laugh emotes/words (KEKW, LUL, lmao, 😂 ...).
``chat_hype``    hype emotes/words (POG, W, LETS GO, 🔥, "clip it" ...).
``chat_shock``   shock emotes/words (WTF, monkaS, 😱, "no way" ...).
``chat_events``  paid/sub events weighted by size (bits, subs, superchats).

Chat reacts *after* the moment — viewers watch, then type, on top of stream
delay. That lag is estimated and removed in :mod:`.fusion`, not here; these
arrays are in raw message time.

Messages are stored as JSON lines (``chat.jsonl``) so a live recorder can
append while the scorer reads.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ChatMessage:
    #: Seconds from the start of the source (stream time, before lag removal).
    t: float
    text: str
    author: str = ""
    #: ``msg``, ``sub``, ``gift``, ``bits``, ``superchat``, ``donation``, ``raid``.
    kind: str = "msg"
    #: Monetary or bits amount, when known (bits are converted at 100 = 1 unit).
    amount: float = 0.0


def write_jsonl(path: Path, messages: Iterable[ChatMessage], *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[ChatMessage]:
    if not path.exists():
        return []
    messages: list[ChatMessage] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                messages.append(ChatMessage(**data))
            except (json.JSONDecodeError, TypeError):
                # A live writer may leave a half-written last line.
                continue
    return messages


# --------------------------------------------------------------------------
# Lexicons
# --------------------------------------------------------------------------

_LAUGH = {
    "kekw",
    "lul",
    "lulw",
    "omegalul",
    "lmao",
    "lmfao",
    "lol",
    "haha",
    "hahaha",
    "xd",
    "icant",
    "kekwait",
    "pepelaugh",
    "lmaooo",
    "rofl",
    "😂",
    "🤣",
    "💀",
    "😭",
}
_HYPE = {
    "pog",
    "pogchamp",
    "poggers",
    "pogu",
    "w",
    "ww",
    "www",
    "letsgo",
    "lets",
    "gg",
    "ez",
    "insane",
    "clip",
    "clipit",
    "goat",
    "🔥",
    "🐐",
    "💯",
    "hype",
    "omg",
    "sheesh",
    "peak",
}
_SHOCK = {
    "wtf",
    "monkas",
    "monkaw",
    "d:",
    "omg",
    "nooo",
    "noway",
    "😱",
    "😳",
    "🤯",
    "what",
    "bruh",
    "holy",
    "jesus",
    "wait",
    "huh",
    "ayo",
}
#: Phrases checked on the whole message before tokenising.
_PHRASES = {
    "clip it": ("hype", 3.0),
    "clip that": ("hype", 3.0),
    "lets go": ("hype", 1.5),
    "let's go": ("hype", 1.5),
    "no way": ("shock", 1.5),
    "what the": ("shock", 1.0),
}
_TOKEN = re.compile(r"[\w:']+|[\U0001F300-\U0001FAFF☀-➿]", re.UNICODE)
_REPEAT = re.compile(r"(.)\1{2,}")


def _classify(text: str) -> tuple[float, float, float]:
    lowered = text.lower()
    laugh = hype = shock = 0.0
    for phrase, (kind, weight) in _PHRASES.items():
        if phrase in lowered:
            if kind == "hype":
                hype += weight
            else:
                shock += weight
    for token in _TOKEN.findall(lowered):
        # "loooool" -> "lool", "nooooo" -> "noo": collapse stretched words.
        squashed = _REPEAT.sub(r"\1\1", token)
        if token in _LAUGH or squashed in _LAUGH or squashed.startswith("hah"):
            laugh += 1.0
        if token in _HYPE or squashed in _HYPE:
            hype += 1.0
        if token in _SHOCK or squashed in _SHOCK or squashed.startswith("noo"):
            shock += 1.0
    # Cap per message so one copy-paste wall doesn't outweigh a hundred viewers.
    return min(laugh, 3.0), min(hype, 3.0), min(shock, 3.0)


_EVENT_WEIGHT = {
    "sub": 3.0,
    "gift": 4.0,
    "bits": 1.0,
    "superchat": 1.0,
    "donation": 1.0,
    "raid": 5.0,
}


def chat_features(messages: list[ChatMessage], length: int) -> dict[str, np.ndarray]:
    """Per-second chat signals over ``length`` seconds."""
    names = ("chat_rate", "chat_unique", "chat_laugh", "chat_hype", "chat_shock", "chat_events")
    out = {name: np.zeros(length, dtype=np.float32) for name in names}
    if not messages or length <= 0:
        return out

    authors: dict[int, set[str]] = {}
    for message in messages:
        second = int(message.t)
        if second < 0 or second >= length:
            continue
        if message.kind == "msg":
            out["chat_rate"][second] += 1.0
            authors.setdefault(second, set()).add(message.author or f"anon{id(message)}")
            laugh, hype, shock = _classify(message.text)
            out["chat_laugh"][second] += laugh
            out["chat_hype"][second] += hype
            out["chat_shock"][second] += shock
        else:
            weight = _EVENT_WEIGHT.get(message.kind, 1.0)
            # Log-scale amounts: a 10x bigger donation is notable, not 10x.
            amount = np.log1p(max(0.0, message.amount)) if message.amount else 1.0
            out["chat_events"][second] += weight * float(amount)
            if message.text:
                laugh, hype, shock = _classify(message.text)
                out["chat_hype"][second] += hype

    for second, names_seen in authors.items():
        out["chat_unique"][second] = float(len(names_seen))
    return out

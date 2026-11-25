"""Signal fusion — many per-second signals in, one "how interesting" curve out.

Three problems make naive summing useless on real streams, and each has a step
here:

1. **Every stream has its own normal.** A loud FPS streamer is always loud; a
   calm podcast never is. Each signal is turned into a *robust z-score* against
   a rolling baseline (median and MAD over the surrounding ~10 minutes), so a
   spike means "unusual for this stream, right now" rather than "loud".
2. **Chat is late.** Viewers react after the moment, on top of stream latency.
   The lag is estimated by cross-correlating chat activity against the
   audio/visual reaction curve, and chat is shifted back by it.
3. **One signal lies; several agreeing rarely do.** A door slam spikes audio
   alone. A real moment usually lights up audio, face and chat together, so
   simultaneous agreement multiplies the score.

Weights depend on the content profile (gaming, just chatting, podcast), which
is inferred from the picture layout unless the user sets it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .timeline import Timeline

log = logging.getLogger(__name__)

#: Signals that are states, not events, or are only used for chapters.
NON_EVENT_PREFIXES = ("vsig_",)
NON_EVENT = {"face_present", "face_size"}

CHAT_SIGNALS = ("chat_rate", "chat_unique", "chat_laugh", "chat_hype", "chat_shock", "chat_events")

PROFILES: dict[str, dict[str, float]] = {
    "gaming": {
        "loudness": 1.0,
        "flux": 0.8,
        "shout": 0.8,
        "motion": 0.6,
        "scene": 0.4,
        "mouth_open": 0.6,
        "face_surprise": 0.8,
        "face_joy": 0.5,
        "head_motion": 0.3,
        "speech_rate": 0.3,
        "exclaim": 0.8,
        "chat_rate": 1.2,
        "chat_unique": 0.8,
        "chat_laugh": 1.0,
        "chat_hype": 1.2,
        "chat_shock": 1.0,
        "chat_events": 0.6,
        "heatmap": 2.0,
        "comments": 1.5,
    },
    "just_chatting": {
        "loudness": 0.8,
        "flux": 0.6,
        "shout": 0.8,
        "motion": 0.3,
        "scene": 0.2,
        "mouth_open": 0.6,
        "face_surprise": 0.9,
        "face_joy": 0.9,
        "head_motion": 0.5,
        "speech_rate": 0.5,
        "exclaim": 0.9,
        "chat_rate": 1.2,
        "chat_unique": 0.8,
        "chat_laugh": 1.2,
        "chat_hype": 1.0,
        "chat_shock": 1.0,
        "chat_events": 0.6,
        "heatmap": 2.0,
        "comments": 1.5,
    },
    "podcast": {
        "loudness": 0.6,
        "flux": 0.4,
        "shout": 0.4,
        "motion": 0.1,
        "scene": 0.1,
        "mouth_open": 0.3,
        "face_surprise": 0.6,
        "face_joy": 0.8,
        "head_motion": 0.2,
        "speech_rate": 0.3,
        "exclaim": 0.8,
        "chat_rate": 1.0,
        "chat_unique": 0.8,
        "chat_laugh": 1.2,
        "chat_hype": 0.8,
        "chat_shock": 0.8,
        "chat_events": 0.5,
        "heatmap": 2.5,
        "comments": 1.5,
    },
}

#: Human-readable names for explaining why a clip was picked.
LABELS = {
    "loudness": "volume spike",
    "flux": "sudden audio change",
    "shout": "shouting/screaming",
    "motion": "lots of on-screen action",
    "scene": "scene change",
    "mouth_open": "big vocal reaction",
    "face_surprise": "surprised face",
    "face_joy": "laughing/smiling",
    "head_motion": "physical reaction",
    "speech_rate": "rapid talking",
    "exclaim": "exclamations",
    "chat_rate": "chat exploded",
    "chat_unique": "many chatters reacting",
    "chat_laugh": "chat laughing",
    "chat_hype": "chat hype",
    "chat_shock": "chat shocked",
    "chat_events": "subs/donations",
    "heatmap": "most replayed part",
    "comments": "timestamped in comments",
}

BASELINE_WINDOW_S = 600
BASELINE_STEP_S = 15
Z_CLIP = (-2.0, 8.0)
AGREE_Z = 1.5


@dataclass
class FusionResult:
    score: np.ndarray
    #: Per-signal weighted positive z contribution, after lag correction.
    contributions: dict[str, np.ndarray] = field(default_factory=dict)
    profile: str = "gaming"
    chat_lag_s: float = 0.0
    used: list[str] = field(default_factory=list)


def robust_z(values: np.ndarray, *, window_s: int = BASELINE_WINDOW_S) -> np.ndarray:
    """Z-score each sample against a rolling median/MAD baseline.

    The baseline is computed on a coarse grid (every ``BASELINE_STEP_S``
    seconds) and interpolated, which is ~15x cheaper than a true per-sample
    rolling median and indistinguishable at a 10-minute window.
    """
    n = len(values)
    if n == 0:
        return values.astype(np.float32)
    x = values.astype(np.float64)
    global_med = float(np.median(x))
    global_mad = float(np.median(np.abs(x - global_med))) * 1.4826
    half = window_s // 2
    centres = np.arange(0, n, BASELINE_STEP_S)
    if centres[-1] != n - 1:
        centres = np.append(centres, n - 1)
    meds = np.empty(len(centres))
    mads = np.empty(len(centres))
    for i, c in enumerate(centres):
        seg = x[max(0, c - half) : min(n, c + half + 1)]
        m = float(np.median(seg))
        meds[i] = m
        mads[i] = float(np.median(np.abs(seg - m))) * 1.4826
    med = np.interp(np.arange(n), centres, meds)
    mad = np.interp(np.arange(n), centres, mads)
    # A locally flat stretch (MAD ~0) would turn any blip into a huge z, so the
    # spread is floored at half the *typical* local spread. Using the typical
    # local spread rather than the global one matters: a stream that is quiet
    # for an hour and loud the next has a huge global spread that would
    # otherwise flatten every real spike in both halves.
    typical = float(np.median(mads)) if np.any(mads > 0) else global_mad
    spread = np.maximum(mad, 0.5 * typical) + 1e-6 + 0.05 * abs(global_med) * (typical == 0)
    z = (x - med) / spread
    return np.clip(z, *Z_CLIP).astype(np.float32)


def smooth(values: np.ndarray, width: int, *, mode: str = "mean") -> np.ndarray:
    if width <= 1 or len(values) == 0:
        return values.astype(np.float32)
    if mode == "max":
        from numpy.lib.stride_tricks import sliding_window_view

        pad = width // 2
        padded = np.pad(values, (pad, width - 1 - pad), mode="edge")
        return sliding_window_view(padded, width).max(axis=1).astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def estimate_chat_lag(
    chat: np.ndarray,
    reaction: np.ndarray,
    *,
    min_lag: int = 1,
    max_lag: int = 30,
    default: float = 6.0,
) -> float:
    """Find how many seconds chat trails the on-stream reaction.

    Returns ``default`` when the correlation is too weak to trust — a quiet
    chat or a stream with no reactions shouldn't produce an arbitrary lag.
    """
    n = min(len(chat), len(reaction))
    if n < max_lag * 4 or chat[:n].std() < 1e-6 or reaction[:n].std() < 1e-6:
        return default
    a = (reaction[:n] - reaction[:n].mean()) / (reaction[:n].std() + 1e-9)
    b = (chat[:n] - chat[:n].mean()) / (chat[:n].std() + 1e-9)
    best_lag, best_corr = default, 0.0
    for lag in range(min_lag, max_lag + 1):
        corr = float(np.mean(a[: n - lag] * b[lag:]))
        if corr > best_corr:
            best_corr, best_lag = corr, float(lag)
    if best_corr < 0.08:
        return default
    log.info("Estimated chat lag %.0fs (r=%.2f).", best_lag, best_corr)
    return best_lag


def infer_profile(timeline: Timeline) -> str:
    """Pick weights from the picture layout found by the visual pass."""
    layout = (timeline.meta.get("layout") or {}).get("kind", "screen")
    if layout == "camera":
        n = timeline.length
        motion = float(np.mean(timeline.get("motion", n))) if timeline.has("motion") else 0.0
        talk = (
            float(np.mean(timeline.get("speech_rate", n) > 0))
            if timeline.has("speech_rate")
            else 0.0
        )
        return "podcast" if motion < 0.02 and talk > 0.6 else "just_chatting"
    return "gaming"


def fuse(
    timeline: Timeline,
    *,
    profile: str = "auto",
    weights: dict[str, float] | None = None,
    chat_lag_s: float | None = None,
) -> FusionResult:
    """Combine every event signal on the timeline into one score per second."""
    n = timeline.length
    if n == 0:
        return FusionResult(score=np.zeros(0, dtype=np.float32))

    chosen = infer_profile(timeline) if profile == "auto" else profile
    table = dict(PROFILES.get(chosen, PROFILES["gaming"]))
    if weights:
        table.update(weights)

    available = [
        name
        for name in timeline.names()
        if name in table
        and table[name] > 0
        and name not in NON_EVENT
        and not name.startswith(NON_EVENT_PREFIXES)
        and timeline.has(name)
        and float(np.std(timeline.get(name, n))) > 1e-6
    ]

    # Reaction curve (no chat) used to time-align chat.
    zs: dict[str, np.ndarray] = {}
    for name in available:
        raw = timeline.get(name, n)
        if name.startswith("chat_"):
            raw = smooth(raw, 5)
        elif name in ("heatmap", "comments"):
            raw = smooth(raw, 9)
        else:
            raw = smooth(raw, 3, mode="max")
        zs[name] = robust_z(raw)
        # Unmeasured stretches (a backfill gap) carry no evidence either way.
        zs[name][~timeline.covered(name, n)] = 0.0

    lag = 0.0
    chat_names = [c for c in CHAT_SIGNALS if c in zs]
    if chat_names:
        if chat_lag_s is None:
            reaction_names = [
                k for k in zs if not k.startswith("chat_") and k not in ("heatmap", "comments")
            ]
            reaction = sum((np.maximum(zs[k], 0) for k in reaction_names), np.zeros(n, np.float32))
            chat_total = sum((np.maximum(zs[k], 0) for k in chat_names), np.zeros(n, np.float32))
            lag = estimate_chat_lag(chat_total, reaction)
        else:
            lag = float(chat_lag_s)
        shift = int(round(lag))
        for name in chat_names:
            if shift > 0:
                zs[name] = np.concatenate([zs[name][shift:], np.zeros(shift, np.float32)])

    contributions: dict[str, np.ndarray] = {}
    total_weight = 0.0
    score = np.zeros(n, dtype=np.float32)
    agree = np.zeros(n, dtype=np.float32)
    for name, z in zs.items():
        w = table[name]
        positive = np.maximum(z, 0.0)
        contributions[name] = (w * positive).astype(np.float32)
        score += contributions[name]
        agree += (z > AGREE_Z).astype(np.float32) * min(w, 1.0)
        total_weight += w

    if total_weight > 0:
        score /= total_weight
    score *= 1.0 + 0.15 * np.minimum(agree, 5.0)
    score = smooth(score, 5)

    log.info(
        "Fused %d signals (%s profile, chat lag %.0fs): %s",
        len(zs),
        chosen,
        lag,
        ", ".join(sorted(zs)),
    )
    return FusionResult(
        score=score.astype(np.float32),
        contributions=contributions,
        profile=chosen,
        chat_lag_s=lag,
        used=sorted(zs),
    )

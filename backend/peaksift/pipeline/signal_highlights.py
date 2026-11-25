"""Multimodal highlight detection — signals find the moments, an LLM (optionally) judges them.

This replaces "ask an LLM to read the whole transcript" as the default:

1. :mod:`peaksift.signals.fusion` turns the timeline into one score per second.
2. :mod:`peaksift.signals.candidates` picks peaks and builds clip windows.
3. If an LLM is available, each candidate is shown to it *with* the signal
   evidence and chat sample, and it returns a quality score, title and hook.
   The final score blends both. Without an LLM the signal score stands alone
   and titles come from the transcript — clipping still works fully offline
   and even on gameplay with no talking at all.
4. :mod:`peaksift.signals.select` applies the chosen view: best overall, best
   per hour, or best per chapter.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from pathlib import Path

from ..config import Settings
from ..db.models import Clip, new_id
from ..providers import DetectionConfig, LLMProvider
from ..providers.base import extract_json_object, load_prompt
from ..signals import select
from ..signals.candidates import Candidate, build_candidates
from ..signals.chat import ChatMessage, read_jsonl
from ..signals.fusion import FusionResult, fuse
from ..signals.timeline import Timeline
from .prepare import Silence
from .report import hms
from .transcript import Transcript

log = logging.getLogger(__name__)

SIGNAL_WEIGHT = 0.6
LLM_WEIGHT = 0.4
MAX_TRANSCRIPT_CHARS = 1800
MAX_CHAT_LINES = 25


def target_clip_count(settings: Settings, duration_s: float, chapters: int = 1) -> int:
    sel = settings.selection
    if sel.view == "per_hour":
        buckets = max(1, math.ceil(duration_s / (sel.bucket_minutes * 60)))
        return buckets * max(sel.per_bucket, sel.min_per_bucket)
    if sel.view == "per_chapter":
        return max(1, chapters) * sel.per_bucket
    return settings.clips.max_clips


async def detect(
    timeline: Timeline,
    transcript: Transcript | None,
    settings: Settings,
    *,
    job_id: str,
    silences: list[Silence] | None = None,
    provider: LLMProvider | None = None,
    detection_config: DetectionConfig | None = None,
    source_dir: Path | None = None,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    fusion: FusionResult | None = None,
) -> list[Clip]:
    """Find, judge and select clips. Returns ranked :class:`Clip` rows."""
    report = on_progress or (lambda _f: None)
    duration = duration_s or timeline.duration_s
    det = settings.detection

    fusion = fusion or fuse(
        timeline, profile=det.profile, weights=det.weights or None, chat_lag_s=det.chat_lag_s
    )
    timeline.meta["profile"] = fusion.profile
    timeline.meta["chat_lag_s"] = fusion.chat_lag_s

    chapters = select.find_chapters(timeline) if settings.selection.view == "per_chapter" else []
    target = target_clip_count(settings, duration, len(chapters) or 1)
    pool_size = max(8, math.ceil(target * det.oversample))
    candidates = build_candidates(
        fusion,
        transcript=transcript,
        silences=silences,
        max_candidates=pool_size,
        min_duration_s=settings.clips.min_duration_s,
        max_duration_s=settings.clips.max_duration_s,
    )
    report(0.3)
    if not candidates:
        log.warning("No candidate moments found.")
        return []
    log.info("Signal engine proposed %d candidates (target %d clips).", len(candidates), target)

    judged: dict[int, dict] = {}
    if provider is not None:
        chat = read_jsonl(source_dir / "chat.jsonl") if source_dir else []
        judged = await _rerank(
            candidates,
            transcript,
            chat,
            provider,
            detection_config or DetectionConfig(),
            fusion.chat_lag_s,
            report,
        )

    for index, candidate in enumerate(candidates):
        verdict = judged.get(index)
        if verdict is not None:
            candidate.score = int(
                round(SIGNAL_WEIGHT * candidate.score + LLM_WEIGHT * verdict["score"])
            )
    report(0.9)

    groups = _apply_view(candidates, timeline, settings, duration, chapters)
    clips: list[Clip] = []
    seen: set[int] = set()
    for label, members in groups:
        for candidate in members:
            key = id(candidate)
            if key in seen:
                continue
            seen.add(key)
            index = candidates.index(candidate)
            clips.append(_to_clip(candidate, judged.get(index), transcript, job_id, label, fusion))

    clips.sort(key=lambda c: c.score, reverse=True)
    for rank, clip in enumerate(clips, start=1):
        clip.rank = rank
    report(1.0)
    return clips


def _apply_view(
    candidates: list[Candidate],
    timeline: Timeline,
    settings: Settings,
    duration: float,
    chapters: list[tuple[float, float]],
) -> list[tuple[str, list[Candidate]]]:
    sel = settings.selection

    def blended(c: Candidate) -> float:
        # Blended 0-100 score first; the raw signal score breaks ties.
        return c.score + c.raw_score * 1e-3

    if sel.view == "per_hour":
        groups = select.per_bucket(
            candidates,
            duration_s=duration,
            bucket_s=sel.bucket_minutes * 60,
            per_bucket=sel.per_bucket,
            min_per_bucket=sel.min_per_bucket,
            key=blended,
        )
        return [(g.label, g.candidates) for g in groups]
    if sel.view == "per_chapter":
        groups = select.per_chapter(
            candidates, chapters or [(0.0, duration)], per_chapter=sel.per_bucket, key=blended
        )
        return [(g.label, g.candidates) for g in groups]
    return [("", select.overall(candidates, settings.clips.max_clips, key=blended))]


def _to_clip(
    candidate: Candidate,
    verdict: dict | None,
    transcript: Transcript | None,
    job_id: str,
    label: str,
    fusion: FusionResult,
) -> Clip:
    title = (verdict or {}).get("title", "").strip()
    if not title:
        title = _fallback_title(candidate, transcript)
    start_word, end_word = candidate.start_word, candidate.end_word
    if transcript is not None and transcript.words and end_word <= start_word:
        first, last = transcript.indices_in_range(candidate.start_s, candidate.end_s)
        start_word, end_word = first, max(first, last - 1)
    return Clip(
        id=new_id(),
        job_id=job_id,
        start_s=candidate.start_s,
        end_s=candidate.end_s,
        start_word=start_word,
        end_word=end_word,
        title=title[:80],
        hook=(verdict or {}).get("hook", "").strip()[:120],
        score=max(0, min(100, candidate.score)),
        reason=((verdict or {}).get("reason") or "").strip() or candidate.reason,
        peak_s=candidate.peak_s,
        group_label=label,
        signals={
            "drivers": [[name, round(share, 3)] for name, share in candidate.drivers],
            "signal_score": round(candidate.raw_score, 4),
            "llm_score": (verdict or {}).get("score"),
            "profile": fusion.profile,
            "chat_lag_s": fusion.chat_lag_s,
        },
    )


def _fallback_title(candidate: Candidate, transcript: Transcript | None) -> str:
    if transcript is not None and transcript.words:
        index = transcript.index_at_time(candidate.peak_s)
        first = max(0, index - 4)
        text = transcript.text_between(first, min(len(transcript.words) - 1, first + 9))
        if text:
            return text
    what = candidate.reason.split(",")[0].capitalize()
    return f"{what} at {hms(candidate.peak_s)}"


# --------------------------------------------------------------------------
# LLM re-ranking
# --------------------------------------------------------------------------


async def _rerank(
    candidates: list[Candidate],
    transcript: Transcript | None,
    chat: list[ChatMessage],
    provider: LLMProvider,
    config: DetectionConfig,
    chat_lag_s: float,
    report: Callable[[float], None],
) -> dict[int, dict]:
    system = load_prompt("rerank_v1")
    concurrency = 1 if provider.name == "ollama" else 3
    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    results: dict[int, dict] = {}

    async def judge(index: int, candidate: Candidate) -> None:
        nonlocal done
        user = _render_candidate(candidate, transcript, chat, chat_lag_s)
        async with semaphore:
            try:
                raw = await provider._complete(system, user, config)
                data = extract_json_object(raw)
                score = data.get("score", 50)
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 50.0
                if 0 < score <= 1:
                    score *= 100
                results[index] = {
                    "score": max(0, min(100, int(round(score)))),
                    "title": str(data.get("title") or ""),
                    "hook": str(data.get("hook") or ""),
                    "reason": str(data.get("reason") or ""),
                }
            except Exception as exc:
                log.warning("LLM re-rank failed for candidate at %.0fs: %s", candidate.peak_s, exc)
        done += 1
        report(0.3 + 0.6 * done / len(candidates))

    await asyncio.gather(*(judge(i, c) for i, c in enumerate(candidates)))
    log.info("LLM judged %d/%d candidates.", len(results), len(candidates))
    return results


def _render_candidate(
    candidate: Candidate, transcript: Transcript | None, chat: list[ChatMessage], chat_lag_s: float
) -> str:
    text = ""
    if transcript is not None and transcript.words:
        first, last = transcript.indices_in_range(candidate.start_s, candidate.end_s)
        text = transcript.text_between(first, max(first, last - 1))
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + " ..."

    lines: list[str] = []
    lo, hi = candidate.start_s + chat_lag_s, candidate.end_s + chat_lag_s
    window = [m for m in chat if lo <= m.t <= hi and m.text]
    if window:
        step = max(1, len(window) // MAX_CHAT_LINES)
        lines = [f"{m.author}: {m.text}"[:120] for m in window[::step][:MAX_CHAT_LINES]]

    return (
        f"Candidate clip {candidate.start_s:.0f}s-{candidate.end_s:.0f}s "
        f"({candidate.duration_s:.0f}s long), peak at {hms(candidate.peak_s)}.\n"
        f"Signals that spiked: {candidate.reason}.\n"
        f"Chat messages in window: {len(window)}.\n\n"
        f"Transcript:\n---\n{text or '(no speech)'}\n---\n\n"
        + ("Chat sample:\n---\n" + "\n".join(lines) + "\n---\n\n" if lines else "")
        + "Respond with ONLY the JSON object."
    )

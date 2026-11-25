"""Caption words from Gemini, timed by the local Whisper transcript.

Opt-in (``whisper.caption_source = "gemini"``). Only the exported clips'
seconds are sent, not the whole video: a 4-hour stream with ten 60-second
clips costs ten short requests, which fits Gemini's free tier. Detection and
cut points keep using the local transcript.

Gemini doesn't return reliable per-word times, so its text is aligned to the
Whisper words of the same seconds (:mod:`.align`) and inherits their timings.
Whenever Gemini fails, or its text doesn't line up with what Whisper heard
(a translation, a summary, a hallucination), the Whisper words are kept.
Results are cached beside the job, so re-rendering a clip is free.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from ..config import Settings, get_secret
from .align import align
from .ffmpeg import ffmpeg_path
from .transcript import Word

log = logging.getLogger(__name__)

#: Tried after the configured model when it is busy (503) or rate limited (429).
FALLBACK_MODELS = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash")
#: Below this share of Gemini words matching Whisper's, Gemini is not trusted.
MIN_MATCHED = 0.4
#: Gemini output whose word count is this far from Whisper's is not trusted.
MAX_LENGTH_RATIO = 2.0

PROMPTS = {
    "hinglish": (
        "Transcribe this audio verbatim. The speaker talks Hindi mixed with English. "
        "Write it in Hinglish: Hindi words in common Roman (Latin) spelling as people "
        "type them in chat, English words in normal English spelling. "
    ),
    "devanagari": (
        "Transcribe this audio verbatim. The speaker talks Hindi mixed with English. "
        "Write Hindi words in Devanagari and English words in normal English spelling. "
    ),
    "other": "Transcribe this audio verbatim in the language that is spoken. ",
}
PROMPT_RULES = (
    "Do not translate, summarise, correct grammar or add anything. Do not label "
    "speakers and leave out descriptions of sounds or music. Output only the words."
)


class CloudCaptionError(RuntimeError):
    pass


def clip_audio_wav(media: Path, start_s: float, end_s: float) -> bytes:
    """16 kHz mono WAV bytes of ``media`` between two times."""
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    if result.returncode != 0 or len(result.stdout) < 1000:
        raise CloudCaptionError(
            f"Could not cut the clip's audio: {result.stderr.decode(errors='replace')[-300:]}"
        )
    return result.stdout


def _prompt(settings: Settings, language: str) -> str:
    if language in ("hi", "ur"):
        key = "devanagari" if settings.whisper.hindi_script == "devanagari" else "hinglish"
    else:
        key = "other"
    return PROMPTS[key] + PROMPT_RULES


def gemini_transcribe(audio_wav: bytes, prompt: str, settings: Settings) -> tuple[str, str]:
    """Gemini's transcript of the audio, and the model that answered."""
    from google import genai
    from google.genai import types

    from ..providers.gemini_provider import normalize_model

    key = get_secret("gemini", settings)
    if not key:
        raise CloudCaptionError("No Gemini API key is set.")
    client = genai.Client(api_key=key)
    configured = normalize_model(settings.provider("gemini").model)
    models = [m for m in (configured, *FALLBACK_MODELS) if m]
    models = list(dict.fromkeys(models))
    errors: list[str] = []
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=audio_wav, mime_type="audio/wav"), prompt],
                config=types.GenerateContentConfig(temperature=0),
            )
        except Exception as exc:  # the SDK raises a variety of error types
            message = str(exc)
            errors.append(f"{model}: {message[:120]}")
            busy = any(code in message for code in ("503", "429", "UNAVAILABLE", "RESOURCE"))
            missing = "404" in message or "NOT_FOUND" in message
            if not (busy or missing):
                break
            time.sleep(2)
            continue
        return " ".join((response.text or "").split()), model
    raise CloudCaptionError("Gemini did not transcribe the clip: " + " | ".join(errors))


def merge(text: str, words: list[Word]) -> tuple[list[Word], float]:
    """Gemini's words with Whisper's timings, and the share that matched.

    Returns the Whisper words unchanged (share 0) when the texts disagree too
    much to trust Gemini's.
    """
    tokens = text.split()
    if not tokens or not words:
        return words, 0.0
    ratio = len(tokens) / len(words)
    if ratio > MAX_LENGTH_RATIO or ratio < 1 / MAX_LENGTH_RATIO:
        return words, 0.0
    aligned, matched = align(tokens, words)
    if matched < MIN_MATCHED:
        return words, matched
    for word in aligned:
        word.speaker = _speaker_at(words, (word.start + word.end) / 2)
    return aligned, matched


def _speaker_at(words: list[Word], t: float) -> str | None:
    best = min(words, key=lambda w: abs((w.start + w.end) / 2 - t))
    return best.speaker


def caption_words(
    media: Path,
    start_s: float,
    end_s: float,
    words: list[Word],
    settings: Settings,
    *,
    language: str = "",
    cache: Path | None = None,
    media_offset_s: float = 0.0,
    fetch: bool = True,
) -> list[Word]:
    """Words for a clip's captions: Gemini's when it can, else ``words``.

    ``words`` are the Whisper words in the clip, on the same clock as
    ``start_s``/``end_s``; that clock is ``media_offset_s`` ahead of the times
    in ``media`` (a live recording segment starts later than the session).
    With ``fetch`` off only a cached answer is used. Never raises.
    """
    if settings.whisper.caption_source != "gemini" or not words:
        return words

    key = f"{start_s:.2f}-{end_s:.2f}"
    if cache is not None and cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("range") == key:
                return [Word(t, s, e, sp) for t, s, e, sp in saved["words"]]
        except (ValueError, KeyError, TypeError):
            pass

    if not fetch:
        return words
    try:
        audio = clip_audio_wav(media, start_s - media_offset_s, end_s - media_offset_s)
        text, model = gemini_transcribe(audio, _prompt(settings, language), settings)
    except Exception as exc:
        log.warning("Gemini captions unavailable (%s); using Whisper's words.", exc)
        return words

    result, matched = merge(text, words)
    if result is words:
        log.warning(
            "Gemini's text didn't match the audio closely enough (%.0f%% matched); "
            "keeping Whisper's words.",
            matched * 100,
        )
        return words
    log.info(
        "Captions from %s: %d words, %.0f%% matched to Whisper.", model, len(result), matched * 100
    )

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {
                    "range": key,
                    "model": model,
                    "matched": round(matched, 3),
                    "text": text,
                    "words": [[w.text, w.start, w.end, w.speaker] for w in result],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return result

"""The words a clip's captions show: the user's edits, else Gemini's, else Whisper's.

One place decides this so the export, a re-render, the caption editor and the
subtitle downloads always agree on what the captions say.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import Settings
from ..db import store
from ..db.models import Clip
from . import cloud_captions
from .transcript import Transcript, Word

Origin = Literal["edited", "gemini", "whisper"]


def gemini_cache(work_dir: Path, clip_id: str) -> Path:
    return work_dir / "captions" / f"{clip_id}.gemini.json"


def automatic_words(clip: Clip, transcript: Transcript) -> list[Word]:
    """Whisper's words that overlap the clip, on the transcript's clock."""
    return [
        Word(w.text, w.start, w.end, w.speaker)
        for w in transcript.words
        if w.end > clip.start_s and w.start < clip.end_s
    ]


def edited_words(clip_id: str) -> list[Word] | None:
    """The user's caption edits for a clip, or None when it has none."""
    edit = store.get_clip_edit(clip_id)
    if edit is None or edit.edited_words is None:
        return None
    return [
        Word(str(w["text"]), float(w["start"]), float(w["end"]), w.get("speaker"))
        for w in edit.edited_words
    ]


def words_for_clip(
    clip: Clip,
    transcript: Transcript,
    settings: Settings,
    *,
    work_dir: Path,
    media: Path | None = None,
    media_offset_s: float = 0.0,
    fetch: bool = False,
) -> tuple[list[Word], Origin]:
    """Caption words for ``clip`` and where they came from.

    ``fetch`` allows a Gemini request (export time); without it only a cached
    Gemini answer is used, so the editor and downloads never cost a request.
    """
    edited = edited_words(clip.id)
    if edited is not None:
        return edited, "edited"
    automatic = automatic_words(clip, transcript)
    words = cloud_captions.caption_words(
        media or Path(),
        clip.start_s,
        clip.end_s,
        automatic,
        settings,
        language=transcript.language,
        cache=gemini_cache(work_dir, clip.id),
        media_offset_s=media_offset_s,
        fetch=fetch and media is not None,
    )
    return words, ("whisper" if words is automatic else "gemini")

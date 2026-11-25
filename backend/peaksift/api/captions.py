"""Caption editor and subtitle downloads for the built-in web UI.

* ``GET/PUT/DELETE /api/ui/clips/{id}/captions`` — the words a clip's captions
  show (edited, Gemini's or Whisper's), saving edits, and going back to the
  automatic words. Edits are used by every later re-render.
* ``GET /api/ui/clips/{id}/subtitles.srt|vtt`` — plain subtitle files with the
  same words, timed from the start of the clip, for uploading with the video.
* ``POST /api/ui/jobs/{id}/replace`` — fix a word everywhere in a job at once
  (a name or game term the model keeps mishearing).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..db import store
from ..db.models import Clip, ClipEdit
from ..pipeline import captions as captions_module
from ..pipeline.clip_words import Origin, words_for_clip
from ..pipeline.runner import JobWorkspace
from ..pipeline.transcript import Transcript, Word
from .schemas import WordOut

router = APIRouter(prefix="/api/ui", tags=["ui"])

MAX_WORDS = 20_000
_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


class CaptionsIn(BaseModel):
    words: list[WordOut] = Field(max_length=MAX_WORDS)


class ReplaceIn(BaseModel):
    find: str = Field(min_length=1, max_length=200)
    replace: str = Field(max_length=200)


def _clip_or_404(clip_id: str) -> Clip:
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return clip


def _context(clip: Clip):
    from ..jobs.queue import _job_settings

    job = store.get_job(clip.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="The clip's job is missing.")
    workspace = JobWorkspace(clip.job_id)
    transcript = (
        Transcript.load(workspace.transcript) if workspace.transcript.exists() else Transcript()
    )
    return _job_settings(job), workspace, transcript


def _words(clip: Clip) -> tuple[list[Word], Origin]:
    settings, workspace, transcript = _context(clip)
    return words_for_clip(clip, transcript, settings, work_dir=workspace.root)


def _payload(clip: Clip) -> dict:
    words, origin = _words(clip)
    exports = store.list_exports(clip.id)
    video = f"/api/exports/{exports[0].id}/download" if exports else None
    return {
        "clip": {
            "id": clip.id,
            "job_id": clip.job_id,
            "rank": clip.rank,
            "title": clip.title,
            "start_s": clip.start_s,
            "end_s": clip.end_s,
        },
        "origin": origin,
        "words": [
            {"text": w.text, "start": w.start, "end": w.end, "speaker": w.speaker} for w in words
        ],
        "video_url": video,
        "subtitles": {
            fmt: f"/api/ui/clips/{clip.id}/subtitles.{fmt}"
            for fmt in captions_module.SUBTITLE_FORMATS
        },
    }


def clean_words(words: list[WordOut]) -> list[dict]:
    """Edited words as stored: empty ones dropped (deleted), times sane and ordered."""
    out: list[dict] = []
    for w in sorted(words, key=lambda w: w.start):
        text = " ".join(w.text.split())
        if not text:
            continue
        start = max(0.0, float(w.start))
        end = max(start, float(w.end))
        out.append({"text": text, "start": start, "end": end, "speaker": w.speaker})
    return out


def _save(clip_id: str, words: list[dict] | None) -> None:
    existing = store.get_clip_edit(clip_id)
    store.upsert_clip_edit(
        ClipEdit(
            clip_id=clip_id,
            edited_words=words,
            caption_style=existing.caption_style if existing else "bold_pop",
            ratio=existing.ratio if existing else "9:16",
        )
    )


@router.get("/clips/{clip_id}/captions")
async def get_captions(clip_id: str) -> dict:
    clip = await asyncio.to_thread(_clip_or_404, clip_id)
    return await asyncio.to_thread(_payload, clip)


@router.put("/clips/{clip_id}/captions")
async def put_captions(clip_id: str, payload: CaptionsIn) -> dict:
    clip = await asyncio.to_thread(_clip_or_404, clip_id)
    await asyncio.to_thread(_save, clip_id, clean_words(payload.words))
    return await asyncio.to_thread(_payload, clip)


@router.delete("/clips/{clip_id}/captions")
async def reset_captions(clip_id: str) -> dict:
    """Drop the edits; the captions go back to the automatic words."""
    clip = await asyncio.to_thread(_clip_or_404, clip_id)
    await asyncio.to_thread(_save, clip_id, None)
    return await asyncio.to_thread(_payload, clip)


@router.get("/clips/{clip_id}/subtitles.{fmt}")
async def subtitles(clip_id: str, fmt: str) -> Response:
    if fmt not in captions_module.SUBTITLE_FORMATS:
        raise HTTPException(status_code=404, detail="Use .srt or .vtt.")
    clip = await asyncio.to_thread(_clip_or_404, clip_id)
    words, _origin = await asyncio.to_thread(_words, clip)
    text = captions_module.subtitle_text(words, fmt, time_offset_s=clip.start_s)
    exports = await asyncio.to_thread(store.list_exports, clip.id)
    stem = Path(exports[0].path).stem if exports else f"clip-{clip.rank:02d}"
    media = {"srt": "application/x-subrip", "vtt": "text/vtt"}[fmt]
    return Response(
        content=text.encode("utf-8"),
        media_type=f"{media}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"'},
    )


def _core(text: str) -> tuple[str, str, str]:
    lead, core, trail = _EDGE.match(text).groups()  # type: ignore[union-attr]
    return lead, core, trail


def replace_in_words(words: list[Word], find: str, replacement: str) -> tuple[list[Word], int]:
    """Replace a word or phrase (ignoring case and punctuation around words).

    Punctuation around the matched words is kept ("phokas," -> "focus,"). A
    phrase replaced by the same number of words keeps each word's timing;
    otherwise the replacement spans the matched words' time.
    """
    target = [t.casefold() for t in find.split()]
    new = replacement.split()
    if not target:
        return words, 0
    out: list[Word] = []
    count = 0
    i = 0
    while i < len(words):
        window = words[i : i + len(target)]
        cores = [_core(w.text)[1].casefold() for w in window]
        if len(window) == len(target) and cores == target:
            lead = _core(window[0].text)[0]
            trail = _core(window[-1].text)[2]
            if len(new) == len(window):
                for k, (old, text) in enumerate(zip(window, new, strict=True)):
                    pre = lead if k == 0 else _core(old.text)[0]
                    post = trail if k == len(new) - 1 else _core(old.text)[2]
                    out.append(Word(pre + text + post, old.start, old.end, old.speaker))
            elif new:
                out.append(
                    Word(
                        lead + " ".join(new) + trail,
                        window[0].start,
                        window[-1].end,
                        window[0].speaker,
                    )
                )
            count += 1
            i += len(window)
            continue
        out.append(words[i])
        i += 1
    return out, count


@router.post("/jobs/{job_id}/replace")
async def replace_everywhere(job_id: str, payload: ReplaceIn) -> dict:
    """Fix a word or phrase in every clip of a job; returns how many were changed."""
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    def run() -> dict:
        clips = store.list_clips(job_id)
        total, touched = 0, []
        for clip in clips:
            words, _origin = _words(clip)
            changed, count = replace_in_words(words, payload.find, payload.replace)
            if count:
                _save(
                    clip.id,
                    [
                        {"text": w.text, "start": w.start, "end": w.end, "speaker": w.speaker}
                        for w in changed
                    ],
                )
                total += count
                touched.append(clip.rank)
        return {"replaced": total, "clips": sorted(touched)}

    return await asyncio.to_thread(run)

"""Multi-signal highlight engine.

Builds a per-second :class:`~.timeline.Timeline` from every available signal —
audio, speech, visual activity, faces and facial reactions, live chat, and
platform hints like YouTube's "most replayed" heatmap — then fuses them into a
single score curve whose peaks become clip candidates.

The same functions serve a finished file (processed in fixed-size chunks so a
five-hour VOD never loads at once) and a live stream (chunks arrive as the
recorder writes them).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..pipeline.transcript import Transcript
from .audio import audio_features, speech_features
from .chat import ChatMessage, chat_features, read_jsonl
from .timeline import Timeline
from .visual import FaceBox, infer_layout, visual_features

log = logging.getLogger(__name__)

__all__ = ["Timeline", "build_timeline", "add_chunk_visual", "refresh_derived", "finalize_layout"]

#: Visual pass chunk length for finished files.
VISUAL_CHUNK_S = 600
#: Sources longer than this use keyframe-only visual decoding unless overridden.
KEYFRAME_MODE_AFTER_S = 2 * 3600


def build_timeline(
    *,
    video: Path | None,
    audio: Path,
    transcript: Transcript | None,
    duration_s: float,
    source_dir: Path | None = None,
    sample_fps: float = 2.0,
    face_fps: float = 1.0,
    visual_mode: str = "auto",
    faces: bool = True,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Timeline:
    """Measure every signal for a finished source and return the timeline."""
    timeline = Timeline()
    length = max(1, int(np.ceil(duration_s)))
    report = on_progress or (lambda _f: None)

    report(0.0)
    timeline.put_all(0, audio_features(audio))
    report(0.1)

    if video is not None:
        keyframes = visual_mode == "keyframes" or (
            visual_mode == "auto" and duration_s > KEYFRAME_MODE_AFTER_S
        )
        boxes: list[FaceBox] = []
        smalls: list[FaceBox] = []
        samples = 0
        start = 0.0
        while start < duration_s:
            if cancelled is not None and cancelled():
                break
            span = min(VISUAL_CHUNK_S, duration_s - start)
            result = visual_features(
                video,
                start_s=start,
                duration_s=span,
                sample_fps=sample_fps,
                face_fps=face_fps,
                keyframes_only=keyframes,
                faces=faces,
                cancelled=cancelled,
            )
            if result.signals:
                timeline.put_all(int(start), result.signals)
                boxes.extend(result.face_boxes)
                smalls.extend(result.small_faces)
                samples += int(np.ceil(span * min(face_fps, 1.0 if keyframes else sample_fps)))
                timeline.meta["visual_mode"] = result.mode
                timeline.meta["source_size"] = [result.width, result.height]
            start += span
            report(0.1 + 0.8 * min(1.0, start / max(duration_s, 1.0)))
        timeline.meta["face_samples"] = samples
        timeline.meta["face_boxes"] = [b.__dict__ for b in boxes[-20000:]]
        timeline.meta["small_faces"] = _pack(smalls)[-MAX_SMALL_FACE_ROWS:]
        finalize_layout(timeline)

    timeline.extend_to(length)
    refresh_derived(timeline, transcript=transcript, source_dir=source_dir)
    report(1.0)
    return timeline


def add_chunk_visual(
    timeline: Timeline,
    video: Path,
    *,
    offset_s: float,
    duration_s: float | None = None,
    sample_fps: float = 2.0,
    face_fps: float = 1.0,
    faces: bool = True,
) -> None:
    """Live path: measure one recorded segment and place it at ``offset_s``."""
    result = visual_features(
        video,
        start_s=0.0,
        duration_s=duration_s,
        sample_fps=sample_fps,
        face_fps=face_fps,
        faces=faces,
    )
    if not result.signals:
        return
    timeline.put_all(int(round(offset_s)), result.signals)
    boxes = timeline.meta.setdefault("face_boxes", [])
    for box in result.face_boxes:
        data = dict(box.__dict__)
        data["t"] = data["t"] + offset_s
        boxes.append(data)
    del boxes[:-20000]
    packed = timeline.meta.setdefault("small_faces", [])
    for row in _pack(result.small_faces):
        row[0] = round(row[0] + offset_s, 2)
        packed.append(row)
    del packed[:-MAX_SMALL_FACE_ROWS]
    span = duration_s if duration_s is not None else result.signals["motion"].size
    timeline.meta["face_samples"] = int(timeline.meta.get("face_samples", 0)) + int(
        np.ceil(span * min(face_fps, sample_fps))
    )
    timeline.meta["source_size"] = [result.width, result.height]
    timeline.meta["visual_mode"] = result.mode


#: Small faces kept in the timeline (about 5 h at 1 sample/s, 3 faces each).
MAX_SMALL_FACE_ROWS = 60_000


def _pack(faces: list[FaceBox]) -> list[list[float]]:
    """Compact ``[t, cx, cy, w, h]`` rows for the timeline's JSON metadata."""
    return [
        [round(f.t, 2), round(f.cx, 4), round(f.cy, 4), round(f.w, 4), round(f.h, 4)] for f in faces
    ]


def small_faces(timeline: Timeline) -> list[FaceBox] | None:
    """The timeline's small faces, or None for one analysed before they were kept."""
    rows = timeline.meta.get("small_faces")
    if rows is None:
        return None
    return [FaceBox(t=r[0], cx=r[1], cy=r[2], w=r[3], h=r[4]) for r in rows]


def finalize_layout(timeline: Timeline) -> None:
    boxes = [FaceBox(**b) for b in timeline.meta.get("face_boxes", [])]
    layout = infer_layout(
        boxes, int(timeline.meta.get("face_samples", 0)), small_faces=small_faces(timeline)
    )
    timeline.meta["layout"] = {
        "kind": layout.kind,
        "facecam": list(layout.facecam) if layout.facecam else None,
        "facecams": [list(region) for region in layout.facecams],
        "face_ratio": layout.face_ratio,
        "median_face_w": layout.median_face_w,
    }


def refresh_derived(
    timeline: Timeline,
    *,
    transcript: Transcript | None,
    source_dir: Path | None,
    chat: list[ChatMessage] | None = None,
) -> None:
    """(Re)compute the signals derived from words, chat and platform hints.

    Cheap enough to rerun after every live chunk, which is how chat that
    arrived late still lands on the right seconds.
    """
    length = timeline.length
    if length == 0:
        return
    if transcript is not None and transcript.words:
        timeline.put_all(0, speech_features(transcript, length))

    if chat is None and source_dir is not None:
        chat = read_jsonl(source_dir / "chat.jsonl")
    if chat:
        timeline.put_all(0, chat_features(chat, length))
        timeline.meta["chat_messages"] = len(chat)

    if source_dir is not None:
        heat = _heatmap(source_dir / "heatmap.json", length)
        if heat is not None:
            timeline.put("heatmap", 0, heat)
        comments = _comment_mentions(source_dir / "comments.json", length)
        if comments is not None:
            timeline.put("comments", 0, comments)


def _heatmap(path: Path, length: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    out = np.zeros(length, dtype=np.float32)
    for entry in entries or []:
        a = int(max(0, entry.get("start_time", 0)))
        b = int(min(length, np.ceil(entry.get("end_time", a + 1))))
        if b > a:
            out[a:b] = max(0.0, float(entry.get("value", 0.0)))
    return out if out.any() else None


def _comment_mentions(path: Path, length: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        stamps = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    out = np.zeros(length, dtype=np.float32)
    for t in stamps or []:
        # People type the moment roughly; spread each mention over +-5 s.
        centre = int(t)
        lo, hi = max(0, centre - 5), min(length, centre + 6)
        if hi > lo:
            out[lo:hi] += 1.0
    return out if out.any() else None

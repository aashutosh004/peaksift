"""Reframe stage — 16:9 source to a 9:16 crop path that follows the speaker.

Pipeline for one clip:

1. Split into shots. Crop paths never interpolate across a cut.
2. Sample face landmarks across the clip.
3. Link observations into per-person tracks.
4. Map diarized speakers onto tracks once, then let turn boundaries drive framing.
5. Per shot, pick a strategy — TRACK, WIDE, or GENERAL.
6. Build a raw crop path, smooth it, and emit segments.

The quality bar is "never jarring": no visible jitter, no cut-off faces, the
speaker on screen for essentially all of their speaking time. Every default here
is biased toward stillness — a locked frame that is slightly off-centre beats a
frame that is always correct and always moving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .. import ffmpeg
from ..transcript import Transcript
from .croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Strategy,
    centre_crop,
    target_crop_size,
)
from .faces import FaceDetectionUnavailable, FaceObservation, sample_faces
from .scenes import MIN_SHOT_S, Shot, detect_shots
from .smoothing import SmoothingConfig, smooth_series
from .speaker import assign_speakers, map_tracks_to_speakers
from .tracker import FaceTrack, build_tracks

log = logging.getLogger(__name__)

__all__ = [
    "CropPath",
    "CropSegment",
    "ReframeConfig",
    "Strategy",
    "build_crop_path",
    "build_stack_path",
]

#: Share of the vertical frame given to the facecam in the gaming layout.
STACK_SPLIT = 0.35

#: Where the eye line sits within the crop, as a fraction of crop height.
#: Portrait convention puts the eyes near the upper third; 0.38 reads as
#: composed without leaving the chin tight to the bottom edge.
EYE_LINE_RATIO = 0.38

#: A face wider than this fraction of the crop is already a close-up; don't
#: track it, lock the frame.
LOCK_IF_FACE_WIDER_THAN = 0.55

#: If detected faces span more than this fraction of the crop width, no crop can
#: hold them and we fall back to a fitted (blurred-background) frame.
FIT_IF_SPREAD_EXCEEDS = 0.95

#: Slow push-in applied to shots with no face at all, to stop a static wide from
#: feeling like a still image. Off by default — see the note in export._zoom_filter.
GENERAL_ZOOM = 0.0


@dataclass
class ReframeConfig:
    aspect_w: int = 9
    aspect_h: int = 16
    sample_fps: float = 5.0
    smoothing: SmoothingConfig | None = None
    #: Skip face detection entirely and centre-crop everything.
    centre_only: bool = False
    #: Two people clearly on screen: stack them (left person on top) instead of
    #: tracking one speaker or framing both wide. Vertical outputs only.
    two_person_split: bool = True


def build_crop_path(
    video: Path,
    *,
    start_s: float,
    end_s: float,
    transcript: Transcript | None = None,
    config: ReframeConfig | None = None,
) -> CropPath:
    """Compute the crop path for one clip.

    Falls back to a centred crop whenever face detection is unavailable or finds
    nothing — a centre crop is an acceptable result, an exception is not.

    Preconditions:
        video exists; 0 <= start_s < end_s <= source duration.
    """
    config = config or ReframeConfig()
    duration = end_s - start_s

    info = ffmpeg.probe(video)
    if not info.has_video or not info.width or not info.height:
        raise ValueError(f"{video.name} has no video stream to reframe.")

    source_w, source_h = info.width, info.height
    crop_w, crop_h = target_crop_size(source_w, source_h, config.aspect_w, config.aspect_h)

    if config.centre_only:
        return centre_crop(
            source_w, source_h, duration, aspect_w=config.aspect_w, aspect_h=config.aspect_h
        )

    try:
        observations = sample_faces(
            video, start_s=start_s, end_s=end_s, sample_fps=config.sample_fps
        )
    except FaceDetectionUnavailable as exc:
        log.warning("Face detection unavailable (%s); using a centre crop.", exc)
        return centre_crop(
            source_w,
            source_h,
            duration,
            aspect_w=config.aspect_w,
            aspect_h=config.aspect_h,
            zoom=GENERAL_ZOOM,
        )

    if not observations:
        log.info("No faces detected in %s; using a centre crop.", video.name)
        return centre_crop(
            source_w,
            source_h,
            duration,
            aspect_w=config.aspect_w,
            aspect_h=config.aspect_h,
            zoom=GENERAL_ZOOM,
        )

    tracks = build_tracks(observations)
    shots = split_at_track_handoffs(detect_shots(video, start_s=start_s, end_s=end_s), tracks)

    turns = _clip_relative_turns(transcript, start_s, end_s)
    speaker_map = map_tracks_to_speakers(tracks, turns) if turns else {}

    segments: list[CropSegment] = []
    for shot in shots:
        segments.extend(
            _segments_for_shot(
                shot,
                tracks=tracks,
                turns=turns,
                speaker_map=speaker_map,
                source_w=source_w,
                source_h=source_h,
                crop_w=crop_w,
                crop_h=crop_h,
                config=config,
            )
        )

    if not segments:
        return centre_crop(
            source_w, source_h, duration, aspect_w=config.aspect_w, aspect_h=config.aspect_h
        )

    segments = _merge_split_runs(segments, source_w)

    # Guarantee the segments tile the clip exactly; a gap or overlap would
    # desync the concatenated render from the audio.
    segments[0].start_s = 0.0
    segments[-1].end_s = duration
    for index in range(len(segments) - 1):
        segments[index].end_s = segments[index + 1].start_s

    segments = [s for s in segments if s.duration_s > 0.01]

    log.info(
        "Reframed %s into %d segment(s): %s",
        video.name,
        len(segments),
        ", ".join(s.strategy.value for s in segments),
    )
    return CropPath(source_width=source_w, source_height=source_h, segments=segments)


def build_stack_path(
    source_w: int,
    source_h: int,
    duration_s: float,
    facecam: tuple[float, float, float, float],
    *,
    gameplay: tuple[float, float, float, float] | None = None,
    out_w: int = 1080,
    out_h: int = 1920,
    split: float = STACK_SPLIT,
    facecam_position: str = "top",
    exact: bool = False,
) -> CropPath:
    """Gaming layout: a facecam panel and a gameplay panel stacked vertically.

    ``facecam`` and ``gameplay`` are normalised ``(x, y, w, h)`` boxes. Without a
    gameplay box, the gameplay panel is a full-height band from the centre of
    the frame. ``split`` is the share of the output height the facecam gets;
    ``facecam_position`` puts it above or below the gameplay.

    ``exact`` is for boxes the user drew: each panel is cropped *inside* its box
    (so nothing outside what they selected appears). Auto-detected facecams use
    the default, which grows the box around the face to fill the panel.
    """
    cam_h = int(round(out_h * split / 2)) * 2
    game_h = out_h - cam_h

    fx, fy, fw, fh = facecam
    cam_px = (fx * source_w, fy * source_h, fw * source_w, fh * source_h)
    fit = _cover_box if exact else _fit_box
    cam_box = fit(*cam_px, aspect=out_w / cam_h, source_w=source_w, source_h=source_h)

    if gameplay is not None:
        gx, gy, gw_n, gh_n = gameplay
        game_box = _cover_box(
            gx * source_w,
            gy * source_h,
            gw_n * source_w,
            gh_n * source_h,
            aspect=out_w / game_h,
            source_w=source_w,
            source_h=source_h,
        )
    else:
        gw = min(source_w, source_h * out_w / game_h)
        game_box = [round((source_w - gw) / 2) // 2 * 2, 0, int(gw) // 2 * 2, source_h // 2 * 2]

    if facecam_position == "bottom":
        panels = {"top": game_box, "bottom": cam_box, "split": game_h / out_h}
    else:
        panels = {"top": cam_box, "bottom": game_box, "split": cam_h / out_h}

    segment = CropSegment(
        start_s=0.0,
        end_s=duration_s,
        width=int(game_box[2]),
        height=int(game_box[3]),
        keyframes=[CropKeyframe(t=0.0, x=float(game_box[0]), y=float(game_box[1]))],
        strategy=Strategy.STACK,
        panels=panels,
    )
    return CropPath(source_width=source_w, source_height=source_h, segments=[segment])


def _cover_box(
    x: float, y: float, w: float, h: float, *, aspect: float, source_w: int, source_h: int
) -> list[int]:
    """Shrink a box around its centre to ``aspect`` so the crop stays inside it."""
    x, y = max(0.0, x), max(0.0, y)
    w, h = min(w, source_w - x), min(h, source_h - y)
    cx, cy = x + w / 2, y + h / 2
    if w / max(h, 1) > aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = max(16.0, w), max(16.0, h)
    left = min(max(0.0, cx - w / 2), source_w - w)
    top = min(max(0.0, cy - h / 2), source_h - h)
    return [int(left) // 2 * 2, int(top) // 2 * 2, int(w) // 2 * 2, int(h) // 2 * 2]


def _fit_box(
    x: float, y: float, w: float, h: float, *, aspect: float, source_w: int, source_h: int
) -> list[int]:
    """Grow a box around its centre to ``aspect`` (w/h), then clamp into the frame."""
    cx, cy = x + w / 2, y + h / 2
    if w / max(h, 1) < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, source_w), min(h, source_h)
    if w / max(h, 1) > aspect:
        w = h * aspect
    else:
        h = w / aspect
    left = min(max(0.0, cx - w / 2), source_w - w)
    top = min(max(0.0, cy - h / 2), source_h - h)
    # Even dimensions keep yuv420p happy through crop and scale.
    return [int(left), int(top), int(w) // 2 * 2, int(h) // 2 * 2]


def _clip_relative_turns(
    transcript: Transcript | None, start_s: float, end_s: float
) -> list[tuple[str, float, float]]:
    """Rebase diarization turns onto the clip's timeline, dropping non-overlapping ones."""
    if transcript is None or not transcript.has_diarization:
        return []

    turns: list[tuple[str, float, float]] = []
    for speaker, turn_start, turn_end in transcript.speaker_turns():
        overlap_start = max(turn_start, start_s)
        overlap_end = min(turn_end, end_s)
        if overlap_end > overlap_start:
            turns.append((speaker, overlap_start - start_s, overlap_end - start_s))
    return turns


def _segments_for_shot(
    shot: Shot,
    *,
    tracks: list[FaceTrack],
    turns: list[tuple[str, float, float]],
    speaker_map: dict[str, FaceTrack],
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    config: ReframeConfig,
) -> list[CropSegment]:
    assignments = assign_speakers(
        tracks,
        shot_start_s=shot.start_s,
        shot_end_s=shot.end_s,
        turns=turns,
        speaker_map=speaker_map,
    )

    segments: list[CropSegment] = []
    for assignment in assignments:
        visible = [
            t for t in tracks if t.observations_between(assignment.start_s, assignment.end_s)
        ]

        pair = (
            _split_pair(visible, assignment.start_s, assignment.end_s, source_w, config)
            if config.two_person_split and config.aspect_h > config.aspect_w
            else None
        )
        if pair is not None:
            segments.append(
                _split_segment(
                    pair,
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    config=config,
                )
            )
            continue

        if assignment.track is not None:
            segments.append(
                _track_segment(
                    assignment.track,
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    config=config,
                )
            )
        elif visible:
            segments.append(
                _wide_segment(
                    visible,
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
            )
        else:
            segments.append(
                _general_segment(
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
            )

    return segments


#: A face track ending and another starting within this gap reads as a cut.
HANDOFF_MAX_GAP_S = 1.5
HANDOFF_MIN_TRACK_S = 1.0


def split_at_track_handoffs(shots: list[Shot], tracks: list[FaceTrack]) -> list[Shot]:
    """Add cuts where one face disappears and another appears right away.

    Content-based cut detection misses cuts between similar-looking angles —
    two people filmed against the same dark backdrop is the classic podcast
    case. If no face track spans the moment, one person vanishing and another
    appearing within a second is a camera cut, and framing must not carry
    across it.
    """
    long_tracks = sorted(
        (t for t in tracks if t.duration_s >= HANDOFF_MIN_TRACK_S), key=lambda t: t.start_s
    )
    cuts: list[float] = []
    for previous, following in zip(long_tracks, long_tracks[1:], strict=False):
        gap = following.start_s - previous.end_s
        if not 0 <= gap <= HANDOFF_MAX_GAP_S:
            continue
        cut = previous.end_s + gap / 2
        spanning = any(
            t.start_s < cut - 0.5 and t.end_s > cut + 0.5
            for t in long_tracks
            if t is not previous and t is not following
        )
        if not spanning:
            cuts.append(cut)
    if not cuts:
        return shots

    result: list[Shot] = []
    for shot in shots:
        start = shot.start_s
        for cut in sorted(c for c in cuts if shot.start_s < c < shot.end_s):
            if cut - start >= MIN_SHOT_S and shot.end_s - cut >= MIN_SHOT_S:
                result.append(Shot(start, cut))
                start = cut
        result.append(Shot(start, shot.end_s))
    if len(result) > len(shots):
        log.info("Added %d cut(s) from face hand-offs.", len(result) - len(shots))
    return result


def _merge_split_runs(segments: list[CropSegment], source_w: int) -> list[CropSegment]:
    """Join back-to-back split segments that frame the same two people.

    Speaker turns cut a shot into many assignments; without this, a static
    two-person split would be re-cut at every turn for no visible reason.
    """
    merged: list[CropSegment] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.panels
            and segment.panels
            and previous.strategy == Strategy.STACK == segment.strategy
            and _panels_close(previous.panels, segment.panels, source_w)
        ):
            previous.end_s = segment.end_s
            continue
        merged.append(segment)
    return merged


def _panels_close(a: dict, b: dict, source_w: int, tolerance: float = 0.04) -> bool:
    limit = tolerance * source_w
    return all(
        abs(float(x) - float(y)) <= limit
        for key in ("top", "bottom")
        for x, y in zip(a[key], b[key], strict=True)
    )


#: Two faces must be at least this far apart (share of frame width) to split.
SPLIT_MIN_SEPARATION = 0.18
#: Ignore faces narrower than this share of the frame (background people).
SPLIT_MIN_FACE_W = 0.035
#: A face must be seen in this share of the samples in the window to count.
SPLIT_MIN_COVERAGE = 0.5
#: The smaller face must be at least this fraction of the larger one's area.
SPLIT_MIN_AREA_RATIO = 0.25


def _split_pair(
    visible: list[FaceTrack],
    start_s: float,
    end_s: float,
    source_w: int,
    config: ReframeConfig,
) -> tuple[FaceTrack, FaceTrack] | None:
    """Two persistent, well-separated, similar-sized faces — or None.

    Guards against the classic auto-split failures: a single speaker cut in half
    (needs two distinct faces), a background extra getting a panel (size and
    coverage checks), and two people sitting close enough to share one crop
    (separation check).
    """
    duration = max(end_s - start_s, 1e-6)
    expected = max(1.0, duration * config.sample_fps)
    candidates: list[tuple[FaceTrack, float, float]] = []
    for track in visible:
        obs = track.observations_between(start_s, end_s)
        if len(obs) < max(3, SPLIT_MIN_COVERAGE * expected):
            continue
        width = sum(o.width for o in obs) / len(obs)
        if width < SPLIT_MIN_FACE_W * source_w:
            continue
        candidates.append((track, width, sum(o.cx for o in obs) / len(obs)))
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    (a, wa, xa), (b, wb, xb) = candidates[0], candidates[1]
    if (wb * wb) / (wa * wa) < SPLIT_MIN_AREA_RATIO:
        return None
    if abs(xa - xb) < SPLIT_MIN_SEPARATION * source_w:
        return None
    return (a, b) if xa <= xb else (b, a)


def _split_segment(
    pair: tuple[FaceTrack, FaceTrack],
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    config: ReframeConfig,
) -> CropSegment:
    """Left person in the top half, right person in the bottom half."""
    # Each panel is half the output height: 9:8 for a 9:16 output.
    aspect = (config.aspect_w * 2) / config.aspect_h
    stats = []
    for track in pair:
        obs = track.observations_between(start_s, end_s)
        stats.append(
            (
                sum(o.cx for o in obs) / len(obs),
                sum(o.eye_y for o in obs) / len(obs),
                sum(o.width for o in obs) / len(obs),
            )
        )
    separation = abs(stats[0][0] - stats[1][0])
    boxes = []
    for cx, eye_y, face_w in stats:
        # Head-and-shoulders: ~3 face widths, but never wide enough to pull the
        # other person into this panel.
        box_w = max(face_w * 1.6, min(face_w * 3.0, separation * 0.95, source_w / 2))
        box_h = box_w / aspect
        boxes.append(
            _fit_box(
                cx - box_w / 2,
                eye_y - EYE_LINE_RATIO * box_h,
                box_w,
                box_h,
                aspect=aspect,
                source_w=source_w,
                source_h=source_h,
            )
        )
    top, bottom = boxes
    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=int(top[2]),
        height=int(top[3]),
        keyframes=[CropKeyframe(t=start_s, x=float(top[0]), y=float(top[1]))],
        strategy=Strategy.STACK,
        panels={"top": top, "bottom": bottom, "split": 0.5},
    )


def _track_segment(
    track: FaceTrack,
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    config: ReframeConfig,
) -> CropSegment:
    """Frame a single subject, tracking them or locking on if they barely move."""
    observations = track.observations_between(start_s, end_s)
    if not observations:
        return _general_segment(
            start_s=start_s,
            end_s=end_s,
            source_w=source_w,
            source_h=source_h,
            crop_w=crop_w,
            crop_h=crop_h,
        )

    raw = [
        (
            o.t,
            _clamp(o.cx - crop_w / 2, 0, source_w - crop_w),
            _clamp(o.eye_y - EYE_LINE_RATIO * crop_h, 0, source_h - crop_h),
        )
        for o in observations
    ]

    mean_face_width = sum(o.width for o in observations) / len(observations)
    spread_x = max(x for _, x, _ in raw) - min(x for _, x, _ in raw)

    # A close-up, or a subject who barely moves, is better locked than tracked.
    should_lock = (
        mean_face_width > crop_w * LOCK_IF_FACE_WIDER_THAN
        or spread_x < (config.smoothing or SmoothingConfig()).dead_zone_px
    )

    if should_lock or len(raw) < 3:
        x = sum(item[1] for item in raw) / len(raw)
        y = sum(item[2] for item in raw) / len(raw)
        keyframes = [CropKeyframe(t=start_s, x=x, y=y)]
    else:
        smoothing = config.smoothing or SmoothingConfig()
        xs = smooth_series([(t, x) for t, x, _ in raw], smoothing)
        ys = smooth_series([(t, y) for t, _, y in raw], smoothing)
        keyframes = [CropKeyframe(t=t, x=x, y=y) for (t, x), (_, y) in zip(xs, ys, strict=True)]
        # Anchor the ends so the expression covers the whole segment.
        if keyframes[0].t > start_s:
            keyframes.insert(0, CropKeyframe(start_s, keyframes[0].x, keyframes[0].y))
        if keyframes[-1].t < end_s:
            keyframes.append(CropKeyframe(end_s, keyframes[-1].x, keyframes[-1].y))

    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=keyframes,
        strategy=Strategy.TRACK,
    )


def _wide_segment(
    visible: list[FaceTrack],
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
) -> CropSegment:
    """Hold several faces at once — locked, never panning.

    Panning between two people who are both on screen is the most obviously
    robotic thing an auto-reframer does. If they fit, lock on their centroid; if
    they don't, fit the whole frame rather than cutting someone out.
    """
    observations: list[FaceObservation] = [
        o for track in visible for o in track.observations_between(start_s, end_s)
    ]
    if not observations:
        return _general_segment(
            start_s=start_s,
            end_s=end_s,
            source_w=source_w,
            source_h=source_h,
            crop_w=crop_w,
            crop_h=crop_h,
        )

    left = min(o.left for o in observations)
    right = max(o.right for o in observations)
    spread = right - left

    if spread > crop_w * FIT_IF_SPREAD_EXCEEDS:
        return CropSegment(
            start_s=start_s,
            end_s=end_s,
            width=source_w,
            height=source_h,
            keyframes=[CropKeyframe(t=start_s, x=0.0, y=0.0)],
            strategy=Strategy.WIDE,
            fit=True,
        )

    centre_x = (left + right) / 2
    eye_y = sum(o.eye_y for o in observations) / len(observations)

    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=[
            CropKeyframe(
                t=start_s,
                x=_clamp(centre_x - crop_w / 2, 0, source_w - crop_w),
                y=_clamp(eye_y - EYE_LINE_RATIO * crop_h, 0, source_h - crop_h),
            )
        ],
        strategy=Strategy.WIDE,
    )


def _general_segment(
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
) -> CropSegment:
    """No usable face — centre crop."""
    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=[CropKeyframe(t=start_s, x=(source_w - crop_w) / 2, y=(source_h - crop_h) / 2)],
        strategy=Strategy.GENERAL,
        zoom=GENERAL_ZOOM,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high)) if high > low else max(0.0, low)

"""Choosing the crop for one clip: gaming stack, speaker tracking, or centre.

Shared by the file pipeline, the live session and ``peaksift rerender`` so all
three frame clips identically.

Gaming clips adapt to what is on screen, second by second, using the face
signals the visual pass already measured:

* **facecam on**  (a small face, i.e. a webcam overlay) → facecam + gameplay stacked,
  with the facecam panel taken from wherever the cam is at that moment (many
  streamers move it between scenes);
* **facecam off** (streamer hid the cam, menu, cutscene)  → the full gameplay
  scene, fitted whole over a blurred fill;
* **full camera** (a big face: "just chatting", a reaction break) → a 9:16 crop
  on the streamer.

Short gaps in face detection (a turned head, a dark frame) don't count as the
cam being off: a state has to hold for a few seconds before the layout
switches, so clips don't flicker between layouts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import Settings
from ..layouts import StackLayout
from ..signals.timeline import Timeline
from . import ffmpeg
from .reframe import ReframeConfig, build_crop_path, build_stack_path
from .reframe.croppath import CropKeyframe, CropPath, CropSegment, Strategy, centre_crop

log = logging.getLogger(__name__)

ASPECTS = {"9:16": (9, 16), "1:1": (1, 1), "16:9": (16, 9)}

#: A face at least this wide (share of the frame) is a full-screen camera.
CAMERA_FACE_W = 0.13
#: How long the cam must be gone before switching to full gameplay.
MIN_OFF_S = 4
#: How long a full-screen camera must last before switching to it.
MIN_CAMERA_S = 3
#: The shortest stretch any layout is kept on screen.
MIN_RUN_S = 2


def layout_choice(
    settings: Settings, stack: StackLayout | None, timeline: Timeline | None = None
) -> str:
    """``gaming``, ``track``, ``center`` or ``full`` for this export."""
    choice = settings.export.layout
    vertical = settings.export.ratio == "9:16"
    if choice == "gaming":
        return "gaming" if vertical and stack is not None else "track"
    if choice == "auto":
        if vertical and stack is not None:
            return "gaming"
        kind = ((timeline.meta.get("layout") or {}).get("kind")) if timeline is not None else None
        # Gameplay/screen with no person to follow: a centre crop would cut away
        # two thirds of the scene, so show all of it.
        if vertical and kind == "screen":
            return "full"
        return "track"
    return choice


def full_scene(width: int, height: int, duration: float) -> CropPath:
    """The whole frame, fitted into the output over a blurred copy of itself."""
    return CropPath(
        source_width=width,
        source_height=height,
        segments=[
            CropSegment(
                start_s=0.0,
                end_s=duration,
                width=width,
                height=height,
                keyframes=[CropKeyframe(t=0.0, x=0.0, y=0.0)],
                strategy=Strategy.WIDE,
                fit=True,
            )
        ],
    )


def crop_for_clip(
    media: Path,
    start_s: float,
    end_s: float,
    *,
    settings: Settings,
    stack: StackLayout | None,
    transcript=None,
    width: int | None = None,
    height: int | None = None,
    timeline: Timeline | None = None,
    timeline_start_s: float | None = None,
) -> CropPath:
    """Build the crop path for ``[start_s, end_s)`` of ``media``.

    ``start_s``/``end_s`` are in ``media``'s own time; the returned path is
    clip-relative, as the export stage expects. ``timeline_start_s`` is where
    ``start_s`` falls on ``timeline`` (they differ for live recordings, where
    the media is a stitched piece of the stream).
    """
    aspect_w, aspect_h = ASPECTS[settings.export.ratio]
    choice = layout_choice(settings, stack, timeline)
    duration = end_s - start_s

    if choice in ("gaming", "center", "full") and (width is None or height is None):
        info = ffmpeg.probe(media)
        width, height = info.width or 1920, info.height or 1080

    if choice == "gaming" and stack is not None:
        stacked = build_stack_path(
            width or 1920,
            height or 1080,
            duration,
            stack.facecam,
            gameplay=stack.gameplay,
            split=stack.split,
            facecam_position=stack.facecam_position,
            exact=stack.origin in ("manual", "preset"),
        )
        if settings.export.auto_switch_layout and timeline is not None:
            begin = start_s if timeline_start_s is None else timeline_start_s
            detected = stack.origin == "auto"
            runs = facecam_runs(timeline, begin, begin + duration, positions=detected)
            if any(state != "stack" for _, _, state in runs):
                stacks = {"stack": stacked}
                regions = ((timeline.meta.get("layout") or {}).get("facecams") or [])[1:]
                for index, region in enumerate(regions if detected else [], start=1):
                    stacks[f"stack{index}"] = build_stack_path(
                        width or 1920,
                        height or 1080,
                        duration,
                        tuple(region),
                        gameplay=stack.gameplay,
                        split=stack.split,
                        facecam_position=stack.facecam_position,
                    )
                return _mixed_path(
                    runs,
                    stacks,
                    timeline,
                    begin,
                    width or 1920,
                    height or 1080,
                    media=media,
                    media_start_s=start_s,
                    config=ReframeConfig(
                        aspect_w=aspect_w,
                        aspect_h=aspect_h,
                        two_person_split=settings.export.two_person == "split",
                    ),
                )
        return stacked
    if choice == "full":
        return full_scene(width or 1920, height or 1080, duration)
    if choice == "center":
        return centre_crop(
            width or 1920, height or 1080, duration, aspect_w=aspect_w, aspect_h=aspect_h
        )
    return build_crop_path(
        media,
        start_s=start_s,
        end_s=end_s,
        transcript=transcript,
        config=ReframeConfig(
            aspect_w=aspect_w,
            aspect_h=aspect_h,
            two_person_split=settings.export.two_person == "split",
        ),
    )


# --------------------------------------------------------------------------
# Facecam on / off / full camera
# --------------------------------------------------------------------------


def facecam_runs(
    timeline: Timeline, start_s: float, end_s: float, *, positions: bool = True
) -> list[tuple[float, float, str]]:
    """Clip-relative ``(start, end, state)`` runs: ``stack``, ``full`` or ``camera``.

    With several detected facecam positions, a stretch where the cam is in
    position *k* (k >= 1) is ``stack<k>``; ``stack`` is the main position.
    ``positions=False`` (a facecam box the user drew) treats the cam anywhere
    as ``stack``.

    Unmeasured seconds (no visual pass there) count as ``stack``, the safe
    default for a layout that was chosen because a facecam was found.
    """
    duration = max(0.0, end_s - start_s)
    if duration <= 0 or not timeline.has("face_present"):
        return [(0.0, duration, "stack")]
    first = int(np.floor(start_s))
    last = int(np.ceil(end_s))
    n = timeline.length
    present = timeline.get("face_present", n)
    size = timeline.get("face_size", n)
    covered = timeline.covered("face_present", n)

    slots = _cam_slots(timeline, first, last)
    states: list[str] = []
    for i, second in enumerate(range(first, last)):
        if second < 0 or second >= n or not covered[second]:
            states.append("stack")
        elif slots is not None:
            # Where the webcam is decides, whatever else is on screen: a game
            # character's big face in a cutscene isn't the streamer.
            slot = slots[i]
            if slot is not None:
                states.append("stack" if slot == 0 or not positions else f"stack{slot}")
            elif present[second] > 0 and size[second] >= CAMERA_FACE_W:
                states.append("camera")
            else:
                states.append("full")
        elif present[second] <= 0:
            states.append("full")
        elif size[second] >= CAMERA_FACE_W:
            states.append("camera")
        else:
            states.append("stack")

    runs = _smooth(states)
    out: list[tuple[float, float, str]] = []
    for a, b, state in runs:
        lo = max(0.0, first + a - start_s)
        hi = min(duration, first + b - start_s)
        if hi > lo:
            out.append((lo, hi, state))
    if out:
        out[0] = (0.0, out[0][1], out[0][2])
        out[-1] = (out[-1][0], duration, out[-1][2])
    return out or [(0.0, duration, "stack")]


def _cam_slots(timeline: Timeline, first: int, last: int) -> list[int | None] | None:
    """For each second, which detected facecam position has a face (or None).

    None overall when the timeline predates per-position tracking (or found
    no facecam position); the size-based rules then apply as before.
    """
    regions = (timeline.meta.get("layout") or {}).get("facecams") or []
    rows = timeline.meta.get("small_faces")
    if len(regions) < 1 or rows is None:
        return None
    out: list[int | None] = [None] * max(0, last - first)
    for t, cx, cy, _w, _h in rows:
        i = int(t) - first
        if not 0 <= i < len(out):
            continue
        for k, (x, y, w, h) in enumerate(regions):
            if x <= cx <= x + w and y <= cy <= y + h:
                if out[i] is None or k < out[i]:
                    out[i] = k
                break
    return out


def _smooth(states: list[str]) -> list[tuple[int, int, str]]:
    """Merge per-second states into runs, absorbing ones too short to show."""
    runs: list[list] = []
    for i, state in enumerate(states):
        if runs and runs[-1][2] == state:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, state])

    minimum = {"full": MIN_OFF_S, "camera": MIN_CAMERA_S}
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for index, (a, b, state) in enumerate(runs):
            if b - a >= minimum.get(state, MIN_RUN_S):
                continue
            # Too short: hand it to the longer neighbour (stack wins ties).
            left = runs[index - 1] if index > 0 else None
            right = runs[index + 1] if index + 1 < len(runs) else None
            target = max(
                (r for r in (left, right) if r is not None),
                key=lambda r: (r[1] - r[0], r[2].startswith("stack")),
            )
            target[0], target[1] = min(target[0], a), max(target[1], b)
            del runs[index]
            merged: list[list] = []
            for run in runs:
                if merged and merged[-1][2] == run[2]:
                    merged[-1][1] = run[1]
                else:
                    merged.append(run)
            runs = merged
            changed = True
            break
    return [(a, b, s) for a, b, s in runs]


def _mixed_path(
    runs: list[tuple[float, float, str]],
    stacks: dict[str, CropPath],
    timeline: Timeline,
    timeline_start_s: float,
    width: int,
    height: int,
    *,
    media: Path | None = None,
    media_start_s: float = 0.0,
    config: ReframeConfig | None = None,
) -> CropPath:
    segments: list[CropSegment] = []
    for a, b, state in runs:
        if state.startswith("stack"):
            template = (stacks.get(state) or stacks["stack"]).segments[0]
            segments.append(
                CropSegment(
                    start_s=a,
                    end_s=b,
                    width=template.width,
                    height=template.height,
                    keyframes=[
                        CropKeyframe(t=a, x=template.keyframes[0].x, y=template.keyframes[0].y)
                    ],
                    strategy=Strategy.STACK,
                    panels=dict(template.panels or {}),
                )
            )
        elif state == "full":
            # The whole gameplay frame, fitted over a blurred copy of itself.
            segments.append(
                CropSegment(
                    start_s=a,
                    end_s=b,
                    width=width,
                    height=height,
                    keyframes=[CropKeyframe(t=a, x=0.0, y=0.0)],
                    strategy=Strategy.WIDE,
                    fit=True,
                )
            )
        else:
            segments.extend(
                _camera_segments(
                    media, media_start_s, timeline, timeline_start_s, a, b, width, height, config
                )
            )
    log.info(
        "Layout switches in clip: %s",
        ", ".join(f"{state} {a:.0f}-{b:.0f}s" for a, b, state in runs),
    )
    return CropPath(source_width=width, source_height=height, segments=segments)


def _camera_segments(
    media: Path | None,
    media_start_s: float,
    timeline: Timeline,
    timeline_start_s: float,
    a: float,
    b: float,
    width: int,
    height: int,
    config: ReframeConfig | None,
) -> list[CropSegment]:
    """Full-camera stretch: let the face reframer frame it.

    It tracks one speaker, or stacks two people when two big cams are on screen
    (a duo stream whose editor enlarges both webcams). Its segments are
    clip-relative to the sub-range, so they're shifted to start at ``a``.
    """
    if media is not None:
        try:
            sub = build_crop_path(
                media, start_s=media_start_s + a, end_s=media_start_s + b, config=config
            )
            shifted: list[CropSegment] = []
            for segment in sub.segments:
                segment.start_s += a
                segment.end_s += a
                for keyframe in segment.keyframes:
                    keyframe.t += a
                shifted.append(segment)
            if shifted:
                shifted[0].start_s = a
                shifted[-1].end_s = b
                return shifted
        except Exception as exc:
            log.warning("Camera reframing failed (%s); centring on the face instead.", exc)
    return [
        _camera_segment(timeline, timeline_start_s + a, timeline_start_s + b, a, b, width, height)
    ]


def _camera_segment(
    timeline: Timeline, t0: float, t1: float, a: float, b: float, width: int, height: int
) -> CropSegment:
    """A still 9:16 crop centred on where the streamer's face was."""
    crop_w = int(min(width, height * 9 / 16)) // 2 * 2
    boxes = [
        box
        for box in (timeline.meta.get("face_boxes") or [])
        if t0 <= float(box.get("t", -1)) < t1 and float(box.get("w", 0)) >= CAMERA_FACE_W
    ]
    cx = float(np.median([box["cx"] for box in boxes])) * width if boxes else width / 2
    x = min(max(0.0, cx - crop_w / 2), width - crop_w)
    return CropSegment(
        start_s=a,
        end_s=b,
        width=crop_w,
        height=height // 2 * 2,
        keyframes=[CropKeyframe(t=a, x=float(int(x) // 2 * 2), y=0.0)],
        strategy=Strategy.TRACK,
    )

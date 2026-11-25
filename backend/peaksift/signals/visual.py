"""Visual signals — motion, scene changes, faces and facial reactions.

One decode pass per chunk produces every visual signal. Frames are pulled
through an ffmpeg pipe at a low sample rate (2 fps by default) and a modest
width, which keeps a multi-hour stream tractable:

``motion``        mean absolute frame difference — visual activity.
``scene``         colour-histogram distance between samples — cuts, map changes,
                  menu/death screens, replays.
``face_present``  a face is visible (streamer cam, podcast guest, etc.).
``face_size``     largest face width as a fraction of frame width.
``mouth_open``    MediaPipe ``jawOpen`` blendshape — shouting, laughing.
``face_surprise`` brow raise + eyes wide + jaw drop.
``face_joy``      smile.
``head_motion``   how much the face moved between samples, in face widths.

Plus a short colour signature per second (``vsig_*``) used to split long
streams into chapters (game changes, "just chatting" segments).

Decoding uses NVDEC when available and falls back to the CPU. For very long
sources a keyframe-only mode decodes only I-frames, which is an order of
magnitude faster at the cost of coarser motion.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..pipeline import ffmpeg

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_FPS = 2.0
DEFAULT_FACE_FPS = 1.0
DECODE_WIDTH = 640
MOTION_WIDTH = 160
SIGNATURE_BINS = 8

#: Blendshape names from MediaPipe's FaceLandmarker (ARKit naming).
_BLEND = (
    "jawOpen",
    "browInnerUp",
    "eyeWideLeft",
    "eyeWideRight",
    "mouthSmileLeft",
    "mouthSmileRight",
)


@dataclass
class FaceBox:
    """Largest face in one sampled frame, in normalised [0,1] coordinates."""

    t: float
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class VisualResult:
    signals: dict[str, np.ndarray]
    face_boxes: list[FaceBox] = field(default_factory=list)
    #: Every small face (webcam-overlay sized) in every face sample — not just
    #: the largest — so a facecam is found even when a game character's face
    #: is bigger, and wherever on screen the streamer moves it.
    small_faces: list[FaceBox] = field(default_factory=list)
    mode: str = "full"
    width: int = 0
    height: int = 0


def visual_features(
    video: Path,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    face_fps: float = DEFAULT_FACE_FPS,
    keyframes_only: bool = False,
    faces: bool = True,
    cancelled=None,
) -> VisualResult:
    """Measure visual signals over ``[start_s, start_s + duration_s)``.

    Arrays are relative to ``start_s``, one value per second.
    """
    info = ffmpeg.probe(video)
    if not info.has_video or not info.width or not info.height:
        return VisualResult(signals={}, mode="none")

    total = duration_s if duration_s is not None else max(0.0, info.duration_s - start_s)
    length = max(1, int(np.ceil(total)))
    out_w = DECODE_WIDTH if info.width >= DECODE_WIDTH else info.width - info.width % 2
    out_h = int(round(info.height * out_w / info.width / 2)) * 2

    fps = 1.0 if keyframes_only else sample_fps
    face_every = max(1, int(round(fps / max(face_fps, 0.01)))) if faces else 0

    import cv2

    landmarker = _make_landmarker() if faces else None

    motion = np.zeros(length, dtype=np.float32)
    scene = np.zeros(length, dtype=np.float32)
    face_present = np.zeros(length, dtype=np.float32)
    face_size = np.zeros(length, dtype=np.float32)
    mouth_open = np.zeros(length, dtype=np.float32)
    surprise = np.zeros(length, dtype=np.float32)
    joy = np.zeros(length, dtype=np.float32)
    head_motion = np.zeros(length, dtype=np.float32)
    signature_sum = np.zeros((length, SIGNATURE_BINS), dtype=np.float32)
    signature_n = np.zeros(length, dtype=np.float32)
    face_boxes: list[FaceBox] = []
    small_faces: list[FaceBox] = []

    previous_small: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    previous_face: FaceBox | None = None
    frame_bytes = out_w * out_h * 3
    index = 0

    for frame in _decode(video, start_s, total, fps, out_w, out_h, keyframes_only, frame_bytes):
        if cancelled is not None and cancelled():
            break
        t = index / fps
        second = min(length - 1, int(t))

        small = cv2.resize(frame, (MOTION_WIDTH, max(2, MOTION_WIDTH * out_h // out_w)))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if previous_small is not None:
            diff = float(np.mean(np.abs(gray - previous_small))) / 255.0
            motion[second] = max(motion[second], diff)
        previous_small = gray

        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if previous_hist is not None:
            distance = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            scene[second] = max(scene[second], distance)
        previous_hist = hist

        hue = np.histogram(hsv[..., 0], bins=SIGNATURE_BINS, range=(0, 180))[0].astype(np.float32)
        signature_sum[second] += hue / max(1.0, hue.sum())
        signature_n[second] += 1

        if landmarker is not None and index % face_every == 0:
            box, blend, smalls = _detect_face(landmarker, frame, int(t * 1000) + 1)
            for small_face in smalls:
                small_face.t = start_s + t
                small_faces.append(small_face)
            if box is not None:
                box.t = start_s + t
                face_boxes.append(box)
                face_present[second] = 1.0
                face_size[second] = max(face_size[second], box.w)
                mouth_open[second] = max(mouth_open[second], blend.get("jawOpen", 0.0))
                s = (
                    blend.get("jawOpen", 0.0)
                    + blend.get("browInnerUp", 0.0)
                    + (blend.get("eyeWideLeft", 0.0) + blend.get("eyeWideRight", 0.0)) / 2
                ) / 3
                j = (blend.get("mouthSmileLeft", 0.0) + blend.get("mouthSmileRight", 0.0)) / 2
                surprise[second] = max(surprise[second], s)
                joy[second] = max(joy[second], j)
                if previous_face is not None and box.w > 0:
                    moved = np.hypot(box.cx - previous_face.cx, box.cy - previous_face.cy)
                    head_motion[second] = max(head_motion[second], float(moved / box.w))
                previous_face = box
            else:
                previous_face = None

        index += 1

    if landmarker is not None:
        landmarker.close()

    signals = {
        "motion": motion,
        "scene": scene,
    }
    if faces:
        signals.update(
            {
                "face_present": face_present,
                "face_size": face_size,
                "mouth_open": mouth_open,
                "face_surprise": surprise,
                "face_joy": joy,
                "head_motion": head_motion,
            }
        )
    filled = signature_n > 0
    signature = np.zeros_like(signature_sum)
    signature[filled] = signature_sum[filled] / signature_n[filled, None]
    for b in range(SIGNATURE_BINS):
        signals[f"vsig_{b}"] = signature[:, b]

    log.info(
        "Visual pass over %s: %d frames, %d face samples (%s).",
        video.name,
        index,
        len(face_boxes),
        "keyframes" if keyframes_only else f"{fps:g} fps",
    )
    return VisualResult(
        signals=signals,
        face_boxes=face_boxes,
        small_faces=small_faces,
        mode="keyframes" if keyframes_only else "full",
        width=info.width,
        height=info.height,
    )


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def _decode(video, start_s, duration_s, fps, out_w, out_h, keyframes_only, frame_bytes):
    """Yield RGB frames; try NVDEC first, then plain CPU decoding."""
    attempts = ["cuda", "cpu"]
    for attempt in attempts:
        produced = 0
        try:
            for frame in _decode_once(
                video, start_s, duration_s, fps, out_w, out_h, keyframes_only, frame_bytes, attempt
            ):
                produced += 1
                yield frame
            return
        except _DecodeFailed as exc:
            if produced:
                # Mid-stream failure after frames were consumed: don't restart
                # and double-count; just stop with what we have.
                log.warning("Visual decode stopped early: %s", exc)
                return
            log.debug("Decode with %s failed (%s); trying the next option.", attempt, exc)
    log.warning("Could not decode %s for visual signals.", video)


class _DecodeFailed(RuntimeError):
    pass


def _decode_once(video, start_s, duration_s, fps, out_w, out_h, keyframes_only, frame_bytes, mode):
    command = [ffmpeg.ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if keyframes_only:
        command += ["-skip_frame", "nokey"]
    if mode == "cuda":
        command += ["-hwaccel", "cuda"]
    if start_s > 0:
        command += ["-ss", f"{start_s:.3f}"]
    command += ["-t", f"{duration_s:.3f}", "-i", str(video), "-an", "-sn"]
    command += [
        "-vf",
        f"fps={fps:g},scale={out_w}:{out_h}:flags=area,format=rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frames = 0
    try:
        while True:
            buffer = proc.stdout.read(frame_bytes)
            if not buffer or len(buffer) < frame_bytes:
                break
            frames += 1
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(out_h, out_w, 3)
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        proc.wait()
    if frames == 0 and proc.returncode not in (0, None):
        raise _DecodeFailed(stderr.strip().splitlines()[-1] if stderr.strip() else "no frames")


# --------------------------------------------------------------------------
# Faces
# --------------------------------------------------------------------------


class _FaceStack:
    """YuNet finds faces (it handles tiny corner facecams); MediaPipe, when it
    can load, reads expressions from an upscaled crop of the face it found.

    MediaPipe's landmarker alone misses a webcam overlay whose face is ~25 px
    wide in the analysis frame — the normal size of a streamer facecam — so it
    can't be the detector here. Run on a crop, it gives the jaw/brow/smile
    blendshapes the reaction signals want.
    """

    CROP = 256

    def __init__(self, detector, landmarker) -> None:
        self.detector = detector
        self.landmarker = landmarker

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()

    def blendshapes(self, frame: np.ndarray, face) -> dict[str, float]:
        if self.landmarker is None:
            return {}
        import cv2
        import mediapipe as mp

        height, width = frame.shape[:2]
        side = max(face.w, face.h) * 1.8
        cx, cy = face.x + face.w / 2, face.y + face.h / 2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return {}
        crop = cv2.resize(
            frame[y0:y1, x0:x1], (self.CROP, self.CROP), interpolation=cv2.INTER_CUBIC
        )
        try:
            result = self.landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
            )
        except Exception as exc:
            log.debug("Blendshapes failed on a face crop: %s", exc)
            return {}
        if not result.face_blendshapes:
            return {}
        return {
            c.category_name: float(c.score)
            for c in result.face_blendshapes[0]
            if c.category_name in _BLEND
        }


def _make_landmarker():
    from ..faces_cv import backend as cv_backend
    from ..faces_cv import mediapipe_usable

    try:
        detector = cv_backend()
    except Exception as exc:
        log.warning("Face signals disabled: %s", exc)
        return None
    if not mediapipe_usable():
        return _FaceStack(detector, None)
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        from .. import models

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models.ensure("face_landmarker"))),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            output_face_blendshapes=True,
        )
        return _FaceStack(detector, FaceLandmarker.create_from_options(options))
    except Exception as exc:  # expressions fall back to the emotion model
        log.info("MediaPipe blendshapes unavailable (%s); using the emotion model.", exc)
        return _FaceStack(detector, None)


def _detect_face(stack, frame: np.ndarray, timestamp_ms: int):
    del timestamp_ms  # IMAGE mode on crops; kept for the call signature
    faces = stack.detector.detect(frame)
    if not faces:
        return None, {}, []
    height, width = frame.shape[:2]

    def as_box(face) -> FaceBox:
        return FaceBox(
            t=0.0,
            cx=(face.x + face.w / 2) / width,
            cy=(face.y + face.h / 2) / height,
            w=face.w / width,
            h=face.h / height,
        )

    smalls = [as_box(f) for f in faces[: MAX_SMALL_FACES * 2] if f.w / width < FACECAM_MAX_FACE_W]
    face = faces[0]
    box = as_box(face)
    blend = stack.blendshapes(frame, face)
    if not blend:
        box, blend = _detect_face_cv(stack.detector, frame, face=face, box=box)
    return box, blend, smalls[:MAX_SMALL_FACES]


def _detect_face_cv(backend, frame: np.ndarray, *, face, box: FaceBox):
    """FER+ emotions on a YuNet face, mapped onto the blendshape keys."""
    emotions = backend.emotions(frame, face)
    # Map onto the names the scoring below expects. Surprise feeds the three
    # surprise components equally; there is no jaw-opening measurement here.
    surprise = emotions.get("surprise", 0.0) + 0.5 * emotions.get("fear", 0.0)
    joy = emotions.get("happiness", 0.0)
    blend = {
        "browInnerUp": surprise,
        "eyeWideLeft": surprise,
        "eyeWideRight": surprise,
        "jawOpen": 0.0,
        "mouthSmileLeft": joy,
        "mouthSmileRight": joy,
    }
    return box, blend


# --------------------------------------------------------------------------
# Layout inference
# --------------------------------------------------------------------------


@dataclass
class Layout:
    """What kind of picture this is, and where the streamer's camera sits."""

    #: ``facecam`` (small camera overlay on gameplay/screen), ``camera``
    #: (the camera *is* the picture: just chatting, podcast, IRL) or ``screen``
    #: (no face: gameplay or screen only).
    kind: str
    #: Facecam region in normalised coordinates (x, y, w, h), for ``facecam``.
    #: The position the cam is in most often.
    facecam: tuple[float, float, float, float] | None = None
    face_ratio: float = 0.0
    median_face_w: float = 0.0
    #: Every position the facecam was found in, most-used first (``facecam``
    #: is the first). Streamers often move the cam between scenes: a corner
    #: during gameplay, another corner during cutscenes.
    facecams: list[tuple[float, float, float, float]] = field(default_factory=list)


#: Faces narrower than this share of the frame are a corner camera overlay.
FACECAM_MAX_FACE_W = 0.13
#: Fraction of face samples that must contain a face to call it a camera stream.
MIN_FACE_PRESENCE = 0.35
#: A corner facecam only has to be on for this share of the video.
MIN_FACECAM_PRESENCE = 0.15
#: Each facecam position must hold for this share of the video on its own.
MIN_POSITION_PRESENCE = 0.06
#: At most this many facecam positions are kept.
MAX_FACECAM_POSITIONS = 3
#: Small faces recorded per sample (the streamer plus a few on-screen others).
MAX_SMALL_FACES = 4
#: Grid used to find spots a face keeps coming back to (share of the frame).
POSITION_CELL = 0.05
#: Median face width for a picture that *is* the camera (just chatting, IRL).
#: Below it the faces are small ones on a screen — game characters, a crowd —
#: and following them would crop away the picture.
CAMERA_MIN_FACE_W = 0.06


def facecam_positions(
    small_faces: list[FaceBox], samples: int
) -> list[tuple[tuple[float, float, float, float], float]]:
    """Spots a small face keeps coming back to: ``(region, share of samples)``.

    A webcam overlay is pinned to the screen, so its face lands in the same few
    grid cells sample after sample; faces in the game or video wander and
    spread out. Each spot must hold for ``MIN_POSITION_PRESENCE`` of the video.
    """
    if samples <= 0 or not small_faces:
        return []
    cells: dict[tuple[int, int], list[FaceBox]] = {}
    for face in small_faces:
        key = (int(face.cx / POSITION_CELL), int(face.cy / POSITION_CELL))
        cells.setdefault(key, []).append(face)

    # 1. Clusters: the busiest grid cell plus its neighbours, repeatedly.
    taken: set[tuple[int, int]] = set()
    clusters: list[list[FaceBox]] = []
    for key, _faces in sorted(cells.items(), key=lambda kv: -len(kv[1])):
        if key in taken:
            continue
        group = [
            k
            for k in cells
            if k not in taken and abs(k[0] - key[0]) <= 1 and abs(k[1] - key[1]) <= 1
        ]
        taken.update(group)
        faces = [f for k in group for f in cells[k]]
        if len({round(f.t, 1) for f in faces}) / samples >= MIN_POSITION_PRESENCE / 2:
            clusters.append(faces)

    # 2. One webcam can span two clusters (the streamer leans, sits up): merge
    #    clusters whose regions overlap, so the crop doesn't hop between them.
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _overlap(_region_of(clusters[i]), _region_of(clusters[j])) > 0.1:
                    clusters[i] = clusters[i] + clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    # 3. Keep steady, frequent spots, busiest first.
    found: list[tuple[tuple[float, float, float, float], float]] = []
    for faces in clusters:
        share = len({round(f.t, 1) for f in faces}) / samples
        widths = np.array([f.w for f in faces])
        # A pinned webcam keeps its size; crowd faces at one spot don't.
        steady = np.median(np.abs(widths - np.median(widths))) <= 0.35 * np.median(widths)
        cx = float(np.median([f.cx for f in faces]))
        cy = float(np.median([f.cy for f in faces]))
        # Overlays sit off-centre; the middle of the frame is the game, where
        # faces are characters (a cutscene close-up keeps landing there too).
        central = 0.3 < cx < 0.7 and 0.25 < cy < 0.75
        if share >= MIN_POSITION_PRESENCE and steady and not central:
            found.append((_region_of(faces), share))
    found.sort(key=lambda item: -item[1])
    return found[:MAX_FACECAM_POSITIONS]


def _region_of(faces: list[FaceBox]) -> tuple[float, float, float, float]:
    return facecam_region(
        float(np.median([f.cx for f in faces])),
        float(np.median([f.cy for f in faces])),
        float(np.median([f.w for f in faces])),
        float(np.median([f.h for f in faces])),
    )


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over the smaller box's area."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(1e-9, min(a[2] * a[3], b[2] * b[3]))


def infer_layout(
    face_boxes: list[FaceBox], samples: int, small_faces: list[FaceBox] | None = None
) -> Layout:
    """Decide between facecam-over-gameplay, full camera, and screen-only.

    A facecam only needs to be on for part of the video (``MIN_FACECAM_PRESENCE``):
    the export switches to the full gameplay scene whenever it's off, so a
    streamer who turns the cam on halfway still gets the stacked layout then.
    ``small_faces`` (every small face per sample) finds a facecam even when a
    bigger face is on screen; without it the largest faces are used.
    """
    if samples <= 0 or not face_boxes:
        return Layout(kind="screen")
    presence = len(face_boxes) / samples
    median_w = float(np.median([b.w for b in face_boxes]))

    smalls = (
        small_faces
        if small_faces is not None
        else [b for b in face_boxes if b.w < FACECAM_MAX_FACE_W]
    )
    positions = facecam_positions(smalls, samples)
    if positions and sum(share for _, share in positions) >= MIN_FACECAM_PRESENCE:
        regions = [region for region, _ in positions]
        return Layout(
            kind="facecam",
            facecam=regions[0],
            facecams=regions,
            face_ratio=presence,
            median_face_w=median_w,
        )
    if presence >= MIN_FACE_PRESENCE and median_w >= CAMERA_MIN_FACE_W:
        return Layout(kind="camera", face_ratio=presence, median_face_w=median_w)
    return Layout(kind="screen", face_ratio=presence, median_face_w=median_w)


def facecam_region(
    cx: float, cy: float, face_w: float, face_h: float, *, aspect: float = 16 / 9
) -> tuple[float, float, float, float]:
    """A camera-overlay-sized box around a face, in normalised coordinates.

    ``aspect`` is width/height in *pixels* of a 16:9 source; the box is sized
    to roughly 3x the face width, which matches typical facecam overlays.
    """
    width = min(1.0, face_w * 3.2)
    # Convert the pixel aspect of the top panel (~1.6) to normalised units.
    height = min(1.0, width * aspect / 1.6)
    height = max(height, face_h * 2.2)
    x = min(max(0.0, cx - width / 2), 1.0 - width)
    y = min(max(0.0, cy - height * 0.45), 1.0 - height)
    return (x, y, width, height)

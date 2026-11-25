"""OpenCV face backend — the fallback when MediaPipe can't run.

MediaPipe ships an unsigned native library that Windows Smart App Control (and
some locked-down machines) refuse to load. OpenCV is already a dependency and
loads fine, so this module provides the same two things from OpenCV's DNN
module and two small ONNX models:

* **YuNet** — fast face detection with five landmarks (eyes, nose, mouth corners).
* **FER+**  — eight-way facial emotion, of which *surprise* and *happiness* feed
  the reaction signals.

Compared to MediaPipe there is no mouth-opening measurement, so the
``mouth_open`` signal stays empty and speaker assignment in the reframe stage
falls back to face prominence. Everything else works the same.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: FER+ output order.
EMOTIONS = ("neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")


@dataclass
class CvFace:
    """One detected face in pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float
    #: (x, y) for right eye, left eye, nose tip, right mouth corner, left mouth corner.
    landmarks: list[tuple[float, float]]

    @property
    def eye_y(self) -> float:
        return (self.landmarks[0][1] + self.landmarks[1][1]) / 2


class CvFaceBackend:
    """YuNet detection + optional FER+ emotions, via cv2.dnn."""

    def __init__(self, *, emotions: bool = True, score_threshold: float = 0.6) -> None:
        import cv2

        from . import models

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            str(models.ensure("yunet")), "", (320, 320), score_threshold, 0.3, 5000
        )
        self._size: tuple[int, int] | None = None
        self._emotion_net = None
        if emotions:
            try:
                self._emotion_net = cv2.dnn.readNetFromONNX(str(models.ensure("ferplus")))
            except Exception as exc:  # emotions are optional
                log.warning("Emotion model unavailable (%s); reactions use audio only.", exc)

    def detect(self, rgb: np.ndarray) -> list[CvFace]:
        height, width = rgb.shape[:2]
        if self._size != (width, height):
            self._detector.setInputSize((width, height))
            self._size = (width, height)
        bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        _, raw = self._detector.detect(bgr)
        faces: list[CvFace] = []
        if raw is None:
            return faces
        for row in raw:
            x, y, w, h = (float(v) for v in row[:4])
            landmarks = [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]
            faces.append(CvFace(x, y, w, h, float(row[14]), landmarks))
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
        return faces

    def emotions(self, rgb: np.ndarray, face: CvFace) -> dict[str, float]:
        """Softmax emotion probabilities for one face (empty if unavailable)."""
        if self._emotion_net is None:
            return {}
        cv2 = self._cv2
        height, width = rgb.shape[:2]
        pad = 0.15 * face.w
        x0, y0 = int(max(0, face.x - pad)), int(max(0, face.y - pad))
        x1 = int(min(width, face.x + face.w + pad))
        y1 = int(min(height, face.y + face.h + pad))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return {}
        gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        blob = cv2.resize(gray, (64, 64)).astype(np.float32).reshape(1, 1, 64, 64)
        self._emotion_net.setInput(blob)
        logits = self._emotion_net.forward().reshape(-1)
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        return {name: float(p) for name, p in zip(EMOTIONS, probs, strict=False)}


_backend: CvFaceBackend | None = None


def backend() -> CvFaceBackend:
    global _backend
    if _backend is None:
        _backend = CvFaceBackend()
    return _backend


def mediapipe_usable() -> bool:
    """True when MediaPipe's native Tasks library can actually be loaded."""
    global _MEDIAPIPE_OK
    if _MEDIAPIPE_OK is None:
        try:
            from mediapipe.tasks.python.core import mediapipe_c_bindings

            mediapipe_c_bindings.load_raw_library()
            _MEDIAPIPE_OK = True
        except Exception as exc:
            log.warning("MediaPipe can't load (%s); using the OpenCV face backend.", exc)
            _MEDIAPIPE_OK = False
    return _MEDIAPIPE_OK


_MEDIAPIPE_OK: bool | None = None

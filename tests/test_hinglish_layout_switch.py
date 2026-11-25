"""Hinglish captions and gaming layouts that follow the webcam on and off."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from peaksift import layouts
from peaksift.config import Settings
from peaksift.hinglish import transliterate_text, transliterate_word
from peaksift.pipeline.framing import crop_for_clip, facecam_runs, layout_choice
from peaksift.pipeline.reframe.croppath import Strategy
from peaksift.pipeline.transcribe import to_hinglish
from peaksift.pipeline.transcript import Segment, Transcript, Word
from peaksift.signals.timeline import Timeline
from peaksift.signals.visual import FaceBox, infer_layout

# --------------------------------------------------------------------------
# Hinglish
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hindi", "hinglish"),
    [
        ("क्या कर रहे हो भाई", "kya kar rahe ho bhai"),
        ("आज बहुत मज़ा आया", "aaj bahut maza aaya"),
        ("अपना समझना करना कमरा", "apna samajhna karna kamra"),
        ("मैं नहीं जानता", "main nahin jaanta"),
        ("यार ये गेम तो बहुत मुश्किल है", "yaar ye game to bahut mushkil hai"),
        ("चलो सब्सक्राइब करो", "chalo subscribe karo"),
        ("अरे बाप रे, ये क्या हो गया।", "are baap re, ye kya ho gaya."),
        ("पढ़ लिया लड़का", "padh liya ladka"),
        ("१०० किल", "100 kill"),
        ("वो देखो", "wo dekho"),
    ],
)
def test_transliteration(hindi: str, hinglish: str) -> None:
    assert transliterate_text(hindi) == hinglish


def test_non_devanagari_words_are_untouched() -> None:
    for word in ("GG", "lol!", "123", "bhai", "😂"):
        assert transliterate_word(word) == word


def test_to_hinglish_keeps_word_timings() -> None:
    words = [Word("क्या", 1.0, 1.3), Word("बात", 1.3, 1.6), Word("है!", 1.6, 2.0)]
    tr = Transcript(words=words, segments=[Segment("क्या बात है!", 1.0, 2.0, 0, 2)], language="hi")
    to_hinglish(tr)
    assert [w.text for w in tr.words] == ["kya", "baat", "hai!"]
    assert [(w.start, w.end) for w in tr.words] == [(1.0, 1.3), (1.3, 1.6), (1.6, 2.0)]
    assert tr.segments[0].text == "kya baat hai!"


# --------------------------------------------------------------------------
# Facecam on / off / full camera
# --------------------------------------------------------------------------


def _timeline(states: str) -> Timeline:
    """One character per second: s = small facecam, - = no face, C = big face."""
    t = Timeline()
    present = [0.0 if c == "-" else 1.0 for c in states]
    size = [0.05 if c == "s" else (0.3 if c == "C" else 0.0) for c in states]
    t.put("face_present", 0, present)
    t.put("face_size", 0, size)
    t.meta["layout"] = {"kind": "facecam", "facecam": [0.8, 0.0, 0.2, 0.25]}
    return t


def test_facecam_runs_switch_and_ignore_blips() -> None:
    #           0         1         2         3
    #           0123456789012345678901234567890123
    timeline = _timeline("ssssss-ssss----------sssssCCCCCsss")
    runs = facecam_runs(timeline, 0, 34)
    states = [s for _, _, s in runs]
    # The one-second gap at 6 is a detection blip, not the cam going off.
    assert states == ["stack", "full", "stack", "camera", "stack"]
    full = next(r for r in runs if r[2] == "full")
    assert full[0] == 11 and full[1] == 21
    assert runs[0][0] == 0 and runs[-1][1] == 34


def test_facecam_runs_are_clip_relative_and_default_to_stack() -> None:
    timeline = _timeline("s" * 10 + "-" * 10 + "s" * 10)
    runs = facecam_runs(timeline, 5.5, 25.5)
    assert [s for _, _, s in runs] == ["stack", "full", "stack"]
    assert runs[1][0] == pytest.approx(4.5) and runs[1][1] == pytest.approx(14.5)
    assert facecam_runs(Timeline(), 0, 10) == [(0.0, 10.0, "stack")]


def test_crop_for_clip_mixes_stack_full_and_camera() -> None:
    settings = Settings()
    stack = layouts.StackLayout(facecam=(0.8, 0.0, 0.2, 0.25))
    timeline = _timeline("s" * 8 + "-" * 8 + "C" * 6 + "s" * 8)
    crop = crop_for_clip(
        Path("x.mp4"),
        0,
        30,
        settings=settings,
        stack=stack,
        width=1920,
        height=1080,
        timeline=timeline,
    )
    kinds = [(seg.strategy, seg.fit) for seg in crop.segments]
    assert kinds == [
        (Strategy.STACK, False),
        (Strategy.WIDE, True),
        (Strategy.TRACK, False),
        (Strategy.STACK, False),
    ]
    assert crop.segments[0].start_s == 0 and crop.segments[-1].end_s == 30
    camera = crop.segments[2]
    assert abs(camera.width / camera.height - 9 / 16) < 0.01

    settings.export.auto_switch_layout = False
    plain = crop_for_clip(
        Path("x.mp4"),
        0,
        30,
        settings=settings,
        stack=stack,
        width=1920,
        height=1080,
        timeline=timeline,
    )
    assert len(plain.segments) == 1 and plain.segments[0].strategy == Strategy.STACK


def test_screen_only_video_shows_the_full_scene() -> None:
    settings = Settings()
    timeline = Timeline()
    timeline.meta["layout"] = {"kind": "screen"}
    assert layout_choice(settings, None, timeline) == "full"
    crop = crop_for_clip(
        Path("x.mp4"),
        0,
        20,
        settings=settings,
        stack=None,
        width=1920,
        height=1080,
        timeline=timeline,
    )
    assert crop.segments[0].fit is True
    timeline.meta["layout"] = {"kind": "camera"}
    assert layout_choice(settings, None, timeline) == "track"


def test_facecam_detected_even_when_only_on_part_of_the_time() -> None:
    cam = [FaceBox(t=i, cx=0.9, cy=0.1, w=0.06, h=0.1) for i in range(25)]
    layout = infer_layout(cam, 100)  # webcam on for a quarter of the video
    assert layout.kind == "facecam" and layout.facecam is not None
    assert infer_layout(cam[:10], 100).kind == "screen"


def test_timeline_face_signal_shape() -> None:
    timeline = _timeline("s-C")
    assert np.allclose(timeline.get("face_present"), [1, 0, 1])


# --------------------------------------------------------------------------
# Hindi speech models and Gemini captions
# --------------------------------------------------------------------------


def test_hindi_model_choice_follows_script() -> None:
    from peaksift.config import WhisperSettings
    from peaksift.pipeline.transcribe import hindi_model_for

    settings = WhisperSettings(model="large-v3-turbo")
    assert hindi_model_for(settings) == "hinglish-prime"
    # Hinglish models write Roman script: not for Devanagari captions.
    settings.hindi_script = "devanagari"
    assert hindi_model_for(settings) == ""
    settings.model = "small"
    assert hindi_model_for(settings) == "large-v3-turbo"
    settings.hindi_model = "large-v3"
    assert hindi_model_for(settings) == "large-v3"
    assert hindi_model_for(WhisperSettings(hindi_model="")) == ""


def test_speech_model_catalogue() -> None:
    from peaksift.speech_models import HINDI_MODELS, lookup, model_source, writes_roman

    assert writes_roman("hinglish-prime")
    assert writes_roman("Hub84/faster-whisper-hinglish-prime")
    assert not writes_roman("large-v3-turbo") and not writes_roman("someone/unknown")
    assert lookup("hinglish-apex").repo == "abhayhadez/anchor-hinglish-apex-ct2"
    # Plain repos and unknown names pass straight through to faster-whisper.
    assert model_source("hinglish-prime") == HINDI_MODELS["hinglish-prime"].repo
    assert model_source("large-v3") == "large-v3"
    assert model_source(r"C:\models\mine") == r"C:\models\mine"


def _words(text: str) -> list[Word]:
    return [Word(t, i * 0.5, i * 0.5 + 0.4) for i, t in enumerate(text.split())]


def test_gemini_text_takes_whisper_timings() -> None:
    from peaksift.pipeline.cloud_captions import merge

    whisper = _words("bhai kitna riyal dikha hai yaar phokas karo")
    gemini = "Bhai kitna real dikha hai yaar, focus karo"
    merged, share = merge(gemini, whisper)
    assert " ".join(w.text for w in merged) == gemini
    assert share >= 0.75
    assert merged[0].start == whisper[0].start and merged[-1].end == whisper[-1].end
    assert all(a.start <= b.start for a, b in zip(merged, merged[1:], strict=False))


def test_gemini_text_rejected_when_it_does_not_match() -> None:
    from peaksift.pipeline.cloud_captions import merge

    whisper = _words("bhai kitna real dikha hai yaar focus karo")
    translated = "Brother how real it looks friend concentrate please"
    merged, _ = merge(translated, whisper)
    assert merged is whisper
    merged, _ = merge("bhai", whisper)  # far too short: a summary, not a transcript
    assert merged is whisper


def test_caption_words_off_or_failing_keeps_whisper(monkeypatch, tmp_path) -> None:
    from peaksift.pipeline import cloud_captions

    settings = Settings()
    whisper = _words("kya scene hai")
    assert cloud_captions.caption_words(tmp_path / "x.mp4", 0, 2, whisper, settings) is whisper

    settings.whisper.caption_source = "gemini"

    def boom(*_a, **_k):
        raise cloud_captions.CloudCaptionError("no key")

    monkeypatch.setattr(cloud_captions, "clip_audio_wav", lambda *_a: b"x" * 2000)
    monkeypatch.setattr(cloud_captions, "gemini_transcribe", boom)
    assert cloud_captions.caption_words(tmp_path / "x.mp4", 0, 2, whisper, settings) is whisper

    calls = []

    def fake(_audio, prompt, _settings):
        calls.append(prompt)
        return "kya scene hai", "gemini-test"

    monkeypatch.setattr(cloud_captions, "gemini_transcribe", fake)
    cache = tmp_path / "c.json"
    out = cloud_captions.caption_words(
        tmp_path / "x.mp4", 0, 2, whisper, settings, language="hi", cache=cache
    )
    assert [w.text for w in out] == ["kya", "scene", "hai"] and cache.exists()
    assert "Hinglish" in calls[0]
    # Second call is served from the cache.
    cloud_captions.caption_words(
        tmp_path / "x.mp4", 0, 2, whisper, settings, language="hi", cache=cache
    )
    assert len(calls) == 1


def test_hinglish_models_decode_in_independent_chunks(monkeypatch) -> None:
    """Fine-tuned Hinglish models repeat themselves on long audio when decoded
    window-by-window; they go through the batched, chunked path instead."""
    import faster_whisper
    from peaksift.pipeline import transcribe as tx

    calls: list[tuple[str, dict]] = []

    class FakeModel:
        def transcribe(self, _samples, **kwargs):
            calls.append(("sequential", kwargs))
            return [], None

    class FakeBatched:
        def __init__(self, model):
            self.model = model

        def transcribe(self, _samples, **kwargs):
            calls.append(("batched", kwargs))
            return [], None

    monkeypatch.setattr(faster_whisper, "BatchedInferencePipeline", FakeBatched)
    tx.decode(FakeModel(), [0.0], "hi", chunked=True)
    tx.decode(FakeModel(), [0.0], "hi", chunked=False)
    assert [c[0] for c in calls] == ["batched", "sequential"]
    assert calls[0][1]["chunk_length"] == tx.HINGLISH_CHUNK_S
    assert calls[0][1]["word_timestamps"] and calls[1][1]["word_timestamps"]
    assert calls[0][1]["suppress_tokens"] == [-1]  # fake model has no tokenizer


def test_learned_nan_token_is_forbidden() -> None:
    from types import SimpleNamespace

    from peaksift.pipeline.transcribe import _junk_tokens

    vocab = {"nan": [17622], " nan": [14067]}
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens: SimpleNamespace(ids=vocab.get(text, [1, 2]))
    )
    assert sorted(_junk_tokens(SimpleNamespace(hf_tokenizer=tokenizer))) == [14067, 17622]


# --------------------------------------------------------------------------
# A facecam that moves between positions
# --------------------------------------------------------------------------


def test_facecam_found_in_each_position_it_moves_to() -> None:
    from peaksift.signals.visual import facecam_positions

    gameplay = [FaceBox(t=i, cx=0.12, cy=0.57, w=0.03, h=0.045) for i in range(60)]
    # Leaning: the same cam, face a little higher for a while.
    leaning = [FaceBox(t=60 + i, cx=0.12, cy=0.48, w=0.03, h=0.045) for i in range(20)]
    cutscene = [FaceBox(t=80 + i, cx=0.12, cy=0.15, w=0.03, h=0.045) for i in range(15)]
    # Game characters: big crowd of faces mid-screen, and wandering ones.
    characters = [FaceBox(t=i, cx=0.5, cy=0.45, w=0.08, h=0.12) for i in range(30)]
    wandering = [
        FaceBox(t=i, cx=(i * 0.137) % 1, cy=(i * 0.311) % 1, w=0.04, h=0.06) for i in range(40)
    ]

    found = facecam_positions(gameplay + leaning + cutscene + characters + wandering, 200)
    centres = [(x + w / 2, y + h / 2) for (x, y, w, h), _ in found]
    assert len(found) == 2  # the lean merged into its cam; characters ignored
    assert centres[0][1] > 0.4 and centres[1][1] < 0.3
    assert found[0][1] == pytest.approx(80 / 200)


def test_small_faces_everywhere_are_not_a_camera_to_follow() -> None:
    crowd = [
        FaceBox(t=i, cx=(i * 0.137) % 1, cy=(i * 0.311) % 1, w=0.03, h=0.05) for i in range(90)
    ]
    assert infer_layout(crowd, 100).kind == "screen"


def test_clip_follows_the_cam_to_its_other_position() -> None:
    settings = Settings()
    stack = layouts.StackLayout(facecam=(0.08, 0.5, 0.09, 0.13), origin="auto")
    t = Timeline()
    #        gameplay (cam bottom-left) | cutscene (cam top-left) | cam off
    t.put("face_present", 0, [1.0] * 30)
    t.put("face_size", 0, [0.2] * 30)  # a big character face is on screen throughout
    t.meta["layout"] = {
        "kind": "facecam",
        "facecam": [0.08, 0.5, 0.09, 0.13],
        "facecams": [[0.08, 0.5, 0.09, 0.13], [0.08, 0.1, 0.09, 0.13]],
    }
    rows = [[float(i), 0.12, 0.57, 0.03, 0.045] for i in range(10)]
    rows += [[float(i), 0.12, 0.16, 0.03, 0.045] for i in range(10, 20)]
    t.meta["small_faces"] = rows
    runs = facecam_runs(t, 0, 30)
    assert [s for _, _, s in runs] == ["stack", "stack1", "camera"]

    crop = crop_for_clip(
        Path("x.mp4"), 0, 30, settings=settings, stack=stack, width=1920, height=1080, timeline=t
    )
    top = [seg.panels["top"] for seg in crop.segments[:2]]
    assert top[0] != top[1]  # each stretch crops the cam where it actually is
    assert top[1][1] < top[0][1]  # [x, y, w, h]: higher up the frame

    # A box the user drew is one fixed place: the cam anywhere counts as on.
    stack.origin = "manual"
    plain = crop_for_clip(
        Path("x.mp4"), 0, 20, settings=settings, stack=stack, width=1920, height=1080, timeline=t
    )
    assert len(plain.segments) == 1

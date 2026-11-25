"""Tests for the multi-signal highlight engine (peaksift.signals)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from peaksift.pipeline.transcript import Segment, Transcript, Word
from peaksift.signals import candidates as cand
from peaksift.signals import fusion, select
from peaksift.signals.audio import audio_features, speech_features
from peaksift.signals.chat import ChatMessage, chat_features, read_jsonl, write_jsonl
from peaksift.signals.timeline import Timeline, per_second
from peaksift.signals.visual import FaceBox, infer_layout


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def test_timeline_put_get_and_roundtrip(tmp_path: Path) -> None:
    t = Timeline()
    t.put("a", 5, [1.0, 2.0, 3.0])
    assert t.length == 8
    assert t.get("a").tolist() == [0, 0, 0, 0, 0, 1, 2, 3]
    assert t.covered("a").tolist() == [False] * 5 + [True] * 3
    t.put("b", 0, np.ones(10))
    assert t.length == 10 and len(t.get("a")) == 10
    t.meta["layout"] = {"kind": "facecam"}
    path = t.save(tmp_path / "timeline.npz")
    loaded = Timeline.load(path)
    assert loaded.get("a").tolist() == t.get("a").tolist()
    assert loaded.meta["layout"]["kind"] == "facecam"
    assert loaded.has("b") and not loaded.has("missing")


def test_per_second_mean_carries_forward() -> None:
    out = per_second(np.array([0.2, 3.1]), np.array([1.0, 3.0]), 5, reduce="mean")
    assert out.tolist() == [1.0, 1.0, 1.0, 3.0, 3.0]


def test_audio_features_find_a_loud_burst(tmp_path: Path) -> None:
    rate = 16000
    seconds = 130  # spans more than one 60 s read block
    t = np.arange(seconds * rate) / rate
    signal = 0.01 * np.sin(2 * np.pi * 220 * t)
    burst = (t >= 70) & (t < 72)
    signal[burst] = 0.8 * np.sign(np.sin(2 * np.pi * 3000 * t[burst]))
    wav = _write_wav(tmp_path / "a.wav", signal)

    feats = audio_features(wav)
    assert len(feats["loudness"]) == seconds
    assert int(np.argmax(feats["loudness"])) in (70, 71)
    assert feats["shout"][70] > feats["shout"][30]
    assert feats["flux"][70] > feats["flux"][30]


def test_speech_features_count_words_and_exclamations() -> None:
    words = [Word("oh", 10.1, 10.3), Word("my", 10.3, 10.4), Word("god!", 10.4, 10.9)]
    tr = Transcript(words=words, segments=[Segment("oh my god!", 10.1, 10.9, 0, 2)])
    feats = speech_features(tr, 20)
    assert feats["speech_rate"][10] == 3
    assert feats["exclaim"][10] >= 2  # "!" plus the phrase


def test_chat_features_and_jsonl(tmp_path: Path) -> None:
    msgs = [
        ChatMessage(t=5.2, text="KEKW KEKW", author="a"),
        ChatMessage(t=5.5, text="clip it POG", author="b"),
        ChatMessage(t=5.9, text="noooo WTF", author="b"),
        ChatMessage(t=7.0, text="", author="c", kind="sub"),
    ]
    feats = chat_features(msgs, 10)
    assert feats["chat_rate"][5] == 3
    assert feats["chat_unique"][5] == 2
    assert feats["chat_laugh"][5] >= 2
    assert feats["chat_hype"][5] >= 3
    assert feats["chat_shock"][5] >= 2
    assert feats["chat_events"][7] > 0

    path = tmp_path / "chat.jsonl"
    write_jsonl(path, msgs)
    path.write_text(path.read_text(encoding="utf-8") + '{"t": 9, "te', encoding="utf-8")
    assert len(read_jsonl(path)) == 4  # half-written trailing line is skipped


def test_robust_z_flags_spike_relative_to_local_baseline() -> None:
    rng = np.random.default_rng(0)
    quiet = rng.normal(0.0, 1.0, 1200)
    loud = rng.normal(20.0, 1.0, 1200)  # a constantly loud second half
    x = np.concatenate([quiet, loud])
    x[600] += 10
    x[1800] += 10
    z = fusion.robust_z(x)
    assert z[600] > 5 and z[1800] > 5
    # Being loud all the time is not a highlight.
    assert abs(float(np.median(z[1300:1700]))) < 1


def test_chat_lag_is_recovered() -> None:
    rng = np.random.default_rng(1)
    reaction = np.zeros(2000)
    for p in rng.choice(np.arange(50, 1950), 30, replace=False):
        reaction[p : p + 3] = 5
    chat = np.roll(reaction, 9) + rng.normal(0, 0.2, 2000)
    assert fusion.estimate_chat_lag(chat, reaction) == 9


def _synthetic_timeline(n: int = 3600) -> Timeline:
    rng = np.random.default_rng(2)
    t = Timeline()
    loud = rng.normal(-30, 1.5, n)
    chat = rng.poisson(2, n).astype(float)
    motion = rng.normal(0.05, 0.005, n)
    for peak in (600, 2000, 3100):
        loud[peak : peak + 4] += 18
        motion[peak - 2 : peak + 6] += 0.08
        chat[peak + 7 : peak + 15] += 25  # chat trails by ~7 s
    # Make the middle one the strongest: more signals agree there.
    shout = rng.normal(0.2, 0.02, n)
    shout[2000:2004] += 0.3
    t.put("shout", 0, shout)
    t.put("loudness", 0, loud)
    t.put("motion", 0, motion)
    t.put("chat_rate", 0, chat)
    t.meta["layout"] = {"kind": "facecam"}
    return t


def test_fuse_and_candidates_find_the_planted_moments() -> None:
    timeline = _synthetic_timeline()
    result = fusion.fuse(timeline)
    assert result.profile == "gaming"
    assert 4 <= result.chat_lag_s <= 10
    found = cand.build_candidates(result, max_candidates=5, min_duration_s=20, max_duration_s=60)
    peaks = sorted(c.peak_s for c in found[:3])
    for planted, got in zip((600, 2000, 3100), peaks, strict=True):
        assert abs(planted - got) <= 6
    best = found[0]
    assert abs(best.peak_s - 2000) <= 6
    assert best.score == 100
    for c in found:
        assert 20 <= c.duration_s <= 60.5
        assert c.start_s < c.peak_s < c.end_s
        assert c.peak_s - c.start_s >= cand.PRE_ROLL_MIN_S - 0.01
    assert "volume spike" in best.reason or "chat" in best.reason


def test_per_bucket_guarantees_a_clip_per_hour_but_ranks_globally() -> None:
    def c(peak: float, score: float) -> cand.Candidate:
        return cand.Candidate(start_s=peak - 10, end_s=peak + 20, peak_s=peak, raw_score=score)

    pool = [c(100, 9), c(500, 8), c(900, 7), c(1300, 6.5), c(4000, 1)]  # hour 2 is quiet
    groups = select.per_bucket(pool, duration_s=7200, per_bucket=3, min_per_bucket=1)
    assert len(groups) == 2
    assert [x.peak_s for x in groups[0].candidates] == [100, 500, 900]
    assert [x.peak_s for x in groups[1].candidates] == [4000]


def test_infer_layout_detects_facecam_and_camera() -> None:
    cam = [FaceBox(t=i, cx=0.88, cy=0.82, w=0.07, h=0.12) for i in range(80)]
    layout = infer_layout(cam, 100)
    assert layout.kind == "facecam"
    x, y, w, h = layout.facecam
    assert x <= 0.88 <= x + w and y <= 0.82 <= y + h

    big = [FaceBox(t=i, cx=0.5, cy=0.4, w=0.25, h=0.4) for i in range(90)]
    assert infer_layout(big, 100).kind == "camera"
    assert infer_layout([], 100).kind == "screen"

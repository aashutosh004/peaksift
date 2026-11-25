"""Caption editor, subtitle files, and fix-a-word-everywhere."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from peaksift import config, paths
from peaksift.api.captions import clean_words, replace_in_words
from peaksift.api.schemas import WordOut
from peaksift.db import store
from peaksift.db.models import Clip, Export, Job, Source, new_id
from peaksift.pipeline.captions import subtitle_text
from peaksift.pipeline.clip_words import gemini_cache, words_for_clip
from peaksift.pipeline.transcript import Segment, Transcript, Word


def _words(text: str, start: float = 0.0) -> list[Word]:
    return [Word(t, start + i * 0.5, start + i * 0.5 + 0.4) for i, t in enumerate(text.split())]


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_subtitles_are_clip_relative_in_both_formats() -> None:
    words = _words("kya scene hai bhai", start=100.0)
    srt = subtitle_text(words, "srt", time_offset_s=100.0)
    vtt = subtitle_text(words, "vtt", time_offset_s=100.0)
    assert "00:00:00,000 --> 00:00:01,900" in srt and "kya scene hai bhai" in srt
    assert vtt.startswith("WEBVTT") and "00:00:00.000 --> 00:00:01.900" in vtt
    with pytest.raises(ValueError):
        subtitle_text(words, "ass")


def test_replace_keeps_punctuation_and_timing() -> None:
    words = _words("bhai phokas, karo PHOKAS")
    out, count = replace_in_words(words, "phokas", "focus")
    assert count == 2
    assert [w.text for w in out] == ["bhai", "focus,", "karo", "focus"]
    assert (out[1].start, out[1].end) == (words[1].start, words[1].end)


def test_replace_phrases_merges_or_maps_and_can_delete() -> None:
    words = _words("o bhai under water discoveries yaar")
    same, n = replace_in_words(words, "under water", "underwater discoveries")
    assert n == 1 and [w.text for w in same][2:4] == ["underwater", "discoveries"]
    merged, n = replace_in_words(words, "under water discoveries", "Underwater Discovery")
    assert n == 1 and merged[2].text == "Underwater Discovery"
    assert (merged[2].start, merged[2].end) == (words[2].start, words[4].end)
    gone, n = replace_in_words(words, "yaar", "")
    assert n == 1 and [w.text for w in gone][-1] == "discoveries"


def test_clean_words_drops_empty_and_orders() -> None:
    words = [
        WordOut(text="b", start=2.0, end=1.0),
        WordOut(text="  ", start=1.0, end=1.5),
        WordOut(text=" a  ", start=0.5, end=0.9),
    ]
    out = clean_words(words)
    assert [w["text"] for w in out] == ["a", "b"]
    assert out[1]["end"] == out[1]["start"] == 2.0


# --------------------------------------------------------------------------
# Which words a clip shows
# --------------------------------------------------------------------------


@pytest.fixture
def job_with_clip(initialised_db):
    source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4", title="t"))
    settings = config.Settings()
    job = store.create_job(
        Job(id=new_id(), source_id=source.id, status="done", settings=settings.model_dump())
    )
    words = _words("intro words here") + _words("bhai phokas karo yaar", start=10.0)
    transcript = Transcript(
        words=words, segments=[Segment("all", 0, 12, 0, len(words) - 1)], language="hi"
    )
    transcript.save(paths.job_work_dir(job.id) / "transcript.json")
    clips = store.replace_clips(
        job.id, [Clip(id=new_id(), job_id=job.id, start_s=10.0, end_s=12.0, rank=1)]
    )
    out = paths.exports_dir() / job.id / "01_bhai-phokas_9x16.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"x")
    store.create_export(Export(id=new_id(), clip_id=clips[0].id, path=str(out)))
    return job, clips[0], transcript


def test_words_for_clip_prefers_edits_then_gemini_then_whisper(job_with_clip) -> None:
    job, clip, transcript = job_with_clip
    settings = config.Settings()
    work = paths.job_work_dir(job.id)

    words, origin = words_for_clip(clip, transcript, settings, work_dir=work)
    assert origin == "whisper" and [w.text for w in words] == ["bhai", "phokas", "karo", "yaar"]

    settings.whisper.caption_source = "gemini"
    cache = gemini_cache(work, clip.id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({"range": "10.00-12.00", "words": [["Bhai", 10.0, 10.4, None]]}),
        encoding="utf-8",
    )
    words, origin = words_for_clip(clip, transcript, settings, work_dir=work)
    assert origin == "gemini" and words[0].text == "Bhai"

    from peaksift.api.captions import _save

    _save(clip.id, [{"text": "edited", "start": 10.0, "end": 10.5, "speaker": None}])
    words, origin = words_for_clip(clip, transcript, settings, work_dir=work)
    assert origin == "edited" and [w.text for w in words] == ["edited"]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def api(initialised_db, monkeypatch) -> TestClient:
    monkeypatch.setenv("PEAKSIFT_NO_WORKER", "1")
    from peaksift.app import create_app

    with TestClient(create_app()) as client:
        yield client


def test_editor_round_trip(api: TestClient, job_with_clip) -> None:
    job, clip, _ = job_with_clip
    url = f"/api/ui/clips/{clip.id}/captions"

    data = api.get(url).json()
    assert data["origin"] == "whisper" and data["video_url"].startswith("/api/exports/")
    assert [w["text"] for w in data["words"]] == ["bhai", "phokas", "karo", "yaar"]

    words = data["words"]
    words[1]["text"] = "focus"
    words[3]["text"] = ""  # deleted
    saved = api.put(url, json={"words": words}).json()
    assert saved["origin"] == "edited"
    assert [w["text"] for w in saved["words"]] == ["bhai", "focus", "karo"]

    srt = api.get(f"/api/ui/clips/{clip.id}/subtitles.srt")
    assert srt.status_code == 200 and "bhai focus karo" in srt.text
    assert 'filename="01_bhai-phokas_9x16.srt"' in srt.headers["content-disposition"]
    assert api.get(f"/api/ui/clips/{clip.id}/subtitles.vtt").text.startswith("WEBVTT")
    assert api.get(f"/api/ui/clips/{clip.id}/subtitles.ass").status_code == 404
    # The older words endpoint agrees with the editor.
    assert [w["text"] for w in api.get(f"/api/clips/{clip.id}/words").json()] == [
        "bhai",
        "focus",
        "karo",
    ]

    reset = api.delete(url).json()
    assert reset["origin"] == "whisper" and reset["words"][1]["text"] == "phokas"


def test_replace_everywhere(api: TestClient, job_with_clip) -> None:
    job, clip, _ = job_with_clip
    out = api.post(f"/api/ui/jobs/{job.id}/replace", json={"find": "PHOKAS", "replace": "focus"})
    assert out.json() == {"replaced": 1, "clips": [1]}
    words = api.get(f"/api/ui/clips/{clip.id}/captions").json()["words"]
    assert [w["text"] for w in words] == ["bhai", "focus", "karo", "yaar"]
    none = api.post(f"/api/ui/jobs/{job.id}/replace", json={"find": "zzz", "replace": "y"})
    assert none.json() == {"replaced": 0, "clips": []}
    assert api.get("/api/ui/clips/nope/captions").status_code == 404


def test_rerender_waits_for_a_locked_file_then_explains(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from peaksift import rerender

    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    real_unlink = Path.unlink
    calls = {"n": 0}

    def flaky(self, missing_ok=False):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("in use")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    rerender._remove(target, wait_s=0)
    assert not target.exists() and calls["n"] == 3

    monkeypatch.setattr(
        Path,
        "unlink",
        lambda self, missing_ok=False: (_ for _ in ()).throw(PermissionError("in use")),
    )
    with pytest.raises(rerender.RerenderError, match="open in another program"):
        rerender._remove(target, attempts=2, wait_s=0)

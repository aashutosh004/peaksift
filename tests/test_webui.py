"""The built-in web UI: static files, UI endpoints, and job-settings plumbing."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from peaksift import config
from peaksift.db import store
from peaksift.db.models import Clip, Job, Source, new_id


@pytest.fixture
def api(initialised_db, monkeypatch) -> TestClient:
    monkeypatch.setenv("PEAKSIFT_NO_WORKER", "1")
    from peaksift.app import create_app

    with TestClient(create_app()) as client:
        yield client


def test_root_serves_the_web_ui(api: TestClient) -> None:
    page = api.get("/")
    assert page.status_code == 200 and "<title>PeakSift</title>" in page.text
    script = api.get("/ui/app.js")
    assert script.status_code == 200
    # No build step means stable file names: browsers must revalidate.
    assert script.headers["cache-control"] == "no-cache"
    assert page.headers["cache-control"] == "no-cache"
    assert api.get("/ui/style.css").status_code == 200


def test_ui_endpoints_validate(api: TestClient) -> None:
    assert api.post("/api/ui/start", json={"url": "not a link"}).status_code == 400
    assert api.get("/api/ui/tasks/nope").status_code == 404
    assert api.get("/api/ui/live").json() == []
    assert api.get("/api/ui/live/nope").status_code == 404
    assert api.post("/api/ui/live/nope/snapshot", json={"seconds": 30}).status_code == 409
    assert api.get("/api/ui/jobs/nope/report").json() == {"clips": []}
    assert api.post("/api/ui/jobs/nope/rerender", json={}).status_code == 404


def test_settings_api_exposes_new_sections(api: TestClient) -> None:
    out = api.get("/api/settings").json()
    assert {"detection", "selection", "live"} <= out.keys()
    assert "hinglish-prime" in [m["key"] for m in out["hindi_models"]]
    put = api.put(
        "/api/settings", json={"selection": {"view": "per_hour"}, "live": {"clips_per_hour": 6}}
    )
    assert put.status_code == 200
    assert (
        put.json()["selection"]["view"] == "per_hour" and put.json()["live"]["clips_per_hour"] == 6
    )


def test_job_overrides_include_new_options(api: TestClient) -> None:
    source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4", title="t"))
    body = {
        "source_id": source.id,
        "settings": {
            "max_clips": 3,
            "view": "per_hour",
            "per_bucket": 2,
            "layout": "gaming",
            "split_ratio": 0.5,
            "facecam_position": "bottom",
            "llm_rerank": "off",
            "profile": "podcast",
            "two_person": "track",
            "hindi_model": "hinglish-apex",
            "caption_source": "gemini",
        },
    }
    job = api.post("/api/jobs", json=body).json()
    saved = store.get_job(job["id"]).settings
    assert saved["clips"]["max_clips"] == 3
    assert saved["selection"] == {**saved["selection"], "view": "per_hour", "per_bucket": 2}
    assert saved["export"]["facecam_position"] == "bottom" and saved["export"]["split_ratio"] == 0.5
    assert saved["detection"]["llm_rerank"] == "off" and saved["detection"]["profile"] == "podcast"
    assert saved["whisper"]["hindi_model"] == "hinglish-apex"
    assert saved["whisper"]["caption_source"] == "gemini"


def test_queue_runs_jobs_with_their_own_settings(initialised_db) -> None:
    from peaksift.jobs.queue import _job_settings

    saved = config.load()
    saved._fallback_secrets = {"openai": "sk-test"}
    config.save(saved)
    custom = config.Settings()
    custom.clips.max_clips = 3
    job = Job(id=new_id(), source_id="s", settings=custom.model_dump(mode="json"))
    settings = _job_settings(job)
    assert settings.clips.max_clips == 3
    assert config.get_secret("openai", settings) == "sk-test"
    assert _job_settings(Job(id=new_id(), source_id="s")).clips.max_clips == saved.clips.max_clips


def test_interrupted_live_jobs_are_not_requeued(initialised_db) -> None:
    from peaksift.jobs.queue import JobQueue

    source = store.create_source(Source(id=new_id(), type="live", path="x"))
    live = store.create_job(
        Job(id=new_id(), source_id=source.id, status="running", current_stage="live")
    )
    normal = store.create_job(
        Job(id=new_id(), source_id=source.id, status="running", current_stage="export")
    )
    rerender = store.create_job(
        Job(id=new_id(), source_id=source.id, status="running", current_stage="rerender")
    )
    JobQueue()._requeue_interrupted()
    assert store.get_job(live.id).status == "failed"
    assert store.get_job(normal.id).status == "queued"
    assert store.get_job(rerender.id).status == "done"


def test_jobs_running_in_another_process_are_left_alone(initialised_db) -> None:
    """The web app must not restart a job an `peaksift clip` window is running."""
    import json
    import subprocess
    import sys

    from peaksift import paths
    from peaksift.jobs.owner import _process_started, claim, owned_elsewhere
    from peaksift.jobs.queue import JobQueue

    source = store.create_source(Source(id=new_id(), type="upload", path="x"))
    busy = store.create_job(
        Job(id=new_id(), source_id=source.id, status="running", current_stage="transcribe")
    )
    crashed = store.create_job(
        Job(id=new_id(), source_id=source.id, status="running", current_stage="transcribe")
    )
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # noqa: S603
    try:
        started = _process_started(other.pid)
        assert started is not None
        for job, pid, when in ((busy, other.pid, started), (crashed, 999_999_999, 0.0)):
            folder = paths.job_work_dir(job.id)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "owner.json").write_text(json.dumps({"pid": pid, "started": when}))
        assert owned_elsewhere(busy.id) and not owned_elsewhere(crashed.id)
        JobQueue()._requeue_interrupted()
        assert store.get_job(busy.id).status == "running"
        assert store.get_job(crashed.id).status == "queued"
    finally:
        other.kill()
        other.wait()
    assert not owned_elsewhere(busy.id)

    # Our own claim is never "elsewhere", and is removed when the work ends.
    with claim(crashed.id):
        assert (paths.job_work_dir(crashed.id) / "owner.json").exists()
        assert not owned_elsewhere(crashed.id)
    assert not (paths.job_work_dir(crashed.id) / "owner.json").exists()


def test_moments_strip_data(api: TestClient) -> None:
    import numpy as np
    from peaksift import paths
    from peaksift.signals.timeline import Timeline

    source = store.create_source(Source(id=new_id(), type="upload", path="x.mp4", title="t"))
    source_id = source.id
    job = store.create_job(Job(id=new_id(), source_id=source_id, status="done"))
    store.replace_clips(
        job.id,
        [Clip(id=new_id(), job_id=job.id, start_s=40, end_s=70, rank=1, peak_s=55.0)],
    )
    # No signals saved yet: clips only, no curve.
    out = api.get(f"/api/ui/jobs/{job.id}/moments").json()
    assert out["points"] == [] and out["clips"][0]["peak_s"] == 55.0

    timeline = Timeline()
    loud = np.zeros(120, dtype=np.float32)
    loud[55] = 10
    timeline.put("loudness", 0, loud)
    timeline.save(paths.job_work_dir(job.id) / "timeline.npz")
    out = api.get(f"/api/ui/jobs/{job.id}/moments").json()
    assert len(out["points"]) == 120 and max(out["points"]) == 1.0
    assert abs(out["points"].index(1.0) - 55) <= 2  # scoring smooths by a second or so
    assert api.get("/api/ui/jobs/nope/moments").status_code == 404

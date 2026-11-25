"""Facecam/gameplay layouts, the picker API, the two-person split, OBS input and snapshots."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from peaksift import layouts, paths
from peaksift.config import Settings
from peaksift.db import store
from peaksift.db.models import Clip, Job, Source, new_id
from peaksift.pipeline import export
from peaksift.pipeline.captions import get_style
from peaksift.pipeline.framing import crop_for_clip, layout_choice
from peaksift.pipeline.reframe import (
    ReframeConfig,
    _merge_split_runs,
    _split_pair,
    _split_segment,
    build_stack_path,
)
from peaksift.pipeline.reframe.croppath import Strategy
from peaksift.pipeline.reframe.faces import FaceObservation
from peaksift.pipeline.reframe.tracker import FaceTrack

# --------------------------------------------------------------------------
# Layout presets and resolution
# --------------------------------------------------------------------------


def test_layout_resolution_priority(initialised_db) -> None:
    meta = {"layout": {"kind": "facecam", "facecam": [0.8, 0.0, 0.18, 0.25]}}
    auto = layouts.resolve(
        job_id="j1", platform="twitch", channel="Someone", timeline_meta=meta, split=0.4
    )
    assert auto.origin == "auto" and auto.split == 0.4 and auto.gameplay is None

    preset = layouts.StackLayout(facecam=(0.0, 0.7, 0.2, 0.3), gameplay=(0.2, 0, 0.6, 1), split=0.5)
    layouts.save_preset("twitch:someone", preset)
    got = layouts.resolve(job_id="j1", platform="twitch", channel="SOMEONE", timeline_meta=meta)
    assert got.origin == "preset" and got.facecam == (0.0, 0.7, 0.2, 0.3)

    layouts.save_job_layout("j1", layouts.StackLayout(facecam=(0.1, 0.1, 0.2, 0.2), split=0.3))
    got = layouts.resolve(job_id="j1", platform="twitch", channel="someone", timeline_meta=meta)
    assert got.origin == "manual" and got.split == 0.3

    assert layouts.resolve(job_id=None, platform="", channel="", timeline_meta={}) is None
    assert "twitch:someone" in layouts.list_presets()
    assert layouts.delete_preset("twitch:someone") and not layouts.list_presets()


def test_layout_clamps_bad_values() -> None:
    layout = layouts.StackLayout(facecam=(-0.5, 0.9, 2.0, 0.5), split=0.95, facecam_position="side")
    x, y, w, h = layout.facecam
    assert x >= 0 and y + h <= 1.0 and w <= 1.0
    assert layout.split == layouts.MAX_SPLIT and layout.facecam_position == "top"


def test_stack_path_with_gameplay_box_split_and_bottom_facecam() -> None:
    path = build_stack_path(
        1920,
        1080,
        20.0,
        (0.8, 0.75, 0.2, 0.25),
        gameplay=(0.1, 0.05, 0.7, 0.9),
        split=0.4,
        facecam_position="bottom",
        exact=True,
    )
    panels = path.segments[0].panels
    cam_h = int(round(1920 * 0.4 / 2)) * 2
    # Gameplay on top now, so the split stored is the gameplay's share.
    assert abs(panels["split"] - (1920 - cam_h) / 1920) < 1e-6
    gx, gy, gw, gh = panels["top"]
    # Exact mode keeps the crop inside the drawn gameplay box.
    assert gx >= int(0.1 * 1920) - 2 and gx + gw <= int(0.8 * 1920) + 2
    assert gy >= int(0.05 * 1080) - 2 and gy + gh <= int(0.95 * 1080) + 2
    assert abs(gw / gh - 1080 / (1920 - cam_h)) < 0.02
    cx, cy, cw, ch = panels["bottom"]
    assert cx >= int(0.8 * 1920) - 2 and cy + ch <= 1080
    assert abs(cw / ch - 1080 / cam_h) < 0.03

    request = export.ExportRequest(
        source=Path("in.mp4"),
        destination=Path("out.mp4"),
        start_s=0,
        end_s=20,
        crop_path=path,
        words=[],
        style=get_style("bold_pop"),
    )
    assert "vstack=inputs=2" in export.build_video_filtergraph(request, subtitle_name=None)


def test_layout_choice_and_crop_for_clip() -> None:
    s = Settings()
    stack = layouts.StackLayout(facecam=(0.8, 0.0, 0.2, 0.25))
    assert layout_choice(s, stack) == "gaming"
    assert layout_choice(s, None) == "track"
    s.export.ratio = "16:9"
    assert layout_choice(s, stack) == "track"
    s.export.ratio = "9:16"
    s.export.layout = "center"
    crop = crop_for_clip(Path("x.mp4"), 0, 10, settings=s, stack=stack, width=1920, height=1080)
    assert crop.segments[0].strategy == Strategy.GENERAL
    s.export.layout = "auto"
    crop = crop_for_clip(Path("x.mp4"), 0, 10, settings=s, stack=stack, width=1920, height=1080)
    assert crop.segments[0].strategy == Strategy.STACK


# --------------------------------------------------------------------------
# Two-person split
# --------------------------------------------------------------------------


def _track(track_id: int, cx: float, width: float, n: int = 50, fps: float = 5.0) -> FaceTrack:
    obs = [
        FaceObservation(
            t=i / fps, cx=cx, cy=400, width=width, height=width * 1.3, eye_y=380, mar=0.1
        )
        for i in range(n)
    ]
    return FaceTrack(id=track_id, observations=obs)


def test_two_person_split_detects_a_real_pair() -> None:
    config = ReframeConfig()
    left, right = _track(1, 600, 180), _track(2, 1350, 170)
    pair = _split_pair([right, left], 0, 10, 1920, config)
    assert pair == (left, right)  # left person goes on top
    segment = _split_segment(pair, start_s=0, end_s=10, source_w=1920, source_h=1080, config=config)
    assert segment.strategy == Strategy.STACK and segment.panels["split"] == 0.5
    top, bottom = segment.panels["top"], segment.panels["bottom"]
    assert top[0] < 600 < top[0] + top[2] and bottom[0] < 1350 < bottom[0] + bottom[2]
    assert top[0] + top[2] <= 1350  # the right person is not in the top panel
    assert abs(top[2] / top[3] - 1080 / 960) < 0.02


@pytest.mark.parametrize(
    "tracks",
    [
        [_track(1, 900, 180), _track(2, 1100, 170)],  # too close together
        [_track(1, 600, 180), _track(2, 1400, 40)],  # background face, too small
        [_track(1, 600, 180), _track(2, 1400, 170, n=8)],  # barely on screen
        [_track(1, 600, 180)],  # one person
    ],
)
def test_two_person_split_guards(tracks) -> None:
    assert _split_pair(tracks, 0, 10, 1920, ReframeConfig()) is None


def test_consecutive_split_segments_merge() -> None:
    config = ReframeConfig()
    pair = (_track(1, 600, 180), _track(2, 1350, 170))
    a = _split_segment(pair, start_s=0, end_s=4, source_w=1920, source_h=1080, config=config)
    b = _split_segment(pair, start_s=4, end_s=9, source_w=1920, source_h=1080, config=config)
    merged = _merge_split_runs([a, b], 1920)
    assert len(merged) == 1 and merged[0].end_s == 9


# --------------------------------------------------------------------------
# Picker API
# --------------------------------------------------------------------------


@pytest.fixture
def picker_job(initialised_db, tmp_path: Path) -> str:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    video = tmp_path / "v.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine",
            "-t",
            "4",
            "-shortest",
            str(video),
        ],
        check=True,
    )
    source = store.create_source(
        Source(
            id=new_id(),
            type="url",
            path=str(video),
            title="Test",
            duration_s=4.0,
            width=640,
            height=360,
            platform="twitch",
            channel="Tester",
        )
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
    meta = paths.job_work_dir(job.id) / "timeline.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"layout": {"kind": "facecam", "facecam": [0.7, 0.0, 0.25, 0.3]}}))
    store.replace_clips(job.id, [Clip(id=new_id(), job_id=job.id, start_s=0, end_s=3, peak_s=2.0)])
    return job.id


@pytest.fixture
def api(monkeypatch) -> TestClient:
    monkeypatch.setenv("PEAKSIFT_NO_WORKER", "1")
    from peaksift.app import create_app

    with TestClient(create_app()) as client:
        yield client


def test_picker_api_roundtrip(picker_job: str, api: TestClient) -> None:
    page = api.get(f"/layout-picker?job={picker_job}")
    assert page.status_code == 200 and "Facecam" in page.text

    ctx = api.get(f"/api/layouts/jobs/{picker_job}/context").json()
    assert ctx["preset_key"] == "twitch:tester"
    assert ctx["detected_facecam"] == [0.7, 0.0, 0.25, 0.3]
    assert ctx["layout"]["origin"] == "auto" and ctx["width"] == 640

    frame = api.get(f"/api/layouts/jobs/{picker_job}/frame?t=1.5")
    assert frame.status_code == 200 and frame.content[:2] == b"\xff\xd8"

    body = {
        "facecam": [0.0, 0.6, 0.3, 0.4],
        "gameplay": [0.3, 0.0, 0.7, 1.0],
        "split": 0.45,
        "facecam_position": "bottom",
        "save_for_channel": True,
    }
    saved = api.put(f"/api/layouts/jobs/{picker_job}", json=body).json()
    assert saved["saved"]["channel"] == "twitch:tester"
    again = api.get(f"/api/layouts/jobs/{picker_job}/context").json()["layout"]
    assert again["origin"] == "manual" and again["facecam_position"] == "bottom"
    assert layouts.load_preset("twitch:tester").split == 0.45

    assert api.get("/api/layouts/jobs/nope/context").status_code == 404


# --------------------------------------------------------------------------
# OBS and snapshots
# --------------------------------------------------------------------------


def test_obs_urls() -> None:
    from peaksift.live.session import is_obs_url, obs_listen_url

    assert is_obs_url("obs") and is_obs_url("rtmp://127.0.0.1:1935/live/x")
    assert not is_obs_url("https://twitch.tv/x")
    assert obs_listen_url("obs", 1940) == "rtmp://127.0.0.1:1940/live/stream"


def test_snapshot_requests_queue_and_trigger_file(tmp_path: Path) -> None:
    from peaksift.live import session as live

    s = live.LiveSession("obs", Settings())
    s.work_dir = tmp_path
    s.origin_epoch = 1000.0
    s._live_edge = lambda: 500.0  # type: ignore[method-assign]
    s.request_snapshot(30)
    (tmp_path / live.SNAPSHOT_TRIGGER).write_text("60\n")

    class NoKeys:
        def read(self):
            return ["l"]

    s._poll_snapshot_triggers(NoKeys())  # type: ignore[arg-type]
    windows = sorted(s._pending_snapshots)
    assert windows == [(440.0, 500.0), (440.0, 500.0), (470.0, 500.0)]
    assert not (tmp_path / live.SNAPSHOT_TRIGGER).exists()
    s.request_snapshot(9999)
    assert s._snapshot_requests[-1] == 180.0


def test_snap_command_targets_running_live_job(initialised_db) -> None:
    from peaksift.live.session import SNAPSHOT_TRIGGER, request_snapshot_for_running_session

    assert request_snapshot_for_running_session(30) is None
    source = store.create_source(Source(id=new_id(), type="live", path="x"))
    job = store.create_job(
        Job(id=new_id(), source_id=source.id, status="running", current_stage="live")
    )
    paths.job_work_dir(job.id).mkdir(parents=True, exist_ok=True)
    trigger = request_snapshot_for_running_session(45)
    assert trigger == paths.job_work_dir(job.id) / SNAPSHOT_TRIGGER
    assert trigger.read_text().strip() == "45"


def test_face_handoff_adds_a_missed_cut() -> None:
    from peaksift.pipeline.reframe import split_at_track_handoffs
    from peaksift.pipeline.reframe.scenes import Shot

    host = _track(1, 650, 190, n=65)  # 0.0 - 12.8 s at 5 fps
    guest = FaceTrack(
        id=2,
        observations=[
            FaceObservation(
                t=13.6 + i / 5, cx=660, cy=400, width=190, height=250, eye_y=380, mar=0.1
            )
            for i in range(100)
        ],
    )
    shots = split_at_track_handoffs([Shot(0.0, 33.6)], [host, guest])
    assert len(shots) == 2 and 12.8 < shots[0].end_s < 13.6

    # A third person on screen through the whole moment means it is not a cut.
    wide = _track(3, 1300, 150, n=170)
    assert len(split_at_track_handoffs([Shot(0.0, 33.6)], [host, guest, wide])) == 1


def test_snapshot_requested_during_last_iteration_is_not_lost(tmp_path: Path, monkeypatch) -> None:
    """A snapshot asked for while the final chunk is processed still gets made."""
    from peaksift.live import session as live
    from peaksift.live.recorder import Segment

    s = live.LiveSession("obs", Settings(), keyboard=False)
    s.work_dir = tmp_path
    s.origin_epoch = 0.0
    s._live_edge = lambda: 90.0  # type: ignore[method-assign]

    class Recorder:
        def __init__(self) -> None:
            import threading

            self.ended = threading.Event()

        def segments(self):
            return [Segment(0, tmp_path / "a.ts", 0.0, 100.0)]

        def stop(self) -> None:
            pass

    recorder = Recorder()
    calls = {"n": 0}

    def start() -> None:
        s.recorder = recorder

    def process() -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            s.request_snapshot(30)  # arrives mid-iteration...
            recorder.ended.set()  # ...and the stream ends in the same iteration
            return True
        return False

    emitted: list[tuple[float, float]] = []
    monkeypatch.setattr(s, "_start", start)
    monkeypatch.setattr(s, "_process_pending_live", process)
    monkeypatch.setattr(s, "_process_one_backfill", lambda: False)
    monkeypatch.setattr(s, "_rescore", lambda final: None)
    monkeypatch.setattr(s, "_emit_snapshot", lambda a, b: emitted.append((a, b)))
    monkeypatch.setattr(s, "_finish", lambda: [])
    s.run()
    assert emitted == [(60.0, 90.0)]

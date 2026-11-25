"""Tests for platform adapters, chat parsers, the v2 schema, the gaming layout, and live merging."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from peaksift import sources
from peaksift.db import schema
from peaksift.pipeline import export
from peaksift.pipeline.captions import get_style
from peaksift.pipeline.reframe import build_stack_path
from peaksift.pipeline.reframe.croppath import CropPath, Strategy
from peaksift.pipeline.transcript import Segment, Transcript, Word
from peaksift.sources import kick, twitch
from peaksift.sources import youtube as yt


def test_platform_detection_and_live_scope() -> None:
    assert sources.platform_of("https://www.youtube.com/watch?v=x") == "youtube"
    assert sources.platform_of("https://youtu.be/x") == "youtube"
    assert sources.platform_of("https://www.twitch.tv/somechannel") == "twitch"
    assert sources.platform_of("https://kick.com/someone") == "kick"
    assert sources.platform_of("https://www.tiktok.com/@a/video/1") == "tiktok"
    assert sources.platform_of("https://fb.watch/abc") == "facebook"
    assert sources.platform_of("https://www.instagram.com/reel/abc") == "instagram"
    assert sources.platform_of("https://example.com/v.mp4") == "other"
    assert sources.supports_live("twitch") and not sources.supports_live("instagram")
    assert sources.twitch_channel("https://www.twitch.tv/Shroud") == "shroud"
    assert sources.twitch_channel("https://www.twitch.tv/videos/123") is None
    assert sources.twitch_vod_id("https://www.twitch.tv/videos/123456") == "123456"
    assert sources.kick_channel("https://kick.com/xqc") == "xqc"


def test_twitch_irc_parsing() -> None:
    line = (
        "@badge-info=;bits=500;display-name=Viewer1;emotes= :viewer1!viewer1@viewer1.tmi.twitch.tv "
        "PRIVMSG #chan :Cheer500 that was insane"
    )
    message = twitch.parse_irc_line(line, 12.5)
    assert message.kind == "bits" and message.amount == 5.0 and message.author == "Viewer1"
    assert message.text == "Cheer500 that was insane"

    plain = twitch.parse_irc_line(":a!a@a.tmi.twitch.tv PRIVMSG #chan :KEKW", 3.0)
    assert plain.kind == "msg" and plain.text == "KEKW" and plain.author == "a"

    sub = twitch.parse_irc_line(
        "@msg-id=resub;display-name=Fan;system-msg=Fan\\ssubscribed "
        ":tmi.twitch.tv USERNOTICE #chan",
        7.0,
    )
    assert sub.kind == "sub" and sub.text == "Fan subscribed"
    assert twitch.parse_irc_line(":tmi.twitch.tv 001 justinfan :Welcome", 0) is None


def test_youtube_live_chat_parsing() -> None:
    replay = {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "61500",
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            "liveChatTextMessageRenderer": {
                                "message": {
                                    "runs": [{"text": "no way "}, {"emoji": {"emojiId": "😂"}}]
                                },
                                "authorName": {"simpleText": "Bob"},
                            }
                        }
                    }
                }
            ],
        }
    }
    [message] = yt.parse_live_chat_line(json.dumps(replay))
    assert message.t == 61.5 and message.text == "no way 😂" and message.author == "Bob"

    paid = {
        "addChatItemAction": {
            "item": {
                "liveChatPaidMessageRenderer": {
                    "timestampUsec": str(int((1000 + 30) * 1e6)),
                    "purchaseAmountText": {"simpleText": "$20.00"},
                    "authorName": {"simpleText": "Rich"},
                    "message": {"simpleText": "GG"},
                }
            }
        }
    }
    [sc] = yt.parse_live_chat_line(json.dumps(paid), origin_epoch=1000)
    assert sc.kind == "superchat" and sc.amount == 20.0 and sc.t == 30.0


def test_comment_timestamps() -> None:
    comments = [
        {"text": "1:02:03 had me dying", "like_count": 100},
        {"text": "best part 12:34 and 0:05", "like_count": 0},
        {"text": "99:99 is not real", "like_count": 0},
    ]
    stamps = yt.comment_timestamps(comments, duration_s=4000)
    assert stamps.count(3723.0) == 5  # liked comment counts more
    assert 754.0 in stamps and 5.0 in stamps


def test_kick_pusher_parsing() -> None:
    raw = json.dumps(
        {
            "event": "App\\Events\\ChatMessageEvent",
            "data": json.dumps({"content": "W stream", "sender": {"username": "k1"}}),
        }
    )
    message = kick.parse_pusher_message(raw, 4.0)
    assert message.text == "W stream" and message.author == "k1"
    assert kick.parse_pusher_message('{"event":"pusher:connection_established"}', 1) is None


def test_v2_migration_preserves_jobs_and_allows_new_types(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "db.sqlite")
    conn.execute("PRAGMA foreign_keys = ON")
    schema.MIGRATIONS[0](conn)
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO sources (id, type, path, created_at) VALUES ('s1', 'youtube', '/x', 'now')"
    )
    conn.execute(
        "INSERT INTO jobs (id, source_id, status, created_at, updated_at) "
        "VALUES ('j1', 's1', 'done', 'now', 'now')"
    )
    conn.commit()

    assert schema.migrate(conn) == schema.SCHEMA_VERSION
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1  # no cascade delete
    assert conn.execute("SELECT platform FROM sources").fetchone()[0] == "youtube"
    conn.execute(
        "INSERT INTO sources (id, type, path, created_at, platform) "
        "VALUES ('s2', 'live', '/y', 'now', 'twitch')"
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(clips)")}
    assert {"peak_s", "signals_json", "group_label"} <= cols
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_stack_layout_crop_and_filtergraph() -> None:
    path = build_stack_path(1920, 1080, 30.0, (0.78, 0.70, 0.2, 0.28))
    [segment] = path.segments
    assert segment.strategy == Strategy.STACK
    tx, ty, tw, th = segment.panels["top"]
    assert tx + tw <= 1920 and ty + th <= 1080 and tw % 2 == 0 and th % 2 == 0
    assert abs(tw / th - 1080 / 672) < 0.02  # matches the top panel's shape
    bx, _, bw, bh = segment.panels["bottom"]
    assert bh == 1080 and abs(bw / bh - 1080 / 1248) < 0.01

    restored = CropPath.from_dict(json.loads(json.dumps(path.to_dict())))
    assert restored.segments[0].panels == segment.panels

    request = export.ExportRequest(
        source=Path("in.mp4"),
        destination=Path("out.mp4"),
        start_s=0,
        end_s=30,
        crop_path=path,
        words=[],
        style=get_style("bold_pop"),
    )
    graph = export.build_video_filtergraph(request, subtitle_name=None)
    assert "vstack=inputs=2" in graph and f"crop={tw}:{th}:{tx}:{ty}" in graph
    assert graph.endswith("[vout]")


def test_live_transcript_merge_offsets_and_dedupes() -> None:
    from peaksift.config import Settings
    from peaksift.live.session import LiveSession

    session = LiveSession("https://www.twitch.tv/x", Settings())

    def tr(words: list[tuple[str, float, float]]) -> Transcript:
        ws = [Word(t, a, b) for t, a, b in words]
        return Transcript(
            words=ws,
            segments=[
                Segment(" ".join(w for w, _, _ in words), ws[0].start, ws[-1].end, 0, len(ws) - 1)
            ],
        )

    session.chunk_transcripts = [
        (60.0, tr([("second", 1.0, 1.5), ("chunk.", 1.5, 2.0)])),
        (0.0, tr([("first", 10.0, 10.4), ("chunk.", 10.4, 11.0)])),
    ]
    merged = session.transcript()
    assert [w.text for w in merged.words] == ["first", "chunk.", "second", "chunk."]
    assert merged.words[2].start == 61.0
    assert merged.segments[1].first_word == 2 and merged.segments[1].last_word == 3

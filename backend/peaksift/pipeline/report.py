"""Clip report — ``clips.json`` and ``clips.md`` next to the exported files.

The report answers the three questions a user has after a long run: which
clips, where in the source, and *why* each one was picked (which signals
spiked, what the LLM thought). Grouped by hour or chapter when that view was
used, so a five-hour stream reads as "best of each hour".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..db import store
from ..db.models import Clip, Source


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def write_report(
    destination_dir: Path,
    *,
    source: Source,
    clips: list[Clip],
    view: str,
    meta: dict[str, Any] | None = None,
    files: dict[str, str] | None = None,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    meta = meta or {}
    if files is None:
        files = {}
        for clip in clips:
            for export in store.list_exports(clip.id):
                files[clip.id] = export.path

    rows = [
        {
            "rank": c.rank,
            "title": c.title,
            "hook": c.hook,
            "score": c.score,
            "start": hms(c.start_s),
            "end": hms(c.end_s),
            "start_s": c.start_s,
            "end_s": c.end_s,
            "peak_s": c.peak_s,
            "group": c.group_label,
            "reason": c.reason,
            "signals": c.signals,
            "file": files.get(c.id, ""),
        }
        for c in sorted(clips, key=lambda c: c.rank)
    ]
    payload = {
        "source": {
            "title": source.title,
            "url": source.url,
            "platform": source.platform,
            "duration_s": source.duration_s,
        },
        "view": view,
        "profile": meta.get("profile"),
        "layout": (meta.get("layout") or {}).get("kind"),
        "chat_lag_s": meta.get("chat_lag_s"),
        "clips": rows,
    }
    (destination_dir / "clips.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [f"# Clips: {source.title}", ""]
    if payload["profile"]:
        lines += [
            f"Profile: {payload['profile']} | layout: {payload['layout']} | view: {view}"
            + (f" | chat lag: {payload['chat_lag_s']:.0f}s" if payload["chat_lag_s"] else ""),
            "",
        ]
    grouped = view != "overall"
    ordered = sorted(rows, key=lambda r: (r["group"], r["start_s"])) if grouped else rows
    current = None
    for row in ordered:
        if grouped and row["group"] != current:
            current = row["group"]
            lines += ["", f"## {current}", ""]
        lines.append(
            f"- **#{row['rank']} {row['title']}** ({row['start']}-{row['end']}, "
            f"score {row['score']}): {row['reason']}"
        )
        if row["file"]:
            lines.append(f"  `{Path(row['file']).name}`")
    path = destination_dir / "clips.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

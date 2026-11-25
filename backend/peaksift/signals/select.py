"""Selection views — which candidates to show, and grouped how.

Processing is continuous; grouping is only a view over the ranked candidates:

``overall``      the best N across the whole source.
``per_hour``     the best K in each hour (or any bucket size), with a guaranteed
                 minimum per bucket so a quiet hour still shows its best moment —
                 but ranked globally, so a great hour isn't capped at a weak
                 hour's level.
``per_chapter``  the best K inside each natural section of the stream. Chapters
                 are found where the picture's colour signature shifts for good
                 (switching games, going from gameplay to "just chatting").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .candidates import Candidate
from .fusion import smooth
from .timeline import Timeline

MIN_CHAPTER_S = 600
CHAPTER_CONTEXT_S = 300


@dataclass
class Group:
    label: str
    start_s: float
    end_s: float
    candidates: list[Candidate] = field(default_factory=list)


def _by_raw(c: Candidate) -> float:
    return c.raw_score


def overall(
    candidates: list[Candidate], limit: int, *, key: Callable[[Candidate], float] = _by_raw
) -> list[Candidate]:
    return sorted(candidates, key=key, reverse=True)[:limit]


def per_bucket(
    candidates: list[Candidate],
    *,
    duration_s: float,
    bucket_s: float = 3600.0,
    per_bucket: int = 3,
    min_per_bucket: int = 1,
    key: Callable[[Candidate], float] = _by_raw,
) -> list[Group]:
    """Top candidates per fixed-length bucket (hours by default)."""
    groups: list[Group] = []
    n_buckets = max(1, int(np.ceil(duration_s / bucket_s)))
    for b in range(n_buckets):
        lo, hi = b * bucket_s, min(duration_s, (b + 1) * bucket_s)
        inside = sorted((c for c in candidates if lo <= c.peak_s < hi), key=key, reverse=True)
        groups.append(
            Group(label=f"{_hms(lo)}-{_hms(hi)}", start_s=lo, end_s=hi, candidates=inside)
        )

    # Global threshold: the score of the Nth best overall, where N = buckets*K.
    budget = n_buckets * per_bucket
    ranked = sorted(candidates, key=key, reverse=True)
    threshold = key(ranked[min(len(ranked), budget) - 1]) if ranked else 0.0
    for group in groups:
        good = [c for c in group.candidates if key(c) >= threshold]
        floor = group.candidates[:min_per_bucket]
        chosen = good[: max(per_bucket, len(floor))]
        for c in floor:
            if c not in chosen:
                chosen.append(c)
        group.candidates = sorted(chosen, key=key, reverse=True)[: max(per_bucket, min_per_bucket)]
    return groups


def find_chapters(
    timeline: Timeline, *, min_chapter_s: int = MIN_CHAPTER_S
) -> list[tuple[float, float]]:
    """Split the source where its colour signature changes and stays changed."""
    n = timeline.length
    names = [name for name in timeline.names() if name.startswith("vsig_")]
    if n < 2 * min_chapter_s or not names:
        return [(0.0, float(n))]
    signature = np.stack([smooth(timeline.get(name, n), 30) for name in names], axis=1)
    ctx = CHAPTER_CONTEXT_S
    distance = np.zeros(n, dtype=np.float32)
    for t in range(ctx, n - ctx, 5):
        left = signature[t - ctx : t].mean(axis=0)
        right = signature[t : t + ctx].mean(axis=0)
        distance[t] = float(np.abs(left - right).sum())
    distance = smooth(distance, 11)

    threshold = max(
        0.25, float(np.percentile(distance[distance > 0], 95)) if (distance > 0).any() else 1.0
    )
    cuts: list[int] = []
    for t in np.argsort(distance)[::-1]:
        if distance[t] < threshold:
            break
        if t < min_chapter_s or n - t < min_chapter_s:
            continue
        if all(abs(int(t) - c) >= min_chapter_s for c in cuts):
            cuts.append(int(t))
    edges = [0, *sorted(cuts), n]
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def per_chapter(
    candidates: list[Candidate],
    chapters: list[tuple[float, float]],
    *,
    per_chapter: int = 3,
    key: Callable[[Candidate], float] = _by_raw,
) -> list[Group]:
    groups: list[Group] = []
    for i, (lo, hi) in enumerate(chapters, start=1):
        inside = sorted((c for c in candidates if lo <= c.peak_s < hi), key=key, reverse=True)
        groups.append(
            Group(
                label=f"Chapter {i} ({_hms(lo)}-{_hms(hi)})",
                start_s=lo,
                end_s=hi,
                candidates=inside[:per_chapter],
            )
        )
    return groups


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"

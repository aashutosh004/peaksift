"""Candidate moments — peaks on the fused score turned into clip windows.

A highlight is not just its peak. A good clip opens on the build-up (the
setup, the approach, the question), hits the peak, and stays for the reaction.
So each peak becomes a window:

* **start**: walk back from the peak until the score has been low for a few
  seconds — the calm before — bounded by a maximum pre-roll;
* **end**: walk forward until the reaction has died down, plus a short tail;
* then stretch or trim to the configured duration range, and snap both edges
  to sentence boundaries and audio silences when there is speech.

Greedy peak picking with a minimum gap keeps one candidate per moment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pipeline import boundaries
from ..pipeline.prepare import Silence
from ..pipeline.transcript import Transcript
from .fusion import LABELS, FusionResult

MIN_GAP_S = 40
PRE_ROLL_MAX_S = 35
PRE_ROLL_MIN_S = 6
POST_ROLL_MAX_S = 30
POST_ROLL_MIN_S = 5
CALM_FRACTION = 0.35
CALM_RUN_S = 4
TAIL_S = 2.0
SNAP_RADIUS_S = 6.0


@dataclass
class Candidate:
    start_s: float
    end_s: float
    peak_s: float
    #: Fused score at the peak (unitless, comparable within one source).
    raw_score: float
    #: 0-100, relative to the strongest moment found.
    score: int = 0
    #: Top contributing signals as (name, share of the peak score).
    drivers: list[tuple[str, float]] = field(default_factory=list)
    start_word: int = 0
    end_word: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def reason(self) -> str:
        parts = [LABELS.get(name, name) for name, share in self.drivers if share >= 0.12]
        return ", ".join(parts[:4]) or "unusual activity"


def find_peaks(
    score: np.ndarray, *, max_peaks: int, min_gap_s: int = MIN_GAP_S, floor: float | None = None
) -> list[int]:
    """Greedy non-maximum suppression over the score curve."""
    n = len(score)
    if n == 0 or max_peaks <= 0:
        return []
    if floor is None:
        positive = score[score > 0]
        floor = float(np.percentile(positive, 80)) if positive.size else 0.0
        floor = max(floor, 1e-3)
    order = np.argsort(score)[::-1]
    taken = np.zeros(n, dtype=bool)
    peaks: list[int] = []
    for index in order:
        if score[index] < floor or len(peaks) >= max_peaks:
            break
        if taken[index]:
            continue
        peaks.append(int(index))
        taken[max(0, index - min_gap_s) : min(n, index + min_gap_s + 1)] = True
    return sorted(peaks)


def window_for_peak(
    score: np.ndarray,
    peak: int,
    *,
    min_duration_s: float,
    max_duration_s: float,
    limit_s: float | None = None,
) -> tuple[float, float]:
    """Pick the build-up start and reaction end around one peak."""
    n = len(score)
    limit = n if limit_s is None else min(n, int(limit_s))
    level = score[peak]
    calm = level * CALM_FRACTION

    start = max(0, peak - PRE_ROLL_MIN_S)
    run = 0
    for t in range(peak - 1, max(-1, peak - PRE_ROLL_MAX_S - 1), -1):
        if t < 0:
            break
        run = run + 1 if score[t] < calm else 0
        start = t
        if run >= CALM_RUN_S:
            start = t + run // 2
            break
    start = min(start, peak - PRE_ROLL_MIN_S) if peak >= PRE_ROLL_MIN_S else 0

    end = min(limit - 1, peak + POST_ROLL_MIN_S)
    run = 0
    for t in range(peak + 1, min(limit, peak + POST_ROLL_MAX_S + 1)):
        run = run + 1 if score[t] < level * 0.4 else 0
        end = t
        if run >= 3:
            end = t - 1
            break
    end_s = min(float(limit), end + TAIL_S)
    start_s = float(max(0, start))

    # Stretch to the minimum length, favouring context before the peak (2:1).
    deficit = min_duration_s - (end_s - start_s)
    if deficit > 0:
        before = min(start_s, deficit * 2 / 3)
        start_s -= before
        end_s = min(float(limit), end_s + (deficit - before))
        if end_s - start_s < min_duration_s:
            start_s = max(0.0, end_s - min_duration_s)
    # Trim to the maximum, keeping the peak and more of the reaction than setup.
    if end_s - start_s > max_duration_s:
        excess = end_s - start_s - max_duration_s
        cut_front = min(excess * 0.6, max(0.0, peak - start_s - PRE_ROLL_MIN_S))
        start_s += cut_front
        end_s -= excess - cut_front
    return start_s, end_s


def snap_to_speech(
    transcript: Transcript | None,
    start_s: float,
    end_s: float,
    silences: list[Silence],
    *,
    max_duration_s: float,
) -> tuple[float, float, int, int]:
    """Move edges to sentence starts/ends when speech is nearby, else keep them."""
    if transcript is None or not transcript.words:
        return start_s, end_s, 0, 0

    first, last = transcript.indices_in_range(start_s, end_s)
    if last <= first:
        return start_s, end_s, first, max(first, last - 1)
    last -= 1

    new_first = boundaries.snap_start_to_sentence(transcript, first)
    new_last = boundaries.snap_end_to_sentence(transcript, last)
    s_word = transcript.words[new_first].start
    e_word = transcript.words[new_last].end

    snapped_start = start_s
    if abs(s_word - start_s) <= SNAP_RADIUS_S:
        snapped_start = boundaries.align_start(s_word, silences)
    else:
        new_first = first
    snapped_end = end_s
    if abs(e_word - end_s) <= SNAP_RADIUS_S:
        snapped_end = boundaries.align_end(e_word, silences)
    else:
        new_last = last

    if snapped_end - snapped_start > max_duration_s:
        snapped_end = snapped_start + max_duration_s
    if snapped_end <= snapped_start:
        return start_s, end_s, first, last
    return max(0.0, snapped_start), snapped_end, new_first, new_last


def build_candidates(
    fusion: FusionResult,
    *,
    transcript: Transcript | None = None,
    silences: list[Silence] | None = None,
    max_candidates: int = 30,
    min_duration_s: float = 20.0,
    max_duration_s: float = 90.0,
    min_gap_s: int = MIN_GAP_S,
    limit_s: float | None = None,
) -> list[Candidate]:
    """Find peaks and turn each into a clip window. Sorted by score, best first."""
    score = fusion.score
    peaks = find_peaks(score, max_peaks=max_candidates, min_gap_s=min_gap_s)
    if limit_s is not None:
        peaks = [p for p in peaks if p < limit_s]
    if not peaks:
        return []

    top = max(float(score[p]) for p in peaks) or 1.0
    candidates: list[Candidate] = []
    for peak in peaks:
        start_s, end_s = window_for_peak(
            score,
            peak,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            limit_s=limit_s,
        )
        start_s, end_s, first_word, last_word = snap_to_speech(
            transcript, start_s, end_s, silences or [], max_duration_s=max_duration_s
        )
        if not start_s <= peak <= end_s:
            # Snapping must never push the moment itself out of the clip.
            start_s, end_s = min(start_s, peak - 2.0), max(end_s, peak + 2.0)
            start_s = max(0.0, start_s)

        # Which signals made this moment, measured around the peak.
        lo, hi = max(0, peak - 3), min(len(score), peak + 4)
        shares = {name: float(c[lo:hi].max()) for name, c in fusion.contributions.items()}
        total = sum(shares.values()) or 1.0
        drivers = sorted(
            ((name, value / total) for name, value in shares.items() if value > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        candidates.append(
            Candidate(
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                peak_s=float(peak),
                raw_score=float(score[peak]),
                score=int(round(40 + 60 * float(score[peak]) / top)),
                drivers=drivers,
                start_word=first_word,
                end_word=last_word,
            )
        )

    candidates = _drop_overlaps(candidates)
    candidates.sort(key=lambda c: c.raw_score, reverse=True)
    return candidates


def _drop_overlaps(candidates: list[Candidate], max_overlap: float = 0.5) -> list[Candidate]:
    """Windows can grow into each other after stretching; keep the stronger one."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.raw_score, reverse=True):
        clash = False
        for other in kept:
            inter = min(candidate.end_s, other.end_s) - max(candidate.start_s, other.start_s)
            if inter > 0 and inter / min(candidate.duration_s, other.duration_s) > max_overlap:
                clash = True
                break
        if not clash:
            kept.append(candidate)
    return kept

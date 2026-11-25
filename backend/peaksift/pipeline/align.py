"""Give an untimed transcript the word timings of a timed one.

Cloud models like Gemini write better Hinglish than Whisper but don't return
reliable per-word times, which word-by-word captions and clean cut points need.
Whisper runs anyway (locally) and its words are timed, so the two are aligned:

1. Both word lists are reduced to a *sound skeleton* — lowercase consonants
   with common Hinglish/English spelling variants folded together
   (ph/f, w/v, c/k, z/j, doubled letters) — so "phokas" lines up with "focus"
   and "karektar" with "character".
2. A Needleman-Wunsch alignment matches the two sequences, scoring pairs by
   skeleton similarity.
3. Matched words take the Whisper word's start/end. Words between two matches
   are spread evenly over the time between them, so every word gets a time and
   times never go backwards.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .transcript import Word

_FOLDS = (
    ("ph", "f"),
    ("ck", "k"),
    ("qu", "kv"),
    ("w", "v"),
    ("c", "k"),
    ("q", "k"),
    ("z", "j"),
    ("x", "ks"),
    ("th", "t"),
    ("dh", "d"),
    ("bh", "b"),
    ("kh", "k"),
    ("gh", "g"),
    ("sh", "s"),
    ("chh", "ch"),
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REPEAT = re.compile(r"(.)\1+")

MATCH_THRESHOLD = 0.5
GAP = -0.45


def skeleton(word: str) -> str:
    """Consonant skeleton of a Latin-script word, with spelling variants folded."""
    text = _NON_ALNUM.sub("", word.lower())
    if not text:
        return ""
    for a, b in _FOLDS:
        text = text.replace(a, b)
    text = _REPEAT.sub(r"\1", text)
    first, rest = text[0], text[1:]
    consonants = re.sub(r"[aeiouy]", "", rest)
    # Keep a leading vowel: "aaj" vs "j" would otherwise collide with "ja".
    return first + consonants


def similarity(a: str, b: str) -> float:
    sa, sb = skeleton(a), skeleton(b)
    if not sa or not sb:
        return 1.0 if _NON_ALNUM.sub("", a.lower()) == _NON_ALNUM.sub("", b.lower()) else 0.0
    if sa == sb:
        return 1.0
    return SequenceMatcher(None, sa, sb).ratio()


def align(text_words: list[str], timed: list[Word]) -> tuple[list[Word], float]:
    """Timed :class:`Word` objects for ``text_words``, and the share matched.

    Preconditions:
        ``timed`` is in time order; both lists may be empty.
    """
    n, m = len(text_words), len(timed)
    if n == 0:
        return [], 1.0
    if m == 0:
        return [Word(w, 0.0, 0.0) for w in text_words], 0.0

    # Banded DP would be faster; clips are short enough for the full table.
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]  # 1 diag, 2 up (skip text), 3 left (skip timed)
    for i in range(1, n + 1):
        score[i][0], move[i][0] = i * GAP, 2
    for j in range(1, m + 1):
        score[0][j], move[0][j] = j * GAP, 3
    for i in range(1, n + 1):
        wi = text_words[i - 1]
        for j in range(1, m + 1):
            sim = similarity(wi, timed[j - 1].text)
            diag = score[i - 1][j - 1] + (sim - MATCH_THRESHOLD) * 2
            up = score[i - 1][j] + GAP
            left = score[i][j - 1] + GAP
            best = max(diag, up, left)
            score[i][j] = best
            move[i][j] = 1 if best == diag else (2 if best == up else 3)

    pairs: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        step = move[i][j]
        if step == 1:
            if similarity(text_words[i - 1], timed[j - 1].text) >= MATCH_THRESHOLD:
                pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif step == 2:
            i -= 1
        else:
            j -= 1

    out: list[Word] = [Word(w, 0.0, 0.0) for w in text_words]
    anchors = sorted(pairs.items())
    for ti, wj in anchors:
        out[ti].start, out[ti].end = timed[wj].start, timed[wj].end

    # Fill the gaps between anchors (and before the first / after the last).
    bounds = (
        [(-1, timed[0].start, timed[0].start)]
        + [(ti, timed[wj].start, timed[wj].end) for ti, wj in anchors]
        + [(n, timed[-1].end, timed[-1].end)]
    )
    for (left_i, _ls, left_end), (right_i, right_start, _re) in zip(
        bounds, bounds[1:], strict=False
    ):
        gap = list(range(left_i + 1, right_i))
        if not gap:
            continue
        span_start = left_end
        span_end = max(right_start, span_start)
        step = (span_end - span_start) / len(gap) if span_end > span_start else 0.0
        for k, index in enumerate(gap):
            out[index].start = span_start + k * step
            out[index].end = span_start + (k + 1) * step if step else span_start
    return out, len(anchors) / n

"""The signal timeline — one value per second per signal, for the whole source.

Everything the highlight engine knows about a video or stream lives here:
audio loudness, motion, chat rate, facial reactions, and so on, each as a float
array indexed by whole seconds from the start of the source.

One second is the right grain. It is fine enough to place a peak within a
sentence, coarse enough that a five-hour stream is 18,000 samples per signal —
small enough to rescore from scratch every time a live chunk lands, which keeps
the live and the after-the-fact paths identical.

Chunked processing writes into the timeline at an offset, so a live stream
grows the arrays as segments arrive, and a backfill worker can fill an earlier
range later without disturbing what is already there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Samples per second. Fixed: every signal module and the fusion code assume it.
RATE_HZ = 1


@dataclass
class Timeline:
    """Per-second signal arrays plus free-form metadata."""

    signals: dict[str, np.ndarray] = field(default_factory=dict)
    #: Which seconds have actually been measured, per signal. A live stream's
    #: backfill range is unmeasured (not zero) until the backfill worker runs.
    coverage: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- shape -------------------------------------------------------------

    @property
    def length(self) -> int:
        return max((len(a) for a in self.signals.values()), default=0)

    @property
    def duration_s(self) -> float:
        return float(self.length) / RATE_HZ

    def names(self) -> list[str]:
        return sorted(self.signals)

    def _grow(self, name: str, length: int) -> None:
        current = self.signals.get(name)
        if current is None:
            self.signals[name] = np.zeros(length, dtype=np.float32)
            self.coverage[name] = np.zeros(length, dtype=bool)
            return
        if len(current) < length:
            pad = length - len(current)
            self.signals[name] = np.concatenate([current, np.zeros(pad, dtype=np.float32)])
            self.coverage[name] = np.concatenate([self.coverage[name], np.zeros(pad, dtype=bool)])

    def extend_to(self, length: int) -> None:
        """Grow every signal to at least ``length`` samples."""
        for name in list(self.signals):
            self._grow(name, length)

    # -- access ------------------------------------------------------------

    def put(self, name: str, start_index: int, values: np.ndarray | list[float]) -> None:
        """Write ``values`` into signal ``name`` starting at second ``start_index``."""
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            return
        start_index = max(0, int(start_index))
        end = start_index + len(array)
        self._grow(name, max(end, self.length))
        self.signals[name][start_index:end] = array
        self.coverage[name][start_index:end] = True

    def put_all(self, start_index: int, values: dict[str, np.ndarray]) -> None:
        for name, array in values.items():
            self.put(name, start_index, array)

    def get(self, name: str, length: int | None = None) -> np.ndarray:
        """Return signal ``name`` padded/truncated to ``length`` (zeros if absent)."""
        length = self.length if length is None else length
        array = self.signals.get(name)
        if array is None:
            return np.zeros(length, dtype=np.float32)
        if len(array) >= length:
            return array[:length]
        return np.concatenate([array, np.zeros(length - len(array), dtype=np.float32)])

    def covered(self, name: str, length: int | None = None) -> np.ndarray:
        length = self.length if length is None else length
        mask = self.coverage.get(name)
        if mask is None:
            return np.zeros(length, dtype=bool)
        if len(mask) >= length:
            return mask[:length]
        return np.concatenate([mask, np.zeros(length - len(mask), dtype=bool)])

    def has(self, name: str) -> bool:
        """True when the signal exists and was measured somewhere."""
        mask = self.coverage.get(name)
        return mask is not None and bool(mask.any())

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Write ``<path>`` (npz arrays) and ``<path>.json`` (metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for name, array in self.signals.items():
            arrays[f"s__{name}"] = array
            arrays[f"c__{name}"] = self.coverage.get(name, np.ones(len(array), dtype=bool))
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(self.meta, default=_json_default), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Timeline:
        path = Path(path)
        timeline = cls()
        with np.load(path) as data:
            for key in data.files:
                kind, _, name = key.partition("__")
                if kind == "s":
                    timeline.signals[name] = data[key].astype(np.float32)
                elif kind == "c":
                    timeline.coverage[name] = data[key].astype(bool)
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            timeline.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return timeline


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def per_second(
    times: np.ndarray, values: np.ndarray, length: int, *, reduce: str = "max"
) -> np.ndarray:
    """Bucket irregular ``(time, value)`` samples into whole seconds.

    ``reduce`` is ``max``, ``mean`` or ``sum``. Empty seconds are 0, except for
    ``mean`` where an empty second carries the previous value forward — a
    sampled signal (faces at 1 fps, say) should not read as a dip between
    samples.
    """
    out = np.zeros(length, dtype=np.float32)
    if length <= 0 or len(times) == 0:
        return out
    idx = np.clip(np.floor(np.asarray(times)).astype(int), 0, length - 1)
    vals = np.asarray(values, dtype=np.float32)
    if reduce == "max":
        np.maximum.at(out, idx, vals)
        return out
    if reduce == "sum":
        np.add.at(out, idx, vals)
        return out
    counts = np.zeros(length, dtype=np.float32)
    np.add.at(out, idx, vals)
    np.add.at(counts, idx, 1.0)
    filled = counts > 0
    out[filled] /= counts[filled]
    last = 0.0
    seen = False
    for i in range(length):
        if filled[i]:
            last = float(out[i])
            seen = True
        elif seen:
            out[i] = last
    return out

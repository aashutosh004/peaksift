"""Speech models offered for Hindi / Hinglish audio, and where to fetch them.

Stock Whisper writes Hindi in Devanagari (or translates Hinglish to English),
so Hinglish captions from it go through a rule-based converter, which can't
recover English words Whisper heard as Hindi ("phokas" for "focus"). Models
fine-tuned on Hinglish write Roman script directly, English words in English.

Every entry here is free, runs locally through faster-whisper (CTranslate2),
and downloads once into the Hugging Face cache on first use. Any other
CTranslate2 Whisper works too: set ``whisper.hindi_model`` to its Hugging Face
repo ID or a local folder.

Scores come from ``scripts/caption_benchmark.py`` on Hindi gaming commentary;
re-run it on your own clips before trusting them for a different kind of
speech.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechModel:
    key: str
    label: str
    #: Hugging Face repo ID, or a stock faster-whisper size name.
    repo: str
    #: What the model writes: "roman" (Hinglish) or "devanagari" (converted).
    script: Literal["roman", "devanagari"]
    download_gb: float
    based_on: str
    license: str
    note: str
    #: Folder inside the repo holding the model, when it isn't the root.
    subfolder: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


HINDI_MODELS: dict[str, SpeechModel] = {
    m.key: m
    for m in (
        SpeechModel(
            key="hinglish-prime",
            label="Hinglish Prime (recommended)",
            repo="Hub84/faster-whisper-hinglish-prime",
            script="roman",
            download_gb=1.6,
            based_on="Oriserve Whisper-Hindi2Hinglish-Prime (large-v3)",
            license="Apache-2.0",
            note=(
                "Best tested: most complete and best-spelled Hinglish "
                "(78% readable words vs 46% for stock Whisper)."
            ),
        ),
        SpeechModel(
            key="hinglish-apex",
            label="Hinglish Apex (lighter)",
            repo="abhayhadez/anchor-hinglish-apex-ct2",
            script="roman",
            download_gb=0.8,
            based_on="Oriserve Whisper-Hindi2Hinglish-Apex (large-v3-turbo)",
            license="Apache-2.0",
            note=(
                "Twice as fast and half the size (~1 GB of GPU memory); "
                "slightly less complete than Prime."
            ),
        ),
        SpeechModel(
            key="hinglish-trelis",
            label="Hinglish (Trelis)",
            repo="nikh360pro/capsbro-hinglish-ct2",
            subfolder="int8",
            script="roman",
            download_gb=1.6,
            based_on="Trelis whisper-hinglish-preview (large-v3, Vaani Hindi)",
            license="Apache-2.0",
            note="Independent fine-tune, no punctuation. A second opinion when the others mishear.",
        ),
        SpeechModel(
            key="large-v3-turbo",
            label="Stock Whisper turbo + converter",
            repo="large-v3-turbo",
            script="devanagari",
            download_gb=1.6,
            based_on="OpenAI Whisper large-v3-turbo",
            license="MIT",
            note="Writes Devanagari, converted to Hinglish by rules. Use for Devanagari captions.",
        ),
        SpeechModel(
            key="large-v3",
            label="Stock Whisper large-v3 + converter",
            repo="large-v3",
            script="devanagari",
            download_gb=3.1,
            based_on="OpenAI Whisper large-v3",
            license="MIT",
            note="Slowest; slightly more accurate Devanagari than turbo.",
        ),
    )
}

DEFAULT_HINDI_MODEL = "hinglish-prime"
#: Used when Devanagari captions are wanted but the Hindi model writes Roman.
DEVANAGARI_FALLBACK = "large-v3-turbo"


def lookup(name: str) -> SpeechModel | None:
    """The catalogue entry for a key or repo ID, if it is one of ours."""
    if name in HINDI_MODELS:
        return HINDI_MODELS[name]
    return next((m for m in HINDI_MODELS.values() if m.repo == name), None)


def writes_roman(name: str) -> bool:
    """Whether a model writes Hinglish itself. Unknown models are assumed not to."""
    model = lookup(name)
    return model is not None and model.script == "roman"


@lru_cache(maxsize=8)
def model_source(name: str) -> str:
    """What to hand ``faster_whisper.WhisperModel`` for a key, repo ID or path.

    faster-whisper downloads plain repo IDs itself. A model kept in a subfolder
    of its repo is fetched here (only that folder) and passed as a local path.
    """
    model = lookup(name)
    if model is None:
        return name
    if not model.subfolder:
        return model.repo
    from huggingface_hub import snapshot_download

    root = snapshot_download(model.repo, allow_patterns=[f"{model.subfolder}/*"])
    return str(Path(root) / model.subfolder)


def catalogue() -> list[dict]:
    return [m.to_dict() for m in HINDI_MODELS.values()]

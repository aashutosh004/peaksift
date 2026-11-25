"""Transcribe stage — audio to word-level timestamps.

faster-whisper does the transcription; WhisperX (optional) adds speaker labels.
Word-level timing is non-negotiable: clip boundaries, caption animation, and
trim-handle snapping all derive from it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..config import WhisperSettings
from ..speech_models import DEVANAGARI_FALLBACK, model_source, writes_roman
from ..system import GPUInfo, report
from .transcript import Segment, Transcript, Word

log = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Transcription could not be completed."""


class NoSpeechError(TranscriptionError):
    """The audio was readable but contained no speech.

    Not fatal for multimodal detection: gameplay with no commentary still has
    audio, motion and chat to clip from.
    """


def resolve_compute(settings: WhisperSettings, gpu: GPUInfo | None = None) -> tuple[str, str]:
    """Return the ``(device, compute_type)`` to run Whisper with.

    An explicit ``settings.compute_type`` wins, but only if the device actually
    supports it. An override the backend will reject fails at model load with a
    message that points at cuDNN rather than at the setting, so an unsupported
    override is warned about and ignored rather than honoured into a crash.
    """
    gpu = gpu if gpu is not None else report().gpu
    automatic = gpu.compute_type

    override = settings.compute_type
    if not override:
        return gpu.device, automatic

    supported = gpu.supported_compute_types
    if supported and override not in supported:
        log.warning(
            "whisper.compute_type is set to %r, which %s does not support "
            "(supported: %s). Using %r instead.",
            override,
            gpu.device,
            ", ".join(sorted(supported)),
            automatic,
        )
        return gpu.device, automatic

    return gpu.device, override


def _model_load_error(exc: Exception, device: str, compute_type: str) -> TranscriptionError:
    """Turn a Whisper model-load failure into something actionable.

    The two causes look identical from the outside and need opposite fixes, so
    they're distinguished rather than lumped under one guess.
    """
    message = str(exc)
    lowered = message.lower()

    if "compute type" in lowered or "not support" in lowered:
        supported = report().gpu.supported_compute_types
        options = ", ".join(sorted(supported)) if supported else "unknown"
        return TranscriptionError(
            f"This device does not support the '{compute_type}' compute type.\n"
            f"Supported on {device}: {options}.\n\n"
            "Clear whisper.compute_type in config.json to let PeakSift choose, "
            "or set it to one of the supported values."
        )

    if device == "cuda" and ("cudnn" in lowered or "library" in lowered or "dll" in lowered):
        return TranscriptionError(
            f"Could not load Whisper on the GPU: {message}\n\n"
            "This is a missing CUDA runtime library. Run the setup again "
            "(Setup-PeakSift.cmd or ./setup.sh), which installs it when it finds "
            "an NVIDIA card, then retry — the job resumes from this stage."
        )

    return TranscriptionError(f"Could not load the Whisper model: {message}")


def _load_audio(audio: Path):
    """Hand Whisper decoded samples when the input is our own 16 kHz WAV.

    faster-whisper otherwise decodes through PyAV, and newer PyAV releases
    dropped an argument faster-whisper still passes (``metadata_errors``), which
    crashes every transcription. The prepare stage already produced 16 kHz mono
    PCM, so reading it directly is both safer and skips a redundant decode.
    Anything else falls back to the path, i.e. faster-whisper's own decoder.
    """
    import wave

    import numpy as np

    try:
        with wave.open(str(audio), "rb") as handle:
            if (
                handle.getframerate() == 16_000
                and handle.getnchannels() == 1
                and handle.getsampwidth() == 2
            ):
                raw = handle.readframes(handle.getnframes())
                return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except (wave.Error, EOFError, OSError):
        pass
    return str(audio)


_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def _load_model(model_cls, name: str, device: str, compute_type: str):
    """Load a Whisper model once per process.

    Live clipping transcribes a new chunk every minute; reloading a multi-GB
    model each time would cost more than the transcription itself.
    """
    key = (name, device, compute_type)
    if key not in _MODEL_CACHE:
        # Keep at most two models resident: the configured one plus the Hindi
        # model, so a Hindi live stream doesn't reload weights every chunk.
        while len(_MODEL_CACHE) >= 2:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = model_cls(name, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def release_models() -> None:
    """Free the cached Whisper models' GPU memory.

    Called after a video's transcription: a large model holds ~3 GB of VRAM, and
    an idle copy left in the web-app process pushes the next job (or the next
    process) past an 8 GB card's memory, where Windows pages to system RAM and
    transcription slows down tenfold. Live sessions keep theirs loaded because
    they transcribe a new chunk every minute.
    """
    if not _MODEL_CACHE:
        return
    import gc

    _MODEL_CACHE.clear()
    gc.collect()
    log.info("Released Whisper model memory.")


#: Share of Hindi+Urdu in Whisper's language guess above which audio is treated
#: as Hindi. Hinglish (Hindi with lots of English) is usually labelled English
#: with only ~5-10% Hindi; plain English speech scores well under 1%.
HINDI_SHARE = 0.05
#: Models too small to transcribe Hindi usefully (they produce garbage, slowly).
SMALL_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en"}


#: Seconds of speech per chunk for Hinglish models (see :func:`decode`).
HINGLISH_CHUNK_S = 10
#: Outputs some fine-tuned models learned from empty training transcripts
#: (stored as pandas NaN). Forbidden while decoding; dropped if seen anyway.
JUNK_SEGMENTS = {"nan"}


def _junk_tokens(model) -> list[int]:
    """Token IDs of the junk words, so the decoder can't produce them.

    Hindi2Hinglish Prime answers about 1 in 10 chunks with "nan" instead of the
    speech in it (at every precision, so it's learned, not a maths error).
    With the token forbidden it writes the actual words.
    """
    ids: list[int] = []
    tokenizer = getattr(model, "hf_tokenizer", None)
    if tokenizer is None:
        return ids
    for junk in JUNK_SEGMENTS:
        for text in (junk, " " + junk):
            encoded = tokenizer.encode(text, add_special_tokens=False).ids
            if len(encoded) == 1:
                ids.append(encoded[0])
    return ids


def decode(model, samples, language: str | None, *, chunked: bool):
    """Run Whisper over the audio; returns ``(segments, info)``.

    Stock Whisper decodes the audio window by window, each window seeing the
    previous one's text, and moves on from where its last word ended.

    Models fine-tuned on Hinglish were trained on short clips and don't keep
    Whisper's long-audio behaviour: they end a 30-second window early, so the
    next window starts back inside speech already written and the same lines
    come out two or three times, or they loop on a phrase. Measured on an
    8-minute Hindi stream: a fifth of all 4-word phrases repeated. Cutting the
    speech into short chunks (at pauses, via the VAD) and decoding each chunk
    exactly once, independently, removes that (1% repeats) and is ~8x faster,
    since chunks are decoded in batches. Stock Whisper does the opposite —
    it loops when decoded this way — so it keeps the sequential path.
    """
    if chunked:
        from faster_whisper import BatchedInferencePipeline

        return BatchedInferencePipeline(model).transcribe(
            samples,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            chunk_length=HINGLISH_CHUNK_S,
            batch_size=8,
            # -1 keeps Whisper's default suppressed tokens.
            suppress_tokens=[-1, *_junk_tokens(model)],
        )
    return model.transcribe(
        samples,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        # Trims long silences before decoding, which both speeds things up
        # and stops Whisper hallucinating text into empty audio.
        vad_parameters={"min_silence_duration_ms": 500},
    )


def hindi_model_for(settings: WhisperSettings) -> str:
    """The model to transcribe Hindi audio with, or "" to keep ``settings.model``.

    Hinglish models write Roman script, so when Devanagari captions are wanted
    they're swapped for stock Whisper. The configured model is kept when it is
    already able to do Hindi and no Hindi model is set.
    """
    choice = (settings.hindi_model or "").strip()
    if settings.hindi_script == "devanagari" and writes_roman(choice):
        choice = DEVANAGARI_FALLBACK if settings.model in SMALL_MODELS else ""
    return choice


def _is_hindi(model, samples) -> bool:
    try:
        language, _prob, probs = model.detect_language(
            samples, vad_filter=True, language_detection_segments=4
        )
    except Exception as exc:  # detection is a hint; never fail the transcription
        log.debug("Language detection failed: %s", exc)
        return False
    shares = dict(probs or [])
    hindi = shares.get("hi", 0.0) + shares.get("ur", 0.0)
    log.info("Language guess: %s; Hindi/Urdu share %.0f%%.", language, hindi * 100)
    return language in ("hi", "ur") or hindi >= HINDI_SHARE


def transcribe(
    audio: Path,
    settings: WhisperSettings | None = None,
    *,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Transcript:
    """Transcribe an audio file into a :class:`Transcript` with word timings.

    Preconditions:
        audio is a decodable audio file, ideally 16 kHz mono as produced by
        :func:`peaksift.pipeline.prepare.extract_audio`.
    """
    settings = settings or WhisperSettings()

    # Must happen before faster-whisper pulls in CTranslate2, which resolves its
    # CUDA dependencies at import.
    from ..cuda import ensure_cuda_libraries

    ensure_cuda_libraries()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - faster-whisper is a core dep
        raise TranscriptionError(
            "faster-whisper is not installed. Run `peaksift doctor` for details."
        ) from exc

    device, compute_type = resolve_compute(settings)
    log.info(
        "Transcribing with model=%s device=%s compute_type=%s",
        settings.model,
        device,
        compute_type,
    )

    try:
        model = _load_model(WhisperModel, settings.model, device, compute_type)
    except Exception as exc:
        raise _model_load_error(exc, device, compute_type) from exc

    samples = _load_audio(audio)
    hinglish = getattr(settings, "hindi_script", "hinglish") == "hinglish"

    language = settings.language or None
    if language is None and _is_hindi(model, samples):
        # Hinglish speech is often labelled English, and Whisper then
        # *translates* it; Hindi is often labelled Urdu (Arabic script).
        # Decoding as Hindi transcribes what was said, English words included.
        language = "hi"
    used = settings.model
    hindi_model = hindi_model_for(settings) if language == "hi" else ""
    if hindi_model and hindi_model != settings.model:
        log.info("Hindi audio: transcribing with %s.", hindi_model)
        try:
            model = _load_model(WhisperModel, model_source(hindi_model), device, compute_type)
            used = hindi_model
        except Exception as exc:
            log.warning("Could not load %s (%s); using %s.", hindi_model, exc, settings.model)

    segments_iter, info = decode(model, samples, language, chunked=writes_roman(used))

    total = duration_s or getattr(info, "duration", 0.0) or 0.0
    transcript = Transcript(
        language=getattr(info, "language", "") or settings.language,
        model=used,
        source="whisper",
    )

    for segment in segments_iter:
        if cancelled is not None and cancelled():
            raise TranscriptionError("Transcription cancelled.")
        if (segment.text or "").strip().lower() in JUNK_SEGMENTS:
            continue

        first_word = len(transcript.words)
        for word in getattr(segment, "words", None) or []:
            text = (word.word or "").strip()
            if not text:
                continue
            transcript.words.append(Word(text=text, start=float(word.start), end=float(word.end)))

        # A segment with no word timings still carries text worth keeping for
        # display, but it can't contribute to word-indexed clip boundaries.
        last_word = max(first_word, len(transcript.words) - 1)
        transcript.segments.append(
            Segment(
                text=(segment.text or "").strip(),
                start=float(segment.start),
                end=float(segment.end),
                first_word=first_word,
                last_word=last_word,
            )
        )

        if on_progress and total:
            on_progress(min(1.0, float(segment.end) / total))

    if not transcript.words:
        raise NoSpeechError(
            "No speech was detected in this audio. If the file really does contain "
            "speech, try a larger Whisper model or check that the correct audio track "
            "was extracted."
        )

    if on_progress:
        on_progress(1.0)

    if hinglish and transcript.language == "hi":
        to_hinglish(transcript)

    log.info(
        "Transcribed %d words in %d segments.", len(transcript.words), len(transcript.segments)
    )
    return transcript


def to_hinglish(transcript: Transcript) -> Transcript:
    """Rewrite Devanagari words and segments in Hinglish, keeping every timing."""
    from ..hinglish import transliterate_text, transliterate_word

    changed = 0
    for word in transcript.words:
        converted = transliterate_word(word.text)
        if converted != word.text:
            word.text = converted
            changed += 1
    for segment in transcript.segments:
        segment.text = transliterate_text(segment.text)
    if changed:
        log.info("Wrote %d Hindi words in Hinglish.", changed)
    return transcript


# --------------------------------------------------------------------------
# Diarization
# --------------------------------------------------------------------------


def diarize(
    audio: Path,
    transcript: Transcript,
    *,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Transcript:
    """Label each word with a speaker, in place.

    Uses pyannote via WhisperX. Returns the transcript unchanged (with a logged
    warning) if diarization isn't available — speaker labels improve reframing
    but are never required for the pipeline to complete.

    Preconditions:
        transcript already has word-level timings.
    """
    if not transcript.words:
        return transcript

    pipeline_cls = _load_diarization_pipeline()
    if pipeline_cls is None:
        log.warning(
            "WhisperX is not installed; skipping diarization. "
            "Install the diarization extra (see docs/LAYOUTS.md)."
        )
        return transcript

    if not hf_token:
        log.warning(
            "Diarization needs a HuggingFace token. Set one with "
            "`peaksift config set-secret huggingface_token`, and accept the pyannote "
            "model licences on huggingface.co. Continuing without speaker labels."
        )
        return transcript

    device = report().gpu.device
    try:
        pipeline = pipeline_cls(use_auth_token=hf_token, device=device)
        diarization = pipeline(str(audio), min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as exc:
        log.warning("Diarization failed (%s); continuing without speaker labels.", exc)
        return transcript

    turns = _diarization_turns(diarization)
    if not turns:
        return transcript

    _assign_speakers(transcript, turns)
    log.info("Diarization labelled %d speakers.", len(transcript.speakers))
    return transcript


def _load_diarization_pipeline():
    """Import WhisperX's diarization pipeline across its API reshuffles."""
    try:
        from whisperx.diarize import DiarizationPipeline

        return DiarizationPipeline
    except ImportError:
        pass
    try:
        from whisperx import DiarizationPipeline  # type: ignore[attr-defined]

        return DiarizationPipeline
    except (ImportError, AttributeError):
        return None


def _diarization_turns(diarization) -> list[tuple[str, float, float]]:
    """Normalise WhisperX output into ``(speaker, start, end)`` tuples.

    Recent versions return a pandas DataFrame; older ones return a pyannote
    ``Annotation``. Both are handled so the extra can be upgraded independently.
    """
    turns: list[tuple[str, float, float]] = []

    if hasattr(diarization, "itertuples"):  # DataFrame
        for row in diarization.itertuples():
            speaker = getattr(row, "speaker", None)
            start = getattr(row, "start", None)
            end = getattr(row, "end", None)
            if speaker is not None and start is not None and end is not None:
                turns.append((str(speaker), float(start), float(end)))
        return turns

    if hasattr(diarization, "itertracks"):  # pyannote Annotation
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((str(speaker), float(segment.start), float(segment.end)))

    return turns


def _assign_speakers(transcript: Transcript, turns: list[tuple[str, float, float]]) -> None:
    """Attach a speaker to every word by maximum temporal overlap.

    Overlap rather than midpoint containment: words straddling a turn boundary
    are common, and the speaker who covers more of the word is the better guess.
    """
    turns = sorted(turns, key=lambda t: t[1])

    for word in transcript.words:
        best_speaker: str | None = None
        best_overlap = 0.0
        for speaker, start, end in turns:
            if start > word.end:
                break
            overlap = min(word.end, end) - max(word.start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        word.speaker = best_speaker

    for segment in transcript.segments:
        segment.speaker = transcript.dominant_speaker(segment.first_word, segment.last_word)

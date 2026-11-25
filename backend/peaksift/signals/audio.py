"""Audio signals — loudness, spectral flux, and shout energy, per second.

All three are cheap, model-free measurements that read a 16 kHz mono WAV in
one-minute blocks, so a five-hour stream never needs more than a few megabytes
of audio in memory:

``loudness``   RMS level in dBFS. Raw level matters less than *change* from the
               stream's own baseline; fusion handles the baseline.
``flux``       Spectral flux — how fast the spectrum is changing. Spikes on
               sudden events: a scream, an explosion, a laugh, a crowd reacting.
``shout``      Share of energy in the 2-5 kHz band where raised voices, screams
               and excited yelling concentrate, relative to the speech band.

Speech-derived signals (speaking rate, exclamations) come from the transcript in
:func:`speech_features`, since the words are already measured there.
"""

from __future__ import annotations

import logging
import re
import wave
from pathlib import Path

import numpy as np

from ..pipeline.transcript import Transcript

log = logging.getLogger(__name__)

FRAME = 1024
HOP = 512
BLOCK_S = 60

#: Silence floor for the loudness signal, so digital silence doesn't produce
#: -inf and dominate the statistics.
FLOOR_DB = -80.0


def audio_features(wav_path: Path, *, offset_s: float = 0.0) -> dict[str, np.ndarray]:
    """Per-second audio signals for a 16-bit mono WAV.

    ``offset_s`` is accepted for symmetry with the other extractors; the
    returned arrays always start at the file's own second zero and the caller
    places them on the timeline.
    """
    del offset_s
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(f"{wav_path.name} must be 16-bit mono PCM.")
        rate = handle.getframerate()
        total_frames = handle.getnframes()
        seconds = int(np.ceil(total_frames / rate))

        loudness = np.full(seconds, FLOOR_DB, dtype=np.float32)
        flux = np.zeros(seconds, dtype=np.float32)
        shout = np.zeros(seconds, dtype=np.float32)

        freqs = np.fft.rfftfreq(FRAME, d=1.0 / rate)
        speech_band = (freqs >= 100) & (freqs <= 8000)
        shout_band = (freqs >= 2000) & (freqs <= 5000)
        window = np.hanning(FRAME).astype(np.float32)

        previous_spectrum: np.ndarray | None = None
        carry = np.zeros(0, dtype=np.float32)
        second_cursor = 0

        while True:
            raw = handle.readframes(rate * BLOCK_S)
            if not raw:
                break
            block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            block_seconds = int(np.ceil(len(block) / rate))

            # Loudness per whole second of this block.
            for i in range(block_seconds):
                chunk = block[i * rate : (i + 1) * rate]
                if chunk.size:
                    rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-12)
                    index = second_cursor + i
                    if index < seconds:
                        loudness[index] = max(FLOOR_DB, 20.0 * np.log10(rms))

            # Spectral features on overlapping frames, carrying the tail of the
            # previous block so frames straddling the boundary aren't lost.
            samples = np.concatenate([carry, block])
            base_time = second_cursor - len(carry) / rate
            n_frames = max(0, 1 + (len(samples) - FRAME) // HOP)
            if n_frames:
                idx = np.arange(FRAME)[None, :] + HOP * np.arange(n_frames)[:, None]
                frames = samples[idx] * window
                spectrum = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
                log_spec = np.log1p(spectrum * 10.0)

                if previous_spectrum is not None:
                    stacked = np.vstack([previous_spectrum[None, :], log_spec])
                else:
                    stacked = np.vstack([log_spec[:1], log_spec])
                frame_flux = np.maximum(0.0, np.diff(stacked, axis=0)).sum(axis=1)
                previous_spectrum = log_spec[-1]

                power = spectrum * spectrum
                speech_energy = power[:, speech_band].sum(axis=1) + 1e-9
                shout_energy = power[:, shout_band].sum(axis=1)
                frame_shout = shout_energy / speech_energy

                centres = base_time + (HOP * np.arange(n_frames) + FRAME / 2) / rate
                sec_idx = np.clip(centres.astype(int), 0, seconds - 1)
                counts = np.bincount(sec_idx, minlength=seconds).astype(np.float32)
                flux_sum = np.bincount(sec_idx, weights=frame_flux, minlength=seconds)
                shout_sum = np.bincount(sec_idx, weights=frame_shout, minlength=seconds)
                touched = counts > 0
                flux[touched] = (flux_sum[touched] / counts[touched]).astype(np.float32)
                shout[touched] = (shout_sum[touched] / counts[touched]).astype(np.float32)

                consumed = n_frames * HOP
                carry = samples[consumed:]
            else:
                carry = samples

            second_cursor += block_seconds

    # Spectral shout ratio is meaningless in silence; zero it there so quiet
    # hiss doesn't read as shouting.
    shout[loudness < -55.0] = 0.0
    return {"loudness": loudness, "flux": flux, "shout": shout}


# --------------------------------------------------------------------------
# Speech-derived signals
# --------------------------------------------------------------------------

#: Phrases streamers say when something just happened. Matched on lowercased,
#: punctuation-stripped text. Deliberately includes common Hinglish reactions.
EXCLAMATIONS = (
    "oh my god",
    "omg",
    "no way",
    "lets go",
    "let's go",
    "what the",
    "wtf",
    "holy",
    "oh no",
    "no no no",
    "yes yes",
    "insane",
    "clip that",
    "clip it",
    "are you kidding",
    "are you serious",
    "bro what",
    "what was that",
    "did you see",
    "oh shit",
    "dude",
    "unbelievable",
    "crazy",
    "bhai",
    "arre",
    "kya baat",
    "abe",
    "oh hell",
)
_NORMALISE = re.compile(r"[^\w\s']+")


def speech_features(transcript: Transcript, length: int) -> dict[str, np.ndarray]:
    """Speaking-rate and exclamation signals from word timings.

    ``speech_rate``  words starting in each second.
    ``exclaim``      exclamation phrases or "!" sentence ends, per second.
    """
    rate = np.zeros(length, dtype=np.float32)
    exclaim = np.zeros(length, dtype=np.float32)
    if not transcript.words or length <= 0:
        return {"speech_rate": rate, "exclaim": exclaim}

    for word in transcript.words:
        index = int(word.start)
        if 0 <= index < length:
            rate[index] += 1.0
            if word.text.strip().endswith("!"):
                exclaim[index] += 1.0

    for segment in transcript.segments:
        text = " " + _NORMALISE.sub(" ", segment.text.lower()) + " "
        hits = sum(text.count(f" {phrase} ") for phrase in EXCLAMATIONS)
        if hits:
            index = int(segment.start)
            span = max(1, int(segment.end) - index)
            for offset in range(span):
                if 0 <= index + offset < length:
                    exclaim[index + offset] += hits / span

    return {"speech_rate": rate, "exclaim": exclaim}

"""Caption benchmark: compare transcription options on the same short clips.

    .venv\\Scripts\\python.exe scripts\\caption_benchmark.py .local\\bench\\clips

Each ``clipN.wav`` (16 kHz mono, ~30 s) in the folder is transcribed by every
system below, and an HTML report with the audio and each system's text side by
side is written next to the clips.

Systems:
  baseline  local Whisper (large-v3-turbo, decoded as Hindi) + Devanagari→Hinglish
  <key>     any Hindi model from peaksift.speech_models (hinglish-prime,
            hinglish-apex, hinglish-trelis, ...), decoded as Hindi
  polish    baseline text cleaned up by a local LLM via Ollama (spelling only)
  gemini    Gemini transcribes; words are timed by aligning to the baseline

Scores:
  * If ``clipN.ref.txt`` exists (a correct hand-written transcript), word error
    rate (WER) against it — the real measure; lower is better.
  * Always: "known words" — the share of words found in an English word list or
    a list of common Hinglish words. A rough readability signal: misspelled
    English ("phokas") and garbled words count against it; it can't tell a
    correct word from a wrong-but-real one.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import re
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from peaksift import config  # noqa: E402
from peaksift.cuda import ensure_cuda_libraries  # noqa: E402
from peaksift.pipeline.align import align  # noqa: E402
from peaksift.pipeline.transcribe import JUNK_SEGMENTS, decode  # noqa: E402
from peaksift.pipeline.transcript import Word  # noqa: E402
from peaksift.speech_models import HINDI_MODELS, model_source, writes_roman  # noqa: E402

BASELINE_MODEL = "large-v3-turbo"
FALLBACK_GEMINI = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
ENGLISH_WORDS_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/master/"
    "google-10000-english-no-swears.txt"
)

#: Common Hinglish words in their usual Roman spellings (plus close variants).
HINGLISH = set(
    """
    hai hain ho hoon hun tha thi the tu tum aap main mai mujhe mujhko mera meri mere tera teri tere
    uska uski uske iska iski iske kya kyu kyun kyon kaise kaisa kaisi kab kahan kidhar idhar udhar
    yahan wahan ye yeh wo woh vo voh ek do teen char paanch nahi nahin na mat haan han ji bhai bro
    yaar yar arre are abe oye accha achha acha theek thik sahi galat bahut bohot bht zyada thoda
    kuch kuchh sab sabhi koi kaun kisne kisko kisi kar karo karna karta karti karte kiya kiye ki ka
    ke ko se me mein par pe tak liye lie wala wali wale aaj kal abhi phir fir bas chal chalo chalte
    dekh dekho dekha dekhna dekhte gaya gayi gaye jaa ja jao jana jaata jaate aa aao aaya aayi aaye
    raha rahi rahe rha rhi rhe hota hoti hote hua hui hue hoga hogi honge le lo lena liya lete de do
    dena diya dete bol bolo bola bolta bolte samajh samjha pata pta matlab bilkul sach kasam lag laga
    lagta lagti mar maar maro maara mara gaadi gadi paisa paise log logon unka unki unke unko inka
    inki inke apna apni apne hum hame humko humne usne maine tune isne kyunki lekin magar aur ya toh
    to bhi hi wahi yahi aise waise jaise itna utna kitna kitne kitni jab tab ab sabse pehle baad
    andar bahar upar neeche niche saath sath ghar baat baatein kaam din raat log aadmi banda bande
    pagal mast badhiya bekaar faltu dimaag dil zindagi duniya khatam shuru ruk ruko chup suno sun
    """.split()  # noqa: SIM905 - a word list reads better as text
)

_TOKEN = re.compile(r"[a-z0-9']+")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_samples(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def known_rate(text: str, english: set[str]) -> float:
    words = tokens(text)
    if not words:
        return 0.0
    return sum(1 for w in words if w in english or w in HINGLISH or w.isdigit()) / len(words)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1] / len(ref)


def english_words(cache: Path) -> set[str]:
    if not cache.exists():
        urllib.request.urlretrieve(ENGLISH_WORDS_URL, cache)  # noqa: S310
    return {w.strip().lower() for w in cache.read_text(encoding="utf-8").split() if w.strip()}


# --------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------


def run_whisper(model_name: str, clips: list[Path], *, hinglish: bool) -> dict[str, dict]:
    from faster_whisper import WhisperModel
    from peaksift.hinglish import transliterate_word

    ensure_cuda_libraries()
    model = WhisperModel(model_source(model_name), device="cuda", compute_type="float16")
    results: dict[str, dict] = {}
    for clip in clips:
        started = time.time()
        # The pipeline's own decoding, so scores match what a real job gets.
        segments, _info = decode(model, load_samples(clip), "hi", chunked=writes_roman(model_name))
        words: list[Word] = []
        for segment in segments:
            if (segment.text or "").strip().lower() in JUNK_SEGMENTS:
                continue
            for w in segment.words or []:
                text = (w.word or "").strip()
                if text:
                    words.append(
                        Word(transliterate_word(text) if hinglish else text, w.start, w.end)
                    )
        results[clip.stem] = {
            "text": " ".join(w.text for w in words),
            "words": [[w.text, round(w.start, 2), round(w.end, 2)] for w in words],
            "seconds": round(time.time() - started, 1),
        }
        print(f"  {model_name} {clip.stem}: {results[clip.stem]['text'][:90]}")
    del model
    gc.collect()
    return results


def run_polish(baseline: dict[str, dict], model: str) -> dict[str, dict]:
    """Ask a local LLM to fix spelling only, one word in -> one word out."""
    results: dict[str, dict] = {}
    for name, base in baseline.items():
        source = base["text"].split()
        started = time.time()
        prompt = (
            "Below is an automatic transcript of a Hindi gaming stream, written in Hinglish "
            "(Hindi in Roman letters). Fix only the spelling: write Hindi words in common "
            "Hinglish spelling and English words in normal English spelling (for example "
            "'phokas' -> 'focus', 'karektar' -> 'character'). Do not translate, add, remove, "
            "merge or reorder words. Reply with exactly "
            f"{len(source)} space-separated words and nothing else.\n\n{base['text']}"
        )
        body = json.dumps(
            {
                "model": model,
                "stream": False,
                "think": False,
                "keep_alive": 0,
                "options": {"temperature": 0},
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                reply = json.loads(response.read())["message"]["content"].strip()
        except Exception as exc:
            reply = ""
            print(f"  polish {name}: failed ({exc})")
        fixed = reply.split()
        kept = len(fixed) == len(source)
        results[name] = {
            "text": " ".join(fixed) if kept else base["text"],
            "seconds": round(time.time() - started, 1),
            "note": ""
            if kept
            else f"LLM returned {len(fixed)} words for {len(source)}; kept original",
        }
        print(f"  polish {name}: {results[name]['text'][:90]}")
    return results


def run_gemini(clips: list[Path], baseline: dict[str, dict], model: str) -> dict[str, dict]:
    from google import genai
    from google.genai import types

    settings = config.load()
    client = genai.Client(api_key=config.get_secret("gemini", settings))
    prompt = (
        "Transcribe this audio verbatim. It is a Hindi-speaking gaming streamer who mixes in "
        "English. Write it in Hinglish: Hindi words in common Roman (Latin) spelling, English "
        "words in normal English spelling. Do not translate, summarise or add anything, do not "
        "label speakers, and leave out sound descriptions. Output only the transcript text."
    )
    results: dict[str, dict] = {}
    for clip in clips:
        started = time.time()
        text, used = "", ""
        # Busy models answer 503/429; retry with backoff, then fall back.
        for name in [model, *FALLBACK_GEMINI]:
            try:
                response = client.models.generate_content(
                    model=name,
                    contents=[
                        types.Part.from_bytes(data=clip.read_bytes(), mime_type="audio/wav"),
                        prompt,
                    ],
                )
                text, used = " ".join((response.text or "").split()), name
                break
            except Exception as exc:
                print(f"  gemini {clip.stem} ({name}): {str(exc)[:80]}")
                time.sleep(3)
        timed = [Word(t, s, e) for t, s, e in baseline[clip.stem]["words"]]
        aligned, matched = align(text.split(), timed)
        results[clip.stem] = {
            "text": text,
            "words": [[w.text, round(w.start, 2), round(w.end, 2)] for w in aligned],
            "seconds": round(time.time() - started, 1),
            "note": f"{used or 'no model answered'}; {matched:.0%} of words matched to Whisper timings",
        }
        print(f"  gemini {clip.stem}: {text[:90]}")
    return results


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

SYSTEM_LABELS = {
    "baseline": "Current: Whisper large-v3-turbo + converter",
    **{key: m.label for key, m in HINDI_MODELS.items() if key != "large-v3-turbo"},
    "polish": "Current + local AI spelling fix (Ollama)",
    "gemini": "Gemini (+ Whisper timings)",
}


def write_report(folder: Path, results: dict[str, dict[str, dict]], english: set[str]) -> Path:
    clips = sorted({c for system in results.values() for c in system})
    summary_rows = []
    for system, per_clip in results.items():
        known = [known_rate(v["text"], english) for v in per_clip.values()]
        secs = [v.get("seconds", 0) for v in per_clip.values()]
        refs = [
            wer((folder / f"{c}.ref.txt").read_text(encoding="utf-8"), per_clip[c]["text"])
            for c in per_clip
            if (folder / f"{c}.ref.txt").exists()
        ]
        summary_rows.append(
            f"<tr><td>{html.escape(SYSTEM_LABELS.get(system, system))}</td>"
            f"<td>{np.mean(known):.0%}</td><td>{np.mean(refs):.0%}</td>"
            if refs
            else f"<tr><td>{html.escape(SYSTEM_LABELS.get(system, system))}</td>"
            f"<td>{np.mean(known):.0%}</td><td>–</td>"
        )
        summary_rows[-1] += f"<td>{np.mean(secs):.1f}s</td></tr>"

    sections = []
    for clip in clips:
        rows = []
        for system, per_clip in results.items():
            item = per_clip.get(clip)
            if not item:
                continue
            note = (
                f"<div class='note'>{html.escape(item.get('note', ''))}</div>"
                if item.get("note")
                else ""
            )
            rows.append(
                f"<tr><th>{html.escape(SYSTEM_LABELS.get(system, system))}"
                f"<span>{known_rate(item['text'], english):.0%} known</span></th>"
                f"<td>{html.escape(item['text']) or '<i>(empty)</i>'}{note}</td></tr>"
            )
        sections.append(
            f"<section><h2>{clip}</h2><audio controls preload='none' src='{clip}.m4a'></audio>"
            f"<table>{''.join(rows)}</table></section>"
        )

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Caption benchmark</title>
<style>
body{{background:#0e1014;color:#e9ebf1;font:14px/1.55 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{font-size:20px}} h2{{font-size:15px;margin:28px 0 8px}} audio{{width:100%;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse}} th,td{{border-bottom:1px solid #2a2f3b;padding:10px 8px;text-align:left;vertical-align:top}}
th{{width:260px;color:#98a1b1;font-weight:500}} th span{{display:block;font-size:12px;color:#7c5cff}}
.note{{color:#98a1b1;font-size:12px;margin-top:4px}} .sum td,.sum th{{width:auto}}
</style></head><body>
<h1>Caption benchmark (Hindi / Hinglish)</h1>
<p>Listen to each clip and read the systems' captions. "Known" = share of words that are real English
or common Hinglish words (rough: misspellings like "phokas" count against it). Add <code>clipN.ref.txt</code>
with the correct words to get a real error rate (WER).</p>
<table class="sum"><tr><th>System</th><th>Known words</th><th>WER (if refs)</th><th>Time / 30 s clip</th></tr>
{"".join(summary_rows)}</table>
{"".join(sections)}
</body></html>"""
    out = folder / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--systems", default="baseline,hinglish-prime,hinglish-apex,hinglish-trelis,gemini"
    )
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--gemini-model", default="gemini-3.8-flash")
    args = parser.parse_args()

    folder: Path = args.folder
    clips = sorted(folder.glob("clip*.wav"))
    systems = args.systems.split(",")
    english = english_words(folder / "english-10k.txt")
    results_path = folder / "results.json"
    results: dict[str, dict] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    )

    if "baseline" in systems:
        print("baseline…")
        results["baseline"] = run_whisper(BASELINE_MODEL, clips, hinglish=True)
    for key in systems:
        if key in HINDI_MODELS and key != BASELINE_MODEL:
            print(f"{key}…")
            results[key] = run_whisper(key, clips, hinglish=True)
    if "polish" in systems and "baseline" in results:
        print("polish…")
        results["polish"] = run_polish(results["baseline"], args.ollama_model)
    if "gemini" in systems and "baseline" in results:
        print("gemini…")
        results["gemini"] = run_gemini(clips, results["baseline"], args.gemini_model)

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    report = write_report(folder, results, english)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

# Captions

Captions are burned into every clip, word by word, with the spoken word
highlighted. Word timings come from Whisper, so clip boundaries, caption
animation and cut points all line up.

## Styles

| Style | Look |
|---|---|
| `bold_pop` (default) | Chunky white, heavy outline; the spoken word grows and turns yellow |
| `karaoke_fill` | Words fill with colour as they're spoken |
| `clean_lower` | Minimal lower third, no animation |
| `boxed` | High-contrast text on a solid block |

Fonts (Anton, Inter) are bundled, so nothing is fetched at runtime.

## Speech models

Whisper runs locally. `small` is the default (fine on a CPU); with an NVIDIA GPU,
choose `large-v3-turbo` in **Settings → Transcription** for clearly better
captions at similar speed. Leave **Language** blank to detect it.

## Hindi and Hinglish

When the audio is Hindi — including Hindi mixed with English — captions are
written in Hinglish, the way people type it: "kya scene hai bhai", "aaj bahut
maza aaya", with English words in English. Choose Devanagari instead in
**Settings → Hindi captions** or with `--hindi devanagari`.

Stock Whisper writes Hindi in Devanagari and hears many English words as Hindi
("phokas" for "focus"), which no conversion can fix. So Hindi audio goes to a
model trained on Hinglish. All are free, run locally and download once on first
use:

| Key | Model | Download | Notes |
|---|---|---|---|
| `hinglish-prime` (default) | Oriserve Hindi2Hinglish Prime | 1.6 GB | Most complete, best-spelled |
| `hinglish-apex` | Oriserve Hindi2Hinglish Apex | 0.8 GB | Twice as fast, ~1 GB of GPU memory |
| `hinglish-trelis` | Trelis whisper-hinglish-preview | 1.6 GB | Independent model, no punctuation; a second opinion |
| `large-v3-turbo` | Stock Whisper + converter | 1.6 GB | Used for Devanagari captions |
| `large-v3` | Stock Whisper + converter | 3.1 GB | Slowest |

Pick one in the web app or with `--hindi-model KEY`; `peaksift hindi-models`
lists them. Any other faster-whisper (CTranslate2) model works: put its Hugging
Face ID or a local folder in `whisper.hindi_model`.

Tested on six 30-second clips of Hindi gaming commentary
(`scripts/caption_benchmark.py` writes a listen-and-compare report you can run
on your own clips):

| Model | Readable words | Time per 30 s clip (RTX 5060 laptop) |
|---|---|---|
| Stock Whisper turbo + converter | 46% | 16.5 s |
| … + a local AI spelling fix | 47% | 10.3 s |
| `hinglish-prime` | 78% | 2.0 s |
| `hinglish-apex` | 75% | 0.9 s |
| `hinglish-trelis` | 72% | 2.4 s |

"Readable words" is the share of words found in an English or common-Hinglish
word list: a rough measure. An AI spelling pass barely helps, because it can't
recover words that were misheard.

**How the Hinglish models are run.** They were trained on short clips and lose
two of Whisper's long-audio habits, which only shows on real videos. Decoded
the usual way they re-read speech they had already written (a fifth of all
phrases repeated on an 8-minute stream, with loops like "var gaya" ×40), so
they're run on 10-second speech chunks, each decoded once, in batches — 1%
repeats, about 8× faster. And Prime sometimes answers a chunk with the word
"nan" instead of the speech; that token is blocked while decoding.

## Captions from Gemini (optional)

Choosing **Gemini** for where caption words come from (or `--captions-from gemini`) has Gemini
re-transcribe each *exported clip* with your Gemini key. Only those seconds of
audio are sent, never the whole video. Detection, cut points and timings still
come from the local transcript: Gemini's text is lined up with Whisper's words
by sound and takes their timings. If Gemini is busy, out of quota, or returns
something that doesn't match the audio (a translation, a summary), the local
words are used. Results are cached, so re-rendering costs nothing.

## Editing captions

No model is perfect. **Edit captions** on any clip lets you fix words and
timings, add or delete words, and replace a misheard name in every clip at
once. See [Web app → Caption editor](WEB_APP.md#caption-editor).

## Subtitle files

Every clip has **.srt** and **.vtt** downloads (with your edits), timed from the
start of the clip, for platforms that take a separate caption file. Tick
**Also save an .srt / .vtt file next to each clip** in Settings to write them
next to every export automatically.

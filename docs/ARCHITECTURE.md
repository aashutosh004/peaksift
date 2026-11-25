# PeakSift architecture

A single Python process serving a REST API and a no-build web app on one port,
with a pipeline that turns long videos and live streams into short clips.

```
                     ┌──────────────────────────────────────┐
  browser  ◀────────▶│ FastAPI (backend/peaksift/app.py)   │
                     │  · /api/*        REST + SSE          │
                     │  · /, /ui/*      web app (plain JS)  │
                     └───────────────┬──────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │ JobQueue — one video at a time, FIFO │
                     └───────────────┬──────────────────────┘
                                     │
   ┌─────────────────────────────────▼─────────────────────────────────┐
   │ PipelineRunner                                                    │
   │                                                                   │
   │ prepare → transcribe → signals → highlights → reframe → captions  │
   │ → export     (LiveSession runs the same stages per 60 s segment)  │
   │                                                                   │
   │  each stage writes artifacts to work/{job_id}/                    │
   └───────────────────────────────────────────────────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │ SQLite (.local/data/peaksift.db)     │
                     └──────────────────────────────────────┘
```

## Why one process

PeakSift is a local tool used by one person at a time. A queue broker, a worker
fleet, and a separate frontend server would each add an install step and a
failure mode in exchange for concurrency nobody needs. The web app is plain
HTML/JS served by the same process, so there is no Node.js and no build step.

The same reasoning drives the FIFO queue: transcription, face detection, and
encoding each want the entire GPU. Running two jobs at once on a 6 GB card makes
both slower than running them in sequence.

## Layout

```
backend/peaksift/
├── app.py            FastAPI application; serves webui/
├── cli.py            Typer CLI — clip, live, snap, layout, rerender, serve, doctor, config
├── portable.py       project-local mode: data, caches and ffmpeg under .local/
├── paths.py          artifact layout under PEAKSIFT_HOME
├── config.py         settings + secrets (config.json in portable mode, else OS keyring)
├── system.py         environment probing (ffmpeg, GPU, deps)
├── models.py         face model download and cache
├── speech_models.py  Hindi/Hinglish speech model catalogue
├── hinglish.py       Devanagari → Hinglish transliteration
├── faces_cv.py       YuNet face detection + FER+ emotions (OpenCV)
├── layouts.py        facecam/gameplay layouts: per job, per channel, detected
├── rerender.py       re-export finished clips without re-detecting
├── api/              REST routers: jobs, clips, settings, ui, captions, layouts
├── db/               SQLite schema, migrations, row models, CRUD
├── jobs/             queue, SSE broker, job ownership across processes
├── providers/        LLM adapters behind one Protocol
├── prompts/          versioned prompt text (highlight, re-rank)
├── sources/          platforms: YouTube, Twitch, Kick — links, chat, VOD replay
├── signals/          per-second audio, visual, face and chat signals; fusion; selection
├── live/             live recorder (segments, OBS listener) and LiveSession
├── webui/            the web app (HTML/CSS/JS, no build step)
├── picker/           the facecam/gameplay picker page
├── assets/fonts/     bundled OFL caption fonts
└── pipeline/
    ├── ffmpeg.py     probing, running, filtergraph escaping
    ├── transcript.py the shared word-level model
    ├── ingest.py     yt-dlp and file upload
    ├── prepare.py    audio extraction, thumbnails, silence map
    ├── transcribe.py faster-whisper, Hindi model routing, WhisperX diarization
    ├── align.py      align untimed text (Gemini) to timed Whisper words
    ├── cloud_captions.py  optional Gemini caption pass
    ├── clip_words.py which words a clip shows: edits > Gemini > Whisper
    ├── signal_highlights.py  multi-signal moments + optional LLM re-rank
    ├── framing.py    per-clip layout choice, webcam on/off/moved switching
    ├── boundaries.py sentence snap, duration clamp, silence alignment
    ├── highlights.py windowing, dedupe, ranking
    ├── captions.py   ASS generation and the four presets
    ├── export.py     the render
    ├── runner.py     stage orchestration and resume
    └── reframe/      scenes, faces, tracker, speaker, smoothing, croppath
```

## Load-bearing decisions

### Word indices, not seconds

Highlight detection returns `start_word_index` / `end_word_index`. Timing is
then looked up from measured word timestamps.

Language models are unreliable at arithmetic and completely reliable at copying
a number they can see. Asking for seconds produces clips that start in the wrong
place; asking for an index the model is reading off the page does not.

### Stages are resumable because artifacts are files

Every stage writes its output to `work/{job_id}/`. A retry checks what exists and
starts at the first missing artifact. When transcription took eleven minutes and
the LLM call then hit a rate limit, that difference matters.

### Crop paths never interpolate across a cut

The reframe stage detects shots first and computes a crop path per shot. Panning
through an edit is the single most obvious sign of an auto-reframed video.

### Per-shot segments, concatenated in one filtergraph

ffmpeg filtergraphs cannot change frame size mid-stream, so a clip containing
both a wide two-shot and a tight single can't use one crop. Each shot is trimmed,
cropped independently, scaled to a common output size, and concatenated — all in
a single pass.

Panning within a segment is a piecewise-linear expression over `t` rather than a
`sendcmd` script: it stays in one filtergraph, survives seeking, and can be read
in a log when a render looks wrong.

### Filtergraph paths use bare relative names

Filter option values pass through two unescaping rounds, so a Windows drive
letter needs `C\\:`, not `C\:`. Quoting interacts badly with apostrophes.

Rather than out-escaping this, renders run with ffmpeg's working directory set to
the render workspace and reference `captions.ass` and `fonts` by bare name.
Nothing needs escaping because nothing has a special character in it. Input and
output paths stay absolute — they are ordinary argv arguments, not filtergraph
values.

### Probe capabilities, don't assume them

Two checks look redundant and are not:

- `nvidia-smi` seeing a GPU does not mean CTranslate2 can use it. Missing cuDNN
  is common, and the failure appears at first transcribe.
- `h264_nvenc` appearing in `ffmpeg -encoders` does not mean it encodes. A build
  can require a newer NVENC API than the installed driver provides, and that
  also only surfaces mid-render.

Both are probed functionally, and `peaksift doctor` reports what it actually
tried.

### Compute type follows the hardware

Pre-Volta CUDA devices have no fp16 tensor cores, so `float16` inference is no
faster than `int8_float16` and often slower. `system.py` picks by compute
capability rather than assuming "GPU means float16".

## Data model

Six tables (`db/schema.py`), migrated by SQLite's `user_version` pragma. Each
migration is append-only: once a user's database is at v3, editing migration 2
silently diverges their schema from a fresh install's.

| table        | holds                                              |
| ------------ | -------------------------------------------------- |
| `sources`    | ingested media and probed metadata                 |
| `jobs`       | pipeline runs, status, progress, settings snapshot |
| `transcripts`| pointer to the transcript JSON, model, diarization  |
| `clips`      | detected clips with boundaries and scores           |
| `clip_edits` | user caption edits, style, ratio                    |
| `exports`    | rendered files                                      |

## Provider abstraction

`providers/base.py` defines one Protocol. Four adapters implement it; the
OpenAI-compatible one is the widest, because a configurable `base_url` also
covers OpenRouter, Groq, DeepSeek, Together, and any local server.

The retry-with-feedback loop lives in the base class: a schema violation is
retried once with the validation error appended to the prompt. Small local
models fail the contract often enough that this converts most failures into
successes, and it costs nothing when the first response is already valid.

## Concurrency

The pipeline is blocking work (ffmpeg, Whisper, MediaPipe) with one async stage.
It runs in a worker thread with its own event loop, so the web server stays
responsive during a job.

That thread boundary is why the SSE broker marshals every publish through
`call_soon_threadsafe` — calling `put_nowait` on the loop's queues from a worker
thread would corrupt them.

SQLite uses WAL so the pipeline can commit progress while the UI reads.
Connections are per-thread, because `sqlite3` objects cannot be shared.

## Testing

- **Unit** — boundary refinement, ASS generation, crop-path expressions,
  provider JSON handling, escaping.
- **Render** (`-m slow`) — real ffmpeg encodes, asserting on probed output and
  frame hashes. These catch what unit tests cannot: a filtergraph that parses but
  produces the wrong thing, a font libass can't resolve, an encoder flag the
  local build rejects.
- **API** — the real app with its lifespan running, so migrations, broker
  binding, and queue startup are covered.
- **Golden** (`-m golden`) — the §6.4 reframe acceptance bar on a fixed
  three-video set. A release gate.

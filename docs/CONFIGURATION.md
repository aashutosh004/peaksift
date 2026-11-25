# Configuration

Most settings are in the web app's **Settings** page and the **Options** on the
New clips page. Everything is stored in `config.json` (`peaksift config path` shows
where); `peaksift config show` prints it without secrets. Command-line options
override it for one run.

## Where files live

When PeakSift runs from its folder, everything stays inside it:

```
.venv/                     Python packages
.local/tools/ffmpeg/       ffmpeg (Windows)
.local/cache/              speech and face models, package caches
.local/data/               PeakSift's home:
    config.json            settings and API keys
    peaksift.db            jobs and clips
    media/                 downloaded videos
    work/<job>/            transcript, signals, crops (lets jobs resume)
    exports/<job>/         your clips, clips.json, clips.md
    layouts.json           saved per-channel layouts
```

This happens automatically when `.local/` exists next to `pyproject.toml`
(`backend/peaksift/portable.py`). Set `PEAKSIFT_HOME` to keep the data
somewhere else. Installed any other way, data goes to `~/.peaksift`.

API keys live in `config.json` rather than your system's credential store, so
deleting the folder removes them too. Keep that file private.

## Settings

### `whisper`

| Key | Default | Meaning |
|---|---|---|
| `model` | `small` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, or a path |
| `language` | `""` | Spoken language code; blank detects it |
| `compute_type` | `""` | Force a precision; blank picks what your hardware supports |
| `diarization` | `false` | Label speakers (needs the `diarization` extra) |
| `hindi_script` | `hinglish` | `hinglish` or `devanagari` captions for Hindi audio |
| `hindi_model` | `hinglish-prime` | Model for Hindi audio (see [Captions](CAPTIONS.md)) |
| `caption_source` | `whisper` | `whisper`, or `gemini` to re-transcribe exported clips |

### `clips`

`min_duration_s` (20), `max_duration_s` (90), `max_clips` (10).

### `detection`

| Key | Default | Meaning |
|---|---|---|
| `mode` | `multimodal` | Signals, or `transcript` for AI-only ranking |
| `profile` | `auto` | `gaming`, `just_chatting`, `podcast`, or picked from the picture |
| `llm_rerank` | `auto` | AI judge: `auto` (when reachable), `on`, `off` |
| `visual_mode` | `auto` | `full`, `keyframes` (fast), or keyframes beyond 2 hours |
| `sample_fps` / `face_fps` | 2 / 1 | Frames measured per second for motion / faces |
| `faces` | `true` | Measure faces and reactions |
| `weights` | `{}` | Per-signal weights, e.g. `{"chat_hype": 2, "motion": 0}` |
| `chat_lag_s` | `null` | Fixed chat delay; `null` estimates it |
| `oversample` | 2.5 | Candidates considered per final clip |

### `selection`

`view` (`overall`, `per_hour`, `per_chapter`), `bucket_minutes` (60),
`per_bucket` (3), `min_per_bucket` (1).

### `export`

| Key | Default | Meaning |
|---|---|---|
| `ratio` | `9:16` | `9:16`, `1:1`, `16:9` |
| `layout` | `auto` | `auto`, `gaming`, `full`, `track`, `center` |
| `split_ratio` | 0.35 | Facecam panel height (0.25–0.70) |
| `facecam_position` | `top` | `top` or `bottom` |
| `two_person` | `split` | `split` or `track` |
| `auto_switch_layout` | `true` | Follow the webcam on/off in gaming clips |
| `caption_style` | `bold_pop` | See `peaksift styles` |
| `loudness_lufs` | -14 | Loudness target |
| `prefer_hardware_encoder` | `true` | Use NVENC when it works |
| `crf` | 18 | Quality for software encoding (lower = better, bigger) |
| `write_srt` / `write_vtt` | `false` | Save subtitle files next to each clip |

### `ingest`

| Key | Default | Meaning |
|---|---|---|
| `ytdlp_format` | best ≤1080p | yt-dlp format selector |
| `cookies_from_browser` | `""` | `chrome`, `edge`, `firefox`, `brave`… for videos that need a login |
| `fetch_chat` | `true` | Download chat replay with VODs |
| `fetch_comments` | `true` | Read YouTube comments for timestamps |
| `max_comments` | 300 | How many comments to read |
| `prefer_youtube_captions` | `false` | Use YouTube's captions instead of Whisper |

### `live`

`segment_s` (60), `max_height` (720), `finalize_after_s` (45), `backfill`
(true), `render_while_live` (true), `clips_per_hour` (4).

### AI providers

`active_provider` (`ollama`, `anthropic`, `openai`, `gemini`) and
`providers.<name>.model`; `providers.openai.base_url` points the OpenAI provider
at any compatible server (OpenRouter, Groq, DeepSeek, LM Studio).

## Environment variables

| Variable | Meaning |
|---|---|
| `PEAKSIFT_HOME` | Data folder |
| `PEAKSIFT_PORTABLE=0` | Don't use the project-local `.local/` folder |
| `PEAKSIFT_NO_KEYRING=1` | Store API keys in `config.json` (the default in the folder install) |
| `PEAKSIFT_<PROVIDER>_KEY` | API key for one run, e.g. `PEAKSIFT_GEMINI_KEY` |
| `PEAKSIFT_NO_WORKER=1` | Serve the API without processing jobs (debugging) |

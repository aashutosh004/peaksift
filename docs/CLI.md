# Command line

From the PeakSift folder, run commands as:

- **Windows:** `.\peaksift <command>` (in PowerShell or cmd)
- **macOS / Linux:** `./peaksift.sh <command>`

Both are thin wrappers around `python -m peaksift` using the project's `.venv`.
Add `--help` to any command for its options.

## Making clips

### `clip <link or file>`

Turn a video into clips. A link that is live right now switches to live mode.

```powershell
.\peaksift clip "https://www.youtube.com/watch?v=..."
.\peaksift clip "https://www.twitch.tv/videos/123" --per-hour 3
.\peaksift clip "D:\recordings\stream.mp4" --layout gaming --split 40
.\peaksift clip "https://www.youtube.com/watch?v=..." --no-llm -n 5
```

| Option | Meaning |
|---|---|
| `-n, --max-clips N` | How many clips (best overall view) |
| `--per-hour N` | Best N of every hour (switches to the per-hour view) |
| `--view overall\|per_hour\|per_chapter` | How the final clips are chosen |
| `--profile auto\|gaming\|just_chatting\|podcast` | Signal weighting |
| `--no-llm` | Rank on signals only, no AI judge |
| `-p, --provider NAME` | AI judge provider for this run (`ollama`, `anthropic`, `openai`, `gemini`) |
| `--mode multimodal\|transcript` | Signals (default), or the transcript-only AI ranking |
| `--layout auto\|gaming\|full\|track\|center` | Framing (see [Layouts](LAYOUTS.md)) |
| `--split 25-70` | Facecam panel height in the gaming layout (%) |
| `--facecam-bottom` | Gameplay on top, facecam below |
| `--no-auto-switch` | Keep facecam + gameplay even while the webcam is off |
| `--two-person split\|track` | Two people on screen: split screen, or follow the speaker |
| `--centre-crop` | Skip face tracking; centre-crop everything |
| `-r, --ratio 9:16\|1:1\|16:9` | Output shape |
| `-s, --style NAME` | Caption style (`peaksift styles` lists them) |
| `--whisper-model NAME` | Whisper model for this run (`small`, `large-v3-turbo`, …) |
| `--hindi hinglish\|devanagari` | Script for Hindi captions |
| `--hindi-model KEY` | Speech model for Hindi audio (`peaksift hindi-models`) |
| `--captions-from whisper\|gemini` | Where caption words come from |
| `--diarize` | Label speakers (needs the `diarization` extra and a HuggingFace token) |

### `live <link | obs>`

Clip a live stream as it happens.

```powershell
.\peaksift live "https://www.twitch.tv/somechannel"
.\peaksift live "https://www.youtube.com/@channel/live" --max-hours 3
.\peaksift live obs --chat https://www.twitch.tv/you
```

| Option | Meaning |
|---|---|
| `--backfill / --no-backfill` | Also clip what happened before you joined (default on) |
| `--max-hours H` | Stop recording after H hours |
| `--per-hour N` | Clips to keep per hour (default 4) |
| `--chat URL` | OBS mode: your channel, to read its live chat |
| `--port N` | OBS mode: local RTMP port (default 1935) |
| `--profile`, `--layout`, `--split`, `--facecam-bottom`, `--no-llm`, `--whisper-model`, `-r` | As for `clip` |

While it runs: **s** clips the last 30 s, **l** the last 60 s, **q** stops.

### `snap [seconds]`

Clip the last N seconds (5–180, default 30) of the live session running in
another window. Handy on a Stream Deck button.

## Fixing clips

| Command | What it does |
|---|---|
| `layout [job]` | Open the facecam/gameplay picker on a frame of the job (`latest` by default); re-renders after you save |
| `layouts [--delete KEY]` | List or delete saved per-channel layouts |
| `rerender [job] [-c RANK]…` | Render a job's clips again (all, or the given ranks) with current settings; nothing is re-detected. Takes `--layout`, `--split`, `--facecam-bottom`, `--two-person` |
| `jobs [-n N]` | List recent jobs and their ids |

## Setup and maintenance

| Command | What it does |
|---|---|
| `serve [--port N] [--no-open]` | Start the web app (default http://localhost:8000) |
| `doctor` | Check Python, ffmpeg, GPU, models and AI providers, and say how to fix anything missing |
| `init` | Create the data folder and database |
| `providers` | Check which AI providers are reachable |
| `hindi-models` | List the speech models for Hindi audio |
| `styles` | List caption styles |
| `fetch-models` | Download the face models ahead of time |
| `update-ytdlp` | Update yt-dlp (do this when a site stops downloading) |
| `config show` / `config path` | Print settings (never secrets) / where `config.json` is |
| `config set-secret NAME` | Store an API key (`anthropic`, `openai`, `gemini`, `huggingface_token`); the value is prompted for |
| `config delete-secret NAME` | Remove a stored key |
| `version` | Print the version |

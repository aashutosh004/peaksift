# Getting started

PeakSift installs entirely inside its own folder. Nothing goes into your system
except what you install yourself (Python, and on macOS/Linux, ffmpeg).

## 1. Prerequisites

| | Windows | macOS | Linux |
|---|---|---|---|
| **Python 3.11 or 3.12** | [python.org](https://www.python.org/downloads/) (tick *Add python.exe to PATH*) or `winget install Python.Python.3.12` | `brew install python@3.12` | `sudo apt install python3.12 python3.12-venv` |
| **ffmpeg** | nothing to do: setup downloads it into the folder | `brew install ffmpeg-full` | `sudo apt install ffmpeg` |
| **Git** (optional) | [git-scm.com](https://git-scm.com) | preinstalled with Xcode tools | `sudo apt install git` |

Python 3.13 isn't supported yet (MediaPipe, used for facial expressions, has no
packages for it). Without Git, download the ZIP from the green **Code** button
on GitHub and unzip it.

**macOS note:** use `ffmpeg-full`. The plain `ffmpeg` formula installs fine but
has no `libass`, so it can't burn captions. If you already have it:
`brew unlink ffmpeg && brew link --force --overwrite ffmpeg-full`.

## 2. Install

### Windows

1. `git clone https://github.com/aashutosh004/peaksift.git` (or unzip the download).
2. Double-click **`Setup-PeakSift.cmd`**.

It finds Python, creates `.venv`, installs the packages (with NVIDIA GPU
support if you have an NVIDIA card), downloads ffmpeg into `.local\tools`, and
runs `peaksift doctor` to check everything. The first run takes a few minutes.
It's safe to run again at any time; it only adds what's missing.

Options (from a terminal): `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cpu`
skips GPU support; `-SkipFfmpeg` uses an ffmpeg already on your PATH.

### macOS / Linux

```bash
git clone https://github.com/aashutosh004/peaksift.git
```

```bash
cd peaksift && ./setup.sh
```

`./setup.sh --cpu` skips GPU support on a Linux machine with an NVIDIA card.
Apple Silicon runs on the CPU path; it works, just slower than an NVIDIA GPU.

## 3. Run it

| | Windows | macOS / Linux |
|---|---|---|
| Web app | double-click `Start-PeakSift.cmd` | `./start.sh` |
| Command line | `.\peaksift <command>` | `./peaksift.sh <command>` |

The web app opens at **http://localhost:8000**. Keep its window open while you
use it; close it (or press Ctrl+C) to stop. It only listens on your own PC.

Try it:

```powershell
.\peaksift clip "https://www.youtube.com/watch?v=..." --max-clips 3
```

Clips land in `.local\data\exports\<job id>\`, with `clips.md` explaining why
each one was picked.

## 4. First-run downloads

The first time each is needed, PeakSift downloads it into `.local/cache`:

- the Whisper speech model (`small` by default: ~0.5 GB; `large-v3-turbo`:
  1.6 GB, recommended with an NVIDIA GPU — pick it in **Settings →
  Transcription**);
- the Hinglish model, the first time Hindi speech is detected (1.6 GB);
- small face-detection models (~40 MB).

## 5. Optional: an AI judge

PeakSift ranks moments from audio, video, faces and chat on its own. An AI
model can re-rank the shortlist and write titles and hooks:

- **Free and local:** install [Ollama](https://ollama.com), run
  `ollama pull qwen3.5:9b` (6.6 GB, fits an 8 GB GPU), then in **Settings → AI
  judge** choose *ollama* and type `qwen3.5:9b` as the model.
- **Hosted:** paste an Anthropic, OpenAI or Gemini key in **Settings**, or run
  `.\peaksift config set-secret gemini`. Only the transcript text of candidate
  moments is sent.

## 6. Windows Smart App Control

Windows 11's Smart App Control blocks unsigned programs, including the
`peaksift.exe` launcher pip creates. Everything PeakSift ships runs through
`python -m peaksift` instead, which isn't affected — so always use
`Start-PeakSift.cmd` and `.\peaksift`, not `.venv\Scripts\peaksift.exe`.

## Updating

```powershell
git pull
```

Then run the setup again (`Setup-PeakSift.cmd` or `./setup.sh`) to pick up new
packages. Your videos, clips and settings in `.local/data` are kept.

## Uninstalling

Delete the folder. That removes the packages, ffmpeg, models, caches, your
clips and your saved API keys. Nothing else was installed.

To keep your clips, copy `.local/data/exports` out first.

## Docker (servers)

For a headless Linux server, `docker compose -f docker/compose.yaml up --build`
runs the web app on port 8000 (add `--profile gpu` for NVIDIA). There's no
login, so only expose it on a network you trust.

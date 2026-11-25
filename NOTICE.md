# Credits and third-party notices

PeakSift is released under the [MIT License](LICENSE). It stands on the work
below; thank you to everyone involved.

## Where PeakSift comes from

PeakSift started as a fork of **[AutoClip](https://github.com/artbyjazi/autoclip)**
by Jad Ghazi (MIT License), which provided the transcription → highlight →
reframe → caption → export pipeline and its database, API and CLI foundations.
PeakSift added the multi-signal highlight engine, live-stream clipping (YouTube,
Twitch, Kick, OBS), long-video views (best per hour / per chapter), the facecam
and gameplay layouts, Hinglish captions, the no-install web app and caption
editor, and project-local installation. The original copyright notice is kept
in [LICENSE](LICENSE), as the MIT License requires.

## Models PeakSift downloads (not included in this repository)

Downloaded on first use into `.local/cache` (or `~/.peaksift`), under their own
licences:

| Model | Used for | Licence |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 conversions by [Systran](https://huggingface.co/Systran) and [Mobius Labs](https://huggingface.co/mobiuslabsgmbh)) | Transcription | MIT |
| [Oriserve Whisper-Hindi2Hinglish Prime](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime) (CTranslate2 conversion: [Hub84/faster-whisper-hinglish-prime](https://huggingface.co/Hub84/faster-whisper-hinglish-prime)) | Hindi → Hinglish captions (default) | Apache-2.0 |
| [Oriserve Whisper-Hindi2Hinglish Apex](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex) (conversion: [abhayhadez/anchor-hinglish-apex-ct2](https://huggingface.co/abhayhadez/anchor-hinglish-apex-ct2)) | Hinglish, lighter option | Apache-2.0 |
| [Trelis whisper-hinglish-preview](https://huggingface.co/Trelis/whisper-hinglish-preview) (conversion: [nikh360pro/capsbro-hinglish-ct2](https://huggingface.co/nikh360pro/capsbro-hinglish-ct2)) | Hinglish, alternative option | Apache-2.0 |
| [YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (OpenCV Zoo) | Finding faces and facecams | MIT |
| [FER+ emotion model](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) (ONNX Model Zoo) | Facial reactions | MIT |
| [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Expressions and face tracking | Apache-2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) (bundled in faster-whisper) | Skipping silence before transcription | MIT |

## Fonts (included)

`backend/peaksift/assets/fonts/`: **Anton** and **Inter**, used in captions,
under the SIL Open Font License 1.1 — see
[OFL.txt](backend/peaksift/assets/fonts/OFL.txt).

`backend/peaksift/webui/fonts/`: **IBM Plex Sans** (© IBM Corp.), used by the web
app, under the SIL Open Font License 1.1 — see
[OFL.txt](backend/peaksift/webui/fonts/OFL.txt).

## Tools PeakSift runs

- **[FFmpeg](https://ffmpeg.org)** — all decoding, cropping, caption burning and
  encoding. Not included in this repository: the Windows setup downloads a build
  by [Gyan Doshi](https://www.gyan.dev/ffmpeg/builds/) (GPL) into `.local/tools`;
  macOS and Linux use the system's ffmpeg. PeakSift calls it as a separate program.
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** (Unlicense) — downloading videos,
  live streams and chat replays from links.
- Python libraries listed in [pyproject.toml](pyproject.toml), each under its own
  licence (FastAPI, Uvicorn, Pydantic, Typer, Rich, NumPy, OpenCV, PySceneDetect,
  pysubs2, httpx, and the Anthropic, OpenAI and Google AI SDKs).

## Optional services

AI re-ranking and titles can use a local model through [Ollama](https://ollama.com)
or a hosted provider (Anthropic, OpenAI, Google Gemini) with your own API key.
Their models and terms are their own.

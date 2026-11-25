# Troubleshooting

Start with `.\peaksift doctor` (Windows) or `./peaksift.sh doctor`: it checks
Python, ffmpeg, the GPU and the AI providers and says how to fix what's
missing. Everything below was hit for real while building PeakSift.

## Installing

**"PeakSift needs Python 3.11 or 3.12".** You have 3.13 or none. Install 3.12
alongside it (see [Getting started](GETTING_STARTED.md)); setup picks the right
one.

**Windows: a program is blocked, or `peaksift.exe` won't start.** Windows 11's
Smart App Control blocks unsigned programs. Use `Start-PeakSift.cmd` and
`.\peaksift`, which run through Python and aren't affected.

**Windows: the setup window closes immediately.** Run it from a terminal to see
the error: `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`.

## Captions

**Captions don't appear, or ffmpeg says "No such filter: ass".** Your ffmpeg has
no libass. macOS: `brew install ffmpeg-full` (not `ffmpeg`). Windows: let setup
download its ffmpeg (don't pass `-SkipFfmpeg`). Linux: the distro package has it.

**Hindi captions have English words spelled wrong ("phokas").** Check that
**Speech model for Hindi** is a Hinglish model (`hinglish-prime`), not
stock Whisper.

**A name is always misheard.** Fix it once in every clip: **Edit captions →
Fix a word everywhere**, then re-render.

## Speed and GPU

**"Library cublas64_12.dll is not found" / transcription falls back to CPU.**
The CUDA libraries aren't installed: run the setup again (it installs GPU
support when `nvidia-smi` works). PeakSift finds them inside `.venv` itself.

**"Requested int8_float16 compute type, but the target device does not
support…"** Clear `whisper.compute_type` in `config.json` and let PeakSift pick.

**Everything is very slow on an 8 GB GPU.** Two big models don't fit at once
(e.g. Whisper large plus a 9B AI model), and Windows then pages GPU memory to
system RAM. PeakSift releases Whisper after transcribing and asks Ollama to
unload soon after ranking; also close other GPU-heavy programs, or pick
`hinglish-apex` (needs ~1 GB) for Hindi.

**Exports are slower than expected.** `doctor` shows whether GPU encoding
(NVENC) works. If your driver is older than the ffmpeg build expects, PeakSift
falls back to CPU encoding: same quality, slower.

## Downloading

**YouTube asks to "sign in to confirm you're not a bot".** Set **Settings →
Downloads → browser cookies** to a browser where you're signed in to YouTube,
and close that browser while downloading (it locks its cookie file). Or
download the video yourself and drop the file in.

**A site that used to work stopped downloading.** Sites change often:
`.\peaksift update-ytdlp`.

**Instagram / Facebook video needs a login.** Same as the YouTube fix: browser
cookies.

## Layout

**The facecam panel shows the wrong thing.** Open **Adjust layout**
and draw the boxes; tick *use for every video of this channel* to keep them.

**The clip keeps switching to the full scene.** The webcam wasn't detected for
a few seconds (dark scene, face turned away). Draw the facecam box in the
picker, or turn off *Show the whole game while the webcam is off* under Options.

## The web app

**Re-render fails with "is open in another program".** A video player (or
another browser tab) has the clip open. Close it and re-render again.

**The page looks outdated after an update.** Reload it (Ctrl+F5).

**Port 8000 is in use.** `.\peaksift serve --port 8080`.

**A job someone else is running in a terminal restarts when the web app
starts.** It won't: jobs running in another PeakSift window are left alone.
A job whose window was closed mid-run is resumed from where it stopped.

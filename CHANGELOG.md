# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/).

## 0.1.0 — first release

PeakSift's first release.

### Finding moments

- **Multi-signal highlights**: audio intensity, shouting, exclamations, speech
  rate, motion, scene changes, faces and facial reactions, live chat (volume,
  laughter, hype, shock, subs, bits, Super Chats), YouTube's "most replayed"
  curve and comment timestamps — each scored against the stream's own rolling
  baseline, with chat delay estimated and removed.
- Profiles for gaming, just chatting and podcasts, chosen automatically.
- **Views**: best overall, best per hour, best per chapter.
- Optional **AI judge** (Ollama, Anthropic, OpenAI, Gemini) re-ranks moments and
  writes titles and hooks; `clips.md` explains every pick.

### Live streams

- Live clipping for **YouTube, Twitch and Kick**, with **backfill** of what
  happened before you joined, and an **OBS input** for your own stream.
- **Instant snapshots**: keys, web buttons or `peaksift snap 30`.

### Framing

- **Facecam + gameplay** layout that follows the webcam: every position it moves
  to, the full scene while it's off, a close-up when the streamer goes
  full-screen.
- **Layout picker** to draw the boxes, with per-channel presets.
- **Two-person split** for podcasts and duo streams.

### Captions

- **Hinglish captions** for Hindi speech from Hinglish-trained models (Prime,
  Apex, Trelis), decoded in short independent chunks to avoid repetition; or
  Devanagari.
- Optional **Gemini caption pass** for exported clips, aligned to Whisper timings.
- **Caption editor**: fix words and timings, fix a word in every clip at once.
- **SRT and VTT** downloads and sidecar files.

### Using it

- **Web app with no build step** (no Node.js): New, Jobs, Job, Live, Settings,
  caption editor and layout picker.
- **One-click setup** for Windows (`Setup-PeakSift.cmd`, which also fetches
  ffmpeg) and `./setup.sh` for macOS/Linux; everything installs inside the
  project folder.
- Command line: `clip`, `live`, `snap`, `layout`, `layouts`, `rerender`,
  `hindi-models`, and the rest.

### Fixed along the way

- Jobs running in a terminal are no longer restarted by a web app that starts
  meanwhile (two copies used to fight over the GPU).
- Re-rendering a clip that's open in the browser works on Windows.
- Whisper models are released after transcription, so an 8 GB GPU doesn't page
  to system memory on the next job.
- Ollama "thinking" models answer directly and unload soon after ranking.
- Web app files are revalidated, so an update never leaves an old page running.

# The web app

Start it with `Start-PeakSift.cmd` (Windows), `./start.sh` (macOS/Linux) or
`peaksift serve`. It opens **http://localhost:8000**, follows your system's
light or dark setting, and is plain HTML and JavaScript served by PeakSift
itself: nothing to build, no Node.js.

## New clips

Paste a link — YouTube, Twitch, Kick, TikTok, Instagram, Facebook or anything
else yt-dlp supports — and press **Find clips**, or **Upload a video file** (or
drop one onto the page).

- A finished video or stream is downloaded (with its chat replay and comments
  when available) and clipped.
- A link that is **live right now** starts a live session (see [Live](LIVE.md)).

**Options** (all optional; the defaults come from Settings):

| Group | Option | What it does |
|---|---|---|
| Finding moments | Type of video | Detect, Gaming, Just chatting or Podcast — how much each signal counts |
| | Which clips | Best overall, best of each hour, or best of each chapter, and how many |
| | AI judge | Use the AI model to re-rank and title: when available, always, or off |
| Framing | Layout | Automatic, Facecam over gameplay, Full scene, Follow the speaker, Centre |
| | Show the whole game while the webcam is off… | Switch layouts as the facecam comes and goes |
| | Facecam height, Facecam position | Size of the facecam panel (25–70%), on top or bottom |
| | Two people on screen | Split screen, or follow whoever is speaking |
| | Shape | 9:16, 1:1 or 16:9 |
| Captions | Style | The caption look |
| | Words come from | This computer (Whisper) or Gemini |
| | Speech model for Hindi | Which model writes Hindi captions (see [Captions](CAPTIONS.md)) |
| If the link is live | Clips per hour, Stop after, Also clip what happened before I joined | Live session options |

**Clip your own stream from OBS** starts a local OBS input (see
[Live](LIVE.md#obs)).

**Recent** lists your last few videos.

## Videos

Every video and live session, newest first, with its state: *Working* (with
progress), *Ready*, *Failed*, or *Live*.

## A video's page

While it works: the current step and a progress bar, with the steps it has
done and has left (prepare, transcribe, read signals, find moments, frame,
captions, export).

When it's done:

- **Moments** — a strip across the whole video showing how much was happening
  at each point (the amber curve), with each clip marked at its peak. Click a
  number to jump to that clip.
- **The clips**, best first (grouped by hour or chapter if you chose that), each
  with a player, its time in the video, its title, and **why it was picked**
  ("Picked for shouting, chat hype and a surprised face").
- On each clip: **Download**, **Edit captions**, and **…** for *Re-render this
  clip* and the *.srt* / *.vtt* subtitle files.

At the top: **Adjust layout** opens the [layout picker](LAYOUTS.md#the-picker),
and **Re-render all** renders every clip again (after a layout or settings
change) without re-detecting anything.

## Caption editor

**Edit captions** on any clip:

- Click a word to jump the video to it and retype it. **Enter** moves to the
  next word; **Backspace** on an empty word removes it.
- The word being spoken is highlighted while the clip plays.
- Select a word to move its start or end by 0.1 s, remove it, or add a word
  after it.
- **Fix a word everywhere** replaces a word or phrase in every clip of the
  video — for a name or term the model keeps mishearing.
- **Save** keeps your edits; **Save and re-render** also puts them into the
  video. Edits are used by every later re-render until you press **Reset to
  automatic**.
- **Subtitles .srt / .vtt** — subtitle files with your edited words.

Under the title it says where the words came from: *Your edits*, *Words from
Gemini*, or *Automatic words*.

## Live session

For a running live session: time recorded, time analysed (or how much of the
earlier part has been filled in), clips so far and chat messages read; **Clip it
now** buttons for the last 15, 30 or 60 seconds; the activity log; and **Stop and
finish**, which runs the final ranking. Clips appear below as they're made.

Closing the app stops any live session; clips already made are kept.

## Settings

- **AI judge** — choose Anthropic, OpenAI, Google Gemini or Ollama, its model,
  and its API key. Keys are stored in `.local/data/config.json` inside the
  PeakSift folder.
- **Transcription** — speech model, language (blank detects it), Hindi caption
  script, where caption words come from, the speech model for Hindi, and whether
  to also save .srt / .vtt files next to each clip.
- **Clips** (length and count), **Downloads** (browser cookies for videos that
  need a login, chat replay, YouTube comments) and **Live streams** defaults.

**Save settings** stays at the bottom of the page and lights up when something
has changed.

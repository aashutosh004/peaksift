# How moments are found

```
link / file → download → transcribe → measure signals → rank moments → frame → caption → export
```

Each stage saves its results in `.local/data/work/<job>/`, so a job that fails
or is stopped resumes where it left off, and re-rendering never re-detects.

## What it measures

Every signal is measured once per second for the whole video:

| Group | Signals | How |
|---|---|---|
| Audio intensity | `loudness`, `flux`, `shout` | Volume, sudden changes, raised-voice energy (2–5 kHz) |
| Speaking | `speech_rate`, `exclaim` | Word timings; exclamations like "no way", "let's go", "clip that", "bhai", "arre" |
| Motion | `motion` | Frame differences, 2 per second |
| Scene changes | `scene` | Colour changes: cuts, deaths, menus, map changes |
| Faces | `face_present`, `face_size` | YuNet face detection, which also finds small facecams |
| Reactions | `mouth_open`, `face_surprise`, `face_joy`, `head_motion` | MediaPipe expressions, or the FER+ emotion model when MediaPipe can't load |
| Chat | `chat_rate`, `chat_unique`, `chat_laugh`, `chat_hype`, `chat_shock`, `chat_events` | Live chat or chat replay: volume, distinct chatters, laughing/hype/shock emotes and words, subs, bits, Super Chats |
| Viewers | `heatmap`, `comments` | YouTube's "most replayed" curve; `12:34` timestamps in top comments |

Gameplay with no talking at all still has audio, motion, scene and chat
signals to clip from.

## How a moment is picked

1. **Compared with this stream, right now.** Each signal becomes a score against
   a rolling 10-minute baseline (median and spread), so what counts is
   "unusual for this stream", not "loud". A constantly loud FPS streamer
   doesn't flood the results.
2. **Chat lag removed.** Chat reacts several seconds after a moment. The delay is
   estimated by lining chat up against the audio and face reactions, and chat
   is shifted back by it.
3. **Weighted, with an agreement bonus.** Weights come from the profile (gaming,
   just chatting, podcast — picked automatically from the picture). Moments
   where several signals spike together score higher.
4. **Peak → clip.** Each peak becomes a clip that opens on the build-up and
   stays for the reaction, fitted to your length range and snapped to the ends
   of sentences and to pauses.
5. **AI judge (optional).** Each shortlisted moment is shown to the AI model with
   its transcript, the signals that spiked and a sample of chat. It returns a
   score, a title and a hook. The final score is 60% signals, 40% AI.
   `--no-llm` skips this.

## Overall, per hour, per chapter

Detection runs over the whole video; the view only decides which ranked
moments become clips:

- **Overall** — the best N in the whole video.
- **Per hour** (`--per-hour 3`) — the best 3 of every hour, with at least one
  per hour so a quiet hour still shows its best moment. Live streams longer than
  an hour use this.
- **Per chapter** — the best few inside each natural section (a game switch, a
  just-chatting segment), found from shifts in the picture's colours.

Each job writes `clips.json` and `clips.md` next to the clips: every clip's
time, group, score and why it was picked.

## Long videos

Multi-hour sources are measured from keyframes only (much faster) beyond two
hours; set `detection.visual_mode` to `full` to measure every frame anyway.

## Framing and captions

See [Layouts](LAYOUTS.md) and [Captions](CAPTIONS.md). The code-level design is
in [ARCHITECTURE.md](ARCHITECTURE.md).

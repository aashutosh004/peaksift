# Layouts

Vertical clips are cut from wide video, so PeakSift decides second by second
what goes in the frame.

| Layout | What you get | Chosen automatically when |
|---|---|---|
| **Facecam + gameplay** (`gaming`) | The webcam panel above the gameplay | A webcam overlay is found |
| **Full scene** (`full`) | The whole frame, fitted over a blurred copy | Screen or gameplay with no webcam |
| **Follow speaker** (`track`) | A crop that follows whoever is talking | The picture *is* the camera (just chatting, podcasts, IRL) |
| **Centre** (`center`) | A plain centre crop | Only when you choose it |

Choose one with **Framing** in the web app or `--layout` on the command line;
`auto` is the default.

## Gaming: following the webcam

In gaming clips the layout follows what's on screen:

- **webcam on** → facecam + gameplay, with the facecam taken from wherever the
  webcam is at that moment. Many streamers move it between scenes (a corner
  during gameplay, another during cutscenes); every position is found
  separately.
- **webcam off** (hidden, a menu, a cutscene without it) → the full gameplay
  scene.
- **full-screen camera** (a reaction break) → a close-up of the streamer.

A layout must hold for a few seconds before it switches, so a missed detection
doesn't make the clip flicker. Faces in the middle of the screen are treated as
the game (characters, cutscenes), not as a webcam. Captions sit on the seam
between the panels, covering neither the face nor the game.

Turn switching off with the checkbox under Options on the New clips page, or `--no-auto-switch`.

## The picker

When the automatic boxes aren't right, set them yourself: **Adjust layout** on
the video's page, or `peaksift layout latest`.

A frame from the job opens (at its strongest moment; you can pick any moment).
Drag the **red** box over the facecam and the **blue** box over the part of the
game you want (or leave gameplay on *auto*), set the facecam panel height
(25–70%) and top or bottom, and watch the 9:16 preview update. Save, then press
**Re-render all** on the video's page (`peaksift layout` re-renders for you): the
clips are rendered again from the original video, with no quality loss.

- **"Also use for every video and stream of <channel>"** saves a preset; every
  later video and live stream from that channel uses it. `peaksift layouts`
  lists presets, `peaksift layouts --delete twitch:name` removes one.
- Priority: this job's layout → the channel's preset → automatic detection.
- A box you draw is one fixed place, so it's used whenever the webcam is on,
  wherever the face is detected.
- Quick tweaks without the picker: `--split 45` and `--facecam-bottom` on
  `clip`, `live` and `rerender`.

## Two people on screen

When two people are clearly on screen (a podcast, a duo stream), the clip
stacks them: left person on top, right person below, each framed head and
shoulders. It only happens when both faces stay through the moment, are well
apart and similar in size, so a single speaker is never cut in half and a
background face never gets a panel. `--two-person track` turns it off and
follows whoever is speaking instead.

Speaker-following works best with `--diarize` (who spoke when). It needs the
`diarization` extra, which pulls in PyTorch (several GB):
`.venv\Scripts\uv pip install --python .venv\Scripts\python.exe -e ".[diarization]"`
(macOS/Linux: `.venv/bin/uv pip install --python .venv/bin/python -e ".[diarization]"`),
plus a HuggingFace token (`peaksift config set-secret huggingface_token`) and
accepting the pyannote model licences on huggingface.co.

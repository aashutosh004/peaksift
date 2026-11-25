# Contributing to PeakSift

Thanks for looking. PeakSift is MIT and meant to be hacked on.

## Getting set up

You need **Python 3.11 or 3.12** (not 3.13 — MediaPipe has no wheels for it)
and **ffmpeg built with libass**. The normal setup (`Setup-PeakSift.cmd` or
`./setup.sh`) installs the `dev` extra too, so a user install is a dev install:

```bash
git clone https://github.com/aashutosh004/peaksift
cd peaksift
./setup.sh            # Windows: Setup-PeakSift.cmd
```

`peaksift doctor` (run at the end of setup) tells you exactly what's missing.

The web app (`backend/peaksift/webui/`) is plain HTML, CSS and JavaScript with no
build step: edit a file and reload the page. `peaksift serve --reload` restarts
the server on Python changes.

## Before you open a PR

```bash
ruff check . && ruff format --check .
pytest -m "not slow"
pytest -m slow          # real ffmpeg renders; slower, worth it
```

CI runs all of this on Linux, macOS, and Windows. The Windows leg is not
decoration — filtergraph path escaping and keyring behaviour genuinely differ
there, and both have already produced bugs.

## Where things are

See [ARCHITECTURE.md](ARCHITECTURE.md) for the map and the reasoning behind the
load-bearing decisions. Skim the "Load-bearing decisions" section before changing
the pipeline; several choices that look arbitrary are working around something
specific.

## Good first contributions

**Prompts.** `backend/peaksift/prompts/highlight_v1.txt` is plain text and has
more effect on output quality than most code changes. Add `highlight_v2.txt` and
compare on the same source video. No Python required.

**Caption styles.** Add a preset to `PRESETS` in `pipeline/captions.py`. Each is
a dataclass; the UI picks it up automatically through `/api/caption-styles`.

**Provider adapters.** Subclass `LLMProvider`, implement `_complete` and
`health_check`, register it in `providers/__init__.py`. The retry loop, JSON
extraction, and index clamping are inherited.

**Language support.** Whisper handles many languages already, but caption
grouping assumes space-separated words and left-to-right layout.

## Things to know before changing the pipeline

**Migrations are append-only.** Once a migration ships, editing it silently
diverges existing databases from fresh installs. Add a new one.

**Never pass a user-controlled path into a filtergraph.** Use
`ffmpeg.relative_filter_workspace()`. The escaping rules are genuinely awkward
and the failure mode is a render that either dies cryptically or quietly drops
captions.

**Reframe changes need the golden set.** The §6.4 acceptance bar — no visible
jitter, no cut-off faces, speaker on screen ≥95% of speaking time — is a release
gate. If you change smoothing, tracking, or strategy selection, run the golden
tests and say what moved.

**Bias toward stillness.** In reframing, a locked frame that is slightly
off-centre beats a frame that is always correct and always moving. Several
defaults exist specifically to stop the crop from micro-correcting.

## Style

Ruff handles formatting and linting; run it and move on. Beyond that:

- Comments explain *why*, not *what*. If a constant looks arbitrary, say what
  it's working around.
- Python docstrings write preconditions as plain lines under a `Preconditions:`
  header.
- Errors should tell the user what to do. `IngestError` and `ProviderError` both
  carry a `hint` for exactly this, and the UI renders it.

## Reporting bugs

Include the output of `peaksift doctor`. Most reports in this domain come down
to an ffmpeg build without libass, a GPU driver mismatch, or Python 3.13 — and
`doctor` identifies all three immediately.

## Legal

PeakSift bundles yt-dlp. It contains no DRM circumvention and will not accept
contributions that add any.

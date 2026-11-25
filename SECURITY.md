# Security

## Reporting a vulnerability

Please **don't** open a public issue for a security problem. Use GitHub's
[private vulnerability reporting](https://github.com/aashutosh004/peaksift/security/advisories/new)
instead, and I'll respond as quickly as I can.

## What PeakSift does with your data

Worth stating plainly, because "local-first" is easy to claim and easy to get wrong:

- **Media never leaves your machine**, with two opt-in exceptions: a hosted AI
  judge (Anthropic, OpenAI, Gemini) receives the *transcript text* of candidate
  moments, and **Caption words from → Gemini** sends the *audio of the exported
  clips* (only those seconds) to Gemini. Choosing Ollama and local captions sends
  nothing anywhere.
- **API keys** are stored in `.local/data/config.json` inside the PeakSift
  folder when it runs from there (the normal install), so deleting the folder
  removes them. Installed elsewhere, they go to your OS keyring (Credential
  Manager, Keychain, Secret Service); if none exists, PeakSift falls back to a
  file and says so in the UI and in `peaksift doctor`.
- **No telemetry, no analytics, no phone-home.** There is no code that contacts
  a server PeakSift's authors control. Downloads go to the sites you link
  (YouTube, Twitch, …), Hugging Face (speech models) and GitHub (ffmpeg on
  Windows).
- **Everything is inside the PeakSift folder** (`.local/` and `.venv/`) —
  media, transcripts, exports, database, settings, models. Delete the folder and
  nothing remains.

## Running it safely

PeakSift **has no authentication**. It binds to `127.0.0.1` by default, and it should stay that way.

Serving it on `0.0.0.0`, or publishing the Docker port beyond loopback, exposes an unauthenticated API that can read any file path the process can reach and execute ffmpeg against it. If you genuinely need remote access, put it behind a reverse proxy that terminates TLS and handles authentication.

## Scope

In scope: anything that lets a third party read your media, exfiltrate your API keys, or execute code through a crafted input file, URL, or API request.

Out of scope: the consequences of deliberately binding to a public interface, and anything requiring an attacker who already has local access to your user account.

## Supported versions

Only the latest release gets fixes. Fixes land on `main`; there are no backported release branches.

# Live streams

## Platforms

| Platform | Finished videos / VODs | Live | Chat |
|---|---|---|---|
| YouTube | ✅ | ✅ + backfill from the start (DVR) | live chat and replay, "most replayed" curve, comment timestamps |
| Twitch | ✅ | ✅ + backfill from the in-progress VOD | live (IRC) and VOD replay |
| Kick | ✅ | ✅ (no backfill) | live |
| TikTok, Instagram, Facebook, others | ✅ uploaded videos | – | – |

Twitch VOD replay and Kick chat use unofficial interfaces and are best effort.
Instagram and Facebook videos that need a login work with **Settings →
Downloads → browser cookies** (opt-in; close that browser while downloading).

## Starting

Paste the live link on the web app's **New clips** page, or:

```powershell
.\peaksift live "https://www.twitch.tv/somechannel"
```

`peaksift clip` on a link that's live switches to live mode by itself.

## How it works

- The stream is recorded in 60-second pieces (720p, enough for vertical clips).
  Each finished piece is transcribed and measured right away.
- **Joining late:** a backfill worker downloads the earlier part in 10-minute
  blocks (YouTube DVR, Twitch's in-progress VOD) while the live edge keeps being
  clipped. The live edge always comes first.
- A moment becomes a clip once 45 seconds past its end have been measured, so a
  reaction is never cut short. Clips are rendered as they're found, a minute or
  two behind live.
- When the stream ends (or you press Stop / Ctrl+C), a final pass re-ranks
  everything with the per-hour view and renders anything missing. Clips that
  didn't make the final cut move to `extras/`; nothing is deleted.

## Instant snapshots

Clip the last moments on demand — they're always kept, under "Snapshots":

- web app: **Last 15 s / 30 s / 60 s** on the live page;
- terminal running `peaksift live`: **s** (last 30 s), **l** (last 60 s);
- anywhere else (a second terminal, a Stream Deck button): `peaksift snap 30`.

## OBS

Clip your own stream locally, without a platform in between:

```powershell
.\peaksift live obs --chat https://www.twitch.tv/you
```

or **Clip your own stream from OBS** on the New clips page. Then in OBS:
*Settings → Stream → Service: Custom*, Server `rtmp://127.0.0.1:1935/live`,
Stream Key `stream`, and start streaming. PeakSift only listens on this PC.

`--chat` reads your channel's live chat for the chat signals, and that
channel's saved layout is used. To stream to a platform and to PeakSift at the
same time, use OBS's multiple outputs or a multi-RTMP plugin.

## Settings

`live.segment_s` (60), `live.max_height` (720), `live.finalize_after_s` (45),
`live.backfill` (true), `live.render_while_live` (true), `live.clips_per_hour`
(4) — see [Configuration](CONFIGURATION.md).

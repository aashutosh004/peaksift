"""Live recording into fixed-length segments.

yt-dlp resolves the stream (YouTube, Twitch, Kick) to a media URL and the
headers it needs; ffmpeg reads it and writes ``seg_NNNNN.ts`` files of
``segment_s`` seconds each, appending a line to ``segments.csv`` as each one
completes. MPEG-TS segments are independently playable, so each finished
segment can be analysed immediately while recording continues.

Stream URLs expire (YouTube after a few hours, Twitch tokens sooner) and
connections drop, so the recorder re-resolves and restarts ffmpeg whenever it
exits while the stream is still live. Each run's segment times are rebased onto
one session clock, so the analysis timeline stays continuous across restarts.
"""

from __future__ import annotations

import csv
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..pipeline import ffmpeg

log = logging.getLogger(__name__)


@dataclass
class Segment:
    index: int
    path: Path
    #: Session time (seconds from the session origin) where the segment starts.
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


#: Prefer a muxed stream (Twitch, Kick); fall back to separate video + audio
#: (YouTube live serves them as two HLS playlists).
LIVE_FORMAT = (
    "best[height<={h}][vcodec!=none][acodec!=none]"
    "/bestvideo[height<={h}]+bestaudio"
    "/best[vcodec!=none][acodec!=none]/bestvideo+bestaudio/best"
)


def resolve_stream(
    url: str, *, max_height: int, cookies_from_browser: str = ""
) -> list[tuple[str, dict]]:
    """Return the media URL(s) and HTTP headers ffmpeg needs.

    One entry for a muxed stream, two (video, audio) when the platform serves
    them separately.
    """
    import yt_dlp

    options: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": LIVE_FORMAT.format(h=max_height),
    }
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    requested = info.get("requested_formats") or []
    if requested:
        inputs = [(f.get("url", ""), dict(f.get("http_headers") or {})) for f in requested]
    else:
        inputs = [(info.get("url", ""), dict(info.get("http_headers") or {}))]
    inputs = [(u, h) for u, h in inputs if u]
    if not inputs:
        raise RuntimeError("Could not resolve a playable stream URL.")
    return inputs


class SegmentRecorder:
    """Record a live stream into segments in a background thread."""

    def __init__(
        self,
        url: str,
        out_dir: Path,
        *,
        origin_epoch: float,
        segment_s: int = 60,
        max_height: int = 720,
        cookies_from_browser: str = "",
        is_still_live=None,
        listen: bool = False,
    ) -> None:
        self.url = url
        #: OBS mode: ffmpeg listens on ``url`` (rtmp://127.0.0.1:...) and OBS
        #: pushes to it. No yt-dlp, no reconnect loop: when OBS stops, we stop.
        self.listen = listen
        self.out_dir = out_dir
        self.origin_epoch = origin_epoch
        self.segment_s = segment_s
        self.max_height = max_height
        self.cookies_from_browser = cookies_from_browser
        self._is_still_live = is_still_live or (lambda: True)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._segments: list[Segment] = []
        self._run = 0
        self.ended = threading.Event()
        self.error: str = ""

    # -- control -------------------------------------------------------------

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="live-recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                # 'q' lets ffmpeg close the current segment cleanly.
                if proc.stdin:
                    proc.stdin.write(b"q")
                    proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        if self._thread:
            self._thread.join(timeout=15)
        self._collect()

    def segments(self) -> list[Segment]:
        """Completed segments so far, in order."""
        self._collect()
        with self._lock:
            return list(self._segments)

    # -- internals -----------------------------------------------------------

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            if self.listen:
                self._run += 1
                self._record([(self.url, {})], None)
                self._collect()
                break
            try:
                inputs = resolve_stream(
                    self.url,
                    max_height=self.max_height,
                    cookies_from_browser=self.cookies_from_browser,
                )
            except Exception as exc:
                failures += 1
                self.error = str(exc)
                if not self._is_still_live() or failures >= 5:
                    log.info("Stream is no longer available (%s); recording finished.", exc)
                    break
                time.sleep(min(30, 5 * failures))
                continue

            self._run += 1
            run_base = time.time() - self.origin_epoch
            code = self._record(inputs, run_base)
            self._collect()
            if self._stop.is_set():
                break
            if code == 0 and not self._is_still_live():
                break
            failures = failures + 1 if code != 0 else 0
            if failures >= 8:
                log.warning("Recorder failed repeatedly; stopping.")
                break
            log.info("Recorder run %d ended (code %s); reconnecting.", self._run, code)
            time.sleep(3)
        self.ended.set()

    def _record(self, inputs: list[tuple[str, dict]], run_base: float | None) -> int:
        """Run one ffmpeg recording. ``run_base`` None means "when data arrives".

        In listen mode ffmpeg may wait minutes for OBS to connect; segment times
        count from the first packet, so the session offset is stamped when the
        first segment file appears rather than when listening began.
        """
        run_dir = self.out_dir / f"run{self._run:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if run_base is not None:
            (run_dir / "base.txt").write_text(f"{run_base:.3f}", encoding="utf-8")
        else:
            threading.Thread(
                target=self._stamp_base_on_first_segment, args=(run_dir,), daemon=True
            ).start()
        command = [ffmpeg.ffmpeg_path(), "-hide_banner", "-loglevel", "error"]
        if self.listen:
            command += ["-listen", "1", "-timeout", "-1"]
        for media_url, headers in inputs:
            header_blob = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            if header_blob:
                command += ["-headers", header_blob]
            if media_url.startswith("http"):
                command += [
                    "-reconnect",
                    "1",
                    "-reconnect_streamed",
                    "1",
                    "-reconnect_delay_max",
                    "10",
                ]
            command += ["-i", media_url]
        audio_input = len(inputs) - 1
        command += [
            "-map",
            "0:v:0?",
            "-map",
            f"{audio_input}:a:0?",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_s),
            "-segment_format",
            "mpegts",
            "-reset_timestamps",
            "1",
            "-segment_list",
            str(run_dir / "segments.csv"),
            "-segment_list_type",
            "csv",
            str(run_dir / "seg_%05d.ts"),
        ]
        log.info("Recording run %d (session offset %.0fs).", self._run, run_base)
        self._proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        _, stderr = self._proc.communicate()
        if self._proc.returncode not in (0, None) and stderr:
            self.error = (
                stderr.decode("utf-8", "replace").strip().splitlines()[-1:][0]
                if stderr.strip()
                else ""
            )
            log.warning("ffmpeg recorder: %s", self.error)
        return self._proc.returncode or 0

    def _stamp_base_on_first_segment(self, run_dir: Path) -> None:
        first = run_dir / "seg_00000.ts"
        while not self._stop.is_set() and not self.ended.is_set():
            if first.exists():
                (run_dir / "base.txt").write_text(
                    f"{time.time() - self.origin_epoch:.3f}", encoding="utf-8"
                )
                log.info("Stream connected; recording.")
                return
            time.sleep(0.25)

    def _collect(self) -> None:
        """Read every run's segment list into one session-time-ordered list."""
        found: list[Segment] = []
        for run_dir in sorted(self.out_dir.glob("run*")):
            listing = run_dir / "segments.csv"
            base_file = run_dir / "base.txt"
            if not listing.exists() or not base_file.exists():
                continue
            base = float(base_file.read_text(encoding="utf-8") or 0)
            try:
                with listing.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.reader(handle))
            except OSError:
                continue
            for row in rows:
                if len(row) < 3:
                    continue
                try:
                    start, end = float(row[1]), float(row[2])
                except ValueError:
                    continue
                path = run_dir / row[0]
                if path.exists() and end > start:
                    found.append(
                        Segment(index=0, path=path, start_s=base + start, end_s=base + end)
                    )
        found.sort(key=lambda s: s.start_s)
        for i, segment in enumerate(found):
            segment.index = i
        with self._lock:
            self._segments = found


def concat_segments(segments: list[Segment], destination: Path) -> Path:
    """Join segments losslessly into one file (for rendering or the final archive)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    listing = destination.with_suffix(".txt")
    listing.write_text("".join(f"file '{s.path.as_posix()}'\n" for s in segments), encoding="utf-8")
    ffmpeg.run(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(destination),
        ]
    )
    listing.unlink(missing_ok=True)
    return destination

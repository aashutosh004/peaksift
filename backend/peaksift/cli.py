"""PeakSift command-line interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__, config, paths, system

app = typer.Typer(
    name="peaksift",
    help="Turn long videos and live streams into captioned vertical clips, on your own PC.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

OK = "[green]OK[/green]"
WARN = "[yellow]WARN[/yellow]"
FAIL = "[red]FAIL[/red]"


def _status(passed: bool, warn_only: bool = False) -> str:
    if passed:
        return OK
    return WARN if warn_only else FAIL


def _ffmpeg_install_hint() -> str:
    """Platform-specific install advice that actually yields a usable ffmpeg.

    The distinction matters more than it looks: the default package on both
    Windows and macOS is a reduced build without libass, so following the
    obvious instruction produces an ffmpeg that installs fine and then cannot
    burn a single caption.
    """
    if sys.platform == "darwin":
        return (
            "  macOS: [cyan]brew install ffmpeg-full[/cyan] — not [cyan]ffmpeg[/cyan], "
            "which Homebrew now ships as a reduced build with no libass.\n"
            "  If you already installed the plain formula:\n"
            "    [cyan]brew install ffmpeg-full && brew unlink ffmpeg && "
            "brew link --force --overwrite ffmpeg-full[/cyan]"
        )
    if sys.platform == "win32":
        return (
            "  Windows: [cyan]winget install Gyan.FFmpeg[/cyan] — the [bold]full[/bold] "
            "build, not [cyan]Gyan.FFmpeg.Essentials[/cyan], which omits libass."
        )
    return (
        "  Debian/Ubuntu: [cyan]sudo apt install ffmpeg[/cyan]\n"
        "  Fedora: [cyan]sudo dnf install ffmpeg[/cyan] (RPM Fusion)\n"
        "  Arch: [cyan]sudo pacman -S ffmpeg[/cyan]"
    )


@app.command()
def version() -> None:
    """Print the PeakSift version."""
    console.print(f"peaksift {__version__}")


@app.command()
def doctor() -> None:
    """Check that this machine can run the pipeline, and explain anything missing."""
    report = system.refresh()
    settings = config.load()

    console.print()
    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold]PeakSift {__version__}[/bold]\n{report.platform}\nHome: {paths.root()}"
            ),
            border_style="cyan",
        )
    )

    remediation: list[str] = []

    # --- Runtime -----------------------------------------------------------
    table = Table(title="Runtime", show_header=True, header_style="bold", title_justify="left")
    table.add_column("Check")
    table.add_column("Status", width=6)
    table.add_column("Detail")

    table.add_row("Python", _status(report.python_ok), report.python_version)
    if not report.python_ok:
        remediation.append(
            "Python must be >=3.11,<3.13 — MediaPipe publishes no wheels for 3.13+, "
            "so the reframe stage cannot run. Create the environment with "
            "[cyan]uv venv --python 3.11[/cyan]."
        )

    ff = report.ffmpeg
    table.add_row("ffmpeg", _status(ff.found), ff.version or "not found")
    table.add_row("ffprobe", _status(ff.ffprobe_found), ff.path or "not found")
    if not ff.found or not ff.ffprobe_found:
        remediation.append(
            "Install ffmpeg and make sure both [cyan]ffmpeg[/cyan] and [cyan]ffprobe[/cyan] "
            f"are on PATH.\n{_ffmpeg_install_hint()}"
        )
    else:
        table.add_row("  libass (captions)", _status(ff.has_libass), "subtitle burn-in")
        table.add_row("  fontconfig", _status(ff.has_fontconfig, warn_only=True), "font resolution")
        table.add_row("  libx264", _status(ff.has_libx264), "software H.264 encode")
        if ff.has_nvenc and not ff.nvenc_works:
            table.add_row("  h264_nvenc", WARN, ff.nvenc_error or "listed but not usable")
            remediation.append(
                "This ffmpeg build lists [cyan]h264_nvenc[/cyan] but the GPU driver is "
                "too old for the NVENC API it was built against, so exports will use "
                "the CPU encoder. Update your NVIDIA driver to enable GPU encoding — "
                "or ignore this, since software encoding produces identical quality, "
                "just slower."
            )
        else:
            table.add_row(
                "  h264_nvenc",
                _status(ff.nvenc_works, warn_only=True),
                "GPU encode" if ff.nvenc_works else "not available (software encode will be used)",
            )
        missing = ff.missing_filters
        table.add_row(
            "  required filters",
            _status(not missing),
            "all present" if not missing else f"missing: {', '.join(missing)}",
        )
        optional_missing = [f for f in system.OPTIONAL_FILTERS if f not in ff.filters]
        if optional_missing:
            table.add_row(
                "  optional filters",
                WARN,
                f"missing: {', '.join(optional_missing)}",
            )
        if not ff.has_libass or "ass" in missing:
            remediation.append(
                "This ffmpeg build has no libass, so captions cannot be burned in — "
                f"which is most of what PeakSift does.\n{_ffmpeg_install_hint()}"
            )

    console.print()
    console.print(table)

    # --- Acceleration ------------------------------------------------------
    gpu = report.gpu
    accel_table = Table(
        title="Acceleration", show_header=True, header_style="bold", title_justify="left"
    )
    accel_table.add_column("Check")
    accel_table.add_column("Status", width=6)
    accel_table.add_column("Detail")

    accel_table.add_row("Mode", OK, gpu.accel.upper())
    if gpu.name:
        vram = f", {gpu.vram_mb} MiB VRAM" if gpu.vram_mb else ""
        cc = f", compute {gpu.compute_capability}" if gpu.compute_capability else ""
        accel_table.add_row("Device", OK, f"{gpu.name}{vram}{cc}")
    if gpu.driver_version:
        accel_table.add_row("Driver", OK, gpu.driver_version)

    if gpu.name and not gpu.ctranslate2_cuda and "ctranslate2" in report.deps.load_errors:
        accel_table.add_row("CUDA for Whisper", FAIL, "the speech engine can't load (below)")
    elif gpu.name and not gpu.ctranslate2_cuda:
        accel_table.add_row("CUDA for Whisper", FAIL, "CTranslate2 cannot see the GPU")
        remediation.append(
            "An NVIDIA GPU is present but CTranslate2 can't use it — usually the CUDA "
            "libraries are missing. Run the setup again (Setup-PeakSift.cmd or "
            "./setup.sh), which installs them when it finds an NVIDIA card, then re-run "
            "doctor. Transcription falls back to CPU until this is fixed."
        )
    elif gpu.ctranslate2_cuda:
        accel_table.add_row("CUDA for Whisper", OK, "CTranslate2 sees the GPU")

    if gpu.supported_compute_types:
        accel_table.add_row(
            "Supported compute types", OK, ", ".join(sorted(gpu.supported_compute_types))
        )

    reason = ""
    if gpu.accel == "cuda" and not gpu.compute_type.startswith("float16"):
        reason = " (this GPU has no usable fp16 path)"
    accel_table.add_row("Whisper compute type", OK, f"{gpu.compute_type}{reason}")

    override = settings.whisper.compute_type
    if override:
        usable = not gpu.supported_compute_types or override in gpu.supported_compute_types
        accel_table.add_row(
            "  override in config",
            _status(usable, warn_only=True),
            f"forced to {override}" if usable else f"{override} is unsupported — ignored",
        )
        if not usable:
            remediation.append(
                f"[cyan]whisper.compute_type[/cyan] is set to [cyan]{override}[/cyan], which "
                "this device does not support, so it is being ignored. Clear it in "
                f"{paths.config_path()} to silence this."
            )

    console.print()
    console.print(accel_table)

    # --- Python dependencies ----------------------------------------------
    deps = report.deps
    dep_table = Table(
        title="Dependencies", show_header=True, header_style="bold", title_justify="left"
    )
    dep_table.add_column("Package")
    dep_table.add_column("Status", width=6)
    dep_table.add_column("Used for")

    dep_table.add_row("faster-whisper", _status(deps.faster_whisper), "transcription")
    dep_table.add_row("mediapipe", _status(deps.mediapipe), "face detection / reframe")
    dep_table.add_row("scenedetect", _status(deps.scenedetect), "shot boundaries")
    dep_table.add_row(
        "whisperx",
        _status(deps.whisperx, warn_only=True),
        "speaker diarization" + ("" if deps.whisperx else " (optional extra)"),
    )

    for module, error in deps.load_errors.items():
        dep_table.add_row(module, FAIL, "installed but won't load")
        if system.blocked_by_windows(error):
            remediation.append(
                f"Windows Smart App Control blocked [cyan]{module}[/cyan] ({error}). "
                "This happens with very new package releases. Run the setup again — it "
                "installs versions known to load — and if it persists, report it on "
                "PeakSift's GitHub page."
            )
        else:
            remediation.append(f"[cyan]{module}[/cyan] is installed but fails to load: {error}")

    if not deps.whisperx and settings.whisper.diarization:
        remediation.append(
            "Diarization is enabled in settings but WhisperX isn't installed. Install "
            "the extra (see docs/LAYOUTS.md) and set a HuggingFace token with "
            "[cyan]peaksift config set-secret huggingface_token[/cyan]."
        )

    for missing, extra in (
        (not deps.faster_whisper, "faster-whisper"),
        (not deps.mediapipe, "mediapipe"),
        (not deps.scenedetect, "scenedetect"),
    ):
        if missing:
            remediation.append(
                f"[cyan]{extra}[/cyan] is not installed — run the setup again "
                "(Setup-PeakSift.cmd or ./setup.sh)."
            )

    console.print()
    console.print(dep_table)

    # --- Providers ---------------------------------------------------------
    prov_table = Table(
        title="LLM providers", show_header=True, header_style="bold", title_justify="left"
    )
    prov_table.add_column("Provider")
    prov_table.add_column("Status", width=6)
    prov_table.add_column("Detail")

    for name in config.KEYED_PROVIDERS:
        has_key = config.get_secret(name, settings) is not None
        model = settings.provider(name).model or "no model set"
        active = " [cyan](active)[/cyan]" if settings.active_provider == name else ""
        prov_table.add_row(
            f"{name}{active}",
            _status(has_key, warn_only=True),
            f"{model}" if has_key else "no API key stored",
        )

    ollama_active = " [cyan](active)[/cyan]" if settings.active_provider == "ollama" else ""
    if deps.ollama_running:
        model_list = ", ".join(deps.ollama_models[:4]) or "no models pulled"
        prov_table.add_row(f"ollama{ollama_active}", OK, model_list)
    else:
        prov_table.add_row(
            f"ollama{ollama_active}", WARN, "not running on localhost:11434 (optional)"
        )

    if not any(config.get_secret(n, settings) for n in config.KEYED_PROVIDERS) and not (
        deps.ollama_running
    ):
        remediation.append(
            "No LLM provider is usable yet. Add an API key with "
            "[cyan]peaksift config set-secret anthropic[/cyan] (or openai / gemini), "
            "or install Ollama for a fully local setup."
        )

    console.print()
    console.print(prov_table)

    # --- Storage -----------------------------------------------------------
    store_table = Table(
        title="Storage", show_header=True, header_style="bold", title_justify="left"
    )
    store_table.add_column("Check")
    store_table.add_column("Status", width=6)
    store_table.add_column("Detail")

    from . import portable

    project_local = portable.PROJECT_ROOT is not None and os.environ.get(
        "PEAKSIFT_NO_KEYRING", ""
    ).lower() in ("1", "true", "yes")
    if project_local:
        store_table.add_row(
            "Secret storage", OK, f"project-local (in {paths.config_path()}, not the OS store)"
        )
    else:
        store_table.add_row(
            "Secret storage",
            _status(deps.keyring_backend, warn_only=True),
            "OS keyring" if deps.keyring_backend else "no keyring backend — plaintext fallback",
        )
    if not deps.keyring_backend and not project_local:
        remediation.append(
            "No OS keyring backend is available, so API keys would be written to "
            f"{paths.config_path()} in plaintext. On headless Linux install "
            "[cyan]keyrings.alt[/cyan] or a Secret Service provider."
        )
    if settings.insecure_secret_storage:
        store_table.add_row("Stored secrets", WARN, "one or more secrets are in plaintext")

    store_table.add_row("Home", OK, str(paths.root()))

    console.print()
    console.print(store_table)

    # --- Verdict -----------------------------------------------------------
    console.print()
    if report.ready:
        console.print(
            Panel.fit(
                "[bold green]Ready.[/bold green] The core pipeline can run on this machine.",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold red]Not ready.[/bold red] Resolve the items below, then re-run "
                "[cyan]peaksift doctor[/cyan].",
                border_style="red",
            )
        )

    if remediation:
        console.print()
        console.print("[bold]What to do:[/bold]")
        for i, item in enumerate(remediation, 1):
            console.print(f"  [bold]{i}.[/bold] {item}")
        console.print()

    raise typer.Exit(0 if report.ready else 1)


config_app = typer.Typer(help="Inspect and modify PeakSift settings.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Print current settings (secrets are never displayed)."""
    settings = config.load()
    console.print_json(settings.model_dump_json(indent=2))


@config_app.command("path")
def config_path_cmd() -> None:
    """Print the path to config.json."""
    console.print(str(paths.config_path()))


@config_app.command("set-secret")
def config_set_secret(
    key: str = typer.Argument(
        ...,
        help="Provider name (anthropic, openai, gemini) or 'huggingface_token'.",
    ),
) -> None:
    """Store an API key or token. The value is prompted for, never passed as an argument.

    Prompting rather than accepting a flag keeps the secret out of your shell
    history and out of the process list.
    """
    valid = (*config.KEYED_PROVIDERS, config.HF_TOKEN_KEY)
    if key not in valid:
        console.print(f"[red]Unknown secret '{key}'.[/red] Expected one of: {', '.join(valid)}")
        raise typer.Exit(2)

    value = typer.prompt(f"Value for {key}", hide_input=True).strip()
    if not value:
        console.print("[yellow]Empty value — nothing stored.[/yellow]")
        raise typer.Exit(1)

    paths.ensure_layout()
    secure = config.set_secret(key, value)
    if secure:
        console.print(f"[green]Stored {key} in the OS keyring.[/green]")
    else:
        console.print(
            f"[yellow]No keyring backend available — {key} was written to "
            f"{paths.config_path()} in plaintext.[/yellow]"
        )


@config_app.command("delete-secret")
def config_delete_secret(key: str = typer.Argument(..., help="Secret to remove.")) -> None:
    """Remove a stored secret."""
    config.delete_secret(key)
    console.print(f"[green]Removed {key}.[/green]")


@app.command("hindi-models")
def hindi_models() -> None:
    """List the speech models offered for Hindi / Hinglish audio."""
    from rich.table import Table

    from .speech_models import HINDI_MODELS

    current = config.load().whisper.hindi_model
    table = Table(show_lines=False)
    for column in ("key", "model", "writes", "download", "based on", "licence"):
        table.add_column(column)
    for m in HINDI_MODELS.values():
        mark = " [green](current)[/green]" if current in (m.key, m.repo) else ""
        writes = "Hinglish" if m.script == "roman" else "Devanagari → converter"
        table.add_row(m.key, m.label + mark, writes, f"{m.download_gb} GB", m.based_on, m.license)
    console.print(table)
    console.print(
        "Use one with [bold]--hindi-model KEY[/bold], or set whisper.hindi_model in "
        "config.json (any faster-whisper Hugging Face ID or local folder also works)."
    )


@app.command()
def init() -> None:
    """Create the PeakSift home directory and initialise the database."""
    from . import db

    version_applied = db.init()
    console.print(f"[green]Initialised[/green] {paths.root()} (schema v{version_applied})")


def _apply_detection_options(
    settings,
    *,
    view: str = "",
    per_hour: int = 0,
    profile: str = "",
    layout: str = "",
    mode: str = "",
    no_llm: bool = False,
    split: float = 0,
    facecam_bottom: bool = False,
    two_person: str = "",
    hindi: str = "",
    hindi_model: str = "",
    captions_from: str = "",
    no_auto_switch: bool = False,
) -> None:
    if hindi:
        settings.whisper.hindi_script = hindi  # type: ignore[assignment]
    if hindi_model:
        settings.whisper.hindi_model = hindi_model
    if captions_from:
        if captions_from not in ("whisper", "gemini"):
            raise typer.BadParameter("--captions-from must be whisper or gemini")
        settings.whisper.caption_source = captions_from  # type: ignore[assignment]
    if no_auto_switch:
        settings.export.auto_switch_layout = False
    if split:
        # Accept 35 or 0.35.
        settings.export.split_ratio = max(0.25, min(0.70, split / 100 if split > 1 else split))
    if facecam_bottom:
        settings.export.facecam_position = "bottom"
    if two_person:
        settings.export.two_person = two_person  # type: ignore[assignment]
    if per_hour:
        settings.selection.view = "per_hour"
        settings.selection.per_bucket = per_hour
    if view:
        settings.selection.view = view  # type: ignore[assignment]
    if profile:
        settings.detection.profile = profile  # type: ignore[assignment]
    if layout:
        settings.export.layout = layout  # type: ignore[assignment]
    if mode:
        settings.detection.mode = mode  # type: ignore[assignment]
    if no_llm:
        settings.detection.llm_rerank = "off"


@app.command()
def clip(
    target: str = typer.Argument(
        ...,
        help=(
            "A video link (YouTube, Twitch, Kick, TikTok, Facebook, Instagram, ...) "
            "or a path to a local media file. Live YouTube/Twitch/Kick links switch "
            "to live mode automatically."
        ),
    ),
    provider: str = typer.Option("", "--provider", "-p", help="Override the active provider."),
    model: str = typer.Option("", "--whisper-model", help="Override the Whisper model."),
    max_clips: int = typer.Option(0, "--max-clips", "-n", help="Override the clip count."),
    style: str = typer.Option("", "--style", "-s", help="Caption style preset."),
    ratio: str = typer.Option("", "--ratio", "-r", help="Output ratio: 9:16, 1:1, or 16:9."),
    diarize: bool = typer.Option(
        False, "--diarize/--no-diarize", help="Label speakers (needs the diarization extra)."
    ),
    centre_crop: bool = typer.Option(
        False, "--centre-crop", help="Skip face tracking and centre-crop everything."
    ),
    view: str = typer.Option(
        "", "--view", help="overall | per_hour | per_chapter (how final clips are chosen)."
    ),
    per_hour: int = typer.Option(
        0, "--per-hour", help="Best N moments of every hour (sets --view per_hour)."
    ),
    profile: str = typer.Option("", "--profile", help="auto | gaming | just_chatting | podcast"),
    layout: str = typer.Option(
        "",
        "--layout",
        help="auto | gaming (facecam over gameplay) | track | center | full (whole frame)",
    ),
    mode: str = typer.Option(
        "", "--mode", help="multimodal (signals, default) | transcript (LLM-only, original)"
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Rank on audio/video/chat signals only; no LLM calls."
    ),
    split: float = typer.Option(
        0, "--split", help="Gaming layout: facecam share of the height, 25-70 (default 35)."
    ),
    facecam_bottom: bool = typer.Option(
        False, "--facecam-bottom", help="Gaming layout: gameplay on top, facecam below."
    ),
    two_person: str = typer.Option(
        "", "--two-person", help="split (stack two people, default) | track (follow the speaker)"
    ),
    hindi: str = typer.Option(
        "", "--hindi", help="Hindi captions: hinglish (Roman script, default) | devanagari"
    ),
    hindi_model: str = typer.Option(
        "",
        "--hindi-model",
        help="Speech model for Hindi audio (see `peaksift hindi-models`), "
        "or any faster-whisper Hugging Face ID.",
    ),
    captions_from: str = typer.Option(
        "",
        "--captions-from",
        help="whisper (local, default) | gemini (re-transcribes only the exported clips; "
        "needs a Gemini key)",
    ),
    no_auto_switch: bool = typer.Option(
        False,
        "--no-auto-switch",
        help="Keep the facecam+gameplay stack even while the webcam is off.",
    ),
) -> None:
    """Turn a video (or a live stream) into captioned vertical clips."""
    import asyncio

    from . import db
    from .db import store
    from .db.models import Job, new_id
    from .pipeline import ingest, runner

    db.init()
    settings = config.load()

    if provider:
        settings.active_provider = provider  # type: ignore[assignment]
    if model:
        settings.whisper.model = model
    if max_clips:
        settings.clips.max_clips = max_clips
    if style:
        settings.export.caption_style = style
    if ratio:
        settings.export.ratio = ratio  # type: ignore[assignment]
    if diarize:
        settings.whisper.diarization = True
    if centre_crop:
        settings.export.layout = "center"
    _apply_detection_options(
        settings,
        view=view,
        per_hour=per_hour,
        profile=profile,
        layout=layout,
        mode=mode,
        no_llm=no_llm,
        split=split,
        facecam_bottom=facecam_bottom,
        two_person=two_person,
        hindi=hindi,
        hindi_model=hindi_model,
        captions_from=captions_from,
        no_auto_switch=no_auto_switch,
    )

    from . import sources

    # --- live links go to live mode ---------------------------------------
    if sources.is_url(target) and sources.supports_live(sources.platform_of(target)):
        try:
            with console.status("[cyan]Checking link...", spinner="dots"):
                link = sources.probe(
                    target, cookies_from_browser=settings.ingest.cookies_from_browser
                )
        except Exception:
            link = None
        if link is not None and link.is_live:
            console.print("[cyan]This stream is live — switching to live mode.[/cyan]")
            _run_live(target, settings, backfill=None, max_hours=None)
            return

    # --- ingest ---------------------------------------------------------
    try:
        status = "[cyan]Downloading source (and chat replay, if any)..."
        with console.status(status, spinner="dots"):
            if sources.is_url(target):
                source = ingest.ingest_url(target, settings.ingest)
            else:
                source = ingest.ingest_file(Path(target))
    except ingest.IngestError as exc:
        console.print(f"\n[red]Ingest failed.[/red] {exc}")
        raise typer.Exit(1) from exc

    store.create_source(source)
    console.print(
        f"[green]Source:[/green] {source.title} "
        f"({_format_duration(source.duration_s)}, {source.width}x{source.height})"
    )

    job = store.create_job(
        Job(
            id=new_id(),
            source_id=source.id,
            provider=settings.active_provider,
            settings=settings.model_dump(mode="json"),
        )
    )

    # --- run ------------------------------------------------------------
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting", total=1.0)

        def on_progress(event: runner.ProgressEvent) -> None:
            progress.update(task, completed=event.overall, description=event.message)

        try:
            clips = asyncio.run(
                runner.PipelineRunner(job, source, settings=settings, on_progress=on_progress).run()
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            raise typer.Exit(130) from None
        except Exception as exc:
            progress.stop()
            console.print(f"\n[red]Pipeline failed.[/red] {exc}")
            raise typer.Exit(1) from exc

    # --- report ---------------------------------------------------------
    _print_clips(clips)
    out_dir = paths.exports_dir() / job.id
    console.print(f"\n[green]Exported to[/green] {out_dir}")
    if (out_dir / "clips.md").exists():
        console.print(f"[green]Report:[/green] {out_dir / 'clips.md'}")


def _print_clips(clips) -> None:
    table = Table(title=f"{len(clips)} clips", header_style="bold", title_justify="left")
    table.add_column("#", width=3)
    table.add_column("Score", width=5)
    table.add_column("At", width=8)
    table.add_column("Length", width=6)
    table.add_column("Group", width=17)
    table.add_column("Title / why")

    for c in sorted(clips, key=lambda c: c.rank):
        at = _format_duration(c.peak_s if c.peak_s is not None else c.start_s)
        why = f"{c.title or '(untitled)'}\n[dim]{c.reason}[/dim]" if c.reason else (c.title or "")
        table.add_row(
            str(c.rank), str(c.score), at, f"{c.duration_s:.0f}s", c.group_label or "", why
        )

    console.print()
    console.print(table)


def _run_live(
    url: str,
    settings,
    *,
    backfill: bool | None,
    max_hours: float | None,
    chat_url: str | None = None,
    port: int = 1935,
) -> None:
    from . import db
    from .live.session import LiveError, LiveSession

    db.init()
    last = {"printed": 0}

    def on_status(status) -> None:
        for message in status.events[last["printed"] :]:
            console.print(f"[cyan]•[/cyan] {message}")
        last["printed"] = len(status.events)

    session = LiveSession(
        url,
        settings,
        backfill=backfill,
        max_hours=max_hours,
        on_status=on_status,
        chat_url=chat_url,
        obs_port=port,
    )
    console.print(
        "[dim]Keys: [bold]s[/bold] = clip the last 30 s, [bold]l[/bold] = last 60 s, "
        "[bold]q[/bold] or Ctrl+C = stop (clips so far are kept and ranked). "
        "From another terminal: peaksift snap 30[/dim]"
    )
    try:
        clips = session.run()
    except LiveError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    _print_clips(clips)


@app.command()
def live(
    url: str = typer.Argument(
        ...,
        help=(
            "A live YouTube, Twitch or Kick stream/channel URL — or 'obs' to receive "
            "your own stream straight from OBS on this PC."
        ),
    ),
    backfill: bool = typer.Option(
        True,
        "--backfill/--no-backfill",
        help="Also clip the part of the stream that happened before you joined.",
    ),
    max_hours: float = typer.Option(0, "--max-hours", help="Stop recording after this long."),
    per_hour: int = typer.Option(0, "--per-hour", help="Clips to keep per hour (default 4)."),
    profile: str = typer.Option("", "--profile", help="auto | gaming | just_chatting | podcast"),
    layout: str = typer.Option("", "--layout", help="auto | gaming | track | center | full"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Signals only; no LLM calls."),
    whisper_model: str = typer.Option("", "--whisper-model", help="Override the Whisper model."),
    ratio: str = typer.Option("", "--ratio", "-r", help="Output ratio: 9:16, 1:1, or 16:9."),
    chat: str = typer.Option(
        "",
        "--chat",
        help="OBS mode: your channel URL to read live chat from (Twitch/YouTube/Kick).",
    ),
    port: int = typer.Option(1935, "--port", help="OBS mode: local RTMP port."),
    split: float = typer.Option(0, "--split", help="Gaming layout: facecam share, 25-70."),
    facecam_bottom: bool = typer.Option(False, "--facecam-bottom", help="Facecam below gameplay."),
) -> None:
    """Clip a live stream as it happens (and backfill what you missed)."""
    settings = config.load()
    if per_hour:
        settings.live.clips_per_hour = per_hour
        settings.selection.per_bucket = per_hour
    if whisper_model:
        settings.whisper.model = whisper_model
    if ratio:
        settings.export.ratio = ratio  # type: ignore[assignment]
    _apply_detection_options(
        settings,
        profile=profile,
        layout=layout,
        no_llm=no_llm,
        split=split,
        facecam_bottom=facecam_bottom,
    )
    _run_live(
        url,
        settings,
        backfill=backfill,
        max_hours=max_hours or None,
        chat_url=chat or None,
        port=port,
    )


@app.command()
def snap(
    seconds: float = typer.Argument(30, help="How many seconds back to clip (5-180)."),
) -> None:
    """Clip the last N seconds of the live session running in another terminal."""
    from . import db
    from .live.session import request_snapshot_for_running_session

    db.init()
    trigger = request_snapshot_for_running_session(seconds)
    if trigger is None:
        console.print("[yellow]No live session is running.[/yellow]")
        raise typer.Exit(1)
    console.print(f"[green]Snapshot requested[/green] (last {seconds:.0f}s).")


def _resolve_job_id(job_id: str) -> str:
    from .db import store

    if job_id in ("latest", "last"):
        jobs = store.list_jobs(limit=1)
        if not jobs:
            console.print("[red]There are no jobs yet.[/red]")
            raise typer.Exit(1)
        return jobs[0].id
    if store.get_job(job_id) is None:
        console.print(f"[red]Job {job_id} not found.[/red] See `peaksift jobs`.")
        raise typer.Exit(1)
    return job_id


@app.command()
def layout(
    job_id: str = typer.Argument("latest", help="Job to pick the layout on ('latest' by default)."),
    port: int = typer.Option(8765, "--port", help="Local port for the picker page."),
    rerender_after: bool = typer.Option(
        True, "--rerender/--no-rerender", help="Re-render this job's clips after you save."
    ),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open the browser."),
) -> None:
    """Draw the facecam and gameplay boxes on a frame and set the split (opens a browser)."""
    import os
    import threading
    import time
    import webbrowser

    import uvicorn

    from . import db
    from .api import layouts as layout_api

    db.init()
    job_id = _resolve_job_id(job_id)
    os.environ["PEAKSIFT_NO_WORKER"] = "1"
    url = f"http://127.0.0.1:{port}/layout-picker?job={job_id}"

    from .app import create_app

    server = uvicorn.Server(
        uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.0)
    console.print(f"[green]Layout picker:[/green] [cyan]{url}[/cyan]")
    console.print("[dim]Draw the boxes, press Save, then come back here. Ctrl+C to cancel.[/dim]")
    if open_browser:
        webbrowser.open(url)

    layout_api.saved_event.clear()
    try:
        while not layout_api.saved_event.wait(0.5):
            if not thread.is_alive():
                console.print("[red]The picker server stopped unexpectedly.[/red]")
                raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled; nothing changed.[/yellow]")
        server.should_exit = True
        raise typer.Exit(130) from None
    time.sleep(0.5)  # let the browser receive the save response
    server.should_exit = True
    thread.join(timeout=5)
    console.print("[green]Layout saved.[/green]")
    if rerender_after:
        _rerender(job_id, ranks=None)


@app.command("layouts")
def layouts_cmd(
    delete: str = typer.Option("", "--delete", help="Delete the preset with this key."),
) -> None:
    """List saved facecam/gameplay presets (per channel)."""
    from . import layouts as layout_store

    if delete:
        ok = layout_store.delete_preset(delete)
        console.print(
            f"[green]Deleted {delete}.[/green]" if ok else f"[yellow]No preset {delete}.[/yellow]"
        )
        return
    presets = layout_store.list_presets()
    if not presets:
        console.print("No presets yet. Create one with `peaksift layout <job>`.")
        return
    table = Table(title="Layout presets", header_style="bold", title_justify="left")
    for column in ("Channel", "Facecam (x,y,w,h)", "Gameplay", "Split", "Facecam at"):
        table.add_column(column)
    for key, item in presets.items():
        fmt = lambda b: ", ".join(f"{v:.2f}" for v in b) if b else "auto (centre)"  # noqa: E731
        table.add_row(
            key, fmt(item.facecam), fmt(item.gameplay), f"{item.split:.0%}", item.facecam_position
        )
    console.print(table)


def _rerender(job_id: str, ranks: list[int] | None) -> None:
    from .rerender import RerenderError, rerender_job

    settings = config.load()

    def progress(index: int, total: int, clip) -> None:
        console.print(f"  [{index}/{total}] {clip.title or clip.id}")

    console.print("[cyan]Re-rendering clips with the new layout...[/cyan]")
    try:
        files = rerender_job(job_id, settings, ranks=ranks, on_progress=progress)
    except RerenderError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Re-rendered {len(files)} clip(s)[/green] in {paths.exports_dir() / job_id}"
    )


@app.command()
def rerender(
    job_id: str = typer.Argument("latest", help="Job whose clips to re-render."),
    clip: list[int] = typer.Option([], "--clip", "-c", help="Only these clip ranks (repeatable)."),
    split: float = typer.Option(0, "--split", help="Override the facecam share, 25-70."),
    facecam_bottom: bool = typer.Option(False, "--facecam-bottom", help="Facecam below gameplay."),
    layout_opt: str = typer.Option("", "--layout", help="auto | gaming | track | center | full"),
    two_person: str = typer.Option("", "--two-person", help="split | track"),
) -> None:
    """Re-render a job's clips (e.g. after changing the layout) without re-detecting."""
    from . import db

    db.init()
    job_id = _resolve_job_id(job_id)
    if split or facecam_bottom or layout_opt or two_person:
        settings = config.load()
        _apply_detection_options(
            settings,
            layout=layout_opt,
            split=split,
            facecam_bottom=facecam_bottom,
            two_person=two_person,
        )
        config.save(settings)
        console.print("[dim]Saved these layout options as your defaults.[/dim]")
    _rerender(job_id, ranks=list(clip) or None)


@app.command()
def jobs(limit: int = typer.Option(15, "--limit", "-n")) -> None:
    """List recent jobs."""
    from . import db
    from .db import store

    db.init()
    table = Table(title="Recent jobs", header_style="bold", title_justify="left")
    table.add_column("ID", width=16)
    table.add_column("Status", width=10)
    table.add_column("Stage", width=12)
    table.add_column("Progress", width=8)
    table.add_column("Source")

    for job in store.list_jobs(limit=limit):
        source = store.get_source(job.source_id)
        colour = {
            "done": "green",
            "failed": "red",
            "running": "cyan",
            "cancelled": "yellow",
        }.get(job.status, "white")
        table.add_row(
            job.id,
            f"[{colour}]{job.status}[/{colour}]",
            job.current_stage,
            f"{job.progress * 100:.0f}%",
            (source.title if source else "?")[:50],
        )

    console.print(table)


@app.command()
def providers() -> None:
    """Check which LLM providers are reachable right now."""
    import asyncio

    from .providers import PROVIDERS, build_provider

    settings = config.load()

    async def check_all():
        results = []
        for name in PROVIDERS:
            try:
                provider = build_provider(name, settings)
                results.append(await provider.health_check())
            except Exception as exc:
                from .providers import ProviderStatus

                results.append(ProviderStatus(name=name, available=False, detail=str(exc)[:120]))
        return results

    with console.status("[cyan]Checking providers...", spinner="dots"):
        statuses = asyncio.run(check_all())

    table = Table(title="LLM providers", header_style="bold", title_justify="left")
    table.add_column("Provider", width=12)
    table.add_column("Status", width=6)
    table.add_column("Detail")

    for status in statuses:
        active = " [cyan]*[/cyan]" if settings.active_provider == status.name else ""
        table.add_row(
            f"{status.name}{active}", _status(status.available, warn_only=True), status.detail
        )

    console.print(table)
    if any(s.available for s in statuses):
        console.print("\n[dim]* = active provider[/dim]")


@app.command("fetch-models")
def fetch_models() -> None:
    """Download the ML model bundles the reframe stage needs."""
    from . import models

    for key, spec in models.MODELS.items():
        if models.is_available(key):
            console.print(f"[green]OK[/green] {spec.filename} already downloaded")
            continue
        with console.status(f"[cyan]Downloading {spec.filename}...", spinner="dots"):
            try:
                path = models.ensure(key)
            except models.ModelDownloadError as exc:
                console.print(f"[red]FAILED[/red] {spec.filename}\n{exc}")
                raise typer.Exit(1) from exc
        console.print(f"[green]OK[/green] {path}")


@app.command("update-ytdlp")
def update_ytdlp() -> None:
    """Update yt-dlp, which YouTube changes force often."""
    import subprocess
    import sys

    console.print("[cyan]Updating yt-dlp...[/cyan]")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        console.print(f"[red]Update failed.[/red]\n{result.stderr}")
        raise typer.Exit(1)

    import importlib.metadata

    try:
        version_installed = importlib.metadata.version("yt-dlp")
        console.print(f"[green]yt-dlp is now at {version_installed}.[/green]")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        console.print("[green]Update complete.[/green]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open a browser once the server is up."
    ),
) -> None:
    """Start the PeakSift web app."""
    import threading
    import webbrowser

    import uvicorn

    from . import db

    db.init()

    # The web app (peaksift/webui) is plain HTML/JS: no build step, no Node.
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    console.print(f"[green]PeakSift[/green] starting on [cyan]{url}[/cyan]")

    if host == "0.0.0.0":  # noqa: S104
        console.print(
            "[yellow]Binding to 0.0.0.0 exposes PeakSift to your whole network.[/yellow] "
            "There is no authentication — only do this on a network you trust."
        )

    if open_browser and not reload:
        # Delayed so the browser doesn't race the server's first bind.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "peaksift.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def styles() -> None:
    """List the available caption styles."""
    from .pipeline.captions import PRESETS

    table = Table(title="Caption styles", header_style="bold", title_justify="left")
    table.add_column("Key", width=14)
    table.add_column("Name", width=14)
    table.add_column("Description")

    for style in PRESETS.values():
        table.add_row(style.key, style.label, style.description)

    console.print(table)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


if __name__ == "__main__":  # pragma: no cover
    app()

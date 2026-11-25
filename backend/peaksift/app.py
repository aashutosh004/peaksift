"""FastAPI application — REST API plus the web app, on one port.

The web app is plain HTML/JS/CSS served from the same origin as the API, which
keeps PeakSift a single process with no CORS configuration, no second port, and
no Node.js or build step.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, db, paths
from .api import api_router
from .jobs.events import broker
from .jobs.queue import queue

log = logging.getLogger(__name__)

#: Set to "1" to serve the API without starting the background job worker.
ENV_NO_WORKER = "PEAKSIFT_NO_WORKER"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    import asyncio

    paths.ensure_layout()
    db.init()

    broker.bind_loop(asyncio.get_running_loop())

    # Serving the API without a worker is useful for debugging a stuck queue and
    # makes API tests deterministic — jobs stay queued instead of racing off.
    worker_enabled = os.environ.get(ENV_NO_WORKER) != "1"
    if worker_enabled:
        await queue.start()
    else:
        log.info("Job worker disabled by %s=1.", ENV_NO_WORKER)

    log.info("PeakSift %s ready. Artifacts in %s", __version__, paths.root())

    try:
        yield
    finally:
        if worker_enabled:
            await queue.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PeakSift",
        version=__version__,
        description="Local-first AI video clipper.",
        lifespan=lifespan,
    )

    app.include_router(api_router)

    @app.get("/api/health")
    async def health() -> dict:
        status = queue.status()
        return {
            "status": "ok",
            "version": __version__,
            "running_job": status.running_job_id,
            "queued": status.queued,
        }

    _mount_frontend(app)
    return app


#: The built-in web UI: plain HTML/JS/CSS served as-is, no Node or build step.
WEBUI_DIR = Path(__file__).resolve().parent / "webui"


class _RevalidatedFiles(StaticFiles):
    """Static files the browser must re-check before reuse.

    The web UI has no build step, so its file names never change between
    versions; without this, browsers keep running an old app.js after an
    update. Unchanged files still come back as a cheap 304.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _mount_frontend(app: FastAPI) -> None:
    app.mount("/ui", _RevalidatedFiles(directory=WEBUI_DIR), name="webui")

    @app.get("/", include_in_schema=False, response_model=None)
    async def webui_index() -> FileResponse | JSONResponse:
        index = WEBUI_DIR / "index.html"
        if index.exists():
            return FileResponse(
                index, media_type="text/html", headers={"Cache-Control": "no-cache"}
            )
        return JSONResponse(
            status_code=503,
            content={"error": "No web UI found.", "hint": "The API is available at /docs."},
        )


app = create_app()

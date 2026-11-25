"""REST API routers."""

from __future__ import annotations

from fastapi import APIRouter

from . import captions, clips, jobs, layouts, settings, sources, ui

__all__ = ["api_router"]

api_router = APIRouter()
api_router.include_router(sources.router)
api_router.include_router(jobs.router)
api_router.include_router(clips.router)
api_router.include_router(settings.router)
api_router.include_router(layouts.router)
api_router.include_router(ui.router)
api_router.include_router(captions.router)

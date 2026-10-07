"""Completion routes attached to Gradio's existing authenticated application."""

import inspect
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .catalog import Catalog
from .config import FIELDS, SettingsStore
from .frequency import FrequencyStore


@lru_cache(maxsize=1)
def services():
    import modules.config as config
    from modules.sdxl_styles import legal_style_names

    settings = SettingsStore(
        Path(config.config_path).resolve().parent / "tagcomplete_settings.json"
    )
    catalog = Catalog(
        config.path_tagcomplete,
        config.paths_loras,
        config.path_embeddings,
        config.path_wildcards,
        legal_style_names,
    )
    frequency = FrequencyStore(
        Path(config.path_tagcomplete_cache) / "frequency.sqlite3"
    )
    return settings, catalog, frequency


async def current_user(request: Request):
    app = request.app
    dependency = getattr(app, "auth_dependency", None)
    user = None
    if dependency:
        user = dependency(request)
        if inspect.isawaitable(user):
            user = await user
    elif getattr(app, "auth", None) is not None:
        cookie_id = app.cookie_id
        token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(
            f"access-token-unsecure-{cookie_id}"
        )
        user = app.tokens.get(token)
    else:
        return "local"
    if user is None:
        raise HTTPException(401, "Not authenticated")
    return user


class UsageEvent(BaseModel):
    name: str = Field(min_length=1, max_length=4096)
    kind: str = Field(
        pattern="^(tag|extra|lora|embedding|wildcard|wildcard_value|chant|style)$"
    )
    negative: bool = False


def create_router(store, catalog, frequency):
    router = APIRouter(prefix="/tagcomplete/v1", dependencies=[Depends(current_user)])

    @router.get("/bootstrap")
    def bootstrap():
        return {
            "settings": store.load(),
            "catalog": catalog.get(),
            "schema": [
                {"key": key, "kind": kind, "bounds": bounds}
                for key, _, _, kind, _, bounds in FIELDS
            ],
        }

    @router.get("/catalog")
    def get_catalog():
        return catalog.get()

    @router.post("/refresh")
    async def refresh():
        return await run_in_threadpool(catalog.refresh)

    @router.get("/asset/{identifier}")
    def asset(identifier: str):
        try:
            path = catalog.asset(identifier, {"dataset", "preview", "wildcard"})
        except FileNotFoundError:
            raise HTTPException(404, "Completion asset not found") from None
        return FileResponse(path, headers={"Cache-Control": "private, no-cache"})

    @router.get("/usage")
    def usage(user=Depends(current_user)):
        return frequency.get(user)

    @router.post("/usage")
    def increase(event: UsageEvent, user=Depends(current_user)):
        frequency.increase(user, event.name, event.kind, event.negative)
        return {"saved": True}

    @router.delete("/usage")
    def clear(user=Depends(current_user)):
        frequency.clear(user)
        return {"cleared": True}

    return router


@asynccontextmanager
async def lifespan(app):
    app.include_router(create_router(*services()))
    from modules.prompt_all_in_one.api import create_router as editor_router, services as editor_services
    app.include_router(editor_router(*editor_services()))
    yield


def model_capabilities(base_name, refiner_name):
    from safetensors import SafetensorError

    import modules.config as config
    from modules.anima import is_anima_file
    from modules.util import get_file_from_folder_list

    def is_anima(name):
        if not name or name == "None":
            return False
        try:
            return is_anima_file(
                get_file_from_folder_list(name, config.paths_checkpoints)
            )
        except (OSError, ValueError, SafetensorError):
            return False

    base_anima, refiner_anima = is_anima(base_name), is_anima(refiner_name)
    return {
        "base": base_name,
        "refiner": refiner_name,
        "base_anima": base_anima,
        "refiner_anima": refiner_anima,
        "embeddings": not (base_anima or refiner_anima),
    }

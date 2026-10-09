"""Authenticated native editor API; no A1111 imports or package-install routes."""

import csv
import io
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, get_args

import httpx
import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modules.tagcomplete.api import current_user

from .config import SettingsStore
from .providers import ProviderError, completion, translate, validate_endpoint
from .storage import Store

User = Annotated[str, Depends(current_user)]
Scope = str
Provider = Literal[
    "dictionary",
    "openai",
    "deepl",
    "libretranslate",
    "microsoft",
    "mbart50",
    "google",
    "baidu",
    "amazon",
    "alibaba",
    "yandex",
    "youdao",
    "tencent",
    "mymemory",
    "niutrans",
    "caiyun",
    "volcengine",
    "iflytekV1",
    "iflytekV2",
]
SCOPE_PATTERN = r"^(positive|negative|inpaint|enhance_(negative_)?[0-9]{1,3})$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tag(StrictModel):
    raw: str = Field(max_length=100000)
    separator: str = Field(default="", max_length=10000, pattern=r"^[,\s]*$")
    disabled: bool = False


class Entry(StrictModel):
    scope: Scope = Field(pattern=SCOPE_PATTERN)
    prompt: str = Field(max_length=100000)
    tags: list[Tag] = Field(default_factory=list, max_length=3000)
    name: str = Field(default="", max_length=200)
    favorite: bool = False
    limit: int = Field(default=100, ge=10, le=1000)

    @model_validator(mode="after")
    def bound_snapshot(self):
        if sum(len(tag.raw) + len(tag.separator) for tag in self.tags) > 100000:
            raise ValueError("Editor snapshot exceeds 100000 characters")
        return self


class UpdateEntry(StrictModel):
    name: str | None = Field(default=None, max_length=200)
    position: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)


class ProviderConfig(StrictModel):
    endpoint: str = Field(default="", max_length=2048)
    key: str = Field(default="", max_length=4096)
    model: str = Field(default="", max_length=200)
    region: str = Field(default="", max_length=200)
    model_path: str = Field(default="", max_length=2048)
    clear_key: bool = False
    options: dict[str, str] = Field(default_factory=dict, max_length=10)


class Translation(StrictModel):
    provider: Provider = "dictionary"
    texts: list[str] = Field(max_length=100, min_length=1)
    source: str = Field(
        default="auto", pattern=r"^(auto|zh|en|ja|ko|de|fr|es|it|pt|ru)$"
    )
    target: str = Field(default="zh", pattern=r"^(zh|en|ja|ko|de|fr|es|it|pt|ru)$")
    language: str = Field(default="zh_CN", pattern=r"^[a-z]{2}_[A-Z]{2}$")


class Generation(StrictModel):
    text: str = Field(max_length=20000, min_length=1)


class TokenRequest(StrictModel):
    text: str = Field(max_length=100000)
    base: str = Field(default="", max_length=4096)
    refiner: str = Field(default="None", max_length=4096)


@lru_cache(maxsize=1)
def services():
    from modules import config

    directory = (
        Path(config.config_path).resolve().parent / "userdata" / "prompt_all_in_one"
    )
    return SettingsStore(directory / "settings.json"), Store(
        directory / "editor.sqlite3"
    )


def private_paths():
    _, store = services()
    return [str(store.path.parent.parent)]


@lru_cache(maxsize=12)
def groups(language):
    if language not in DEFAULTS_LANGUAGES:
        language = "default"
    file = Path(__file__).parent / "group_tags" / (language + ".yaml")
    if not file.is_file():
        file = file.with_name("default.yaml")
    # BaseLoader preserves tag names such as "on"/"off" as strings; YAML's
    # implicit boolean conversion would otherwise rename them. Upstream also
    # includes layout-only {type: wrap} nodes which are not tag groups.
    content = yaml.load(file.read_text(encoding="utf-8"), Loader=yaml.BaseLoader) or []
    return [
        {
            "name": category["name"],
            "groups": [
                group
                for group in category.get("groups", [])
                if "name" in group and isinstance(group.get("tags"), dict)
            ],
        }
        for category in content
        if isinstance(category, dict) and "name" in category
    ]


DEFAULTS_LANGUAGES = {
    "zh_CN",
    "zh_TW",
    "ja_JP",
    "ko_KR",
    "de_DE",
    "fr_FR",
    "es_ES",
    "it_IT",
    "pt_PT",
    "ru_RU",
    "zh_HK",
}


def dictionary(language):
    result = {}
    for category in groups(language):
        for group in category["groups"]:
            result.update(
                {
                    key.replace("_", " ").casefold(): label
                    for key, label in group["tags"].items()
                    if label
                }
            )
    # Share CSV translations selected in Tag Autocomplete's server defaults.
    from modules.tagcomplete.api import services as completion_services

    settings, catalog, _ = completion_services()
    configuration = settings.load()
    name = configuration["translation_file"]
    entry = next(
        (entry for entry in catalog.get()["datasets"] if entry["name"] == name), None
    )
    if entry:
        file = catalog.asset(entry["id"], {"dataset"})
        for row in csv.reader(io.StringIO(file.read_text(encoding="utf-8-sig"))):
            index = 2 if configuration["translation_old_format"] else 1
            if len(row) > index:
                result[row[0].replace("_", " ").casefold()] = row[index]
    return result


def create_router(settings, store):
    router = APIRouter(
        prefix="/prompt-all-in-one/v1", dependencies=[Depends(current_user)]
    )

    @router.get("/bootstrap")
    def bootstrap(user: User):
        from .providers.base import METADATA

        return {
            "provider_fields": {key: item["fields"] for key, item in METADATA.items()},
            "settings": settings.load(),
            "user": user,
            "providers": list(get_args(Provider)),
        }

    @router.get("/groups")
    def get_groups(language: str = "zh_CN"):
        return groups(language)

    @router.get("/entries")
    def get_entries(scope: str, user: User, favorite: bool = False):
        return store.list(user, scope, favorite)

    @router.post("/entries")
    def add_entry(entry: Entry, user: User):
        return store.add(
            user,
            entry.scope,
            entry.prompt,
            [tag.model_dump() for tag in entry.tags],
            entry.name,
            entry.favorite,
            entry.limit,
        )

    @router.patch("/entries/{identifier}")
    def update_entry(identifier: str, entry: UpdateEntry, user: User):
        if not store.update(user, identifier, entry.name, entry.position):
            raise HTTPException(404, "Entry not found")
        return {"saved": True}

    @router.delete("/entries")
    def delete_entries(
        scope: str,
        user: User,
        favorite: bool = False,
        identifier: str | None = None,
    ):
        store.delete(user, scope, favorite, identifier)
        return {"deleted": True}

    @router.get("/providers/{provider}")
    def get_provider(provider: Provider, user: User):
        return store.public_provider(user, provider)

    @router.put("/providers/{provider}")
    def set_provider(provider: Provider, values: ProviderConfig, user: User):
        from .providers.base import METADATA

        fields = {item["key"] for item in METADATA.get(provider, {}).get("fields", [])}
        if set(values.options) - fields or any(
            len(value) > 4096 for value in values.options.values()
        ):
            raise HTTPException(422, "Invalid provider configuration fields")
        if values.endpoint:
            try:
                validate_endpoint(values.endpoint)
            except ProviderError as error:
                raise HTTPException(422, str(error)) from None
        return store.save_provider(user, provider, values.model_dump())

    @router.post("/translate")
    async def translate_texts(values: Translation, user: User):
        if sum(len(text) for text in values.texts) > 20000:
            raise HTTPException(422, "Translation text exceeds 20000 characters")
        if values.provider == "dictionary":
            from starlette.concurrency import run_in_threadpool

            locales = {
                "zh": "zh_CN",
                "en": "en_US",
                "ja": "ja_JP",
                "ko": "ko_KR",
                "de": "de_DE",
                "fr": "fr_FR",
                "es": "es_ES",
                "it": "it_IT",
                "pt": "pt_PT",
                "ru": "ru_RU",
            }
            language = (
                values.language
                if values.target == "en"
                or values.language.startswith(values.target + "_")
                else locales[values.target]
            )
            if language == "en_US":
                language = locales.get(values.source, "zh_CN")
            mapping = await run_in_threadpool(dictionary, language)
            if values.target == "en":
                mapping = {value.casefold(): key for key, value in mapping.items()}
            result = [
                mapping.get(text.strip().replace("_", " ").casefold(), text)
                for text in values.texts
            ]
        else:
            try:
                result = await translate(
                    values.provider,
                    store.provider(user, values.provider),
                    values.texts,
                    values.source,
                    values.target,
                )
            except Exception:  # noqa: BLE001 - optional SDK failures may contain secrets.
                # Provider exceptions can contain keys/URLs/request bodies. Never echo them.
                raise HTTPException(
                    502,
                    "Translation failed; check provider configuration and server connectivity",
                ) from None
        if (
            not isinstance(result, list)
            or len(result) != len(values.texts)
            or any(not isinstance(text, str) or len(text) > 100000 for text in result)
        ):
            raise HTTPException(502, "Invalid translation response")
        return {"texts": result}

    @router.post("/generate")
    async def generate_prompt(values: Generation, user: User):
        try:
            prompt = await completion(
                store.provider(user, "openai"),
                "Write an English image generation prompt from the user's description. Return only the prompt. Preserve specified details. Do not invent model, LoRA or embedding filenames.",
                values.text,
            )
        except (
            ProviderError,
            ImportError,
            ValueError,
            KeyError,
            TypeError,
            httpx.HTTPError,
        ):
            raise HTTPException(
                502,
                "Prompt generation failed; check the OpenAI-compatible provider settings",
            ) from None
        return {"prompt": prompt}

    @router.post("/tokens")
    def tokens(values: TokenRequest):
        from modules.tagcomplete.api import model_capabilities

        from .tokens import count

        caps = model_capabilities(values.base, values.refiner)
        result = {"base": count(values.text, caps["base_anima"])}
        if values.refiner not in ("", "None"):
            result["refiner"] = count(values.text, caps["refiner_anima"])
        return result

    return router

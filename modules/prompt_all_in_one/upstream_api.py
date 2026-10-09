"""Host adapter for the copied Prompt All-in-One UI and operation contracts."""

import asyncio
import importlib.util
import json
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from modules.localization import translate as translate_ui
from modules.tagcomplete.api import current_user
from modules.tagcomplete.config import DEFAULT_TRANSLATION_FILE

from .api import User, dictionary
from .history import History
from .providers import completion, translate, validate_endpoint
from .providers.base import METADATA

ROOT = Path(__file__).resolve().parents[2]
STYLES = ROOT / "frontend/prompt_all_in_one/styles"
LANGUAGES = ROOT / "frontend/prompt_all_in_one/src/i18n.json"
SCOPE = r"^(positive|negative|inpaint|enhance_(negative_)?[0-9]{1,3})$"
KEY = r"^[a-zA-Z0-9_.:-]{1,200}$"
ALIASES = {"myMemory": "mymemory", "myMemory_free": "mymemory"}
_free_lock = threading.Lock()
OPTION_KEYS = {
    "languageCode": "language",
    "translateApi": "provider",
    "autoTranslateToEnglish": "auto_translate",
}


@lru_cache(maxsize=1)
def translation_metadata():
    return json.loads(
        Path(__file__).with_name("translate_apis.json").read_text(encoding="utf-8")
    )


def localized_metadata(value):
    """Translate host labels without modifying IDs, provider fields or values."""
    if isinstance(value, list):
        return [localized_metadata(item) for item in value]
    if isinstance(value, dict):
        return {
            key: translate_ui(item)
            if key in {"name", "title"} and isinstance(item, str)
            else localized_metadata(item)
            for key, item in value.items()
        }
    return value


def provider_item(name):
    for group in translation_metadata()["apis"]:
        for item in group["children"]:
            if item["key"] == name:
                return item
    raise HTTPException(422, "Unknown translation provider")


def credential_key(key):
    return key == "chatgpt_key" or key.startswith("translate_api.")


def provider_name(key):
    return "openai" if key == "chatgpt_key" else key.split(".", 1)[1]


def legacy_config(store, user, name):
    values = store.provider(user, ALIASES.get(name, name))
    if not values:
        return {}
    return {
        **values.get("options", {}),
        "api_key": values.get("key", ""),
        "api_base": values.get("endpoint", ""),
        "endpoint": values.get("endpoint", ""),
        "model": values.get("model", ""),
        "region": values.get("region", ""),
        "model_path": values.get("model_path", ""),
    }


def public_value(key, value):
    if not credential_key(key) or not isinstance(value, dict):
        return value
    fields = provider_item(provider_name(key)).get("config", [])
    result = dict(value)
    for field in fields:
        if field.get("secret"):
            result.pop(field["key"], None)
    return result


def native_provider_config(values):
    return {
        "key": values.get("api_key", ""),
        "endpoint": values.get("api_base") or values.get("endpoint", ""),
        "model": values.get("model", ""),
        "region": values.get("region", ""),
        "model_path": values.get("model_path", ""),
        "options": values,
    }


class DataValue(BaseModel):
    key: str = Field(pattern=KEY)
    data: object = None


class HistoryAction(BaseModel):
    type: str = Field(pattern=SCOPE)
    id: str = Field(default="", max_length=200)
    tags: list[dict] = Field(default_factory=list, max_length=3000)
    prompt: str = Field(default="", max_length=100000)
    name: str = Field(default="", max_length=200)
    limit: int = Field(default=100, ge=10, le=1000)
    record: bool = True


class ListAction(BaseModel):
    key: str = Field(pattern=KEY)
    item: object = None
    index: int = Field(default=0, ge=0)


class TranslationAction(BaseModel):
    text: str = Field(default="", max_length=20000)
    texts: list[str] = Field(default_factory=list, max_length=100)
    from_lang: str = Field(default="en_US", max_length=32)
    to_lang: str = Field(default="zh_CN", max_length=32)
    api: str = Field(default="dictionary", max_length=50)
    api_config: dict = Field(default_factory=dict)


def create_router(settings, store):
    router = APIRouter(
        prefix="/prompt-all-in-one/upstream", dependencies=[Depends(current_user)]
    )

    def read_value(storage, user, key):
        if not re.fullmatch(KEY, key):
            raise HTTPException(422, "Invalid storage key")
        value = storage.get(key)
        if value is None and credential_key(key):
            value = legacy_config(store, user, provider_name(key))
        if value is None and key in OPTION_KEYS:
            value = settings.load()[OPTION_KEYS[key]]
            if key == "translateApi" and value == "mymemory":
                value = "myMemory"
        if value is None and key == "tagCompleteFile":
            from modules.tagcomplete.api import services as completion_services

            _, catalog, _ = completion_services()
            value = next(
                (
                    entry["id"]
                    for entry in catalog.get()["datasets"]
                    if entry["name"] == DEFAULT_TRANSLATION_FILE
                ),
                None,
            )
        return value

    def save_value(storage, user, key, value):
        if not re.fullmatch(KEY, key) or key.startswith(("history.", "favorite.")):
            raise HTTPException(422, "Invalid preference key")
        if len(json.dumps(value)) > 100000:
            raise HTTPException(422, "Preference exceeds 100000 characters")
        if credential_key(key):
            if not isinstance(value, dict):
                raise HTTPException(422, "Provider settings must be an object")
            item = provider_item(provider_name(key))
            fields = {field["key"]: field for field in item.get("config", [])}
            if set(value) - fields.keys() - {"clear_key"}:
                raise HTTPException(422, "Invalid provider field")
            previous = read_value(storage, user, key) or {}
            for name, field in fields.items():
                if field.get("secret"):
                    value[name] = (
                        ""
                        if value.get("clear_key")
                        else value.get(name) or previous.get(name, "")
                    )
            value.pop("clear_key", None)
            for field in ("api_base", "endpoint", "host"):
                if value.get(field):
                    try:
                        validate_endpoint(value[field])
                    except ValueError:
                        raise HTTPException(422, "Invalid provider endpoint") from None
        storage.set(key, value)

    @router.get("/get_config")
    def config():
        return {
            "i18n": json.loads(LANGUAGES.read_text(encoding="utf-8")),
            "translate_apis": localized_metadata(translation_metadata()),
            "python": sys.executable,
        }

    @router.get("/get_data")
    def get_data(user: User, key: str = Query(pattern=KEY)):
        with store.upstream(user) as storage:
            return {"data": public_value(key, read_value(storage, user, key))}

    @router.get("/get_datas")
    def get_datas(user: User, keys: str = Query(max_length=20000)):
        with store.upstream(user) as storage:
            return {
                "datas": {
                    key: public_value(key, read_value(storage, user, key))
                    for key in keys.split(",")
                    if key
                }
            }

    @router.post("/set_data")
    def set_data(values: DataValue, user: User):
        with store.upstream(user) as storage:
            save_value(storage, user, values.key, values.data)
        return {"success": True}

    @router.post("/set_datas")
    def set_datas(user: User, values: Annotated[dict, Body()]):
        if len(values) > 100:
            raise HTTPException(422, "Too many preferences")
        with store.upstream(user) as storage:
            for key, value in values.items():
                save_value(storage, user, key, value)
        return {"success": True}

    def read_list(storage, key):
        if credential_key(key) or key.startswith(("history.", "favorite.")):
            raise HTTPException(422, "Invalid list key")
        data = storage.get(key) or []
        if not isinstance(data, list):
            raise HTTPException(422, "Preference is not a list")
        return data

    @router.get("/get_data_list_item")
    def list_item(user: User, key: str = Query(pattern=KEY), index: int = Query(ge=0)):
        with store.upstream(user) as storage:
            data = read_list(storage, key)
            return {"item": data[index] if index < len(data) else None}

    def list_route(operation):
        def mutate(values: ListAction, user: User):
            with store.upstream(user) as storage:
                data = read_list(storage, values.key)
                item = None
                if operation == "push":
                    if len(data) >= 3000:
                        raise HTTPException(422, "Too many list entries")
                    data.append(values.item)
                elif operation == "clear":
                    data = []
                elif data:
                    index = (
                        len(data) - 1
                        if operation == "pop"
                        else values.index
                        if operation == "remove"
                        else 0
                    )
                    if index >= len(data):
                        raise HTTPException(422, "List index out of range")
                    item = data.pop(index)
                save_value(storage, user, values.key, data)
                return {"success": True, "item": item}

        return mutate

    for operation in ("push", "pop", "shift", "remove", "clear"):
        router.add_api_route(
            "/" + operation + "_data_list", list_route(operation), methods=["POST"]
        )

    @router.get("/get_histories")
    def histories(user: User, type: str = Query(pattern=SCOPE)):
        with store.upstream(user) as storage:
            return {"histories": History(storage, [type]).get_histories(type)}

    @router.get("/get_favorites")
    def favorites(user: User, type: str = Query(pattern=SCOPE)):
        with store.upstream(user) as storage:
            return {"favorites": History(storage, [type]).get_favorites(type)}

    @router.get("/get_latest_history")
    def latest_history(user: User, type: str = Query(pattern=SCOPE)):
        with store.upstream(user) as storage:
            return {"history": History(storage, [type]).get_latest_history(type)}

    def history_route(method):
        def operation(values: HistoryAction, user: User):
            if len(json.dumps(values.tags)) > 1000000:
                raise HTTPException(422, "History snapshot too large")
            with store.upstream(user) as storage:
                history = History(storage, [values.type], values.limit)
                function = getattr(history, method)
                if method in ("push_history", "push_favorite"):
                    result = (
                        function(values.type, values.tags, values.prompt, values.name)
                        if values.record or method == "push_favorite"
                        else True
                    )
                elif method == "set_history":
                    result = (
                        function(
                            values.type,
                            values.id,
                            values.tags,
                            values.prompt,
                            values.name,
                        )
                        if values.record
                        else True
                    )
                elif method in ("set_history_name", "set_favorite_name"):
                    result = function(values.type, values.id, values.name)
                elif method == "remove_histories":
                    result = function(values.type)
                else:
                    result = function(values.type, values.id)
                return {"success": bool(result)}

        return operation

    for endpoint, method in {
        "push_history": "push_history",
        "push_favorite": "push_favorite",
        "move_up_favorite": "move_up_favorite",
        "move_down_favorite": "move_down_favorite",
        "set_history": "set_history",
        "set_history_name": "set_history_name",
        "set_favorite_name": "set_favorite_name",
        "dofavorite": "dofavorite",
        "unfavorite": "unfavorite",
        "delete_history": "remove_history",
        "delete_histories": "remove_histories",
    }.items():
        router.add_api_route("/" + endpoint, history_route(method), methods=["POST"])

    async def run_translation(values, user, texts):
        if sum(map(len, texts)) > 20000:
            raise HTTPException(422, "Translation text exceeds 20000 characters")
        item = provider_item(values.api)
        with store.upstream(user) as storage:
            configuration = (
                read_value(storage, user, "translate_api." + values.api) or {}
            )
        # Unsaved form values support the original Test action. Blank secrets retain saved values.
        allowed = {field["key"] for field in item.get("config", [])}
        configuration.update(
            {
                key: value
                for key, value in values.api_config.items()
                if key in allowed and value
            }
        )
        try:
            source = values.from_lang.split("_")[0]
            target = values.to_lang.split("_")[0]
            if values.api == "dictionary":
                mapping = await asyncio.to_thread(
                    dictionary, values.to_lang if target != "en" else values.from_lang
                )
                if target == "en":
                    mapping = {value.casefold(): key for key, value in mapping.items()}
                result = [
                    mapping.get(text.strip().replace("_", " ").casefold(), text)
                    for text in texts
                ]
            elif item.get("type") == "translators":
                from .providers.translators_translator import TranslatorsTranslator

                def translate_free():
                    with _free_lock:
                        adapter = (
                            TranslatorsTranslator(values.api)
                            .set_translator(item["translator"])
                            .set_api_config(configuration)
                        )
                        adapter.from_lang = item["support"][values.from_lang]
                        adapter.to_lang = item["support"][values.to_lang]
                        return adapter.translate_batch(texts)

                result = await asyncio.to_thread(translate_free)
            else:
                result = await translate(
                    ALIASES.get(values.api, values.api),
                    native_provider_config(configuration),
                    texts,
                    values.from_lang
                    if ALIASES.get(values.api, values.api) in METADATA
                    else item.get("support", {}).get(values.from_lang) or source,
                    values.to_lang
                    if ALIASES.get(values.api, values.api) in METADATA
                    else item.get("support", {}).get(values.to_lang) or target,
                )
            if len(result) != len(texts) or any(
                not isinstance(text, str) for text in result
            ):
                raise ValueError("Invalid translation result")
            return {"success": True, "translated_text": result}
        except Exception:  # noqa: BLE001 - Optional services can include credentials.
            return {
                "success": False,
                "message": "Translation failed; check provider settings and optional dependencies",
            }

    @router.post("/translates")
    async def translates(values: TranslationAction, user: User):
        return await run_translation(values, user, values.texts)

    @router.post("/translate")
    async def translate_one(values: TranslationAction, user: User):
        result = await run_translation(values, user, [values.text])
        if result.get("success"):
            result["translated_text"] = result["translated_text"][0]
        return result

    @router.post("/gen_openai")
    async def gen_openai(user: User, values: Annotated[dict, Body()]):
        messages = values.get("messages", [])
        if (
            not isinstance(messages, list)
            or len(messages) > 20
            or len(json.dumps(messages)) > 40000
        ):
            raise HTTPException(422, "Invalid prompt messages")
        with store.upstream(user) as storage:
            configuration = (
                read_value(storage, user, "chatgpt_key")
                or read_value(storage, user, "translate_api.openai")
                or {}
            )
        configuration.update(
            {
                key: value
                for key, value in values.get("api_config", {}).items()
                if key in {"api_key", "api_base", "model"} and value
            }
        )
        try:
            prompt = await completion(
                native_provider_config(configuration),
                messages[0]["content"]
                if messages
                else "Write an English image generation prompt.",
                "\n".join(message["content"] for message in messages[1:]),
            )
            return {"success": True, "result": prompt}
        except Exception:  # noqa: BLE001 - service exceptions can contain secrets.
            return {
                "success": False,
                "message": "Prompt generation failed; check OpenAI-compatible settings",
            }

    @router.post("/token_counter")
    def token_counter(values: Annotated[dict, Body()]):
        from modules.tagcomplete.api import model_capabilities

        from .tokens import count

        text = values.get("text", "")
        if not isinstance(text, str) or len(text) > 100000:
            raise HTTPException(422, "Invalid prompt")
        caps = model_capabilities(values.get("base", ""), values.get("refiner", "None"))
        counts = {"base": count(text, caps["base_anima"])}
        if values.get("refiner") not in (None, "", "None"):
            counts["refiner"] = count(text, caps["refiner_anima"])
        label = " | ".join(
            translate_ui("Base model" if name == "base" else "Refiner model")
            + ": "
            + ", ".join(
                f"{kind} {number}"
                for kind, number in data.items()
                if kind in {"clip", "qwen", "t5"}
            )
            for name, data in counts.items()
        )
        return {
            "token_count": next(iter(counts["base"].values())),
            "max_length": 75,
            "label": label,
        }

    @router.get("/get_group_tags")
    def group_tags(lang: str = "zh_CN"):
        file = (
            Path(__file__).parent / "group_tags" / (lang + ".yaml")
            if re.fullmatch(r"[a-z]{2}_[A-Z]{2}", lang)
            else None
        )
        if not file or not file.is_file():
            file = Path(__file__).parent / "group_tags/default.yaml"
        # The upstream UI consumes raw YAML including colors and layout nodes.
        return {"tags": file.read_text(encoding="utf-8")}

    @router.get("/get_csvs")
    def csvs():
        from modules.tagcomplete.api import services as completion_services

        _, catalog, _ = completion_services()
        directory = catalog.tags.resolve()
        if directory.is_relative_to(ROOT):
            directory = directory.relative_to(ROOT)
        return {
            "directory": directory.as_posix(),
            "csvs": [
                {"key": entry["id"], "name": entry["name"]}
                for entry in catalog.refresh()["datasets"]
                if Path(entry["name"]).suffix.lower() == ".csv"
            ]
        }

    @router.get("/get_csv")
    def csv_file(key: str):
        from modules.tagcomplete.api import services as completion_services

        _, catalog, _ = completion_services()
        try:
            file = catalog.asset(key, {"dataset"})
            if file.suffix.lower() != ".csv":
                raise FileNotFoundError(key)
            content = file.read_text(encoding="utf-8-sig")
        except (KeyError, ValueError, FileNotFoundError):
            raise HTTPException(404, "CSV not found") from None
        return Response(content, media_type="text/csv")

    @router.get("/get_extension_css_list")
    def extension_styles(user: User):
        result = []
        with store.upstream(user) as storage:
            for manifest in sorted((STYLES / "extensions").glob("*/manifest.json")):
                directory = manifest.parent.name
                if (manifest.parent / "style.min.css").is_file():
                    key = "extensionSelect." + directory
                    result.append(
                        {
                            "dir": directory,
                            "dataName": key,
                            "selected": storage.get(key) or False,
                            "manifest": manifest.read_text(encoding="utf-8"),
                            "style": "extensions/" + directory + "/style.min.css",
                        }
                    )
        return {"css_list": result}

    @router.get("/styles")
    def styles(file: str):
        target = (STYLES / file).resolve()
        if (
            not target.is_relative_to(STYLES)
            or not target.is_file()
            or target.suffix not in {".css", ".svg", ".png", ".jpg"}
        ):
            raise HTTPException(404, "Style asset not found")
        if target.suffix == ".css":
            content = target.read_text(encoding="utf-8")

            def asset_url(match):
                asset = match[1].strip("\"'")
                if asset.startswith(("data:", "http:", "https:", "/")):
                    return match[0]
                resolved = (target.parent / asset).resolve()
                if not resolved.is_relative_to(STYLES):
                    return match[0]
                return (
                    'url("styles?file='
                    + quote(resolved.relative_to(STYLES).as_posix())
                    + '")'
                )

            return Response(
                re.sub(r"url\(([^)]+)\)", asset_url, content), media_type="text/css"
            )
        return FileResponse(target)

    @router.get("/get_extra_networks")
    def extra_networks():
        # Real model data is shared with Fooocus's existing completion catalog.
        from modules.tagcomplete.api import services as completion_services

        _, catalog, _ = completion_services()
        data = catalog.get()
        result = []
        for key, name, title in (
            ("loras", "lora", "LoRA"),
            ("embeddings", "textual inversion", "Embeddings"),
            ("wildcards", "wildcards", "Wildcards"),
        ):
            items = []
            for item in data.get(key, []):
                filename = item["name"]
                stem = filename
                prompt = (
                    f"<lora:{filename}:{item.get('weight', 1)}>"
                    if name == "lora"
                    else "embedding:" + stem
                    if name == "textual inversion"
                    else "__" + stem + "__"
                )
                items.append(
                    {
                        "name": stem,
                        "basename": filename,
                        "dirname": filename.rpartition("/")[0],
                        "output_name": filename,
                        "prompt": json.dumps(prompt),
                        "preview": "./tagcomplete/v1/asset/" + item["preview"]
                        if item.get("preview")
                        else "",
                        "description": item.get("description", ""),
                        "keywords": item.get("keywords", ""),
                        "civitai_info": {
                            "trainedWords": item.get("keywords", "").split(",")
                            if item.get("keywords")
                            else []
                        },
                    }
                )
            result.append({"name": name, "title": translate_ui(title), "items": items})
        from modules import config

        result.insert(
            0,
            {
                "name": "checkpoints",
                "title": translate_ui("Base models"),
                "items": [
                    {
                        "name": name,
                        "basename": name,
                        "dirname": name.rpartition("/")[0],
                        "output_name": name,
                        "prompt": json.dumps(""),
                        "preview": "./prompt-all-in-one/upstream/checkpoint-preview?name="
                        + quote(name),
                        "description": "",
                        "civitai_info": {},
                    }
                    for name in config.model_filenames
                ],
            },
        )
        return {"extra_networks": result}

    @router.get("/checkpoint-preview")
    def checkpoint_preview(name: str):
        from modules import config
        from modules.tagcomplete.catalog import contained_file

        if name not in config.model_filenames:
            raise HTTPException(404, "Model is no longer available")
        for folder in config.paths_checkpoints:
            for suffix in (".preview.png", ".png", ".jpg", ".jpeg", ".webp"):
                try:
                    preview = contained_file(folder, Path(name).with_suffix(suffix))
                except FileNotFoundError:
                    continue
                return FileResponse(preview)
        return Response(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="120"><rect width="100%" height="100%" fill="#303030"/></svg>',
            media_type="image/svg+xml",
        )

    @router.get("/get_packages_state")
    def packages(provider: str = "dictionary"):
        requirements = []
        if provider == "amazon":
            requirements = [("boto3", "boto3")]
        elif provider == "alibaba":
            requirements = [
                ("aliyun-python-sdk-core", "aliyunsdkcore"),
                ("aliyun-python-sdk-alimt", "aliyunsdkalimt"),
            ]
        elif (
            provider.endswith("_free")
            and provider_item(provider).get("type") == "translators"
        ):
            requirements = [
                ("PyExecJS", "execjs"),
                ("pathos", "pathos"),
                ("lxml", "lxml"),
            ]
        return {
            "packages_state": [
                {
                    "name": module,
                    "package": name,
                    "state": importlib.util.find_spec(module) is not None,
                }
                for name, module in requirements
            ]
        }

    @router.post("/install_package")
    def install_package(values: Annotated[dict, Body()]):
        from .packages import install_package as install_optional

        return {"result": install_optional(values.get("name"), values.get("package"))}

    @router.get("/get_extensions")
    def extensions():
        return {
            "extensions": [
                {
                    "name": translate_ui("Fooocus Tag Autocomplete"),
                    "enabled": True,
                    "url": "https://github.com/licyk/Fooocus/blob/feat-1/readme.md#prompt-assistance",
                }
            ]
        }

    @router.get("/get_version")
    def version():
        return {
            "version": "5179da83c7a6ba688f924f0c81cb775d5695a414",
            "latest_version": "5179da83c7a6ba688f924f0c81cb775d5695a414",
        }

    @router.get("/get_remote_versions")
    def versions(
        page: int = Query(default=1, ge=1),
        per_page: int = Query(default=100, ge=1, le=100),
    ):
        data = json.loads(
            Path(__file__)
            .with_name("upstream_versions.json")
            .read_text(encoding="utf-8")
        )
        start = (page - 1) * per_page
        return {"versions": data[start : start + per_page]}

    @router.post("/mbart50_initialize")
    async def mbart(user: User):
        with store.upstream(user) as storage:
            configuration = read_value(storage, user, "translate_api.mbart50") or {}
        try:
            await translate(
                "mbart50", native_provider_config(configuration), ["test"], "en", "zh"
            )
            return {"success": True}
        except Exception:  # noqa: BLE001 - service exceptions can contain secrets.
            return {
                "success": False,
                "message": "Cannot initialize MBart50; check local model directory and dependencies",
            }

    return router

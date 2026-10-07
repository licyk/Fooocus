"""Validated browser preferences and explicitly saved server defaults."""

import json
import os
import re
import tempfile
from pathlib import Path

# key, label, default, allowed choices (None means a checkbox)
FIELDS = (
    ("enabled", "Enable Prompt All-in-One", False, None),
    ("positive", "Main positive prompt", True, None),
    ("negative", "Main negative prompt", True, None),
    ("inpaint", "Inpaint prompt", True, None),
    ("enhance", "Enhance prompts", True, None),
    ("hide_native", "Hide original prompt while editor is active", False, None),
    ("bilingual", "Show translated tags", True, None),
    ("auto_translate", "Automatically translate added text to English", False, None),
    ("history", "Record editing history", True, None),
    ("hotkeys", "Enable editor shortcuts", True, None),
    (
        "language",
        "Editor language",
        "zh_CN",
        (
            "zh_CN",
            "zh_TW",
            "en_US",
            "ja_JP",
            "ko_KR",
            "de_DE",
            "fr_FR",
            "es_ES",
            "it_IT",
            "pt_PT",
            "ru_RU",
            "zh_HK",
        ),
    ),
    (
        "target_language",
        "Translation language",
        "zh",
        ("zh", "en", "ja", "ko", "de", "fr", "es", "it", "pt", "ru"),
    ),
    (
        "source_language",
        "Translation source language",
        "auto",
        ("auto", "zh", "en", "ja", "ko", "de", "fr", "es", "it", "pt", "ru"),
    ),
    (
        "provider",
        "Translation provider",
        "dictionary",
        (
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
        ),
    ),
    ("separator", "Separator for newly added tags", "comma", ("comma", "newline")),
    ("theme", "Editor theme", "auto", ("auto", "light", "dark")),
    ("lora_color", "LoRA highlight color", "#be70e1", "color"),
    ("embedding_color", "Embedding highlight color", "#5d9ee2", "color"),
    ("wildcard_color", "Wildcard highlight color", "#56b58b", "color"),
    ("weight_step", "Weight adjustment step", 0.1, "number"),
    ("history_limit", "Maximum history per prompt", 100, "integer"),
    ("blacklist", "Blacklisted tags (one per line)", "", "text"),
    ("custom_groups", "Custom tag groups (JSON)", "[]", "json"),
)
DEFAULTS = {key: value for key, _, value, _ in FIELDS}


def validate_settings(values):
    if not isinstance(values, dict):
        raise TypeError("Editor settings must be an object")
    result = dict(DEFAULTS)
    for key, _, _, choices in FIELDS:
        if key not in values:
            continue
        value = values[key]
        if choices is None:
            valid = type(value) is bool
        elif choices == "number":
            valid = type(value) in (int, float) and 0.01 <= value <= 1
        elif choices == "integer":
            valid = type(value) is int and 10 <= value <= 1000
        elif choices == "color":
            valid = isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value)
        elif choices in ("text", "json"):
            valid = isinstance(value, str) and len(value) <= 100000
        else:
            valid = value in choices
        if not valid:
            raise ValueError(f"Invalid editor setting: {key}")
        if choices == "json":
            groups = json.loads(value)
            if not isinstance(groups, list) or len(groups) > 100:
                raise ValueError(
                    "Custom groups must be a JSON list (maximum 100 groups)"
                )
            for group in groups:
                if (
                    not isinstance(group, dict)
                    or not isinstance(group.get("name"), str)
                    or not isinstance(group.get("tags"), dict)
                ):
                    raise TypeError(
                        'Custom groups use [{"name":"My tags","tags":{"cat":"猫"}}]'
                    )
                if any(
                    not isinstance(tag, str) or not isinstance(label, (str, type(None)))
                    for tag, label in group["tags"].items()
                ):
                    raise ValueError("Custom tags and translations must be strings")
        result[key] = value
    return result


class SettingsStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        try:
            return validate_settings(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return dict(DEFAULTS)

    def save(self, values):
        data = validate_settings(values)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".prompt-editor-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return data

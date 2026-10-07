"""Lazy host for upstream signed translation adapters; no A1111 dependencies."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import ProviderError

METADATA = json.loads(
    Path(__file__).with_name("metadata.json").read_text(encoding="utf-8")
)
LANGUAGES = {
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
CLASSES = {
    "google": "GoogleTranslator",
    "baidu": "BaiduTranslator",
    "amazon": "AmazonTranslator",
    "alibaba": "AlibabaTranslator",
    "yandex": "YandexTranslator",
    "youdao": "YoudaoTranslator",
    "tencent": "TencentTranslator",
    "mymemory": "MyMemoryTranslator",
    "niutrans": "NiutransTranslator",
    "caiyun": "CaiyunTranslator",
    "volcengine": "VolcengineTranslator",
    "iflytekV1": "IflytekV1Translator",
    "iflytekV2": "IflytekV2Translator",
}


class Requests:
    """Keep upstream HTTP requests bounded without modifying their signing code."""

    @staticmethod
    def request(method, url, **kwargs):
        import requests as http

        kwargs.setdefault("timeout", 30)
        return http.request(method, url, **kwargs)

    @classmethod
    def get(cls, url, **kwargs):
        return cls.request("GET", url, **kwargs)

    @classmethod
    def post(cls, url, **kwargs):
        return cls.request("POST", url, **kwargs)


requests = Requests()


def get_lang(key, values=None):
    return key + (": " + ", ".join(values.values()) if values else "")


class BaseTranslator:
    def __init__(self, name):
        self.api = name
        self.api_config = {}
        self.from_lang = self.to_lang = None

    def set_api_config(self, configuration):
        self.api_config = configuration
        return self

    def get_concurrent(self):
        return 10

    def translate_batch(self, texts):
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(self.translate, texts))


def translate_upstream(name, configuration, texts, source, target):
    mapping = METADATA[name]["languages"]
    if (
        source == "auto"
        or LANGUAGES.get(source, source) not in mapping
        or LANGUAGES.get(target, target) not in mapping
    ):
        raise ProviderError(
            "This provider requires supported explicit source/target languages"
        )
    module = "google_tanslator" if name == "google" else name + "_translator"
    adapter = getattr(
        importlib.import_module("." + module, __package__), CLASSES[name]
    )()
    adapter.from_lang = mapping[LANGUAGES.get(source, source)]
    adapter.to_lang = mapping[LANGUAGES.get(target, target)]
    adapter.set_api_config(configuration.get("options", {}))
    return adapter.translate_batch(texts)

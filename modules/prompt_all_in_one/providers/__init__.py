"""Optional translation services, invoked only after explicit user action/configuration."""

import asyncio
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class ProviderError(ValueError):
    pass


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderError(
            "Provider endpoint must be an HTTP(S) URL without credentials or query parameters"
        )
    return endpoint.rstrip("/")


async def completion(config, instruction, text):
    endpoint = validate_endpoint(config.get("endpoint") or "https://api.openai.com/v1")
    if not config.get("key") or not config.get("model"):
        raise ProviderError("Configure an API key and model first")
    async with httpx.AsyncClient(timeout=40) as client:
        response = await client.post(
            endpoint + "/chat/completions",
            headers={"Authorization": "Bearer " + config["key"]},
            json={
                "model": config["model"],
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


# The optional offline model is never downloaded or initialized at UI startup.
_offline_lock = None
_offline_model = None
_offline_thread_lock = threading.Lock()


def offline_translate(config, texts, source, target):
    # A cancelled HTTP coroutine cannot interrupt an already running CPU thread.
    # Keep tokenizer/model mutation protected until that thread actually exits.
    with _offline_thread_lock:
        return _offline_translate(config, texts, source, target)


def _offline_translate(config, texts, source, target):
    global _offline_model
    from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

    languages = {
        "zh": "zh_CN",
        "en": "en_XX",
        "ja": "ja_XX",
        "ko": "ko_KR",
        "de": "de_DE",
        "fr": "fr_XX",
        "es": "es_XX",
        "it": "it_IT",
        "pt": "pt_XX",
        "ru": "ru_RU",
    }
    if source not in languages or target not in languages:
        raise ProviderError("Offline translation requires an explicit source language")
    model_path = Path(config.get("model_path", ""))
    if not config.get("model_path") or not model_path.is_dir():
        raise ProviderError("Configure a local MBart50 model directory first")
    model_path = str(model_path.resolve())
    if _offline_model is None or _offline_model[0] != model_path:
        tokenizer = MBart50TokenizerFast.from_pretrained(
            model_path, local_files_only=True
        )
        model = MBartForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True
        ).eval()
        _offline_model = (model_path, tokenizer, model)
    _, tokenizer, model = _offline_model
    tokenizer.src_lang = languages[source]
    import torch

    with torch.inference_mode():
        tokens = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        generated = model.generate(
            **tokens,
            forced_bos_token_id=tokenizer.lang_code_to_id[languages[target]],
            max_new_tokens=512,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


async def translate(provider, config, texts, source, target):
    from .base import METADATA, translate_upstream

    if provider in METADATA:
        return await asyncio.to_thread(
            translate_upstream, provider, config, texts, source, target
        )
    if provider == "mbart50":
        global _offline_lock
        if _offline_lock is None:
            _offline_lock = asyncio.Lock()
        async with _offline_lock:
            return await asyncio.to_thread(
                offline_translate, config, texts, source, target
            )
    if provider == "openai":
        result = await completion(
            config,
            f"Translate the JSON array into {target}. Preserve prompt syntax, weights and model names. Return ONLY a JSON array of {len(texts)} strings, in the same order.",
            json.dumps(texts, ensure_ascii=False),
        )
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0]
        values = json.loads(result)
        if (
            not isinstance(values, list)
            or len(values) != len(texts)
            or any(not isinstance(value, str) for value in values)
        ):
            raise ProviderError("Translation response has an invalid shape")
        return values
    async with httpx.AsyncClient(timeout=30) as client:
        if provider == "deepl":
            if not config.get("key"):
                raise ProviderError("Configure a DeepL API key first")
            endpoint = validate_endpoint(
                config.get("endpoint") or "https://api-free.deepl.com/v2"
            )
            body = {"text": texts, "target_lang": target.upper()}
            if source != "auto":
                body["source_lang"] = source.upper()
            response = await client.post(
                endpoint + "/translate",
                json=body,
                headers={"Authorization": "DeepL-Auth-Key " + config["key"]},
            )
            response.raise_for_status()
            return [entry["text"] for entry in response.json()["translations"]]
        if provider == "libretranslate":
            endpoint = validate_endpoint(config.get("endpoint", ""))
            response = await client.post(
                endpoint + "/translate",
                json={
                    "q": texts,
                    "source": source,
                    "target": target,
                    "format": "text",
                    "api_key": config.get("key", ""),
                },
            )
            response.raise_for_status()
            return response.json()["translatedText"]
        if provider == "microsoft":
            if not config.get("key"):
                raise ProviderError("Configure a Microsoft Translator API key first")
            endpoint = validate_endpoint(
                config.get("endpoint")
                or "https://api.cognitive.microsofttranslator.com"
            )
            params = {"api-version": "3.0", "to": target}
            if source != "auto":
                params["from"] = source
            response = await client.post(
                endpoint + "/translate",
                params=params,
                headers={
                    "Ocp-Apim-Subscription-Key": config["key"],
                    "Ocp-Apim-Subscription-Region": config.get("region", ""),
                },
                json=[{"Text": text} for text in texts],
            )
            response.raise_for_status()
            return [entry["translations"][0]["text"] for entry in response.json()]
    raise ProviderError("Unsupported translation provider")

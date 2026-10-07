import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.prompt_all_in_one.api import create_router
from modules.prompt_all_in_one.config import DEFAULTS, SettingsStore, validate_settings
from modules.prompt_all_in_one.storage import Store


class TestPromptEditor(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.settings = SettingsStore(self.directory / "settings.json")
        self.store = Store(self.directory / "editor.sqlite3")
        self.app = FastAPI()
        self.app.auth = None
        self.app.include_router(create_router(self.settings, self.store))
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def entry(self, **values):
        return {
            "scope": "positive",
            "prompt": "cat, dog",
            "tags": [
                {"raw": "cat", "separator": ", ", "disabled": False},
                {"raw": "dog", "disabled": True},
            ],
            **values,
        }

    def test_default_disabled_and_validated_atomic_preferences(self):
        self.assertFalse(self.store.path.exists())
        self.assertFalse(
            self.client.get("/prompt-all-in-one/v1/bootstrap").json()["settings"][
                "enabled"
            ]
        )
        self.assertFalse(self.store.path.exists())
        self.settings.save({"enabled": True})
        self.assertTrue(self.settings.load()["enabled"])
        for values in (
            {"enabled": 1},
            {"history_limit": True},
            {"weight_step": float("nan")},
            {"custom_groups": "{}"},
            {"provider": "unknown"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_settings(values)
        self.assertEqual(validate_settings({"unknown": 123}), DEFAULTS)
        self.settings.path.write_text("broken", encoding="utf-8")
        self.assertEqual(self.settings.load(), DEFAULTS)

    def test_history_deduplicated_bounded_favorites_durable(self):
        first = self.store.add("alice", "positive", "cat", [], limit=10)
        self.assertEqual(
            first["id"], self.store.add("alice", "positive", "cat", [], limit=10)["id"]
        )
        for index in range(12):
            self.store.add("alice", "positive", str(index), [], limit=10)
        self.assertEqual(len(self.store.list("alice", "positive")), 10)
        favorite = self.store.add("alice", "positive", "cat", [], favorite=True)
        self.store.delete("alice", "positive")
        self.assertEqual(
            Store(self.store.path).list("alice", "positive", True)[0]["id"],
            favorite["id"],
        )

    def test_storage_and_routes_isolate_users_and_fields(self):
        def auth(request):
            return request.headers.get("x-test-user")

        self.app.auth_dependency = auth
        self.assertEqual(
            self.client.get("/prompt-all-in-one/v1/bootstrap").status_code, 401
        )
        alice = {"x-test-user": "alice"}
        bob = {"x-test-user": "bob"}
        result = self.client.post(
            "/prompt-all-in-one/v1/entries", json=self.entry(), headers=alice
        )
        self.assertEqual(result.status_code, 200)
        identifier = result.json()["id"]
        self.assertTrue(result.json()["tags"][1]["disabled"])
        self.assertEqual(
            self.client.get(
                "/prompt-all-in-one/v1/entries?scope=positive", headers=bob
            ).json(),
            [],
        )
        self.assertEqual(
            self.client.get(
                "/prompt-all-in-one/v1/entries?scope=negative", headers=alice
            ).json(),
            [],
        )
        self.assertEqual(
            self.client.patch(
                "/prompt-all-in-one/v1/entries/" + identifier,
                json={"name": "stolen"},
                headers=bob,
            ).status_code,
            404,
        )
        self.client.delete(
            "/prompt-all-in-one/v1/entries?scope=positive&identifier=" + identifier,
            headers=bob,
        )
        self.assertEqual(
            len(
                self.client.get(
                    "/prompt-all-in-one/v1/entries?scope=positive", headers=alice
                ).json()
            ),
            1,
        )
        self.assertEqual(
            self.client.post(
                "/prompt-all-in-one/v1/entries",
                json=self.entry(scope="../outside"),
                headers=alice,
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/prompt-all-in-one/v1/entries",
                json=self.entry(extra="unknown"),
                headers=alice,
            ).status_code,
            422,
        )

    def test_credentials_never_return_even_short_keys_and_blank_retains(self):
        prefix = "/prompt-all-in-one/v1/providers/openai"
        self.client.put(
            prefix,
            json={"key": "abc", "endpoint": "https://example.test/v1", "model": "test"},
        )
        result = self.client.get(prefix).json()
        self.assertTrue(result["configured"])
        self.assertNotIn("key", result)
        self.assertNotIn("abc", json.dumps(result))
        self.client.put(prefix, json={"key": "", "model": "another"})
        self.assertEqual(self.store.provider("local", "openai")["key"], "abc")
        self.client.put(prefix, json={"clear_key": True})
        self.assertFalse(self.client.get(prefix).json()["configured"])
        self.assertEqual(
            self.client.put(
                prefix, json={"endpoint": "file:///etc/passwd"}
            ).status_code,
            422,
        )

    def test_provider_failures_do_not_leak_secrets(self):
        with patch(
            "modules.prompt_all_in_one.api.translate",
            AsyncMock(side_effect=ValueError("secret-key")),
        ):
            result = self.client.post(
                "/prompt-all-in-one/v1/translate",
                json={"provider": "openai", "texts": ["cat"]},
            )
        self.assertEqual(result.status_code, 502)
        self.assertNotIn("secret-key", result.text)
        self.assertEqual(
            self.client.post(
                "/prompt-all-in-one/v1/translate", json={"texts": ["x" * 20001]}
            ).status_code,
            422,
        )

    def test_dictionary_translation_uses_local_data(self):
        with patch(
            "modules.prompt_all_in_one.api.dictionary", return_value={"cat": "猫"}
        ):
            result = self.client.post(
                "/prompt-all-in-one/v1/translate",
                json={"texts": ["cat", "<lora:cat:1>"]},
            )
            self.assertEqual(result.json()["texts"], ["猫", "<lora:cat:1>"])
            result = self.client.post(
                "/prompt-all-in-one/v1/translate",
                json={"texts": ["猫"], "target": "en"},
            )
            self.assertEqual(result.json()["texts"], ["cat"])

    def test_native_counter_reports_separate_anima_and_refiner_counts(self):
        from modules.prompt_all_in_one.tokens import count

        self.assertGreater(count("(cat:1.2), dog")["clip"], 0)
        anima = count("(cat:1.2), dog", True)
        self.assertGreater(anima["qwen"], 0)
        self.assertGreater(anima["t5"], 0)
        self.assertNotIn("chunk_size", anima)
        with patch(
            "modules.tagcomplete.api.model_capabilities",
            return_value={"base_anima": False, "refiner_anima": True},
        ):
            result = self.client.post(
                "/prompt-all-in-one/v1/tokens",
                json={"text": "cat", "base": "sdxl", "refiner": "anima"},
            )
        self.assertEqual(result.json()["base"]["kind"], "sdxl")
        self.assertEqual(result.json()["refiner"]["kind"], "anima")

    def test_upstream_provider_configs_keep_every_secret_server_side(self):
        from modules.prompt_all_in_one.providers.base import METADATA

        for name, item in METADATA.items():
            with self.subTest(provider=name):
                secret = {
                    field["key"]: "private-" + field["key"]
                    for field in item["fields"]
                    if field["secret"]
                }
                result = self.client.put(
                    "/prompt-all-in-one/v1/providers/" + name, json={"options": secret}
                )
                self.assertEqual(result.status_code, 200)
                self.assertNotIn("private-", result.text)
                self.assertTrue(result.json()["configured"])
                result = self.client.put(
                    "/prompt-all-in-one/v1/providers/" + name, json={"options": {}}
                )
                self.assertNotIn("private-", result.text)
                self.assertTrue(
                    all(
                        self.store.provider("local", name)["options"][key] == value
                        for key, value in secret.items()
                    )
                )
                self.client.put(
                    "/prompt-all-in-one/v1/providers/" + name, json={"clear_key": True}
                )
                self.assertFalse(
                    self.client.get("/prompt-all-in-one/v1/providers/" + name).json()[
                        "configured"
                    ]
                )

    def test_ported_google_and_baidu_use_mapped_languages_and_bounded_http(self):
        from modules.prompt_all_in_one.providers.base import translate_upstream

        response = Mock()
        response.json.return_value = {
            "data": {"translations": [{"translatedText": "猫"}]}
        }
        with patch("requests.request", return_value=response) as request:
            values = translate_upstream(
                "google", {"options": {"api_key": "secret"}}, ["cat"], "en", "zh"
            )
        self.assertEqual(values, ["猫"])
        self.assertEqual(request.call_args.kwargs["params"]["source"], "en")
        self.assertEqual(request.call_args.kwargs["params"]["target"], "zh-CN")
        self.assertGreater(request.call_args.kwargs["timeout"], 0)
        response.json.return_value = {"trans_result": [{"dst": "cat"}, {"dst": "dog"}]}
        with patch("requests.request", return_value=response):
            values = translate_upstream(
                "baidu",
                {"options": {"app_id": "id", "app_secret": "secret"}},
                ["猫", "狗"],
                "zh",
                "en",
            )
        self.assertEqual(values, ["cat", "dog"])

    def test_native_library_is_bounded_to_known_languages(self):
        from modules.prompt_all_in_one.api import groups

        self.assertEqual(groups("../outside"), groups("default"))
        self.assertGreater(len(groups("zh_CN")), 1)
        for language in (
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
            "default",
        ):
            for category in groups(language):
                for group in category["groups"]:
                    self.assertIsInstance(group["name"], str)
                    self.assertTrue(
                        all(
                            isinstance(tag, str) and isinstance(label, str)
                            for tag, label in group["tags"].items()
                        )
                    )
        from modules.prompt_all_in_one.api import dictionary

        settings = Mock()
        settings.load.return_value = {"translation_file": "None"}
        catalog = Mock()
        catalog.get.return_value = {"datasets": []}
        with patch(
            "modules.tagcomplete.api.services", return_value=(settings, catalog, None)
        ):
            self.assertEqual(dictionary("zh_CN")["cat"], "猫")
            self.assertEqual(dictionary("zh_CN")["dog"], "狗")


class TestOfflineTranslationCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_cpu_request_cannot_overlap_the_next_model_use(self):
        import asyncio
        import threading

        from modules.prompt_all_in_one import providers

        started = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        def translate_model(config, texts, source, target):
            if texts == ["first"]:
                started.set()
                release.wait(timeout=5)
            else:
                second_started.set()
            return texts

        with (
            patch.object(providers, "_offline_lock", None),
            patch.object(providers, "_offline_translate", side_effect=translate_model),
        ):
            first = asyncio.create_task(
                providers.translate("mbart50", {}, ["first"], "en", "zh")
            )
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(
                providers.translate("mbart50", {}, ["second"], "en", "zh")
            )
            try:
                await asyncio.sleep(0.05)
                self.assertFalse(second_started.is_set())
            finally:
                release.set()
            self.assertEqual(await asyncio.wait_for(second, 3), ["second"])


if __name__ == "__main__":
    unittest.main()

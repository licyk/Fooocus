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


class TestUpstreamPromptEditor(unittest.TestCase):
    def setUp(self):
        from modules.prompt_all_in_one.upstream_api import (
            create_router as upstream_router,
        )

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "editor.sqlite3")
        self.settings = SettingsStore(Path(temporary.name) / "settings.json")
        self.app = FastAPI()
        self.app.auth_dependency = lambda request: request.headers.get("x-test-user")
        self.app.include_router(upstream_router(self.settings, self.store))
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.alice = {"x-test-user": "alice"}
        self.bob = {"x-test-user": "bob"}
        self.prefix = "/prompt-all-in-one/upstream"

    def post(self, endpoint, **values):
        response = self.client.post(
            self.prefix + "/" + endpoint, json=values, headers=self.alice
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def get(self, endpoint):
        response = self.client.get(self.prefix + "/" + endpoint, headers=self.alice)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_full_upstream_config_authenticated_and_credential_defaults_empty(self):
        self.assertEqual(self.client.get(self.prefix + "/get_config").status_code, 401)
        config = self.get("get_config")
        self.assertEqual(len(config["i18n"]["languages"]), 108)
        apis = [
            item
            for group in config["translate_apis"]["apis"]
            for item in group["children"]
        ]
        self.assertGreater(len(apis), 35)
        for api in apis:
            for field in api.get("config", []):
                if field.get("secret"):
                    self.assertEqual(field.get("default"), "")
        self.assertEqual(
            self.get("get_packages_state?provider=dictionary")["packages_state"], []
        )

    def test_original_history_favorite_link_rename_reorder_and_delete(self):
        tags = [
            {
                "id": "1",
                "value": "cat",
                "localValue": "猫",
                "disabled": True,
                "type": "text",
                "weightNum": 1.2,
            }
        ]
        self.post("push_history", type="positive", tags=tags, prompt="", name="first")
        first = self.get("get_latest_history?type=positive")["history"]
        self.assertEqual(first["tags"], tags)
        self.post("dofavorite", type="positive", id=first["id"])
        self.post("set_history_name", type="positive", id=first["id"], name="renamed")
        self.assertEqual(
            self.get("get_favorites?type=positive")["favorites"][0]["name"], "renamed"
        )
        self.post(
            "push_favorite", type="positive", tags=tags, prompt="dog", name="second"
        )
        entries = self.get("get_favorites?type=positive")["favorites"]
        self.post("move_up_favorite", type="positive", id=entries[1]["id"])
        self.assertEqual(
            self.get("get_favorites?type=positive")["favorites"][0]["name"], "second"
        )
        self.post("delete_histories", type="positive")
        self.assertEqual(self.get("get_histories?type=positive")["histories"], [])
        self.assertEqual(len(self.get("get_favorites?type=positive")["favorites"]), 2)
        self.post("unfavorite", type="positive", id=first["id"])
        self.assertEqual(len(self.get("get_favorites?type=positive")["favorites"]), 1)

    def test_native_history_migration_preserves_disabled_metadata_and_user(self):
        old = self.store.add(
            "alice",
            "enhance_2",
            "cat",
            [{"raw": "dog", "disabled": True}],
            favorite=True,
        )
        migrated = self.get("get_favorites?type=enhance_2")["favorites"][0]
        self.assertEqual(migrated["id"], old["id"])
        self.assertEqual(migrated["tags"][0]["value"], "dog")
        self.assertTrue(migrated["tags"][0]["disabled"])
        self.assertEqual(
            self.client.get(
                self.prefix + "/get_favorites?type=enhance_2", headers=self.bob
            ).json()["favorites"],
            [],
        )

    def test_private_credentials_keep_clear_and_users_are_isolated(self):
        self.post(
            "set_data",
            key="translate_api.openai",
            data={
                "api_key": "private-alice",
                "model": "test-model",
                "api_base": "http://127.0.0.1:9999/v1",
            },
        )
        self.assertNotIn(
            "private-alice",
            json.dumps(self.get("get_datas?keys=translate_api.openai,chatgpt_key")),
        )
        self.post(
            "set_data",
            key="translate_api.openai",
            data={"api_key": "", "model": "another"},
        )
        with self.store.upstream("alice") as storage:
            self.assertEqual(
                storage.get("translate_api.openai")["api_key"], "private-alice"
            )
        with self.store.upstream("bob") as storage:
            self.assertIsNone(storage.get("translate_api.openai"))
        self.post("set_data", key="translate_api.openai", data={"clear_key": True})
        with self.store.upstream("alice") as storage:
            self.assertEqual(storage.get("translate_api.openai")["api_key"], "")

    def test_scope_key_and_style_path_validation(self):
        self.assertEqual(
            self.client.post(
                self.prefix + "/push_history",
                json={"type": "../../auth"},
                headers=self.alice,
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                self.prefix + "/set_data",
                json={"key": "../auth", "data": 1},
                headers=self.alice,
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.get(
                self.prefix + "/styles?file=../../../auth.json", headers=self.alice
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get(
                self.prefix + "/styles?file=icons/copy.svg", headers=self.alice
            ).status_code,
            200,
        )

    def test_upstream_raw_groups_retain_colors_wrap_nodes_and_local_translate(self):
        import yaml

        raw = self.get("get_group_tags?lang=zh_CN")["tags"]
        categories = yaml.load(raw, Loader=yaml.BaseLoader)
        self.assertTrue(
            any(
                group.get("color")
                for category in categories
                for group in category.get("groups", [])
            )
        )
        self.assertTrue(
            any(
                group.get("type") == "wrap"
                for category in categories
                for group in category.get("groups", [])
            )
        )
        with patch(
            "modules.prompt_all_in_one.upstream_api.dictionary",
            return_value={"cat": "猫"},
        ):
            self.assertEqual(
                self.post(
                    "translates",
                    texts=["猫"],
                    from_lang="zh_CN",
                    to_lang="en_US",
                    api="dictionary",
                )["translated_text"],
                ["cat"],
            )

    def test_history_limit_and_disabled_recording(self):
        for index in range(12):
            self.post("push_history", type="negative", prompt=str(index), limit=10)
        self.assertEqual(len(self.get("get_histories?type=negative")["histories"]), 10)
        self.post("push_history", type="negative", prompt="disabled", record=False)
        self.assertNotEqual(
            self.get("get_latest_history?type=negative")["history"]["prompt"],
            "disabled",
        )

    def test_bulk_preferences_and_original_list_contracts(self):
        self.post("set_datas", sampleList=["first", "second"], sampleFlag=True)
        self.assertTrue(self.get("get_data?key=sampleFlag")["data"])
        self.post("push_data_list", key="sampleList", item="third")
        self.assertEqual(
            self.get("get_data_list_item?key=sampleList&index=1")["item"], "second"
        )
        self.assertEqual(
            self.post("shift_data_list", key="sampleList")["item"], "first"
        )
        self.assertEqual(self.post("pop_data_list", key="sampleList")["item"], "third")
        self.post("remove_data_list", key="sampleList", index=0)
        self.post("push_data_list", key="sampleList", item="last")
        self.post("clear_data_list", key="sampleList")
        self.assertIsNone(self.get("get_data_list_item?key=sampleList&index=0")["item"])
        self.assertEqual(
            self.client.post(
                self.prefix + "/push_data_list",
                json={"key": "translate_api.openai", "item": "secret"},
                headers=self.alice,
            ).status_code,
            422,
        )

    def test_optional_installer_allowlist_uses_current_interpreter(self):
        import sys

        with patch("modules.prompt_all_in_one.packages.subprocess.run") as run:
            result = self.post("install_package", name="execjs", package="PyExecJS")
            self.assertTrue(result["result"]["state"])
            self.assertEqual(
                run.call_args.args[0],
                [sys.executable, "-m", "pip", "install", "PyExecJS"],
            )
            run.reset_mock()
            self.assertFalse(
                self.post(
                    "install_package", name="execjs", package="--arbitrary-option"
                )["result"]["state"]
            )
            run.assert_not_called()

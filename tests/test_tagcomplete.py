import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from random import Random
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.tagcomplete.api import create_router
from modules.tagcomplete.catalog import Catalog
from modules.tagcomplete.config import DEFAULTS, SettingsStore, validate_settings
from modules.tagcomplete.frequency import FrequencyStore


class TestCompletion(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in (
            "tags",
            "loras/one",
            "loras/two",
            "embeddings",
            "wildcards/hair",
        ):
            (self.root / directory).mkdir(parents=True)
        (self.root / "tags/main.csv").write_text(
            'cat,0,20,"kitty,feline"\n', encoding="utf-8"
        )
        (self.root / "wildcards/hair/color.txt").write_text(
            "# ignored\nred_hair\nblue_hair\n", encoding="utf-8"
        )
        for name in ("one/cat", "two/cat"):
            (self.root / f"loras/{name}.safetensors").write_bytes(b"weights not loaded")
        (self.root / "loras/one/cat.json").write_text(
            json.dumps({"preferred weight": 0.7, "activation text": "cat style"})
        )
        (self.root / "loras/one/cat.png").write_bytes(b"preview")
        self.catalog = Catalog(
            self.root / "tags",
            [self.root / "loras"],
            self.root / "embeddings",
            self.root / "wildcards",
            ["Fooocus V2"],
        )
        self.store = SettingsStore(self.root / "settings.json")
        self.frequency = FrequencyStore(self.root / "usage.sqlite3")
        self.app = FastAPI()
        self.app.auth = None
        self.app.auth_dependency = None
        self.app.include_router(create_router(self.store, self.catalog, self.frequency))
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_settings_round_trip_atomic_write_and_unknown_keys(self):
        settings = self.store.save(
            {**DEFAULTS, "delay_ms": 250, "future_key": "ignored"}
        )
        self.assertEqual(self.store.load(), settings)
        self.assertNotIn("future_key", settings)
        self.assertEqual(self.store.load()["delay_ms"], 250)
        self.assertFalse(list(self.root.glob(".tagcomplete-*")))

    def test_invalid_values_never_replace_saved_settings(self):
        self.store.save(DEFAULTS)
        for key, value in [
            ("max_results", True),
            ("delay_ms", -1),
            ("lora_weight", float("nan")),
            ("tag_file", "../private.csv"),
            ("colors", []),
            ("categories", ["99"]),
            ("keymap", {**DEFAULTS["keymap"], "choose": "Tab"}),
        ]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.store.save({key: value})
                self.assertEqual(self.store.load(), DEFAULTS)

    def test_corrupt_defaults_fall_back_without_overwriting_file(self):
        for document in ("invalid JSON", "[]", "null"):
            self.store.path.write_text(document)
            self.assertEqual(self.store.load(), DEFAULTS)
            self.assertEqual(self.store.path.read_text(), document)

    def test_alias_only_enables_alias_search(self):
        result = validate_settings({"only_alias": True, "search_aliases": False})
        self.assertTrue(result["search_aliases"])

    def test_catalog_retains_nested_names_and_local_metadata(self):
        data = self.catalog.get()
        self.assertEqual([row["name"] for row in data["loras"]], ["one/cat", "two/cat"])
        self.assertEqual(data["loras"][0]["weight"], 0.7)
        self.assertEqual(data["loras"][0]["keywords"], "cat style")
        self.assertEqual(data["wildcards"][0]["name"], "hair/color")
        self.assertIn("preview", data["loras"][0])

    def test_assets_never_serve_model_weights_or_arbitrary_paths(self):
        lora = self.catalog.get()["loras"][0]
        self.assertEqual(
            self.client.get(f"/tagcomplete/v1/asset/{lora['id']}").status_code, 404
        )
        self.assertEqual(
            self.client.get("/tagcomplete/v1/asset/unknown").status_code, 404
        )
        dataset = self.catalog.get()["datasets"][0]
        self.assertIn(
            "kitty", self.client.get(f"/tagcomplete/v1/asset/{dataset['id']}").text
        )

    def test_symlink_outside_tag_root_is_excluded_and_rechecked(self):
        secret = self.root / "secret.csv"
        secret.write_text("secret")
        link = self.root / "tags/link.csv"
        link.symlink_to(secret)
        self.assertNotIn(
            "link.csv", [row["name"] for row in self.catalog.refresh()["datasets"]]
        )
        dataset = self.catalog.get()["datasets"][0]
        path = self.root / "tags/main.csv"
        path.unlink()
        path.symlink_to(secret)
        self.assertEqual(
            self.client.get(f"/tagcomplete/v1/asset/{dataset['id']}").status_code, 404
        )

    def test_refresh_invalidates_revision_and_discovers_new_files(self):
        before = self.catalog.get()["revision"]
        (self.root / "tags/new.csv").write_text("new_tag,0,1,")
        after = self.client.post("/tagcomplete/v1/refresh").json()
        self.assertNotEqual(before, after["revision"])
        self.assertIn("new.csv", [row["name"] for row in after["datasets"]])

    def test_frequency_is_lazy_and_separates_users_and_roles(self):
        self.assertEqual(self.frequency.get("alice"), [])
        self.assertFalse(self.frequency.path.exists())
        self.frequency.increase("alice", "cat", "tag", False)
        self.frequency.increase("alice", "cat", "tag", False)
        self.frequency.increase("alice", "cat", "tag", True)
        self.frequency.increase("bob", "cat", "tag", False)
        self.assertEqual([row["count"] for row in self.frequency.get("alice")], [2, 1])
        self.frequency.clear("alice")
        self.assertEqual(self.frequency.get("alice"), [])
        self.assertEqual(self.frequency.get("bob")[0]["count"], 1)

    def test_all_routes_require_existing_gradio_login(self):
        self.app.auth = lambda user, password: True
        self.app.cookie_id = "test"
        self.app.tokens = {"secret": "alice", "other": "bob"}
        for method, path in [
            ("GET", "bootstrap"),
            ("GET", "catalog"),
            ("GET", "asset/unknown"),
            ("GET", "usage"),
            ("DELETE", "usage"),
            ("POST", "refresh"),
        ]:
            with self.subTest(path=path):
                self.assertEqual(
                    self.client.request(method, "/tagcomplete/v1/" + path).status_code,
                    401,
                )
        self.client.cookies.set("access-token-test", "secret")
        self.assertEqual(self.client.get("/tagcomplete/v1/bootstrap").status_code, 200)
        self.assertEqual(
            self.client.post(
                "/tagcomplete/v1/usage", json={"name": "cat", "kind": "tag"}
            ).status_code,
            200,
        )
        self.assertEqual(self.frequency.get("alice")[0]["count"], 1)
        self.assertEqual(self.frequency.get("bob"), [])

    def test_async_auth_dependency(self):
        async def auth(request):
            return "alice" if request.headers.get("x-test-auth") else None

        self.app.auth_dependency = auth
        self.assertEqual(self.client.get("/tagcomplete/v1/bootstrap").status_code, 401)
        self.assertEqual(
            self.client.get(
                "/tagcomplete/v1/bootstrap", headers={"x-test-auth": "yes"}
            ).status_code,
            200,
        )

    def test_invalid_frequency_event_rejected(self):
        response = self.client.post(
            "/tagcomplete/v1/usage", json={"name": "cat", "kind": "unknown"}
        )
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.frequency.path.exists())

    def test_nested_lora_resolution_preserves_legacy_stems(self):
        from modules.util import parse_lora_references_from_prompt

        filenames = ["one/cat.safetensors", "two/cat.safetensors"]
        loras, text = parse_lora_references_from_prompt(
            "scene, <lora:two/cat:.7>", [], lora_filenames=filenames
        )
        self.assertEqual(loras, [("two/cat.safetensors", 0.7)])
        self.assertEqual(text, "scene")
        loras, _ = parse_lora_references_from_prompt(
            "<lora:cat:1>", [], lora_filenames=filenames
        )
        self.assertEqual(loras, [("one/cat.safetensors", 1)])

    def test_nested_wildcard_output_matches_completion_and_ignores_comments(self):
        from modules import config, util

        with (
            patch.object(config, "path_wildcards", str(self.root / "wildcards")),
            patch.object(config, "wildcard_filenames", ["hair/color.txt"]),
        ):
            self.assertEqual(
                util.apply_wildcards("__hair/color__", Random(42), 0, True), "red_hair"
            )
            self.assertEqual(
                util.apply_wildcards("__color__", Random(42), 1, True), "blue_hair"
            )

    def test_wildcard_spaces_and_multiple_placeholders(self):
        from modules import config, util

        (self.root / "wildcards/hair/my color (v1).txt").write_text("green_hair")
        (self.root / "wildcards/ending_.txt").write_text("legacy")
        with (
            patch.object(config, "path_wildcards", str(self.root / "wildcards")),
            patch.object(
                config,
                "wildcard_filenames",
                ["hair/color.txt", "hair/my color (v1).txt", "ending_.txt"],
            ),
        ):
            self.assertEqual(
                util.apply_wildcards(
                    "__hair/my color (v1)__ and __hair/color__", Random(42), 0, True
                ),
                "green_hair and red_hair",
            )
            self.assertEqual(
                util.apply_wildcards("__ending___", Random(42), 0, True), "legacy"
            )

    def test_model_capabilities_inspect_headers_and_tolerate_corrupt_files(self):
        import torch
        from safetensors.torch import save_file
        from modules import config
        from modules.tagcomplete.api import model_capabilities

        save_file(
            {
                "net.llm_adapter.blocks.0.cross_attn.q_proj.weight": torch.zeros(1),
                "net.x_embedder.proj.1.weight": torch.zeros(1),
            },
            str(self.root / "anima.safetensors"),
        )
        (self.root / "corrupt.safetensors").write_bytes(b"not a checkpoint")
        with patch.object(config, "paths_checkpoints", [str(self.root)]):
            self.assertFalse(
                model_capabilities("anima.safetensors", "None")["embeddings"]
            )
            self.assertFalse(
                model_capabilities("None", "anima.safetensors")["embeddings"]
            )
            self.assertTrue(
                model_capabilities("corrupt.safetensors", "missing.safetensors")[
                    "embeddings"
                ]
            )

    def test_catalog_excludes_names_that_prompt_parsers_cannot_use(self):
        for filename in (
            "embeddings/two words.pt",
            "loras/invalid,name.pt",
            "wildcards/hair/invalid:name.txt",
        ):
            (self.root / filename).write_bytes(b"fixture")
        data = self.catalog.refresh()
        self.assertEqual(data["embeddings"], [])
        self.assertEqual(len(data["loras"]), 2)
        self.assertEqual(len(data["wildcards"]), 1)

    @unittest.skipUnless(
        shutil.which("node"), "Node is needed for frontend engine tests"
    )
    def test_browser_engine(self):
        script = Path(__file__).with_name("tagcomplete_engine.test.js")
        result = subprocess.run(
            ["node", str(script)],
            input=json.dumps(DEFAULTS),
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

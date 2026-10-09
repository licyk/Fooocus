import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from extras import wd14tagger
from extras.wd14_tagger.cl import CLTaggerInterrogator
from extras.wd14_tagger.interrogator import Interrogator
from extras.wd14_tagger.models import MODELS
from extras.wd14_tagger.wd14 import WaifuDiffusionInterrogator
from modules import flags
from modules.describe import describe_image


class FakeSession:
    def __init__(self, kind, output):
        self.kind = kind
        self.output = np.array([output], dtype=np.float32)
        self.inputs = []

    def get_inputs(self):
        shape = [1, 3, 4, 6] if self.kind == "cl" else [1, 4, 4, 3]
        return [SimpleNamespace(name="image", shape=shape)]

    def get_outputs(self):
        return [SimpleNamespace(name="probabilities")]

    def run(self, outputs, inputs):
        self.inputs.append(inputs["image"])
        return [self.output]


class TestTaggerInference(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv_path = self.root / "selected_tags.csv"
        with self.csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["tag_id", "name", "category"])
            # Intentionally interleave the categories, including a rating away
            # from the first four rows and a general tag after a character.
            writer.writerows(
                [
                    [0, "red_hair", 0],
                    [1, "miku_(vocaloid)", 4],
                    [2, "safe", 9],
                    [3, "blue_eyes", 0],
                ]
            )
        self.mapping_path = self.root / "tag_mapping.json"
        self.mapping = {
            "idx_to_tag": {
                "0": "safe",
                "1": "explicit",
                "2": "1girl",
                "3": "miku",
                "4": "vocaloid",
                "5": "artist_a",
                "6": "commentary",
                "7": "best_quality",
                "8": "low_quality",
                "9": "anima",
            },
            "tag_to_category": {
                "safe": "Rating",
                "explicit": "Rating",
                "1girl": "General",
                "miku": "Character",
                "vocaloid": "Copyright",
                "artist_a": "Artist",
                "commentary": "Meta",
                "best_quality": "Quality",
                "low_quality": "Quality",
                "anima": "Model",
            },
        }
        self.mapping_path.write_text(json.dumps(self.mapping))
        self.image = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        self.image.putpixel((0, 0), (0, 255, 0, 0))

    def loaded(self, cls, name, output):
        tagger = cls(name, MODELS[name], self.root, ["CPUExecutionProvider"])
        path = self.mapping_path if MODELS[name]["kind"] == "cl" else self.csv_path
        session = FakeSession(MODELS[name]["kind"], output)
        with (
            patch.object(
                tagger, "download", return_value=(self.root / "model.onnx", path)
            ),
            patch(
                "extras.wd14_tagger.interrogator.ort.InferenceSession",
                return_value=session,
            ),
        ):
            tagger.load()
        return tagger, session

    def test_all_17_models_dispatch_to_their_own_preprocessing_pipeline(self):
        manager = wd14tagger.TaggerManager()
        self.addCleanup(manager.unload)
        downloads = []

        def download(tagger):
            downloads.append((tagger.name, tagger.spec["revision"]))
            return (
                self.root / tagger.name / "model.onnx",
                self.mapping_path if tagger.spec["kind"] == "cl" else self.csv_path,
            )

        def session(path, **kwargs):
            kind = MODELS[Path(path).parent.name]["kind"]
            return FakeSession(
                kind, [1] * 10 if kind == "cl" else [0.8, 0.9, 0.99, 0.7]
            )

        with (
            patch("modules.config.path_clip_vision", str(self.root)),
            patch.object(Interrogator, "download", download),
            patch(
                "extras.wd14_tagger.interrogator.ort.InferenceSession",
                side_effect=session,
            ),
            patch.object(
                wd14tagger, "execution_providers", return_value=["CPUExecutionProvider"]
            ),
        ):
            for name in MODELS:
                with self.subTest(model=name):
                    _, tags, categories = manager.interrogate(self.image, name)
                    self.assertTrue(tags)
                    self.assertIn("character", categories.values())
                    shape = manager.tagger.model.inputs[0].shape
                    self.assertEqual(
                        shape,
                        (1, 3, 4, 6) if MODELS[name]["kind"] == "cl" else (1, 4, 4, 3),
                    )
        self.assertEqual(len(downloads), 17)
        self.assertEqual(MODELS["wd14-vit-v2"]["revision"], "v2.0")
        self.assertEqual(MODELS["wd14-vit-v2-git"]["revision"], "main")

    def test_wd_alpha_bgr_and_noncontiguous_categories(self):
        tagger, session = self.loaded(
            WaifuDiffusionInterrogator, "wd14-moat-v2", [0.8, 0.9, 0.99, 0.7]
        )
        ratings, tags = tagger.interrogate(self.image)
        tensor = session.inputs[0]
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tensor.shape, (1, 4, 4, 3))
        np.testing.assert_array_equal(tensor[0, 1, 1], [255, 255, 255])
        np.testing.assert_array_equal(tensor[0, 2, 2], [0, 0, 255])
        self.assertEqual(list(ratings), ["safe"])
        self.assertEqual(list(tags), ["red_hair", "miku_(vocaloid)", "blue_eyes"])
        self.assertEqual(tagger.categories()["miku_(vocaloid)"], "character")

    def test_wd_without_character_tags_and_bad_output(self):
        tagger, session = self.loaded(
            WaifuDiffusionInterrogator, "wd14-vit", [0.8, 0.9, 0.99, 0.7]
        )
        tagger.tags = [row for row in tagger.tags if row["category"] != "4"]
        session.output = np.array([[0.8, 0.99, 0.7]], dtype=np.float32)
        _, tags = tagger.interrogate(self.image)
        self.assertEqual(set(tags), {"red_hair", "blue_eyes"})
        session.output = np.array([[0.2]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "different lengths"):
            tagger.interrogate(self.image)

    def test_cl_layout_normalization_sigmoid_and_all_categories(self):
        tagger, session = self.loaded(
            CLTaggerInterrogator, "cl_tagger_1_01", [2, -2, 1, 3, 2, 2, 2, 4, -4, 2]
        )
        ratings, tags = tagger.interrogate(Image.new("RGB", (4, 4), (255, 0, 0)))
        tensor = session.inputs[0]
        self.assertEqual(tensor.shape, (1, 3, 4, 6))
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [-1, -1, 1])
        self.assertEqual(list(ratings), ["safe"])
        self.assertAlmostEqual(ratings["safe"], 1 / (1 + np.exp(-2)), places=6)
        self.assertIn("best_quality", tags)
        self.assertNotIn("low_quality", tags)
        self.assertEqual(
            set(tagger.categories().values()),
            {
                "rating",
                "general",
                "character",
                "copyright",
                "artist",
                "meta",
                "quality",
                "model",
            },
        )

    def test_cl_alternative_mapping_and_sparse_indices(self):
        alternative = {
            idx: {"tag": name, "category": self.mapping["tag_to_category"][name]}
            for idx, name in self.mapping["idx_to_tag"].items()
        }
        alternative["12"] = {"tag": "solo", "category": "General"}
        self.mapping_path.write_text(json.dumps(alternative))
        tagger = CLTaggerInterrogator(
            "cl_tagger_1_01", MODELS["cl_tagger_1_01"], self.root, []
        )
        tagger.load_labels(self.mapping_path)
        self.assertEqual(len(tagger.tags[0].names), 13)
        self.assertIsNone(tagger.tags[0].names[10])
        self.assertIn(12, tagger.tags[0].general)

    def test_cl_dynamic_dimensions_use_source_448px_default(self):
        tagger = CLTaggerInterrogator(
            "cl_tagger_1_01", MODELS["cl_tagger_1_01"], self.root, []
        )
        session = FakeSession("cl", [0] * 10)
        session.get_inputs = lambda: [
            SimpleNamespace(name="image", shape=["batch", 3, "height", "width"])
        ]
        with (
            patch.object(
                tagger,
                "download",
                return_value=(self.root / "model.onnx", self.mapping_path),
            ),
            patch(
                "extras.wd14_tagger.interrogator.ort.InferenceSession",
                return_value=session,
            ),
        ):
            tagger.load()
        tagger.interrogate(self.image)
        self.assertEqual(session.inputs[0].shape, (1, 3, 448, 448))

    def test_downloads_separate_versions_and_reuse_only_complete_legacy_pair(self):
        with patch(
            "extras.wd14_tagger.interrogator.load_file_from_url",
            return_value="/cache/file",
        ) as download:
            for name in ["wd14-vit-v2", "wd14-vit-v2-git", "cl_tagger_1_01"]:
                Interrogator(name, MODELS[name], self.root, []).download()
            calls = download.call_args_list
            self.assertIn("/resolve/v2.0/", calls[0].kwargs["url"])
            self.assertIn("/resolve/main/", calls[2].kwargs["url"])
            self.assertNotEqual(
                calls[0].kwargs["model_dir"], calls[2].kwargs["model_dir"]
            )
            self.assertIn("cl_tagger_1_01/tag_mapping.json", calls[5].kwargs["url"])
            self.assertEqual(calls[5].kwargs["file_name"], "tag_mapping.json")
            tagger = Interrogator("wd14-moat-v2", MODELS["wd14-moat-v2"], self.root, [])
            legacy = [
                self.root / f"wd-v1-4-moat-tagger-v2.{ext}" for ext in ["onnx", "csv"]
            ]
            legacy[0].touch()
            download.reset_mock()
            tagger.download()
            self.assertEqual(download.call_count, 2)
            legacy[1].touch()
            download.reset_mock()
            self.assertEqual(tagger.download(), legacy)
            download.assert_not_called()

    def test_model_switch_cache_unload_failure_and_device_changes(self):
        manager = wd14tagger.TaggerManager()
        self.addCleanup(manager.unload)
        tagger = Mock()
        tagger.interrogate.return_value = ({}, {"tag": 1.0})
        tagger.categories.return_value = {"tag": "general"}
        with (
            patch(
                "extras.wd14_tagger.wd14.WaifuDiffusionInterrogator",
                return_value=tagger,
            ) as cls,
            patch.object(
                wd14tagger,
                "execution_providers",
                side_effect=lambda cpu: (
                    ["CPUExecutionProvider"]
                    if cpu
                    else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ),
            ),
        ):
            manager.interrogate(self.image, "wd14-moat-v2")
            manager.interrogate(self.image, "wd14-moat-v2")
            self.assertEqual(cls.call_count, 1)
            manager.interrogate(self.image, "wd14-moat-v2", use_cpu=True)
            self.assertEqual(cls.call_count, 2)
            tagger.unload.assert_called_once()
            manager.interrogate(
                self.image, "wd14-vit-v2", use_cpu=True, unload_model_after_running=True
            )
            self.assertIsNone(manager.tagger)
            tagger.interrogate.side_effect = RuntimeError("bad model")
            with self.assertRaisesRegex(RuntimeError, "bad model"):
                manager.interrogate(self.image, "wd14-moat-v2")
            self.assertIsNone(manager.tagger)
        with self.assertRaisesRegex(ValueError, "Unknown tagger"):
            manager.interrogate(self.image, "not-a-model")


class TestTaggerOptions(unittest.TestCase):
    def setUp(self):
        self.ratings = {"safe": 0.9}
        self.tags = {
            "z_tag": 0.8,
            "a_tag": 0.6,
            "character_a": 0.8,
            "character_b": 0.9,
            "series_a": 0.7,
            "^_^": 0.5,
        }
        self.categories = {name: "general" for name in self.tags} | {
            "character_a": "character",
            "character_b": "character",
            "series_a": "copyright",
        }
        patcher = patch.object(
            wd14tagger.manager,
            "interrogate",
            return_value=(self.ratings, self.tags, self.categories),
        )
        self.infer = patcher.start()
        self.addCleanup(patcher.stop)

    def test_independent_thresholds_category_filter_and_confidence_order(self):
        text, ratings, tags = wd14tagger.interrogate_image(None)
        self.assertEqual(text, "character b, z tag, series a, a tag, ^_^")
        self.assertEqual(ratings, self.ratings)
        self.assertNotIn("character a", tags)
        text, _, _ = wd14tagger.interrogate_image(
            None, categories=["general"], threshold=0.8
        )
        self.assertEqual(text, "z tag")
        text, _, _ = wd14tagger.interrogate_image(None, categories=[])
        self.assertEqual(text, "")

    def test_add_exclude_original_names_and_format_options(self):
        text, _, tags = wd14tagger.interrogate_image(
            None,
            additional_tags=" extra_tag, character_a, , skip_me ",
            exclude_tags="skip_me,a_tag",
            sort_by_alphabetical_order=True,
            character_threshold=0.95,
            replace_underscore=False,
        )
        self.assertEqual(text, "^_^, character_a, extra_tag, series_a, z_tag")
        self.assertEqual(tags["character_a"], 1.0)
        self.assertEqual(self.tags["character_a"], 0.8)
        text, _, _ = wd14tagger.interrogate_image(
            None,
            additional_tags=r"tag_(x)\y",
            categories=[],
            add_confident_as_weight=True,
        )
        self.assertEqual(text, r"(tag \(x\)\\y:1.0)")

    def test_backwards_string_api_and_runtime_options_forwarded(self):
        text = wd14tagger.default_interrogator(
            None,
            0.9,
            0.8,
            "character_a",
            model_name="wd-vit-v3",
            use_cpu=True,
            unload_model_after_running=True,
        )
        self.assertEqual(text, "character b")
        self.infer.assert_called_with(None, "wd-vit-v3", True, True)

    def test_manual_description_styles_and_confidence(self):
        options = wd14tagger.DEFAULTS | {
            "model_name": "cl_tagger_1_01",
            "show_confidence": True,
            "threshold": 0.7,
        }
        text, styles, ratings, tags = describe_image(
            [flags.describe_type_anime],
            np.zeros((8, 8, 3), dtype=np.uint8),
            True,
            *options.values(),
        )
        self.assertIn("z tag", text)
        self.assertEqual(styles, ["Fooocus Masterpiece", "Fooocus V2"])
        self.assertEqual(ratings, self.ratings)
        self.assertIn("z tag", tags)
        self.infer.assert_called_with(unittest.mock.ANY, "cl_tagger_1_01", False, False)
        _, styles, ratings, tags = describe_image(
            [flags.describe_type_anime], np.zeros((8, 8, 3), dtype=np.uint8), False
        )
        self.assertEqual(styles, {"__type__": "update"})
        self.assertIsNone(ratings)
        self.assertIsNone(tags)

    def test_empty_image_reports_actionable_error(self):
        import gradio as gr

        with self.assertRaises(gr.Error):
            describe_image([flags.describe_type_anime], None, False)
        self.infer.assert_not_called()

    def test_cpu_flag_and_installed_providers(self):
        import args_manager

        with (
            patch.object(args_manager.args, "always_cpu", None),
            patch.object(args_manager.args, "gpu_device_id", None),
            patch.object(
                wd14tagger.ort,
                "get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
        ):
            self.assertEqual(
                wd14tagger.execution_providers(False),
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.assertEqual(
                wd14tagger.execution_providers(True), ["CPUExecutionProvider"]
            )
            with patch.object(args_manager.args, "gpu_device_id", 2):
                self.assertEqual(
                    wd14tagger.execution_providers(False)[0],
                    ("CUDAExecutionProvider", {"device_id": 2}),
                )
            with patch.object(args_manager.args, "always_cpu", -1):
                self.assertEqual(
                    wd14tagger.execution_providers(False), ["CPUExecutionProvider"]
                )

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from extras import expansion
from ldm_patched.modules import model_management
from ldm_patched.modules.model_base import SDXL
from modules import core


class TestLazyPipeline(unittest.TestCase):
    def setUp(self):
        self.load_model = self.start_patch(core, "load_model")
        self.expansion = self.start_patch(expansion, "FooocusExpansion")
        self.load_gpu = self.start_patch(model_management, "load_models_gpu")
        path = Path(__file__).resolve().parents[1] / "modules/default_pipeline.py"
        spec = importlib.util.spec_from_file_location("lazy_pipeline_under_test", path)
        self.pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pipeline)

    def start_patch(self, target, name):
        patcher = patch.object(target, name)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def model(self, filename, vae_filename=None):
        unet = SimpleNamespace(model=SDXL.__new__(SDXL))
        clip = SimpleNamespace(patcher=object(), fcs_cond_cache={"old": "cached"})
        return SimpleNamespace(
            filename=filename,
            vae_filename=vae_filename,
            unet=unet,
            unet_with_lora=unet,
            clip=clip,
            clip_with_lora=clip,
            vae=object(),
            refresh_loras=Mock(),
        )

    def refresh(self, **options):
        with patch.object(self.pipeline, "is_anima_file", return_value=False):
            self.pipeline.refresh_everything(
                refiner_model_name="None",
                base_model_name="selected.safetensors",
                loras=[("selected-lora.safetensors", 0.75)],
                **options,
            )

    def test_import_does_not_load_any_weights(self):
        self.load_model.assert_not_called()
        self.expansion.assert_not_called()
        self.load_gpu.assert_not_called()
        self.assertIsNone(self.pipeline.model_base.unet)
        self.assertIsNone(self.pipeline.model_refiner.unet)
        self.assertIsNone(self.pipeline.final_clip)
        self.assertIsNone(self.pipeline.final_expansion)
        self.assertEqual(self.pipeline.loaded_ControlNets, {})

    def test_encoder_warmup_before_generation_leaves_weights_unloaded(self):
        self.pipeline.prepare_text_encoder()
        self.load_model.assert_not_called()
        self.expansion.assert_not_called()
        self.load_gpu.assert_not_called()

    def test_first_task_loads_selected_model_vae_and_loras(self):
        self.load_model.side_effect = self.model
        self.refresh(vae_name="selected-vae.safetensors")
        filename, vae_filename = self.load_model.call_args.args
        self.assertTrue(filename.endswith("selected.safetensors"))
        self.assertTrue(vae_filename.endswith("selected-vae.safetensors"))
        self.pipeline.model_base.refresh_loras.assert_called_once_with(
            [("selected-lora.safetensors", 0.75)]
        )
        self.expansion.assert_called_once_with()
        self.load_gpu.assert_called_once_with(
            [self.pipeline.final_clip.patcher, self.expansion.return_value.patcher]
        )
        self.assertEqual(self.pipeline.final_clip.fcs_cond_cache, {})

    def test_repeated_tasks_reuse_loaded_model_and_expansion(self):
        self.load_model.side_effect = self.model
        self.refresh()
        first_unet = self.pipeline.final_unet
        self.refresh()
        self.assertIs(self.pipeline.final_unet, first_unet)
        self.load_model.assert_called_once()
        self.expansion.assert_called_once()
        self.assertEqual(self.load_gpu.call_count, 2)

    def test_first_task_also_loads_selected_refiner(self):
        self.load_model.side_effect = self.model
        with patch.object(self.pipeline, "is_anima_file", return_value=False):
            self.pipeline.refresh_everything(
                refiner_model_name="selected-refiner.safetensors",
                base_model_name="selected.safetensors",
                loras=[],
            )
        self.assertEqual(self.load_model.call_count, 2)
        filenames = [call.args[0] for call in self.load_model.call_args_list]
        self.assertTrue(filenames[0].endswith("selected-refiner.safetensors"))
        self.assertTrue(filenames[1].endswith("selected.safetensors"))
        self.assertIs(
            self.pipeline.final_refiner_unet, self.pipeline.model_refiner.unet_with_lora
        )

    def test_failed_first_load_can_be_retried(self):
        def load(filename, vae_filename=None):
            if self.load_model.call_count == 1:
                raise RuntimeError("load failed")
            return self.model(filename, vae_filename)

        self.load_model.side_effect = load
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            self.refresh()
        self.pipeline.prepare_text_encoder()
        self.load_gpu.assert_not_called()
        self.refresh()
        self.assertIsNotNone(self.pipeline.final_unet)
        self.assertEqual(self.load_model.call_count, 2)
        self.expansion.assert_called_once()


if __name__ == "__main__":
    unittest.main()

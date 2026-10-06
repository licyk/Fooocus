import tempfile
import unittest
from pathlib import Path

import torch
from transformers import CLIPConfig, CLIPTextConfig, CLIPTextModel, CLIPVisionConfig

from extras.safety_checker.models.safety_checker import StableDiffusionSafetyChecker
from ldm_patched.modules.lora import model_lora_keys_clip
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.sd1_clip import SDClipModel
from modules.patch_clip import patch_all_clip


class TestCLIPModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patch_all_clip()

    def setUp(self):
        self.config = CLIPTextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=8,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        self.reference = CLIPTextModel(self.config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            self.config.to_json_file(path)
            self.model = SDClipModel(
                textmodel_json_config=str(path), max_length=8,
                special_tokens={'start': 1, 'end': 31, 'pad': 0},
            )
        self.weights = {'text_model.' + key: value for key, value in self.reference.state_dict().items()}
        with torch.no_grad():
            missing, unexpected = self.model.load_sd(dict(self.weights))
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.ids = torch.tensor([[1, 4, 5, 31, 0, 0, 0, 0]])

    def test_sdxl_checkpoint_paths_and_hidden_outputs(self):
        self.assertEqual(set(self.model.transformer.state_dict()), set(self.weights))
        with torch.no_grad():
            expected = self.reference(self.ids, output_hidden_states=True)
            actual, pooled = self.model(self.ids.tolist())
            torch.testing.assert_close(actual, expected.last_hidden_state)
            torch.testing.assert_close(pooled, expected.pooler_output)
            self.model.clip_layer(-2)
            actual, _ = self.model(self.ids.tolist())
            normalized = self.reference.final_layer_norm(expected.hidden_states[-2])
            torch.testing.assert_close(actual, normalized)

    def test_textual_inversion_restores_embeddings_and_masks_padding(self):
        original = self.model.transformer.get_input_embeddings()
        self.model.enable_attention_masks = True
        vector = torch.randn(16)
        with torch.no_grad():
            actual, pooled = self.model([[1, vector, 5, 31, 0, 0, 0, 0]])
        self.assertEqual(actual.shape, (1, 8, 16))
        self.assertEqual(pooled.shape, (1, 16))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertIs(self.model.transformer.get_input_embeddings(), original)

    def test_sdxl_lora_mapping_and_patch_restoration(self):
        sdxl = torch.nn.Module()
        sdxl.clip_l = self.model
        sdxl.clip_g = self.model
        keys = model_lora_keys_clip(sdxl, {})
        for encoder, prefix in [('clip_l', 'lora_te1'), ('clip_g', 'lora_te2')]:
            key = f'{encoder}.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight'
            self.assertEqual(keys[f'{prefix}_text_model_encoder_layers_0_self_attn_q_proj'], key)
        key = keys['lora_te1_text_model_encoder_layers_0_self_attn_q_proj']
        original = sdxl.state_dict()[key].clone()
        up, down = torch.randn(16, 2), torch.randn(2, 16)
        patcher = ModelPatcher(sdxl, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
        self.assertEqual(patcher.add_patches({key: ('lora', (up, down, None, None))}), [key])
        patcher.patch_model()
        torch.testing.assert_close(sdxl.state_dict()[key], original + up @ down)
        patcher.unpatch_model()
        torch.testing.assert_close(sdxl.state_dict()[key], original)

    def test_safety_checker_checkpoint_preserves_vision_paths_and_outputs(self):
        vision = CLIPVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, image_size=32, patch_size=16,
        )
        config = CLIPConfig(text_config=self.config.to_dict(), vision_config=vision.to_dict(), projection_dim=8)
        model = StableDiffusionSafetyChecker(config).eval()
        self.assertIn('vision_model.vision_model.embeddings.patch_embedding.weight', model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = StableDiffusionSafetyChecker.from_pretrained(directory).eval()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key])
        pixels = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model.vision_model(pixels).pooler_output,
                                       restored.vision_model(pixels).pooler_output)

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3Model

from ldm_patched.ldm.anima.model import Anima
from ldm_patched.ldm.anima.operations import AnimaOperations, apply_rope
from ldm_patched.ldm.anima.sampling import sample_euler_ancestral_flow
from ldm_patched.modules.anima import (
    AnimaLatentFormat,
    AnimaModel,
    AnimaModelConfig,
    AnimaTextEncoder,
    AnimaTokenizer,
    AnimaVAE,
    detect_anima_config,
    detect_qwen_config,
)
from ldm_patched.modules.lora import model_lora_keys_clip, model_lora_keys_unet
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.model_sampling import EPS, ModelSamplingDiscreteFlow
from modules import anima, flags
from modules.lora import match_lora


def tiny_config():
    return dict(
        image_model="anima",
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        model_channels=64,
        num_blocks=1,
        num_heads=2,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        crossattn_emb_channels=32,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False,
        adapter_config=dict(
            source_dim=64, target_dim=32, model_dim=32, num_layers=1, num_heads=2
        ),
        dtype=torch.float32,
    )


class TestAnima(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AnimaTokenizer()
        cls.model = AnimaModel(AnimaModelConfig(tiny_config())).eval()
        with torch.no_grad():
            for name, parameter in cls.model.named_parameters():
                if "norm" in name and parameter.ndim == 1:
                    parameter.fill_(1)
                else:
                    torch.nn.init.normal_(parameter, std=0.02)

    def test_detection_uses_checkpoint_structure_instead_of_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "animaPencilXL_v500.safetensors"
            save_file(
                {"model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1)},
                filename,
            )
            self.assertFalse(anima.is_anima_file(filename))
            for prefix in anima.PREFIXES:
                with self.subTest(prefix=prefix):
                    save_file(
                        {
                            prefix + key: torch.zeros(1)
                            for key in (anima.SIGNATURE, "x_embedder.proj.1.weight")
                        },
                        filename,
                    )
                    self.assertTrue(anima.is_anima_file(filename))

    def test_detected_dit_config_strictly_loads_native_weights(self):
        weights = self.model.diffusion_model.state_dict()
        detected = detect_anima_config(weights)
        detected["dtype"] = torch.float32
        restored = Anima(**detected, operations=AnimaOperations)
        restored.load_state_dict(weights, strict=True)
        self.assertEqual(detected["adapter_config"], tiny_config()["adapter_config"])
        self.assertEqual(detected["num_heads"], 2)

    def test_rope_rotates_head_halves_with_sequence_head_layout(self):
        x = torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4)
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        frequencies = rotation.expand(1, 2, 1, 2, 2, 2)
        actual = apply_rope(x, frequencies)
        torch.testing.assert_close(actual, torch.cat((-x[..., 2:], x[..., :2]), dim=-1))

    def test_four_dimensional_dit_preserves_odd_latent_dimensions(self):
        model = self.model.diffusion_model
        latent = torch.randn(1, 16, 5, 7)
        context = torch.randn(1, 8, 32)
        with torch.no_grad():
            image = model(latent, torch.tensor([0.6]), context)
            video = model(latent.unsqueeze(2), torch.tensor([0.6]), context)
        self.assertEqual(image.shape, latent.shape)
        self.assertTrue(torch.isfinite(image).all())
        torch.testing.assert_close(image, video.squeeze(2))

    def test_adapter_weights_apply_before_zero_padding(self):
        model = self.model.diffusion_model
        context, ids = torch.randn(1, 3, 64), torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            plain = model.preprocess_text_embeds(context, ids)
            weighted = model.preprocess_text_embeds(
                context, ids, t5xxl_weights=torch.tensor([[[1.0], [2.0], [0.5]]])
            )
        self.assertEqual(plain.shape, (1, 512, 32))
        torch.testing.assert_close(weighted[:, 1], plain[:, 1] * 2)
        torch.testing.assert_close(weighted[:, 2], plain[:, 2] * 0.5)
        self.assertEqual(weighted[:, 3:].count_nonzero().item(), 0)

    def test_tokenization_weights_unicode_and_untruncated_prompts(self):
        plain = self.tokenizer.tokenize_with_weights("cat")
        weighted = self.tokenizer.tokenize_with_weights("(cat:2)")
        self.assertEqual(plain["qwen3_06b"], weighted["qwen3_06b"])
        self.assertEqual(
            weighted["t5xxl"][:-1], [(token, 2.0) for token, _ in plain["t5xxl"][:-1]]
        )
        self.assertEqual(weighted["t5xxl"][-1], (1, 1.0))
        self.assertEqual(
            self.tokenizer.tokenize_with_weights("")["qwen3_06b"], [151643]
        )
        tokens = self.tokenizer.tokenize_with_weights("café, 猫\n" + "word " * 200)
        self.assertGreater(len(tokens["qwen3_06b"]), 77)
        self.assertGreater(len(tokens["t5xxl"]), 77)

    def test_qwen_detection_preserves_native_checkpoint_and_outputs(self):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
        )
        reference = Qwen3Model(config).eval()
        weights = {
            "model." + key: value for key, value in reference.state_dict().items()
        }
        restored = AnimaTextEncoder(detect_qwen_config(weights)).eval()
        restored.load_state_dict(weights, strict=True)
        ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            torch.testing.assert_close(restored(ids), reference(ids).last_hidden_state)

    def test_flow_shift_initial_noise_and_partial_denoise(self):
        sampling = ModelSamplingDiscreteFlow()
        self.assertEqual(sampling.sigma_max.item(), 1.0)
        self.assertAlmostEqual(sampling.sigma(torch.tensor(0.5)).item(), 0.75)
        noise, latent = torch.randn(1, 16, 3, 3), torch.randn(1, 16, 3, 3)
        torch.testing.assert_close(
            sampling.noise_scaling(1.0, noise, latent, True), noise
        )
        torch.testing.assert_close(
            sampling.noise_scaling(0.6, noise, latent), 0.6 * noise + 0.4 * latent
        )
        self.assertIs(sampling.calculate_input(torch.tensor([0.6]), noise), noise)
        torch.testing.assert_close(
            sampling.calculate_denoised(torch.tensor([0.6]), latent, noise),
            noise - 0.6 * latent,
        )

    def test_latent_normalization_roundtrip_and_vae_family_validation(self):
        latent_format = AnimaLatentFormat()
        latent = torch.randn(2, 16, 4, 6)
        torch.testing.assert_close(
            latent_format.process_out(latent_format.process_in(latent)), latent
        )
        torch.testing.assert_close(
            latent_format.process_in(latent_format.mean), torch.zeros(1, 16, 1, 1)
        )
        with self.assertRaisesRegex(ValueError, "16-channel"):
            AnimaVAE(
                {
                    "decoder.head.0.gamma": torch.zeros(96),
                    "conv2.weight": torch.zeros(1, 48),
                }
            )

    def test_sdxl_noise_scaling_preserves_legacy_formula(self):
        sampling = EPS()
        sampling.sigma_data = 1.0
        noise, latent = torch.randn(1, 4, 3, 3), torch.randn(1, 4, 3, 3)
        for full_denoise in (False, True):
            expected = noise * ((1 + 0.6**2) ** 0.5 if full_denoise else 0.6) + latent
            torch.testing.assert_close(
                sampling.noise_scaling(torch.tensor(0.6), noise, latent, full_denoise),
                expected,
            )

    def test_flow_ancestral_seed_and_euler_limit(self):
        from ldm_patched.k_diffusion.sampling import sample_euler

        def denoiser(x, sigma, **kwargs):
            return x - sigma[:, None, None, None] * torch.tanh(x)

        noise = torch.randn(1, 16, 3, 3)
        sigmas = torch.tensor([1.0, 0.75, 0.4, 0.1, 0.0])
        first = sample_euler_ancestral_flow(
            denoiser, noise.clone(), sigmas, extra_args={"seed": 42}, disable=True
        )
        repeated = sample_euler_ancestral_flow(
            denoiser, noise.clone(), sigmas, extra_args={"seed": 42}, disable=True
        )
        other = sample_euler_ancestral_flow(
            denoiser, noise.clone(), sigmas, extra_args={"seed": 43}, disable=True
        )
        torch.testing.assert_close(first, repeated)
        self.assertFalse(torch.allclose(first, other))
        deterministic = sample_euler_ancestral_flow(
            denoiser, noise.clone(), sigmas, eta=0, disable=True
        )
        torch.testing.assert_close(
            deterministic, sample_euler(denoiser, noise.clone(), sigmas, disable=True)
        )

    def test_native_and_peft_lora_keys_patch_and_restore(self):
        mapping = model_lora_keys_unet(self.model, {})
        key = "diffusion_model.blocks.0.self_attn.q_proj.weight"
        original = self.model.state_dict()[key].clone()
        up, down = torch.randn(64, 2), torch.randn(2, 64)
        for name in (
            "lora_unet_blocks_0_self_attn_q_proj",
            "net.blocks.0.self_attn.q_proj",
            "diffusion_model.blocks.0.self_attn.q_proj",
        ):
            with self.subTest(name=name):
                patches, remaining = match_lora(
                    {
                        name + ".lora_B.weight": up,
                        name + ".lora_A.weight": down,
                        name + ".alpha": torch.tensor(2.0),
                    },
                    mapping,
                )
                self.assertEqual(remaining, {})
                patcher = ModelPatcher(
                    self.model, torch.device("cpu"), torch.device("cpu")
                )
                patcher.add_patches(patches, 0.5)
                patcher.patch_model()
                torch.testing.assert_close(
                    self.model.state_dict()[key], original + 0.5 * up @ down
                )
                patcher.unpatch_model()
                torch.testing.assert_close(self.model.state_dict()[key], original)

    def test_vae_memory_fallback_uses_tiles_and_preserves_other_errors(self):
        vae = AnimaVAE.__new__(AnimaVAE)
        pixels, latent = torch.zeros(1, 64, 64, 3), torch.zeros(1, 16, 8, 8)
        vae._encode = Mock(side_effect=torch.OutOfMemoryError("fixture"))
        vae._decode = Mock(side_effect=torch.OutOfMemoryError("fixture"))
        vae.encode_tiled = Mock(return_value=latent)
        vae.decode_tiled = Mock(return_value=pixels)
        with patch(
            "ldm_patched.modules.anima.model_management.soft_empty_cache"
        ) as clear:
            self.assertIs(vae.encode(pixels), latent)
            self.assertIs(vae.decode(latent), pixels)
            self.assertEqual(clear.call_count, 2)
        vae.encode_tiled.assert_called_once_with(
            pixels, tile_x=256, tile_y=256, overlap=32
        )
        vae.decode_tiled.assert_called_once_with(
            latent, tile_x=32, tile_y=32, overlap=8
        )
        vae._encode = Mock(side_effect=ValueError("invalid pixels"))
        with self.assertRaisesRegex(ValueError, "invalid pixels"):
            vae.encode(pixels)

    def test_qwen_lora_aliases(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(32, 16)
        keys = model_lora_keys_clip(model, {})
        self.assertEqual(
            keys["lora_te_model_embed_tokens"], "model.embed_tokens.weight"
        )
        self.assertEqual(
            keys["text_encoder.model.embed_tokens"], "model.embed_tokens.weight"
        )

    def test_task_disables_sdxl_components_and_rejects_active_controls(self):
        task = SimpleNamespace(
            base_model_name="anima.safetensors",
            performance_selection=flags.Performance.QUALITY,
            sampler_name="euler",
            scheduler_name="simple",
            input_image_checkbox=False,
            current_tab="ip",
            mixing_image_prompt_and_vary_upscale=False,
            mixing_image_prompt_and_inpaint=False,
            cn_tasks={"ip": ["old hidden image"]},
            refiner_model_name="sdxl.safetensors",
            refiner_swap_method="separate",
            freeu_enabled=True,
            inpaint_engine="v2.6",
        )
        with patch.object(anima, "is_anima_file", return_value=True):
            self.assertTrue(anima.prepare_task(task))
            self.assertEqual(task.refiner_model_name, "None")
            self.assertEqual(task.refiner_swap_method, "joint")
            self.assertFalse(task.freeu_enabled)
            self.assertEqual(task.inpaint_engine, "None")
            task.input_image_checkbox = True
            with self.assertRaisesRegex(ValueError, "Image Prompt"):
                anima.prepare_task(task)
            task.performance_selection = flags.Performance.LIGHTNING
            with self.assertRaisesRegex(ValueError, "Speed or Quality"):
                anima.prepare_task(task)

    def test_rejects_sdxl_schedules_and_legacy_unipc(self):
        for sampler, scheduler in [
            ("uni_pc", "simple"),
            ("uni_pc_bh2", "simple"),
            ("euler", "turbo"),
            ("euler", "align_your_steps"),
            ("lcm", "simple"),
            ("euler", "tcd"),
        ]:
            with (
                self.subTest(sampler=sampler, scheduler=scheduler),
                self.assertRaises(ValueError),
            ):
                anima.validate_sampling(sampler, scheduler)
        anima.validate_sampling("euler", "simple")
        anima.validate_sampling("dpmpp_2m", "karras")

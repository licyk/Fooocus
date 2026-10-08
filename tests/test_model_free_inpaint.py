import ast
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from extras.inpaint_masking import expand_crop_region, fill, get_crop_region_v2
from extras.soft_inpainting import (
    SoftInpaintingSettings,
    adaptive_mask,
    get_gaussian_kernel,
    get_modified_nmask,
    latent_blend,
    weighted_histogram_filter,
)
from ldm_patched.k_diffusion.sampling import sample_euler
from ldm_patched.modules.model_sampling import EPS, ModelSamplingDiscreteFlow
from ldm_patched.modules.samplers import KSAMPLER, UNIPC, UNIPCBH2, KSamplerX0Inpaint
from modules import inpaint_worker
from modules.meta_parser import (
    A1111MetadataParser,
    FooocusMetadataParser,
    load_parameter_button_click,
)
from modules.model_free_inpaint import (
    DEFAULTS,
    ModelFreeDenoiser,
    ModelFreeInpaintWorker,
    parameterized_enabled,
    prepare_mask,
    read_settings,
)
from modules.patch import (
    PatchSettings,
    patch_settings,
    patched_KSamplerX0Inpaint_forward,
    patched_unet_forward,
)
from tests.test_deep_shrink import model_patcher


def worker_function(name, env):
    source = ast.parse(
        (Path(__file__).resolve().parents[1] / "modules/async_worker.py").read_text()
    )
    node = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(  # noqa: S102
        compile(ast.Module(body=[node], type_ignores=[]), "async_worker.py", "exec"),
        env,
    )
    return env[name]


def image_and_mask():
    image = np.indices((64, 96)).sum(axis=0).astype(np.uint8)
    image = np.repeat(image[:, :, None], 3, axis=2)
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[24:40, 40:56] = 255
    return image, mask


def make_task(**settings):
    image, mask = image_and_mask()
    settings = DEFAULTS | {"mask_blur": 0, "padding": 0} | settings
    return ModelFreeInpaintWorker(image, prepare_mask(mask, settings), 64, 64, settings)


class TestModelFreeMasking(unittest.TestCase):
    def test_crop_padding_clamps_and_expands_to_aspect_ratio(self):
        _, mask = image_and_mask()
        self.assertEqual(get_crop_region_v2(mask, 8), (32, 16, 64, 48))
        self.assertEqual(get_crop_region_v2(mask, 256), (0, 0, 96, 64))
        self.assertIsNone(get_crop_region_v2(np.zeros_like(mask)))
        self.assertEqual(
            expand_crop_region((40, 24, 56, 40), 128, 64, 96, 64), (32, 24, 64, 40)
        )

    def test_mask_modes_invert_before_blur_and_keep_gray_for_soft(self):
        mask = np.array([[0, 64, 128, 255]], dtype=np.uint8)
        settings = DEFAULTS | {"mask_blur": 0}
        np.testing.assert_array_equal(prepare_mask(mask, settings), [[0, 0, 0, 255]])
        np.testing.assert_array_equal(
            prepare_mask(mask, settings | {"mask_mode": "unmasked"}), [[255, 255, 0, 0]]
        )
        np.testing.assert_array_equal(
            prepare_mask(mask, settings | {"soft_enabled": True}), mask
        )
        np.testing.assert_array_equal(
            prepare_mask(
                mask,
                settings | {"soft_enabled": True, "mask_mode": "unmasked"},
                invert=True,
            ),
            mask,
        )
        _, mask = image_and_mask()
        blurred = prepare_mask(mask, DEFAULTS | {"soft_enabled": True})
        self.assertTrue(np.any((blurred > 0) & (blurred < 255)))

    def test_fill_uses_context_and_does_not_download_upscaler(self):
        image = np.full((32, 32, 3), 80, dtype=np.uint8)
        image[12:20, 12:20] = 230
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[12:20, 12:20] = 255
        result = np.array(
            fill(
                Image.fromarray(image),
                Image.fromarray(mask),
            )
        )
        self.assertLess(result[16, 16].mean(), 100)
        with patch(
            "modules.upscaler.perform_upscale",
            side_effect=AssertionError("must not upscale"),
        ):
            task = make_task(content="fill")
        self.assertEqual(task.interested_fill.shape, (64, 64, 3))

    def test_only_masked_preserves_exact_outside_pixels(self):
        task = make_task()
        result = task.post_process(np.full((64, 64, 3), 250, dtype=np.uint8))
        self.assertEqual(result.shape, task.image.shape)
        np.testing.assert_array_equal(
            result[task.mask == 0], task.image[task.mask == 0]
        )
        self.assertTrue(np.all(result[task.mask == 255] == 250))

    def test_whole_picture_uses_selected_resolution(self):
        task = make_task(area="whole")
        self.assertEqual(task.image.shape, (64, 64, 3))
        self.assertEqual(
            task.post_process(np.zeros_like(task.image)).shape, (64, 64, 3)
        )

    def test_empty_and_inverted_full_masks(self):
        image, mask = image_and_mask()
        empty = ModelFreeInpaintWorker(image, mask * 0, 64, 64, DEFAULTS)
        self.assertTrue(empty.empty)
        np.testing.assert_array_equal(
            empty.post_process(np.zeros((64, 64, 3), dtype=np.uint8)), image
        )
        mask = prepare_mask(
            mask * 0, DEFAULTS | {"mask_mode": "unmasked", "mask_blur": 0}
        )
        self.assertTrue(np.all(mask == 255))

    def test_outpaint_preserves_extended_canvas_even_with_whole_picture(self):
        image, mask = image_and_mask()
        image = np.pad(image, ((16, 0), (0, 0), (0, 0)), mode="edge")
        mask = np.pad(mask * 0, ((16, 0), (0, 0)), constant_values=255)
        task = ModelFreeInpaintWorker(
            image,
            mask,
            64,
            64,
            DEFAULTS | {"area": "whole", "mask_blur": 0},
            outpaint=True,
        )
        result = task.post_process(np.full((64, 64, 3), 200, dtype=np.uint8))
        self.assertEqual(result.shape, (80, 96, 3))
        np.testing.assert_array_equal(result[16:], image[16:])


class TestModelFreeLatents(unittest.TestCase):
    def test_four_content_modes_use_seeded_noise_and_channel_independent_masks(self):
        for channels in [4, 16]:
            for content in ["original", "fill", "latent_noise", "latent_nothing"]:
                with self.subTest(channels=channels, content=content):
                    task = make_task(area="whole", content=content)
                    base = torch.ones(1, channels, 8, 8)
                    task.load_latent(base)
                    initial = task.prepare_latent(42)["samples"].clone()
                    torch.testing.assert_close(
                        initial, task.prepare_latent(42)["samples"]
                    )
                    mask = task.latent_mask.expand_as(base)
                    torch.testing.assert_close(initial[mask == 0], base[mask == 0])
                    if content == "latent_noise":
                        self.assertFalse(
                            torch.equal(initial, task.prepare_latent(43)["samples"])
                        )
                    elif content == "latent_nothing":
                        self.assertTrue(torch.all(initial[mask == 1] == 0))
                    else:
                        torch.testing.assert_close(initial, base)
                    torch.testing.assert_close(base, torch.ones_like(base))

    def test_soft_keeps_fractional_mask_and_swap_restores_dimensions(self):
        task = make_task(area="whole", soft_enabled=True, mask_blur=4)
        task.load_latent(torch.ones(1, 4, 8, 8), latent_swap=torch.ones(1, 16, 8, 8))
        self.assertTrue(torch.any((task.latent_mask > 0) & (task.latent_mask < 1)))
        task.prepare_latent(42)
        task.swap()
        self.assertEqual(task.latent.shape[1], 16)
        task.unswap()
        self.assertEqual(task.latent.shape[1], 4)

    def test_noise_and_nothing_are_initialized_in_scaled_shifted_model_space(self):
        from ldm_patched.modules.anima import AnimaLatentFormat
        from ldm_patched.modules.latent_formats import SDXL, SDXL_Playground_2_5

        for format, channels in [
            (SDXL(), 4),
            (SDXL_Playground_2_5(), 4),
            (AnimaLatentFormat(), 16),
        ]:
            model = SimpleNamespace(
                process_latent_in=format.process_in,
                process_latent_out=format.process_out,
            )
            for content in ["latent_noise", "latent_nothing"]:
                task = make_task(content=content, area="whole")
                original = torch.ones(1, channels, 8, 8)
                task.load_latent(original, latent_model=model)
                raw = task.prepare_latent(42)["samples"]
                processed = format.process_in(raw)
                mask = task.latent_mask.expand_as(processed)
                expected = (
                    torch.randn(
                        original.shape, generator=torch.Generator().manual_seed(42)
                    )
                    if content == "latent_noise"
                    else torch.zeros_like(original)
                )
                torch.testing.assert_close(
                    processed[mask == 1], expected[mask == 1], atol=1e-6, rtol=1e-6
                )
                torch.testing.assert_close(
                    raw[mask == 0], original[mask == 0], atol=1e-6, rtol=1e-6
                )

    def test_sampler_protects_unmasked_input_and_output_in_eps_and_flow(self):
        for sampling, channels in [(EPS(), 4), (ModelSamplingDiscreteFlow(), 16)]:
            for soft in [False, True]:
                with self.subTest(sampling=type(sampling).__name__, soft=soft):
                    task = make_task(area="whole", soft_enabled=soft)
                    task.load_latent(torch.full((1, channels, 8, 8), 3.0))
                    task.prepare_latent(42)
                    task.latent_mask[:, :, :, :4] = 0
                    task.latent_mask[:, :, :, 4:] = 1
                    model = SimpleNamespace(
                        model_sampling=sampling, process_latent_in=lambda x: x
                    )
                    denoiser = Mock(return_value=torch.full_like(task.latent, 7.0))
                    denoiser.inner_model = model
                    sampler = SimpleNamespace(
                        inner_model=denoiser, noise=torch.full_like(task.latent, 2.0)
                    )
                    with patch.object(inpaint_worker, "current_task", task):
                        result = patched_KSamplerX0Inpaint_forward(
                            sampler,
                            torch.zeros_like(task.latent),
                            torch.tensor([0.6]),
                            None,
                            None,
                            1,
                            task.latent_mask,
                            seed=42,
                        )
                    expected = sampling.noise_scaling(
                        torch.tensor([[[[0.6]]]]), sampler.noise, task.latent
                    )
                    torch.testing.assert_close(
                        denoiser.call_args.args[0][:, :, :, :4], expected[:, :, :, :4]
                    )
                    torch.testing.assert_close(
                        result[:, :, :, :4], torch.full_like(result[:, :, :, :4], 3)
                    )
                    torch.testing.assert_close(
                        result[:, :, :, 4:], torch.full_like(result[:, :, :, 4:], 7)
                    )

    def test_noise_schedule_changes_soft_blending_without_losing_endpoints(self):
        settings = SoftInpaintingSettings()
        mask = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(get_modified_nmask(settings, mask, 2.0), mask)
        torch.testing.assert_close(
            get_modified_nmask(settings, mask, 0.5), mask.pow(0.25)
        )
        a = torch.tensor([[[[2.0]]], [[[4.0]]]])
        b = a * 2
        result = latent_blend(settings, a, b, torch.full_like(a, 0.5))
        self.assertTrue(torch.all(result > (a + b) / 2))
        self.assertTrue(torch.isfinite(result).all())

    def test_weighted_filter_matches_scalar_weighted_percentiles_at_edges(self):
        image = np.arange(35, dtype=np.float32).reshape(5, 7)
        kernel, center = get_gaussian_kernel()
        padded = np.pad(image, center)
        for low, high in [(0.9, 1), (0.25, 0.75)]:
            expected = np.empty_like(image)
            for y, x in np.ndindex(image.shape):
                values = padded[y : y + 5, x : x + 5].ravel()
                order = np.argsort(values)
                values, weights = values[order], kernel.ravel()[order]
                cumulative = np.cumsum(weights)
                lower, upper = cumulative[-1] * low, cumulative[-1] * high
                if upper - lower < 1:
                    midpoint = (lower + upper) / 2
                    lower, upper = (
                        max(0, midpoint - 0.5),
                        min(cumulative[-1], midpoint + 0.5),
                    )
                overlap = np.maximum(
                    0,
                    np.minimum(upper, cumulative)
                    - np.maximum(lower, np.r_[0, cumulative[:-1]]),
                )
                expected[y, x] = np.sum(values * overlap) / np.sum(overlap)
            np.testing.assert_allclose(
                weighted_histogram_filter(image, kernel, center, low, high),
                expected,
                rtol=1e-6,
            )

    def test_adaptive_mask_uses_latent_difference_and_handles_zero_parameters(self):
        original = torch.zeros(1, 16, 8, 8)
        nmask = torch.full((1, 1, 8, 8), 0.5)
        same = adaptive_mask(
            SoftInpaintingSettings(), nmask, original, original, (64, 64)
        )
        changed = adaptive_mask(
            SoftInpaintingSettings(), nmask, original, original + 4, (64, 64)
        )
        self.assertEqual(np.array(same).max(), 0)
        self.assertGreater(np.array(changed).mean(), 200)
        for influence in [0, 1]:
            settings = SoftInpaintingSettings(
                composite_difference_threshold=0, composite_mask_influence=influence
            )
            with np.errstate(all="raise"):
                result = adaptive_mask(
                    settings, nmask, original, original + 4, (64, 64)
                )
            self.assertEqual(result.size, (64, 64))

    def test_soft_final_composite_does_not_change_pixels_outside_requested_mask(self):
        task = make_task(soft_enabled=True)
        task.load_latent(torch.zeros(1, 16, 8, 8))
        task.prepare_latent(42)
        task.finish_sample(
            torch.ones(1, 16, 8, 8) * 4, SimpleNamespace(process_latent_in=lambda x: x)
        )
        result = task.post_process(np.full((64, 64, 3), 250, dtype=np.uint8))
        np.testing.assert_array_equal(
            result[task.mask == 0], task.image[task.mask == 0]
        )


class TestModelFreeIntegration(unittest.TestCase):
    def test_image_input_does_not_download_head_lora_or_upscaler_by_default(self):
        image, mask = image_and_mask()
        task = SimpleNamespace(
            current_tab="inpaint",
            inpaint_input_image={
                "image": image,
                "mask": np.repeat(mask[:, :, None], 3, axis=2),
            },
            inpaint_advanced_masking_checkbox=False,
            inpaint_erode_or_dilate=0,
            invert_mask_checkbox=False,
            inpaint_settings=DEFAULTS,
            outpaint_selections=[],
            mixing_image_prompt_and_inpaint=False,
            mixing_image_prompt_and_vary_upscale=False,
            inpaint_additional_prompt="",
            prompt="",
            refiner_model_name="None",
        )
        config = SimpleNamespace(
            downloading_upscale_model=Mock(), downloading_inpaint_models=Mock()
        )
        env = {
            "np": np,
            "modules": SimpleNamespace(config=config),
            "prepare_mask": prepare_mask,
            "HWC3": lambda x: x,
            "progressbar": Mock(),
        }
        apply = worker_function("apply_image_input", env)
        goals, loras = [], []
        apply(
            task,
            loras,
            None,
            None,
            None,
            goals,
            None,
            None,
            None,
            False,
            None,
            None,
            None,
            False,
            False,
        )
        self.assertEqual(goals, ["inpaint"])
        self.assertEqual(loras, [])
        config.downloading_upscale_model.assert_not_called()
        config.downloading_inpaint_models.assert_not_called()

    def test_empty_mask_and_zero_denoise_do_not_sample_or_change_image(self):
        for empty, strength in [(True, 1.0), (False, 0.0)]:
            task = make_task(area="masked")
            task.empty = empty
            task.load_latent(torch.ones(1, 4, 8, 8))
            async_task = SimpleNamespace(
                last_stop=False,
                black_out_nsfw=False,
                disable_intermediate_results=False,
            )
            pipeline = SimpleNamespace(
                process_diffusion=Mock(side_effect=AssertionError("must not sample"))
            )
            save = Mock(return_value=["test.png"])
            env = {
                "inpaint_worker": inpaint_worker,
                "pipeline": pipeline,
                "modules": SimpleNamespace(
                    config=SimpleNamespace(default_black_out_nsfw=False)
                ),
                "save_and_log": save,
                "yield_result": Mock(),
                "progressbar": Mock(),
            }
            process = worker_function("process_task", env)
            with patch.object(inpaint_worker, "current_task", task):
                process(
                    30,
                    async_task,
                    None,
                    None,
                    None,
                    0,
                    strength,
                    "normal",
                    ["inpaint"],
                    None,
                    30,
                    24,
                    [],
                    [],
                    {"task_seed": 42},
                    [],
                    False,
                    False,
                    64,
                    64,
                    10,
                    10,
                    1,
                    True,
                )
            pipeline.process_diffusion.assert_not_called()
            np.testing.assert_array_equal(save.call_args.args[2][0], task.image)

    def test_real_unet_sampling_and_unipc_use_masked_denoiser(self):
        host = model_patcher(real_unet=True)
        sampling = EPS()
        sampling.sigma_max = torch.tensor(1.0)
        sampling.sigma_data = 1.0
        host.model.model_sampling.noise_scaling = sampling.noise_scaling
        host.model.process_latent_in = lambda x: x
        task = make_task(area="whole", soft_enabled=True, mask_blur=4)
        base = torch.randn(1, 4, 8, 8)
        task.load_latent(base)
        latent = task.prepare_latent(42)["samples"]

        class Denoiser:
            inner_model = host.model

            def __call__(self, x, sigma, **kwargs):
                output = patched_unet_forward(
                    host.model.diffusion_model,
                    x,
                    torch.ones(x.shape[0]) * 500,
                    transformer_options={"sigmas": sigma},
                )
                return x - output * sigma[:, None, None, None]

        noise = torch.randn_like(base)
        kwargs = {
            "cond": [],
            "uncond": [],
            "cond_scale": 1.0,
            "model_options": {},
            "seed": 42,
        }
        with (
            patch.object(inpaint_worker, "current_task", task),
            patch.object(
                KSamplerX0Inpaint, "forward", patched_KSamplerX0Inpaint_forward
            ),
            patch.dict(patch_settings, {os.getpid(): PatchSettings()}),
            torch.no_grad(),
        ):
            for sampler in [KSAMPLER(sample_euler), UNIPC(), UNIPCBH2()]:
                wrapped = Denoiser()
                mask = task.latent_mask
                if not isinstance(sampler, KSAMPLER):
                    wrapped = ModelFreeDenoiser(task, wrapped, noise)
                    mask = None
                result = sampler.sample(
                    wrapped,
                    torch.tensor([0.8, 0.4, 0.1, 0.0]),
                    kwargs.copy(),
                    None,
                    noise,
                    latent,
                    mask,
                    True,
                )
                result = task.finish_sample(result, host.model)
                self.assertEqual(result.shape, latent.shape)
                self.assertTrue(torch.isfinite(result).all())
                torch.testing.assert_close(
                    result[task.latent_mask.expand_as(base) == 0],
                    base[task.latent_mask.expand_as(base) == 0],
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_worker_vae_path_never_uses_special_inpaint_encoder_or_models(self):
        for channels in [4, 16]:
            task = SimpleNamespace(
                outpaint_selections=[],
                inpaint_settings=DEFAULTS,
                debugging_inpaint_preprocessor=False,
                black_out_nsfw=False,
                steps=30,
                refiner_swap_method="joint",
            )
            identity = SimpleNamespace(
                process_latent_in=lambda x: x, process_latent_out=lambda x: x
            )
            pipeline = SimpleNamespace(
                get_candidate_vae=Mock(return_value=("base_vae", None)),
                final_vae="base_vae",
                final_refiner_vae=None,
                final_unet=SimpleNamespace(model=identity),
                final_refiner_unet=None,
            )
            core = SimpleNamespace(
                numpy_to_pytorch=lambda x: torch.from_numpy(x.copy())[None] / 255.0,
                encode_vae=Mock(
                    return_value={"samples": torch.ones(1, channels, 8, 8)}
                ),
                encode_vae_inpaint=Mock(
                    side_effect=AssertionError("no specialized encoder")
                ),
            )
            env = {
                "ModelFreeInpaintWorker": ModelFreeInpaintWorker,
                "inpaint_worker": inpaint_worker,
                "core": core,
                "pipeline": pipeline,
                "progressbar": Mock(),
            }
            apply = worker_function("apply_inpaint", env)
            image, mask = image_and_mask()
            with patch.object(inpaint_worker, "current_task", None):
                strength, latent, width, height, _ = apply(
                    task,
                    None,
                    None,
                    image,
                    mask,
                    False,
                    0.5,
                    0.618,
                    24,
                    False,
                    1,
                    True,
                    width=64,
                    height=64,
                )
                self.assertEqual(latent["samples"].shape, (1, channels, 8, 8))
                self.assertEqual((strength, width, height), (0.5, 64, 64))
                self.assertIn("noise_mask", latent)
                core.encode_vae.assert_called_once()
                core.encode_vae_inpaint.assert_not_called()

    def test_backend_default_and_parameterized_opt_in(self):
        self.assertFalse(parameterized_enabled(DEFAULTS, "v2.6"))
        self.assertTrue(
            parameterized_enabled(DEFAULTS | {"backend": "fooocus"}, "v2.6")
        )
        self.assertFalse(
            parameterized_enabled(DEFAULTS | {"backend": "fooocus"}, "None")
        )

    def test_metadata_roundtrip_and_malformed_defaults(self):
        settings = DEFAULTS | {
            "soft_enabled": True,
            "mask_blur": 20,
            "mask_mode": "unmasked",
            "content": "latent_noise",
            "padding": 64,
            "area": "whole",
        }
        self.assertEqual(read_settings(json.dumps(settings)), settings)
        for value in [
            None,
            [],
            "broken",
            {"mask_blur": -1},
            {"padding": 2.5},
            {"soft_enabled": "false"},
            {"backend": "invalid"},
            {"mask_blend_scale": float("nan")},
        ]:
            self.assertEqual(read_settings(value), DEFAULTS)
        results = load_parameter_button_click(
            {"inpaint_settings": json.dumps(settings), "inpaint_strength": 0.432},
            False,
            "Inpaint or Outpaint (default)",
        )
        # The complete settings appear together before the LoRA outputs.
        expected = list(settings.values()) + [0.432]
        self.assertTrue(
            any(
                results[i : i + len(expected)] == expected
                for i in range(len(results) - len(expected) + 1)
            )
        )

    def test_fooocus_and_a1111_image_metadata_formats(self):
        data = {
            "resolution": "(1024, 1024)",
            "sampler": "euler",
            "scheduler": "normal",
            "seed": "1",
            "guidance_scale": 7,
            "sharpness": 2,
            "adm_guidance": "(1.5, 0.8, 0.3)",
            "base_model": "model.safetensors",
            "performance": "Speed",
            "vae": "vae.safetensors",
            "version": "test",
            "inpaint_settings": json.dumps(DEFAULTS),
            "inpaint_strength": 0.432,
        }
        for parser_type in [FooocusMetadataParser, A1111MetadataParser]:
            parser = parser_type()
            parser.raw_prompt = "test"
            parser.full_prompt = ["test"]
            parser.full_negative_prompt = []
            text = parser.to_string([(key, key, value) for key, value in data.items()])
            metadata = parser.to_json(
                json.loads(text) if parser_type is FooocusMetadataParser else text
            )
            self.assertEqual(read_settings(metadata["inpaint_settings"]), DEFAULTS)
            self.assertAlmostEqual(float(metadata["inpaint_strength"]), 0.432)

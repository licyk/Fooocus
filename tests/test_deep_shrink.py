import ast
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from ldm_patched.contrib.external_freelunch import FreeU_V2
from ldm_patched.contrib.external_model_downscale import PatchModelAddDownscale
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm_patched.modules.anima import AnimaModel, AnimaModelConfig
from ldm_patched.modules.conds import CONDRegular
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.model_sampling import ModelSamplingDiscrete
from ldm_patched.modules.samplers import calc_cond_uncond_batch
from modules.deep_shrink import DEFAULTS, RESIZE_METHODS, apply_deep_shrink
from modules.meta_parser import (
    A1111MetadataParser,
    FooocusMetadataParser,
    get_deep_shrink,
)
from modules.patch import PatchSettings, patch_settings, patched_unet_forward
from modules.patch_precision import patched_register_schedule
from tests.test_anima import tiny_config


def model_patcher(real_unet=False):
    if real_unet:
        unet = UNetModel(
            image_size=32,
            in_channels=4,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            channel_mult=(1, 2),
            num_heads=1,
            transformer_depth=[0, 0],
            transformer_depth_output=[0, 0, 0, 0],
            transformer_depth_middle=-1,
        )
        for parameter in unet.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
    else:
        unet = UNetModel.__new__(UNetModel)
        torch.nn.Module.__init__(unet)
    model = torch.nn.Module()
    model.diffusion_model = unet
    with patch.object(
        ModelSamplingDiscrete, "_register_schedule", patched_register_schedule
    ):
        model.model_sampling = ModelSamplingDiscrete()
    model.model_config = SimpleNamespace(unet_config={"model_channels": 32})
    model.sentinel = torch.nn.Parameter(torch.zeros(1))
    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


def input_patch(model, after_skip=True):
    key = "input_block_patch_after_skip" if after_skip else "input_block_patch"
    return model.model_options["transformer_options"]["patches"][key][-1]


def output_patch(model):
    return model.model_options["transformer_options"]["patches"]["output_block_patch"][
        -1
    ]


class TestDeepShrinkPatch(unittest.TestCase):
    def setUp(self):
        self.model = model_patcher()
        self.settings = DEFAULTS | {"enabled": True}
        self.h = torch.randn(1, 8, 8, 12)

    def options(self, progress=0.1, block=3):
        sigma = self.model.model.model_sampling.percent_to_sigma(progress)
        return {"block": ("input", block), "sigmas": torch.tensor([sigma])}

    def test_defaults_match_upstream_node(self):
        schema = PatchModelAddDownscale.INPUT_TYPES()["required"]
        for key in [
            "block_number",
            "downscale_factor",
            "start_percent",
            "end_percent",
            "downscale_after_skip",
        ]:
            self.assertEqual(DEFAULTS[key], schema[key][1]["default"])
        self.assertEqual(DEFAULTS["downscale_method"], RESIZE_METHODS[0])
        self.assertFalse(DEFAULTS["enabled"])

    def test_sigma_window_and_block_selection(self):
        model = apply_deep_shrink(
            self.model, **(self.settings | {"start_percent": 0.1})
        )
        shrink = input_patch(model)
        for progress in [0.1, 0.2, 0.35]:
            self.assertEqual(shrink(self.h, self.options(progress)).shape[-2:], (4, 6))
        for progress, block in [(0.05, 3), (0.36, 3), (0.2, 2), (0.2, 32)]:
            self.assertIs(shrink(self.h, self.options(progress, block)), self.h)

    def test_before_and_after_skip_use_correct_hooks(self):
        for after_skip in [False, True]:
            model = apply_deep_shrink(
                self.model, **(self.settings | {"downscale_after_skip": after_skip})
            )
            self.assertEqual(
                input_patch(model, after_skip)(self.h, self.options()).shape[-2:],
                (4, 6),
            )
            other_key = (
                "input_block_patch" if after_skip else "input_block_patch_after_skip"
            )
            self.assertNotIn(
                other_key, model.model_options["transformer_options"]["patches"]
            )

    def test_all_interpolation_methods_preserve_device_dtype_and_skip(self):
        for method in RESIZE_METHODS:
            with self.subTest(method=method):
                model = apply_deep_shrink(
                    self.model,
                    **(
                        self.settings
                        | {"downscale_method": method, "upscale_method": method}
                    ),
                )
                h = input_patch(model)(self.h, self.options())
                restored, skip = output_patch(model)(h, self.h, {})
                self.assertEqual(h.shape[-2:], (4, 6))
                self.assertEqual(restored.shape, self.h.shape)
                self.assertEqual(restored.dtype, self.h.dtype)
                self.assertEqual(restored.device, self.h.device)
                self.assertIs(skip, self.h)
                self.assertTrue(torch.isfinite(restored).all())

    def test_output_restores_width_only_mismatch(self):
        model = apply_deep_shrink(self.model, **self.settings)
        restored, skip = output_patch(model)(self.h[:, :, :, :4], self.h, {})
        self.assertEqual(restored.shape, self.h.shape)
        self.assertIs(skip, self.h)
        same, skip = output_patch(model)(self.h, self.h, {})
        self.assertIs(same, self.h)

    def test_small_features_and_factors_below_one(self):
        model = apply_deep_shrink(
            self.model, **(self.settings | {"downscale_factor": 9.0})
        )
        self.assertEqual(
            input_patch(model)(self.h[:, :, :1, :1], self.options()).shape[-2:], (1, 1)
        )
        model = apply_deep_shrink(
            self.model, **(self.settings | {"downscale_factor": 0.5})
        )
        self.assertEqual(
            input_patch(model)(self.h, self.options()).shape[-2:], (16, 24)
        )

    def test_scheduler_object_patch_controls_sigma_thresholds(self):
        sampling = Mock()
        sampling.percent_to_sigma.side_effect = [10.0, 5.0]
        self.model.add_object_patch("model_sampling", sampling)
        model = apply_deep_shrink(
            self.model, **(self.settings | {"start_percent": 0.2, "end_percent": 0.6})
        )
        self.assertEqual(
            sampling.percent_to_sigma.call_args_list,
            [unittest.mock.call(0.2), unittest.mock.call(0.6)],
        )
        options = {"block": ("input", 3), "sigmas": torch.tensor([7.0])}
        self.assertEqual(input_patch(model)(self.h, options).shape[-2:], (4, 6))
        options["sigmas"] = torch.tensor([4.0])
        self.assertIs(input_patch(model)(self.h, options), self.h)

    def test_model_cache_is_unchanged_and_non_unet_is_skipped(self):
        self.assertIs(apply_deep_shrink(self.model, **DEFAULTS), self.model)
        patched = apply_deep_shrink(self.model, **self.settings)
        self.assertIsNot(patched, self.model)
        self.assertIs(patched.model, self.model.model)
        self.assertNotIn("patches", self.model.model_options["transformer_options"])
        self.assertIsNone(apply_deep_shrink(None, **self.settings))
        self.model.model.diffusion_model = torch.nn.Identity()
        with patch("builtins.print"):
            self.assertIs(apply_deep_shrink(self.model, **self.settings), self.model)

    def test_restored_sampling_object_is_used_when_no_override_exists(self):
        original_sampling = Mock()
        original_sampling.percent_to_sigma.side_effect = [10.0, 5.0]
        self.model.object_patches_backup["model_sampling"] = original_sampling
        self.model.model.model_sampling.percent_to_sigma = Mock()
        model = apply_deep_shrink(self.model, **self.settings)
        self.assertEqual(original_sampling.percent_to_sigma.call_count, 2)
        self.model.model.model_sampling.percent_to_sigma.assert_not_called()
        options = {"block": ("input", 3), "sigmas": torch.tensor([7.0])}
        self.assertEqual(input_patch(model)(self.h, options).shape[-2:], (4, 6))

    def test_worker_patches_base_and_refiner_and_disabled_next_task(self):
        source = ast.parse(
            (
                Path(__file__).resolve().parents[1] / "modules/async_worker.py"
            ).read_text()
        )
        node = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.FunctionDef) and node.name == "patch_samplers"
        )
        refiner = model_patcher()
        pipeline = SimpleNamespace(final_unet=self.model, final_refiner_unet=refiner)
        env = {"pipeline": pipeline, "apply_deep_shrink": apply_deep_shrink}
        exec(  # noqa: S102
            compile(
                ast.Module(body=[node], type_ignores=[]), "async_worker.py", "exec"
            ),
            env,
        )
        task = SimpleNamespace(scheduler_name="karras", deep_shrink=self.settings)
        self.assertEqual(env["patch_samplers"](task), "karras")
        self.assertIsNot(pipeline.final_unet, self.model)
        self.assertIsNot(pipeline.final_refiner_unet, refiner)
        pipeline.final_unet, pipeline.final_refiner_unet = self.model, refiner
        task.deep_shrink = DEFAULTS.copy()
        env["patch_samplers"](task)
        self.assertIs(pipeline.final_unet, self.model)
        self.assertIs(pipeline.final_refiner_unet, refiner)

    def test_real_fooocus_unet_forward_with_freeu_and_skip_modes(self):
        model = model_patcher(real_unet=True)
        inputs = torch.randn(1, 4, 16, 24)
        timesteps = torch.tensor([900.0])
        sigma = model.model.model_sampling.percent_to_sigma(0.1)
        with (
            patch.dict(patch_settings, {os.getpid(): PatchSettings()}),
            torch.no_grad(),
        ):
            for after_skip in [False, True]:
                for freeu in [False, True]:
                    with self.subTest(after_skip=after_skip, freeu=freeu):
                        base = (
                            FreeU_V2().patch(model, 1.01, 1.02, 0.99, 0.95)[0]
                            if freeu
                            else model
                        )
                        patched = apply_deep_shrink(
                            base,
                            **(self.settings | {"downscale_after_skip": after_skip}),
                        )
                        options = patched.model_options["transformer_options"] | {
                            "sigmas": torch.tensor([sigma])
                        }
                        output = patched_unet_forward(
                            model.model.diffusion_model,
                            inputs,
                            timesteps,
                            transformer_options=options,
                        )
                        self.assertEqual(output.shape, inputs.shape)
                        self.assertTrue(torch.isfinite(output).all())


class TestAnimaDeepShrink(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = tiny_config() | {"num_blocks": 5}
        cls.model = AnimaModel(AnimaModelConfig(config)).eval()
        with torch.no_grad():
            for name, parameter in cls.model.named_parameters():
                if "norm" in name and parameter.ndim == 1:
                    parameter.fill_(1)
                else:
                    torch.nn.init.normal_(parameter, std=0.02)
        cls.patcher = ModelPatcher(cls.model, torch.device("cpu"), torch.device("cpu"))

    def forward(self, settings=None, progress=0.1, shape=(1, 16, 9, 15)):
        model = apply_deep_shrink(
            self.patcher, **(DEFAULTS | {"enabled": True} | (settings or {}))
        )
        sigma = model.model.model_sampling.percent_to_sigma(progress)
        options = model.model_options["transformer_options"] | {
            "sigmas": torch.tensor([sigma])
        }
        observed = []

        def observe(block, args, kwargs):
            hidden = args[0]
            observed.append(tuple(hidden.shape[1:4]))
            rope = kwargs["rope_emb_L_1_1_D"]
            self.assertEqual(
                rope.shape[1], hidden.shape[1] * hidden.shape[2] * hidden.shape[3]
            )
            extra = kwargs["extra_per_block_pos_emb"]
            if extra is not None:
                self.assertEqual(extra.shape, hidden.shape)

        hooks = [
            block.register_forward_pre_hook(observe, with_kwargs=True)
            for block in model.model.diffusion_model.blocks
        ]
        self.addCleanup(lambda: [hook.remove() for hook in hooks])
        torch.manual_seed(1234)
        latent = torch.randn(shape)
        context = torch.randn(shape[0], 8, 32)
        try:
            with torch.no_grad():
                output = model.model.apply_model(
                    latent,
                    torch.full((shape[0],), sigma),
                    c_crossattn=context,
                    transformer_options=options,
                )
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(output.shape, latent.shape)
        self.assertEqual(output.dtype, latent.dtype)
        self.assertTrue(torch.isfinite(output).all())
        self.assertNotIn("patches", self.patcher.model_options["transformer_options"])
        self.assertNotIn("block", model.model_options["transformer_options"])
        return observed, output

    def test_real_dit_shrinks_at_selected_block_before_or_after(self):
        for after_skip in [False, True]:
            with self.subTest(after_skip=after_skip):
                observed, output = self.forward({"downscale_after_skip": after_skip})
                full_blocks = 4 if after_skip else 3
                self.assertEqual(
                    observed,
                    [(1, 5, 8)] * full_blocks + [(1, 2, 4)] * (5 - full_blocks),
                )
                _, baseline = self.forward({"enabled": False})
                self.assertFalse(torch.allclose(output, baseline))

    def test_flow_sigma_window_and_invalid_block_leave_forward_unchanged(self):
        _, baseline = self.forward({"enabled": False}, progress=0.7)
        for settings in [{"end_percent": 0.35}, {"block_number": 32}]:
            observed, output = self.forward(settings, progress=0.7)
            self.assertEqual(observed, [(1, 5, 8)] * 5)
            torch.testing.assert_close(output, baseline)
        for progress in [0.1, 0.2, 0.35]:
            observed, _ = self.forward({"start_percent": 0.1}, progress=progress)
            self.assertEqual(observed[-1], (1, 2, 4))
        observed, _ = self.forward({"start_percent": 0.1}, progress=0.05)
        self.assertEqual(observed, [(1, 5, 8)] * 5)

    def test_all_resize_methods_and_extreme_factors_restore_odd_output_size(self):
        for method in RESIZE_METHODS:
            with self.subTest(method=method):
                observed, _ = self.forward(
                    {"downscale_method": method, "upscale_method": method}
                )
                self.assertEqual(observed[-1], (1, 2, 4))
        for factor, expected in [
            (9.0, (1, 1, 1)),
            (0.5, (1, 10, 16)),
            (1.0, (1, 5, 8)),
        ]:
            observed, _ = self.forward({"downscale_factor": factor})
            self.assertEqual(observed[-1], expected)

    def test_batches_video_and_padding_masks_preserve_frames_and_channels(self):
        for shape in [(2, 16, 9, 15), (1, 16, 2, 9, 15)]:
            observed, _ = self.forward(shape=shape)
            frames = 1 if len(shape) == 4 else shape[2]
            self.assertEqual(observed[-1], (frames, 2, 4))

    def test_disabled_task_after_enabled_task_uses_full_grid(self):
        self.forward()
        observed, _ = self.forward({"enabled": False})
        self.assertEqual(observed, [(1, 5, 8)] * 5)

    def test_euler_cfg_sampling_switches_back_to_full_grid_after_window(self):
        from ldm_patched.k_diffusion.sampling import sample_euler

        patched = apply_deep_shrink(self.patcher, **(DEFAULTS | {"enabled": True}))
        positive = [
            {"model_conds": {"c_crossattn": CONDRegular(torch.randn(1, 8, 32))}}
        ]
        negative = [
            {"model_conds": {"c_crossattn": CONDRegular(torch.zeros(1, 8, 32))}}
        ]
        observed = []
        hook = self.model.diffusion_model.blocks[-1].register_forward_pre_hook(
            lambda block, args: observed.append(tuple(args[0].shape[1:4]))
        )

        def denoiser(latent, sigma):
            cond, uncond = calc_cond_uncond_batch(
                patched.model, positive, negative, latent, sigma, patched.model_options
            )
            return uncond + 4.5 * (cond - uncond)

        try:
            with torch.no_grad():
                output = sample_euler(
                    denoiser,
                    torch.randn(1, 16, 9, 15),
                    torch.tensor([1.0, 0.5, 0.0]),
                    disable=True,
                )
        finally:
            hook.remove()
        self.assertEqual(output.shape, (1, 16, 9, 15))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIn((1, 2, 4), observed)
        self.assertEqual(observed[-1], (1, 5, 8))

    def test_learned_positions_resize_when_expanding_past_table_size(self):
        config = tiny_config() | {
            "num_blocks": 5,
            "extra_per_block_abs_pos_emb": True,
            "max_img_h": 12,
            "max_img_w": 16,
        }
        model = AnimaModel(AnimaModelConfig(config)).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=0.02)
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        with patch.object(self, "patcher", patcher):
            observed, _ = self.forward({"downscale_factor": 0.5})
        self.assertEqual(observed[-1], (1, 10, 16))

    def test_worker_patches_anima_base_refiner_and_sdxl_anima_pair(self):
        source = ast.parse(
            (
                Path(__file__).resolve().parents[1] / "modules/async_worker.py"
            ).read_text()
        )
        node = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.FunctionDef) and node.name == "patch_samplers"
        )
        for base in [self.patcher, model_patcher()]:
            pipeline = SimpleNamespace(final_unet=base, final_refiner_unet=self.patcher)
            env = {"pipeline": pipeline, "apply_deep_shrink": apply_deep_shrink}
            exec(  # noqa: S102
                compile(
                    ast.Module(body=[node], type_ignores=[]), "async_worker.py", "exec"
                ),
                env,
            )
            env["patch_samplers"](
                SimpleNamespace(
                    scheduler_name="simple", deep_shrink=DEFAULTS | {"enabled": True}
                )
            )
            self.assertIsNot(pipeline.final_unet, base)
            self.assertIsNot(pipeline.final_refiner_unet, self.patcher)
            patches = pipeline.final_refiner_unet.model_options["transformer_options"][
                "patches"
            ]
            self.assertIn("dit_input_block_patch_after_skip", patches)
            self.assertIn("dit_output_block_patch", patches)
            self.assertNotIn("output_block_patch", patches)


class TestDeepShrinkMetadata(unittest.TestCase):
    def read(self, value):
        result = []
        get_deep_shrink(value, result)
        return dict(zip(DEFAULTS, result))

    def test_metadata_round_trip_and_missing_fields(self):
        settings = DEFAULTS | {
            "enabled": True,
            "block_number": 4,
            "downscale_factor": 1.75,
            "end_percent": 0.5,
            "downscale_after_skip": False,
            "upscale_method": "bilinear",
        }
        self.assertEqual(self.read({"deep_shrink": json.dumps(settings)}), settings)
        self.assertEqual(self.read({"UNet Deep Shrink": settings}), settings)
        self.assertEqual(
            self.read({"deep_shrink": {"enabled": True}}), DEFAULTS | {"enabled": True}
        )

    def test_old_or_malformed_metadata_resets_defaults(self):
        for value in [
            None,
            "invalid JSON",
            [],
            {"enabled": "false"},
            {"block_number": "3"},
            {"downscale_factor": 0},
            {"start_percent": float("nan")},
            {"upscale_method": "unknown"},
        ]:
            with self.subTest(value=value):
                self.assertEqual(self.read({"deep_shrink": value}), DEFAULTS)
        self.assertEqual(self.read({}), DEFAULTS)

    def test_fooocus_and_a1111_image_metadata_round_trip(self):
        settings = DEFAULTS | {"enabled": True, "downscale_factor": 1.75}
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
            "deep_shrink": json.dumps(settings),
        }
        for parser_type in [FooocusMetadataParser, A1111MetadataParser]:
            with self.subTest(parser=parser_type.__name__):
                parser = parser_type()
                parser.raw_prompt = "test prompt"
                parser.full_prompt = ["test prompt"]
                parser.full_negative_prompt = []
                text = parser.to_string(
                    [(key, key, value) for key, value in data.items()]
                )
                metadata = (
                    json.loads(text)
                    if isinstance(parser, FooocusMetadataParser)
                    else text
                )
                self.assertEqual(self.read(parser.to_json(metadata)), settings)

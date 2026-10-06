import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from ldm_patched.modules.anima import AnimaModel
from ldm_patched.modules.conds import CONDCrossAttn
from ldm_patched.modules.model_base import SDXL, SDXLRefiner
from ldm_patched.modules.model_sampling import (
    EPS,
    ModelSamplingDiscreteFlow,
)
from ldm_patched.modules.sample import convert_cond
from ldm_patched.modules.samplers import KSAMPLER
from modules import anima, anima_refiner, flags
from modules.inpaint_worker import InpaintWorker
from modules.sample_hijack import clip_separate, clip_separate_after_preparation


def anima_target():
    model = AnimaModel.__new__(AnimaModel)
    torch.nn.Module.__init__(model)
    return model


def qwen_conditioning():
    return [
        [
            torch.randn(1, 3, 64),
            {
                "t5xxl_ids": torch.tensor([[1, 2, 3]]),
                "t5xxl_weights": torch.tensor([[[1.0], [2.0], [0.5]]]),
            },
        ]
    ]


class TestAnimaRefinerConditioning(unittest.TestCase):
    def test_sdxl_context_replaced_by_refiner_qwen_preserving_region(self):
        own = qwen_conditioning()
        mask = torch.ones(1, 8, 8)
        source = [
            [
                torch.randn(1, 77, 2048),
                {
                    "pooled_output": torch.randn(1, 1280),
                    "area": (4, 4, 0, 0),
                    "mask": mask,
                    "strength": 0.7,
                    "anima_refiner_conditioning": own,
                },
            ]
        ]
        result = clip_separate(source, target_model=anima_target())
        torch.testing.assert_close(result[0][0], own[0][0])
        torch.testing.assert_close(
            result[0][1]["t5xxl_weights"], own[0][1]["t5xxl_weights"]
        )
        torch.testing.assert_close(result[0][1]["mask"], mask)
        self.assertEqual(result[0][1]["area"], (4, 4, 0, 0))
        self.assertEqual(result[0][1]["strength"], 0.7)
        self.assertNotIn("pooled_output", result[0][1])
        with torch.inference_mode():
            result[0][0].zero_()
            result[0][1]["t5xxl_ids"].zero_()
        self.assertGreater(own[0][0].count_nonzero(), 0)
        self.assertEqual(own[0][1]["t5xxl_ids"].tolist(), [[1, 2, 3]])

    def test_same_family_uses_full_raw_qwen_context(self):
        source = qwen_conditioning()
        result = clip_separate(source, target_model=anima_target())
        torch.testing.assert_close(result[0][0], source[0][0])
        self.assertEqual(result[0][0].shape[-1], 64)
        self.assertNotEqual(result[0][0].data_ptr(), source[0][0].data_ptr())

    def test_prepared_context_does_not_reuse_base_adapter_output(self):
        source = qwen_conditioning()
        own = qwen_conditioning()
        source[0][1]["anima_refiner_conditioning"] = own
        prepared = convert_cond(source)
        prepared[0]["model_conds"]["c_crossattn"] = CONDCrossAttn(
            torch.zeros(1, 512, 32)
        )
        result = clip_separate_after_preparation(prepared, target_model=anima_target())
        torch.testing.assert_close(result[0]["cross_attn"], own[0][0])
        torch.testing.assert_close(
            result[0]["model_conds"]["c_crossattn"].cond, own[0][0]
        )
        self.assertEqual(result[0]["t5xxl_ids"].tolist(), [[1, 2, 3]])

    def test_clip_only_context_rejected_for_anima(self):
        with self.assertRaisesRegex(ValueError, "Qwen/T5"):
            clip_separate([[torch.zeros(1, 77, 2048), {}]], target_model=anima_target())

    def test_legacy_sdxl_and_refiner_clip_slices(self):
        context, pooled = torch.randn(1, 77, 2048), torch.randn(1, 1280)
        for cls, expected in [(SDXL, context), (SDXLRefiner, context[..., -1280:])]:
            with self.subTest(model=cls.__name__):
                target = cls.__new__(cls)
                torch.nn.Module.__init__(target)
                result = clip_separate(
                    [[context, {"pooled_output": pooled}]], target_model=target
                )
                torch.testing.assert_close(result[0][0], expected)
                torch.testing.assert_close(result[0][1]["pooled_output"], pooled)


class TestAnimaRefinerTask(unittest.TestCase):
    def task(self):
        return SimpleNamespace(
            base_model_name="base.safetensors",
            refiner_model_name="refiner.safetensors",
            refiner_swap_method="separate",
            performance_selection=flags.Performance.QUALITY,
            sampler_name="euler",
            scheduler_name="simple",
            input_image_checkbox=False,
            current_tab="ip",
            mixing_image_prompt_and_vary_upscale=False,
            mixing_image_prompt_and_inpaint=False,
            cn_tasks={"ip": []},
            freeu_enabled=True,
            inpaint_engine="v2.6",
        )

    def test_sdxl_base_anima_refiner_uses_vae_bridge_and_guards(self):
        task = self.task()
        with patch.object(anima, "is_anima_file", side_effect=[False, True]):
            self.assertTrue(anima.prepare_task(task))
        self.assertEqual(task.refiner_model_name, "refiner.safetensors")
        self.assertEqual(task.refiner_swap_method, "vae")
        self.assertFalse(task.freeu_enabled)
        self.assertEqual(task.inpaint_engine, "None")

    def test_same_family_keeps_refiner_and_selected_mode(self):
        for method in ("joint", "separate", "vae"):
            with self.subTest(method=method):
                task = self.task()
                task.refiner_swap_method = method
                with patch.object(anima, "is_anima_file", return_value=True):
                    self.assertTrue(anima.prepare_task(task))
                self.assertEqual(task.refiner_swap_method, method)
                self.assertEqual(task.refiner_model_name, "refiner.safetensors")

    def test_base_only_anima_uses_joint(self):
        task = self.task()
        task.refiner_model_name = "None"
        with patch.object(anima, "is_anima_file", return_value=True) as detect:
            self.assertTrue(anima.prepare_task(task))
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(task.refiner_swap_method, "joint")

    def test_reverse_cross_family_rejected(self):
        with patch.object(anima, "is_anima_file", side_effect=[True, False]):
            with self.assertRaisesRegex(ValueError, "Anima refiner"):
                anima.prepare_task(self.task())

    def test_sdxl_pair_keeps_existing_configuration(self):
        task = self.task()
        before = vars(task).copy()
        with patch.object(anima, "is_anima_file", return_value=False):
            self.assertFalse(anima.prepare_task(task))
        self.assertEqual(vars(task), before)

    def test_anima_refiner_rejects_active_sdxl_control(self):
        task = self.task()
        task.input_image_checkbox = True
        task.cn_tasks = {"ip": ["active"]}
        with patch.object(anima, "is_anima_file", side_effect=[False, True]):
            with self.assertRaisesRegex(ValueError, "Image Prompt"):
                anima.prepare_task(task)


class TestFlowContinuation(unittest.TestCase):
    def initial_sample(self, sampling, resume):
        noise = torch.full((1, 16, 2, 2), 2.0)
        latent = torch.full_like(noise, 3.0)

        def capture(model, initial, sigmas, **kwargs):
            return initial.clone()

        wrapper = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=sampling))
        return KSAMPLER(capture).sample(
            wrapper,
            torch.tensor([0.6, 0.0]),
            {"model_options": {"anima_resume": resume}},
            None,
            noise,
            latent,
        )

    def test_flow_resume_keeps_existing_noisy_latent(self):
        result = self.initial_sample(ModelSamplingDiscreteFlow(), True)
        torch.testing.assert_close(result, torch.full_like(result, 3.0))

    def test_fresh_flow_sampling_interpolates_noise_and_image(self):
        result = self.initial_sample(ModelSamplingDiscreteFlow(), False)
        torch.testing.assert_close(result, torch.full_like(result, 0.6 * 2 + 0.4 * 3))

    def test_resume_does_not_change_eps_initialization(self):
        sampling = EPS()
        sampling.sigma_max = torch.tensor(1.0)
        result = self.initial_sample(sampling, True)
        torch.testing.assert_close(result, torch.full_like(result, 0.6 * 2 + 3))


class TestAnimaVAEBridge(unittest.TestCase):
    def setUp(self):
        self.base, self.refiner = object(), SimpleNamespace(model=anima_target())
        self.base_vae, self.refiner_vae = object(), object()
        self.sigmas = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])
        self.latent = {
            "samples": torch.zeros(1, 4, 8, 8),
            "noise_mask": torch.ones(1, 1, 8, 8),
        }
        self.kwargs = dict(
            base_model=self.base,
            refiner_model=self.refiner,
            base_vae=self.base_vae,
            refiner_vae=self.refiner_vae,
            positive=qwen_conditioning(),
            negative=qwen_conditioning(),
            latent=self.latent,
            steps=4,
            switch=2,
            seed=42,
            callback=Mock(),
            sampler_name="euler",
            scheduler_name="simple",
            cfg=4.5,
            denoise=0.5,
            tiled=True,
            disable_preview=False,
            refiner_sigmas=self.sigmas,
        )
        self.sample_calls = []

        def sample(**kwargs):
            self.sample_calls.append(kwargs)
            return kwargs["latent"]

        self.sampler = patch.object(
            anima_refiner.core, "ksampler", side_effect=sample
        ).start()
        self.decode = patch.object(
            anima_refiner.core, "decode_vae", return_value=torch.zeros(1, 64, 64, 3)
        ).start()
        self.encode = patch.object(
            anima_refiner.core,
            "encode_vae",
            side_effect=lambda *a, **kw: {"samples": torch.ones(1, 16, 8, 8)},
        ).start()
        patch.object(anima_refiner.inpaint_worker, "current_task", None).start()
        self.addCleanup(patch.stopall)

    def test_bridge_converts_channels_and_uses_own_flow_schedule(self):
        anima_refiner.sample_with_vae_bridge(**self.kwargs)
        base, refiner = self.sample_calls
        self.assertIs(base["model"], self.base)
        self.assertTrue(base["force_full_denoise"])
        self.assertEqual(base["last_step"], 2)
        self.assertEqual(base["denoise"], 0.5)
        self.assertEqual(refiner["latent"]["samples"].shape[1], 16)
        self.assertIs(refiner["latent"]["noise_mask"], self.latent["noise_mask"])
        torch.testing.assert_close(refiner["sigmas"], self.sigmas[2:])
        self.assertEqual(refiner["seed"], 43)
        self.assertEqual(refiner["steps"], 2)
        self.assertEqual((refiner["previewer_start"], refiner["previewer_end"]), (2, 4))
        self.decode.assert_any_call(self.base_vae, self.latent, tiled=True)
        self.encode.assert_called_once()
        self.assertIs(self.encode.call_args.args[0], self.refiner_vae)

    def test_zero_switch_skips_base_sampling(self):
        self.kwargs["switch"] = 0
        anima_refiner.sample_with_vae_bridge(**self.kwargs)
        self.assertEqual(len(self.sample_calls), 1)
        self.assertIs(self.sample_calls[0]["model"], self.refiner)
        torch.testing.assert_close(self.sample_calls[0]["sigmas"], self.sigmas)

    def test_full_switch_skips_refiner_and_vae_encoding(self):
        self.kwargs["switch"] = 4
        anima_refiner.sample_with_vae_bridge(**self.kwargs)
        self.assertEqual(len(self.sample_calls), 1)
        self.assertIs(self.sample_calls[0]["model"], self.base)
        self.encode.assert_not_called()

    def test_inpaint_restored_even_when_refiner_sampling_raises(self):
        task = InpaintWorker.__new__(InpaintWorker)
        task.swapped = False
        task.latent = self.latent["samples"]
        task.latent_after_swap = None
        with patch.object(anima_refiner.inpaint_worker, "current_task", task):

            def failing_sample(**kwargs):
                if kwargs["model"] is self.refiner:
                    self.assertTrue(task.swapped)
                    self.assertEqual(task.latent.shape[1], 16)
                    raise RuntimeError("refiner failure")
                return kwargs["latent"]

            self.sampler.side_effect = failing_sample
            with self.assertRaisesRegex(RuntimeError, "refiner failure"):
                anima_refiner.sample_with_vae_bridge(**self.kwargs)
        self.assertFalse(task.swapped)
        self.assertEqual(task.latent.shape[1], 4)
        self.assertEqual(task.latent_after_swap.shape[1], 16)


if __name__ == "__main__":
    unittest.main()

"""Model-free masked img2img, adapted from Forge Classic processing.py.

Only the selected diffusion model and its VAE are required. No inpaint head,
patch LoRA or learned image upscaler is used in this path.
"""

import json
import math

import cv2
import numpy as np
import torch
from PIL import Image

from extras.inpaint_masking import (
    expand_crop_region,
    fill,
    get_crop_region_v2,
    resize_image,
)
from extras.soft_inpainting import (
    SoftInpaintingSettings,
    adaptive_mask,
    get_modified_nmask,
    latent_blend,
)
from modules.inpaint_worker import InpaintWorker


class ModelFreeDenoiser:
    """Give UniPC the same masked denoiser used by K-diffusion samplers."""

    def __init__(self, task, model_wrap, noise):
        self.task, self.model_wrap, self.noise = task, model_wrap, noise

    @property
    def inner_model(self):
        return self.model_wrap.inner_model

    def __call__(self, x, sigma, **kwargs):
        from types import SimpleNamespace

        sampler = SimpleNamespace(inner_model=self.model_wrap, noise=self.noise)
        return self.task.sample(sampler, x, sigma, **kwargs)


BACKENDS = [
    ("Standard inpaint (no dedicated inpaint model)", "standard"),
    ("Fooocus dedicated inpaint model", "fooocus"),
]
MASK_MODES = [("Inpaint masked", "masked"), ("Inpaint not masked", "unmasked")]
CONTENTS = [
    ("fill", "fill"),
    ("original", "original"),
    ("latent noise", "latent_noise"),
    ("latent nothing", "latent_nothing"),
]
AREAS = [("Whole picture", "whole"), ("Only masked", "masked")]
DEFAULTS = {
    "backend": "standard",
    "mask_blur": 4,
    "mask_transparency": 50,
    "mask_mode": "masked",
    "content": "original",
    "padding": 32,
    "area": "masked",
    "soft_enabled": False,
    "mask_blend_power": 1.0,
    "mask_blend_scale": 0.5,
    "inpaint_detail_preservation": 4.0,
    "composite_mask_influence": 0.0,
    "composite_difference_threshold": 0.5,
    "composite_difference_contrast": 2.0,
}
RANGES = {
    "mask_blur": (0, 64),
    "mask_transparency": (0, 100),
    "padding": (0, 256),
    "mask_blend_power": (0, 8),
    "mask_blend_scale": (0, 8),
    "inpaint_detail_preservation": (1, 32),
    "composite_mask_influence": (0, 1),
    "composite_difference_threshold": (0, 8),
    "composite_difference_contrast": (0, 8),
}
CHOICES = {
    "backend": BACKENDS,
    "mask_mode": MASK_MODES,
    "content": CONTENTS,
    "area": AREAS,
}


def read_settings(value):
    """Safely restore complete settings from JSON image metadata."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = None
    if not isinstance(value, dict):
        return DEFAULTS.copy()
    settings = {key: value.get(key, default) for key, default in DEFAULTS.items()}
    for key, (low, high) in RANGES.items():
        number = settings[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or not low <= number <= high
        ):
            return DEFAULTS.copy()
    for key in ("mask_blur", "mask_transparency", "padding"):
        if int(settings[key]) != settings[key]:
            return DEFAULTS.copy()
        settings[key] = int(settings[key])
    if not isinstance(settings["soft_enabled"], bool):
        return DEFAULTS.copy()
    for key, choices in CHOICES.items():
        if settings[key] not in [value for _, value in choices]:
            return DEFAULTS.copy()
    return settings


def parameterized_enabled(settings, engine):
    return settings["backend"] == "fooocus" and engine != "None"


def prepare_mask(mask, settings, *, invert=False, apply_blur=True):
    """Invert before Gaussian blur; preserve fractional masks for soft inpaint."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if invert ^ (settings["mask_mode"] == "unmasked"):
        mask = 255 - mask
    if not settings["soft_enabled"]:
        mask = (mask > 128).astype(np.uint8) * 255
    return blur_mask(mask, settings["mask_blur"]) if apply_blur else mask


def blur_mask(mask, blur):
    if blur > 0:
        kernel = 2 * int(2.5 * blur + 0.5) + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), blur)
    return mask


class ModelFreeInpaintWorker(InpaintWorker):
    """Forge crop, fill, latent preservation and pixel composition adapter."""

    is_model_free = True

    def __init__(self, image, mask, width, height, settings, *, outpaint=False):
        self.settings = read_settings(settings)
        self.soft = SoftInpaintingSettings(
            **{
                key: self.settings[key]
                for key in SoftInpaintingSettings.__dataclass_fields__
            }
        )
        mask = blur_mask(mask, self.settings["mask_blur"])
        source = Image.fromarray(image).convert("RGB")
        mask_image = Image.fromarray(mask).convert("L")
        # No selected pixels: retain the source rather than falling back to
        # unmasked img2img as older WebUI versions do.
        self.empty = mask_image.getbbox() is None
        only_masked = self.settings["area"] == "masked" or outpaint
        if only_masked:
            crop = get_crop_region_v2(mask_image, self.settings["padding"])
            crop = (
                expand_crop_region(crop, width, height, source.width, source.height)
                if crop
                else (0, 0, source.width, source.height)
            )
            x1, y1, x2, y2 = crop
            self.image = np.array(source)
            self.mask = np.array(mask_image)
            self.interested_area = (y1, y2, x1, x2)
            source = source.crop(crop)
            mask_image = mask_image.crop(crop)
        else:
            self.image = np.array(
                source.resize((width, height), Image.Resampling.LANCZOS)
            )
            # Forge's whole-picture overlay boosts mask opacity using sqrt.
            full_mask = (
                np.array(mask_image.resize((width, height), Image.Resampling.LANCZOS))
                / 255.0
            )
            self.mask = (
                np.clip(np.sqrt(full_mask) * 255, 0, 255).round().astype(np.uint8)
            )
            self.interested_area = (0, height, 0, width)
        self.interested_image = np.array(
            resize_image(2 if only_masked else 0, source, width, height)
        )
        self.interested_mask = np.array(
            resize_image(2 if only_masked else 0, mask_image, width, height)
        )
        self.interested_fill = (
            np.array(
                fill(
                    Image.fromarray(self.interested_image),
                    Image.fromarray(self.interested_mask),
                )
            )
            if self.settings["content"] != "original"
            else self.interested_image.copy()
        )
        self.latent = self.latent_after_swap = self.latent_mask = None
        self.swapped = False
        self.inpaint_head_feature = None
        self.overlay_mask = None

    def load_latent(
        self,
        latent_fill,
        latent_mask=None,
        latent_swap=None,
        latent_model=None,
        swap_model=None,
    ):
        super().load_latent(latent_fill, latent_mask, latent_swap)
        self.base_fill = latent_fill
        self.swap_fill = latent_swap
        self.base_model, self.swap_model = latent_model, swap_model
        mask = torch.from_numpy(self.interested_mask.copy()).float()[None, None] / 255
        self.latent_mask = torch.nn.functional.interpolate(
            mask, size=latent_fill.shape[-2:], mode="bilinear", align_corners=False
        )
        if not self.settings["soft_enabled"]:
            self.latent_mask = self.latent_mask.round()

    def _initial_content(self, latent, seed, model=None):
        # Forge initializes noise/zero in the denoiser's latent space. VAE
        # tensors in Fooocus are unscaled and may also have mean/std offsets.
        initial = model.process_latent_in(latent) if model is not None else latent
        mask = self.latent_mask.to(initial)
        if self.settings["content"] == "latent_noise":
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(
                initial.shape, generator=generator, dtype=initial.dtype, device="cpu"
            ).to(initial)
            initial = initial * (1 - mask) + noise * mask
        if self.settings["content"] == "latent_nothing":
            initial = initial * (1 - mask)
        if self.settings["content"] in ("latent_noise", "latent_nothing"):
            return model.process_latent_out(initial) if model is not None else initial
        return latent

    def prepare_latent(self, seed):
        self.unswap()
        self.latent = self._initial_content(self.base_fill, seed, self.base_model)
        self.latent_after_swap = (
            self._initial_content(self.swap_fill, seed, self.swap_model)
            if self.swap_fill is not None
            else None
        )
        self.overlay_mask = None
        return {"samples": self.latent, "noise_mask": self.latent_mask}

    def sample(self, sampler, x, sigma, **kwargs):
        """Preserve unmasked latents using the active model's noise convention."""
        model = sampler.inner_model.inner_model
        original = model.process_latent_in(self.latent).to(x)
        mask = self.latent_mask.to(x)
        energy_sigma = sigma.reshape([sigma.shape[0]] + [1] * (x.ndim - 1))
        noisy_original = model.model_sampling.noise_scaling(
            energy_sigma, sampler.noise.to(x), original
        )
        # Fractional masks are blended after denoising by Forge's soft formula.
        input_mask = (mask > 0).to(x) if self.settings["soft_enabled"] else mask
        x = x * input_mask + noisy_original * (1 - input_mask)
        denoised = sampler.inner_model(x, sigma, **kwargs)
        if self.settings["soft_enabled"]:
            t = get_modified_nmask(self.soft, mask, energy_sigma)
            blended = latent_blend(self.soft, original, denoised, t)
            # Keep exact endpoints despite upstream's norm epsilon or 0**0.
            return torch.where(
                mask == 0, original, torch.where(mask == 1, denoised, blended)
            )
        return denoised * mask + original * (1 - mask)

    def finish_sample(self, samples, model):
        mask = self.latent_mask.to(samples)
        original = self.latent.to(samples)
        # Forge performs a final composition for ordinary inpaint. Soft inpaint
        # keeps fractional regions untouched but still protects exact-zero mask
        # pixels, including UniPC's small residual at its nonzero last sigma.
        samples = (
            torch.where(mask == 0, original, samples)
            if self.settings["soft_enabled"]
            else samples * mask + original * (1 - mask)
        )
        if self.settings["soft_enabled"]:
            self.overlay_mask = adaptive_mask(
                self.soft,
                self.latent_mask.to(samples),
                model.process_latent_in(self.latent).to(samples),
                model.process_latent_in(samples),
                (self.interested_image.shape[1], self.interested_image.shape[0]),
            )
        return samples

    def post_process(self, img):
        if self.empty:
            return self.image.copy()
        a, b, c, d = self.interested_area
        content = np.array(resize_image(1, Image.fromarray(img), d - c, b - a))
        result = self.image.copy()
        result[a:b, c:d] = content
        if self.settings["soft_enabled"] and self.overlay_mask is not None:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask[a:b, c:d] = np.array(resize_image(1, self.overlay_mask, d - c, b - a))
            # Never extend the requested paint region, even when mask influence
            # is zero and VAE reconstruction differs across the whole image.
            mask = np.where(self.mask > 0, mask, 0)
        else:
            mask = self.mask
        return np.array(
            Image.composite(
                Image.fromarray(result),
                Image.fromarray(self.image),
                Image.fromarray(mask),
            )
        )

"""Fooocus integration for ComfyUI's Kohya Deep Shrink UNet patch."""

from ldm_patched.contrib.external_model_downscale import PatchModelAddDownscale
from ldm_patched.ldm.anima.predict2 import MiniTrainDIT
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel

DEFAULTS = {
    "enabled": False,
    "block_number": 3,
    "downscale_factor": 2.0,
    "start_percent": 0.0,
    "end_percent": 0.35,
    "downscale_after_skip": True,
    "downscale_method": "bicubic",
    "upscale_method": "bicubic",
}
RESIZE_METHODS = PatchModelAddDownscale.upscale_methods


def apply_deep_shrink(model, enabled=False, **settings):
    if not enabled or model is None:
        return model
    if not isinstance(model.model.diffusion_model, (UNetModel, MiniTrainDIT)):
        print("[Deep Shrink] Skipped: unsupported denoiser architecture.")
        return model
    (patched,) = PatchModelAddDownscale().patch(model, **settings)
    return patched

# Adapted from ComfyUI comfy_extras/nodes_model_downscale.py (Kohya Deep Shrink).

import torch
import ldm_patched.modules.utils
from ldm_patched.ldm.anima.predict2 import MiniTrainDIT

class PatchModelAddDownscale:
    upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                              "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
                              "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "downscale_after_skip": ("BOOLEAN", {"default": True}),
                              "downscale_method": (s.upscale_methods,),
                              "upscale_method": (s.upscale_methods,),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method):
        model_sampling = model.object_patches.get("model_sampling")
        if model_sampling is None:
            model_sampling = model.object_patches_backup.get("model_sampling", model.model.model_sampling)
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        def input_block_patch(h, transformer_options):
            if transformer_options["block"][1] == block_number:
                sigmas = transformer_options["sigmas"]
                sigma = sigmas[0].item()
                # Compare at the sampler's precision so flow schedule boundary
                # steps are included despite Python float -> tensor rounding.
                start, end = torch.as_tensor((sigma_start, sigma_end), dtype=sigmas.dtype, device="cpu").tolist()
                if sigma <= start and sigma >= end:
                    h = ldm_patched.modules.utils.common_upscale(h, max(1, round(h.shape[-1] / downscale_factor)), max(1, round(h.shape[-2] / downscale_factor)), downscale_method, "disabled")
            return h

        def output_block_patch(h, hsp, transformer_options):
            if h.shape[-2:] != hsp.shape[-2:]:
                h = ldm_patched.modules.utils.common_upscale(h, hsp.shape[-1], hsp.shape[-2], upscale_method, "disabled")
            return h, hsp

        m = model.clone()
        if isinstance(model.model.diffusion_model, MiniTrainDIT):
            # DiT blocks have no UNet skip stack. Run before/after the selected
            # transformer block, then restore the spatial grid before output.
            name = "dit_input_block_patch_after_skip" if downscale_after_skip else "dit_input_block_patch"
            m.set_model_patch(input_block_patch, name)
            m.set_model_patch(output_block_patch, "dit_output_block_patch")
        else:
            if downscale_after_skip:
                m.set_model_input_block_patch_after_skip(input_block_patch)
            else:
                m.set_model_input_block_patch(input_block_patch)
            m.set_model_output_block_patch(output_block_patch)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "PatchModelAddDownscale": PatchModelAddDownscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Sampling
    "PatchModelAddDownscale": "PatchModelAddDownscale (Kohya Deep Shrink)",
}

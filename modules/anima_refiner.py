"""Refine decoded base-model images in Anima's own latent and flow spaces."""

import torch

import modules.core as core
import modules.inpaint_worker as inpaint_worker
from modules.sample_hijack import clip_separate


@torch.no_grad()
@torch.inference_mode()
def sample_with_vae_bridge(
    *,
    base_model,
    refiner_model,
    base_vae,
    refiner_vae,
    positive,
    negative,
    latent,
    steps,
    switch,
    seed,
    callback,
    sampler_name,
    scheduler_name,
    cfg,
    denoise,
    tiled,
    disable_preview,
    refiner_sigmas,
):
    switch = max(0, min(steps, switch))
    task = inpaint_worker.current_task
    if task is not None:
        task.unswap()
    try:
        if switch > 0:
            latent = core.ksampler(
                model=base_model,
                positive=positive,
                negative=negative,
                latent=latent,
                steps=steps,
                start_step=0,
                last_step=switch,
                force_full_denoise=True,
                seed=seed,
                denoise=denoise,
                callback_function=callback,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler_name,
                previewer_start=0,
                previewer_end=steps,
                disable_preview=disable_preview,
            )
        pixels = core.decode_vae(base_vae, latent, tiled=tiled)
        if switch == steps:
            return pixels

        refiner_positive = clip_separate(positive, target_model=refiner_model.model)
        refiner_negative = clip_separate(negative, target_model=refiner_model.model)
        encoded = core.encode_vae(refiner_vae, pixels, tiled=tiled)
        if "noise_mask" in latent:
            encoded["noise_mask"] = latent["noise_mask"]
        if task is not None:
            if task.latent_after_swap is None:
                protected_pixels = core.decode_vae(
                    base_vae, {"samples": task.latent}, tiled=tiled
                )
                task.latent_after_swap = core.encode_vae(
                    refiner_vae, protected_pixels, tiled=tiled
                )["samples"]
            task.swap()
        sigmas = refiner_sigmas[switch:]
        refiner_steps = len(sigmas) - 1
        print(
            "[Anima Refiner] VAE bridge; sampling in 16-channel latents with the Anima flow schedule."
        )
        refined = core.ksampler(
            model=refiner_model,
            positive=refiner_positive,
            negative=refiner_negative,
            latent=encoded,
            steps=refiner_steps,
            start_step=0,
            last_step=refiner_steps,
            force_full_denoise=True,
            seed=seed + 1,
            denoise=1.0,
            callback_function=callback,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
            sigmas=sigmas,
            disable_preview=disable_preview,
        )
        return core.decode_vae(refiner_vae, refined, tiled=tiled)
    finally:
        if task is not None:
            task.unswap()

"""Rectified-flow Euler ancestral sampling, adapted from ComfyUI (GPL-3.0)."""

import torch
from tqdm.auto import trange


@torch.no_grad()
def sample_euler_ancestral_flow(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
):
    extra_args = {} if extra_args is None else extra_args
    if noise_sampler is None:
        seed = extra_args.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed + (1 if x.device.type == "cpu" else 0))

        def noise_sampler(sigma, sigma_next):
            return torch.randn(
                x.shape, dtype=x.dtype, device=x.device, generator=generator
            )

    batch = x.new_ones([x.shape[0]])
    for index in trange(len(sigmas) - 1, disable=disable):
        sigma, next_sigma = sigmas[index], sigmas[index + 1]
        denoised = model(x, sigma * batch, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": index,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                }
            )
        if next_sigma == 0:
            x = denoised
        else:
            down_ratio = 1 + (next_sigma / sigma - 1) * eta
            sigma_down = next_sigma * down_ratio
            alpha_next, alpha_down = 1 - next_sigma, 1 - sigma_down
            renoise = (
                next_sigma**2 - sigma_down**2 * alpha_next**2 / alpha_down**2
            ).sqrt()
            ratio = sigma_down / sigma
            x = ratio * x + (1 - ratio) * denoised
            if eta > 0:
                x = (alpha_next / alpha_down) * x + noise_sampler(
                    sigma, next_sigma
                ) * s_noise * renoise
    return x

"""Soft inpainting algorithms ported from Forge Classic's built-in extension.

Source: extensions-builtin/soft-inpainting/scripts/soft_inpainting.py (AGPL-3.0).
The latent blend and noise-dependent mask formulas retain upstream semantics.
Fooocus adapters replace WebUI callbacks, device helpers and image processing.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFilter


@dataclass
class SoftInpaintingSettings:
    mask_blend_power: float = 1.0
    mask_blend_scale: float = 0.5
    inpaint_detail_preservation: float = 4.0
    composite_mask_influence: float = 0.0
    composite_difference_threshold: float = 0.5
    composite_difference_contrast: float = 2.0


def float64(tensor):
    return torch.float32 if tensor.device.type in ("mps", "xpu") else torch.float64


def weighted_histogram_filter(
    img, kernel, kernel_center, percentile_min=0.0, percentile_max=1.0, min_width=1.0
):
    """Upstream weighted percentile filter, vectorized in bounded row chunks.

    This retains the zero padding and percentile overlap calculation without
    starting joblib worker processes for every generated image.
    """
    radius = int(kernel_center)
    padded = np.pad(img, radius, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    result = np.empty_like(img)
    weights = kernel.reshape(-1)
    # Keep sort buffers bounded even for large latent maps.
    rows = max(1, 262144 // (img.shape[1] * kernel.size))
    for start in range(0, img.shape[0], rows):
        values = windows[start : start + rows].reshape(-1, kernel.size)
        order = np.argsort(values, axis=-1)
        values = np.take_along_axis(values, order, axis=-1)
        cumulative = np.cumsum(
            np.take_along_axis(np.broadcast_to(weights, order.shape), order, axis=-1),
            axis=-1,
        )
        total = cumulative[:, -1:]
        low = np.maximum(0, total * percentile_min)
        high = np.minimum(total, total * percentile_max)
        narrow = high - low < min_width
        center = (low + high) / 2
        low = np.where(narrow, np.maximum(0, center - min_width / 2), low)
        high = np.where(narrow, np.minimum(total, center + min_width / 2), high)
        previous = np.concatenate((np.zeros_like(total), cumulative[:, :-1]), axis=-1)
        overlap = np.maximum(
            0, np.minimum(high, cumulative) - np.maximum(low, previous)
        )
        denominator = overlap.sum(axis=-1)
        filtered = np.divide(
            (values * overlap).sum(axis=-1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        result[start : start + rows] = filtered.reshape(-1, img.shape[1])
    return result


def adaptive_mask(settings, nmask, latent_orig, latent_processed, size):
    """Forge's latent-difference overlay mask, adapted to a single Fooocus image."""
    latent_mask = nmask[:, 0].float().clamp(0, 1)
    mask_scalar = 1 - latent_mask.pow(settings.mask_blend_scale / 2)
    mask_scalar = (
        0.5 * (1 - settings.composite_mask_influence)
        + mask_scalar * settings.composite_mask_influence
    )
    mask_scalar = (mask_scalar / (1.00001 - mask_scalar)).cpu().numpy()[0]
    distance = (
        torch.norm(latent_processed - latent_orig, p=2, dim=1)[0].float().cpu().numpy()
    )
    kernel, center = get_gaussian_kernel(stddev_radius=1.5, max_radius=2)
    distance = weighted_histogram_filter(distance, kernel, center, 0.9, 1, 1)
    distance = weighted_histogram_filter(distance, kernel, center, 0.25, 0.75, 1)
    # Upstream divides by zero when threshold/influence are zero. Define the
    # limiting behavior explicitly: no change keeps original, any change repaints.
    half_distance = settings.composite_difference_threshold * mask_scalar
    ratio = np.divide(
        distance,
        half_distance,
        out=np.where(distance > 0, np.inf, 0.0).astype(distance.dtype),
        where=half_distance > 0,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        opacity = 1 / (1 + ratio**settings.composite_difference_contrast)
    mask = np.clip((1 - smootherstep(opacity)) * 255, 0, 255).astype(np.uint8)
    mask = Image.fromarray(mask).resize(size, Image.Resampling.LANCZOS)
    return mask.filter(ImageFilter.GaussianBlur(radius=4))


def latent_blend(
    settings: "SoftInpaintingSettings",
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolates two latent image representations according to the parameter `t`,
    where the interpolated vectors' magnitudes are also interpolated separately.
    The `detail_preservation` factor biases the magnitude interpolation towards
    the larger of the two magnitudes.
    """

    ndim = a.ndim

    if t.ndim == 3:
        t = t.unsqueeze(0)

    if ndim == 5 and t.ndim == 4:
        t = t.unsqueeze(2)

    one_minus_t = 1 - t

    image_interp = a * one_minus_t
    image_interp.add_(b * t)
    result_type = image_interp.dtype

    current_magnitude = (
        torch.norm(image_interp, p=2, dim=1, keepdim=True)
        .to(float64(image_interp))
        .add_(0.00001)
    )

    a_magnitude = (
        torch.norm(a, p=2, dim=1, keepdim=True)
        .to(float64(a))
        .pow_(settings.inpaint_detail_preservation)
        * one_minus_t
    )
    b_magnitude = (
        torch.norm(b, p=2, dim=1, keepdim=True)
        .to(float64(b))
        .pow_(settings.inpaint_detail_preservation)
        * t
    )

    desired_magnitude = a_magnitude
    desired_magnitude.add_(b_magnitude).pow_(1 / settings.inpaint_detail_preservation)

    scale = desired_magnitude.div_(current_magnitude).to(result_type)
    image_interp.mul_(scale)

    return image_interp


def get_modified_nmask(
    settings: "SoftInpaintingSettings", nmask: torch.Tensor, sigma: float
) -> torch.Tensor:
    """
    Converts a negative mask representing the transparency of the original latent vectors being overlaid
    to a mask that is scaled according to the denoising strength for this step.

    Where:
        0 = fully opaque, infinite density, fully masked
        1 = fully transparent, zero density, fully unmasked

    We bring this transparency to a power, as this allows one to simulate N number of blending operations
    where N can be any positive real value. Using this one can control the balance of influence between
    the denoiser and the original latents according to the sigma value.
    """
    return torch.pow(
        nmask, (sigma**settings.mask_blend_power) * settings.mask_blend_scale
    )


def smootherstep(x):
    """
    The smootherstep function, input should be clamped to 0-1 range.
    Turns a diagonal line (f(x) = x) into a sigmoid-like curve.
    """
    return x * x * x * (x * (6 * x - 15) + 10)


def get_gaussian_kernel(stddev_radius=1.0, max_radius=2):
    """
    Creates a Gaussian kernel with thresholded edges.

    Args:
        stddev_radius (float):
            Standard deviation of the gaussian kernel, in pixels.
        max_radius (int):
            The size of the filter kernel. The number of pixels is (max_radius*2+1) ** 2.
            The kernel is thresholded so that any values one pixel beyond this radius
            is weighted at 0.

    Returns:
        (nparray, nparray): A kernel array (shape: (N, N)), its center coordinate (shape: (2))
    """

    # Evaluates a 0-1 normalized gaussian function for a given square distance from the mean.
    def gaussian(sqr_mag):
        return math.exp(-sqr_mag / (stddev_radius * stddev_radius))

    # Helper function for converting a tuple to an array.
    def vec(x):
        return np.array(x)

    """
    Since a gaussian is unbounded, we need to limit ourselves
    to a finite range.
    We taper the ends off at the end of that range so they equal zero
    while preserving the maximum value of 1 at the mean.
    """
    zero_radius = max_radius + 1.0
    gauss_zero = gaussian(zero_radius * zero_radius)
    gauss_kernel_scale = 1 / (1 - gauss_zero)

    def gaussian_kernel_func(coordinate):
        x = coordinate[0] ** 2.0 + coordinate[1] ** 2.0
        x = gaussian(x)
        x -= gauss_zero
        x *= gauss_kernel_scale
        x = max(0.0, x)
        return x

    size = max_radius * 2 + 1
    kernel_center = max_radius
    kernel = np.zeros((size, size))

    for index in np.ndindex(kernel.shape):
        kernel[index] = gaussian_kernel_func(vec(index) - kernel_center)

    return kernel, kernel_center

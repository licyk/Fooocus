"""Anima model files and Fooocus feature compatibility."""

import functools
import os

import torch
from safetensors import safe_open

from ldm_patched.modules import model_management, utils
from ldm_patched.modules.anima import (
    AnimaCLIP,
    AnimaModel,
    AnimaModelConfig,
    AnimaVAE,
    detect_anima_config,
)
from ldm_patched.modules.model_patcher import ModelPatcher
from modules import config, flags
from modules.model_loader import load_file_from_url
from modules.util import get_file_from_folder_list

MODEL_URL = "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files"
PREFIXES = ("", "net.", "model.diffusion_model.", "diffusion_model.")
SIGNATURE = "llm_adapter.blocks.0.cross_attn.q_proj.weight"


@functools.lru_cache(maxsize=32)
def _is_anima_file(filename, mtime, size):
    if not filename.lower().endswith(".safetensors"):
        return False
    with safe_open(filename, framework="pt", device="cpu") as file:
        keys = set(file.keys())
    return any(
        prefix + SIGNATURE in keys and prefix + "x_embedder.proj.1.weight" in keys
        for prefix in PREFIXES
    )


def is_anima_file(filename):
    stat = os.stat(filename)
    return _is_anima_file(os.path.abspath(filename), stat.st_mtime_ns, stat.st_size)


def resolve_vae(vae_filename=None):
    if vae_filename is not None:
        return vae_filename
    return get_file_from_folder_list(config.anima_vae, config.path_vae)


def load_model(filename, vae_filename=None):
    vae_filename = resolve_vae(vae_filename)
    encoder_filename = get_file_from_folder_list(
        config.anima_text_encoder, config.path_text_encoders
    )
    for path, folder in [(encoder_filename, "text_encoders"), (vae_filename, "vae")]:
        if not os.path.isfile(path):
            load_file_from_url(
                url=f"{MODEL_URL}/{folder}/{os.path.basename(path)}",
                model_dir=os.path.dirname(path),
                file_name=os.path.basename(path),
            )

    sd = utils.load_torch_file(filename, safe_load=True)
    prefix = next(prefix for prefix in PREFIXES if prefix + SIGNATURE in sd)
    sd = {
        key[len(prefix) :]: value for key, value in sd.items() if key.startswith(prefix)
    }
    if any(
        value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        for value in sd.values()
    ):
        raise ValueError(
            "Use an unquantized FP16/BF16/FP32 Anima checkpoint; scaled FP8 and GGUF are not supported."
        )
    model_config = AnimaModelConfig(detect_anima_config(sd))
    dtype = model_management.unet_dtype(
        model_params=sum(value.numel() for value in sd.values())
    )
    model_config.unet_config["dtype"] = dtype
    model = AnimaModel(model_config, device=model_management.unet_offload_device())
    model.diffusion_model.load_state_dict(sd, strict=True)
    del sd
    unet = ModelPatcher(
        model,
        load_device=model_management.get_torch_device(),
        offload_device=model_management.unet_offload_device(),
    )
    clip = AnimaCLIP(utils.load_torch_file(encoder_filename, safe_load=True))
    vae = AnimaVAE(utils.load_torch_file(vae_filename, safe_load=True))
    return unet, clip, vae, vae_filename


def validate_sampling(sampler, scheduler):
    if sampler not in (
        "euler",
        "euler_ancestral",
        "heun",
        "heunpp2",
        "dpm_2",
        "lms",
        "dpmpp_2m",
        "ddim",
    ) or scheduler in (
        "lcm",
        "tcd",
        "edm_playground_v2.5",
        "turbo",
        "align_your_steps",
    ):
        raise ValueError(
            "Anima: use a flow-compatible sampler such as Euler with the simple scheduler."
        )


def prepare_task(task):
    filename = get_file_from_folder_list(task.base_model_name, config.paths_checkpoints)
    if not is_anima_file(filename):
        return False
    if task.performance_selection in (
        flags.Performance.EXTREME_SPEED,
        flags.Performance.LIGHTNING,
        flags.Performance.HYPER_SD,
    ):
        raise ValueError(
            "Anima: select Speed or Quality. LCM, Lightning and Hyper-SD use SDXL-only LoRAs."
        )
    validate_sampling(task.sampler_name, task.scheduler_name)
    image_prompt_active = task.input_image_checkbox and (
        task.current_tab == "ip"
        or task.mixing_image_prompt_and_vary_upscale
        or task.mixing_image_prompt_and_inpaint
    )
    if image_prompt_active and any(task.cn_tasks.values()):
        raise ValueError(
            "Anima cannot use SDXL Image Prompt, FaceSwap or ControlNet models. Use text prompts or Vary instead."
        )
    task.refiner_model_name = "None"
    task.refiner_swap_method = "joint"
    task.freeu_enabled = False
    task.inpaint_engine = "None"
    print("[Anima] SDXL refiner, FreeU and parameterized inpainter disabled.")
    return True

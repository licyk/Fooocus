# Anima

This fork supports the native Anima architecture using the model definitions and inference settings from ComfyUI. It loads split diffusion, text encoder and VAE files without requiring a ComfyUI installation.

Start with `python entry_with_update.py --preset anima`, or select the **anima** preset in the WebUI. The preset uses Euler, the simple scheduler, 40 steps and CFG 4.5. Anima Turbo checkpoints can use CFG 1 and 8–12 steps, configured through the existing advanced settings.

Model files from [CircleStone Labs](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files) belong in these directories:

| File | Fooocus directory |
| --- | --- |
| `anima-base-v1.0.safetensors` or another native Anima checkpoint | `models/checkpoints/` |
| `qwen_3_06b_base.safetensors` | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` |

The preset downloads its base checkpoint through the existing model downloader. Selecting Anima downloads its missing default text encoder and VAE when the model is first loaded. You can also install all three files manually. Custom locations can be configured with `path_checkpoints`, `path_text_encoders`, `path_vae`, `anima_text_encoder` and `anima_vae` in `config.txt`; the existing VAE dropdown overrides `anima_vae` for a generation.

Supported paths include text-to-image, negative prompts, weighted prompts, styles, wildcards, compatible Anima LoRAs, Vary, diffusion upscaling, [UNet Deep Shrink](deep-shrink.md), and standard mask-based inpaint/outpaint. Anima uses the full Qwen3 output and its own text adapter; CLIP Skip and SDXL sampling sharpness do not apply. Prompt weights are applied after the adapter, matching ComfyUI.

Select another Anima checkpoint in **Refiner (SDXL / SD 1.5 / Anima)** to refine an Anima base model. **Refiner Switch** controls the number of base steps before refinement. Joint switching keeps the same 16-channel noisy latent and uses the refiner's own text adapter. Separate switching resumes the existing flow latent without adding or scaling its noise again. VAE switching decodes the base image and re-encodes it for the refiner.

An Anima refiner also works after an SDXL base model: Fooocus automatically uses VAE conversion from SDXL's 4-channel latent to Anima's 16-channel latent, and encodes the same prompts with the refiner's Qwen3 text encoder. The second stage uses Anima's own flow schedule, with the remaining steps selected by Refiner Switch. CLIP tensors, SDXL noise schedules and the SDXL-to-SD1.5 interposer are not reused for Anima. Both models must use a supported Anima sampler/scheduler combination such as Euler/simple. An Anima base supports Anima refiners; an SDXL or SD1.5 refiner after an Anima base is currently rejected.

FreeU and the Fooocus SDXL inpaint patch are disabled when either stage uses Anima. Image Prompt, FaceSwap, SDXL ControlNet, LCM, Lightning and Hyper-SD require different model components and are rejected before downloading them. Use Speed or Quality. Anima's separate LLLite controls are not implemented. Scaled FP8 and GGUF checkpoints are not supported; use FP16, BF16 or FP32 weights.

Supported samplers are Euler, Euler ancestral (with rectified-flow noise scaling), Heun, Heun++, DPM2, LMS, DPM++ 2M and DDIM. Other ancestral/SDE samplers, UniPC, the SDXL Turbo scheduler and Align Your Steps are rejected; their legacy implementations need additional flow-specific changes. The `anima` preset selects Euler with the simple scheduler.

The implementation is adapted from ComfyUI commit `d49e888586dd8ae012c0667b33466b815fee07f7`: `comfy/ldm/anima/model.py`, `comfy/ldm/cosmos/predict2.py`, `comfy/ldm/cosmos/position_embedding.py`, `comfy/ldm/wan/vae.py`, `comfy/text_encoders/anima.py`, and the Anima model detection, flow sampling and latent normalization definitions. ComfyUI adaptations are GPL-3.0; NVIDIA Cosmos and Alibaba Wan source attribution is preserved in the imported modules.

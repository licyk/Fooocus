import copy
import os

import torch
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast
from transformers.initialization import no_init_weights

from ldm_patched.ldm.anima.model import Anima
from ldm_patched.ldm.anima.operations import AnimaOperations
from ldm_patched.ldm.anima.wan_vae import WanVAE
from ldm_patched.modules import conds, model_base, model_management, utils
from ldm_patched.modules.latent_formats import LatentFormat
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.sd1_clip import (
    escape_important,
    token_weights,
    unescape_important,
)
from modules.ops import use_patched_ops


class AnimaLatentFormat(LatentFormat):
    latent_channels = 16
    latent_rgb_factors = [
        [-0.1299, -0.1692, 0.2932],
        [0.0671, 0.0406, 0.0442],
        [0.3568, 0.2548, 0.1747],
        [0.0372, 0.2344, 0.1420],
        [0.0313, 0.0189, -0.0328],
        [0.0296, -0.0956, -0.0665],
        [-0.3477, -0.4059, -0.2925],
        [0.0166, 0.1902, 0.1975],
        [-0.0412, 0.0267, -0.1364],
        [-0.1293, 0.0740, 0.1636],
        [0.0680, 0.3019, 0.1128],
        [0.0032, 0.0581, 0.0639],
        [-0.1251, 0.0927, 0.1699],
        [0.0060, -0.0633, 0.0005],
        [0.3477, 0.2275, 0.2950],
        [0.1984, 0.0913, 0.1861],
    ]
    latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

    def __init__(self):
        self.mean = torch.tensor(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ]
        ).view(1, 16, 1, 1)
        self.std = torch.tensor(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ]
        ).view(1, 16, 1, 1)

    def process_in(self, latent):
        return (latent - self.mean.to(latent)) / self.std.to(latent)

    def process_out(self, latent):
        return latent * self.std.to(latent) + self.mean.to(latent)


class AnimaModelConfig:
    manual_cast_dtype = None
    sampling_settings = {"shift": 3.0}

    def __init__(self, config):
        self.unet_config = config
        self.latent_format = AnimaLatentFormat()


class AnimaModel(model_base.BaseModel):
    def __init__(self, config, device=None):
        super().__init__(
            config,
            model_type=model_base.ModelType.FLOW,
            device=device,
            unet_model=Anima,
            operations=AnimaOperations,
        )

    def extra_conds(self, **kwargs):
        context = kwargs["cross_attn"].to(
            device=kwargs["device"], dtype=self.get_dtype()
        )
        ids = kwargs["t5xxl_ids"].unsqueeze(0).to(context.device)
        weights = kwargs["t5xxl_weights"][None, :, None].to(context)
        context = self.diffusion_model.preprocess_text_embeds(
            context, ids, t5xxl_weights=weights
        )
        return {"c_crossattn": conds.CONDRegular(context)}


def detect_anima_config(sd):
    channels = sd["x_embedder.proj.1.weight"].shape[0]
    target_dim = sd["llm_adapter.embed.weight"].shape[1]
    model_dim = sd["llm_adapter.blocks.0.cross_attn.q_proj.weight"].shape[0]
    return dict(
        image_model="anima",
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=True,
        model_channels=channels,
        num_blocks=len({key.split(".")[1] for key in sd if key.startswith("blocks.")}),
        num_heads=channels // sd["blocks.0.self_attn.q_norm.weight"].shape[0],
        crossattn_emb_channels=target_dim,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        use_adaln_lora=True,
        adaln_lora_dim=sd["blocks.0.adaln_modulation_self_attn.1.weight"].shape[0],
        rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False,
        adapter_config=dict(
            source_dim=sd["llm_adapter.blocks.0.cross_attn.k_proj.weight"].shape[1],
            target_dim=target_dim,
            model_dim=model_dim,
            num_layers=len(
                {
                    key.split(".")[2]
                    for key in sd
                    if key.startswith("llm_adapter.blocks.")
                }
            ),
            num_heads=model_dim
            // sd["llm_adapter.blocks.0.cross_attn.q_norm.weight"].shape[0],
        ),
    )


class AnimaTokenizer:
    def __init__(self):
        root = os.path.join(os.path.dirname(__file__), "anima_tokenizers")
        self.qwen = Qwen2Tokenizer.from_pretrained(
            os.path.join(root, "qwen"), local_files_only=True
        )
        self.t5 = T5TokenizerFast.from_pretrained(
            os.path.join(root, "t5"), local_files_only=True
        )

    def tokenize_with_weights(self, text):
        qwen, t5 = [], []
        for segment, weight in token_weights(escape_important(text), 1.0):
            segment = unescape_important(segment)
            qwen.extend(self.qwen(segment, add_special_tokens=False)["input_ids"])
            t5.extend(
                (token, weight)
                for token in self.t5(segment, add_special_tokens=False)["input_ids"]
            )
        return {
            "qwen3_06b": qwen or [151643],
            "t5xxl": t5 + [(self.t5.eos_token_id, 1.0)],
        }


class AnimaTextEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        with use_patched_ops(AnimaOperations), no_init_weights():
            self.model = Qwen3Model(config)

    def forward(self, ids):
        return self.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).last_hidden_state


def detect_qwen_config(sd):
    head_dim = sd["model.layers.0.self_attn.q_norm.weight"].shape[0]
    return Qwen3Config(
        vocab_size=sd["model.embed_tokens.weight"].shape[0],
        hidden_size=sd["model.embed_tokens.weight"].shape[1],
        intermediate_size=sd["model.layers.0.mlp.gate_proj.weight"].shape[0],
        num_hidden_layers=len(
            {key.split(".")[2] for key in sd if key.startswith("model.layers.")}
        ),
        num_attention_heads=sd["model.layers.0.self_attn.q_proj.weight"].shape[0]
        // head_dim,
        num_key_value_heads=sd["model.layers.0.self_attn.k_proj.weight"].shape[0]
        // head_dim,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
    )


class AnimaCLIP:
    def __init__(self, sd, config=None):
        self.tokenizer = AnimaTokenizer()
        config = config or detect_qwen_config(sd)
        dtype = (
            torch.float32
            if model_management.text_encoder_device().type == "cpu"
            else model_management.text_encoder_dtype()
        )
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "Anima Qwen3 requires FP16, BF16 or FP32 text encoder precision."
            )
        self.cond_stage_model = AnimaTextEncoder(config).to(dtype).eval()
        self.cond_stage_model.load_state_dict(sd, strict=True)
        self.patcher = ModelPatcher(
            self.cond_stage_model,
            load_device=model_management.text_encoder_device(),
            offload_device=model_management.text_encoder_offload_device(),
        )
        self.fcs_cond_cache = {}

    def clone(self):
        result = copy.copy(self)
        result.patcher = self.patcher.clone()
        result.fcs_cond_cache = {}
        return result

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def encode(self, text):
        if text in self.fcs_cond_cache:
            return self.fcs_cond_cache[text]
        tokens = self.tokenizer.tokenize_with_weights(text)
        model_management.load_model_gpu(self.patcher)
        device = self.patcher.load_device
        context = (
            self.cond_stage_model(torch.tensor([tokens["qwen3_06b"]], device=device))
            .float()
            .cpu()
        )
        result = [
            [
                context,
                {
                    "t5xxl_ids": torch.tensor(
                        [token for token, _ in tokens["t5xxl"]], dtype=torch.long
                    ),
                    "t5xxl_weights": torch.tensor(
                        [weight for _, weight in tokens["t5xxl"]]
                    ),
                },
            ]
        ]
        self.fcs_cond_cache[text] = result
        return result


class AnimaVAE:
    downscale_ratio = 8
    latent_channels = 16

    def __init__(self, sd):
        if (
            "decoder.upsamples.0.upsamples.0.residual.2.weight" in sd
            or "decoder.head.0.gamma" not in sd
            or "conv2.weight" not in sd
            or sd["conv2.weight"].shape[1] != 16
        ):
            raise ValueError(
                "Anima requires the 16-channel Qwen-Image / Wan 2.1 VAE (qwen_image_vae.safetensors)."
            )
        self.first_stage_model = WanVAE(
            dim=sd["decoder.head.0.gamma"].shape[0],
            z_dim=16,
            temperal_downsample=[False, True, True],
        ).eval()
        self.vae_dtype = model_management.vae_dtype()
        self.first_stage_model.to(self.vae_dtype)
        self.first_stage_model.load_state_dict(sd, strict=True)
        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=model_management.vae_device(),
            offload_device=model_management.vae_offload_device(),
        )

    def encode(self, pixels):
        try:
            return self._encode(pixels)
        except torch.OutOfMemoryError:
            model_management.soft_empty_cache()
            print("[Anima] VAE encode memory limit reached; using tiles.")
            return self.encode_tiled(pixels, tile_x=256, tile_y=256, overlap=32)

    def _encode(self, pixels):
        memory = (
            1500
            * pixels.shape[1]
            * pixels.shape[2]
            * model_management.dtype_size(self.vae_dtype)
        )
        model_management.load_models_gpu([self.patcher], memory_required=memory)
        results = []
        for image in pixels.split(1):
            image = (
                image.to(device=self.patcher.load_device, dtype=self.vae_dtype)
                .movedim(-1, 1)
                .unsqueeze(2)
                * 2
                - 1
            )
            results.append(
                self.first_stage_model.encode(image).squeeze(2).float().cpu()
            )
        return torch.cat(results)

    def decode(self, latent):
        try:
            return self._decode(latent)
        except torch.OutOfMemoryError:
            model_management.soft_empty_cache()
            print("[Anima] VAE decode memory limit reached; using tiles.")
            return self.decode_tiled(latent, tile_x=32, tile_y=32, overlap=8)

    def _decode(self, latent):
        memory = (
            2200
            * latent.shape[2]
            * latent.shape[3]
            * 64
            * model_management.dtype_size(self.vae_dtype)
        )
        model_management.load_models_gpu([self.patcher], memory_required=memory)
        results = []
        for sample in latent.split(1):
            pixels = self.first_stage_model.decode(
                sample.to(
                    device=self.patcher.load_device, dtype=self.vae_dtype
                ).unsqueeze(2)
            )
            results.append(
                ((pixels.squeeze(2).movedim(1, -1) + 1) / 2).clamp(0, 1).float().cpu()
            )
        return torch.cat(results)

    def encode_tiled(self, pixels, tile_x=512, tile_y=512, overlap=64):
        if pixels.shape[1] <= tile_y and pixels.shape[2] <= tile_x:
            return self._encode(pixels)
        return utils.tiled_scale(
            pixels.movedim(-1, 1),
            lambda x: self._encode(x.movedim(1, -1)),
            tile_x,
            tile_y,
            overlap,
            upscale_amount=1 / 8,
            out_channels=16,
            output_device="cpu",
        )

    def decode_tiled(self, latent, tile_x=64, tile_y=64, overlap=16):
        if latent.shape[2] <= tile_y and latent.shape[3] <= tile_x:
            return self._decode(latent)
        return utils.tiled_scale(
            latent,
            lambda x: self._decode(x).movedim(-1, 1),
            tile_x,
            tile_y,
            overlap,
            upscale_amount=8,
            out_channels=3,
            output_device="cpu",
        ).movedim(1, -1)

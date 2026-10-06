# Adapted from ComfyUI (GPL-3.0), comfy/ldm/cosmos/predict2.py.
# Original: NVIDIA Cosmos-Predict2 (Apache-2.0).
import logging
import math
from typing import Callable, Optional, Tuple

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision import transforms

from ldm_patched.ldm.modules.attention import optimized_attention

from .operations import apply_rope, pad_to_patch_size
from .position_embedding import LearnablePosEmbAxis, VideoRopePosition3DEmb


class GPT2FeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device=None, dtype=None, operations=None
    ) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = operations.Linear(
            d_model, d_ff, bias=False, device=device, dtype=dtype
        )
        self.layer2 = operations.Linear(
            d_ff, d_model, bias=False, device=device, dtype=dtype
        )
        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    transformer_options: Optional[dict] = {},
    preferred_attention=None,
) -> torch.Tensor:
    heads = q_B_S_H_D.shape[-2]
    q, k, v = [tensor.flatten(-2) for tensor in (q_B_S_H_D, k_B_S_H_D, v_B_S_H_D)]
    return optimized_attention(q, k, v, heads)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        super().__init__()
        logging.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using {n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.q_proj = operations.Linear(
            query_dim, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.q_norm = operations.RMSNorm(
            self.head_dim, eps=1e-06, device=device, dtype=dtype
        )
        self.k_proj = operations.Linear(
            context_dim, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.k_norm = operations.RMSNorm(
            self.head_dim, eps=1e-06, device=device, dtype=dtype
        )
        self.v_proj = operations.Linear(
            context_dim, inner_dim, bias=False, device=device, dtype=dtype
        )
        self.v_norm = nn.Identity()
        self.output_proj = operations.Linear(
            inner_dim, query_dim, bias=False, device=device, dtype=dtype
        )
        self.output_dropout = nn.Dropout(dropout) if dropout > 0.0001 else nn.Identity()
        self.attn_op = torch_attention_op
        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = x if context is None else context
        q_input = x
        k_input = context
        v_input = context
        transformer_patches = transformer_options.get("patches", {})
        patch_name = "attn1_patch" if self.is_selfattn else "attn2_patch"
        if patch_name in transformer_patches:
            extra_options = transformer_options.copy()
            extra_options["n_heads"] = self.n_heads
            extra_options["dim_head"] = self.head_dim
            for patch in transformer_patches[patch_name]:
                out = patch(
                    q_input,
                    k_input,
                    v_input,
                    pe=rope_emb,
                    attn_mask=None,
                    extra_options=extra_options,
                )
                q_input = out.get("q", q_input)
                k_input = out.get("k", k_input)
                v_input = out.get("v", v_input)
                rope_emb = out.get("pe", rope_emb)
        q = self.q_proj(q_input)
        k = self.k_proj(k_input)
        v = self.v_proj(v_input)
        q, k, v = map(
            lambda t: rearrange(
                t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim
            ),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            rope_emb: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            v = self.v_norm(v)
            if self.is_selfattn and rope_emb is not None:
                q = apply_rope(self.q_norm(q), rope_emb)
                k = apply_rope(self.k_norm(k), rope_emb)
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)
            return (q, k, v)

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)
        return (q, k, v)

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        result = self.attn_op(q, k, v, transformer_options=transformer_options)
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(
            x, context, rope_emb=rope_emb, transformer_options=transformer_options
        )
        return self.compute_attention(q, k, v, transformer_options=transformer_options)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)
        return rearrange(
            emb, "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1]
        )


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_adaln_lora: bool = False,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        logging.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = operations.Linear(
            in_features,
            out_features,
            bias=not use_adaln_lora,
            device=device,
            dtype=dtype,
        )
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = operations.Linear(
                out_features, 3 * out_features, bias=False, device=device, dtype=dtype
            )
        else:
            self.linear_2 = operations.Linear(
                out_features, out_features, bias=False, device=device, dtype=dtype
            )

    def forward(
        self, sample: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb
        return (emb_B_T_D, adaln_lora_B_T_3D)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            operations.Linear(
                in_channels
                * spatial_patch_size
                * spatial_patch_size
                * temporal_patch_size,
                out_channels,
                bias=False,
                device=device,
                dtype=dtype,
            ),
        )
        self.dim = (
            in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0, (
            f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.layer_norm = operations.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-06
        )
        self.linear = operations.Linear(
            hidden_size,
            spatial_patch_size
            * spatial_patch_size
            * temporal_patch_size
            * out_channels,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    hidden_size, adaln_lora_dim, bias=False, device=device, dtype=dtype
                ),
                operations.Linear(
                    adaln_lora_dim,
                    self.n_adaln_chunks * hidden_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    hidden_size,
                    self.n_adaln_chunks * hidden_size,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D)
                + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)
        shift_B_T_1_1_D, scale_B_T_1_1_D = (
            rearrange(shift_B_T_D, "b t d -> b t 1 1 d"),
            rearrange(scale_B_T_D, "b t d -> b t 1 1 d"),
        )

        def _fn(
            _x_B_T_H_W_D: torch.Tensor,
            _norm_layer: nn.Module,
            _scale_B_T_1_1_D: torch.Tensor,
            _shift_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        x_B_T_H_W_D = _fn(
            x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D
        )
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


class Block(nn.Module):
    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = operations.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-06, device=device, dtype=dtype
        )
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.layer_norm_cross_attn = operations.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-06, device=device, dtype=dtype
        )
        self.cross_attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.layer_norm_mlp = operations.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-06, device=device, dtype=dtype
        )
        self.mlp = GPT2FeedForward(
            x_dim,
            int(x_dim * mlp_ratio),
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype
                ),
                operations.Linear(
                    adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype
                ),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype
                ),
                operations.Linear(
                    adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype
                ),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    x_dim, adaln_lora_dim, bias=False, device=device, dtype=dtype
                ),
                operations.Linear(
                    adaln_lora_dim, 3 * x_dim, bias=False, device=device, dtype=dtype
                ),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype
                ),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype
                ),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    x_dim, 3 * x_dim, bias=False, device=device, dtype=dtype
                ),
            )

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        transformer_options: Optional[dict] = {},
    ) -> torch.Tensor:
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb
        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)
            )
        shift_self_attn_B_T_1_1_D = rearrange(
            shift_self_attn_B_T_D, "b t d -> b t 1 1 d"
        )
        scale_self_attn_B_T_1_1_D = rearrange(
            scale_self_attn_B_T_D, "b t d -> b t 1 1 d"
        )
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")
        shift_cross_attn_B_T_1_1_D = rearrange(
            shift_cross_attn_B_T_D, "b t d -> b t 1 1 d"
        )
        scale_cross_attn_B_T_1_1_D = rearrange(
            scale_cross_attn_B_T_D, "b t d -> b t 1 1 d"
        )
        gate_cross_attn_B_T_1_1_D = rearrange(
            gate_cross_attn_B_T_D, "b t d -> b t 1 1 d"
        )
        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")
        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        result_B_T_H_W_D = rearrange(
            self.self_attn(
                rearrange(
                    normalized_x_B_T_H_W_D.to(compute_dtype), "b t h w d -> b (t h w) d"
                ),
                None,
                rope_emb=rope_emb_L_1_1_D,
                transformer_options=transformer_options,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = torch.addcmul(
            x_B_T_H_W_D,
            gate_self_attn_B_T_1_1_D.to(residual_dtype),
            result_B_T_H_W_D.to(residual_dtype),
        )

        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
            transformer_options: Optional[dict] = {},
        ) -> torch.Tensor:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D,
                layer_norm_cross_attn,
                _scale_cross_attn_B_T_1_1_D,
                _shift_cross_attn_B_T_1_1_D,
            )
            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(
                        _normalized_x_B_T_H_W_D.to(compute_dtype),
                        "b t h w d -> b (t h w) d",
                    ),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                    transformer_options=transformer_options,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
            transformer_options=transformer_options,
        )
        x_B_T_H_W_D = torch.addcmul(
            x_B_T_H_W_D,
            gate_cross_attn_B_T_1_1_D.to(residual_dtype),
            result_B_T_H_W_D.to(residual_dtype),
        )
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D
        ).to(compute_dtype)
        patches = transformer_options.get("patches", {})
        if "mlp_patch" in patches:
            args = {
                "x": normalized_x_B_T_H_W_D,
                "transformer_options": transformer_options,
            }
            for patch in patches["mlp_patch"]:
                args = patch(args)
            normalized_x_B_T_H_W_D = args["x"]
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = torch.addcmul(
            x_B_T_H_W_D,
            gate_mlp_B_T_1_1_D.to(residual_dtype),
            result_B_T_H_W_D.to(residual_dtype),
        )
        return x_B_T_H_W_D


class MiniTrainDIT(nn.Module):
    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        image_model=None,
        device=None,
        dtype=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.build_pos_embed(device=device, dtype=dtype)
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(
                model_channels,
                model_channels,
                use_adaln_lora=use_adaln_lora,
                device=device,
                dtype=dtype,
                operations=operations,
            ),
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    device=device,
                    dtype=dtype,
                    operations=operations,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.t_embedding_norm = operations.RMSNorm(
            model_channels, eps=1e-06, device=device, dtype=dtype
        )

    def build_pos_embed(self, device=None, dtype=None) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")
        logging.debug(
            f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}"
        )
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
            device=device,
        )
        self.pos_embedder = cls_type(**kwargs)
        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            kwargs["device"] = device
            kwargs["dtype"] = dtype
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.concat_padding_mask:
            if padding_mask is None:
                padding_mask = torch.zeros(
                    x_B_C_T_H_W.shape[0],
                    1,
                    x_B_C_T_H_W.shape[3],
                    x_B_C_T_H_W.shape[4],
                    dtype=x_B_C_T_H_W.dtype,
                    device=x_B_C_T_H_W.device,
                )
            else:
                padding_mask = transforms.functional.resize(
                    padding_mask,
                    list(x_B_C_T_H_W.shape[-2:]),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1),
                ],
                dim=1,
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(
                x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype
            )
        else:
            extra_pos_emb = None
        if "rope" in self.pos_emb_cls.lower():
            return (
                x_B_T_H_W_D,
                self.pos_embedder(x_B_T_H_W_D, fps=fps, device=x_B_C_T_H_W.device),
                extra_pos_emb,
            )
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(
            x_B_T_H_W_D, device=x_B_C_T_H_W.device
        )
        return (x_B_T_H_W_D, None, extra_pos_emb)

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self._forward(x, timesteps, context, fps, padding_mask, **kwargs)

    def _forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        orig_shape = list(x.shape)
        x = pad_to_patch_size(
            x, (self.patch_temporal, self.patch_spatial, self.patch_spatial)
        )
        x_B_C_T_H_W = x
        timesteps_B_T = timesteps
        crossattn_emb = context
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = (
            self.prepare_embedded_sequence(
                x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
            )
        )
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](
            self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
        )
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )
        transformer_options = kwargs.get("transformer_options", {})
        patches = transformer_options.get("patches", {})
        if "post_input" in patches:
            transformer_options = transformer_options.copy()
            transformer_options["model_patch_data"] = {}
        if "post_input" in patches:
            for patch in patches["post_input"]:
                out = patch(
                    {
                        "img": x_B_T_H_W_D,
                        "x": x_B_C_T_H_W,
                        "transformer_options": transformer_options,
                    }
                )
                x_B_T_H_W_D = out["img"]
        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            "transformer_options": transformer_options,
        }
        if x_B_T_H_W_D.dtype == torch.float16:
            x_B_T_H_W_D = x_B_T_H_W_D.float()
        for block_index, block in enumerate(self.blocks):
            transformer_options["block_index"] = block_index
            x_B_T_H_W_D = block(
                x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs
            )
        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D.to(crossattn_emb.dtype),
            t_embedding_B_T_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
        )
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)[
            :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
        ]
        return x_B_C_Tt_Hp_Wp

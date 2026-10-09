import torch

from ldm_patched.ldm.modules.attention import optimized_attention
from ldm_patched.modules.ops import manual_cast


class AnimaOperations(manual_cast):
    class RMSNorm(torch.nn.RMSNorm):
        ldm_patched_cast_weights = True

        def reset_parameters(self):
            pass

        def forward(self, x):
            return torch.nn.functional.rms_norm(
                x, self.normalized_shape, self.weight.to(x), self.eps
            )

    class Embedding(torch.nn.Embedding):
        ldm_patched_cast_weights = True

        def reset_parameters(self):
            pass

        def forward(self, ids, out_dtype=None):
            weight = self.weight.to(
                device=ids.device, dtype=out_dtype or self.weight.dtype
            )
            return torch.nn.functional.embedding(ids, weight, self.padding_idx)


def apply_rope(x, freqs):
    # Cosmos rotates the two halves of each head, rather than adjacent elements.
    paired = torch.stack(x.float().chunk(2, dim=-1), dim=-1)
    rotated = (
        freqs[..., 0] * paired[..., 0, None] + freqs[..., 1] * paired[..., 1, None]
    )
    return torch.cat((rotated[..., 0], rotated[..., 1]), dim=-1).to(x.dtype)


def pad_to_patch_size(x, patch_size):
    padding = ()
    for size, patch in zip(x.shape[2:], patch_size):
        padding = (0, (-size) % patch) + padding
    return torch.nn.functional.pad(x, padding, mode="circular")


def torch_cat_if_needed(tensors, dim):
    tensors = [tensor for tensor in tensors if tensor is not None]
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=dim)


def vae_attention():
    def attention(q, k, v):
        shape = q.shape
        q, k, v = [x.flatten(2).transpose(1, 2) for x in (q, k, v)]
        return optimized_attention(q, k, v, 1).transpose(1, 2).reshape(shape)

    return attention

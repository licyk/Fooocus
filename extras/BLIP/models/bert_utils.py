"""Attention helpers for BLIP's custom BERT architecture."""

import torch


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size, dtype=torch.bool)
    for head in heads:
        index = head - sum(previous < head for previous in already_pruned_heads)
        mask[index] = False
    return heads, torch.arange(mask.numel())[mask.flatten()]


class BertModelUtils:
    def invert_attention_mask(self, attention_mask):
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None, :, :]
        return (1.0 - attention_mask.to(self.dtype)) * torch.finfo(self.dtype).min

    def get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is None:
            return [None] * num_hidden_layers
        if head_mask.ndim == 1:
            head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.ndim == 2:
            head_mask = head_mask[:, None, :, None, None]
        return head_mask.to(self.dtype)

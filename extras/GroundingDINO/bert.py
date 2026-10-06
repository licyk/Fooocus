import torch
from transformers import BertModel


class GroundingDinoBertModel(BertModel):
    """Use GroundingDINO's per-token attention masks with Transformers 5."""

    def forward(self, *args, attention_mask=None, **kwargs):
        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = (1.0 - attention_mask[:, None].to(self.dtype)) * torch.finfo(self.dtype).min
        return super().forward(*args, attention_mask=attention_mask, **kwargs)


def wrap_bert_model(bert_model):
    # Retain the loaded parameters and checkpoint names when replacing the
    # package's wrapper, which uses BERT interfaces removed in Transformers 5.
    bert_model.__class__ = GroundingDinoBertModel
    bert_model.set_attn_implementation('eager')
    return bert_model

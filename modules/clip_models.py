"""Keep Fooocus checkpoint and LoRA paths around the Transformers CLIP encoders."""

import torch
from transformers import CLIPTextModel, CLIPVisionModel


class FooocusCLIPTextModel(torch.nn.Module):
    """Preserve ``text_model.*`` keys used by SDXL and its text encoder LoRAs."""

    def __init__(self, config):
        super().__init__()
        self.text_model = CLIPTextModel(config)

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        self.text_model.set_input_embeddings(embeddings)

    def forward(self, *args, **kwargs):
        return self.text_model(*args, **kwargs)


class FooocusCLIPVisionModel(torch.nn.Module):
    """Preserve ``vision_model.*`` keys in the existing safety checker weights."""

    def __init__(self, config):
        super().__init__()
        self.vision_model = CLIPVisionModel(config)

    def forward(self, *args, **kwargs):
        return self.vision_model(*args, **kwargs)

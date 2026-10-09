"""All WD14 extension taggers, with a bounded, thread-safe ONNX session cache."""

import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

from extras.wd14_tagger.interrogator import Interrogator
from extras.wd14_tagger.models import DEFAULT_MODEL, MODELS

KAOMOJI = "0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||"
CATEGORIES = [
    ("General tags", "general"),
    ("Character tags", "character"),
    ("Copyright tags", "copyright"),
    ("Artist tags", "artist"),
    ("Meta tags", "meta"),
    ("Quality tags", "quality"),
    ("Model tags", "model"),
]
DEFAULTS = {
    "model_name": DEFAULT_MODEL,
    "threshold": 0.35,
    "character_threshold": 0.85,
    "categories": [value for _, value in CATEGORIES],
    "additional_tags": "",
    "exclude_tags": "",
    "sort_by_alphabetical_order": False,
    "add_confident_as_weight": False,
    "replace_underscore": True,
    "replace_underscore_excludes": KAOMOJI,
    "escape_tag": True,
    "use_cpu": False,
    "unload_model_after_running": False,
    "show_confidence": False,
}


def split_str(value):
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def execution_providers(use_cpu):
    from args_manager import args

    if use_cpu or args.always_cpu:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [
        name
        for name in (
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
        )
        if name in available
    ]
    if args.gpu_device_id is not None:
        providers = [
            (name, {"device_id": args.gpu_device_id})
            if name
            in (
                "CUDAExecutionProvider",
                "ROCMExecutionProvider",
                "DmlExecutionProvider",
            )
            else name
            for name in providers
        ]
    return providers + ["CPUExecutionProvider"]


class TaggerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.key = None
        self.tagger = None

    def unload(self):
        with self.lock:
            if self.tagger is not None:
                self.tagger.unload()
            self.tagger = self.key = None

    def interrogate(
        self, image, model_name, use_cpu=False, unload_model_after_running=False
    ):
        if model_name not in MODELS:
            raise ValueError(f"Unknown tagger model: {model_name}")
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        providers = execution_providers(use_cpu)
        key = (model_name, repr(providers))
        with self.lock:
            if key != self.key:
                self.unload()
                from extras.wd14_tagger.cl import CLTaggerInterrogator
                from extras.wd14_tagger.wd14 import WaifuDiffusionInterrogator
                from modules.config import path_clip_vision

                spec = MODELS[model_name]
                cls = (
                    CLTaggerInterrogator
                    if spec["kind"] == "cl"
                    else WaifuDiffusionInterrogator
                )
                self.tagger = cls(model_name, spec, path_clip_vision, providers)
                self.key = key
            try:
                ratings, tags = self.tagger.interrogate(image)
                return ratings, tags, self.tagger.categories()
            except Exception:
                self.unload()
                raise
            finally:
                if unload_model_after_running:
                    self.unload()


manager = TaggerManager()


def interrogate_image(image_rgb, **options):
    settings = DEFAULTS | options
    ratings, tags, categories = manager.interrogate(
        image_rgb,
        settings["model_name"],
        settings["use_cpu"],
        settings["unload_model_after_running"],
    )
    selected = {
        tag: confidence
        for tag, confidence in tags.items()
        if categories.get(tag, "general") in settings["categories"]
        and confidence
        >= (
            settings["character_threshold"]
            if categories.get(tag) == "character"
            else settings["threshold"]
        )
    }
    processed = Interrogator.postprocess_tags(
        selected,
        threshold=0,
        additional_tags=split_str(settings["additional_tags"]),
        exclude_tags=split_str(settings["exclude_tags"]),
        sort_by_alphabetical_order=settings["sort_by_alphabetical_order"],
        add_confident_as_weight=settings["add_confident_as_weight"],
        replace_underscore=settings["replace_underscore"],
        replace_underscore_excludes=split_str(settings["replace_underscore_excludes"]),
        escape_tag=settings["escape_tag"],
    )
    return ", ".join(processed), ratings, processed


def default_interrogator(
    image_rgb, threshold=0.35, character_threshold=0.85, exclude_tags="", **options
):
    """Keep the existing string-returning API for scripts and other callers."""
    return interrogate_image(
        image_rgb,
        threshold=threshold,
        character_threshold=character_threshold,
        exclude_tags=exclude_tags,
        **options,
    )[0]

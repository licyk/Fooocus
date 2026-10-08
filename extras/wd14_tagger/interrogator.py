"""Shared ONNX adapter and original extension tag formatting."""

import re
from collections.abc import Sequence
from pathlib import Path

import onnxruntime as ort

from modules.model_loader import load_file_from_url

tag_escape_pattern = re.compile(r"([\\()])")


class Interrogator:
    def __init__(self, name, spec, model_dir, providers):
        self.name = name
        self.spec = spec
        self.model_dir = Path(model_dir)
        self.providers = providers
        self.model = None

    def download(self):
        # Retain the original Fooocus MOAT cache when the complete pair exists.
        legacy = [
            self.model_dir / f"wd-v1-4-moat-tagger-v2.{ext}" for ext in ("onnx", "csv")
        ]
        if self.name == "wd14-moat-v2" and all(path.is_file() for path in legacy):
            return legacy
        folder = self.model_dir / "taggers" / self.name
        paths = []
        for key in ("model_path", "tags_path"):
            filename = self.spec[key]
            paths.append(
                Path(
                    load_file_from_url(
                        url=f"https://huggingface.co/{self.spec['repo_id']}/resolve/{self.spec['revision']}/{filename}",
                        model_dir=str(folder),
                        file_name=Path(filename).name,
                    )
                )
            )
        return paths

    def load(self):
        model_path, tags_path = self.download()
        self.load_labels(tags_path)
        model = ort.InferenceSession(str(model_path), providers=self.providers)
        shape = model.get_inputs()[0].shape
        spatial = shape[2:4] if self.spec["kind"] == "cl" else shape[1:3]
        # The CL export exposes symbolic height/width. Its source pipeline uses
        # 448px for those dimensions; honor concrete dimensions when supplied.
        if self.spec["kind"] == "cl":
            spatial = [dim if isinstance(dim, int) else 448 for dim in spatial]
        if len(shape) != 4 or any(
            not isinstance(dim, int) or dim <= 0 for dim in spatial
        ):
            raise ValueError(f"Unsupported tagger input shape: {shape}")
        self.image_size = (spatial[1], spatial[0])
        if self.spec["kind"] == "wd" and spatial[0] != spatial[1]:
            raise ValueError(f"WD taggers require square input: {shape}")
        self.model = model

    def unload(self):
        self.model = None
        if hasattr(self, "tags"):
            del self.tags

    @staticmethod
    def postprocess_tags(
        tags: dict[str, float],
        threshold=0.35,
        additional_tags: Sequence[str] = (),
        exclude_tags: Sequence[str] = (),
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: Sequence[str] = (),
        escape_tag=False,
    ) -> dict[str, float]:
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c
            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order,
            )
            # filter tags
            if (c >= threshold and t not in exclude_tags)
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace("_", " ")

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r"\\\1", new_tag)

            if add_confident_as_weight:
                new_tag = f"({new_tag}:{tags[tag]})"

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

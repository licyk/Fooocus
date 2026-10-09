"""WD ONNX preprocessing/inference ported from tagger/wd14.py.

Use CSV categories rather than slice boundaries so models with no characters or
non-contiguous categories also work. No pandas or A1111 runtime is required.
"""

import csv

import numpy as np
from PIL import Image

from . import dbimutils
from .interrogator import Interrogator


class WaifuDiffusionInterrogator(Interrogator):
    def load_labels(self, path):
        with open(path, encoding="utf-8-sig", newline="") as file:
            self.tags = list(csv.DictReader(file))
        if not self.tags or any(
            "name" not in row or "category" not in row for row in self.tags
        ):
            raise ValueError("Invalid WD tag CSV")

    def categories(self):
        categories = {
            "0": "general",
            "1": "artist",
            "3": "copyright",
            "4": "character",
            "5": "meta",
            "9": "rating",
        }
        return {
            row["name"]: categories.get(row["category"], "general") for row in self.tags
        }

    def preprocess_image(self, image):
        height = self.image_size[1]
        # Original extension: alpha over white, then RGB -> BGR, square padding
        # and OpenCV area/cubic resizing, without normalization.
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = np.asarray(new_image.convert("RGB"))[:, :, ::-1]
        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        return np.expand_dims(image.astype(np.float32), 0)

    def interrogate(self, image):
        if self.model is None:
            self.load()
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        probs = self.model.run(
            [output_name], {input_name: self.preprocess_image(image)}
        )[0][0]
        if len(probs) != len(self.tags):
            raise ValueError("WD model output and tag CSV have different lengths")
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        categories = self.categories()
        ratings, tags = {}, {}
        for row, probability in zip(self.tags, probs):
            target = ratings if categories[row["name"]] == "rating" else tags
            target[row["name"]] = float(probability)
        return ratings, tags

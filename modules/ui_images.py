import numpy as np
from PIL import Image


def editor_to_inpaint(value):
    """Convert ImageEditor layers into the image and mask used by the worker."""
    if value is None:
        return None

    background = value.get('background')
    if background is None:
        background = value.get('composite')
    if background is None:
        return None

    image = Image.fromarray(np.asarray(background)).convert('RGBA')
    opaque = Image.new('RGBA', image.size, 'white')
    opaque.alpha_composite(image)
    rgb = np.asarray(opaque.convert('RGB'))
    mask = Image.new('RGBA', image.size)
    for layer in value.get('layers') or []:
        if layer is not None:
            layer = Image.fromarray(np.asarray(layer)).convert('RGBA')
            if layer.size != image.size:
                layer = layer.resize(image.size, Image.Resampling.NEAREST)
            mask.alpha_composite(layer)
    alpha = np.asarray(mask.getchannel('A'))
    return {'image': rgb, 'mask': np.repeat(alpha[:, :, None], 3, axis=2)}

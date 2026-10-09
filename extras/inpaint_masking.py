"""Mask crop and fill helpers ported from Forge Classic modules/masking.py (AGPL-3.0)."""

from PIL import Image, ImageFilter, ImageOps


def resize_image(mode, image, width, height):
    """Forge images.resize_image modes 0/1/2, without a learned upscaler."""

    def resize(im, w, h):
        return im.resize((w, h), Image.Resampling.LANCZOS)

    if mode == 0:
        return resize(image, width, height)
    ratio = width / height
    src_ratio = image.width / image.height
    if mode == 1:
        src_w = width if ratio > src_ratio else image.width * height // image.height
        src_h = height if ratio <= src_ratio else image.height * width // image.width
        resized = resize(image, src_w, src_h)
        result = Image.new(image.mode, (width, height))
        result.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return result
    src_w = width if ratio < src_ratio else image.width * height // image.height
    src_h = height if ratio >= src_ratio else image.height * width // image.width
    resized = resize(image, src_w, src_h)
    result = Image.new(image.mode, (width, height))
    result.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        if fill_height > 0:
            result.paste(
                resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0)
            )
            result.paste(
                resized.resize(
                    (width, fill_height), box=(0, resized.height, width, resized.height)
                ),
                box=(0, fill_height + src_h),
            )
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        if fill_width > 0:
            result.paste(
                resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0)
            )
            result.paste(
                resized.resize(
                    (fill_width, height), box=(resized.width, 0, resized.width, height)
                ),
                box=(fill_width + src_w, 0),
            )
    return result


def get_crop_region_v2(mask, pad=0):
    """
    Finds a rectangular region that contains all masked ares in a mask.
    Returns None if mask is completely black mask (all 0)

    Parameters:
    mask: PIL.Image.Image L mode or numpy 1d array
    pad: int number of pixels that the region will be extended on all sides
    Returns: (x1, y1, x2, y2) | None

    Introduced post 1.9.0
    """
    mask = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
    if box := mask.getbbox():
        x1, y1, x2, y2 = box
        return (
            (
                max(x1 - pad, 0),
                max(y1 - pad, 0),
                min(x2 + pad, mask.size[0]),
                min(y2 + pad, mask.size[1]),
            )
            if pad
            else box
        )


def get_crop_region(mask, pad=0):
    """
    Same function as get_crop_region_v2 but handles completely black mask (all 0) differently
    when mask all black still return coordinates but the coordinates may be invalid ie x2>x1 or y2>y1
    Notes: it is possible for the coordinates to be "valid" again if pad size is sufficiently large
    (mask_size.x-pad, mask_size.y-pad, pad, pad)

    Extension developer should use get_crop_region_v2 instead unless for compatibility considerations.
    """
    mask = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
    if box := get_crop_region_v2(mask, pad):
        return box
    x1, y1 = mask.size
    x2 = y2 = 0
    return (
        max(x1 - pad, 0),
        max(y1 - pad, 0),
        min(x2 + pad, mask.size[0]),
        min(y2 + pad, mask.size[1]),
    )


def expand_crop_region(
    crop_region, processing_width, processing_height, image_width, image_height
):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2 - y1))
        y1 -= desired_height_diff // 2
        y2 += desired_height_diff - desired_height_diff // 2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        y2 = min(image_height, y2)
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2 - x1))
        x1 -= desired_width_diff // 2
        x2 += desired_width_diff - desired_width_diff // 2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        x2 = min(image_width, x2)

    return x1, y1, x2, y2


def fill(image, mask):
    """fills masked regions with colors from image using blur. Not extremely effective."""

    image_mod = Image.new("RGBA", (image.width, image.height))

    image_masked = Image.new("RGBa", (image.width, image.height))
    image_masked.paste(
        image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert("L"))
    )

    image_masked = image_masked.convert("RGBa")

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")

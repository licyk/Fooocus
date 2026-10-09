"""Shared manual/automatic image description with selectable taggers."""

import gradio as gr

from extras.wd14tagger import DEFAULTS
from modules import flags
from modules.localization import translate


def describe_image(modes, img, apply_styles, *tagger_options):
    if img is None:
        raise gr.Error(translate("Upload an image before describing it."))
    settings = DEFAULTS | dict(zip(DEFAULTS, tagger_options))
    prompts, styles = [], set()
    ratings = tags = None
    if flags.describe_type_photo in modes:
        from extras.interrogate import default_interrogator

        prompts.append(default_interrogator(img))
        styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])
    if flags.describe_type_anime in modes:
        from extras.wd14tagger import interrogate_image

        try:
            text, ratings, tags = interrogate_image(img, **settings)
        except Exception as error:
            raise gr.Error(
                translate("Tagger failed: {error}", {"error": str(error)})
            ) from error
        prompts.append(text)
        styles.update(["Fooocus V2", "Fooocus Masterpiece"])
    if not settings["show_confidence"]:
        ratings = tags = None
    return (
        ", ".join(prompts) if prompts else gr.update(),
        sorted(styles) if styles and apply_styles else gr.update(),
        ratings,
        tags,
    )

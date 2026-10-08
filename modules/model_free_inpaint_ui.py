"""Gradio controls for the Forge Classic inpainting port."""

import gradio as gr

from modules.localization import translate_choices
from modules.model_free_inpaint import AREAS, BACKENDS, CONTENTS, DEFAULTS, MASK_MODES


def build_settings(default_backend="standard"):
    controls = {}
    controls["backend"] = gr.Radio(
        label="Inpaint implementation",
        choices=translate_choices(BACKENDS),
        value=default_backend,
        elem_id="inpaint_backend",
        info="Standard inpaint uses the selected model and VAE. No dedicated inpaint model is downloaded. These settings also apply to enhancement inpainting.",
    )
    with gr.Accordion(
        "Standard inpaint settings",
        open=True,
        elem_id="standard_inpaint_settings",
        visible=default_backend == "standard",
    ) as settings_panel:
        controls["mask_blur"] = gr.Slider(
            label="Mask blur",
            minimum=0,
            maximum=64,
            step=1,
            value=4,
            elem_id="inpaint_mask_blur",
        )
        controls["mask_transparency"] = gr.Slider(
            label="Mask transparency",
            minimum=0,
            maximum=100,
            step=1,
            value=50,
            elem_id="inpaint_mask_transparency",
            info="Display transparency of the drawn mask. It does not change the generated mask strength.",
        )
        controls["mask_mode"] = gr.Radio(
            label="Mask mode",
            choices=translate_choices(MASK_MODES),
            value="masked",
            elem_id="inpaint_mask_mode",
        )
        controls["content"] = gr.Radio(
            label="Masked content",
            choices=translate_choices(CONTENTS),
            value="original",
            elem_id="inpaint_masked_content",
        )
        controls["padding"] = gr.Slider(
            label="Only masked padding, pixels",
            minimum=0,
            maximum=256,
            step=8,
            value=32,
            elem_id="inpaint_padding",
            info="Context padding on all four sides of the masked region. Used only with Only masked.",
        )
        controls["area"] = gr.Radio(
            label="Inpaint area",
            choices=translate_choices(AREAS),
            value="masked",
            elem_id="inpaint_area",
        )
        with gr.Accordion(
            "Soft inpainting", open=False, elem_id="soft_inpaint_settings"
        ):
            controls["soft_enabled"] = gr.Checkbox(
                label="Enable soft inpainting",
                value=False,
                elem_id="soft_inpaint_enabled",
                info="Blend original and generated content using mask opacity, noise level and latent differences. Larger mask blur values are recommended.",
            )
            with gr.Column(
                visible=False, elem_id="soft_inpaint_parameters"
            ) as soft_panel:
                specs = [
                    (
                        "mask_blend_power",
                        "Schedule bias",
                        0,
                        8,
                        0.1,
                        "Shifts when preservation of original content occurs during denoising.",
                    ),
                    (
                        "mask_blend_scale",
                        "Preservation strength",
                        0,
                        8,
                        0.05,
                        "How strongly partially masked content should be preserved.",
                    ),
                    (
                        "inpaint_detail_preservation",
                        "Transition contrast boost",
                        1,
                        32,
                        0.5,
                        "Amplifies the contrast that may be lost in partially masked regions.",
                    ),
                    (
                        "composite_mask_influence",
                        "Mask influence",
                        0,
                        1,
                        0.05,
                        "How strongly the original mask should bias the difference threshold.",
                    ),
                    (
                        "composite_difference_threshold",
                        "Difference threshold",
                        0,
                        8,
                        0.25,
                        "How much an image region can change before the original pixels are not blended in anymore.",
                    ),
                    (
                        "composite_difference_contrast",
                        "Difference contrast",
                        0,
                        8,
                        0.25,
                        "How sharp the transition should be between blended and not blended.",
                    ),
                ]
                for key, label, low, high, step, info in specs:
                    controls[key] = gr.Slider(
                        label=label,
                        minimum=low,
                        maximum=high,
                        step=step,
                        value=DEFAULTS[key],
                        info=info,
                        elem_id="soft_inpaint_" + key,
                    )
            controls["soft_enabled"].change(
                lambda value: gr.update(visible=value),
                inputs=controls["soft_enabled"],
                outputs=soft_panel,
                queue=False,
                show_progress="hidden",
            )
    controls["backend"].change(
        lambda value: gr.update(visible=value == "standard"),
        inputs=controls["backend"],
        outputs=settings_panel,
        queue=False,
        show_progress="hidden",
    )
    controls["mask_transparency"].change(
        lambda value: None,
        inputs=controls["mask_transparency"],
        js="(value) => {window.setInpaintMaskTransparency(value); return value;}",
        queue=False,
        show_progress="hidden",
    )
    return [controls[key] for key in DEFAULTS]

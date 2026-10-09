"""Model selection and single-image options from the WD14 Tagger extension."""

import gradio as gr

from extras.wd14tagger import CATEGORIES, DEFAULTS, MODELS, manager
from modules.localization import translate, translate_choices


def build_settings(default_model, visible):
    controls = {}
    with gr.Column(visible=visible, elem_id="describe_tagger_settings") as panel:
        controls["model_name"] = gr.Dropdown(
            label="Tagger model",
            choices=sorted(MODELS),
            value=default_model,
            elem_id="describe_tagger_model",
            info="Models are downloaded on first use. Settings also apply to automatic Art/Anime description.",
        )
        with gr.Accordion(
            "Tagger parameters", open=False, elem_id="describe_tagger_parameters"
        ):
            controls["threshold"] = gr.Slider(
                label="General tag threshold",
                minimum=0,
                maximum=1,
                step=0.01,
                value=DEFAULTS["threshold"],
                elem_id="describe_tagger_threshold",
            )
            controls["character_threshold"] = gr.Slider(
                label="Character tag threshold",
                minimum=0,
                maximum=1,
                step=0.01,
                value=DEFAULTS["character_threshold"],
                elem_id="describe_tagger_character_threshold",
            )
            controls["categories"] = gr.CheckboxGroup(
                label="Tag categories",
                choices=translate_choices(CATEGORIES),
                value=DEFAULTS["categories"],
                info="WD models return general and character tags. CL Tagger also supports copyright, artist, meta, quality and model categories.",
                elem_id="describe_tagger_categories",
            )
            controls["additional_tags"] = gr.Textbox(
                label="Additional tags (split by comma)",
                value="",
                elem_id="describe_tagger_additional_tags",
            )
            controls["exclude_tags"] = gr.Textbox(
                label="Exclude tags (split by comma)",
                value="",
                elem_id="describe_tagger_exclude_tags",
            )
            controls["sort_by_alphabetical_order"] = gr.Checkbox(
                label="Sort by alphabetical order",
                value=False,
                elem_id="describe_tagger_sort",
            )
            controls["add_confident_as_weight"] = gr.Checkbox(
                label="Include tag confidence as prompt weight",
                value=False,
                elem_id="describe_tagger_weights",
            )
            controls["replace_underscore"] = gr.Checkbox(
                label="Use spaces instead of underscore",
                value=True,
                elem_id="describe_tagger_spaces",
            )
            controls["replace_underscore_excludes"] = gr.Textbox(
                label="Keep underscores in these tags (split by comma)",
                value=DEFAULTS["replace_underscore_excludes"],
                elem_id="describe_tagger_keep_underscores",
            )
            controls["escape_tag"] = gr.Checkbox(
                label="Escape brackets and backslashes",
                value=True,
                elem_id="describe_tagger_escape",
            )
            controls["use_cpu"] = gr.Checkbox(
                label="Run tagger on CPU", value=False, elem_id="describe_tagger_cpu"
            )
            controls["unload_model_after_running"] = gr.Checkbox(
                label="Unload model after running",
                value=False,
                elem_id="describe_tagger_auto_unload",
            )
            controls["show_confidence"] = gr.Checkbox(
                label="Show rating and tag confidence",
                value=False,
                elem_id="describe_tagger_show_confidence",
            )
            unload = gr.Button("Unload tagger model", elem_id="describe_tagger_unload")
            status = gr.Markdown(elem_id="describe_tagger_status")

            def unload_model():
                manager.unload()
                return translate("Tagger model unloaded.")

            unload.click(
                unload_model, outputs=status, queue=True, show_progress="hidden"
            )
        ratings = gr.Label(
            label="Rating confidence", visible=False, elem_id="describe_tagger_ratings"
        )
        tags = gr.Label(
            label="Tag confidence", visible=False, elem_id="describe_tagger_confidence"
        )
        controls["show_confidence"].change(
            lambda enabled: [gr.update(visible=enabled), gr.update(visible=enabled)],
            inputs=controls["show_confidence"],
            outputs=[ratings, tags],
            queue=False,
            show_progress="hidden",
        )
    return [controls[key] for key in DEFAULTS], panel, ratings, tags

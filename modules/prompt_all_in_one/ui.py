"""Native Gradio preferences; generation continues to use original Textboxes."""

import json

import gradio as gr

from modules.localization import translate_choices

from .api import services
from .config import DEFAULTS, FIELDS, validate_settings


def build_settings():
    store, _ = services()
    defaults = store.load()
    fields = tuple(
        field
        for field in FIELDS
        if field[0]
        in {
            "enabled",
            "positive",
            "negative",
            "inpaint",
            "enhance",
            "hide_native",
            "history",
            "history_limit",
        }
    )
    controls = []
    with gr.Accordion(
        "Prompt All-in-One", open=False, elem_id="prompt_all_in_one_settings"
    ):
        gr.Markdown(
            "Full upstream Prompt All-in-One editor. Language, translation, formatting, mouse shortcuts, blacklist and themes are configured in its original toolbar and dialogs. "
            "Preferences apply to this browser. The original prompts remain the generation inputs. "
            "External services require configuration in the editor."
        )
        status = gr.Markdown(
            "Loading Prompt All-in-One preferences...", elem_id="prompt_all_in_one_status"
        )
        for key, label, _, choices in fields:
            arguments = {
                "label": label,
                "value": defaults[key],
                "elem_id": f"prompt_all_in_one_{key}",
            }
            if choices is None:
                control = gr.Checkbox(**arguments)
            elif choices == "number":
                control = gr.Slider(**arguments, minimum=0.01, maximum=1, step=0.01)
            elif choices == "integer":
                control = gr.Slider(**arguments, minimum=10, maximum=1000, step=10)
            elif choices in ("text", "json"):
                control = gr.Textbox(**arguments, lines=4)
            elif choices == "color":
                control = gr.ColorPicker(**arguments)
            else:
                control = gr.Dropdown(**arguments, choices=translate_choices(choices))
            control.input(
                fn=None,
                inputs=[control],
                queue=False,
                show_progress="hidden",
                js=f"(value) => {{ window.FooocusPromptAllInOne?.setSetting('{key}', value); }}",
            )
            controls.append(control)
        payload = gr.JSON(value=defaults, visible="hidden")
        bridge = gr.Textbox(
            elem_id="prompt_all_in_one_settings_bridge", visible="hidden"
        )

        def to_ui(configuration):
            return [configuration[key] for key, *_ in fields] + [configuration]

        def restore_browser(raw):
            try:
                document = json.loads(raw)
                configuration = validate_settings(document["settings"])
                revision = document["revision"]
            except (ValueError, TypeError, KeyError):
                configuration = store.load()
                revision = 0
                gr.Info("Invalid editor preferences; restored server defaults.")
            result = to_ui(configuration)
            result[-1] = {**configuration, "_bridge_revision": revision}
            return result

        bridge.change(
            restore_browser,
            inputs=[bridge],
            outputs=[*controls, payload],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=None,
            inputs=[payload],
            queue=False,
            js="(config) => window.FooocusPromptAllInOne?.finishSettings(config)",
        )
        with gr.Accordion("Actions", open=False):
            actions = []
            for label, description in (
                ("Save as default", "Save these preferences for new browsers."),
                (
                    "Load server defaults",
                    "Apply the saved server defaults to this browser.",
                ),
                (
                    "Restore built-in defaults",
                    "Restore this browser's settings; keep saved history and favorites.",
                ),
            ):
                with gr.Row():
                    actions.append(gr.Button(label, scale=1))
                    gr.Markdown(description, scale=2)
        save, reload, reset = actions

        def save_defaults(*values):
            try:
                configuration = store.save(
                    {field[0]: value for field, value in zip(fields, values)}
                )
            except ValueError as error:
                raise gr.Error(str(error)) from error
            return configuration, "Editor defaults saved."

        save.click(
            save_defaults, inputs=controls, outputs=[payload, status], queue=False
        ).then(
            fn=None,
            inputs=[payload],
            queue=False,
            js="(config) => window.FooocusPromptAllInOne?.useSettings(config)",
        )
        for button, loader in ((reload, store.load), (reset, lambda: dict(DEFAULTS))):
            button.click(
                lambda loader=loader: to_ui(loader()),
                outputs=[*controls, payload],
                queue=False,
            ).then(
                fn=None,
                inputs=[payload],
                queue=False,
                js="(config) => window.FooocusPromptAllInOne?.useSettings(config)",
            )
    return controls


def bind_models(base_model):
    """Route original Extra Networks checkpoint clicks through Gradio state."""
    from modules import config

    bridge = gr.Textbox(elem_id="prompt_all_in_one_model_bridge", visible="hidden")

    def select_model(name):
        if name not in config.model_filenames:
            raise gr.Error("Model is no longer available; refresh the model list.")
        return gr.update(value=name)

    bridge.change(
        select_model,
        inputs=bridge,
        outputs=base_model,
        queue=False,
        show_progress="hidden",
    )

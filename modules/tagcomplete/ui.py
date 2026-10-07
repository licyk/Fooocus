"""Gradio controls for per-browser completion preferences and saved defaults."""

import json

import gradio as gr

from modules.localization import translate_choices

from .api import model_capabilities, services
from .config import DEFAULTS, FIELDS, validate_settings

CHOICE_LABELS = {"wildcard_mode": {"full": "Full wildcard path"}}


def build_settings():
    store, catalog, _ = services()
    defaults = store.load()
    datasets = catalog.get()["datasets"]
    controls = []
    gr.Markdown(
        "Settings apply to this browser immediately. **Save as default** changes defaults for new browsers. "
        "Add CSV/Chants files to the configured tags folder, then refresh the datasets."
    )
    status = gr.Markdown("Loading prompt completion…", elem_id="tagcomplete_status")
    groups = list(dict.fromkeys(field[4] for field in FIELDS))
    for group in groups:
        with gr.Accordion(group, open=group == "General"):
            for key, label, _, kind, field_group, bounds in FIELDS:
                if group != field_group:
                    continue
                value = defaults[key]
                args = {"label": label, "value": value, "elem_id": f"tagcomplete_{key}"}
                if kind == "bool":
                    control = gr.Checkbox(**args)
                elif kind in ("int", "float"):
                    control = gr.Slider(
                        **args,
                        minimum=bounds[0],
                        maximum=bounds[1],
                        step=1 if kind == "int" else 0.05,
                    )
                elif kind in ("csv", "json_file"):
                    suffix = ".csv" if kind == "csv" else ".json"
                    choices = ["None"] + [
                        row["name"] for row in datasets if row["name"].endswith(suffix)
                    ]
                    control = gr.Dropdown(
                        **args,
                        choices=translate_choices(list(dict.fromkeys([*choices, value]))),
                    )
                elif kind == "choice":
                    labels = CHOICE_LABELS.get(key, {})
                    choices = [(labels.get(choice, choice), choice) for choice in bounds]
                    control = gr.Dropdown(**args, choices=translate_choices(choices))
                elif kind == "categories":
                    control = gr.CheckboxGroup(
                        **args, choices=[str(i) for i in range(-1, 9)]
                    )
                else:
                    if kind == "object":
                        args["value"] = json.dumps(value, indent=2, ensure_ascii=False)
                    control = gr.Textbox(**args, lines=8 if kind == "object" else 1)
                control.change(
                    fn=None,
                    inputs=[control],
                    queue=False,
                    show_progress="hidden",
                    js=f"(value) => {{ window.FooocusTagComplete?.setSetting('{key}', value); }}",
                )
                controls.append(control)

    payload = gr.JSON(value=defaults, visible="hidden")
    bridge = gr.Textbox(elem_id="tagcomplete_settings_bridge", visible="hidden")

    def values_to_ui(config):
        return [
            json.dumps(config[key], indent=2, ensure_ascii=False)
            if kind == "object"
            else config[key]
            for key, _, _, kind, *_ in FIELDS
        ] + [config]

    def restore_browser(raw):
        try:
            config = validate_settings(json.loads(raw))
        except (ValueError, TypeError):
            config = store.load()
            gr.Info("Invalid saved browser preferences; restored completion defaults.")
        return values_to_ui(config)

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
        js="(config) => { window.FooocusTagComplete?.finishSettings(config); }",
    )

    actions = [
        (
            "tagcomplete_save",
            "Save as default",
            (
                "Save these settings as server defaults for new browsers. "
                "Existing browsers keep their preferences."
            ),
        ),
        (
            "tagcomplete_reload",
            "Load server defaults",
            "Apply the saved server defaults to this browser.",
        ),
        (
            "tagcomplete_reset",
            "Restore built-in defaults",
            (
                "Restore this browser to the built-in defaults without changing "
                "the saved server defaults."
            ),
        ),
        (
            "tagcomplete_refresh",
            "Refresh completion datasets",
            (
                "Rescan tags, translations, prompt snippets, models and wildcards "
                "after adding or editing files."
            ),
        ),
        (
            "tagcomplete_clear_usage",
            "Clear my completion usage",
            (
                "Delete completion usage for the current login and reset frequency ranking. "
                "Without login, this clears shared local usage."
            ),
        ),
    ]
    buttons = []
    with gr.Accordion("Actions", open=False, elem_id="tagcomplete_actions"):
        for identifier, label, description in actions:
            with gr.Row():
                buttons.append(
                    gr.Button(label, elem_id=identifier, scale=1, min_width=160)
                )
                gr.Markdown(description, scale=2, min_width=180)
    save, reload, reset, refresh, clear = buttons

    def save_defaults(*values):
        try:
            config = store.save(
                {field[0]: value for field, value in zip(FIELDS, values)}
            )
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return (
            config,
            "Completion defaults saved. Existing browsers retain their preferences.",
        )

    def refresh_datasets(*values):
        data = catalog.refresh()
        result = []
        for field, value in zip(FIELDS, values):
            kind = field[3]
            if kind in ("csv", "json_file"):
                suffix = ".csv" if kind == "csv" else ".json"
                choices = ["None"] + [
                    row["name"]
                    for row in data["datasets"]
                    if row["name"].endswith(suffix)
                ]
                result.append(
                    gr.update(
                        choices=translate_choices(choices),
                        value=value if value in choices else "None",
                    )
                )
            else:
                result.append(gr.update())
        return result

    save.click(
        save_defaults, inputs=controls, outputs=[payload, status], queue=False
    ).then(
        fn=None,
        inputs=[payload],
        queue=False,
        js="(config) => { window.FooocusTagComplete?.useSettings(config); }",
    )
    for button, loader in (
        (reload, store.load),
        (reset, lambda: validate_settings(DEFAULTS)),
    ):
        button.click(
            lambda loader=loader: values_to_ui(loader()),
            outputs=[*controls, payload],
            queue=False,
        ).then(
            fn=None,
            inputs=[payload],
            queue=False,
            js="(config) => { window.FooocusTagComplete?.useSettings(config); }",
        )
    refresh.click(
        refresh_datasets, inputs=controls, outputs=controls, queue=False
    ).then(
        fn=None,
        queue=False,
        js="async () => { await window.FooocusTagComplete?.refresh(); }",
    )
    clear.click(
        fn=None,
        queue=False,
        js="async () => { await window.FooocusTagComplete?.clearUsage(); }",
    )


def bind_context(blocks, base_model, refiner_model, styles):
    capabilities = gr.JSON(visible="hidden")
    for trigger in (base_model.change, refiner_model.change, blocks.load):
        trigger(
            model_capabilities,
            inputs=[base_model, refiner_model],
            outputs=[capabilities],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=None,
            inputs=[capabilities],
            queue=False,
            js="(value) => { window.FooocusTagComplete?.setCapabilities(value); }",
        )
    style_bridge = gr.Textbox(elem_id="tagcomplete_style_bridge", visible="hidden")

    def select_style(raw, selected):
        from modules.sdxl_styles import legal_style_names

        try:
            name = json.loads(raw)["name"]
        except (ValueError, KeyError, TypeError):
            raise gr.Error("Invalid style completion.") from None
        if name not in legal_style_names:
            raise gr.Error("Unknown Fooocus style.")
        return list(dict.fromkeys([*(selected or []), name]))

    style_bridge.change(
        select_style,
        inputs=[style_bridge, styles],
        outputs=[styles],
        queue=False,
        show_progress="hidden",
    )

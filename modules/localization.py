import json
import os
import re

current_translation = {}
localization_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'language')


def load_localization(filename):
    """Load the dictionary before constructing controls or serving scripts."""
    global current_translation
    current_translation = {}

    if isinstance(filename, str):
        full_name = os.path.abspath(os.path.join(localization_root, filename + '.json'))
        if os.path.exists(full_name):
            try:
                with open(full_name, encoding='utf-8') as f:
                    translation = json.load(f)
                    if not isinstance(translation, dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in translation.items()
                    ):
                        raise ValueError('Translations must be a dictionary of strings')
                    current_translation = translation
            except (OSError, ValueError) as e:
                print(str(e))
                print(f'Failed to load localization file {full_name}')

    return current_translation


def translate(text, params=None):
    """Translate a UI message, replacing only explicitly supplied parameters."""
    result = current_translation.get(text, text)
    if params is not None:
        result = re.sub(
            r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}',
            lambda match: str(params[match[1]]) if match[1] in params else match[0],
            result,
        )
    return result


def translate_choices(choices):
    """Translate display labels while retaining Gradio's original choice values."""
    return [
        (translate(str(choice[0])), choice[1])
        if isinstance(choice, (tuple, list))
        else (translate(str(choice)), choice)
        for choice in choices
    ]


def localization_js(filename):
    load_localization(filename)

    payload = json.dumps(current_translation).replace('<', '\\u003c')
    return f"window.localization = {payload}"


def dump_english_config(components):
    all_texts = []
    for c in components:
        label = getattr(c, 'label', None)
        value = getattr(c, 'value', None)
        choices = getattr(c, 'choices', None)
        info = getattr(c, 'info', None)

        if isinstance(label, str):
            all_texts.append(label)
        if isinstance(value, str):
            all_texts.append(value)
        if isinstance(info, str):
            all_texts.append(info)
        if isinstance(choices, list):
            for x in choices:
                if isinstance(x, str):
                    all_texts.append(x)
                if isinstance(x, tuple):
                    for y in x:
                        if isinstance(y, str):
                            all_texts.append(y)

    config_dict = {k: k for k in all_texts if k != "" and 'progress-container' not in k}
    full_name = os.path.abspath(os.path.join(localization_root, 'en.json'))

    with open(full_name, "w", encoding="utf-8") as json_file:
        json.dump(config_dict, json_file, indent=4)

"""Validated defaults shared by the settings UI and completion client."""

import copy
import json
import math
import os
import tempfile
import threading
from pathlib import Path


KEYMAP = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "page_up": "PageUp",
    "page_down": "PageDown",
    "first": "Home",
    "last": "End",
    "choose": "Enter",
    "choose_first": "Tab",
    "close": "Escape",
}
COLORS = {
    "0": ["lightblue", "dodgerblue"],
    "1": ["indianred", "firebrick"],
    "3": ["violet", "darkorchid"],
    "4": ["lightgreen", "darkgreen"],
    "5": ["orange", "darkorange"],
    "default": ["#e0e0e0", "#222222"],
}

# key, label, default, editor, group, bounds/choices
FIELDS = [
    ("enabled", "Enable prompt completion", True, "bool", "General", None),
    ("positive", "Positive prompts", True, "bool", "General", None),
    ("negative", "Negative prompts", True, "bool", "General", None),
    ("inpaint", "Inpaint additional prompt", True, "bool", "General", None),
    ("enhance", "Enhancement prompts", True, "bool", "General", None),
    ("tag_file", "Tag dataset", "danbooru.csv", "csv", "Datasets", None),
    ("translation_file", "Tag translations", "None", "csv", "Datasets", None),
    (
        "translation_old_format",
        "Legacy three-column translations",
        False,
        "bool",
        "Datasets",
        None,
    ),
    ("extra_file", "Extra tags", "extra-quality-tags.csv", "csv", "Datasets", None),
    (
        "extra_mode",
        "Extra tag priority",
        "before",
        "choice",
        "Datasets",
        ["before", "after"],
    ),
    (
        "chant_file",
        "Prompt snippets (Chants)",
        "demo-chants.json",
        "json_file",
        "Datasets",
        None,
    ),
    ("search_aliases", "Search aliases", True, "bool", "Search", None),
    ("only_alias", "Display aliases without arrows", False, "bool", "Search", None),
    ("search_translations", "Search translations", True, "bool", "Search", None),
    (
        "match_mode",
        "Tag matching",
        "word",
        "choice",
        "Search",
        ["word", "prefix", "substring"],
    ),
    ("categories", "Tag categories (empty = all)", [], "categories", "Search", None),
    (
        "model_list",
        "Model names to filter (comma separated)",
        "",
        "text",
        "Search",
        None,
    ),
    (
        "model_list_mode",
        "Model filter",
        "blacklist",
        "choice",
        "Search",
        ["blacklist", "whitelist"],
    ),
    ("max_results", "Visible result rows", 20, "int", "Display", (1, 100)),
    (
        "show_all",
        "Show all matches in scrollable batches",
        False,
        "bool",
        "Display",
        None,
    ),
    ("batch_size", "Rows per batch", 100, "int", "Display", (10, 500)),
    ("delay_ms", "Completion delay (milliseconds)", 150, "int", "Display", (0, 2000)),
    ("min_chars", "Minimum tag search length", 1, "int", "Display", (1, 10)),
    ("follow_cursor", "Follow the text cursor", True, "bool", "Display", None),
    ("previews", "Show local model previews", True, "bool", "Display", None),
    ("wiki_links", "Show tag wiki links", False, "bool", "Display", None),
    (
        "live_translation",
        "Show translations below prompts",
        False,
        "bool",
        "Display",
        None,
    ),
    (
        "replace_underscores",
        "Replace tag underscores with spaces",
        True,
        "bool",
        "Insertion",
        None,
    ),
    (
        "underscore_exclusions",
        "Keep underscores in these tags (comma separated)",
        "0_0,>_<,^_^,o_o,u_u,x_x,@_@",
        "text",
        "Insertion",
        None,
    ),
    (
        "escape_parentheses",
        "Escape parentheses in tags",
        True,
        "bool",
        "Insertion",
        None,
    ),
    ("append_comma", "Append comma", True, "bool", "Insertion", None),
    ("append_space", "Append space", True, "bool", "Insertion", None),
    ("space_at_end", "Always append space at the end", True, "bool", "Insertion", None),
    ("artist_trigger", "Search artist tags with @", True, "bool", "Insertion", None),
    (
        "artist_prefix",
        "Add @ to artist tags",
        "when_triggered",
        "choice",
        "Insertion",
        ["when_triggered", "always", "never"],
    ),
    ("use_loras", "Complete LoRA names", True, "bool", "Models and wildcards", None),
    (
        "use_embeddings",
        "Complete embeddings (supported models only)",
        True,
        "bool",
        "Models and wildcards",
        None,
    ),
    (
        "embeddings_in_tags",
        "Include embeddings in ordinary tag searches",
        False,
        "bool",
        "Models and wildcards",
        None,
    ),
    (
        "use_wildcards",
        "Complete wildcard names and contents",
        True,
        "bool",
        "Models and wildcards",
        None,
    ),
    (
        "use_styles",
        "Complete Fooocus Styles with $",
        True,
        "bool",
        "Models and wildcards",
        None,
    ),
    (
        "model_sort",
        "Model and wildcard ordering",
        "name",
        "choice",
        "Models and wildcards",
        ["name", "newest", "oldest"],
    ),
    (
        "lora_weight",
        "Default LoRA weight",
        1.0,
        "float",
        "Models and wildcards",
        (-10, 10),
    ),
    (
        "trigger_words",
        "Insert local LoRA trigger words",
        False,
        "bool",
        "Models and wildcards",
        None,
    ),
    (
        "trigger_location",
        "LoRA trigger word location",
        "before_lora",
        "choice",
        "Models and wildcards",
        ["start", "end", "before_lora"],
    ),
    (
        "wildcard_sort",
        "Sort wildcard contents alphabetically",
        True,
        "bool",
        "Models and wildcards",
        None,
    ),
    (
        "wildcard_exclusions",
        "Exclude wildcard folders (comma separated)",
        "",
        "text",
        "Models and wildcards",
        None,
    ),
    (
        "wildcard_mode",
        "Tab completion of nested wildcard paths",
        "next_folder",
        "choice",
        "Models and wildcards",
        ["next_folder", "first_difference", "full"],
    ),
    (
        "frequency",
        "Record completion usage and boost frequent results",
        False,
        "bool",
        "Usage",
        None,
    ),
    (
        "frequency_function",
        "Frequency ranking",
        "weak",
        "choice",
        "Usage",
        ["weak", "strong", "usage_first"],
    ),
    ("frequency_min", "Minimum uses before boosting", 3, "int", "Usage", (1, 1000)),
    (
        "frequency_days",
        "Days until usage expires (0 = never)",
        30,
        "int",
        "Usage",
        (0, 3650),
    ),
    (
        "frequency_cap",
        "Maximum boosted results (0 = unlimited)",
        10,
        "int",
        "Usage",
        (0, 1000),
    ),
    (
        "frequency_aliases",
        "Boost matches found through aliases",
        False,
        "bool",
        "Usage",
        None,
    ),
    (
        "keymap",
        "Completion keyboard mappings (JSON)",
        KEYMAP,
        "object",
        "Advanced",
        None,
    ),
    (
        "colors",
        "Category colors: [dark, light] (JSON)",
        COLORS,
        "object",
        "Advanced",
        None,
    ),
]
DEFAULTS = {key: copy.deepcopy(value) for key, _, value, *_ in FIELDS}


def validate_settings(values):
    if not isinstance(values, dict):
        raise ValueError("Completion settings must be a JSON object.")
    result = copy.deepcopy(DEFAULTS)
    for key, _, default, kind, _, bounds in FIELDS:
        if key not in values:
            continue
        value = values[key]
        if kind == "bool":
            valid = isinstance(value, bool)
        elif kind in ("int", "float"):
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and bounds[0] <= value <= bounds[1]
            )
            if kind == "int":
                valid = valid and int(value) == value
                if valid:
                    value = int(value)
        elif kind == "choice":
            valid = value in bounds
        elif kind == "categories":
            valid = isinstance(value, list) and all(
                str(x) in {str(i) for i in range(-1, 9)} for x in value
            )
            value = [str(x) for x in value] if valid else value
        elif kind == "object":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    raise ValueError(f"Invalid JSON for {key}.") from None
            valid = isinstance(value, dict)
            if valid and key == "keymap":
                valid = (
                    set(value) == set(KEYMAP)
                    and all(isinstance(x, str) and len(x) <= 64 for x in value.values())
                    and len([x for x in value.values() if x])
                    == len(set(x for x in value.values() if x))
                )
            elif valid:
                valid = all(
                    isinstance(k, str)
                    and isinstance(v, list)
                    and len(v) == 2
                    and all(isinstance(x, str) and len(x) <= 64 for x in v)
                    for k, v in value.items()
                )
        else:
            valid = isinstance(value, str) and len(value) <= 4096
            if valid and kind in ("csv", "json_file") and value != "None":
                suffix = ".csv" if kind == "csv" else ".json"
                valid = (
                    Path(value).name == value
                    and value.endswith(suffix)
                    and "/" not in value
                    and "\\" not in value
                )
        if not valid:
            raise ValueError(f"Invalid completion setting: {key}.")
        result[key] = value
    if result["only_alias"]:
        result["search_aliases"] = True
    return result


class SettingsStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def load(self):
        with self.lock:
            if self.path.is_file():
                try:
                    document = json.loads(self.path.read_text(encoding="utf-8"))
                    if not isinstance(document, dict):
                        raise ValueError("Completion defaults must be a JSON object.")
                    return validate_settings(document.get("settings", document))
                except (OSError, ValueError, TypeError) as error:
                    print(
                        f"[Tag completion] Cannot read defaults: {error}; using built-in defaults."
                    )
            return copy.deepcopy(DEFAULTS)

    def save(self, values):
        settings = validate_settings(values)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".tagcomplete-", dir=self.path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    json.dump(
                        {"version": 1, "settings": settings},
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return settings

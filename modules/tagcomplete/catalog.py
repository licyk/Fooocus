"""Expose registered completion assets, never arbitrary client filesystem paths."""

import hashlib
import json
import re
import threading
from pathlib import Path


def contained_file(root, filename):
    root = Path(root).resolve()
    path = (root / filename).resolve()
    if path.is_file() and path.is_relative_to(root):
        return path
    raise FileNotFoundError(filename)


class Catalog:
    def __init__(self, tags, loras, embeddings, wildcards, styles):
        self.tags = Path(tags)
        self.loras = [Path(path) for path in loras]
        self.embeddings = Path(embeddings)
        self.wildcards = Path(wildcards)
        self.styles = styles
        self.lock = threading.RLock()
        self.files = {}
        self.data = {}
        self.refresh()

    def register(self, path, kind):
        identifier = hashlib.sha256((kind + str(path)).encode()).hexdigest()[:24]
        self.files[identifier] = (kind, path)
        return identifier

    def scan(self, roots, suffixes, kind):
        entries, names = [], set()
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() not in suffixes or not path.is_file():
                    continue
                try:
                    path = contained_file(root, path.relative_to(root))
                except FileNotFoundError:
                    continue
                name = path.relative_to(root.resolve()).with_suffix("").as_posix()
                # Suggest only identifiers accepted by Fooocus' prompt parsers.
                if kind == "wildcard" and (
                    "__" in name
                    or not re.fullmatch(r"[\w .()-]+(?:/[\w .()-]+)*", name)
                ):
                    continue
                if kind == "lora" and any(char in name for char in ",:<>\r\n"):
                    continue
                if kind == "embedding" and (
                    any(char.isspace() for char in name)
                    or any(char in name for char in ",:<>()[\\]")
                ):
                    continue
                if name in names:
                    continue  # Same precedence as Fooocus' configured search roots.
                names.add(name)
                entry = {
                    "name": name,
                    "id": self.register(path, kind),
                    "modified": path.stat().st_mtime,
                }
                if kind in ("lora", "embedding"):
                    for suffix in (".preview.png", ".png", ".jpg", ".jpeg", ".webp"):
                        try:
                            preview = contained_file(
                                root,
                                path.with_suffix(suffix).relative_to(root.resolve()),
                            )
                        except FileNotFoundError:
                            continue
                        entry["preview"] = self.register(preview, "preview")
                        break
                if kind == "lora":
                    entry.update(self.lora_info(root, path))
                entries.append(entry)
        return entries

    def lora_info(self, root, path):
        result = {}
        try:
            info_path = contained_file(
                root, path.with_suffix(".json").relative_to(root.resolve())
            )
            if info_path.stat().st_size <= 1024 * 1024:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                weight = info.get("preferred weight")
                if (
                    isinstance(weight, (int, float))
                    and not isinstance(weight, bool)
                    and -10 <= weight <= 10
                ):
                    result["weight"] = weight
                keywords = info.get("activation text", info.get("trigger_words", ""))
                if isinstance(keywords, list):
                    keywords = ", ".join(x for x in keywords if isinstance(x, str))
                if isinstance(keywords, str):
                    result["keywords"] = keywords[:4096]
        except (FileNotFoundError, OSError, ValueError, AttributeError):
            pass
        return result

    def refresh(self):
        with self.lock:
            self.files = {}
            datasets = []
            if self.tags.is_dir():
                for path in sorted(self.tags.iterdir()):
                    if path.suffix not in (".csv", ".json"):
                        continue
                    try:
                        path = contained_file(self.tags, path.name)
                    except FileNotFoundError:
                        continue
                    datasets.append(
                        {
                            "name": path.name,
                            "id": self.register(path, "dataset"),
                            "size": path.stat().st_size,
                            "modified": path.stat().st_mtime,
                        }
                    )
            self.data = {
                "datasets": datasets,
                "loras": self.scan(
                    self.loras, {".safetensors", ".ckpt", ".pt", ".bin", ".pth"}, "lora"
                ),
                "embeddings": self.scan(
                    [self.embeddings], {".safetensors", ".pt", ".bin"}, "embedding"
                ),
                "wildcards": self.scan([self.wildcards], {".txt"}, "wildcard"),
                "styles": [{"name": name} for name in self.styles],
            }
            self.data["revision"] = hashlib.sha256(
                json.dumps(self.data, sort_keys=True).encode()
            ).hexdigest()[:16]
            return self.data.copy()

    def get(self):
        with self.lock:
            return self.data.copy()

    def asset(self, identifier, kinds):
        with self.lock:
            kind, path = self.files.get(identifier, (None, None))
        if kind not in kinds or path is None or not path.is_file():
            raise FileNotFoundError(identifier)
        # Re-check containment after refresh in case an asset became a symlink.
        roots = {
            "dataset": [self.tags],
            "wildcard": [self.wildcards],
            "preview": [self.embeddings, *self.loras],
        }[kind]
        if not any(path.resolve().is_relative_to(root.resolve()) for root in roots):
            raise FileNotFoundError(identifier)
        suffixes = {
            "dataset": {".csv", ".json"},
            "wildcard": {".txt"},
            "preview": {".png", ".jpg", ".jpeg", ".webp"},
        }[kind]
        if path.resolve().suffix.lower() not in suffixes:
            raise FileNotFoundError(identifier)
        return path

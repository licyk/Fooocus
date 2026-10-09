"""Text-only token previews using bundled tokenizers, without model weights."""

import threading
from functools import lru_cache
from pathlib import Path

_lock = threading.Lock()
ROOT = Path(__file__).resolve().parents[2] / "ldm_patched" / "modules"


@lru_cache(maxsize=3)
def tokenizer(kind):
    from transformers import CLIPTokenizer, Qwen2Tokenizer, T5TokenizerFast

    if kind == "qwen":
        return Qwen2Tokenizer.from_pretrained(
            ROOT / "anima_tokenizers" / "qwen", local_files_only=True
        )
    if kind == "t5":
        return T5TokenizerFast.from_pretrained(
            ROOT / "anima_tokenizers" / "t5", local_files_only=True
        )
    return CLIPTokenizer.from_pretrained(ROOT / "sd1_tokenizer", local_files_only=True)


def count(text, anima=False):
    from ldm_patched.modules.sd1_clip import (
        escape_important,
        token_weights,
        unescape_important,
    )

    segments = [
        unescape_important(segment)
        for segment, _ in token_weights(escape_important(text), 1.0)
    ]
    with _lock:
        if anima:
            qwen = sum(
                len(tokenizer("qwen")(segment, add_special_tokens=False)["input_ids"])
                for segment in segments
            )
            t5 = sum(
                len(tokenizer("t5")(segment, add_special_tokens=False)["input_ids"])
                for segment in segments
            )
            return {
                "qwen": max(qwen, 1),
                "t5": t5 + 1,
                "kind": "anima",
                "text_only": True,
            }
        clip = sum(
            len(tokenizer("clip")(segment, add_special_tokens=False)["input_ids"])
            for segment in segments
        )
        return {"clip": clip, "chunk_size": 75, "kind": "sdxl", "text_only": True}

"""运行环境配置。"""

import logging
import os
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
REQUIREMENTS_PATH = ROOT_PATH / "requirements_versions.txt"
LOGGER_NAME = "Fooocus"
LOGGER_LEVEL = logging.INFO
LOGGER_COLOR = os.getenv("NO_COLOR") is None

TORCH_MIN_VERSION = "2.8.0"
TORCHVISION_MIN_VERSION = "0.23.0"
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl"

# 同一次解析安装配套版本，不接受低于项目最低要求的构建。
_PACKAGES = [f"torch>={TORCH_MIN_VERSION}", f"torchvision>={TORCHVISION_MIN_VERSION}"]
PYTORCH_INSTALL_ARGS: dict[str, list[str]] = {
    "cuda": [*_PACKAGES, "--index-url", f"{PYTORCH_INDEX_URL}/cu130"],
    "cuda128": [*_PACKAGES, "--index-url", f"{PYTORCH_INDEX_URL}/cu128"],
    "cuda126": [*_PACKAGES, "--index-url", f"{PYTORCH_INDEX_URL}/cu126"],
    "rocm": [
        "torch[device-all]==2.13.0+rocm10.0.0",
        _PACKAGES[1],
        "--index-url",
        "https://stable.repo.amd.com/rocm/whl-next",
    ],
    "xpu": [*_PACKAGES, "--index-url", f"{PYTORCH_INDEX_URL}/xpu"],
    "mps": [*_PACKAGES],
    "cpu": [*_PACKAGES, "--index-url", f"{PYTORCH_INDEX_URL}/cpu"],
}
TORCH_BACKEND_CHOICES = ["auto", *PYTORCH_INSTALL_ARGS]

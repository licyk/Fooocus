"""Upstream optional-package UI operations using Fooocus's current interpreter."""

import importlib.util
import subprocess
import sys

from .providers.base import get_lang

packages = {
    "chardet": "chardet",
    "fastapi": "fastapi",
    "execjs": "PyExecJS",
    "lxml": "lxml",
    "tqdm": "tqdm",
    "pathos": "pathos",
    "cryptography": "cryptography",
    # The following packages are required for translation service. If you do not need translation service, you can remove them.
    # 以下是翻译所需的包，如果不需要翻译服务，可以删除掉它们。
    "openai": "openai",
    "boto3": "boto3",
    "aliyunsdkcore": "aliyun-python-sdk-core",
    "aliyunsdkalimt": "aliyun-python-sdk-alimt",
}


def get_packages_state():
    states = []
    for package_name, package in packages.items():
        item = {"name": package_name, "package": package, "state": False}
        if importlib.util.find_spec(package_name) is not None:
            item["state"] = True

        states.append(item)

    return states


def install_package(name, package):
    result = {"state": False, "message": ""}
    if packages.get(name) != package:
        result["message"] = "Unknown optional package"
        return result
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        importlib.invalidate_caches()
        result["state"] = True
        result["message"] = get_lang("install_success", {"0": package})
    except (subprocess.SubprocessError, OSError) as e:
        print(e)
        print(f"Warning: Failed to install {package}, some preprocessors may not work.")
        result["message"] = get_lang("install_failed", {"0": package}) + "\n" + str(e)
    return result

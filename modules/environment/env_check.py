"""检查 PyTorch 和项目依赖，缺失或损坏时自动安装。"""

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

from modules.environment.cmd_runner import run_cmd
from modules.environment.config import (
    LOGGER_COLOR,
    LOGGER_LEVEL,
    LOGGER_NAME,
    PYTORCH_INSTALL_ARGS,
    REQUIREMENTS_PATH,
    TORCH_MIN_VERSION,
    TORCHVISION_MIN_VERSION,
)
from modules.environment.gpu_detector import detect_torch_backend
from modules.environment.logger import get_logger
from modules.environment.package_analyzer import (
    get_package_version_from_library,
    is_package_installed,
    parse_package_spec,
    parse_requirement_list,
    read_packages_from_requirements_file,
    validate_package_metadata_dependencies,
    validate_requirements,
)

logger = get_logger(name=LOGGER_NAME, level=LOGGER_LEVEL, color=LOGGER_COLOR)
TORCH_REQUIREMENT = f"torch>={TORCH_MIN_VERSION}"
TORCHVISION_REQUIREMENT = f"torchvision>={TORCHVISION_MIN_VERSION}"


def ensure_pip() -> None:
    """为未安装 Pip 的 Python 环境初始化 Pip。"""
    try:
        run_cmd([sys.executable, "-m", "pip", "--version"], live=False, shell=False)
    except RuntimeError:
        run_cmd([sys.executable, "-m", "ensurepip", "--upgrade"], shell=False)
    if not is_package_installed("pip>=24.2"):
        logger.info("更新 Pip，启用规范的软件包名称与元数据解析")
        run_cmd(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip>=24.2"],
            shell=False,
        )
        importlib.invalidate_caches()


def pip_install(*args: str) -> None:
    """使用当前解释器安装软件包，失败时终止环境准备。"""
    ensure_pip()
    custom_env = os.environ.copy()
    custom_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    run_cmd(
        [sys.executable, "-m", "pip", "install", *args],
        custom_env=custom_env,
        shell=False,
    )
    importlib.invalidate_caches()


def probe_pytorch() -> str | None:
    """在独立进程验证导入和 torchvision 算子，避免重装后使用旧模块。"""
    script = """
import json, sys
import torch, torchvision
from torchvision.ops import nms
nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
if torch.version.hip:
    backend = "rocm"
elif torch.version.cuda:
    cuda = torch.version.cuda.split(".")
    backend = "cuda" if int(cuda[0]) >= 13 else "cuda" + "".join(cuda[:2])
elif "+xpu" in torch.__version__ or torch.xpu.is_available():
    backend = "xpu"
elif sys.platform == "darwin":
    backend = "mps"
else:
    backend = "cpu"
print(json.dumps({"backend": backend}))
"""
    try:
        output = run_cmd([sys.executable, "-c", script], live=False, shell=False) or ""
        return json.loads(output.strip().splitlines()[-1])["backend"]
    except (RuntimeError, ValueError, KeyError, IndexError) as exc:
        logger.warning("PyTorch / torchvision 验证失败: %s", exc)
        return None


def check_pytorch(backend: str = "auto", reinstall: bool = False) -> str:
    """检查配套 PyTorch，按显卡类型安装或切换到指定后端。"""
    if backend != "auto" and backend not in PYTORCH_INSTALL_ARGS:
        raise ValueError(f"未知的 PyTorch 类型: {backend}")
    logger.info("检查 PyTorch 中")
    versions_ok = is_package_installed(TORCH_REQUIREMENT) and is_package_installed(
        TORCHVISION_REQUIREMENT
    )
    pair_ok = versions_ok and validate_package_metadata_dependencies("torchvision")
    installed_backend = probe_pytorch() if pair_ok and not reinstall else None
    if installed_backend and (backend == "auto" or installed_backend == backend):
        logger.info("PyTorch / torchvision 已安装，类型: %s", installed_backend)
        return installed_backend

    if backend == "auto":
        backend = detect_torch_backend()
    logger.info("重新安装 PyTorch / torchvision，类型: %s", backend)
    ensure_pip()
    # 一并清理依赖旧 PyTorch 的二进制扩展，防止残留的 ABI 冲突。
    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "torch",
            "torchvision",
            "torchaudio",
            "xformers",
            "-y",
        ],
        shell=False,
    )
    importlib.invalidate_caches()
    install_args = list(PYTORCH_INSTALL_ARGS[backend])
    if custom_index := os.getenv("TORCH_INDEX_URL"):
        if "--index-url" in install_args:
            install_args[install_args.index("--index-url") + 1] = custom_index
        else:
            install_args += ["--index-url", custom_index]
    pip_install(*install_args)
    if not is_package_installed(TORCH_REQUIREMENT) or not is_package_installed(
        TORCHVISION_REQUIREMENT
    ):
        raise RuntimeError(
            "PyTorch / torchvision 安装后仍未满足最低版本要求，请检查上方的 Pip 输出"
        )
    installed_backend = probe_pytorch()
    if installed_backend != backend:
        raise RuntimeError(
            f"PyTorch 安装后验证失败，期望类型 {backend}，实际类型 {installed_backend}"
        )
    logger.info("PyTorch 安装完成")
    return installed_backend


def _requirements_satisfied(requirements_path: Path) -> bool:
    """检查顶层依赖和各软件包声明的依赖。"""
    if not validate_requirements(requirements_path):
        return False
    packages = parse_requirement_list(
        read_packages_from_requirements_file(requirements_path)
    )
    names = dict.fromkeys(parse_package_spec(package)[0] for package in packages)
    for name in names:
        if not validate_package_metadata_dependencies(name):
            logger.warning("%s 声明的依赖不完整", name)
            return False
    return True


def check_requirements(
    index_url: str | None = None, requirements_path: Path | None = None
) -> None:
    """安装缺失依赖并复查，保留已选定的 PyTorch 构建。"""
    requirements_path = resolve_requirements_path(requirements_path)
    logger.info("检查 Fooocus 依赖中")
    if _requirements_satisfied(requirements_path):
        logger.info("Fooocus 依赖完整")
        return
    logger.info("安装 Fooocus 依赖中")
    # 精确锁住含 local version 的构建，避免 PyPI 或镜像替换 CUDA / XPU / CPU 版本。
    with tempfile.TemporaryDirectory(prefix="fooocus-requirements-") as directory:
        constraints = Path(directory) / "pytorch.txt"
        constraints.write_text(
            "".join(
                f"{name}=={get_package_version_from_library(name)}\n"
                for name in ("torch", "torchvision")
            ),
            encoding="utf-8",
        )
        args = ["-r", str(requirements_path), "-c", str(constraints), "--prefer-binary"]
        if index_url:
            args += ["--index-url", index_url]
        pip_install(*args)
    if not _requirements_satisfied(requirements_path):
        raise RuntimeError("Fooocus 依赖安装后仍不完整，请检查上方的 Pip 输出")
    logger.info("Fooocus 依赖安装完成")


def resolve_requirements_path(path: Path | None = None) -> Path:
    """确认依赖文件存在，支持通过 REQS_FILE 指定文件。"""
    path = (
        Path(path or os.getenv("REQS_FILE", str(REQUIREMENTS_PATH)))
        .expanduser()
        .resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"在 {path} 中未找到依赖记录文件，请检查项目文件是否完整"
        )
    return path


def check_environment(
    backend: str = "auto", reinstall_torch: bool = False, index_url: str | None = None
) -> str:
    """依次检查 PyTorch 和项目依赖，返回可用的 PyTorch 类型。"""
    requirements_path = resolve_requirements_path()
    installed_backend = check_pytorch(backend=backend, reinstall=reinstall_torch)
    check_requirements(index_url=index_url, requirements_path=requirements_path)
    return installed_backend

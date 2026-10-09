import ast
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.environment import cmd_runner, env_check, gpu_detector
from modules.environment.package_analyzer import (
    dependency_categorizer,
    is_package_installed,
    parse_requirement_list,
    validate_package_metadata_dependencies,
    validate_requirements,
)

ROOT = Path(__file__).resolve().parents[1]


class TestGPUDetection(unittest.TestCase):
    def test_backends(self):
        cases = [
            ("linux", [], 0, "cpu"),
            ("darwin", [], 0, "mps"),
            (
                "linux",
                [{"Name": "RTX 5090", "AdapterCompatibility": "NVIDIA"}],
                13,
                "cuda",
            ),
            (
                "win32",
                [{"Name": "RTX 4090", "AdapterCompatibility": "NVIDIA"}],
                12.8,
                "cuda128",
            ),
            (
                "linux",
                [{"Name": "Tesla T4", "AdapterCompatibility": "NVIDIA"}],
                12.6,
                "cuda126",
            ),
            (
                "linux",
                [{"Name": "RTX 3090", "AdapterCompatibility": "NVIDIA"}],
                11.8,
                "cpu",
            ),
            (
                "linux",
                [{"Name": "RTX 5090", "AdapterCompatibility": "NVIDIA"}],
                0,
                "cuda",
            ),
            (
                "linux",
                [
                    {
                        "Name": "Radeon RX 7900",
                        "AdapterCompatibility": "Advanced Micro Devices, Inc.",
                    }
                ],
                0,
                "rocm",
            ),
            (
                "win32",
                [
                    {
                        "Name": "Intel(R) Arc(TM) A770 Graphics",
                        "AdapterCompatibility": "Intel",
                    }
                ],
                0,
                "xpu",
            ),
            (
                "linux",
                [
                    {
                        "Name": "Intel Core Ultra Graphics",
                        "AdapterCompatibility": "Intel",
                    }
                ],
                0,
                "xpu",
            ),
            (
                "linux",
                [{"Name": "HD Graphics 630", "AdapterCompatibility": "Intel"}],
                0,
                "cpu",
            ),
        ]
        for platform, gpus, cuda, expected in cases:
            with (
                self.subTest(platform=platform, gpus=gpus, cuda=cuda),
                patch.object(gpu_detector.sys, "platform", platform),
                patch.object(gpu_detector, "get_cuda_version", return_value=cuda),
            ):
                self.assertEqual(gpu_detector.detect_torch_backend(gpus), expected)

    def test_windows_single_and_multiple_gpu(self):
        gpu = {"Name": "NVIDIA RTX", "AdapterCompatibility": "NVIDIA"}
        for data in (
            gpu,
            [gpu, {"Name": "Intel Arc", "AdapterCompatibility": "Intel"}],
        ):
            with (
                self.subTest(data=data),
                patch.object(
                    gpu_detector,
                    "_run_detection_command",
                    return_value=json.dumps(data),
                ),
            ):
                self.assertEqual(
                    gpu_detector.get_windows_gpu_list()[0]["Name"], gpu["Name"]
                )

    def test_corrupt_windows_output(self):
        with patch.object(
            gpu_detector, "_run_detection_command", return_value="invalid JSON"
        ):
            self.assertEqual(gpu_detector.get_windows_gpu_list(), [])

    def test_linux_compute_card_and_vendor(self):
        with (
            patch.object(gpu_detector.shutil, "which", return_value="lspci"),
            patch.object(
                gpu_detector,
                "_run_detection_command",
                side_effect=["", "Device:\tTesla T4\nVendor:\tNVIDIA Corporation\n"],
            ) as run,
        ):
            self.assertTrue(gpu_detector.has_nvidia_gpu(gpu_detector.get_lspci_gpus()))
            self.assertEqual(run.call_args_list[1].args[0][-1], "::0302")

    def test_linux_merges_driver_information(self):
        with (
            patch.object(
                gpu_detector,
                "get_nvidia_smi_gpus",
                return_value=[{"Name": "RTX 4090", "DriverVersion": "580.1"}],
            ),
            patch.object(
                gpu_detector,
                "get_lshw_gpus",
                return_value=[{"Name": "rtx 4090", "AdapterCompatibility": "NVIDIA"}],
            ),
            patch.object(gpu_detector, "get_lspci_gpus", return_value=[]),
        ):
            result = gpu_detector.get_linux_gpu_list()
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["DriverVersion"], "580.1")
            self.assertTrue(gpu_detector.has_nvidia_gpu(result))

    def test_detection_failures_are_bounded(self):
        for error in (
            FileNotFoundError(),
            subprocess.TimeoutExpired("nvidia-smi", 15),
            subprocess.CalledProcessError(1, "nvidia-smi", stderr="driver failed"),
        ):
            with (
                self.subTest(error=error),
                patch.object(gpu_detector.subprocess, "run", side_effect=error) as run,
            ):
                self.assertIsNone(gpu_detector._run_detection_command(["nvidia-smi"]))
                self.assertEqual(run.call_args.kwargs["timeout"], 15)


class TestEnvironment(unittest.TestCase):
    def test_healthy_torch_does_not_reinstall(self):
        with (
            patch.object(env_check, "is_package_installed", return_value=True),
            patch.object(
                env_check, "validate_package_metadata_dependencies", return_value=True
            ),
            patch.object(env_check, "probe_pytorch", return_value="cuda128"),
            patch.object(env_check, "pip_install") as install,
        ):
            self.assertEqual(env_check.check_pytorch(), "cuda128")
            install.assert_not_called()

    def check_repair(self, backend, reinstall=False, installed=True, probes=None):
        with (
            patch.object(
                env_check,
                "is_package_installed",
                side_effect=[True, True, True, True]
                if installed
                else [False, True, True],
            ),
            patch.object(
                env_check, "validate_package_metadata_dependencies", return_value=True
            ),
            patch.object(env_check, "probe_pytorch", side_effect=probes or [backend]),
            patch.object(env_check, "ensure_pip"),
            patch.object(env_check, "run_cmd") as run,
            patch.object(env_check, "pip_install") as install,
        ):
            self.assertEqual(env_check.check_pytorch(backend, reinstall), backend)
            self.assertIn("torchvision", run.call_args.args[0])
            self.assertIn("xformers", run.call_args.args[0])
            self.assertFalse(run.call_args.kwargs["shell"])
            self.assertTrue(
                any(arg.startswith("torchvision") for arg in install.call_args.args)
            )
            return install.call_args.args

    def test_forced_reinstallation_all_backends(self):
        for backend in env_check.PYTORCH_INSTALL_ARGS:
            with self.subTest(backend=backend), patch.dict(os.environ, {}, clear=True):
                self.check_repair(backend, reinstall=True)

    def test_missing_torch_installs(self):
        self.check_repair("cpu", installed=False)

    def test_broken_import_repairs(self):
        self.check_repair("cpu", probes=[None, "cpu"])

    def test_explicit_backend_switches(self):
        self.check_repair("cuda", probes=["cpu", "cuda"])

    def test_custom_torch_index(self):
        with patch.dict(
            os.environ, {"TORCH_INDEX_URL": "https://example.invalid/wheels"}
        ):
            args = self.check_repair("cpu", reinstall=True)
            self.assertEqual(
                args[args.index("--index-url") + 1], "https://example.invalid/wheels"
            )

    def test_failed_torch_install_stops(self):
        with (
            patch.object(env_check, "is_package_installed", return_value=False),
            patch.object(env_check, "ensure_pip"),
            patch.object(env_check, "run_cmd"),
            patch.object(
                env_check, "pip_install", side_effect=RuntimeError("pip failed")
            ),
            self.assertRaisesRegex(RuntimeError, "pip failed"),
        ):
            env_check.check_pytorch("cpu")

    def test_wrong_backend_after_install_stops(self):
        with (
            patch.object(env_check, "is_package_installed", return_value=True),
            patch.object(
                env_check, "validate_package_metadata_dependencies", return_value=True
            ),
            patch.object(env_check, "ensure_pip"),
            patch.object(env_check, "run_cmd"),
            patch.object(env_check, "pip_install"),
            patch.object(env_check, "probe_pytorch", return_value="cpu"),
            self.assertRaisesRegex(RuntimeError, "实际类型 cpu"),
        ):
            env_check.check_pytorch("cuda", reinstall=True)

    def test_probe_does_not_import_torch_into_launcher(self):
        with patch.object(
            env_check, "run_cmd", return_value='warning\n{"backend":"cpu"}\n'
        ) as run:
            self.assertEqual(env_check.probe_pytorch(), "cpu")
            self.assertEqual(run.call_args.args[0][:2], [sys.executable, "-c"])
            self.assertFalse(run.call_args.kwargs["shell"])

    def test_metadata_dependency_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requirements.txt"
            path.write_text("example==1.0\n")
            with (
                patch.object(env_check, "validate_requirements", return_value=True),
                patch.object(
                    env_check,
                    "validate_package_metadata_dependencies",
                    return_value=False,
                ) as check,
            ):
                self.assertFalse(env_check._requirements_satisfied(path))
                check.assert_called_once_with("example")

    def test_install_uses_constraints_and_checks_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requirements file.txt"
            path.touch()

            def install(*args):
                self.assertEqual(args[args.index("-r") + 1], str(path))
                constraint = Path(args[args.index("-c") + 1]).read_text()
                self.assertEqual(
                    constraint, "torch==2.8.0+cu128\ntorchvision==0.23.0+cu128\n"
                )
                self.assertEqual(
                    args[args.index("--index-url") + 1],
                    "https://example.invalid/simple",
                )

            with (
                patch.object(
                    env_check, "_requirements_satisfied", side_effect=[False, True]
                ) as check,
                patch.object(
                    env_check,
                    "get_package_version_from_library",
                    side_effect=["2.8.0+cu128", "0.23.0+cu128"],
                ),
                patch.object(env_check, "pip_install", side_effect=install),
            ):
                env_check.check_requirements("https://example.invalid/simple", path)
                self.assertEqual(check.call_count, 2)

    def test_failed_requirements_recheck_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requirements.txt"
            path.touch()
            with (
                patch.object(env_check, "_requirements_satisfied", return_value=False),
                patch.object(env_check, "pip_install"),
                patch.object(
                    env_check, "get_package_version_from_library", return_value="2.8.0"
                ),
                self.assertRaisesRegex(RuntimeError, "依赖安装后仍不完整"),
            ):
                env_check.check_requirements(requirements_path=path)

    def test_missing_requirements_precedes_torch_install(self):
        with (
            patch.dict(
                os.environ, {"REQS_FILE": "/nonexistent/fooocus-test-requirements"}
            ),
            patch.object(env_check, "check_pytorch") as check,
        ):
            with self.assertRaises(FileNotFoundError):
                env_check.check_environment()
            check.assert_not_called()

    def test_pip_bootstrap(self):
        with (
            patch.object(
                env_check, "run_cmd", side_effect=[RuntimeError("no pip"), None]
            ) as run,
            patch.object(env_check, "is_package_installed", return_value=True),
        ):
            env_check.ensure_pip()
            self.assertEqual(
                run.call_args.args[0], [sys.executable, "-m", "ensurepip", "--upgrade"]
            )

    def test_old_pip_is_upgraded(self):
        with (
            patch.object(env_check, "run_cmd") as run,
            patch.object(env_check, "is_package_installed", return_value=False),
        ):
            env_check.ensure_pip()
            self.assertEqual(
                run.call_args.args[0],
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip>=24.2"],
            )

    def test_subprocess_errors_propagate(self):
        with (
            patch.object(
                cmd_runner.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="install failed"
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "install failed"),
        ):
            cmd_runner.run_cmd(["python", "-m", "pip"], shell=False, live=False)


class TestPackageAnalysis(unittest.TestCase):
    def test_versions_and_local_builds(self):
        with patch.object(importlib.metadata, "version", return_value="2.8.0+cu128"):
            self.assertTrue(is_package_installed("torch>=2.8.0,<3"))
            self.assertTrue(is_package_installed("torch==2.8.*"))
            self.assertFalse(is_package_installed("torch<2.8"))
            self.assertFalse(is_package_installed("torch>=2.9"))
        with patch.object(importlib.metadata, "version", return_value="2.7.1"):
            self.assertFalse(is_package_installed("torch>=2.8.0"))

    def test_markers_comments_and_multiple_constraints(self):
        requirements = parse_requirement_list(
            [
                'missing-package; python_version < "1.0"',
                "torch>=2.8,<3 # comment",
                'numpy; python_version >= "3.10"',
                "# empty",
            ]
        )
        self.assertNotIn("missing-package", requirements)
        self.assertIn("torch>=2.8", requirements)
        self.assertIn("torch<3", requirements)
        self.assertIn("numpy", requirements)

    def test_metadata_markers_and_optional_extras(self):
        metadata = [
            "numpy>=1",
            'windows-only; sys_platform == "nonexistent"',
            'requests; extra == "client"',
        ]
        with (
            patch.object(dependency_categorizer, "requires", return_value=metadata),
            patch.object(
                importlib.metadata,
                "version",
                side_effect=lambda name: "1.26" if name == "numpy" else None,
            ),
        ):
            self.assertTrue(validate_package_metadata_dependencies("example"))
            self.assertFalse(validate_package_metadata_dependencies("example[client]"))

    def test_requirements_version_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requirements.txt"
            path.write_text("torch>=2.8\n")
            with patch.object(importlib.metadata, "version", return_value="2.7.1"):
                self.assertFalse(validate_requirements(path))


class TestLauncher(unittest.TestCase):
    def test_help_in_empty_environment(self):
        result = subprocess.run(
            [sys.executable, "-S", "launch.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--reinstall-torch", result.stdout)
        self.assertNotIn("安装 PyTorch 中", result.stdout)

    def prepare(self, args):
        tree = ast.parse((ROOT / "launch.py").read_text())
        definition = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_environment"
        )
        scope = {
            "args": args,
            "os": os,
            "sys": sys,
            "fooocus_version": SimpleNamespace(version="test"),
            "logger": env_check.logger,
        }
        exec(  # noqa: S102
            compile(
                ast.Module(body=[definition], type_ignores=[]), "launch.py", "exec"
            ),
            scope,
        )
        return scope

    def args(self, **changes):
        values = {
            "skip_check": False,
            "torch_backend": "auto",
            "always_cpu": None,
            "reinstall_torch": False,
            "index_url": None,
            "directml": None,
        }
        return SimpleNamespace(**(values | changes))

    def test_skip_prevents_install(self):
        scope = self.prepare(self.args(skip_check=True))
        # No check_environment binding: invoking it would fail this test.
        scope["prepare_environment"]()

    def test_cpu_option_and_detected_cpu(self):
        for cpu in (None, -1, 4):
            with self.subTest(cpu=cpu):
                args = self.args(always_cpu=cpu)
                scope = self.prepare(args)
                scope["check_environment"] = lambda cpu=cpu, **kwargs: (
                    self.assertEqual(
                        kwargs["backend"], "auto" if cpu is None else "cpu"
                    )
                    or "cpu"
                )
                scope["prepare_environment"]()
                self.assertEqual(args.always_cpu, cpu if cpu is not None else -1)

    def test_install_failure_exits_before_ui(self):
        scope = self.prepare(self.args())

        def fail(**kwargs):
            raise RuntimeError("pip failed")

        scope["check_environment"] = fail
        with self.assertRaises(SystemExit) as exc:
            scope["prepare_environment"]()
        self.assertEqual(exc.exception.code, 1)


if __name__ == "__main__":
    unittest.main()

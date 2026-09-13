"""Tests for running the analysis on the GPU, without a GPU.

``ECG_DEVICE`` has to reach both stages: the digitizer, through a copy of its config with
the two ``device`` keys changed, and ECGFounder, through the ``device`` argument. torch is
replaced by a stand-in wherever CUDA availability matters, so the suite behaves the same
on a laptop, in CI, and on a machine that does have a GPU.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

from ecg_pipeline import cli, devices, digitizer, pipeline


def fake_torch(available: bool, count: int = 1, cuda_build: str | None = "12.1") -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: available, device_count=lambda: count)  # type: ignore[attr-defined]
    module.version = types.SimpleNamespace(cuda=cuda_build)  # type: ignore[attr-defined]
    return module


def fake_checkout() -> Path:
    home = Path(tempfile.mkdtemp())
    (home / "src").mkdir()
    (home / "src" / "digitize.py").write_text("# stand-in for the real digitizer\n")
    return home


def device_values(config: Path) -> list[str]:
    data = yaml.safe_load(config.read_text())
    values = []
    for keys in digitizer.DEVICE_KEYS:
        node = data
        for key in keys:
            node = node[key]
        values.append(node)
    return values


class TestResolveDevice(unittest.TestCase):
    def test_cpu_is_the_default(self):
        self.assertEqual(devices.resolve_device(env={}), "cpu")

    def test_the_environment_selects_the_gpu(self):
        self.assertEqual(devices.resolve_device(env={devices.ENV_DEVICE: " CUDA "}), "cuda")

    def test_an_explicit_argument_beats_the_environment(self):
        self.assertEqual(devices.resolve_device("cpu", env={devices.ENV_DEVICE: "cuda"}), "cpu")

    def test_an_indexed_gpu_is_accepted(self):
        self.assertEqual(devices.resolve_device("cuda:1", env={}), "cuda:1")

    def test_unknown_devices_are_rejected(self):
        for raw in ("gpu", "cuda:x", "mps", "cuda:"):
            with self.subTest(raw=raw), self.assertRaises(devices.DeviceError):
                devices.resolve_device(raw, env={})


class TestRequireDevice(unittest.TestCase):
    def test_cpu_never_imports_torch(self):
        # None in sys.modules makes any `import torch` raise.
        with mock.patch.dict(sys.modules, {"torch": None}), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(devices.require_device(), "cpu")

    def test_cuda_without_a_visible_gpu_fails_with_a_clear_message(self):
        with mock.patch.dict(sys.modules, {"torch": fake_torch(available=False)}):
            with self.assertRaises(devices.DeviceUnavailable) as ctx:
                devices.require_device("cuda")
        message = str(ctx.exception)
        self.assertIn("ECG_DEVICE=cuda", message)
        self.assertIn("sees no GPU", message)
        self.assertIn("ECG_DEVICE=cpu", message)

    def test_a_cpu_only_torch_build_is_named_as_the_cause(self):
        with mock.patch.dict(sys.modules, {"torch": fake_torch(available=False, cuda_build=None)}):
            with self.assertRaises(devices.DeviceUnavailable) as ctx:
                devices.require_device("cuda")
        self.assertIn("no CUDA support", str(ctx.exception))

    def test_an_index_past_the_last_gpu_fails(self):
        with mock.patch.dict(sys.modules, {"torch": fake_torch(available=True, count=1)}):
            self.assertEqual(devices.require_device("cuda:0"), "cuda:0")
            with self.assertRaises(devices.DeviceUnavailable):
                devices.require_device("cuda:1")

    def test_an_available_gpu_passes(self):
        with mock.patch.dict(sys.modules, {"torch": fake_torch(available=True)}):
            self.assertEqual(devices.require_device("cuda"), "cuda")


class TestConfigForDevice(unittest.TestCase):
    def test_cpu_passes_the_config_through_untouched(self):
        self.assertEqual(digitizer.config_for_device(digitizer.DEFAULT_CONFIG, "cpu"), digitizer.DEFAULT_CONFIG)

    def test_the_shipped_config_has_both_device_keys_on_cpu(self):
        self.assertEqual(device_values(digitizer.DEFAULT_CONFIG), ["cpu", "cpu"])

    def test_cuda_changes_both_device_keys_and_nothing_else(self):
        generated = digitizer.config_for_device(digitizer.DEFAULT_CONFIG, "cuda")

        self.assertNotEqual(generated, digitizer.DEFAULT_CONFIG)
        self.assertEqual(device_values(generated), ["cuda", "cuda"])

        original = yaml.safe_load(digitizer.DEFAULT_CONFIG.read_text())
        changed = yaml.safe_load(generated.read_text())
        original["MODEL"]["KWARGS"]["device"] = "cuda"
        original["MODEL"]["KWARGS"]["config"]["LAYOUT_IDENTIFIER"]["KWARGS"]["device"] = "cuda"
        self.assertEqual(changed, original)

    def test_the_same_request_maps_to_the_same_file(self):
        first = digitizer.config_for_device(digitizer.DEFAULT_CONFIG, "cuda")
        second = digitizer.config_for_device(digitizer.DEFAULT_CONFIG, "cuda")
        self.assertEqual(first, second)
        self.assertNotEqual(first, digitizer.config_for_device(digitizer.DEFAULT_CONFIG, "cuda:1"))

    def test_an_edited_config_does_not_reuse_a_stale_copy(self):
        config = Path(tempfile.mkdtemp()) / "custom.yml"
        text = digitizer.DEFAULT_CONFIG.read_text()
        config.write_text(text)
        before = digitizer.config_for_device(config, "cuda")
        config.write_text(text.replace("resample_size: 3000", "resample_size: 2500"))
        after = digitizer.config_for_device(config, "cuda")
        self.assertNotEqual(before, after)
        self.assertEqual(yaml.safe_load(after.read_text())["MODEL"]["KWARGS"]["resample_size"], 2500)

    def test_a_config_without_the_device_keys_is_refused(self):
        config = Path(tempfile.mkdtemp()) / "partial.yml"
        config.write_text("MODEL:\n  KWARGS:\n    device: 'cpu'\n")
        with self.assertRaises(ValueError) as ctx:
            digitizer.config_for_device(config, "cuda")
        self.assertIn("LAYOUT_IDENTIFIER", str(ctx.exception))


class TestDigitizeOnDevice(unittest.TestCase):
    """Which config each mode is handed. Subprocess and persistent must agree."""

    def setUp(self) -> None:
        self.images = Path(tempfile.mkdtemp())
        self.out = Path(tempfile.mkdtemp())
        self.enterContext(mock.patch.dict(os.environ, {}, clear=False))
        for name in (devices.ENV_DEVICE, digitizer.ENV_DEBUG_PNG):
            os.environ.pop(name, None)

    def subprocess_cmd(self, **kwargs) -> list[str]:
        with mock.patch("ecg_pipeline.digitizer.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            digitizer.digitize(self.images, self.out, home=fake_checkout(), quiet=True, mode="subprocess", **kwargs)
        return list(run.call_args.args[0])

    def config_arg(self, cmd: list[str]) -> Path:
        return Path(cmd[cmd.index("--config") + 1])

    def test_cpu_runs_the_checked_in_config_as_before(self):
        cmd = self.subprocess_cmd()
        self.assertEqual(self.config_arg(cmd), digitizer.DEFAULT_CONFIG.resolve())
        self.assertFalse(any(arg.startswith(digitizer.SAVE_MODE_KEY) for arg in cmd))

    def test_cuda_in_subprocess_mode_runs_a_cuda_config(self):
        with mock.patch.object(devices, "require_device", return_value="cuda") as require:
            cmd = self.subprocess_cmd(device="cuda")
        require.assert_called_once_with("cuda")
        self.assertEqual(device_values(self.config_arg(cmd)), ["cuda", "cuda"])

    def test_the_environment_alone_moves_the_digitizer_to_the_gpu(self):
        os.environ[devices.ENV_DEVICE] = "cuda"
        with mock.patch.dict(sys.modules, {"torch": fake_torch(available=True)}):
            cmd = self.subprocess_cmd()
        self.assertEqual(device_values(self.config_arg(cmd)), ["cuda", "cuda"])

    def test_cuda_in_persistent_mode_starts_the_server_on_a_cuda_config(self):
        server = mock.Mock()
        server.run.return_value = ""
        with (
            mock.patch.object(devices, "require_device", return_value="cuda"),
            mock.patch.object(digitizer, "persistent_digitizer", return_value=server) as factory,
        ):
            digitizer.digitize(self.images, self.out, home=fake_checkout(), quiet=True, mode="persistent")
        config = factory.call_args.args[2]
        self.assertEqual(device_values(Path(config)), ["cuda", "cuda"])

    def test_no_gpu_fails_before_the_digitizer_is_started(self):
        with (
            mock.patch.dict(sys.modules, {"torch": fake_torch(available=False)}),
            mock.patch("ecg_pipeline.digitizer.subprocess.run") as run,
            mock.patch.object(digitizer, "persistent_digitizer") as factory,
        ):
            for mode in digitizer.MODES:
                with self.subTest(mode=mode), self.assertRaises(devices.DeviceUnavailable):
                    digitizer.digitize(self.images, self.out, home=fake_checkout(), mode=mode, device="cuda")
        run.assert_not_called()
        factory.assert_not_called()

    def test_the_debug_png_is_kept_unless_turned_off(self):
        for raw, expected in (("1", False), ("true", False), ("0", True), ("off", True)):
            os.environ[digitizer.ENV_DEBUG_PNG] = raw
            with self.subTest(raw=raw):
                cmd = self.subprocess_cmd()
                self.assertEqual(f"{digitizer.SAVE_MODE_KEY}=timeseries_only" in cmd, expected)

    def test_a_malformed_debug_png_setting_is_rejected(self):
        os.environ[digitizer.ENV_DEBUG_PNG] = "maybe"
        with self.assertRaises(ValueError):
            self.subprocess_cmd()


class TestPipelineDevice(unittest.TestCase):
    def setUp(self) -> None:
        self.images = Path(tempfile.mkdtemp())
        self.out = Path(tempfile.mkdtemp())

    def test_one_device_reaches_the_digitizer_and_ecgfounder(self):
        import ecg_pipeline.interpret.interpret_ecg as interpret

        csv = self.out / f"ecg{digitizer.CANONICAL_SUFFIX}"
        csv.write_text("I\n0\n")

        with (
            mock.patch.object(devices, "require_device", return_value="cuda") as require,
            mock.patch.object(pipeline.preprocess, "prepare_images", return_value={"ecg": {}}),
            mock.patch.object(pipeline.digitizer, "digitize", return_value=[csv]) as digitize,
            mock.patch.object(interpret, "build_model_for", return_value=(object(), "ckpt.pth")) as build,
            mock.patch.object(interpret, "interpret_csv", return_value={"topk": []}) as interpret_csv,
            mock.patch.object(pipeline, "_record_warnings", return_value=[]),
            mock.patch.object(pipeline, "_layout_template_warnings", return_value=([], False, [])),
        ):
            pipeline.run(self.images, self.out, device="cuda", quiet=True)

        require.assert_called_once_with("cuda")
        self.assertEqual(digitize.call_args.kwargs["device"], "cuda")
        self.assertEqual(build.call_args.kwargs["device"], "cuda")
        self.assertEqual(interpret_csv.call_args.kwargs["device"], "cuda")

    def test_no_gpu_fails_before_any_image_is_touched(self):
        with (
            mock.patch.object(devices, "require_device", side_effect=devices.DeviceUnavailable("no GPU")),
            mock.patch.object(pipeline.preprocess, "prepare_images") as prepare,
            mock.patch.object(pipeline.digitizer, "digitize") as digitize,
        ):
            with self.assertRaises(devices.DeviceUnavailable):
                pipeline.run(self.images, self.out, device="cuda", quiet=True)
        prepare.assert_not_called()
        digitize.assert_not_called()

    def test_the_cli_reports_a_missing_gpu_as_an_error(self):
        stderr = io.StringIO()
        with (
            mock.patch.object(cli.pipeline, "run", side_effect=devices.DeviceUnavailable("ECG_DEVICE=cuda: no GPU")),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(["--images", "in", "--out", "out", "--device", "cuda"])
        self.assertEqual(code, 1)
        self.assertIn("no GPU", stderr.getvalue())

    def test_the_cli_leaves_the_device_to_the_environment_by_default(self):
        with mock.patch.object(cli.pipeline, "run", return_value=[]) as run, contextlib.redirect_stderr(io.StringIO()):
            cli.main(["--images", "in", "--out", "out", "--quiet"])
        self.assertIsNone(run.call_args.kwargs["device"])


if __name__ == "__main__":
    unittest.main()

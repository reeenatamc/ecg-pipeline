"""Tests for the digitizer's run modes, working resolution and thread count.

The persistent client is exercised against a fake ``src/serve.py`` that speaks the same
line protocol as the real one (patches/0004) but writes a stub CSV instead of running any
model, so no digitizer checkout or weights are needed.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from ecg_pipeline import digitizer, threads

FAKE_SERVE = textwrap.dedent(
    """
    import json, os, sys
    protocol = sys.stdout
    print("noise before ready goes nowhere near the protocol", file=sys.stderr)
    protocol.write(json.dumps({"event": "ready", "load_seconds": 0.01}) + "\\n"); protocol.flush()
    for line in sys.stdin:
        request = json.loads(line)
        overrides = request["overrides"]
        images = overrides["DATA.images_path"]
        if "crash" in images:
            os._exit(9)
        reply = {"id": request["id"], "ok": True, "log": "", "seconds": 0.0}
        if "fail" in images:
            reply.update(ok=False, error="Traceback: boom")
        else:
            out = overrides["DATA.output_path"]
            for name in sorted(os.listdir(images)):
                stem = os.path.splitext(name)[0]
                with open(os.path.join(out, stem + "_timeseries_canonical.csv"), "w") as fh:
                    fh.write("I\\n0\\n")
            reply["log"] = "pid=%d threads=%s overrides=%s" % (
                os.getpid(), os.environ.get("OMP_NUM_THREADS"), json.dumps(overrides, sort_keys=True))
        protocol.write(json.dumps(reply) + "\\n"); protocol.flush()
    """
)


def fake_checkout(with_server: bool = True) -> Path:
    home = Path(tempfile.mkdtemp())
    (home / "src").mkdir()
    (home / "src" / "digitize.py").write_text("# stand-in for the real digitizer\n")
    if with_server:
        (home / "src" / "serve.py").write_text(FAKE_SERVE)
    return home


def image_dir(name: str = "images") -> Path:
    directory = Path(tempfile.mkdtemp()) / name
    directory.mkdir()
    (directory / "ecg.png").write_bytes(b"not really a png")
    return directory


class TestResolveMode(unittest.TestCase):
    def test_persistent_is_the_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(digitizer.resolve_mode(), digitizer.MODE_PERSISTENT)

    def test_the_environment_selects_subprocess(self):
        with mock.patch.dict(os.environ, {digitizer.ENV_MODE: "Subprocess"}):
            self.assertEqual(digitizer.resolve_mode(), digitizer.MODE_SUBPROCESS)

    def test_an_explicit_argument_beats_the_environment(self):
        with mock.patch.dict(os.environ, {digitizer.ENV_MODE: "subprocess"}):
            self.assertEqual(digitizer.resolve_mode("persistent"), digitizer.MODE_PERSISTENT)

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            digitizer.resolve_mode("threaded")


class TestResampleOverride(unittest.TestCase):
    def test_unset_leaves_the_config_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(digitizer.resample_override(), {})

    def test_a_valid_size_becomes_the_model_override(self):
        with mock.patch.dict(os.environ, {digitizer.ENV_RESAMPLE_SIZE: " 1600 "}):
            self.assertEqual(digitizer.resample_override(), {digitizer.RESAMPLE_KEY: "1600"})

    def test_garbage_and_tiny_sizes_are_rejected(self):
        for raw in ("big", "1.5", "100"):
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {digitizer.ENV_RESAMPLE_SIZE: raw}):
                with self.assertRaises(ValueError):
                    digitizer.resample_override()

    def test_the_config_default_is_a_downscale_the_digitizer_understands(self):
        """The shipped default must be an int (a long side), or the digitizer ignores it."""
        import yaml

        cfg = yaml.safe_load(digitizer.DEFAULT_CONFIG.read_text())
        size = cfg["MODEL"]["KWARGS"]["resample_size"]
        self.assertIsInstance(size, int)
        self.assertGreaterEqual(size, digitizer.MIN_RESAMPLE_SIZE)


class TestThreads(unittest.TestCase):
    def test_environment_value_is_used(self):
        self.assertEqual(threads.resolve_threads({threads.ENV_THREADS: "3"}), 3)

    def test_default_is_bounded(self):
        value = threads.resolve_threads({})
        self.assertGreaterEqual(value, 1)
        self.assertLessEqual(value, threads.MAX_DEFAULT_THREADS)
        self.assertLessEqual(value, threads.available_cpus())

    def test_invalid_values_are_rejected(self):
        for raw in ("0", "-2", "many"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                threads.resolve_threads({threads.ENV_THREADS: raw})

    def test_child_env_carries_the_count_to_openmp_and_mkl(self):
        env = threads.child_env({threads.ENV_THREADS: "5", "PATH": "/bin"})
        self.assertEqual(env["OMP_NUM_THREADS"], "5")
        self.assertEqual(env["MKL_NUM_THREADS"], "5")
        self.assertEqual(env["PATH"], "/bin")


class TestSubprocessModeEnvironment(unittest.TestCase):
    def _call(self, env: dict[str, str], **kwargs):
        images, out = image_dir(), tempfile.mkdtemp()
        with mock.patch.dict(os.environ, env), mock.patch("ecg_pipeline.digitizer.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            digitizer.digitize(images, out, home=fake_checkout(False), quiet=True, mode="subprocess", **kwargs)
        return run.call_args

    def test_thread_count_reaches_the_child(self):
        call = self._call({threads.ENV_THREADS: "2"})
        self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "2")

    def test_resample_size_from_the_environment_is_passed(self):
        call = self._call({digitizer.ENV_RESAMPLE_SIZE: "1400"})
        self.assertIn(f"{digitizer.RESAMPLE_KEY}=1400", call.args[0])

    def test_a_caller_override_beats_the_environment(self):
        call = self._call({digitizer.ENV_RESAMPLE_SIZE: "1400"}, overrides={digitizer.RESAMPLE_KEY: "2000"})
        self.assertIn(f"{digitizer.RESAMPLE_KEY}=2000", call.args[0])
        self.assertNotIn(f"{digitizer.RESAMPLE_KEY}=1400", call.args[0])


class TestPersistentMode(unittest.TestCase):
    def setUp(self) -> None:
        self.home = fake_checkout()
        self.config = Path(tempfile.mkdtemp()) / "config.yml"
        self.config.write_text("MODEL: {}\n")
        self.addCleanup(digitizer.close_persistent_digitizer)
        self.enterContext(mock.patch.dict(os.environ, {threads.ENV_THREADS: "3"}))

    def digitize(self, images: Path, **kwargs):
        kwargs.setdefault("quiet", True)
        out = Path(tempfile.mkdtemp())
        paths = digitizer.digitize(
            images, out, config=self.config, home=self.home, python_exe=sys.executable, mode="persistent", **kwargs
        )
        return paths, out

    def server(self) -> digitizer.PersistentDigitizer:
        return digitizer.persistent_digitizer(self.home.resolve(), sys.executable, self.config.resolve())

    def test_one_process_serves_consecutive_studies(self):
        paths1, _ = self.digitize(image_dir())
        paths2, _ = self.digitize(image_dir())

        self.assertEqual([p.name for p in paths1], ["ecg_timeseries_canonical.csv"])
        self.assertEqual([p.name for p in paths2], ["ecg_timeseries_canonical.csv"])
        self.assertEqual(self.server().starts, 1)

    def test_overrides_and_threads_reach_the_server(self):
        with mock.patch("builtins.print") as printed:
            self.digitize(image_dir(), overrides={digitizer.RESAMPLE_KEY: "1600"}, quiet=False)
        log = printed.call_args.args[0]
        self.assertIn("threads=3", log)
        self.assertIn(f'"{digitizer.RESAMPLE_KEY}": "1600"', log)
        self.assertIn('"DATA.output_path"', log)

    def test_a_failed_request_raises_and_keeps_the_process(self):
        self.digitize(image_dir())
        with self.assertRaises(digitizer.DigitizerFailed) as ctx:
            self.digitize(image_dir("fail"))
        self.assertIn("boom", str(ctx.exception))

        self.digitize(image_dir())
        self.assertEqual(self.server().starts, 1)

    def test_a_dead_process_raises_and_the_next_call_restarts_it(self):
        self.digitize(image_dir())
        with self.assertRaises(digitizer.DigitizerFailed) as ctx:
            self.digitize(image_dir("crash"))
        self.assertIn("exited unexpectedly", str(ctx.exception))

        paths, _ = self.digitize(image_dir())
        self.assertEqual(len(paths), 1)
        self.assertEqual(self.server().starts, 2)

    def test_a_different_thread_count_replaces_the_process(self):
        self.digitize(image_dir())
        first = self.server()
        with mock.patch.dict(os.environ, {threads.ENV_THREADS: "1"}):
            self.digitize(image_dir())
            self.assertIsNot(self.server(), first)
        self.assertFalse(first.alive)

    def test_close_stops_the_process(self):
        self.digitize(image_dir())
        server = self.server()
        self.assertTrue(server.alive)
        digitizer.close_persistent_digitizer()
        self.assertFalse(server.alive)


if __name__ == "__main__":
    unittest.main()

"""Tests for locating and invoking the external Open-ECG-Digitizer checkout.

No digitizer checkout is required: these cover resolution and argument construction,
not a real digitization run.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ecg_pipeline import digitizer


def fake_checkout() -> str:
    d = tempfile.mkdtemp()
    (Path(d) / "src").mkdir()
    (Path(d) / "src" / "digitize.py").write_text("# stand-in for the real digitizer\n")
    return d


class TestDigitizerHome(unittest.TestCase):
    def test_env_var_pointing_at_a_real_checkout_is_used(self):
        home = fake_checkout()
        with mock.patch.dict(os.environ, {digitizer.ENV_HOME: home}):
            self.assertEqual(digitizer.digitizer_home(), Path(home).resolve())

    def test_a_bad_env_var_raises_instead_of_falling_back(self):
        """Silently using a different checkout would attribute results to the wrong
        digitizer version, which matters once a specific one has been validated."""
        with mock.patch.dict(os.environ, {digitizer.ENV_HOME: "/definitely/not/here"}):
            with self.assertRaises(digitizer.DigitizerNotFound) as ctx:
                digitizer.digitizer_home()

        message = str(ctx.exception)
        self.assertIn("/definitely/not/here", message)
        self.assertNotIn("Could not find an Open-ECG-Digitizer checkout at", message)

    def test_falls_back_to_the_sibling_when_no_env_var_is_set(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            sibling = digitizer.REPO_ROOT.parent / "Open-ECG-Digitizer"
            if (sibling / "src" / "digitize.py").is_file():
                self.assertEqual(digitizer.digitizer_home(), sibling.resolve())
            else:
                with self.assertRaises(digitizer.DigitizerNotFound):
                    digitizer.digitizer_home()


class TestLeadLayoutOverride(unittest.TestCase):
    def test_bare_filename_resolves_against_configs(self):
        override = digitizer.lead_layout_override("lead_layouts_limbaug_right.yml")
        path = Path(override[digitizer.LAYOUT_KEY])

        self.assertTrue(path.is_absolute(), "the digitizer resolves this against its own root")
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent, digitizer.REPO_ROOT / "configs")

    def test_missing_layout_raises(self):
        with self.assertRaises(FileNotFoundError):
            digitizer.lead_layout_override("does_not_exist.yml")


class TestDigitizeArguments(unittest.TestCase):
    """digitize() shells out; these check what it would run, not the run itself."""

    def _run_capture(self, **kwargs):
        images, out = tempfile.mkdtemp(), tempfile.mkdtemp()
        with mock.patch("ecg_pipeline.digitizer.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            digitizer.digitize(image_dir=images, output_dir=out, home=Path(fake_checkout()), quiet=True, **kwargs)
        return run.call_args, images, out

    def test_input_and_output_are_passed_as_positional_overrides(self):
        """The digitizer's CLI takes overrides positionally; a --overrides flag makes it
        exit 2."""
        call, images, out = self._run_capture()
        cmd = call.args[0]

        self.assertNotIn("--overrides", cmd)
        self.assertIn(f"DATA.images_path={Path(images).resolve()}/", cmd)
        self.assertIn(f"DATA.output_path={Path(out).resolve()}", cmd)

    def test_runs_from_the_checkout_root(self):
        """Its config uses paths relative to its own root, so CWD is load-bearing."""
        call, _, _ = self._run_capture()
        self.assertTrue((Path(call.kwargs["cwd"]) / "src" / "digitize.py").is_file())

    def test_caller_overrides_are_merged(self):
        call, _, _ = self._run_capture(overrides={"MODEL.KWARGS.device": "cuda:0"})
        self.assertIn("MODEL.KWARGS.device=cuda:0", call.args[0])

    def test_a_nonzero_exit_raises_with_the_captured_output(self):
        images, out = tempfile.mkdtemp(), tempfile.mkdtemp()
        with mock.patch("ecg_pipeline.digitizer.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=2, stdout="", stderr="boom")
            with self.assertRaises(digitizer.DigitizerFailed) as ctx:
                digitizer.digitize(image_dir=images, output_dir=out, home=Path(fake_checkout()), quiet=True)

        self.assertIn("boom", str(ctx.exception))

    def test_a_missing_image_directory_raises_before_spawning(self):
        with mock.patch("ecg_pipeline.digitizer.subprocess.run") as run:
            with self.assertRaises(FileNotFoundError):
                digitizer.digitize(image_dir="/no/such/dir", output_dir=tempfile.mkdtemp(), home=Path(fake_checkout()))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Tests for the pre-digitization image preparation.

The upscale is what separates a usable digitization from a fragmented one on a
low-resolution scan, so the parts that decide *whether* and *by how much* are pinned here.
Real images are generated with Pillow; no sample data is needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ecg_pipeline.preprocess import (
    MAX_FACTOR,
    TARGET_WIDTH,
    DuplicateRecordName,
    find_images,
    prepare_images,
    preprocessing_warnings,
    upscale_factor,
)


def write_image(directory: Path, name: str, size: tuple[int, int], mode: str = "RGB") -> Path:
    path = directory / name
    Image.new(mode, size, color=1 if mode == "P" else "white").save(path)
    return path


class TestUpscaleFactor(unittest.TestCase):
    def test_the_reference_ecg_gets_the_factor_that_was_validated(self):
        # 797x446 native was digitized at 26% lead-I coverage; x3 -> 2391 took it to 99%.
        self.assertEqual(upscale_factor(797), 3)

    def test_a_full_resolution_photograph_is_left_alone(self):
        self.assertEqual(upscale_factor(3004), 1)

    def test_the_right_sided_ecg_is_capped_at_the_factor_that_was_validated(self):
        # 488 px wants x5 to reach the target, but x3 -> 1464 fits the grid at cost 0.17 and
        # x4 -> 1952 at 0.70. The cap is what keeps it on the better of the two.
        self.assertEqual(upscale_factor(488), 3)

    def test_factor_is_capped(self):
        self.assertEqual(upscale_factor(10), MAX_FACTOR)

    def test_the_threshold_is_the_boundary(self):
        self.assertEqual(upscale_factor(TARGET_WIDTH), 1)
        self.assertEqual(upscale_factor(TARGET_WIDTH - 1), 2)

    def test_a_degenerate_width_does_not_divide_by_zero(self):
        self.assertEqual(upscale_factor(0), 1)


class TestFindImages(unittest.TestCase):
    def test_extensions_match_case_insensitively(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_image(directory, "a.png", (10, 10))
            write_image(directory, "b.JPG", (10, 10))
            (directory / "notes.txt").write_text("ignore me")

            self.assertEqual([p.name for p in find_images(directory)], ["a.png", "b.JPG"])

    def test_a_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            find_images("/nonexistent/directory")

    def test_subdirectories_are_descended_into(self):
        # The digitizer walks its input tree. Staging only the top level would drop every
        # nested image without saying so.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_image(directory, "top.png", (10, 10))
            (directory / "nested").mkdir()
            write_image(directory / "nested", "deep.png", (10, 10))

            self.assertEqual([p.name for p in find_images(directory)], ["deep.png", "top.png"])


class TestPrepareImages(unittest.TestCase):
    def prepare(self, sizes: dict[str, tuple[int, int]], **kwargs):
        """Build a source directory, prepare it, and hand back the records and work dir."""
        source = Path(self.enterContext(tempfile.TemporaryDirectory()))
        work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, size in sizes.items():
            write_image(source, name, size)
        return prepare_images(source, work, **kwargs), work

    def test_a_small_image_is_upscaled_and_the_factor_reported(self):
        prepared, work = self.prepare({"normal.png": (797, 446)})

        record = prepared["normal"]
        self.assertEqual(record["upscale_factor"], 3)
        self.assertEqual(record["original_size"], [797, 446])
        self.assertEqual(record["size"], [2391, 1338])
        with Image.open(work / "normal.png") as written:
            self.assertEqual(written.size, (2391, 1338))

    def test_a_large_image_is_passed_through_untouched(self):
        prepared, work = self.prepare({"photo.jpg": (3004, 1599)})

        self.assertEqual(prepared["photo"]["upscale_factor"], 1)
        self.assertTrue((work / "photo.jpg").exists())
        with Image.open(work / "photo.jpg") as written:
            self.assertEqual(written.size, (3004, 1599))

    def test_off_keeps_the_original_resolution(self):
        prepared, work = self.prepare({"tiny.png": (400, 300)}, upscale="off")

        self.assertEqual(prepared["tiny"]["upscale_factor"], 1)
        with Image.open(work / "tiny.png") as written:
            self.assertEqual(written.size, (400, 300))

    def test_an_explicit_factor_overrides_the_width_rule(self):
        # Wide enough that 'auto' would have left it alone; the strip is short only to keep
        # the resampling in this test cheap.
        prepared, _ = self.prepare({"photo.jpg": (3004, 120)}, upscale=2)

        self.assertEqual(prepared["photo"]["upscale_factor"], 2)
        self.assertEqual(prepared["photo"]["size"], [6008, 240])

    def test_a_factor_below_one_raises(self):
        with self.assertRaises(ValueError):
            self.prepare({"a.png": (10, 10)}, upscale=0)

    def test_the_record_name_survives_the_png_re_encode(self):
        # The digitizer names its output after the input basename and the pipeline joins on
        # it, so an upscaled foo.jpg must still be staged as foo.*, never foo_x3.*.
        prepared, work = self.prepare({"foo.jpg": (500, 300)})

        self.assertIn("foo", prepared)
        self.assertEqual([p.name for p in sorted(work.iterdir())], ["foo.png"])

    def test_colliding_basenames_raise_instead_of_overwriting(self):
        with self.assertRaises(DuplicateRecordName):
            self.prepare({"ecg.png": (500, 300), "ecg.jpg": (500, 300)})

    def test_a_nested_image_keeps_its_relative_path(self):
        source = Path(self.enterContext(tempfile.TemporaryDirectory()))
        work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (source / "batch-1").mkdir()
        write_image(source / "batch-1", "ecg.png", (2400, 4))

        prepared = prepare_images(source, work)

        # The key carries the subdirectory, and the tree is mirrored: the digitizer names its
        # output from the path relative to its input root, and the pipeline joins on that.
        self.assertEqual(list(prepared), ["batch-1/ecg"])
        self.assertTrue((work / "batch-1" / "ecg.png").exists())

    def test_the_same_basename_in_two_subdirectories_is_not_a_collision(self):
        source = Path(self.enterContext(tempfile.TemporaryDirectory()))
        work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for batch in ("batch-1", "batch-2"):
            (source / batch).mkdir()
            write_image(source / batch, "ecg.png", (2400, 4))

        prepared = prepare_images(source, work)

        self.assertEqual(sorted(prepared), ["batch-1/ecg", "batch-2/ecg"])

    def test_a_palette_image_is_converted_before_resampling(self):
        source = Path(self.enterContext(tempfile.TemporaryDirectory()))
        work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_image(source, "paletted.png", (500, 300), mode="P")

        prepare_images(source, work)

        with Image.open(work / "paletted.png") as written:
            self.assertEqual(written.mode, "RGB")

    def test_an_empty_directory_prepares_nothing(self):
        prepared, _ = self.prepare({})

        self.assertEqual(prepared, {})


class TestPreprocessingWarnings(unittest.TestCase):
    def test_a_usable_image_produces_no_warning(self):
        self.assertEqual(preprocessing_warnings({"original_size": [797, 446], "upscale_factor": 3}), [])

    def test_the_right_sided_ecg_is_not_warned_about(self):
        # 488 -> 1464 px is under the 2000 px target but digitizes well, so it must not warn.
        self.assertEqual(preprocessing_warnings({"original_size": [488, 305], "upscale_factor": 3}), [])

    def test_an_image_too_small_to_rescue_warns(self):
        # 300 px at the cap is 900, under anything this pipeline has evidence for:
        # interpolation cannot invent the resolution and the caller should re-scan.
        warnings = preprocessing_warnings({"original_size": [300, 200], "upscale_factor": MAX_FACTOR})

        self.assertEqual(len(warnings), 1)
        self.assertIn("Re-scan", warnings[0])

    def test_no_record_produces_no_warning(self):
        self.assertEqual(preprocessing_warnings(None), [])


if __name__ == "__main__":
    unittest.main()

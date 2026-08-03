"""Tests for reading the digitizer's quality metadata and turning it into warnings.

The layout gate lives here. It is the only thing that catches an ECG whose leads were
never identified -- the coverage gate cannot, because misidentified leads can still be
fully covered.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from ecg_pipeline.pipeline import (
    METADATA_FILENAME,
    UNKNOWN_LAYOUT,
    digitization_warnings,
    read_digitization_metadata,
)

HEADER = "file_path,matching_cost,is_flipped,lead_layout\n"


def metadata_dir(body: str) -> str:
    d = tempfile.mkdtemp()
    (Path(d) / METADATA_FILENAME).write_text(HEADER + body)
    return d


class TestReadMetadata(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        self.assertEqual(read_digitization_metadata(tempfile.mkdtemp()), {})

    def test_parses_a_normal_row(self):
        m = read_digitization_metadata(metadata_dir("ecg1,0.41,False,standard_3x4_with_r3\n"))

        self.assertEqual(set(m), {"ecg1"})
        self.assertAlmostEqual(m["ecg1"]["matching_cost"], 0.41)
        self.assertFalse(m["ecg1"]["is_flipped"])
        self.assertEqual(m["ecg1"]["lead_layout"], "standard_3x4_with_r3")

    def test_last_row_wins_for_a_repeated_record(self):
        """The digitizer appends and only writes a header when the file is absent, so a
        reused output directory accumulates stale rows."""
        m = read_digitization_metadata(metadata_dir("ecg1,0.1,False,standard\necg1,0.9,True,Unknown layout\n"))

        self.assertAlmostEqual(m["ecg1"]["matching_cost"], 0.9)
        self.assertEqual(m["ecg1"]["lead_layout"], UNKNOWN_LAYOUT)

    def test_unparseable_cost_becomes_nan_without_crashing(self):
        m = read_digitization_metadata(metadata_dir("ecg1,NOTANUMBER,False,standard\n"))
        self.assertTrue(math.isnan(m["ecg1"]["matching_cost"]))

    def test_rows_without_a_record_name_are_skipped(self):
        self.assertEqual(read_digitization_metadata(metadata_dir(",0.5,False,standard\n")), {})

    def test_is_flipped_parsing_is_case_insensitive(self):
        m = read_digitization_metadata(metadata_dir("a,0.1,TRUE,standard\nb,0.1,false,standard\n"))
        self.assertTrue(m["a"]["is_flipped"])
        self.assertFalse(m["b"]["is_flipped"])


class TestDigitizationWarnings(unittest.TestCase):
    def test_clean_metadata_produces_no_warnings(self):
        self.assertEqual(
            digitization_warnings({"lead_layout": "standard_3x4_with_r3", "matching_cost": 0.41, "is_flipped": False}),
            [],
        )

    def test_unknown_layout_always_warns(self):
        w = digitization_warnings({"lead_layout": UNKNOWN_LAYOUT, "matching_cost": 1.0, "is_flipped": False})

        self.assertEqual(len(w), 1)
        self.assertIn("could not identify the lead layout", w[0])

    def test_absent_metadata_warns_rather_than_passing_silently(self):
        w = digitization_warnings(None)
        self.assertEqual(len(w), 1)
        self.assertIn("unverified", w[0])

    def test_flipped_image_warns(self):
        w = digitization_warnings({"lead_layout": "standard", "matching_cost": 0.4, "is_flipped": True})
        self.assertTrue(any("flipped" in x for x in w))

    def test_matching_cost_is_not_thresholded_by_default(self):
        """The cost is an unbounded residual, so no universal cutoff is defensible."""
        self.assertEqual(
            digitization_warnings({"lead_layout": "standard", "matching_cost": 99.0, "is_flipped": False}), []
        )

    def test_matching_cost_warns_once_a_limit_is_configured(self):
        w = digitization_warnings(
            {"lead_layout": "standard", "matching_cost": 0.9, "is_flipped": False}, max_matching_cost=0.5
        )
        self.assertTrue(any("exceeds the configured limit" in x for x in w))

    def test_cost_limit_does_not_double_report_an_unknown_layout(self):
        """1.0 is a hardcoded sentinel for 'no match', not a measurement, so it must not
        also surface as a cost-limit breach."""
        w = digitization_warnings(
            {"lead_layout": UNKNOWN_LAYOUT, "matching_cost": 1.0, "is_flipped": False}, max_matching_cost=0.5
        )
        self.assertEqual(len(w), 1)
        self.assertIn("could not identify", w[0])

    def test_nan_cost_does_not_trigger_the_limit(self):
        w = digitization_warnings(
            {"lead_layout": "standard", "matching_cost": float("nan"), "is_flipped": False}, max_matching_cost=0.5
        )
        self.assertEqual(w, [])


if __name__ == "__main__":
    unittest.main()

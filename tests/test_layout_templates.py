"""Tests for reading the digitizer's lead-layout templates as data.

``pipeline._layout_definitions`` is what gates 4 and 5 (a matched layout missing leads it
defines, or a wildcard rhythm strip landing on an unconventional lead) check the digitized
signal against. ``tests/fixtures/lead_layouts_excerpt.yml`` is a small copy of upstream's own
template file, kept only so this exercises real YAML shapes without needing a live
Open-ECG-Digitizer checkout at test time.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ecg_pipeline import digitizer, pipeline

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lead_layouts_excerpt.yml"


class LayoutFixtureCase(unittest.TestCase):
    """Points ``digitizer_home()`` at a throwaway checkout holding the fixture template file."""

    def setUp(self) -> None:
        pipeline._layout_definitions.cache_clear()
        self.addCleanup(pipeline._layout_definitions.cache_clear)
        home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config_dir = home / "src" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "lead_layouts_all.yml").write_text(FIXTURE.read_text())
        self.enterContext(mock.patch.object(digitizer, "digitizer_home", return_value=home))


class TestLayoutDefinitions(LayoutFixtureCase):
    def test_loads_upstream_templates_by_name(self):
        definitions = pipeline._layout_definitions()
        self.assertIn("cabrera_6x1_limb", definitions)
        self.assertIn("standard_3x4_with_r1", definitions)

    def test_own_right_sided_layout_is_merged_in(self):
        definitions = pipeline._layout_definitions()
        self.assertIn("limb_aug_right_3x3", definitions)

    def test_returns_none_when_the_checkout_is_missing(self):
        pipeline._layout_definitions.cache_clear()
        with mock.patch.object(digitizer, "digitizer_home", side_effect=digitizer.DigitizerNotFound("nope")):
            self.assertIsNone(pipeline._layout_definitions())

    def test_returns_none_on_malformed_yaml(self):
        pipeline._layout_definitions.cache_clear()
        home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        config_dir = home / "src" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "lead_layouts_all.yml").write_text("not: [valid: yaml: at: all")
        with mock.patch.object(digitizer, "digitizer_home", return_value=home):
            self.assertIsNone(pipeline._layout_definitions())


class TestLayoutLeads(LayoutFixtureCase):
    def test_flat_leads_list_ignores_polarity_marker(self):
        # cabrera_6x1_limb writes the inverted lead as "-aVR"; the "-" is a polarity flag,
        # not part of the lead's identity.
        self.assertEqual(pipeline._layout_leads("cabrera_6x1_limb"), {"aVL", "I", "aVR", "II", "aVF", "III"})

    def test_nested_leads_grid_is_flattened(self):
        expected = {"I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"}
        self.assertEqual(pipeline._layout_leads("standard_3x4_with_r1"), expected)

    def test_wildcard_rhythm_leads_carry_no_identity(self):
        # standard_3x4_with_r3 adds three "Any" rhythm leads on top of the 12 grid leads;
        # none of them name a real lead, so the set is still exactly the 12.
        self.assertEqual(len(pipeline._layout_leads("standard_3x4_with_r3")), 12)

    def test_unknown_layout_name_returns_none(self):
        self.assertIsNone(pipeline._layout_leads("no_such_layout"))

    def test_returns_none_when_definitions_are_unavailable(self):
        pipeline._layout_definitions.cache_clear()
        with mock.patch.object(digitizer, "digitizer_home", side_effect=digitizer.DigitizerNotFound("nope")):
            self.assertIsNone(pipeline._layout_leads("standard_3x1"))


class TestWildcardRhythm(LayoutFixtureCase):
    def test_true_for_a_layout_with_any_rhythm_leads(self):
        self.assertTrue(pipeline._layout_has_wildcard_rhythm("standard_3x4_with_r1"))

    def test_false_for_a_layout_without_rhythm_leads(self):
        self.assertFalse(pipeline._layout_has_wildcard_rhythm("standard_12x1"))
        self.assertFalse(pipeline._layout_has_wildcard_rhythm("standard_6x2"))

    def test_false_for_an_unknown_layout(self):
        self.assertFalse(pipeline._layout_has_wildcard_rhythm("no_such_layout"))


if __name__ == "__main__":
    unittest.main()

"""Tests for per-class threshold resolution: the loader, and ``interpret_csv``'s fallback
to it when no explicit ``--thresholds``/``--threshold`` was given.

``interpret_csv`` is exercised with a fake model rather than a real checkpoint, the same
way the rest of the suite avoids the ~740 MB of weights (see test_quality.py,
test_pipeline_flow.py): ``predict_probs`` only needs something callable that returns
logits shaped ``(1, 150)``, so torch never has to load a real ``Net1D``.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from ecg_pipeline.interpret import interpret_ecg
from ecg_pipeline.interpret.interpret_ecg import default_thresholds, interpret_csv, load_tasks, thresholded_labels
from ecg_pipeline.interpret.waveform import CANONICAL_LEADS

N = 1000
TASKS = load_tasks()
SINUS_RHYTHM_INDEX = TASKS.index("SINUS RHYTHM")


class FakeModel:
    """A stand-in Net1D: callable, returns fixed logits regardless of the input signal.

    ``probs`` maps class index -> the sigmoid probability ``predict_probs`` should read
    back for it; every other class is pinned near zero. Deterministic, so a test can assert
    on exactly which classes get flagged without a real checkpoint.
    """

    def __init__(self, probs: dict[int, float] | None = None):
        self.probs = probs or {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, len(TASKS)), -10.0)
        for index, p in self.probs.items():
            logits[0, index] = torch.logit(torch.tensor(float(p)))
        return logits


def write_canonical_csv(path: Path, coverage: dict[str, float]) -> None:
    """A canonical CSV where each lead is valid for the given fraction of the record."""
    data = np.full((N, len(CANONICAL_LEADS)), np.nan)
    for column, lead in enumerate(CANONICAL_LEADS):
        valid = int(round(coverage.get(lead, 0.0) * N))
        if valid:
            data[:valid, column] = np.linspace(-1.0, 1.0, valid)
    with path.open("w") as fh:
        fh.write(",".join(CANONICAL_LEADS) + "\n")
        np.savetxt(fh, data, delimiter=",")


class TestDefaultThresholdsLoader(unittest.TestCase):
    def test_returns_none_when_the_directory_has_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", tmp):
            self.assertIsNone(default_thresholds("rhythm"))

    def test_loads_the_1lead_ii_file_for_rhythm_and_1lead(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "thresholds_1lead_II.json").write_text(json.dumps({"SINUS RHYTHM": 0.5}))
            with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", tmp):
                self.assertEqual(default_thresholds("rhythm"), {"SINUS RHYTHM": 0.5})
                self.assertEqual(default_thresholds("1lead"), {"SINUS RHYTHM": 0.5})

    def test_loads_the_12lead_file_for_morphology_and_12lead(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "thresholds_12lead.json").write_text(json.dumps({"ABNORMAL ECG": 0.7}))
            with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", tmp):
                self.assertEqual(default_thresholds("morphology"), {"ABNORMAL ECG": 0.7})
                self.assertEqual(default_thresholds("12lead"), {"ABNORMAL ECG": 0.7})

    def test_ignores_subdirectories(self):
        # configs/thresholds/provisional/ holds the fold-10 derivation deliberately not
        # auto-loaded: default_thresholds only ever reads THRESHOLDS_DIR/thresholds_*.json
        # directly, never a subdirectory, so shipping those files does not silently change
        # what a caller who asked for nothing gets.
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "provisional" / "fold10_2026-09-11"
            sub.mkdir(parents=True)
            (sub / "thresholds_1lead_II.json").write_text(json.dumps({"SINUS RHYTHM": 0.5}))
            with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", tmp):
                self.assertIsNone(default_thresholds("rhythm"))

    def test_an_unknown_pathway_is_none_rather_than_a_lookup_error(self):
        self.assertIsNone(default_thresholds("not-a-pathway"))

    def test_malformed_json_returns_none_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "thresholds_1lead_II.json").write_text("{not json")
            stderr = io.StringIO()
            with (
                mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", tmp),
                mock.patch("sys.stderr", stderr),
            ):
                result = default_thresholds("rhythm")

        self.assertIsNone(result)
        self.assertIn("could not load thresholds", stderr.getvalue())


class TestThresholdedLabels(unittest.TestCase):
    """The helper ``to_observations`` reads to tell "no threshold to clear" apart from
    "checked and absent" -- see contract.to_observations and configs/thresholds/README.md.
    """

    def test_a_flat_default_covers_every_task(self):
        self.assertEqual(thresholded_labels(["B", "A"], default=0.5), ["A", "B"])

    def test_a_per_class_dict_covers_only_its_own_entries(self):
        # A partial set (this repo's provisional fold-10 derivation, 23 of 150 classes) must
        # not silently expand to "every task" -- that is exactly the bug this exists to fix.
        self.assertEqual(thresholded_labels(["A", "B", "C"], thresholds={"B": 0.3}), ["B"])

    def test_neither_given_is_empty(self):
        self.assertEqual(thresholded_labels(["A", "B"]), [])

    def test_a_flat_default_wins_even_alongside_a_partial_dict(self):
        # Mirrors flagged_findings' own fallback: a class missing from thresholds still
        # gets `default`, so every task ends up with a threshold applied.
        self.assertEqual(thresholded_labels(["A", "B"], thresholds={"A": 0.2}, default=0.5), ["A", "B"])


class TestInterpretCsvDefaultThresholds(unittest.TestCase):
    """``interpret_csv`` resolves thresholds itself only when the caller gave neither
    ``thresholds`` nor ``flat_threshold`` -- both existing entry points into that choice.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.csv_path = Path(self.tmpdir.name) / "ecg_timeseries_canonical.csv"
        # II full-length, everything else a 2.5s grid fragment -- select_rhythm_leads picks
        # up only II at the default coverage_min, same shape test_quality.py's fixtures use.
        coverage = {lead: 0.25 for lead in CANONICAL_LEADS} | {"II": 1.0}
        write_canonical_csv(self.csv_path, coverage)

    def test_no_thresholds_file_present_reports_none(self):
        with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", self.tmpdir.name):
            result = interpret_csv(str(self.csv_path), pathway="rhythm", model=FakeModel())

        self.assertEqual(result["threshold_source"], "none")
        self.assertNotIn("flagged", result)

    def test_default_thresholds_are_picked_up_when_none_given_explicitly(self):
        (Path(self.tmpdir.name) / "thresholds_1lead_II.json").write_text(json.dumps({"SINUS RHYTHM": 0.5}))

        with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", self.tmpdir.name):
            result = interpret_csv(str(self.csv_path), pathway="rhythm", model=FakeModel({SINUS_RHYTHM_INDEX: 0.9}))

        self.assertEqual(result["threshold_source"], "default:thresholds_1lead_II.json")
        self.assertEqual([row["label"] for row in result["flagged"]], ["SINUS RHYTHM"])
        # Only the one class this (deliberately partial) default file covers.
        self.assertEqual(result["thresholded_labels"], ["SINUS RHYTHM"])

    def test_a_partial_default_set_does_not_claim_every_class_was_thresholded(self):
        # The regression this exists for: a 23-of-150-class default file (this repo's
        # provisional fold-10 derivation) must not make every other class report
        # aboveThreshold=false downstream -- thresholded_labels is how contract.to_observations
        # tells "no threshold to clear" apart from "checked and absent".
        (Path(self.tmpdir.name) / "thresholds_1lead_II.json").write_text(
            json.dumps({"SINUS RHYTHM": 0.99, "ATRIAL FIBRILLATION": 0.5})
        )

        with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", self.tmpdir.name):
            result = interpret_csv(str(self.csv_path), pathway="rhythm", model=FakeModel({SINUS_RHYTHM_INDEX: 0.9}))

        self.assertEqual(result["thresholded_labels"], sorted(["SINUS RHYTHM", "ATRIAL FIBRILLATION"]))
        self.assertNotIn("NORMAL ECG", result["thresholded_labels"])

    def test_an_explicit_flat_threshold_wins_over_the_default_file(self):
        (Path(self.tmpdir.name) / "thresholds_1lead_II.json").write_text(json.dumps({"SINUS RHYTHM": 0.99}))

        with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", self.tmpdir.name):
            result = interpret_csv(
                str(self.csv_path),
                pathway="rhythm",
                model=FakeModel({SINUS_RHYTHM_INDEX: 0.9}),
                flat_threshold=0.1,
            )

        self.assertEqual(result["threshold_source"], "flat=0.1")
        # A flat threshold applies to every class, unlike a partial per-class file.
        self.assertEqual(result["thresholded_labels"], sorted(TASKS))

    def test_an_explicit_thresholds_dict_wins_over_the_default_file(self):
        (Path(self.tmpdir.name) / "thresholds_1lead_II.json").write_text(json.dumps({"SINUS RHYTHM": 0.99}))

        with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", self.tmpdir.name):
            result = interpret_csv(
                str(self.csv_path),
                pathway="rhythm",
                model=FakeModel({SINUS_RHYTHM_INDEX: 0.9}),
                thresholds={"SINUS RHYTHM": 0.1},
            )

        self.assertEqual(result["threshold_source"], "per-class")


class TestInterpretRhythmLeadSelection(unittest.TestCase):
    """interpret_csv's rhythm_leads (through interpret_rhythm/select_rhythm_leads) is where
    assess_quality's lead selection actually shows up in a result -- see test_quality.py for
    the selection logic itself.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.csv_path = Path(self.tmpdir.name) / "ecg_timeseries_canonical.csv"

    def test_v1_is_left_out_of_rhythm_leads_when_a_preferred_strip_is_present(self):
        # The realistic 3x4-with-rhythm-strips shape: II, V1 and V5 all full length. Only
        # the preferred ones (II, V5) end up in rhythm_leads and therefore in the averaged
        # result; V1 is reported as available (signal_quality) but not used.
        coverage = {lead: 0.25 for lead in CANONICAL_LEADS} | {"II": 1.0, "V1": 1.0, "V5": 1.0}
        write_canonical_csv(self.csv_path, coverage)

        with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", self.tmpdir.name):
            result = interpret_csv(str(self.csv_path), pathway="rhythm", model=FakeModel())

        self.assertEqual(result["rhythm_leads"], ["II", "V5"])
        self.assertEqual(result["signal_quality"]["full_length_leads"], ["II", "V5", "V1"])
        self.assertTrue(any("V1" in w and "left out" in w for w in result["warnings"]))

    def test_v1_alone_still_drives_rhythm_leads(self):
        # No preferred lead reached full length, so the fallback (every full-length lead)
        # applies and V1 is used, same as select_rhythm_leads on its own.
        coverage = {lead: 0.25 for lead in CANONICAL_LEADS} | {"V1": 1.0}
        write_canonical_csv(self.csv_path, coverage)

        with mock.patch.object(interpret_ecg, "THRESHOLDS_DIR", self.tmpdir.name):
            result = interpret_csv(str(self.csv_path), pathway="rhythm", model=FakeModel())

        self.assertEqual(result["rhythm_leads"], ["V1"])
        self.assertFalse(result["degraded"])


if __name__ == "__main__":
    unittest.main()

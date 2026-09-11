"""Tests for the app-EKG contract emitter.

Three of these guard conversions that would put a plausible lie on a screen: microvolts
read as millivolts, a gap in the trace turned into signal, and a right-sided lead shown
under a left-sided name.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest

import numpy as np

from ecg_pipeline.contract import (
    MAX_BRIDGED_GAP_SECONDS,
    bridged_holes,
    failure_reason,
    lead_segments,
    observation_category,
    observation_id,
    observed_leads,
    signal_bridging_report,
    signal_from_csv,
    to_analysis,
    to_observations,
    to_signal,
)
from ecg_pipeline.interpret.waveform import CANONICAL_LEADS
from ecg_pipeline.label_categories import DEFAULT_CATEGORY, LABEL_CATEGORIES

VALID_CATEGORIES = {
    "ritmo",
    "conduccion",
    "repolarizacion",
    "isquemia_infarto",
    "marcapasos",
    "eje",
    "hipertrofia",
    "tecnico",
    "resumen",
    "otro",
}

FS = 500


class TestLeadSegments(unittest.TestCase):
    def test_microvolts_become_millivolts(self):
        segments = lead_segments(np.array([1000.0, -500.0, 250.0]), FS)

        self.assertEqual(segments[0]["values"], [1.0, -0.5, 0.25])

    def test_a_gap_splits_the_lead_rather_than_being_filled(self):
        # The failure this prevents: a straight line drawn across 7.5 s the heart was never
        # recorded for, indistinguishable from a real flat trace.
        values = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0])

        segments = lead_segments(values, fs=2)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["startSecond"], 0.0)
        self.assertEqual(segments[1]["startSecond"], 2.0)

    def test_each_segment_starts_at_the_second_it_was_recorded(self):
        values = np.full(1000, np.nan)
        values[250:500] = 1000.0

        segments = lead_segments(values, FS)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["startSecond"], 0.5)
        self.assertEqual(len(segments[0]["values"]), 250)

    def test_a_lead_with_no_signal_has_no_segments(self):
        self.assertEqual(lead_segments(np.full(100, np.nan), FS), [])

    def test_segment_values_never_contain_nan(self):
        values = np.array([1.0, np.nan, 2.0])

        for segment in lead_segments(values, fs=1):
            self.assertFalse(any(v != v for v in segment["values"]))


class TestBridgedDropouts(unittest.TestCase):
    """Bridging a digitizer dropout must never touch a real gap.

    Measured on a real record (study 43be167a): V1-V4 arrive split by 14-34 ms holes where
    the digitizer lost the trace under a label or a bold grid line, while the real gaps on
    that record are hundreds of milliseconds or more. 0.04 s sits between the two.
    """

    def test_a_three_sample_hole_is_bridged_with_linear_interpolation(self):
        # 3 samples at 500 Hz = 6 ms, well under the 40 ms threshold.
        values = np.array([1000.0, 2000.0, np.nan, np.nan, np.nan, 6000.0, 7000.0])

        segments = lead_segments(values, FS)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["values"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_a_fifty_millisecond_hole_is_not_bridged_and_still_splits(self):
        # 25 samples at 500 Hz = 50 ms, over the threshold: "gaps stay gaps" still applies.
        values = np.concatenate([[1.0, 2.0], np.full(25, np.nan), [3.0, 4.0]])

        segments = lead_segments(values, FS)

        self.assertEqual(len(segments), 2)

    def test_bridged_holes_reports_only_what_was_bridged(self):
        values = np.concatenate([[1.0, 2.0], np.full(3, np.nan), [3.0], np.full(25, np.nan), [4.0]])

        self.assertEqual(bridged_holes(values, FS), [6.0])

    def test_a_hole_touching_the_edge_is_never_bridged(self):
        # Nothing on one side to interpolate from -- the lead has not started yet, this is
        # not a dropout inside a printed stretch.
        values = np.concatenate([np.full(3, np.nan), [1.0, 2.0]])

        self.assertEqual(bridged_holes(values, FS), [])
        self.assertEqual(len(lead_segments(values, FS)), 1)

    def test_threshold_is_shorter_than_any_qrs_complex(self):
        # QRS complexes run roughly 60-120 ms; the bridge must stay well under that so it
        # can never span, and so hide, part of a beat.
        self.assertLess(MAX_BRIDGED_GAP_SECONDS, 0.06)


class TestSignalFromCsv(unittest.TestCase):
    """The public entry point for the trace alone, without interpretation."""

    def _write_canonical_csv(self, path):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CANONICAL_LEADS)
            for _ in range(10):
                writer.writerow([1000.0] * len(CANONICAL_LEADS))

    def test_matches_what_to_analysis_builds_internally(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ecg_timeseries_canonical.csv")
            self._write_canonical_csv(path)

            signal = signal_from_csv(path, fs=FS)

            self.assertEqual(signal["samplingRateHz"], FS)
            self.assertEqual(signal["durationSeconds"], round(10 / FS, 6))
            self.assertEqual([lead["name"] for lead in signal["leads"]], CANONICAL_LEADS)
            self.assertEqual(signal["leads"][0]["segments"][0]["values"][0], 1.0)

    def test_relabels_right_sided_leads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ecg_timeseries_canonical.csv")
            self._write_canonical_csv(path)

            signal = signal_from_csv(path, lead_layout="limb_aug_right_3x3", fs=FS)

            self.assertIn("V4R", [lead["name"] for lead in signal["leads"]])


class TestSignalBridgingReport(unittest.TestCase):
    def test_reports_bridged_holes_per_lead(self):
        canonical = np.full((len(CANONICAL_LEADS), 20), 1000.0)
        index = CANONICAL_LEADS.index("II")
        canonical[index, 5:8] = np.nan  # 3 samples, bridged

        report = signal_bridging_report(canonical, list(CANONICAL_LEADS), FS)

        self.assertEqual(list(report.keys()), ["II"])
        self.assertEqual(report["II"], bridged_holes(canonical[index], FS))

    def test_a_lead_with_no_bridged_holes_is_absent_from_the_report(self):
        canonical = np.full((len(CANONICAL_LEADS), 20), 1000.0)

        self.assertEqual(signal_bridging_report(canonical, list(CANONICAL_LEADS), FS), {})


class TestToSignal(unittest.TestCase):
    def frame(self, coverage: dict[str, float], n: int = 1000):
        canonical = np.full((len(CANONICAL_LEADS), n), np.nan)
        for index, lead in enumerate(CANONICAL_LEADS):
            valid = int(coverage.get(lead, 0.0) * n)
            if valid:
                canonical[index, :valid] = 1000.0
        return canonical, list(CANONICAL_LEADS)

    def test_duration_covers_the_gaps_too(self):
        canonical, names = self.frame({"II": 0.25})

        signal = to_signal(canonical, names, FS)

        self.assertEqual(signal["durationSeconds"], 2.0)
        self.assertEqual(signal["samplingRateHz"], FS)

    def test_leads_without_signal_are_omitted(self):
        canonical, names = self.frame({"II": 1.0, "V1": 1.0})

        signal = to_signal(canonical, names, FS)

        self.assertEqual([lead["name"] for lead in signal["leads"]], ["II", "V1"])

    def test_a_right_sided_layout_relabels_the_precordial_slots(self):
        # The digitizer identifies leads by position, so on this layout the V4/V5/V6 slots
        # hold V4R/V5R/V6R. Showing them as left-sided would misreport which side of the
        # heart was recorded.
        canonical, names = self.frame({l: 1.0 for l in ("I", "V4", "V5", "V6")})

        signal = to_signal(canonical, names, FS, lead_layout="limb_aug_right_3x3")

        self.assertEqual([lead["name"] for lead in signal["leads"]], ["I", "V4R", "V5R", "V6R"])

    def test_a_standard_layout_keeps_its_lead_names(self):
        canonical, names = self.frame({l: 1.0 for l in ("I", "V4", "V5", "V6")})

        signal = to_signal(canonical, names, FS, lead_layout="standard_3x4_with_r3")

        self.assertEqual([lead["name"] for lead in signal["leads"]], ["I", "V4", "V5", "V6"])


class TestObservations(unittest.TestCase):
    def test_labels_pass_through_unrewritten(self):
        result = {"pathway": "rhythm", "rhythm_leads": ["II"], "topk": [{"label": "SINUS RHYTHM", "prob": 0.99}]}

        observations = to_observations(result)

        self.assertEqual(observations[0]["label"], "SINUS RHYTHM")
        self.assertEqual(observations[0]["id"], "sinus-rhythm")
        self.assertEqual(observations[0]["confidence"], 0.99)

    def test_every_observation_needs_review(self):
        result = {
            "pathway": "rhythm",
            "rhythm_leads": ["II"],
            "topk": [{"label": "A", "prob": 0.9}, {"label": "B", "prob": 0.1}],
        }

        self.assertTrue(all(o["needsReview"] for o in to_observations(result)))

    def test_observations_point_at_the_leads_that_were_read(self):
        rhythm = {"pathway": "rhythm", "rhythm_leads": ["II", "V5"], "topk": []}
        single = {"pathway": "1lead", "lead": "II", "topk": []}
        morphology = {"pathway": "morphology", "signal_quality": {"leads_with_signal": ["I", "II"]}, "topk": []}

        self.assertEqual(observed_leads(rhythm), ["II", "V5"])
        self.assertEqual(observed_leads(single), ["II"])
        self.assertEqual(observed_leads(morphology), ["I", "II"])

    def test_ids_survive_punctuation(self):
        self.assertEqual(observation_id("1st DEGREE AV BLOCK"), "1st-degree-av-block")

    def test_a_known_label_carries_its_category(self):
        result = {"pathway": "rhythm", "rhythm_leads": ["II"], "topk": [{"label": "ATRIAL FIBRILLATION", "prob": 0.9}]}

        observations = to_observations(result)

        self.assertEqual(observations[0]["category"], "ritmo")

    def test_a_label_the_csv_does_not_cover_falls_back_to_otro(self):
        result = {"pathway": "rhythm", "rhythm_leads": ["II"], "topk": [{"label": "MADE UP LABEL", "prob": 0.9}]}

        observations = to_observations(result)

        self.assertEqual(observations[0]["category"], DEFAULT_CATEGORY)
        self.assertEqual(observation_category("MADE UP LABEL"), "otro")

    def test_no_flagged_key_means_ranking_only(self):
        # interpret_csv sets threshold_source but not "flagged" only when it resolved
        # neither an explicit nor a default threshold -- see interpret_ecg.default_thresholds.
        result = {"pathway": "rhythm", "rhythm_leads": ["II"], "topk": [{"label": "A", "prob": 0.9}]}

        observations = to_observations(result)

        self.assertIsNone(observations[0]["aboveThreshold"])

    def test_flagged_findings_come_first_marked_above_threshold(self):
        # A flat threshold applies to every class (interpret_csv's thresholded_labels
        # wiring), so the non-flagged class still comes back False, same as before this
        # field existed.
        result = {
            "pathway": "rhythm",
            "rhythm_leads": ["II"],
            "topk": [{"label": "A", "prob": 0.9}, {"label": "B", "prob": 0.2}],
            "flagged": [{"label": "A", "prob": 0.9, "threshold": 0.5}],
            "thresholded_labels": ["A", "B"],
        }

        observations = to_observations(result)

        self.assertEqual([(o["label"], o["aboveThreshold"]) for o in observations], [("A", True), ("B", False)])

    def test_a_non_flagged_class_with_no_threshold_applied_is_null_not_false(self):
        # The bug this fixes: a partial per-class threshold set (e.g. 23 of 150 classes)
        # means most non-flagged classes were never checked against a threshold at all.
        # "B" is not in thresholded_labels, so it must read null ("no verdict"), not false
        # ("checked and absent").
        result = {
            "pathway": "rhythm",
            "rhythm_leads": ["II"],
            "topk": [{"label": "A", "prob": 0.9}, {"label": "B", "prob": 0.2}],
            "flagged": [{"label": "A", "prob": 0.9, "threshold": 0.5}],
            "thresholded_labels": ["A"],
        }

        observations = to_observations(result)

        self.assertEqual([(o["label"], o["aboveThreshold"]) for o in observations], [("A", True), ("B", None)])

    def test_absent_thresholded_labels_key_treats_every_non_flagged_row_as_null(self):
        # Defensive default for a hand-built result dict that predates thresholded_labels
        # (e.g. an older interpretation JSON on disk): with no list to consult, "not
        # flagged" cannot be distinguished from "never checked", so it reads null.
        result = {
            "pathway": "rhythm",
            "rhythm_leads": ["II"],
            "topk": [{"label": "A", "prob": 0.9}, {"label": "B", "prob": 0.2}],
            "flagged": [{"label": "A", "prob": 0.9, "threshold": 0.5}],
        }

        observations = to_observations(result)

        self.assertIsNone(next(o for o in observations if o["label"] == "B")["aboveThreshold"])

    def test_a_flagged_finding_outside_the_topk_still_appears(self):
        # flagged_findings scans every class; topk is only the top slice, so a class can
        # clear its threshold and still fall outside it.
        result = {
            "pathway": "rhythm",
            "rhythm_leads": ["II"],
            "topk": [{"label": "A", "prob": 0.9}],
            "flagged": [
                {"label": "A", "prob": 0.9, "threshold": 0.5},
                {"label": "Z", "prob": 0.6, "threshold": 0.5},
            ],
        }

        observations = to_observations(result)

        self.assertEqual([o["label"] for o in observations], ["A", "Z"])
        self.assertTrue(all(o["aboveThreshold"] for o in observations))


class TestLabelCategories(unittest.TestCase):
    def test_every_label_has_one_of_the_ten_fixed_categories(self):
        self.assertEqual(len(LABEL_CATEGORIES), 150)
        for label, category in LABEL_CATEGORIES.items():
            self.assertIn(category, VALID_CATEGORIES, f"{label!r} has an invalid category {category!r}")

    def test_default_category_is_one_of_the_fixed_slugs(self):
        self.assertIn(DEFAULT_CATEGORY, VALID_CATEGORIES)


class TestFailureReason(unittest.TestCase):
    def test_a_clean_result_has_no_failure(self):
        self.assertIsNone(failure_reason({"degraded": False, "source_csv": "x.csv"}))

    def test_an_image_that_produced_nothing_is_unreadable(self):
        self.assertEqual(failure_reason({"degraded": True, "error": "skipped"}), "unreadable-image")

    def test_an_unidentified_layout_is_an_unsupported_mount(self):
        result = {"degraded": True, "source_csv": "x.csv", "digitization": {"lead_layout": "Unknown layout"}}

        self.assertEqual(failure_reason(result), "unsupported-mount")

    def test_an_interpretation_error_is_a_server_error(self):
        result = {"degraded": True, "source_csv": "x.csv", "error": "RuntimeError: boom"}

        self.assertEqual(failure_reason(result), "server-error")

    def test_a_fragmented_trace_with_a_known_layout_is_trace_incomplete(self):
        # The layout matched and the CSV exists, but no lead was printed at full length:
        # the photograph is the problem, and the user should be told to retake it rather
        # than that the server failed.
        result = {
            "degraded": True,
            "source_csv": "x.csv",
            "digitization": {"lead_layout": "standard_3x4"},
            "signal_quality": {
                "leads_with_signal": ["I", "II"],
                "full_length_leads": [],
                "needs_full_length": True,
                "degraded": True,
            },
        }

        self.assertEqual(failure_reason(result), "trace-incomplete")

    def test_no_signal_at_all_is_trace_incomplete(self):
        result = {
            "degraded": True,
            "source_csv": "x.csv",
            "digitization": {"lead_layout": "standard_3x4"},
            "signal_quality": {"leads_with_signal": [], "full_length_leads": [], "needs_full_length": True},
        }

        self.assertEqual(failure_reason(result), "trace-incomplete")

    def test_a_scan_that_yielded_no_signal_is_not_a_server_error(self):
        # Measured on a blank image pushed through the whole pipeline. The digitizer
        # wrote a CSV, interpretation raised "No usable lead found in canonical CSV",
        # and the error branch blamed the service. Nothing was broken here: the image
        # had nothing on it, and the record says so -- every lead at zero coverage --
        # before any exception is considered.
        result = {
            "degraded": True,
            "source_csv": "x.csv",
            "error": "ValueError: No usable lead found in canonical CSV.",
            "signal_quality": {"leads_with_signal": [], "full_length_leads": [], "needs_full_length": True},
        }

        self.assertEqual(failure_reason(result), "trace-incomplete")

    def test_a_crash_with_signal_present_is_still_a_server_error(self):
        # The other half. Blaming the image for every failure would be the same mistake
        # in the other direction, and it is this service that the user cannot fix.
        result = {
            "degraded": True,
            "source_csv": "x.csv",
            "error": "RuntimeError: boom",
            "signal_quality": {"leads_with_signal": ["II", "V5"]},
        }

        self.assertEqual(failure_reason(result), "server-error")

    def test_a_pathway_that_degrades_itself_is_not_trace_incomplete(self):
        # morphology and 12lead set degraded on their own quality dict for reasons that say
        # nothing about the trace; that must not be reported as a bad photograph.
        result = {
            "degraded": True,
            "source_csv": "x.csv",
            "digitization": {"lead_layout": "standard_3x4"},
            "signal_quality": {
                "leads_with_signal": ["I", "II", "V1"],
                "full_length_leads": ["II"],
                "needs_full_length": False,
                "degraded": True,
            },
        }

        self.assertEqual(failure_reason(result), "unexpected")


class TestFailureReasonFromGates(unittest.TestCase):
    def test_no_signal_outranks_the_interpretation_error_it_causes(self):
        # A blank photograph recovers nothing, so interpretation fails too; the cause the
        # user needs is the photograph, not the server.
        result = {"degraded": True, "source_csv": "x.csv", "gates": ["no-signal", "interpretation-error"]}

        self.assertEqual(failure_reason(result), "trace-incomplete")

    """``result["gates"]`` (pipeline.py's stable gate ids) is now the preferred source for
    ``failure_reason``; the field-based logic above stays only as a fallback for a result
    that predates it. Each of these results carries deliberately misleading fields -- no
    ``source_csv``, no ``error``, an "Unknown layout" -- to prove the gate id wins even when
    the field-based fallback would have picked a different (or no) reason.
    """

    def base(self, gate: str) -> dict:
        return {"degraded": True, "source_csv": "x.csv", "gates": [gate]}

    def test_digitizer_no_output_is_unreadable_image(self):
        self.assertEqual(failure_reason(self.base("digitizer-no-output")), "unreadable-image")

    def test_interpretation_error_is_server_error(self):
        self.assertEqual(failure_reason(self.base("interpretation-error")), "server-error")

    def test_layout_unknown_is_unsupported_mount(self):
        self.assertEqual(failure_reason(self.base("layout-unknown")), "unsupported-mount")

    def test_leads_missing_from_template_is_unsupported_mount(self):
        self.assertEqual(failure_reason(self.base("leads-missing-from-template")), "unsupported-mount")

    def test_rhythm_strip_unverified_is_unsupported_mount(self):
        self.assertEqual(failure_reason(self.base("rhythm-strip-unverified")), "unsupported-mount")

    def test_no_signal_is_trace_incomplete(self):
        self.assertEqual(failure_reason(self.base("no-signal")), "trace-incomplete")

    def test_no_full_length_lead_is_trace_incomplete(self):
        self.assertEqual(failure_reason(self.base("no-full-length-lead")), "trace-incomplete")

    def test_a_blank_scan_whose_interpretation_raised_is_trace_incomplete(self):
        # The field-based fix for a blank photograph, carried over to the gates. On an empty
        # trace interpretation always raises, so both ids fire; "interpretation-error" is
        # listed first and would otherwise blame the server for a picture with nothing on it.
        both = {"degraded": True, "source_csv": "x.csv", "gates": ["no-signal", "interpretation-error"]}
        only_the_error = {
            "degraded": True,
            "source_csv": "x.csv",
            "gates": ["interpretation-error"],
            "signal_quality": {"leads_with_signal": []},
        }

        self.assertEqual(failure_reason(both), "trace-incomplete")
        self.assertEqual(failure_reason(only_the_error), "trace-incomplete")

    def test_an_interpretation_error_with_signal_present_is_still_server_error(self):
        result = {
            "degraded": True,
            "source_csv": "x.csv",
            "gates": ["interpretation-error"],
            "signal_quality": {"leads_with_signal": ["II", "V5"]},
        }

        self.assertEqual(failure_reason(result), "server-error")

    def test_an_empty_gates_list_falls_through_to_unexpected(self):
        # degraded for a reason none of the gate ids cover -- gates is present (so the
        # field-based fallback does not run) but empty.
        self.assertEqual(failure_reason({"degraded": True, "source_csv": "x.csv", "gates": []}), "unexpected")

    def test_first_match_wins_in_severity_order(self):
        # An unknown layout also fails lead completeness by construction; the more specific
        # "layout-unknown" is listed first in _GATE_REASONS and must win.
        result = {"degraded": True, "source_csv": "x.csv", "gates": ["leads-missing-from-template", "layout-unknown"]}

        self.assertEqual(failure_reason(result), "unsupported-mount")

    def test_digitizer_no_output_wins_even_with_other_gates_present(self):
        result = {"degraded": True, "gates": ["layout-unknown", "digitizer-no-output"]}

        self.assertEqual(failure_reason(result), "unreadable-image")

    def test_a_clean_result_with_gates_present_has_no_failure(self):
        self.assertIsNone(failure_reason({"degraded": False, "gates": []}))

    def test_a_result_without_gates_still_uses_the_field_based_fallback(self):
        # No "gates" key at all -- e.g. older interpretation JSON written before this field
        # existed. Same result as test_an_unidentified_layout_is_an_unsupported_mount above.
        result = {"degraded": True, "source_csv": "x.csv", "digitization": {"lead_layout": "Unknown layout"}}

        self.assertEqual(failure_reason(result), "unsupported-mount")


class TestToAnalysis(unittest.TestCase):
    def test_a_degraded_result_is_not_handed_over_as_ready(self):
        # The distinction the whole pipeline exists for. The contract has no third state,
        # so a reading that must not be trusted goes over as failed rather than as
        # observations on a screen.
        result = {
            "degraded": True,
            "source_csv": None,
            "error": "skipped",
            "topk": [{"label": "SINUS RHYTHM", "prob": 0.99}],
        }

        analysis = to_analysis(result, "study-1", completed_at="2026-08-05T12:00:00Z")

        self.assertEqual(analysis["status"], "failed")
        self.assertEqual(analysis["observations"], [])
        self.assertIsNone(analysis["completedAt"])
        self.assertEqual(analysis["failure"], "unreadable-image")

    def test_measurements_are_never_invented(self):
        result = {"degraded": True, "source_csv": None, "error": "skipped"}

        self.assertIsNone(to_analysis(result, "study-1")["measurements"])


if __name__ == "__main__":
    unittest.main()

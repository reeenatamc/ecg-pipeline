"""Tests for the app-EKG contract emitter.

Three of these guard conversions that would put a plausible lie on a screen: microvolts
read as millivolts, a gap in the trace turned into signal, and a right-sided lead shown
under a left-sided name.
"""

from __future__ import annotations

import unittest

import numpy as np

from ecg_pipeline.contract import (
    failure_reason,
    lead_segments,
    observation_id,
    observed_leads,
    to_analysis,
    to_observations,
    to_signal,
)
from ecg_pipeline.interpret.waveform import CANONICAL_LEADS

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

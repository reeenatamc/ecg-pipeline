"""Tests for the signal-quality logic that decides whether a result is trustworthy.

This is the safety-critical path: it is what stops a failed digitization from being
reported as a confident diagnosis. Everything here runs on synthetic arrays -- no model
weights, no digitizer checkout, no images.
"""

from __future__ import annotations

import unittest

import numpy as np

from ecg_pipeline.interpret.interpret_ecg import (
    CANONICAL_LEADS,
    assess_quality,
    clean_lead,
    select_rhythm_leads,
    zscore,
)

N = 1000


def frame(coverage: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """Build a canonical (12, N) frame where each lead has the given valid fraction."""
    canonical = np.full((len(CANONICAL_LEADS), N), np.nan)
    for i, lead in enumerate(CANONICAL_LEADS):
        k = int(round(coverage.get(lead, 0.0) * N))
        if k:
            canonical[i, :k] = np.linspace(-1.0, 1.0, k)
    return canonical, list(CANONICAL_LEADS)


class TestAssessQuality(unittest.TestCase):
    def test_full_length_rhythm_strips_are_not_degraded(self):
        canonical, names = frame({l: 0.25 for l in CANONICAL_LEADS} | {"II": 1.0, "V1": 1.0, "V5": 1.0})
        q = assess_quality(canonical, names)

        self.assertFalse(q["degraded"])
        self.assertEqual(q["warnings"], [])
        self.assertEqual(q["full_length_leads"], ["II", "V1", "V5"])
        self.assertEqual(q["selected_leads"], ["II", "V1", "V5"])

    def test_preferred_rhythm_leads_come_first(self):
        canonical, names = frame({"I": 1.0, "V5": 1.0, "aVR": 1.0, "II": 1.0})
        q = assess_quality(canonical, names)

        # II and V5 are preferred rhythm leads and must precede the incidental ones.
        self.assertEqual(q["selected_leads"][:2], ["II", "V5"])
        self.assertEqual(sorted(q["selected_leads"][2:]), ["I", "aVR"])

    def test_no_full_length_lead_degrades_and_warns(self):
        """The regression this suite exists for: a 2.5 s fragment must not pass as a
        rhythm-strip ensemble."""
        canonical, names = frame({l: 0.25 for l in CANONICAL_LEADS} | {"V2": 0.52})
        q = assess_quality(canonical, names)

        self.assertTrue(q["degraded"])
        self.assertEqual(q["selected_leads"], ["V2"])  # best-covered, not a rhythm strip
        self.assertTrue(any("coverage" in w for w in q["warnings"]))

    def test_coverage_min_is_the_boundary(self):
        canonical, names = frame({"II": 0.6})
        self.assertFalse(assess_quality(canonical, names, coverage_min=0.6)["degraded"])
        self.assertTrue(assess_quality(canonical, names, coverage_min=0.61)["degraded"])

    def test_few_leads_with_signal_warns_even_when_those_are_full_length(self):
        """The infarct case: 3 leads at ~98% pass the coverage gate but the digitization
        is still incomplete."""
        canonical, names = frame({"I": 0.98, "aVR": 0.98, "aVL": 0.98})
        q = assess_quality(canonical, names)

        self.assertEqual(q["leads_with_signal"], ["I", "aVR", "aVL"])
        self.assertTrue(any("Only 3 of 12" in w for w in q["warnings"]))

    def test_a_six_lead_recovery_warns_and_names_what_is_missing(self):
        """The wrong-layout case. Upscaling the right-sided ECG stopped the digitizer from
        reporting 'Unknown layout' -- it matched a 6-lead precordial_3x2 at a cost
        indistinguishable from a correct match. The six absent leads are what give it away,
        so the threshold is any missing lead, not a handful."""
        canonical, names = frame({l: 0.98 for l in ("V1", "V2", "V3", "V4", "V5", "V6")})
        q = assess_quality(canonical, names)

        self.assertFalse(q["degraded"])  # the coverage gate has no complaint; that is the point
        self.assertEqual(q["leads_missing"], ["I", "II", "III", "aVR", "aVL", "aVF"])
        warning = next(w for w in q["warnings"] if "Only 6 of 12" in w)
        self.assertIn("missing I, II, III, aVR, aVL, aVF", warning)
        self.assertIn("--lead-layout", warning)

    def test_a_pathway_that_needs_no_strip_is_not_degraded_for_lacking_one(self):
        """A 3x3 print has no rhythm strip at all, by construction. Reading its morphology
        off ~2.5 s column windows is what the representative beat does on purpose, so the
        rhythm pathway's complaint must not travel with it."""
        canonical, names = frame({l: 0.33 for l in CANONICAL_LEADS})

        rhythm = assess_quality(canonical, names)
        morphology = assess_quality(canonical, names, needs_full_length=False)

        self.assertTrue(rhythm["degraded"])
        self.assertFalse(morphology["degraded"])
        self.assertEqual(morphology["warnings"], [])

    def test_a_frame_with_no_signal_degrades_for_every_pathway(self):
        # needs_full_length only relaxes the strip requirement; nothing rescues an empty frame.
        canonical, names = frame({})

        self.assertTrue(assess_quality(canonical, names, needs_full_length=False)["degraded"])

    def test_lead_completeness_applies_to_every_pathway(self):
        canonical, names = frame({l: 0.33 for l in ("V1", "V2", "V3", "V4", "V5", "V6")})

        q = assess_quality(canonical, names, needs_full_length=False)

        self.assertTrue(any("Only 6 of 12" in w for w in q["warnings"]))

    def test_a_complete_digitization_says_nothing_about_lead_count(self):
        canonical, names = frame({l: 0.25 for l in CANONICAL_LEADS} | {"II": 1.0})
        q = assess_quality(canonical, names)

        self.assertEqual(q["leads_missing"], [])
        self.assertEqual(q["warnings"], [])

    def test_empty_frame_degrades_without_crashing(self):
        canonical, names = frame({})
        q = assess_quality(canonical, names)

        self.assertTrue(q["degraded"])
        self.assertEqual(q["selected_leads"], [])
        self.assertTrue(q["warnings"])

    def test_select_rhythm_leads_keeps_its_signature(self):
        canonical, names = frame({"II": 1.0, "V1": 1.0})
        self.assertEqual(select_rhythm_leads(canonical, names), ["II", "V1"])


class TestNormalization(unittest.TestCase):
    def test_zscore_is_unit_invariant(self):
        """The digitizer emits microvolts; ECGFounder was trained on millivolts. Only the
        z-score makes that safe. If this ever fails, every diagnosis is off by 1000x.

        Invariance is near-exact rather than exact: the ``+ 1e-8`` guard in the
        denominator does not scale with the signal, so it contributes ~5e-8 relative
        error at millivolt amplitudes and ~5e-11 at microvolt amplitudes. Both are far
        below float32 model-input precision, and the digitizer's larger microvolt
        numbers are the safer side of that.
        """
        mv = np.array([0.1, -0.2, 0.35, 0.0, -0.05])
        uv = mv * 1000.0

        np.testing.assert_allclose(zscore(mv), zscore(uv), rtol=1e-6)

    def test_zscore_matches_ecgfounder_global_formula(self):
        """ECGFounder normalizes with one mean/std over the whole (12, T) array, not
        per lead. Relative amplitudes between leads are diagnostic."""
        x = np.arange(24, dtype=np.float64).reshape(2, 12)
        expected = (x - np.mean(x)) / (np.std(x) + 1e-8)

        np.testing.assert_allclose(zscore(x), expected, rtol=1e-9)

    def test_zscore_survives_a_flat_signal(self):
        self.assertTrue(np.all(np.isfinite(zscore(np.zeros(10)))))


class TestCleanLead(unittest.TestCase):
    def test_trims_to_valid_span_and_interpolates_internal_holes(self):
        x = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan, np.nan])
        out = clean_lead(x)

        # Trailing NaNs are dropped, the internal hole is filled by interpolation.
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_does_not_resample(self):
        x = np.concatenate([np.full(10, np.nan), np.ones(37)])
        self.assertEqual(clean_lead(x).size, 37)

    def test_all_nan_lead_raises(self):
        with self.assertRaises(ValueError):
            clean_lead(np.full(50, np.nan))


if __name__ == "__main__":
    unittest.main()

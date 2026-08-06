"""Tests for representative-beat construction.

The property that matters is phase: after this runs, every lead's beat must sit at the same
index. That is the whole point of the pathway -- the naive montage failed precisely because
it did not hold. Everything runs on synthetic spike trains; no weights, no sample data.
"""

from __future__ import annotations

import unittest

import numpy as np

from ecg_pipeline.interpret.representative_beat import (
    MAX_DESYNC_LAG_SECONDS,
    MIN_RR_SECONDS,
    POST_SECONDS,
    PRE_SECONDS,
    baseline_correct,
    detect_fiducials,
    representative_beat,
    residual_desync_ms,
    tile_to_length,
)
from ecg_pipeline.interpret.waveform import CANONICAL_LEADS, STANDARD_3X4_COLUMNS

FS = 500
N = 5000  # 10 s, as the digitizer emits
PERIOD = 300  # samples between beats: 100 bpm
FIDUCIAL = int(PRE_SECONDS * FS)
WINDOW = FIDUCIAL + int(POST_SECONDS * FS)


def spikes(n: int, period: int = PERIOD, width: float = 6.0, amplitude: float = 1.0) -> np.ndarray:
    """A train of sharp positive peaks on a fixed global grid."""
    t = np.arange(n)
    x = np.zeros(n)
    for centre in range(0, n, period):
        x += amplitude * np.exp(-0.5 * ((t - centre) / width) ** 2)
    return x


def synthetic_3x4(rhythm_leads: tuple[str, ...] = ("II",), amplitudes: dict[str, float] | None = None):
    """A canonical frame shaped like a 3x4 print: each column valid in its own quarter.

    Every column's beats land on one global grid, so leads that end up in phase prove the
    alignment worked rather than that they were never apart.
    """
    amplitudes = amplitudes or {}
    canonical = np.full((len(CANONICAL_LEADS), N), np.nan)
    per_column = N // len(STANDARD_3X4_COLUMNS)
    for column_index, column_leads in enumerate(STANDARD_3X4_COLUMNS):
        start, end = column_index * per_column, (column_index + 1) * per_column
        for lead in column_leads:
            index = CANONICAL_LEADS.index(lead)
            full = spikes(N, amplitude=amplitudes.get(lead, 1.0))
            if lead in rhythm_leads:
                canonical[index] = full
            else:
                canonical[index, start:end] = full[start:end]
    return canonical, list(CANONICAL_LEADS)


class TestDetectFiducials(unittest.TestCase):
    def test_finds_every_beat_of_a_clean_train(self):
        # 10 beats on the grid, 9 detected: a peak sitting exactly on sample 0 has no left
        # neighbour to be a peak against. Harmless -- a beat that close to the edge has no
        # room for its own window and would be dropped anyway.
        self.assertEqual(detect_fiducials(spikes(3000), FS).size, 9)

    def test_finds_beats_when_the_dominant_deflection_is_negative(self):
        # V1-V4 have a dominant negative QRS. The detector orients the signal first, so it
        # locks onto the S wave there -- it must still find one fiducial per beat.
        self.assertEqual(detect_fiducials(-spikes(3000), FS).size, 9)

    def test_two_peaks_closer_than_the_refractory_period_are_one_beat(self):
        x = np.zeros(1000)
        x[400] = 1.0
        x[400 + int(0.5 * MIN_RR_SECONDS * FS)] = 0.9

        self.assertEqual(detect_fiducials(x, FS).size, 1)

    def test_a_flat_lead_yields_no_beats(self):
        self.assertEqual(detect_fiducials(np.zeros(2000), FS).size, 0)


def centred_beat(n: int = WINDOW) -> np.ndarray:
    """One complex in the middle of the window, isoelectric at both ends."""
    t = np.arange(n)
    return np.exp(-0.5 * ((t - n // 2) / 6.0) ** 2)


class TestBaselineCorrect(unittest.TestCase):
    def test_linear_drift_is_removed(self):
        drifting = centred_beat() + np.linspace(0.0, 4.0, WINDOW)

        corrected = baseline_correct(drifting)

        self.assertAlmostEqual(float(corrected[:20].mean()), float(corrected[-20:].mean()), places=6)

    def test_the_beat_itself_survives(self):
        beat = centred_beat()

        corrected = baseline_correct(beat + 7.0)

        self.assertEqual(int(np.argmax(corrected)), int(np.argmax(beat)))


class TestRepresentativeBeat(unittest.TestCase):
    def test_every_lead_ends_up_on_the_same_fiducial(self):
        # The property the whole pathway exists for: 12 leads recorded up to 7.5 s apart,
        # all reading at one instant of the cardiac cycle afterwards.
        beat, meta = representative_beat(*synthetic_3x4(), fs=FS)

        peaks = {lead: int(np.argmax(beat[i])) for i, lead in enumerate(CANONICAL_LEADS)}
        self.assertEqual(set(peaks.values()), {meta["fiducial_index"]})

    def test_the_output_is_shaped_for_the_12_lead_model(self):
        beat, meta = representative_beat(*synthetic_3x4(), fs=FS)

        self.assertEqual(beat.shape, (12, WINDOW))
        self.assertEqual(meta["fiducial_index"], FIDUCIAL)
        self.assertEqual(meta["window_samples"], WINDOW)

    def test_beat_counts_are_reported_per_lead(self):
        # This is the number to quote when asked what the median is a median of.
        _, meta = representative_beat(*synthetic_3x4(), fs=FS)

        grid_counts = {meta["beats_per_lead"][l] for l in ("I", "III", "V1", "V6")}
        self.assertEqual(grid_counts, {3})
        self.assertEqual(meta["leads_without_a_beat"], [])

    def test_a_rhythm_strip_contributes_more_beats_than_a_grid_lead(self):
        # It spans the whole paper, so it has ~10 beats where a column window has 3.
        _, meta = representative_beat(*synthetic_3x4(rhythm_leads=("II",)), fs=FS)

        self.assertGreater(meta["beats_per_lead"]["II"], meta["beats_per_lead"]["I"])

    def test_the_column_fiducial_comes_from_its_strongest_lead(self):
        canonical, names = synthetic_3x4(rhythm_leads=(), amplitudes={"III": 5.0})

        _, meta = representative_beat(canonical, names, fs=FS)

        self.assertEqual(meta["column_reference"][0]["lead"], "III")

    def test_a_weak_lead_is_timed_by_its_column_not_by_itself(self):
        # The reason fiducials are shared within a column: a lead too flat to time on its
        # own still gets a correctly placed beat, because its neighbours are simultaneous.
        canonical, names = synthetic_3x4(rhythm_leads=(), amplitudes={"I": 0.02})

        beat, meta = representative_beat(canonical, names, fs=FS)

        self.assertEqual(meta["beats_per_lead"]["I"], meta["beats_per_lead"]["II"])
        self.assertEqual(int(np.argmax(beat[CANONICAL_LEADS.index("I")])), meta["fiducial_index"])

    def test_an_outlier_beat_is_rejected_by_the_median(self):
        # A median, not a mean: one corrupted beat out of three must not reach the output.
        canonical, names = synthetic_3x4(rhythm_leads=())
        index = CANONICAL_LEADS.index("I")
        canonical[index, 900 - 40 : 900 + 40] += 50.0  # wreck the second beat of column 0

        beat, _ = representative_beat(canonical, names, fs=FS)

        self.assertEqual(int(np.argmax(beat[index])), FIDUCIAL)

    def test_a_lead_with_no_signal_is_reported_rather_than_silently_flat(self):
        canonical, names = synthetic_3x4(rhythm_leads=())
        canonical[CANONICAL_LEADS.index("V6")] = np.nan

        beat, meta = representative_beat(canonical, names, fs=FS)

        self.assertIn("V6", meta["leads_without_a_beat"])
        # Flat, not zero: the frame is z-scored as a whole, which offsets an empty lead off
        # zero without giving it any shape. Same behaviour as the naive montage.
        self.assertEqual(len(np.unique(beat[CANONICAL_LEADS.index("V6")])), 1)

    def test_an_all_nan_frame_does_not_crash(self):
        canonical = np.full((len(CANONICAL_LEADS), N), np.nan)

        beat, meta = representative_beat(canonical, list(CANONICAL_LEADS), fs=FS)

        self.assertEqual(beat.shape, (12, WINDOW))
        self.assertEqual(len(meta["leads_without_a_beat"]), 12)


class TestTileToLength(unittest.TestCase):
    def test_it_reaches_the_requested_length(self):
        beat, _ = representative_beat(*synthetic_3x4(), fs=FS)

        self.assertEqual(tile_to_length(beat, 5000).shape, (12, 5000))

    def test_it_repeats_rather_than_stretches(self):
        beat, _ = representative_beat(*synthetic_3x4(), fs=FS)

        tiled = tile_to_length(beat, WINDOW * 3)

        np.testing.assert_allclose(tiled[:, :WINDOW], tiled[:, WINDOW : 2 * WINDOW], atol=1e-12)

    def test_a_beat_longer_than_the_target_is_truncated(self):
        beat, _ = representative_beat(*synthetic_3x4(), fs=FS)

        self.assertEqual(tile_to_length(beat, 100).shape, (12, 100))


class TestResidualDesync(unittest.TestCase):
    def test_aligned_leads_report_no_offset(self):
        beat, _ = representative_beat(*synthetic_3x4(), fs=FS)

        self.assertEqual(set(residual_desync_ms(beat, FS).values()), {0.0})

    def test_a_shifted_lead_is_measured(self):
        beat, _ = representative_beat(*synthetic_3x4(), fs=FS)
        shift = 15  # samples -> 30 ms at 500 Hz, the offset previously observed in V6
        index = CANONICAL_LEADS.index("V6")
        beat[index] = np.roll(beat[index], shift)

        desync = residual_desync_ms(beat, FS)

        self.assertAlmostEqual(abs(desync["V6"]), 1000.0 * shift / FS, places=1)

    def test_an_inverted_lead_is_not_reported_as_a_timing_error(self):
        # aVR is normally inverted. Polarity is not phase.
        beat, _ = representative_beat(*synthetic_3x4(), fs=FS)
        beat[CANONICAL_LEADS.index("aVR")] *= -1

        self.assertEqual(residual_desync_ms(beat, FS)["aVR"], 0.0)

    def test_a_dominant_late_wave_is_not_reported_as_a_timing_error(self):
        # aVF on the reference normal ECG has a T wave taller than its QRS. Searching every
        # lag, the best match is that T against the other leads' R, and the measurement comes
        # back 272 ms -- a statement about which deflection is biggest, not about timing.
        beat, meta = representative_beat(*synthetic_3x4(), fs=FS)
        index = CANONICAL_LEADS.index("aVF")
        t = np.arange(beat.shape[1])
        beat[index] += 5.0 * np.exp(-0.5 * ((t - (meta["fiducial_index"] + int(0.27 * FS))) / 15.0) ** 2)

        self.assertLessEqual(abs(residual_desync_ms(beat, FS)["aVF"]), 1000 * MAX_DESYNC_LAG_SECONDS)

    def test_a_frame_with_nothing_in_it_reports_nothing(self):
        self.assertEqual(residual_desync_ms(np.zeros((12, WINDOW)), FS), {})


if __name__ == "__main__":
    unittest.main()

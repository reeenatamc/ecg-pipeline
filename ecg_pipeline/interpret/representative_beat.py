"""Build a representative beat: one median complex per lead, all in the same cardiac phase.

The problem it solves
---------------------
A standard 3x4 paper ECG records its four columns at *different times* -- up to 7.5 s apart.
Stacking those columns into a 12-lead frame (``build_12lead_montage``) hands the model
twelve leads whose beats are in unrelated phases, and the model reads that misalignment as
pathology: ``LATERAL INFARCT`` 0.997 on a verified-normal ECG.

The fix is what commercial electrocardiographs (GE's 12SL, the Glasgow algorithm) already
do internally before they measure anything: a **representative complex**. Within one column
the three leads *are* simultaneous, so R peaks found in the strongest lead of a column time
every lead in it. Cut one beat per detected peak, take the **median** across beats per lead,
and assemble the twelve median beats on a common fiducial. All twelve are now in phase.

Measured on the reference normal ECG (naive montage -> aligned beat, tiled to 10 s):

    LATERAL INFARCT   0.997 (false)  ->  out of the ranking
    LOW VOLTAGE QRS   0.988 (false)  ->  out of the ranking
    NORMAL ECG        0.10           ->  0.88
    ABNORMAL ECG      0.999          ->  0.55

What it destroys
----------------
A median beat is by construction perfectly regular, so it **erases rhythm**. On a confirmed
atrial-fibrillation ECG, ``ATRIAL FIBRILLATION`` falls from 0.927 (read through the
``rhythm`` pathway) out of the top ranking, and the model reads ``SINUS RHYTHM`` 0.982 on a
patient in AF. This is not a bug to fix; it is what averaging beats means. Rhythm must keep
coming from a full-length strip -- the ``rhythm`` pathway -- and this one answers morphology
only. ``interpret_ecg`` enforces that split.

Known limitation, worth fixing next
-----------------------------------
The fiducial is each segment's *dominant* deflection, which is the R wave in most leads but
the S wave in V1-V4 (dominant negative deflection there). That mismatch leaves a residual
inter-lead offset -- measured on the reference ECG at 0-10 ms across eleven leads and -22 ms
in V6, against a QRS that lasts 80-100 ms. ``residual_desync_ms`` reports it per record so
it is never a guess. The known fix is to move the fiducial to QRS onset, or to a global one
computed from the vector magnitude of all twelve leads, as commercial devices do.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ecg_pipeline.interpret.waveform import (
    CANONICAL_FS,
    CANONICAL_LEADS,
    STANDARD_3X4_COLUMNS,
    interpolate_internal_nans,
    lead_windows,
    zscore,
)

# Beat window around the fiducial. 0.3 s before covers P wave and PR interval, 0.5 s after
# covers QRS, ST segment and T wave -- the morphology this pathway exists to show.
PRE_SECONDS = 0.30
POST_SECONDS = 0.50

# Physiological refractory floor: two R peaks closer than this are one peak detected twice.
# 0.3 s also caps the detector at 200 bpm.
MIN_RR_SECONDS = 0.30

# Peak prominence in standard deviations of the segment.
PEAK_PROMINENCE = 1.0

# Widest fiducial error worth calling a timing offset. R-to-S within a QRS spans tens of
# milliseconds, so 100 ms covers the realistic mismatch with room to spare; beyond it, a
# cross-correlation is matching a different wave entirely.
MAX_DESYNC_LAG_SECONDS = 0.10

# Above this coverage a lead was printed as a full-length rhythm strip, so it carries its
# own beats (9-12 of them) instead of borrowing its column's ~2.5 s window (2-4).
RHYTHM_COVERAGE_MIN = 0.5


def detect_fiducials(
    segment: npt.NDArray[np.float64],
    fs: int = CANONICAL_FS,
    min_rr_seconds: float = MIN_RR_SECONDS,
    prominence: float = PEAK_PROMINENCE,
) -> npt.NDArray[np.int_]:
    """Sample indices of the dominant deflection of each beat in ``segment``.

    The deflection is oriented upward first, so a lead whose QRS points down (V1-V4) is
    still detected -- on the S wave rather than the R wave. See the module docstring on the
    residual offset this causes.
    """
    from scipy.signal import find_peaks

    x = np.nan_to_num(segment)
    x = x - np.median(x)
    if abs(x.min()) > abs(x.max()):
        x = -x
    z = (x - x.mean()) / (x.std() + 1e-8)
    peaks, _ = find_peaks(z, distance=max(1, int(min_rr_seconds * fs)), prominence=prominence)
    return peaks


def baseline_correct(beat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Remove offset and linear drift using the beat's isoelectric ends.

    Without this, beats cut from a drifting baseline sit at different vertical offsets and
    the median across them smears the ST segment -- the exact region this pathway is read
    for.
    """
    edge = max(5, beat.size // 12)
    left, right = float(np.mean(beat[:edge])), float(np.mean(beat[-edge:]))

    # Each mean describes the middle of its window, not the end of the beat. Anchoring the
    # line there and extrapolating removes a linear drift exactly; drawing it from sample 0
    # to sample -1 instead leaves a residual slope of roughly edge/size (~8% here), tilting
    # the ST segment this pathway is read for.
    left_x = (edge - 1) / 2.0
    right_x = beat.size - 1 - left_x
    slope = (right - left) / (right_x - left_x)
    drift = left + slope * (np.arange(beat.size) - left_x)
    return beat - drift


def representative_beat(
    canonical: npt.NDArray[np.float64],
    names: list[str],
    fs: int = CANONICAL_FS,
    pre_seconds: float = PRE_SECONDS,
    post_seconds: float = POST_SECONDS,
    rhythm_coverage_min: float = RHYTHM_COVERAGE_MIN,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Median beat per lead, all aligned on a common fiducial.

    Returns ``(beat, meta)`` where ``beat`` is ``(12, pre+post)`` z-scored as one frame (so
    inter-lead amplitude ratios survive), and ``meta`` records how many beats each lead
    contributed -- the number to quote when someone asks what the median is a median *of*.
    """
    pre, post = int(pre_seconds * fs), int(post_seconds * fs)
    window = pre + post
    info = lead_windows(canonical, names)
    nominal = canonical.shape[1] // len(STANDARD_3X4_COLUMNS)

    beat = np.zeros((len(CANONICAL_LEADS), window), dtype=np.float64)
    beats_per_lead: dict[str, int] = {}
    column_reference: dict[int, dict[str, Any]] = {}

    for column_index, column_leads in enumerate(STANDARD_3X4_COLUMNS):
        present = [l for l in column_leads if l in names and info.get(l, {}).get("span")]
        if not present:
            continue

        # The column's own time window: the widest span among its grid leads. A rhythm strip
        # printed in this column spans the whole paper and would not describe the column.
        grid = [l for l in present if info[l]["coverage"] <= rhythm_coverage_min]
        if grid:
            start, end = info[max(grid, key=lambda l: info[l]["span"][1] - info[l]["span"][0])]["span"]
        else:
            start, end = column_index * nominal, min((column_index + 1) * nominal, canonical.shape[1]) - 1

        segments = {l: interpolate_internal_nans(canonical[names.index(l)][start : end + 1]) for l in present}
        # Fiducials come from the column's strongest signal and are valid for every lead in
        # it, including the ones too flat to time reliably on their own.
        reference = max(segments, key=lambda l: float(np.ptp(np.nan_to_num(segments[l]))))
        column_peaks = detect_fiducials(segments[reference], fs)
        column_reference[column_index] = {"lead": reference, "beats_detected": int(column_peaks.size)}

        for lead, segment in segments.items():
            if info[lead]["coverage"] > rhythm_coverage_min:
                # A rhythm strip: use its full trace, which holds more beats than the column
                # window and none of that window's edge effects.
                source = np.nan_to_num(interpolate_internal_nans(canonical[names.index(lead)]))
                peaks = detect_fiducials(source, fs)
            else:
                source = np.nan_to_num(segment)
                peaks = column_peaks

            cut = [
                baseline_correct(source[p - pre : p + post]) for p in peaks if p - pre >= 0 and p + post <= source.size
            ]
            beats_per_lead[lead] = len(cut)
            if cut:
                beat[CANONICAL_LEADS.index(lead)] = np.median(np.stack(cut), axis=0)

    meta: dict[str, Any] = {
        "fs": fs,
        "window_samples": window,
        "fiducial_index": pre,
        "beats_per_lead": beats_per_lead,
        "column_reference": column_reference,
        "leads_without_a_beat": [l for l in CANONICAL_LEADS if not beats_per_lead.get(l)],
    }
    return zscore(beat), meta


def tile_to_length(beat: npt.NDArray[np.float64], length: int) -> npt.NDArray[np.float64]:
    """Repeat the representative beat until it fills ``length`` samples.

    ECGFounder was trained on ~10 s records and scores a single 0.8 s beat poorly, so the
    beat is repeated to that duration. The repetition is a shape the model expects, NOT a
    recording: it is perfectly regular by construction and must never be read as rhythm.
    """
    if beat.shape[1] >= length:
        return zscore(beat[:, :length])
    repeats = int(np.ceil(length / beat.shape[1]))
    return zscore(np.tile(beat, (1, repeats))[:, :length])


def residual_desync_ms(
    beat: npt.NDArray[np.float64],
    fs: int = CANONICAL_FS,
    max_lag_seconds: float = MAX_DESYNC_LAG_SECONDS,
) -> dict[str, float]:
    """How far each lead's beat still sits from the others, in milliseconds.

    Alignment is only as good as the fiducial, and the fiducial is the dominant deflection,
    which is the S wave in the leads with a negative QRS. This cross-correlates each lead
    against the mean of the others' normalized beats and reports the lag of the best match.
    A perfect alignment is 0 ms; against a QRS of 80-100 ms, tens of milliseconds are not
    negligible for fine morphology.

    The search is bounded because past a certain lag the best match is no longer the same
    wave: on the reference normal ECG, aVF's T wave is taller than its QRS, and an unbounded
    search aligns that T wave against everyone else's R wave and reports 272 ms. A lag that
    large is a statement about which deflection dominates, not about timing.
    """
    present = [i for i in range(beat.shape[0]) if np.any(beat[i])]
    if len(present) < 2:
        return {}

    normalized = {}
    for i in present:
        x = beat[i] - beat[i].mean()
        # Orient every lead the same way before comparing, or a normally-inverted lead
        # (aVR) reports a half-beat lag that is polarity, not timing.
        if abs(x.min()) > abs(x.max()):
            x = -x
        normalized[i] = x / (np.linalg.norm(x) + 1e-12)

    max_lag = max(1, int(max_lag_seconds * fs))
    out: dict[str, float] = {}
    for i in present:
        others = [normalized[j] for j in present if j != i]
        template = np.mean(np.stack(others), axis=0)
        correlation = np.correlate(normalized[i], template, mode="full")
        zero_lag = normalized[i].size - 1
        searched = correlation[max(0, zero_lag - max_lag) : zero_lag + max_lag + 1]
        lag = int(np.argmax(searched)) - min(max_lag, zero_lag)
        out[CANONICAL_LEADS[i]] = round(1000.0 * lag / fs, 1)
    return out

"""Emit a pipeline result in the shape app-EKG's ``EcgAnalysisService`` consumes.

The app talks to ``api-EKG``, which runs this pipeline behind it. Nothing here knows about
HTTP: this is the translation layer, so that whoever builds that service is mapping fields
rather than deciding what the pipeline meant.

Three conversions are not cosmetic, and getting any of them wrong would put a plausible
lie on a screen:

* **Units.** The digitizer writes microvolts; ``EcgSignal`` is specified in millivolts.
* **Gaps are structural.** A lead is a list of continuous segments, each stamped with the
  second it starts, never a padded array. On a 3x4 print each grid lead exists for 2.5 of
  the 10 seconds; the app's signal model is built the way it is precisely so that drawing a
  line across the other 7.5 takes deliberate effort rather than a missing null check. NaNs
  become absent segments, not zeros.
* **Right-sided leads.** The digitizer places a right-sided print's V4R/V5R/V6R into the
  V4/V5/V6 slots, because it identifies leads by position. The app's ``LeadName`` has the
  R names, so the relabel happens here rather than leaving the app to show a right-sided
  trace under a left-sided name.

What this cannot supply
-----------------------
``EcgMeasurements`` -- rate, PR, QRS, QT, QTc, axis -- comes back ``None``. The pipeline
does not delineate waves, and the contract's type is all-or-nothing, so there is no honest
partial answer: filling PR and QT with anything would be inventing measurements of a
patient. See the README.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import numpy.typing as npt

from ecg_pipeline.interpret.waveform import CANONICAL_FS, load_canonical_csv

MICROVOLTS_TO_MILLIVOLTS = 1e-3

# Millivolts, rounded to 0.1 uV. Finer than any paper ECG resolves, and it keeps a 12-lead
# 10 s record from serialising to several megabytes of full float repr.
SAMPLE_DECIMALS = 4

# The custom right-sided layout in configs/. The digitizer fills the V4/V5/V6 slots by
# position, so on this layout those traces are really the right-sided leads.
RIGHT_SIDED_LAYOUTS = {"limb_aug_right_3x3"}
RIGHT_SIDED_RELABEL = {"V4": "V4R", "V5": "V5R", "V6": "V6R"}


def lead_segments(
    values: npt.NDArray[np.float64], fs: int = CANONICAL_FS, decimals: int = SAMPLE_DECIMALS
) -> list[dict[str, Any]]:
    """Split one lead into its continuous recorded stretches, in millivolts.

    Each NaN run ends a segment: outside a segment there is no datum, and the contract's
    type is built so that this cannot be quietly interpolated later.
    """
    valid = np.flatnonzero(~np.isnan(values))
    if valid.size == 0:
        return []

    segments: list[dict[str, Any]] = []
    for run in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
        start, end = int(run[0]), int(run[-1])
        samples = values[start : end + 1] * MICROVOLTS_TO_MILLIVOLTS
        segments.append(
            {
                "startSecond": round(start / fs, 6),
                "values": [round(float(v), decimals) for v in samples],
            }
        )
    return segments


def to_signal(
    canonical: npt.NDArray[np.float64],
    names: list[str],
    fs: int = CANONICAL_FS,
    lead_layout: str = "",
) -> dict[str, Any]:
    """Build ``EcgSignal`` from a canonical frame."""
    relabel = RIGHT_SIDED_RELABEL if lead_layout in RIGHT_SIDED_LAYOUTS else {}
    leads = []
    for index, name in enumerate(names):
        segments = lead_segments(canonical[index], fs)
        if segments:
            leads.append({"name": relabel.get(name, name), "segments": segments})
    return {
        "samplingRateHz": fs,
        "durationSeconds": round(canonical.shape[1] / fs, 6),
        "leads": leads,
    }


def observation_id(label: str) -> str:
    """A stable slug for a model label: ``LATERAL INFARCT`` -> ``lateral-infarct``."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def observed_leads(result: dict[str, Any]) -> list[str]:
    """Which leads the reading actually rests on, for an observation to point at."""
    if result.get("pathway") == "rhythm":
        return list(result.get("rhythm_leads", []))
    if result.get("pathway") == "1lead":
        return [result["lead"]] if result.get("lead") else []
    return list(result.get("signal_quality", {}).get("leads_with_signal", []))


def to_observations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn the model's ranking into observations.

    Labels pass through as the model produced them. Rewriting ``SINUS RHYTHM`` into a
    phrase for a patient to read is a clinical and product decision, and one taken with a
    cardiologist -- not something to improvise in a serialiser.

    The whole ranking is emitted, not a top slice: the contract is explicit that this is a
    ranking rather than a verdict, and where to cut it is the interface's call.
    """
    leads = observed_leads(result)
    return [
        {
            "id": observation_id(row["label"]),
            "label": row["label"],
            "leads": leads,
            "confidence": row["prob"],
            "needsReview": True,
        }
        for row in result.get("topk", [])
    ]


def trace_incomplete(quality: dict[str, Any] | None) -> bool:
    """Did the digitization come back too fragmented to read, by ``assess_quality``'s rules?

    Mirrors the two conditions under which that function degrades a record: no lead
    carries any signal, or the pathway needs a full-length strip and none was printed. It
    is read off the fields rather than off ``quality["degraded"]`` because the morphology
    and 12lead pathways set that flag for reasons of their own that say nothing about the
    trace.
    """
    if not quality:
        return False
    if not quality.get("leads_with_signal"):
        return True
    return bool(quality.get("needs_full_length")) and not quality.get("full_length_leads")


def failure_reason(result: dict[str, Any]) -> str | None:
    """Map a pipeline failure onto the contract's closed set of causes.

    ``grid-not-detected`` is never produced: when the digitizer cannot find the grid it
    raises per-image and writes nothing at all, which reaches us as an image that produced
    no output and is indistinguishable from any other unreadable one.

    ``trace-incomplete`` is the case the contract lacked for a long time: the image was
    read and a layout was found, but the trace came back too fragmented to interpret. It
    used to land on ``unexpected``, which told the user something had gone wrong on the
    server when what had gone wrong was the photograph.
    """
    if not result.get("degraded"):
        return None
    if not result.get("source_csv"):
        return "unreadable-image"

    # A run that recovered no lead at all failed on the image, not on this service,
    # even when interpretation also raised -- and on an empty trace it always does,
    # with "No usable lead found in canonical CSV". Checking the error first put a
    # blank photograph on server-error, which sends the user to wait for a fix
    # instead of to take the picture again. Measured on a blank image pushed through
    # the whole pipeline. With no signal the error is a consequence, not a fault, so
    # it is skipped and the checks below name the cause.
    no_signal = (result.get("signal_quality") or {}).get("leads_with_signal") == []
    if "error" in result and not no_signal:
        return "server-error"
    layout = (result.get("digitization") or {}).get("lead_layout")
    if layout == "Unknown layout":
        return "unsupported-mount"
    if trace_incomplete(result.get("signal_quality")):
        return "trace-incomplete"
    return "unexpected"


def to_analysis(
    result: dict[str, Any],
    study_id: str,
    completed_at: str | None = None,
    fs: int = CANONICAL_FS,
) -> dict[str, Any]:
    """One pipeline result as an ``EcgAnalysis``.

    A ``degraded`` result comes back ``failed``. The pipeline's whole point is the
    distinction between a reading and a reading that must not be trusted, and the contract
    has no third state: handing a degraded result over as ``ready`` would put observations
    on screen that the pipeline has already said are unsafe to read.

    ``completed_at`` is passed in rather than read from the clock, so that the caller owns
    the timestamp and this stays a pure function.
    """
    failure = failure_reason(result)
    ready = failure is None

    signal = None
    if result.get("source_csv"):
        canonical, names = load_canonical_csv(result["source_csv"])
        layout = (result.get("digitization") or {}).get("lead_layout", "")
        signal = to_signal(canonical, names, fs, layout)

    return {
        "studyId": study_id,
        "status": "ready" if ready else "failed",
        "signal": signal,
        # Never fabricated: the pipeline does not delineate waves, and EcgMeasurements has
        # no partial form. See the module docstring.
        "measurements": None,
        "observations": to_observations(result) if ready else [],
        "failure": failure,
        "completedAt": completed_at if ready else None,
    }

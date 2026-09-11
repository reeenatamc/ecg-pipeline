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


# ``result["gates"]`` (pipeline.py) to the contract's closed set of causes, first match
# wins. Both "unsupported-mount" entries read as the same underlying problem: the layout
# the digitizer chose does not actually describe what is on the print -- either it names
# leads the signal does not have, or it cannot even say which lead its own rhythm strip is.
_GATE_REASONS = [
    ("digitizer-no-output", "unreadable-image"),
    ("interpretation-error", "server-error"),
    ("layout-unknown", "unsupported-mount"),
    ("leads-missing-from-template", "unsupported-mount"),
    ("rhythm-strip-unverified", "unsupported-mount"),
    ("no-signal", "trace-incomplete"),
    ("no-full-length-lead", "trace-incomplete"),
]


def _recovered_no_signal(result: dict[str, Any]) -> bool:
    """True when the record itself says the digitized trace carried no lead at all."""
    return (result.get("signal_quality") or {}).get("leads_with_signal") == []


def failure_reason(result: dict[str, Any]) -> str | None:
    """Map a pipeline failure onto the contract's closed set of causes.

    Reads ``result["gates"]`` when present -- the stable ids pipeline.py's gates attach to
    a degraded record -- rather than re-deriving the cause from ``warnings`` text or from
    fields that were never meant to double as a classification. ``_GATE_REASONS`` is
    checked in order and the first id present in ``result["gates"]`` wins; a record can
    trip more than one gate (an unknown layout also fails lead completeness), and the
    earlier entries are the more specific/severe causes.

    For a result written before ``gates`` existed (older JSON on disk, or a caller that
    built a result dict by hand), the same set of fields this function has always read are
    used instead. ``grid-not-detected`` is never produced either way: when the digitizer
    cannot find the grid it raises per-image and writes nothing at all, which reaches us as
    an image that produced no output and is indistinguishable from any other unreadable one.

    ``trace-incomplete`` is the case the contract lacked for a long time: the image was
    read and a layout was found, but the trace came back too fragmented to interpret (gate
    2's two conditions -- no signal at all, or no full-length lead where the pathway needs
    one). It used to land on ``unexpected``, which told the user something had gone wrong
    on the server when what had gone wrong was the photograph.
    """
    if not result.get("degraded"):
        return None

    if "gates" in result:
        fired = set(result["gates"])
        # A run that recovered no lead at all failed on the image, even when interpretation
        # also raised -- on an empty trace it always does, "No usable lead found in
        # canonical CSV". "interpretation-error" is listed before "no-signal", so without
        # this a blank photograph came back as server-error, which sends the user to wait
        # for a fix instead of to take the picture again. The error is a consequence here,
        # not a fault.
        if "no-signal" in fired or _recovered_no_signal(result):
            fired.discard("interpretation-error")
            fired.add("no-signal")
        for gate_id, reason in _GATE_REASONS:
            if gate_id in fired:
                return reason
        return "unexpected"

    if not result.get("source_csv"):
        return "unreadable-image"

    # Same reasoning as the gates branch above: with no signal recovered, the
    # interpretation error is a consequence, not a fault. Measured on a blank image pushed
    # through the whole pipeline, which used to land on server-error here.
    if "error" in result and not _recovered_no_signal(result):
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

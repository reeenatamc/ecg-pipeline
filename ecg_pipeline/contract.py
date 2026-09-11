"""Emit a pipeline result in the shape app-EKG's ``EcgAnalysisService`` consumes.

The app talks to ``api-EKG``, which runs this pipeline behind it. Nothing here knows about
HTTP: this is the translation layer, so that whoever builds that service is mapping fields
rather than deciding what the pipeline meant.

Three conversions are not cosmetic, and getting any of them wrong would put a plausible
lie on a screen:

* **Units.** The digitizer writes microvolts; ``EcgSignal`` is specified in millivolts.
* **Gaps are structural, but a dropout is not a gap.** A lead is a list of continuous
  segments, each stamped with the second it starts, never a padded array. On a 3x4 print
  each grid lead exists for 2.5 of the 10 seconds; the app's signal model is built the way
  it is precisely so that drawing a line across the other 7.5 takes deliberate effort
  rather than a missing null check. Within a printed stretch, though, the digitizer
  sometimes loses the trace for a handful of milliseconds under a label or a bold grid
  line; ``lead_segments`` bridges those (see ``MAX_BRIDGED_GAP_SECONDS``) and leaves every
  longer hole -- the kind that means the machine never recorded that stretch -- as a real
  split. NaNs that are not bridged become absent segments, not zeros.
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

# Longest internal NaN run ``lead_segments`` will bridge by linear interpolation, in
# seconds. 0.04 s is one small grid square at the paper's standard 25 mm/s sweep speed --
# the unit a reader measures the tracing with -- and it is shorter than any QRS complex, so
# a bridge can never invent or hide a beat. Measured on a real record (study 43be167a):
# V1-V4 arrive split into 2-3 segments by 14-34 ms holes where the digitizer lost the trace
# under a label or a bold grid line, while the real gaps on that same record -- a lead only
# printed for its 2.5 s column, a 1.45 s dropout in a rhythm strip -- are hundreds of
# milliseconds or more. This threshold sits well below the smallest real gap and above the
# largest observed digitization artifact.
MAX_BRIDGED_GAP_SECONDS = 0.04


def _internal_nan_runs(values: npt.NDArray[np.float64]) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of NaN runs with a valid sample on both sides.

    A run touching either edge of the array has nothing to interpolate from -- it is a
    lead not yet started or already ended, not a dropout inside a printed stretch -- so it
    is never a candidate for bridging.
    """
    idx = np.flatnonzero(np.isnan(values))
    if idx.size == 0:
        return []
    runs = []
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        start, end = int(run[0]), int(run[-1])
        if start == 0 or end == values.size - 1:
            continue
        runs.append((start, end))
    return runs


def bridged_holes(values: npt.NDArray[np.float64], fs: int = CANONICAL_FS) -> list[float]:
    """Durations, in ms, of the internal NaN runs ``lead_segments`` bridges for this lead.

    Mirrors the same ``MAX_BRIDGED_GAP_SECONDS`` threshold rather than re-deriving it, so a
    caller (``signal_bridging_report``) can report exactly what was interpolated.
    """
    max_samples = MAX_BRIDGED_GAP_SECONDS * fs
    return [
        round(1000 * (end - start + 1) / fs, 3)
        for start, end in _internal_nan_runs(values)
        if (end - start + 1) <= max_samples
    ]


def _bridge_dropouts(values: npt.NDArray[np.float64], fs: int) -> npt.NDArray[np.float64]:
    """Linearly interpolate internal NaN runs no longer than ``MAX_BRIDGED_GAP_SECONDS``.

    Everything longer is left as NaN for ``lead_segments`` to split on: this only touches
    the digitization artifact, never the real "lead wasn't printed here" gap. The model
    path already does the analogous thing for its own purposes (``waveform.clean_lead``);
    this is the contract's own pass because it must also report what it bridged.
    """
    bridged = values.astype(np.float64).copy()
    max_samples = MAX_BRIDGED_GAP_SECONDS * fs
    for start, end in _internal_nan_runs(values):
        if (end - start + 1) <= max_samples:
            bridged[start : end + 1] = np.interp(
                np.arange(start, end + 1), [start - 1, end + 1], [values[start - 1], values[end + 1]]
            )
    return bridged


def lead_segments(
    values: npt.NDArray[np.float64], fs: int = CANONICAL_FS, decimals: int = SAMPLE_DECIMALS
) -> list[dict[str, Any]]:
    """Split one lead into its continuous recorded stretches, in millivolts.

    Short internal dropouts (see ``MAX_BRIDGED_GAP_SECONDS``) are bridged first. What
    remains is a real gap: outside a segment there is no datum, and the contract's type is
    built so that this cannot be quietly interpolated later.
    """
    bridged = _bridge_dropouts(values, fs)
    valid = np.flatnonzero(~np.isnan(bridged))
    if valid.size == 0:
        return []

    segments: list[dict[str, Any]] = []
    for run in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
        start, end = int(run[0]), int(run[-1])
        samples = bridged[start : end + 1] * MICROVOLTS_TO_MILLIVOLTS
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
    """Build ``EcgSignal`` from a canonical frame.

    Shape is exact and stable (``samplingRateHz``, ``durationSeconds``, ``leads``): app-EKG
    parses it strictly. Bridged-dropout counts are reported separately, through
    ``signal_bridging_report``, rather than grown onto this dict.
    """
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


def signal_from_csv(csv_path: str, lead_layout: str = "", fs: int = CANONICAL_FS) -> dict[str, Any]:
    """Build ``EcgSignal`` straight from a digitized canonical CSV.

    The entry point a backend calls right after digitization, before any interpretation:
    it is exactly what ``to_analysis`` builds internally for its ``signal`` field, exposed
    on its own so a caller that only needs the trace -- to show it to a user while
    interpretation is still running, say -- does not have to reach into
    ``load_canonical_csv`` and ``to_signal`` itself.
    """
    canonical, names = load_canonical_csv(csv_path)
    return to_signal(canonical, names, fs, lead_layout)


def signal_bridging_report(
    canonical: npt.NDArray[np.float64], names: list[str], fs: int = CANONICAL_FS
) -> dict[str, list[float]]:
    """Per lead, the durations (ms) of the internal dropouts ``to_signal`` bridged.

    ``to_signal``'s own return shape is fixed and app-EKG parses it strictly, so this is a
    side channel: a caller such as api-EKG's diagnostics can store it without the app ever
    seeing it. Only leads with at least one bridged hole are present.
    """
    report = {}
    for index, name in enumerate(names):
        holes = bridged_holes(canonical[index], fs)
        if holes:
            report[name] = holes
    return report


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
    ranking rather than a verdict, and where to cut it is the interface's call. When
    ``interpret_csv`` applied per-class thresholds (``result["flagged"]`` present), the
    classes that cleared theirs come first with ``aboveThreshold: true``, followed by the
    rest of the top-k with ``aboveThreshold: false`` -- so the app can choose to show only
    the flagged findings or the full ranking without a second request. When no thresholds
    applied, every row is the same top-k as always with ``aboveThreshold: null``: that is
    "no verdict was computed", not "found absent".
    """
    leads = observed_leads(result)

    def observation(row: dict[str, Any], above_threshold: bool | None) -> dict[str, Any]:
        return {
            "id": observation_id(row["label"]),
            "label": row["label"],
            "leads": leads,
            "confidence": row["prob"],
            "needsReview": True,
            "aboveThreshold": above_threshold,
        }

    if "flagged" in result:
        flagged_labels = {row["label"] for row in result["flagged"]}
        observations = [observation(row, True) for row in result["flagged"]]
        observations += [
            observation(row, False) for row in result.get("topk", []) if row["label"] not in flagged_labels
        ]
        return observations

    return [observation(row, None) for row in result.get("topk", [])]


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
        for gate_id, reason in _GATE_REASONS:
            if gate_id in fired:
                return reason
        return "unexpected"

    if not result.get("source_csv"):
        return "unreadable-image"
    if "error" in result:
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
        layout = (result.get("digitization") or {}).get("lead_layout", "")
        signal = signal_from_csv(result["source_csv"], layout, fs)

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

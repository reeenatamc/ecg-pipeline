"""Pure signal primitives shared by the interpretation pathways.

Nothing here loads a model, reads a file, or makes a clinical judgement -- it is the layer
both ``interpret_ecg`` and ``representative_beat`` build on, which is why it exists as its
own module rather than living in either of them.

Two conventions the digitizer sets and everything downstream inherits: samples a lead did
not have (because that lead was only printed for part of the paper's width) are NaN, and
the canonical frame is ``(n_leads, n_samples)`` in ``CANONICAL_LEADS`` order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# Canonical lead order emitted by the digitizer (matches ECGFounder's expected order).
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Leads most commonly printed as full-length rhythm strips, in preference order.
PREFERRED_RHYTHM_LEADS = ["II", "V1", "V5"]

# Standard 3x4 paper layout: the four time columns and the leads printed in each. The three
# leads within one column ARE simultaneous; different columns are not.
STANDARD_3X4_COLUMNS = [
    ["I", "II", "III"],
    ["aVR", "aVL", "aVF"],
    ["V1", "V2", "V3"],
    ["V4", "V5", "V6"],
]

# The digitizer resamples every record to 5000 samples over the standard 10 s of paper.
CANONICAL_FS = 500


def interpolate_internal_nans(segment: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Linearly interpolate NaNs that sit between valid samples."""
    segment = segment.astype(np.float64).copy()
    holes = np.isnan(segment)
    if holes.any() and not holes.all():
        index = np.arange(segment.size)
        segment[holes] = np.interp(index[holes], index[~holes], segment[~holes])
    return segment


def resample_to(x: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    """Linear resample a 1-D array to ``n`` samples (matches ECGFounder resample_unequal)."""
    if x.size == n or x.size == 0:
        return x
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, x.size), x)


def zscore(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ECGFounder's normalization: global mean/std over the whole array.

    For 1 lead this is per-lead; for a 12-lead frame it preserves the relative amplitudes
    between leads (diagnostically meaningful). Units cancel either way.
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def extract_lead(canonical: npt.NDArray[np.float64], names: list[str], lead: str) -> npt.NDArray[np.float64]:
    if lead not in names:
        raise ValueError(f"Lead {lead!r} not found. Available: {names}")
    return canonical[names.index(lead)].astype(np.float64)


def clean_lead(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Trim to the valid ``[first..last]`` span and interpolate internal NaNs.

    Does NOT resample/stretch: the lead's native time base (and heart rate) is kept.
    """
    valid = np.where(~np.isnan(x))[0]
    if valid.size == 0:
        raise ValueError("Lead is entirely NaN - nothing to interpret.")
    return interpolate_internal_nans(x[valid[0] : valid[-1] + 1])


def load_canonical_csv(path: str) -> tuple[npt.NDArray[np.float64], list[str]]:
    """Read the digitizer's ``*_timeseries_canonical.csv`` (rows=time, cols=leads).

    Returns (signal, lead_names) with signal shaped (n_leads, n_samples). Missing
    samples (leads outside their printed window) are NaN, as written by the digitizer.
    """
    with open(path, "r") as fin:
        header = fin.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[:, None]
    return data.T, header


def lead_windows(canonical: npt.NDArray[np.float64], names: list[str]) -> dict[str, dict[str, Any]]:
    """Per-lead valid span + coverage fraction."""
    info: dict[str, dict[str, Any]] = {}
    n = canonical.shape[1]
    for lead in CANONICAL_LEADS:
        if lead not in names:
            continue
        x = canonical[names.index(lead)]
        valid = np.where(~np.isnan(x))[0]
        info[lead] = {
            "coverage": float(valid.size / n) if n else 0.0,
            "span": (int(valid[0]), int(valid[-1])) if valid.size else None,
        }
    return info


def assess_quality(
    canonical: npt.NDArray[np.float64],
    names: list[str],
    coverage_min: float = 0.6,
    needs_full_length: bool = True,
) -> dict[str, Any]:
    """Single source of truth for which leads are usable and how much to trust them.

    Lives here, next to the primitives, because the pipeline needs it for ``--digitize-only``
    runs that must not pay for importing torch.

    A full-length lead is what makes a rhythm reading meaningful. When none was printed, the
    caller still gets an answer -- built from the single best-covered lead, which on a 3x4
    layout is a ~2.5 s fragment. That is a materially weaker basis, so it is reported as
    ``degraded`` rather than silently passed off as a rhythm-strip ensemble.

    ``needs_full_length=False`` for a pathway that does not read a continuous strip. The
    representative beat is built from the ~2.5 s column windows by design, so telling it off
    for having no full-length lead describes a requirement it never had -- and on a 3x3 print
    with no rhythm strip at all it would degrade a perfectly good morphology reading. Lead
    completeness still applies: that one is about whether the digitization is trustworthy at
    all, which every pathway needs.
    """
    info = lead_windows(canonical, names)
    full = [l for l in CANONICAL_LEADS if l in info and info[l]["coverage"] >= coverage_min]
    ordered = [l for l in PREFERRED_RHYTHM_LEADS if l in full] + [l for l in full if l not in PREFERRED_RHYTHM_LEADS]

    # A lead with no samples at all is not a fallback candidate: selecting it would only
    # push the failure downstream into clean_lead as an "entirely NaN" error.
    with_signal = {l: v for l, v in info.items() if v["coverage"] > 0}

    warnings: list[str] = []
    degraded = False
    if ordered:
        selected = ordered
    elif with_signal:
        selected = [max(with_signal, key=lambda l: with_signal[l]["coverage"])]
        if needs_full_length:
            degraded = True
            pct = 100 * info[selected[0]]["coverage"]
            warnings.append(
                f"No lead reached {100 * coverage_min:.0f}% coverage, so no lead was printed at "
                f"full length; the best-covered one is {selected[0]} at {pct:.0f}%. Anything read "
                f"from a single ~2.5 s fragment -- rhythm and rate above all -- is unreliable."
            )
    else:
        selected = []
        degraded = True
        warnings.append("No leads were recovered from the digitized signal.")

    usable = list(with_signal)
    # Any lead short of the full twelve, not just a handful. A digitizer that matched the
    # wrong layout still returns a complete-looking result for the leads that layout has:
    # the right-sided ECG matched a 6-lead `precordial_3x2` at a cost indistinguishable from
    # a correct match, and only the missing six give it away. Naming them is the whole point
    # -- "9 of 12" on a 3x3 print is expected, "6 of 12" on a 3x4 is a wrong layout.
    missing = [l for l in CANONICAL_LEADS if l not in with_signal]
    if info and missing:
        warnings.append(
            f"Only {len(usable)} of 12 leads carry any signal ({', '.join(usable) or 'none'}); "
            f"missing {', '.join(missing)}. Either the print holds fewer than 12 leads, or the "
            f"digitizer matched the wrong layout -- check `lead_layout`, and pass --lead-layout "
            f"if the print is a non-standard montage."
        )

    return {
        "coverage": {l: round(info[l]["coverage"], 3) for l in info},
        "leads_with_signal": usable,
        "leads_missing": missing,
        "full_length_leads": ordered,
        "selected_leads": selected,
        "coverage_min": coverage_min,
        "needs_full_length": needs_full_length,
        "degraded": degraded,
        "warnings": warnings,
    }


def select_rhythm_leads(canonical: npt.NDArray[np.float64], names: list[str], coverage_min: float = 0.6) -> list[str]:
    """Full-length leads usable as rhythm strips, preferred order; else best-covered single lead.

    Thin wrapper over ``assess_quality``; use that directly when you need to know whether
    the selection was degraded.
    """
    return list(assess_quality(canonical, names, coverage_min)["selected_leads"])

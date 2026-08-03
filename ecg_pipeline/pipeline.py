"""End-to-end pipeline: ECG image -> digitized time series -> interpretation.

This is the orchestration layer that owns both stages. Stage 1 shells out to
Open-ECG-Digitizer (see ``digitizer.py`` for the licensing boundary); stage 2 runs
ECGFounder in-process via ``ecg_pipeline.interpret``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ecg_pipeline import digitizer
from ecg_pipeline.digitizer import CANONICAL_SUFFIX

INTERPRETATION_SUFFIX = "_interpretation.json"
METADATA_FILENAME = "digitization_metadata.csv"

# The digitizer writes this literal string when no layout matched at all. In that case
# it also hardcodes matching_cost to 1.0 -- the cost is a sentinel, not a measurement --
# and canonicalization returns an all-NaN frame, so whatever signal survives was
# recovered by rhythm-strip cosine matching and its lead identities are guesses.
UNKNOWN_LAYOUT = "Unknown layout"


def read_digitization_metadata(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Parse the digitizer's per-image quality metadata, keyed by record name.

    The digitizer appends to this file and only writes a header when it does not exist,
    so a reused output directory can hold stale rows. Later rows win.
    """
    path = Path(output_dir) / METADATA_FILENAME
    if not path.is_file():
        return {}

    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("file_path") or "").strip()
            if not name:
                continue
            try:
                cost = float(row.get("matching_cost", "nan"))
            except ValueError:
                cost = float("nan")
            out[name] = {
                "matching_cost": cost,
                "is_flipped": (row.get("is_flipped") or "").strip().lower() == "true",
                "lead_layout": (row.get("lead_layout") or "").strip(),
            }
    return out


def digitization_warnings(meta: dict[str, Any] | None, max_matching_cost: float | None = None) -> list[str]:
    """Warnings derived from the digitizer's own assessment of the image."""
    if not meta:
        return ["No digitization metadata was found; digitization quality is unverified."]

    warnings: list[str] = []
    layout = meta.get("lead_layout", "")
    cost = meta.get("matching_cost", float("nan"))

    if layout == UNKNOWN_LAYOUT:
        warnings.append(
            "The digitizer could not identify the lead layout. Which trace belongs to "
            "which lead is unreliable, so any diagnosis derived from it is unsafe. "
            "Re-scan the ECG or supply a matching layout via --lead-layout."
        )
    elif not layout:
        warnings.append("The digitizer reported no lead layout.")

    # Only checked when the caller supplies a threshold: matching_cost is an unbounded
    # residual (mean grid distance x scaling factor), not a normalized score, so there is
    # no defensible universal cutoff. Calibrate one on your own data.
    if (
        max_matching_cost is not None
        and layout != UNKNOWN_LAYOUT
        and not math.isnan(cost)
        and cost > max_matching_cost
    ):
        warnings.append(
            f"Layout match cost {cost:.3f} exceeds the configured limit {max_matching_cost:.3f}; "
            f"the layout fit is poor."
        )

    if meta.get("is_flipped"):
        warnings.append("The image was detected as flipped and was corrected; verify lead polarity.")

    return warnings


def run(
    image_dir: str | Path,
    output_dir: str | Path,
    config: str | Path = digitizer.DEFAULT_CONFIG,
    lead_layout: str | Path | None = None,
    pathway: str = "rhythm",
    lead: str = "II",
    top_k: int = 10,
    device: str = "cpu",
    thresholds: dict[str, float] | None = None,
    flat_threshold: float | None = None,
    max_matching_cost: float | None = None,
    skip_interpretation: bool = False,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Digitize every image in ``image_dir``, then interpret each result.

    Returns one result dict per successfully digitized ECG. Interpretation failures are
    captured per-record rather than aborting the batch, so one bad trace does not cost
    you the whole run.

    Each result carries ``digitization`` (the digitizer's own layout-match metadata),
    ``warnings``, and ``degraded``. A confident-looking probability on an ECG whose
    layout was never identified is the failure mode this guards against: read
    ``degraded`` before reading ``topk``.
    """
    csv_paths = digitizer.digitize(
        image_dir=image_dir,
        output_dir=output_dir,
        config=config,
        overrides=digitizer.lead_layout_override(lead_layout) if lead_layout else None,
        quiet=quiet,
    )
    if not csv_paths:
        return []

    metadata = read_digitization_metadata(output_dir)

    if skip_interpretation:
        return [
            {
                "source_csv": str(p),
                "digitization": metadata.get(Path(str(p)[: -len(CANONICAL_SUFFIX)]).name),
            }
            for p in csv_paths
        ]

    # Imported lazily: loading torch + the ECGFounder checkpoint is expensive and
    # pointless for digitization-only runs.
    from ecg_pipeline.interpret.interpret_ecg import interpret_csv

    results: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        record = str(csv_path)[: -len(CANONICAL_SUFFIX)]
        name = Path(record).name
        meta = metadata.get(name)
        dig_warnings = digitization_warnings(meta, max_matching_cost)

        try:
            result = interpret_csv(
                str(csv_path),
                pathway=pathway,
                lead=lead,
                k=top_k,
                device=device,
                thresholds=thresholds,
                flat_threshold=flat_threshold,
            )
        except Exception as exc:
            result = {"source_csv": str(csv_path), "error": f"{type(exc).__name__}: {exc}"}

        # Digitization problems come first: they invalidate everything downstream.
        result["digitization"] = meta
        result["warnings"] = dig_warnings + list(result.get("warnings", []))
        layout_failed = bool(meta and meta.get("lead_layout") == UNKNOWN_LAYOUT)
        result["degraded"] = bool(result.get("degraded")) or layout_failed or "error" in result

        out_path = Path(record + INTERPRETATION_SUFFIX)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        result["interpretation_path"] = str(out_path)
        results.append(result)

        if not quiet:
            flag = "  [DEGRADED]" if result["degraded"] else ""
            if "error" in result:
                print(f"  {name}: interpretation failed - {result['error']}")
            else:
                top = ", ".join(f"{r['label']} {r['prob']:.2f}" for r in result["topk"][:3])
                print(f"  {name} [{pathway}]{flag}: {top}")
            for w in result["warnings"]:
                print(f"      ! {w}")

    return results

"""End-to-end pipeline: ECG image -> digitized time series -> interpretation.

This is the orchestration layer that owns both stages. Stage 1 shells out to
Open-ECG-Digitizer (see ``digitizer.py`` for the licensing boundary); stage 2 runs
ECGFounder in-process via ``ecg_pipeline.interpret``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ecg_pipeline import digitizer
from ecg_pipeline.digitizer import CANONICAL_SUFFIX

INTERPRETATION_SUFFIX = "_interpretation.json"


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
    skip_interpretation: bool = False,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Digitize every image in ``image_dir``, then interpret each result.

    Returns one result dict per successfully digitized ECG. Interpretation failures are
    captured per-record rather than aborting the batch, so one bad trace does not cost
    you the whole run.
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

    if skip_interpretation:
        return [{"source_csv": str(p)} for p in csv_paths]

    # Imported lazily: loading torch + the ECGFounder checkpoint is expensive and
    # pointless for digitization-only runs.
    from ecg_pipeline.interpret.interpret_ecg import interpret_csv

    results: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        record = str(csv_path)[: -len(CANONICAL_SUFFIX)]
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

        out_path = Path(record + INTERPRETATION_SUFFIX)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        result["interpretation_path"] = str(out_path)
        results.append(result)

        if not quiet:
            name = Path(record).name
            if "error" in result:
                print(f"  {name}: interpretation failed - {result['error']}")
            else:
                top = ", ".join(f"{r['label']} {r['prob']:.2f}" for r in result["topk"][:3])
                print(f"  {name} [{pathway}]: {top}")

    return results

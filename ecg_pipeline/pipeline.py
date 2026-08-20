"""End-to-end pipeline: ECG image -> digitized time series -> interpretation.

This is the orchestration layer that owns both stages. Stage 1 shells out to
Open-ECG-Digitizer (see ``digitizer.py`` for the licensing boundary); stage 2 runs
ECGFounder in-process via ``ecg_pipeline.interpret``.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from ecg_pipeline import digitizer, preprocess
from ecg_pipeline.digitizer import CANONICAL_SUFFIX

INTERPRETATION_SUFFIX = "_interpretation.json"
METADATA_FILENAME = "digitization_metadata.csv"

# The digitizer writes this literal string when no layout matched at all. In that case
# it also hardcodes matching_cost to 1.0 -- the cost is a sentinel, not a measurement --
# and canonicalization returns an all-NaN frame, so whatever signal survives was
# recovered by rhythm-strip cosine matching and its lead identities are guesses.
UNKNOWN_LAYOUT = "Unknown layout"


def read_digitization_metadata(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Parse the digitizer's per-image quality metadata, keyed by record.

    The digitizer writes one of these files *per output subdirectory*, each row naming only
    a basename, so a nested batch produces several. They are all read and keyed by the
    record's path relative to ``output_dir`` -- which is also what two images sharing a
    basename in different subdirectories need in order not to overwrite each other here.

    The digitizer appends and only writes a header when the file does not exist, so a
    reused output directory can hold stale rows. Later rows win.
    """
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        return {}

    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob(METADATA_FILENAME)):
        prefix = path.parent.relative_to(root)
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("file_path") or "").strip()
                if not name:
                    continue
                try:
                    cost = float(row.get("matching_cost", "nan"))
                except ValueError:
                    cost = float("nan")
                out[(prefix / name).as_posix()] = {
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
    if max_matching_cost is not None and layout != UNKNOWN_LAYOUT and not math.isnan(cost) and cost > max_matching_cost:
        warnings.append(
            f"Layout match cost {cost:.3f} exceeds the configured limit {max_matching_cost:.3f}; "
            f"the layout fit is poor."
        )

    if meta.get("is_flipped"):
        warnings.append("The image was detected as flipped and was corrected; verify lead polarity.")

    return warnings


def _report_preprocessing(prepared: dict[str, dict[str, Any]]) -> None:
    """Say what the image was changed to. An upscale should never be invisible."""
    upscaled = {name: rec for name, rec in prepared.items() if rec["upscale_factor"] > 1}
    if not upscaled:
        return
    print(f"Upscaled {len(upscaled)} of {len(prepared)} image(s) before digitization:")
    for name, rec in upscaled.items():
        (width, height), (new_width, new_height) = rec["original_size"], rec["size"]
        print(f"  {name}: {width}x{height} -> {new_width}x{new_height}  (x{rec['upscale_factor']}, Lanczos)")


def record_name(csv_path: str | Path, output_root: Path) -> str:
    """The key everything joins on: the record's path under the output root, no suffix.

    A bare basename would collapse ``batch-1/ecg`` and ``batch-2/ecg`` onto each other, and
    the digitizer mirrors the input tree rather than flattening it.
    """
    # Resolved on both sides before comparing: on macOS the output directory resolves
    # /var -> /private/var, and an unresolved path would fail to look relative to it and
    # silently degrade to a bare basename -- which is exactly the collapse this avoids.
    stem = Path(str(csv_path)[: -len(CANONICAL_SUFFIX)]).expanduser().resolve()
    try:
        return stem.relative_to(output_root).as_posix()
    except ValueError:
        return stem.name


def clear_previous_outputs(output_root: Path, records: list[str]) -> None:
    """Delete this pipeline's own artifacts for the records about to be rewritten.

    The digitizer offers to do this itself, but ``clear_output_dir_if_exists`` empties the
    whole output directory -- every file in it, ours or not. This removes the three files a
    record produces and nothing else.

    It matters beyond tidiness: a record that succeeded last run and fails this one would
    otherwise leave its old CSV in place, and the run would read a stale digitization as if
    it were fresh.
    """
    for name in records:
        for suffix in (CANONICAL_SUFFIX, INTERPRETATION_SUFFIX, ".png"):
            stale = output_root / f"{name}{suffix}"
            if stale.is_file():
                stale.unlink()


def _layout_failed(meta: dict[str, Any] | None) -> bool:
    return bool(meta and meta.get("lead_layout") == UNKNOWN_LAYOUT)


def _record_warnings(
    meta: dict[str, Any] | None, prep: dict[str, Any] | None, max_matching_cost: float | None
) -> list[str]:
    """Everything wrong with the image and its digitization, before any interpretation."""
    return preprocess.preprocessing_warnings(prep) + digitization_warnings(meta, max_matching_cost)


def _report_record(name: str, result: dict[str, Any], pathway: str | None) -> None:
    """One line per record, then its warnings. ``pathway`` is None for digitize-only runs."""
    flag = "  [DEGRADED]" if result.get("degraded") else ""
    if "error" in result:
        print(f"  {name}: {result['error']}")
    elif "topk" in result:
        top = ", ".join(f"{r['label']} {r['prob']:.2f}" for r in result["topk"][:3])
        print(f"  {name} [{pathway}]{flag}: {top}")
    else:
        leads = result.get("signal_quality", {}).get("leads_with_signal", [])
        print(f"  {name}: digitized, {len(leads)}/12 leads{flag}")
    for warning in result.get("warnings", []):
        print(f"      ! {warning}")


def missing_record_results(prepared: dict[str, dict[str, Any]], recovered: set[str]) -> list[dict[str, Any]]:
    """One failed result per image the digitizer never produced output for.

    The digitizer catches its own per-image exceptions, prints them, and exits 0, so an
    image that blows up simply does not appear among the CSVs. Without this, a batch of ten
    that lost three comes back as a clean run of seven and nobody is told. Reporting them as
    degraded results keeps the count honest and lets --fail-on-degraded catch them.
    """
    return [
        {
            "record": name,
            "source_image": record["source"],
            "preprocessing": record,
            "error": "The digitizer produced no output for this image; it was skipped.",
            "degraded": True,
            "warnings": preprocess.preprocessing_warnings(record)
            + [
                "Digitization failed for this image. The digitizer reports per-image failures "
                "on its own output; re-run without --quiet to see why."
            ],
        }
        for name, record in prepared.items()
        if name not in recovered
    ]


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
    upscale: str | int = preprocess.AUTO,
    skip_interpretation: bool = False,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Digitize every image in ``image_dir``, then interpret each result.

    Returns one result dict per successfully digitized ECG. Interpretation failures are
    captured per-record rather than aborting the batch, so one bad trace does not cost
    you the whole run.

    Each result carries ``digitization`` (the digitizer's own layout-match metadata),
    ``preprocessing`` (what was done to the image first), ``warnings``, and ``degraded``.
    A confident-looking probability on an ECG whose layout was never identified is the
    failure mode this guards against: read ``degraded`` before reading ``topk``.
    """
    # The digitizer reads a directory, so the upscaled images are staged into a scratch
    # one. They are inputs, not results: keeping them would leave a second copy of every
    # ECG image on disk next to the outputs.
    output_root = Path(output_dir).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="ecg-pipeline-") as staged:
        prepared = preprocess.prepare_images(image_dir, staged, upscale=upscale)
        if not quiet:
            _report_preprocessing(prepared)
        clear_previous_outputs(output_root, list(prepared))
        csv_paths = digitizer.digitize(
            image_dir=staged,
            output_dir=output_dir,
            config=config,
            overrides=digitizer.lead_layout_override(lead_layout) if lead_layout else None,
            quiet=quiet,
        )

    # The digitizer no longer empties the output directory, so what it wrote has to be told
    # apart from what an earlier run left there. Only records submitted this time count.
    csv_paths = [p for p in csv_paths if record_name(p, output_root) in prepared]
    metadata = read_digitization_metadata(output_root)
    recovered = {record_name(p, output_root) for p in csv_paths}

    if skip_interpretation:
        # The gates run here too. They used to not, which made --digitize-only the one way
        # to get output past --fail-on-degraded: with no ``degraded`` key on any record, the
        # CLI found nothing to fail on however badly the digitization had gone.
        from ecg_pipeline.interpret.waveform import assess_quality, load_canonical_csv

        results = []
        for csv_path in csv_paths:
            name = record_name(csv_path, output_root)
            meta, prep = metadata.get(name), prepared.get(name)
            try:
                quality = assess_quality(*load_canonical_csv(str(csv_path)))
            except Exception as exc:  # a CSV too malformed to read is itself the finding
                quality = {"degraded": True, "warnings": [f"Could not read the digitized CSV: {exc}"]}
            result = {
                "record": name,
                "source_csv": str(csv_path),
                "digitization": meta,
                "preprocessing": prep,
                "signal_quality": quality,
            }
            result["warnings"] = _record_warnings(meta, prep, max_matching_cost) + list(quality["warnings"])
            result["degraded"] = bool(quality["degraded"]) or _layout_failed(meta)
            results.append(result)
            if not quiet:
                _report_record(name, result, pathway=None)
        missing = missing_record_results(prepared, recovered)
        if not quiet:
            for failed in missing:
                _report_record(failed["record"], failed, pathway=None)
        return results + missing

    # Imported lazily: loading torch + the ECGFounder checkpoint is expensive and
    # pointless for digitization-only runs.
    from ecg_pipeline.interpret.interpret_ecg import build_model_for, interpret_csv

    # Loaded once for the whole batch rather than inside interpret_csv per record: the
    # checkpoint is 370 MB and takes over a second to read. Skipped entirely when the
    # digitizer produced nothing.
    model, ckpt = build_model_for(pathway, device=device) if csv_paths else (None, None)

    results: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        record = str(csv_path)[: -len(CANONICAL_SUFFIX)]
        name = record_name(csv_path, output_root)
        meta = metadata.get(name)
        prep = prepared.get(name)
        dig_warnings = _record_warnings(meta, prep, max_matching_cost)

        try:
            result = interpret_csv(
                str(csv_path),
                pathway=pathway,
                lead=lead,
                ckpt=ckpt,
                k=top_k,
                device=device,
                thresholds=thresholds,
                flat_threshold=flat_threshold,
                model=model,
            )
        except Exception as exc:
            result = {"source_csv": str(csv_path), "error": f"{type(exc).__name__}: {exc}"}

        # Digitization problems come first: they invalidate everything downstream.
        result["record"] = name
        result["digitization"] = meta
        result["preprocessing"] = prep
        result["warnings"] = dig_warnings + list(result.get("warnings", []))
        result["degraded"] = bool(result.get("degraded")) or _layout_failed(meta) or "error" in result

        out_path = Path(record + INTERPRETATION_SUFFIX)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        result["interpretation_path"] = str(out_path)
        results.append(result)

        if not quiet:
            _report_record(name, result, pathway)

    missing = missing_record_results(prepared, recovered)
    if not quiet:
        for failed in missing:
            _report_record(failed["record"], failed, pathway)

    return results + missing

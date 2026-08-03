"""Command-line entry point for the ECG pipeline.

    python -m ecg_pipeline --images path/to/images --out path/to/output
    python -m ecg_pipeline --images in/ --out out/ --pathway 1lead --lead II
    python -m ecg_pipeline --images in/ --out out/ --digitize-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecg_pipeline import digitizer, pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecg_pipeline",
        description="Digitize ECG images and interpret them with ECGFounder.",
    )
    parser.add_argument("--images", required=True, help="Directory of ECG images to process.")
    parser.add_argument("--out", required=True, help="Directory for digitized CSVs and interpretations.")
    parser.add_argument(
        "--config",
        default=str(digitizer.DEFAULT_CONFIG),
        help="Digitizer config (default: configs/digitizer_cpu.yml).",
    )
    parser.add_argument(
        "--lead-layout",
        default=None,
        help="Lead-layout file from configs/ (e.g. lead_layouts_limbaug_right.yml for "
        "right-sided limb leads). Defaults to the digitizer's full layout set.",
    )
    parser.add_argument(
        "--pathway",
        default="rhythm",
        choices=["rhythm", "1lead", "12lead"],
        help="Interpretation pathway. 'rhythm' (default) is the trustworthy one; "
        "'12lead' is experimental and over-calls pathology on 3x4 layouts.",
    )
    parser.add_argument("--lead", default="II", help="Lead to use when --pathway 1lead.")
    parser.add_argument("--top-k", type=int, default=10, help="How many diagnoses to report.")
    parser.add_argument("--device", default="cpu", help="Torch device for interpretation.")
    parser.add_argument("--threshold", type=float, default=None, help="Flat threshold for the flagged list.")
    parser.add_argument("--thresholds", default=None, help="JSON file of per-class thresholds.")
    parser.add_argument(
        "--max-matching-cost",
        type=float,
        default=None,
        help="Warn when the digitizer's layout match cost exceeds this. Off by default: the "
        "cost is an unbounded residual, not a normalized score, so calibrate it on your "
        "own data before relying on it. An unidentified layout is always flagged.",
    )
    parser.add_argument(
        "--fail-on-degraded",
        action="store_true",
        help="Exit non-zero if any ECG came back degraded (unidentified layout, no "
        "full-length rhythm strip, or interpretation error). Use this in a backend.",
    )
    parser.add_argument("--digitize-only", action="store_true", help="Skip the interpretation stage.")
    parser.add_argument("--json", action="store_true", help="Print the full result list as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-record progress output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    thresholds = None
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text())

    try:
        results = pipeline.run(
            image_dir=args.images,
            output_dir=args.out,
            config=args.config,
            lead_layout=args.lead_layout,
            pathway=args.pathway,
            lead=args.lead,
            top_k=args.top_k,
            device=args.device,
            thresholds=thresholds,
            flat_threshold=args.threshold,
            max_matching_cost=args.max_matching_cost,
            skip_interpretation=args.digitize_only,
            quiet=args.quiet,
        )
    except (digitizer.DigitizerNotFound, digitizer.DigitizerFailed, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(f"No ECGs were digitized from {args.images}", file=sys.stderr)
        return 1

    degraded = [r for r in results if r.get("degraded")]

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif not args.quiet:
        print(f"\n{len(results)} ECG(s) processed -> {args.out}")
        if degraded:
            names = ", ".join(Path(r["source_csv"]).name for r in degraded)
            print(f"{len(degraded)} of {len(results)} came back DEGRADED: {names}")
            print("Do not read these as diagnoses; see the warnings above.")

    if degraded and args.fail_on_degraded:
        print(f"error: {len(degraded)} ECG(s) degraded and --fail-on-degraded was set", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

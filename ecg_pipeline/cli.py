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

from ecg_pipeline import devices, digitizer, pipeline, preprocess
from ecg_pipeline.interpret import PATHWAYS


def upscale_arg(value: str) -> str | int:
    """Parse --upscale: the two keywords, or a positive integer factor."""
    if value in (preprocess.AUTO, preprocess.OFF):
        return value
    try:
        factor = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'auto', 'off' or an integer, got {value!r}") from None
    if factor < 1:
        raise argparse.ArgumentTypeError(f"factor must be >= 1, got {factor}")
    return factor


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
        choices=list(PATHWAYS),
        help="Interpretation pathway. 'rhythm' (default) is the one to trust for rhythm and "
        "rate; 'morphology' reads a phase-aligned representative beat and must not be read "
        "for rhythm; '12lead' is the naive montage, kept for comparison.",
    )
    parser.add_argument("--lead", default="II", help="Lead to use when --pathway 1lead.")
    parser.add_argument("--top-k", type=int, default=10, help="How many diagnoses to report.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for the digitizer and ECGFounder: cpu, cuda or cuda:N "
        f"(default: ${devices.ENV_DEVICE}, else cpu).",
    )
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
    parser.add_argument(
        "--upscale",
        type=upscale_arg,
        default=preprocess.AUTO,
        help="Image upscaling before digitization: 'auto' (default, factor chosen per image "
        f"so it reaches {preprocess.TARGET_WIDTH} px wide), 'off', or an integer factor. A "
        "low-resolution scan digitizes into fragments without this; see the README.",
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
            upscale=args.upscale,
            skip_interpretation=args.digitize_only,
            quiet=args.quiet,
        )
    except (
        digitizer.DigitizerNotFound,
        digitizer.DigitizerFailed,
        devices.DeviceError,
        devices.DeviceUnavailable,
        preprocess.DuplicateRecordName,
        FileNotFoundError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(f"No ECGs were digitized from {args.images}", file=sys.stderr)
        return 1

    # A record with no CSV is an image the digitizer skipped. Counted separately because a
    # run where every image failed must not exit 0 just for having produced a result list.
    digitized = [r for r in results if r.get("source_csv")]
    degraded = [r for r in results if r.get("degraded")]

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif not args.quiet:
        print(f"\n{len(digitized)} of {len(results)} image(s) digitized -> {args.out}")
        if degraded:
            names = ", ".join(r["record"] for r in degraded)
            print(f"{len(degraded)} of {len(results)} came back DEGRADED: {names}")
            print("Do not read these as diagnoses; see the warnings above.")

    if not digitized:
        print(f"error: all {len(results)} image(s) failed to digitize", file=sys.stderr)
        return 1

    if degraded and args.fail_on_degraded:
        print(f"error: {len(degraded)} ECG(s) degraded and --fail-on-degraded was set", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

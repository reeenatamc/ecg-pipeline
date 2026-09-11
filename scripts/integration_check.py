#!/usr/bin/env python3
"""Golden integration check: run the real digitizer on one image, compare against a
stored reference.

Deliberately outside ``tests/`` unittest discovery. The fast suite runs on synthetic
arrays and a mocked digitizer subprocess so it needs neither the Open-ECG-Digitizer
checkout nor its model weights; this script needs both, plus about two minutes of CPU, so
it is opt-in rather than run on every test invocation. Run it by hand after touching
``patches/``, ``configs/``, or updating the digitizer checkout -- the synthetic tests
cannot see a regression that only shows up when the real segmentation network reads a
real scan differently than before.

    python scripts/integration_check.py                  # run + compare (default)
    python scripts/integration_check.py --update-reference  # regenerate the reference

See tests/integration/README.md for where the fixture image comes from.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "tests" / "integration" / "images"
EXPECTED_DIR = REPO_ROOT / "tests" / "integration" / "expected"
RECORD_NAME = "normal"
EXPECTED_CSV = EXPECTED_DIR / f"{RECORD_NAME}_timeseries_canonical.csv"
EXPECTED_META = EXPECTED_DIR / f"{RECORD_NAME}_expected.json"

# Tolerances, not exact equality: the digitizer's segmentation network and this repo's
# Lanczos upscale are floating-point pipelines, and a rerun of the identical image can
# differ in the last bit or two. What must not drift is which leads were recovered, how
# much of each was recovered, and whether the recovered trace is still the same trace.
COVERAGE_TOLERANCE = 0.02
MIN_CORRELATION = 0.99


def run_digitize_only(out_dir: Path) -> dict[str, Any]:
    """Run ``python -m ecg_pipeline --digitize-only --json`` on the one fixture image."""
    cmd = [
        sys.executable,
        "-m",
        "ecg_pipeline",
        "--images",
        str(IMAGES_DIR),
        "--out",
        str(out_dir),
        "--digitize-only",
        "--json",
        "--quiet",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    # Exit 2 is --fail-on-degraded, never passed here; the pipeline itself only ever exits
    # 0 or 1, and 1 covers "every image failed", which a missing/unreadable result below
    # already reports on its own terms.
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"pipeline exited {proc.returncode}")

    results = json.loads(proc.stdout)
    matches = [r for r in results if r.get("record") == RECORD_NAME]
    if not matches:
        raise RuntimeError(f"no result for record {RECORD_NAME!r} in: {[r.get('record') for r in results]}")
    return matches[0]


def leads_with_signal(canonical: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Lead name -> boolean validity mask, for every lead that has at least one sample."""
    from ecg_pipeline.interpret.waveform import CANONICAL_LEADS

    out = {}
    for lead in CANONICAL_LEADS:
        if lead not in names:
            continue
        valid = ~np.isnan(canonical[names.index(lead)])
        if valid.any():
            out[lead] = valid
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        # A flat (or single-valued) overlap: correlation is undefined, not necessarily
        # wrong. Treat identical constants as a match and anything else as a mismatch.
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compare(expected_csv: Path, expected_layout: str, actual_csv: Path, actual_layout: str) -> tuple[bool, list[tuple]]:
    from ecg_pipeline.interpret.waveform import CANONICAL_LEADS, load_canonical_csv

    exp_canon, exp_names = load_canonical_csv(str(expected_csv))
    act_canon, act_names = load_canonical_csv(str(actual_csv))
    exp_leads = leads_with_signal(exp_canon, exp_names)
    act_leads = leads_with_signal(act_canon, act_names)

    rows: list[tuple] = []
    passed = True

    layout_ok = expected_layout == actual_layout
    passed &= layout_ok
    rows.append(("layout", expected_layout, actual_layout, "OK" if layout_ok else "FAIL"))

    leads_ok = set(exp_leads) == set(act_leads)
    passed &= leads_ok
    rows.append(
        (
            "leads_with_signal",
            ",".join(sorted(exp_leads)),
            ",".join(sorted(act_leads)),
            "OK" if leads_ok else "FAIL",
        )
    )

    for lead in CANONICAL_LEADS:
        if lead not in exp_leads or lead not in act_leads:
            continue
        exp_cov = float(exp_leads[lead].sum()) / exp_canon.shape[1]
        act_cov = float(act_leads[lead].sum()) / act_canon.shape[1]
        cov_ok = abs(exp_cov - act_cov) <= COVERAGE_TOLERANCE

        overlap = exp_leads[lead] & act_leads[lead]
        corr = pearson(exp_canon[exp_names.index(lead)][overlap], act_canon[act_names.index(lead)][overlap])
        corr_ok = not np.isnan(corr) and corr >= MIN_CORRELATION

        row_ok = cov_ok and corr_ok
        passed &= row_ok
        rows.append(
            (
                f"  {lead}",
                f"cov {exp_cov:.3f}",
                f"cov {act_cov:.3f}  corr {corr:.4f}",
                "OK" if row_ok else "FAIL",
            )
        )

    return passed, rows


def print_table(rows: list[tuple]) -> None:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


def update_reference() -> None:
    print(f"Running digitizer on {IMAGES_DIR} to (re)generate the reference (about 2 min CPU)...")
    with tempfile.TemporaryDirectory(prefix="ecg-pipeline-integration-") as tmp:
        result = run_digitize_only(Path(tmp))
        csv_path = Path(result["source_csv"])
        layout = (result.get("digitization") or {}).get("lead_layout", "")

        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        EXPECTED_CSV.write_bytes(csv_path.read_bytes())
        EXPECTED_META.write_text(json.dumps({"lead_layout": layout, "record": RECORD_NAME}, indent=2) + "\n")
    print(f"Wrote {EXPECTED_CSV} and {EXPECTED_META}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-reference",
        action="store_true",
        help="Regenerate the stored reference instead of comparing against it.",
    )
    args = parser.parse_args()

    if args.update_reference:
        update_reference()
        return 0

    if not EXPECTED_CSV.is_file() or not EXPECTED_META.is_file():
        print(f"error: no reference at {EXPECTED_CSV} -- run with --update-reference first", file=sys.stderr)
        return 1

    expected_meta = json.loads(EXPECTED_META.read_text())

    with tempfile.TemporaryDirectory(prefix="ecg-pipeline-integration-") as tmp:
        result = run_digitize_only(Path(tmp))
        actual_csv = Path(result["source_csv"])
        actual_layout = (result.get("digitization") or {}).get("lead_layout", "")

        passed, rows = compare(EXPECTED_CSV, expected_meta["lead_layout"], actual_csv, actual_layout)

    header = [("field", "expected", "actual", "status")]
    print_table(header + rows)

    if not passed:
        print("\nFAIL: digitized output drifted from the stored reference.", file=sys.stderr)
        return 1
    print("\nOK: matches the stored reference within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

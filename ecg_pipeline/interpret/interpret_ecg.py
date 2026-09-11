"""Interpret digitized ECG signals with ECGFounder.

Downstream interpretation stage for Open-ECG-Digitizer. Takes the digitizer's
canonical time-series output and runs it through the pretrained ECGFounder model
(PKUDigitalHealth, NEJM AI 2025, MIT license) to produce a top-K list of the
150 diagnostic classes with probabilities.

A standard 3x4 paper ECG only prints ~2.5 s of most leads plus a few full-length
rhythm strips, so no pathway invents signal:

* ``rhythm`` (default) - run the 1-lead checkpoint on the preferred full-length rhythm
      strip(s) (II / V5) and average the opinions; falls back to whichever other
      conventional strip (II / V1 / V5) or lead is full length only when neither preferred
      one was printed. Trustworthy for rhythm/rate.
* ``1lead``  - the 1-lead checkpoint on a single chosen lead (inspection).
* ``morphology`` - a representative (median) beat per lead, phase-aligned across all
      twelve, tiled to 10 s -> 12-lead checkpoint. This is the pathway to use for
      morphology; it removes the false pathology ``12lead`` produces. It ERASES rhythm
      (a median beat is perfectly regular), so rhythm and rate must come from
      ``rhythm``. See ``representative_beat.py``.
* ``12lead`` - each lead's own ~2.5 s column window assembled into a concurrent
      montage -> 12-lead checkpoint. The naive assembly, kept for comparison: a 3x4's
      columns are acquired at different times, so the beats are phase-misaligned and
      this pathway over-calls pathology (false LATERAL INFARCT 0.997 on a normal ECG).
      Prefer ``morphology``.

``net1d.Net1D`` uses global average pooling, so it accepts variable input length;
we keep each lead's native time base rather than stretching a short window to 10 s.

Calibration note: the raw sigmoid outputs are NOT calibrated present/absent scores
(e.g. ``ABNORMAL ECG`` scores high alongside ``NORMAL...``). ECGFounder's own
ptbxl_eval.py binarizes with per-class *optimal thresholds* derived on PTB-XL, not
0.5. Pass ``--thresholds`` (a {label: threshold} JSON) or a flat ``--threshold`` to
get a "flagged" present list; without it the output is a ranking only.

Usage:
    python -m ecg_pipeline.interpret.interpret_ecg --csv out/ecg_timeseries_canonical.csv
    python -m ecg_pipeline.interpret.interpret_ecg --csv <csv> --pathway 1lead --lead II
    python -m ecg_pipeline.interpret.interpret_ecg --csv <csv> --pathway 12lead

Normally you do not call this directly: ``python -m ecg_pipeline`` runs digitization
and interpretation together. This entry point stays for inspecting a single CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

# Re-exported deliberately: these were defined here before ``waveform.py`` existed, and
# callers (including the test suite) import them from this module.
from ecg_pipeline.interpret import PATHWAYS
from ecg_pipeline.interpret.net1d import Net1D  # vendored from ECGFounder (MIT); see NOTICE
from ecg_pipeline.interpret.representative_beat import representative_beat, residual_desync_ms, tile_to_length
from ecg_pipeline.interpret.waveform import (
    CANONICAL_FS,
    CANONICAL_LEADS,
    PREFERRED_RHYTHM_LEADS,
    STANDARD_3X4_COLUMNS,
    assess_quality,
    clean_lead,
    extract_lead,
    interpolate_internal_nans,
    lead_windows,
    load_canonical_csv,
    resample_to,
    select_rhythm_leads,
    zscore,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_PATH = os.path.join(HERE, "tasks.txt")

# Checkpoints resolve against the repo root (overridable with $ECGFOUNDER_WEIGHTS_DIR)
# rather than the process CWD, so the pipeline works from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
WEIGHTS_DIR = os.environ.get("ECGFOUNDER_WEIGHTS_DIR", os.path.join(REPO_ROOT, "weights"))
DEFAULT_1LEAD_CKPT = os.path.join(WEIGHTS_DIR, "1_lead_ECGFounder.pth")
DEFAULT_12LEAD_CKPT = os.path.join(WEIGHTS_DIR, "12_lead_ECGFounder.pth")

# Same override convention as WEIGHTS_DIR above, for the per-class thresholds
# scripts/derive_thresholds_colab.ipynb produces. See configs/thresholds/README.md.
THRESHOLDS_DIR = os.environ.get("ECGFOUNDER_THRESHOLDS_DIR", os.path.join(REPO_ROOT, "configs", "thresholds"))

# rhythm and 1lead run the 1-lead checkpoint; morphology and 12lead run the 12-lead one.
# The 1-lead file is II, not I: the checkpoint was fine-tuned on lead I, but the strips
# this pipeline actually feeds it are II/V1/V5 (a 3x4 print has no full-length lead I), II
# is the one PREFERRED_RHYTHM_LEADS picks first, and II sits inside the checkpoint's own
# training rotation set per the ECGFounder paper -- see configs/thresholds/README.md for
# the full reasoning.
_PATHWAY_THRESHOLD_FILES = {
    "rhythm": "thresholds_1lead_II.json",
    "1lead": "thresholds_1lead_II.json",
    "morphology": "thresholds_12lead.json",
    "12lead": "thresholds_12lead.json",
}

# A few labels worth surfacing explicitly as a normality summary.
SUMMARY_LABELS = ["NORMAL ECG", "NORMAL SINUS RHYTHM", "SINUS RHYTHM", "ABNORMAL ECG"]

# The representative beat is repeated to this duration before scoring: ECGFounder was
# trained on ~10 s records and reads a lone 0.8 s beat poorly (on the reference normal ECG,
# NORMAL ECG 0.12 as a single beat against 0.88 tiled).
MORPHOLOGY_TILE_SECONDS = 10.0

# Exact constructors used by ECGFounder for the released checkpoints
# (see ptbxl_eval.py / finetune_model.py upstream). Only in_channels differs.
_COMMON_KWARGS: dict[str, Any] = dict(
    base_filters=64,
    ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16,
    stride=2,
    groups_width=16,
    use_bn=False,
    use_do=False,
    n_classes=150,
)
MODEL_KWARGS_1LEAD = dict(in_channels=1, **_COMMON_KWARGS)
MODEL_KWARGS_12LEAD = dict(in_channels=12, **_COMMON_KWARGS)


# --------------------------------------------------------------------------- IO


def load_tasks(path: str = TASKS_PATH) -> list[str]:
    """The 150 diagnostic class names, index-aligned to the model's output logits."""
    with open(path, "r") as fin:
        return [line.strip() for line in fin if line.strip()]


def load_thresholds(path: str) -> dict[str, float]:
    with open(path, "r") as fin:
        return {str(k): float(v) for k, v in json.load(fin).items()}


def default_thresholds(pathway: str) -> dict[str, float] | None:
    """The notebook-derived thresholds for whatever checkpoint ``pathway`` runs on.

    Returns ``None`` -- never raises -- when the file is absent (nothing derived yet, this
    repo's state until the Colab run lands) or malformed (partial download, hand-edited
    typo): either way the caller is meant to fall back to a ranking, not to crash a batch
    over one bad threshold file. See configs/thresholds/README.md for which file is which
    and why the 1-lead one is II.
    """
    name = _PATHWAY_THRESHOLD_FILES.get(pathway)
    if name is None:
        return None
    path = os.path.join(THRESHOLDS_DIR, name)
    if not os.path.isfile(path):
        return None
    try:
        return load_thresholds(path)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"[warn] {path}: could not load thresholds ({exc}); ranking only", file=sys.stderr)
        return None


def build_12lead_montage(
    canonical: npt.NDArray[np.float64],
    names: list[str],
    target_len: int = 1250,
    partial_coverage_max: float = 0.5,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Assemble a concurrent 12-lead montage from a 3x4 canonical output (EXPERIMENTAL).

    Each lead contributes its own ~2.5 s column window (real morphology only). A
    full-length rhythm-strip lead borrows the time window of a montage lead sharing
    its layout column. Every lead is resampled to ``target_len`` and stacked, then
    globally z-scored. Note: the columns are non-simultaneous, so beats are not phase
    aligned across leads - this pathway is not yet trustworthy (see module docstring).
    """
    info = lead_windows(canonical, names)
    col_of = {lead: ci for ci, leads in enumerate(STANDARD_3X4_COLUMNS) for lead in leads}
    n = canonical.shape[1]
    nominal = n // len(STANDARD_3X4_COLUMNS)

    col_window: dict[int, tuple[int, int]] = {}
    for ci, leads in enumerate(STANDARD_3X4_COLUMNS):
        cand = [l for l in leads if l in info and info[l]["span"] and info[l]["coverage"] <= partial_coverage_max]
        if cand:
            best = max(cand, key=lambda l: info[l]["span"][1] - info[l]["span"][0])
            col_window[ci] = info[best]["span"]
        else:
            col_window[ci] = (ci * nominal, min((ci + 1) * nominal, n) - 1)

    montage = np.zeros((len(CANONICAL_LEADS), target_len), dtype=np.float64)
    used: dict[str, str] = {}
    for li, lead in enumerate(CANONICAL_LEADS):
        if lead not in names or lead not in info or info[lead]["span"] is None:
            used[lead] = "missing->zeros"
            continue
        x = canonical[names.index(lead)]
        if info[lead]["coverage"] <= partial_coverage_max:
            s, e = info[lead]["span"]
            used[lead] = "own-window"
        else:
            s, e = col_window[col_of[lead]]
            used[lead] = "column-window"
        seg = interpolate_internal_nans(x[s : e + 1])
        if np.isnan(seg).all():
            used[lead] = "empty->zeros"
            continue
        montage[li] = resample_to(np.nan_to_num(seg, nan=0.0), target_len)

    meta = {
        "target_len": target_len,
        "column_windows": {i: list(w) for i, w in col_window.items()},
        "lead_source": used,
    }
    return zscore(montage), meta


# ---------------------------------------------------------------------- model


def _build_model(kwargs: dict[str, Any], ckpt_path: str, device: str) -> Net1D:
    model = Net1D(**kwargs)
    # ``weights_only=False`` is explicit rather than left to the default, because the default
    # flips to True in torch 2.6 and these checkpoints do not load under it: alongside the
    # state_dict they carry the optimizer and scheduler state and a ``val_auroc`` stored as a
    # numpy scalar, which the restricted unpickler rejects. The allowlist that would admit it
    # (``torch.serialization.add_safe_globals``) only exists from 2.4. This is a full pickle
    # of a file downloaded from HuggingFace, so keep ``download_weights.sh`` pointed at the
    # published checkpoints and nowhere else.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    log = model.load_state_dict(state_dict, strict=False)
    if log.missing_keys or log.unexpected_keys:
        print(
            f"[warn] {os.path.basename(ckpt_path)}: state_dict mismatch "
            f"missing={len(log.missing_keys)} unexpected={len(log.unexpected_keys)}",
            file=sys.stderr,
        )
    return model.to(device).eval()


def build_1lead_model(ckpt_path: str = DEFAULT_1LEAD_CKPT, device: str = "cpu") -> Net1D:
    return _build_model(MODEL_KWARGS_1LEAD, ckpt_path, device)


def build_12lead_model(ckpt_path: str = DEFAULT_12LEAD_CKPT, device: str = "cpu") -> Net1D:
    return _build_model(MODEL_KWARGS_12LEAD, ckpt_path, device)


def default_checkpoint(pathway: str) -> str:
    """Which of the two checkpoints a pathway runs on."""
    return DEFAULT_1LEAD_CKPT if pathway in ("rhythm", "1lead") else DEFAULT_12LEAD_CKPT


def build_model_for(pathway: str, ckpt_path: str | None = None, device: str = "cpu") -> tuple[Net1D, str]:
    """Load the model a pathway needs, once, for a caller that will reuse it.

    ``interpret_csv`` builds its own when not given one, which is right for a single CSV and
    wasteful for a batch: reading a 370 MB checkpoint costs over a second per record, paid
    again for every ECG in the run. Returns the checkpoint path alongside the model because
    the result reports which weights produced it.
    """
    ckpt_path = ckpt_path or default_checkpoint(pathway)
    builder = build_1lead_model if pathway in ("rhythm", "1lead") else build_12lead_model
    return builder(ckpt_path, device), ckpt_path


@torch.no_grad()
def predict_probs(model: Net1D, signal: npt.NDArray[np.float64], device: str = "cpu") -> npt.NDArray[np.float64]:
    """Run the model on a (C, L) or (L,) signal and return the 150 sigmoid probabilities."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    x = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, L)
    return torch.sigmoid(model(x)).cpu().numpy().ravel()


def top_k(probs: npt.NDArray[np.float64], tasks: list[str], k: int = 10) -> list[dict[str, Any]]:
    order = np.argsort(probs)[::-1][:k]
    return [
        {"rank": r + 1, "index": int(i), "label": tasks[i], "prob": round(float(probs[i]), 4)}
        for r, i in enumerate(order)
    ]


def flagged_findings(
    probs: npt.NDArray[np.float64],
    tasks: list[str],
    thresholds: dict[str, float] | None = None,
    default: float | None = None,
) -> list[dict[str, Any]]:
    """Classes whose probability clears their per-class threshold (or a flat default)."""
    out: list[dict[str, Any]] = []
    for i, p in enumerate(probs):
        thr = thresholds.get(tasks[i]) if thresholds else None
        if thr is None:
            thr = default
        if thr is not None and p >= thr:
            out.append({"label": tasks[i], "prob": round(float(p), 4), "threshold": round(float(thr), 4)})
    out.sort(key=lambda r: -r["prob"])
    return out


def thresholded_labels(
    tasks: list[str],
    thresholds: dict[str, float] | None = None,
    default: float | None = None,
) -> list[str]:
    """Which labels actually had a threshold to clear, sorted -- distinct from which cleared it.

    A flat ``default`` applies to every label, so it covers all of ``tasks``. A per-class
    ``thresholds`` dict covers only the labels it has an entry for: with a partial threshold
    set (this repo's provisional fold-10 derivation supplies 23 of 150 classes, see
    configs/thresholds/README.md), most labels have no entry and therefore no threshold was
    ever applied to them -- not "checked and absent", just never checked. ``to_observations``
    reads this list to tell those two apart instead of defaulting every non-flagged row to
    ``aboveThreshold: false``.
    """
    if default is not None:
        return sorted(tasks)
    if thresholds:
        return sorted(t for t in tasks if t in thresholds)
    return []


def summary_probs(probs: npt.NDArray[np.float64], tasks: list[str]) -> dict[str, float]:
    idx = {t: i for i, t in enumerate(tasks)}
    return {lab: round(float(probs[idx[lab]]), 4) for lab in SUMMARY_LABELS if lab in idx}


# ------------------------------------------------------------------- pathways


def interpret_rhythm(
    canonical: npt.NDArray[np.float64],
    names: list[str],
    model: Net1D,
    leads: list[str] | None = None,
    coverage_min: float = 0.6,
    method: str = "mean",
    device: str = "cpu",
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]], list[str]]:
    """1-lead model over the selected full-length rhythm strip(s); opinions combined per class.

    ``leads`` (via ``select_rhythm_leads``/``assess_quality``) is not necessarily every
    full-length lead: when at least one of ``PREFERRED_RHYTHM_LEADS`` (II, V5) is full
    length, only the preferred lead(s) are used, because averaging in a lead the checkpoint
    reads poorly (V1, at chance on morphology classes per the fold-10 derivation -- see
    configs/thresholds/README.md) would dilute every class in the combined result. Any other
    full-length lead is used only when none of the preferred ones is present.
    """
    if leads is None:
        leads = select_rhythm_leads(canonical, names, coverage_min)
    if not leads:
        raise ValueError("No usable lead found in canonical CSV.")
    per_lead: dict[str, npt.NDArray[np.float64]] = {}
    for lead in leads:
        signal = zscore(clean_lead(extract_lead(canonical, names, lead)))
        per_lead[lead] = predict_probs(model, signal, device)
    stack = np.stack([per_lead[l] for l in leads])
    combined = stack.max(axis=0) if method == "max" else stack.mean(axis=0)
    return combined, per_lead, leads


def interpret_csv(
    csv_path: str,
    pathway: str = "rhythm",
    lead: str = "II",
    ckpt: str | None = None,
    k: int = 10,
    device: str = "cpu",
    combine: str = "mean",
    coverage_min: float = 0.6,
    thresholds: dict[str, float] | None = None,
    flat_threshold: float | None = None,
    model: Net1D | None = None,
) -> dict[str, Any]:
    """Interpret one canonical CSV.

    ``model`` lets a batch caller load the checkpoint once and hand the same model to every
    record; left None, one is built here, which is the right thing for a single CSV and over
    a second of wasted checkpoint reading per record otherwise.
    """
    tasks = load_tasks()
    canonical, names = load_canonical_csv(csv_path)
    # Only the pathways that read a continuous strip need a full-length lead. The
    # representative beat is built from the ~2.5 s column windows on purpose, so holding it
    # to that requirement would attach a rhythm-pathway complaint to a morphology result --
    # and degrade a 3x3 print, which has no rhythm strip by construction.
    quality = assess_quality(canonical, names, coverage_min, needs_full_length=pathway in ("rhythm", "1lead"))
    warnings: list[str] = list(quality["warnings"])

    # The 12lead pathway is known to over-call pathology -- it returns LATERAL INFARCT
    # ~0.99 on a verified-normal ECG -- because a 3x4's columns are recorded at different
    # times and the montage is therefore phase-misaligned. That is a property of the
    # pathway, not of the input, so it is degraded unconditionally: no clean scan should
    # let a 12lead result pass a --fail-on-degraded gate.
    if pathway == "12lead":
        quality["degraded"] = True
        warnings.append(
            "The 12lead pathway over-calls pathology (a 3x4 layout records its columns at "
            "different times, so the montage is phase-misaligned). Every 12lead result is "
            "marked degraded. Use 'morphology', which aligns the leads first."
        )

    # Two separate reasons, both permanent properties of the pathway rather than of the
    # input. The first is the one that gets people into trouble: the numbers look better
    # than the rhythm pathway's, on a signal from which rhythm has been averaged away.
    if pathway == "morphology":
        quality["degraded"] = True
        warnings.append(
            "The morphology pathway reads a median beat, which is perfectly regular by "
            "construction: it ERASES rhythm. On a confirmed atrial-fibrillation ECG it "
            "reports SINUS RHYTHM 0.98. Disregard every rhythm and rate label here and "
            "take those from the 'rhythm' pathway."
        )
        warnings.append(
            "It corrects the false pathology of the '12lead' montage, but has not yet been "
            "validated against ECGs with confirmed morphological diagnoses, so it is "
            "reported as degraded."
        )

    if pathway == "rhythm":
        ckpt = ckpt or DEFAULT_1LEAD_CKPT
        model = build_1lead_model(ckpt, device) if model is None else model
        probs, per_lead, leads = interpret_rhythm(canonical, names, model, None, coverage_min, combine, device)
        detail: dict[str, Any] = {
            "rhythm_leads": leads,
            "combine": combine,
            "per_lead_topk": {l: top_k(per_lead[l], tasks, 5) for l in leads},
        }
    elif pathway == "1lead":
        ckpt = ckpt or DEFAULT_1LEAD_CKPT
        raw = extract_lead(canonical, names, lead)
        n_total, n_valid = int(raw.size), int(np.sum(~np.isnan(raw)))
        signal = zscore(clean_lead(raw))
        model = build_1lead_model(ckpt, device) if model is None else model
        probs = predict_probs(model, signal, device)
        detail = {
            "lead": lead,
            "coverage": {"valid_samples": n_valid, "total_samples": n_total, "pct": round(100 * n_valid / n_total, 1)},
            "samples_used": int(signal.size),
        }
    elif pathway == "morphology":
        ckpt = ckpt or DEFAULT_12LEAD_CKPT
        beat, beat_meta = representative_beat(canonical, names)
        tiled = tile_to_length(beat, int(MORPHOLOGY_TILE_SECONDS * beat_meta["fs"]))
        model = build_12lead_model(ckpt, device) if model is None else model
        probs = predict_probs(model, tiled, device)
        detail = {
            "representative_beat": {
                **beat_meta,
                "tiled_to_samples": int(tiled.shape[1]),
                # Reported, not gated on: what counts as tolerable is a clinical question,
                # and the answer is being sought. Against an 80-100 ms QRS, tens of ms are
                # not negligible for fine morphology.
                "residual_desync_ms": residual_desync_ms(beat, beat_meta["fs"]),
            }
        }
        if beat_meta["leads_without_a_beat"]:
            warnings.append(
                f"No beat could be extracted for {', '.join(beat_meta['leads_without_a_beat'])}; "
                f"those leads are flat in the montage and contribute nothing."
            )
    elif pathway == "12lead":
        ckpt = ckpt or DEFAULT_12LEAD_CKPT
        montage, meta = build_12lead_montage(canonical, names)
        model = build_12lead_model(ckpt, device) if model is None else model
        probs = predict_probs(model, montage, device)
        detail = {"montage": meta, "experimental": True}
    else:
        raise ValueError(f"Unknown pathway {pathway!r} (use one of: {', '.join(PATHWAYS)}).")

    result: dict[str, Any] = {
        "source_csv": csv_path,
        "model": os.path.basename(ckpt),
        "pathway": pathway,
        **detail,
        "summary": summary_probs(probs, tasks),
        "topk": top_k(probs, tasks, k),
        "signal_quality": quality,
        "degraded": bool(quality["degraded"]),
        "warnings": warnings,
    }
    # Neither an explicit --thresholds file nor a flat --threshold was given: fall back to
    # whatever the notebook derived for this pathway's checkpoint, if it has landed yet.
    # In the rhythm pathway ``probs`` is already the mean over strips (interpret_rhythm
    # above), so these thresholds are applied to that combined vector, not to any one lead.
    default_source: str | None = None
    if thresholds is None and flat_threshold is None:
        thresholds = default_thresholds(pathway)
        if thresholds is not None:
            default_source = _PATHWAY_THRESHOLD_FILES[pathway]

    if thresholds is not None or flat_threshold is not None:
        result["flagged"] = flagged_findings(probs, tasks, thresholds, flat_threshold)
        result["thresholded_labels"] = thresholded_labels(tasks, thresholds, flat_threshold)
        if default_source is not None:
            result["threshold_source"] = f"default:{default_source}"
        else:
            result["threshold_source"] = "per-class" if thresholds else f"flat={flat_threshold}"
    else:
        result["threshold_source"] = "none"
    return result


# --------------------------------------------------------------------- report


def _print_report(res: dict[str, Any]) -> None:
    if res["pathway"] == "rhythm":
        head = f"rhythm 1-lead ensemble over {res['rhythm_leads']}  |  combine={res['combine']}"
    elif res["pathway"] == "1lead":
        cov = res["coverage"]
        head = f"lead {res['lead']}  |  {cov['valid_samples']}/{cov['total_samples']} real ({cov['pct']}%)  |  fed {res['samples_used']}"
    elif res["pathway"] == "morphology":
        beat = res["representative_beat"]
        counts = beat["beats_per_lead"].values()
        desync = beat["residual_desync_ms"].values()
        head = (
            f"representative beat  |  {min(counts, default=0)}-{max(counts, default=0)} beats/lead  |  "
            f"worst desync {max((abs(v) for v in desync), default=0.0):.0f} ms  |  MORPHOLOGY ONLY"
        )
    else:
        head = f"12-lead 2.5s montage  |  {res['montage']['target_len']} samples/lead  |  phase-misaligned"
    print(f"\nECGFounder [{res['pathway']}, 150-class]  |  {head}")
    if res.get("warnings"):
        banner = "DEGRADED" if res.get("degraded") else "CAUTION"
        print(f"\n  !! {banner} !!")
        for w in res["warnings"]:
            print(f"   - {w}")
        print()
    if res.get("summary"):
        print("  summary: " + "   ".join(f"{lab}={p:.2f}" for lab, p in res["summary"].items()))
    print("-" * 68)
    print(f"{'#':>2}  {'prob':>6}  label")
    for row in res["topk"]:
        print(f"{row['rank']:>2}  {row['prob']:>6.3f}  {row['label']}")
    if "flagged" in res:
        print(f"\nflagged present ({res['threshold_source']}):")
        if res["flagged"]:
            for row in res["flagged"]:
                print(f"   {row['prob']:>6.3f}  {row['label']}  (thr {row['threshold']})")
        else:
            print("   (none cleared threshold)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interpret a digitized ECG with ECGFounder (150-class).")
    ap.add_argument("--csv", required=True, help="Digitizer canonical timeseries CSV")
    ap.add_argument("--pathway", choices=list(PATHWAYS), default="rhythm")
    ap.add_argument("--lead", default="II", help="[1lead] Lead to interpret (a full rhythm strip is best)")
    ap.add_argument("--combine", choices=["mean", "max"], default="mean", help="[rhythm] combine leads")
    ap.add_argument(
        "--coverage-min", type=float, default=0.6, help="[rhythm] min coverage to treat a lead as full-length"
    )
    ap.add_argument("--ckpt", default=None, help="Checkpoint path (defaults per pathway)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--thresholds", default=None, help="JSON {label: threshold} for present/absent flags")
    ap.add_argument("--threshold", type=float, default=None, help="Flat threshold (fallback if no --thresholds)")
    ap.add_argument("--json-out", default=None, help="Optional path to write the full result as JSON")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    thresholds = load_thresholds(args.thresholds) if args.thresholds else None
    res = interpret_csv(
        args.csv,
        args.pathway,
        args.lead,
        args.ckpt,
        args.topk,
        args.device,
        combine=args.combine,
        coverage_min=args.coverage_min,
        thresholds=thresholds,
        flat_threshold=args.threshold,
    )
    _print_report(res)
    if args.json_out:
        with open(args.json_out, "w") as fout:
            json.dump(res, fout, indent=2, ensure_ascii=False)
        print(f"\nJSON -> {args.json_out}")


if __name__ == "__main__":
    main()

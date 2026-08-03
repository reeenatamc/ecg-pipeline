"""Interpret digitized ECG signals with ECGFounder.

Downstream interpretation stage for Open-ECG-Digitizer. Takes the digitizer's
canonical time-series output and runs it through the pretrained ECGFounder model
(PKUDigitalHealth, NEJM AI 2025, MIT license) to produce a top-K list of the
150 diagnostic classes with probabilities.

A standard 3x4 paper ECG only prints ~2.5 s of most leads plus a few full-length
rhythm strips, so no pathway invents signal:

* ``rhythm`` (default) - run the 1-lead checkpoint on every full-length rhythm
      strip (II / V1 / V5) and average the opinions. Trustworthy for rhythm/rate.
* ``1lead``  - the 1-lead checkpoint on a single chosen lead (inspection).
* ``12lead`` - each lead's own ~2.5 s column window assembled into a concurrent
      montage -> 12-lead checkpoint. EXPERIMENTAL: a 3x4's columns are acquired at
      different times, so the beats are phase-misaligned and this pathway over-calls
      pathology (e.g. false LATERAL INFARCT on a normal ECG). Needs representative-
      beat / R-peak alignment before it can be trusted; kept here for that follow-up.

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

# net1d.py is vendored next to this file (from ECGFounder, MIT). Adding this directory
# to sys.path keeps it importable both as a script and as a module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net1d import Net1D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS_PATH = os.path.join(HERE, "tasks.txt")

# Checkpoints resolve against the repo root (overridable with $ECGFOUNDER_WEIGHTS_DIR)
# rather than the process CWD, so the pipeline works from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
WEIGHTS_DIR = os.environ.get("ECGFOUNDER_WEIGHTS_DIR", os.path.join(REPO_ROOT, "weights"))
DEFAULT_1LEAD_CKPT = os.path.join(WEIGHTS_DIR, "1_lead_ECGFounder.pth")
DEFAULT_12LEAD_CKPT = os.path.join(WEIGHTS_DIR, "12_lead_ECGFounder.pth")

# Canonical lead order emitted by the digitizer (matches ECGFounder's expected order).
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Leads most commonly printed as full-length rhythm strips, in preference order.
PREFERRED_RHYTHM_LEADS = ["II", "V1", "V5"]

# Standard 3x4 paper layout: the four time columns and the leads printed in each.
STANDARD_3X4_COLUMNS = [
    ["I", "II", "III"],
    ["aVR", "aVL", "aVF"],
    ["V1", "V2", "V3"],
    ["V4", "V5", "V6"],
]

# A few labels worth surfacing explicitly as a normality summary.
SUMMARY_LABELS = ["NORMAL ECG", "NORMAL SINUS RHYTHM", "SINUS RHYTHM", "ABNORMAL ECG"]

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


def load_thresholds(path: str) -> dict[str, float]:
    with open(path, "r") as fin:
        return {str(k): float(v) for k, v in json.load(fin).items()}


# ------------------------------------------------------------------ signal prep

def _interp_internal_nans(seg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Linearly interpolate NaNs that sit between valid samples."""
    seg = seg.astype(np.float64).copy()
    holes = np.isnan(seg)
    if holes.any() and not holes.all():
        idx = np.arange(seg.size)
        seg[holes] = np.interp(idx[holes], idx[~holes], seg[~holes])
    return seg


def _resample_to(x: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    """Linear resample a 1-D array to ``n`` samples (matches ECGFounder resample_unequal)."""
    if x.size == n or x.size == 0:
        return x
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, x.size), x)


def zscore(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ECGFounder's normalization: global mean/std over the whole array.

    For 1 lead this is per-lead; for the 12-lead montage it preserves the relative
    amplitudes between leads (diagnostically meaningful). Units cancel either way.
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
    return _interp_internal_nans(x[valid[0] : valid[-1] + 1])


def _lead_windows(canonical: npt.NDArray[np.float64], names: list[str]) -> dict[str, dict[str, Any]]:
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
    canonical: npt.NDArray[np.float64], names: list[str], coverage_min: float = 0.6
) -> dict[str, Any]:
    """Single source of truth for which leads are usable and how much to trust them.

    The rhythm pathway is only meaningful when at least one lead was printed at full
    length. When none was, the caller still gets an answer -- built from the single
    best-covered lead, which on a 3x4 layout is a ~2.5 s fragment. That is a materially
    weaker basis for a rhythm call, so it is reported as ``degraded`` rather than
    silently passed off as a rhythm-strip ensemble.
    """
    info = _lead_windows(canonical, names)
    full = [l for l in CANONICAL_LEADS if l in info and info[l]["coverage"] >= coverage_min]
    ordered = [l for l in PREFERRED_RHYTHM_LEADS if l in full] + [l for l in full if l not in PREFERRED_RHYTHM_LEADS]

    warnings: list[str] = []
    degraded = False
    if ordered:
        selected = ordered
    elif info:
        selected = [max(info, key=lambda l: info[l]["coverage"])]
        degraded = True
        pct = 100 * info[selected[0]]["coverage"]
        warnings.append(
            f"No lead reached {100 * coverage_min:.0f}% coverage, so no full-length rhythm strip was "
            f"available. Fell back to the single best-covered lead ({selected[0]}, {pct:.0f}%). "
            f"Treat rhythm and rate findings as unreliable."
        )
    else:
        selected = []
        degraded = True
        warnings.append("No leads were recovered from the digitized signal.")

    usable = [l for l in info if info[l]["coverage"] > 0]
    if info and len(usable) < 6:
        warnings.append(
            f"Only {len(usable)} of 12 leads carry any signal ({', '.join(usable) or 'none'}). "
            f"The digitization is likely incomplete."
        )

    return {
        "coverage": {l: round(info[l]["coverage"], 3) for l in info},
        "leads_with_signal": usable,
        "full_length_leads": ordered,
        "selected_leads": selected,
        "coverage_min": coverage_min,
        "degraded": degraded,
        "warnings": warnings,
    }


def select_rhythm_leads(canonical: npt.NDArray[np.float64], names: list[str], coverage_min: float = 0.6) -> list[str]:
    """Full-length leads usable as rhythm strips, preferred order; else best-covered single lead.

    Thin wrapper over ``assess_quality``; use that directly when you need to know whether
    the selection was degraded.
    """
    return list(assess_quality(canonical, names, coverage_min)["selected_leads"])


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
    info = _lead_windows(canonical, names)
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
        seg = _interp_internal_nans(x[s : e + 1])
        if np.isnan(seg).all():
            used[lead] = "empty->zeros"
            continue
        montage[li] = _resample_to(np.nan_to_num(seg, nan=0.0), target_len)

    meta = {"target_len": target_len, "column_windows": {i: list(w) for i, w in col_window.items()}, "lead_source": used}
    return zscore(montage), meta


# ---------------------------------------------------------------------- model

def _build_model(kwargs: dict[str, Any], ckpt_path: str, device: str) -> Net1D:
    model = Net1D(**kwargs)
    checkpoint = torch.load(ckpt_path, map_location=device)
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
    return [{"rank": r + 1, "index": int(i), "label": tasks[i], "prob": round(float(probs[i]), 4)} for r, i in enumerate(order)]


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
    """1-lead model over every full-length rhythm strip; opinions combined per class."""
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
) -> dict[str, Any]:
    tasks = load_tasks()
    canonical, names = load_canonical_csv(csv_path)
    quality = assess_quality(canonical, names, coverage_min)
    warnings: list[str] = list(quality["warnings"])

    if pathway == "12lead":
        warnings.append(
            "The 12lead pathway is EXPERIMENTAL. A 3x4 layout records its columns at "
            "different times, so the montage is phase-misaligned and this pathway "
            "over-calls pathology. Prefer 'rhythm'."
        )

    if pathway == "rhythm":
        ckpt = ckpt or DEFAULT_1LEAD_CKPT
        model = build_1lead_model(ckpt, device)
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
        model = build_1lead_model(ckpt, device)
        probs = predict_probs(model, signal, device)
        detail = {
            "lead": lead,
            "coverage": {"valid_samples": n_valid, "total_samples": n_total, "pct": round(100 * n_valid / n_total, 1)},
            "samples_used": int(signal.size),
        }
    elif pathway == "12lead":
        ckpt = ckpt or DEFAULT_12LEAD_CKPT
        montage, meta = build_12lead_montage(canonical, names)
        model = build_12lead_model(ckpt, device)
        probs = predict_probs(model, montage, device)
        detail = {"montage": meta, "experimental": True}
    else:
        raise ValueError(f"Unknown pathway {pathway!r} (use 'rhythm', '1lead' or '12lead').")

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
    if thresholds is not None or flat_threshold is not None:
        result["flagged"] = flagged_findings(probs, tasks, thresholds, flat_threshold)
        result["threshold_source"] = "per-class" if thresholds else f"flat={flat_threshold}"
    return result


# --------------------------------------------------------------------- report

def _print_report(res: dict[str, Any]) -> None:
    if res["pathway"] == "rhythm":
        head = f"rhythm 1-lead ensemble over {res['rhythm_leads']}  |  combine={res['combine']}"
    elif res["pathway"] == "1lead":
        cov = res["coverage"]
        head = f"lead {res['lead']}  |  {cov['valid_samples']}/{cov['total_samples']} real ({cov['pct']}%)  |  fed {res['samples_used']}"
    else:
        head = f"12-lead 2.5s montage  |  {res['montage']['target_len']} samples/lead  |  EXPERIMENTAL"
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
    ap.add_argument("--pathway", choices=["rhythm", "1lead", "12lead"], default="rhythm")
    ap.add_argument("--lead", default="II", help="[1lead] Lead to interpret (a full rhythm strip is best)")
    ap.add_argument("--combine", choices=["mean", "max"], default="mean", help="[rhythm] combine leads")
    ap.add_argument("--coverage-min", type=float, default=0.6, help="[rhythm] min coverage to treat a lead as full-length")
    ap.add_argument("--ckpt", default=None, help="Checkpoint path (defaults per pathway)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--thresholds", default=None, help="JSON {label: threshold} for present/absent flags")
    ap.add_argument("--threshold", type=float, default=None, help="Flat threshold (fallback if no --thresholds)")
    ap.add_argument("--json-out", default=None, help="Optional path to write the full result as JSON")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    thresholds = load_thresholds(args.thresholds) if args.thresholds else None
    res = interpret_csv(
        args.csv, args.pathway, args.lead, args.ckpt, args.topk, args.device,
        combine=args.combine, coverage_min=args.coverage_min,
        thresholds=thresholds, flat_threshold=args.threshold,
    )
    _print_report(res)
    if args.json_out:
        with open(args.json_out, "w") as fout:
            json.dump(res, fout, indent=2, ensure_ascii=False)
        print(f"\nJSON -> {args.json_out}")


if __name__ == "__main__":
    main()

# Per-class thresholds

The JSON files that belong here are not written by hand and not checked into this repo
until they exist: `scripts/derive_thresholds_colab.ipynb` produces them by evaluating
ECGFounder against PTB-XL and finding, per class, the probability that maximizes balanced
accuracy (upstream's own method, see the notebook and `docs/MODELOS.md`). Each file is a
flat `{label: threshold}` JSON, one entry per class the notebook found enough positives
for -- a class below its `MIN_POSITIVES` cutoff has no entry rather than a threshold
computed on noise.

## Which file is which

The notebook names its output `thresholds_<variant>.json`, one file per (checkpoint, lead)
combination it evaluated. This directory only needs the two the pipeline actually resolves
by default:

| File | Checkpoint | Used by pathway |
|---|---|---|
| `thresholds_1lead_II.json` | 1-lead | `rhythm`, `1lead` |
| `thresholds_12lead.json` | 12-lead | `morphology`, `12lead` |

`interpret_ecg.default_thresholds(pathway)` does that mapping. The other `1lead_*` files
the notebook writes (`thresholds_1lead_I.json`, `thresholds_1lead_V1.json`,
`thresholds_1lead_V5.json`, and the `_bandpass` variants) are comparison material for
deciding whether the pipeline should keep using the 1-lead checkpoint on II -- they are not
wired into any pathway and do not need to live here.

## Why II, not I

The 1-lead checkpoint was fine-tuned on lead I, but the `rhythm` pathway (this pipeline's
default and most trustworthy pathway) never sees lead I: a standard 3x4 print's only
full-length strips are II, V1 and V5 (AHA/ACCF/HRS 2007), and II is the one the pipeline
prefers when more than one is available (`PREFERRED_RHYTHM_LEADS` in `waveform.py`). II is
inside the checkpoint's training rotation set per the ECGFounder paper (the 1-lead model
was trained across a rotation of single leads, not on I exclusively), which is why scoring
II with it is not the same kind of extrapolation as, say, scoring aVR with it would be.
`thresholds_1lead_II.json` is therefore the threshold set that matches what `rhythm` and
`1lead` (whose own default `--lead` is II) actually feed the checkpoint.

## A missing file means ranking only

`default_thresholds` returns `None` when its file is absent (nothing derived yet, this
repo's current state) or malformed (a partial download, a hand-edited typo). Nothing raises
either way: `interpret_csv` then reports `threshold_source: "none"` and the caller gets
ECGFounder's raw ranking, exactly as documented in the README's "Scores are rankings, not
calibrated probabilities" section. A ranking without thresholds is a legitimate, expected
mode of this pipeline, not a degraded one.

## Overriding the location

`$ECGFOUNDER_THRESHOLDS_DIR` points `default_thresholds` at a different directory, the same
way `$ECGFOUNDER_WEIGHTS_DIR` overrides where checkpoints are read from -- useful for trying
a threshold set derived on a different notebook run without touching this one.

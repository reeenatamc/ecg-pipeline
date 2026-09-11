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

## Why V1 is no longer a preferred rhythm lead

`PREFERRED_RHYTHM_LEADS` in `waveform.py` used to list `["II", "V1", "V5"]`; it is now
`["II", "V5"]`. A local derivation on PTB-XL `strat_fold` 10 (2198 records, same procedure
as the notebook; see `provisional/fold10_2026-09-11/summary.md`) measured the four 1-lead
checkpoint variants (I, II, V1, V5) separately: on rhythm classes all four are close (AF
0.98, sinus tachycardia 0.99, sinus bradycardia 0.93-0.95), but V1 collapses on morphology
classes (RIGHT BUNDLE BRANCH BLOCK AUROC 0.53, chance; NORMAL ECG 0.67) while II and V5
hold 0.80-0.82. The `rhythm` pathway's combined 150-class vector is the *mean* of every
strip's own vector, so a lead that is fine on rhythm but at chance on morphology dilutes
every morphology class in that average if it is averaged in.

This is not just a reordering: `assess_quality` now actually *excludes* a full-length lead
that is not in `PREFERRED_RHYTHM_LEADS` whenever at least one preferred lead is also full
length. On the common shape (a 3x4 print with II, V1 and V5 all printed full length),
`selected_leads` is `["II", "V5"]`; V1 is still reported in `full_length_leads` (it was
printed, and it is still a legitimate signal) but is not averaged in, and a warning names it
and explains why, so the omission is never silent. `PREFERRED_RHYTHM_LEADS` therefore
controls two things: which leads are averaged when there is a choice, and which one wins
when there isn't. II and V5 are what gets averaged because both are inside or near the
checkpoint's training rotation (see "Why II, not I" above).

A V1-only print (or any single unconventional strip, e.g. a wildcard identified as aVF)
still gets read: `assess_quality` falls back to *every* full-length lead only when *none* of
the preferred ones is present, so a print whose only strip is V1 still selects V1 alone, not
degraded, no omission warning (there is nothing to omit). Gate 5 in `pipeline.py` (unverified
rhythm-strip identity) does not use `PREFERRED_RHYTHM_LEADS` at all; it uses the separate
`CONVENTIONAL_STRIP_LEADS = ("II", "V1", "V5")`, because that gate is about whether the strip
is a *known* lead, not about how well the checkpoint reads it -- a V1 strip excluded from the
rhythm average by the paragraph above is still a verified, non-degrading strip as far as gate
5 is concerned.

## Provisional thresholds (fold 10)

`provisional/fold10_2026-09-11/` holds a threshold set derived locally (Mac, CPU) on PTB-XL
`strat_fold` 10 only -- 2198 of the 21799 records, about a tenth of the dataset -- as a
stopgap while the full-set Colab run (`scripts/derive_thresholds_colab.ipynb`) had not yet
been run. Same procedure as the notebook (z-score, `optimal_threshold` = balanced-accuracy
sweep 0.01-0.99, `MIN_POSITIVES=20`), on a smaller sample. It contains the seven
`thresholds_<variant>.json` files, `metrics.csv` (AUROC/threshold/positives for all 150
classes x 7 variants), and `summary.md` (in Spanish) with the full numbers and reasoning,
including the bandpass comparison (+0.006 mean AUROC on 12-lead, -0.002 on 1-lead II --
not enough of a case to turn bandpass on by default).

**This directory is not auto-loaded.** `default_thresholds` only ever reads
`configs/thresholds/thresholds_1lead_II.json` and `configs/thresholds/thresholds_12lead.json`
-- it does not walk subdirectories, so nothing under `provisional/` is picked up unless a
caller passes it explicitly via `$ECGFOUNDER_THRESHOLDS_DIR` (see
`tests/test_thresholds.py::TestDefaultThresholdsLoader::test_ignores_subdirectories`).

Only 23 of the 150 classes reach `MIN_POSITIVES=20` in this fold, versus the notebook's
full 21799 records, which is expected to clear the cutoff for substantially more classes,
`ATRIAL FLUTTER` (7 positives here) among them. **When the full-set Colab run lands, it
should replace this set**, by copying its own `thresholds_1lead_II.json` and
`thresholds_12lead.json` up one level into `configs/thresholds/` (overwriting nothing that
exists today, since nothing does yet) -- the same two files this README already documents
as the ones `default_thresholds` resolves.

`NORMAL SINUS RHYTHM` and `ABNORMAL ECG` have zero positives in ECGFounder's PTB-XL label
mapping -- not a fold artefact, the same is true across the full dataset per
`ptbxl_label.csv` -- so neither class can ever get a threshold from a PTB-XL derivation
with this label mapping, no matter how much data is used. Both stay ranking-only
permanently; `SUMMARY_LABELS` in `interpret_ecg.py` still reports their raw probabilities,
but a consumer should not expect a `flagged`/`aboveThreshold` verdict for either. More
generally, `docs/etiquetas_es_borrador.csv` already marks `NORMAL ECG` and `ABNORMAL ECG`
as summary-only labels (`mostrar_al_usuario: no`) that are best not shown next to specific
findings in the app -- they describe the whole tracing, not a finding on it.

## Overriding the location

`$ECGFOUNDER_THRESHOLDS_DIR` points `default_thresholds` at a different directory, the same
way `$ECGFOUNDER_WEIGHTS_DIR` overrides where checkpoints are read from -- useful for trying
a threshold set derived on a different notebook run without touching this one.

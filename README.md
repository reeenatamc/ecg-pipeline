# ecg-pipeline

Paper ECG image → digitized time series → diagnostic interpretation.

Two stages:

1. **Digitization**: [Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer)
   converts a scanned or photographed 12-lead ECG into canonical time series.
   It runs as an **external process**; its source is never bundled here.
2. **Interpretation**: [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder)
   scores the digitized signal against 150 diagnostic classes.

## Why the digitizer is not vendored

Open-ECG-Digitizer is **CC BY-SA 4.0**, a ShareAlike (copyleft) licence. Copying its
source here would make this repository Adapted Material and force it under CC BY-SA 4.0
too. Running it as a separate program does not, and CC BY-SA has no network/SaaS clause,
so serving it from a backend triggers nothing. Its *output* is not encumbered either.

This is a structural property of the repo, not a detail. **Do not paste upstream source
into `ecg_pipeline/`.** Behaviour changes go in `configs/` or `patches/`. See
[NOTICE](NOTICE) for the full reasoning and the mandatory citation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/setup_digitizer.sh    # clone + patch the digitizer (uses an existing
                                   # checkout if you already have one)
bash scripts/download_weights.sh   # ECGFounder checkpoints, ~740 MB total
```

If your digitizer checkout lives somewhere non-standard:

```bash
export OPEN_ECG_DIGITIZER_HOME=/path/to/Open-ECG-Digitizer
```

The digitizer needs its own dependencies (`pip install -r $OPEN_ECG_DIGITIZER_HOME/requirements.txt`).
It can share this virtualenv or use its own, if separate, pass the interpreter through
`digitize(python_exe=...)`.

## Usage

```bash
# Full pipeline
python -m ecg_pipeline --images path/to/images --out path/to/output

# Right-sided limb leads
python -m ecg_pipeline --images in/ --out out/ --lead-layout lead_layouts_limbaug_right.yml

# Morphology, on a phase-aligned representative beat
python -m ecg_pipeline --images in/ --out out/ --pathway morphology

# Digitization only, without the default image upscale
python -m ecg_pipeline --images in/ --out out/ --digitize-only --upscale off

# Machine-readable
python -m ecg_pipeline --images in/ --out out/ --json
```

Per ECG the pipeline writes `<name>_timeseries_canonical.csv` and
`<name>_interpretation.json` into the output directory.

As a library:

```python
from ecg_pipeline import pipeline

results = pipeline.run(image_dir="in/", output_dir="out/", pathway="rhythm")
```

## Interpretation pathways

| Pathway | What it does | Trust |
|---|---|---|
| `rhythm` (default) | 1-lead model over the preferred full-length strip(s) (II/V5; falls back to whichever other conventional strip -- II/V1/V5 per AHA/ACCF/HRS 2007 -- or lead is full length when neither is printed -- see `configs/thresholds/README.md`), opinions averaged | Reliable for rhythm and rate |
| `1lead` | 1-lead model on one chosen lead | Inspection/debugging |
| `morphology` | Median beat per lead, phase-aligned across all 12, tiled to 10 s → 12-lead model | Morphology only, **never rhythm** |
| `12lead` | Per-lead ~2.5 s windows assembled into a montage → 12-lead model | Naive; kept for comparison |

A standard 3×4 paper ECG prints only ~2.5 s of most leads, and the columns are recorded
at *different times*, up to 7.5 s apart. The `12lead` pathway feeds the model those
columns as if they were simultaneous, and it reads the misalignment as pathology: a false
`LATERAL INFARCT` at 0.997 on a verified-normal ECG.

`morphology` fixes that the way commercial electrocardiographs do internally, a
**representative complex**. Within one column the three leads *are* simultaneous, so R
peaks found in the column's strongest lead time every lead in it; each lead's beats are
then median-averaged and the twelve are assembled on a common fiducial. On that same
normal ECG:

| | `12lead` | `morphology` |
|---|---:|---:|
| `LATERAL INFARCT` | 0.997 ⛔ false | out of the ranking |
| `LOW VOLTAGE QRS` | 0.988 ⛔ false | out of the ranking |
| `NORMAL ECG` | 0.10 | **0.88** |
| `ABNORMAL ECG` | 0.999 | 0.55 |

**It erases rhythm, and that is not a bug.** A median beat is perfectly regular by
construction. On a confirmed atrial-fibrillation ECG, `ATRIAL FIBRILLATION` falls from
0.927 (via `rhythm`) out of the top ranking, and the model reads `SINUS RHYTHM` 0.982 on a
patient in AF. Take rhythm and rate from `rhythm`; take morphology from here. Both are
reported as `degraded`, `morphology` because it has not yet been validated against ECGs
with confirmed morphological diagnoses.

Each `morphology` result reports `beats_per_lead` (2–4 for grid leads, 10–12 for rhythm
strips) and `residual_desync_ms`, the offset each lead still carries after alignment
(measured: 0–10 ms in eleven leads, −22 ms in V6). The fiducial is each lead's *dominant*
deflection, which is the S wave where the QRS points down (V1–V4), moving it to QRS onset,
or to a global fiducial from the 12-lead vector magnitude, is the known next improvement.

**Scores are rankings, not calibrated probabilities.** Raw sigmoid outputs are not
present/absent decisions, `ABNORMAL ECG` can score high alongside `NORMAL ECG`.
ECGFounder binarizes with per-class thresholds derived on PTB-XL. Pass `--thresholds`
(a `{label: threshold}` JSON) or `--threshold` to get a flagged list; without one you get
a ranking only. Upstream does not publish the thresholds, it computes them at evaluation
time; `scripts/derive_thresholds_colab.ipynb` reproduces that computation on a Colab GPU
(about half an hour) and also measures two things this README leaves open: whether the
bandpass upstream uses in fine-tuning helps the pretrained model, and how much the 1-lead
checkpoint, built for lead I, loses on the II/V1/V5 strips the `rhythm` pathway feeds it.

You do not have to pass `--thresholds` yourself. `interpret_csv` calls
`interpret_ecg.default_thresholds(pathway)`, which reads whatever the notebook produced
from `configs/thresholds/` (`thresholds_1lead_II.json` for `rhythm`/`1lead`,
`thresholds_12lead.json` for `morphology`/`12lead`; override the directory with
`$ECGFOUNDER_THRESHOLDS_DIR`) whenever neither `--thresholds` nor `--threshold` was given
explicitly. Every result now carries `threshold_source`: `"per-class"` or `"flat=<n>"` for
an explicit choice, `"default:<file>"` when one of those files supplied it, and `"none"`
when no threshold applied at all -- so a consumer can always tell whether `flagged` reflects
a real binarization or is simply absent. See `configs/thresholds/README.md` for which file
is which and why the 1-lead one is II rather than I.

## Image preprocessing

Images are upscaled before digitization, by default. This is not cosmetic: the digitizer
segments the trace with a U-Net, and on a low-resolution scan the printed line is thin
enough that the network loses most of it. Measured on the reference normal ECG
(797×446, Lanczos ×3 → 2391×1338):

| | native | ×3 |
|---|---:|---:|
| lead I samples recovered | 26 % | **99 %** |
| Einthoven `II = I + III` (corr) | 0.06 | **0.98** |
| leads carrying signal | 11/12 | **12/12** |
| verdict | `DEGRADED` | clean |

`--upscale auto` (the default) picks the smallest integer factor that brings the image to
2000 px wide, capped at ×3. The cap is a measurement, not caution: on the 488 px
right-sided ECG, ×3 → 1464 px matches the paper grid at cost 0.17 while ×4 → 1952 px
recovers the same 9 leads at 0.70. Pass `--upscale off` or an explicit factor to override.
The factor applied is reported per record under `preprocessing`, and printed on the run 
it is never silent. Lanczos resampling invents no signal; it gives the segmentation network
enough pixels to find the trace already printed on the paper.

## Quality gating

A digitizer that fails still produces a CSV, and ECGFounder will happily score noise at
0.99. Every result therefore carries `degraded`, `warnings`, `digitization`, and
`signal_quality`. **Read `degraded` before `topk`.**

Five independent gates, all needed. None catches the others' failures:

1. **Layout** (from the digitizer's `digitization_metadata.csv`). When no layout matches,
   the digitizer emits `lead_layout: "Unknown layout"`, canonicalization returns an
   all-NaN frame, and whatever signal survives was recovered by rhythm-strip cosine
   matching, so *which trace is which lead* is a guess. Always flagged.
2. **Coverage** (from the signal itself). A reading is only meaningful when a lead was
   printed at full length. If none reaches `--coverage-min`, the result is built from a
   single ~2.5 s fragment; that is flagged rather than passed off as an ensemble.
3. **Lead completeness.** Gate 1 only catches the digitizer *admitting* it found nothing.
   It says nothing about a layout matched confidently and wrongly, which returns a
   complete-looking result for whatever leads that layout happens to have. Any lead short
   of the full twelve is therefore named in the warnings, "9 of 12" on a 3×3 print is
   expected, "6 of 12" on a 3×4 is a wrong layout.
4. **Template completeness.** Gate 3 only compares against the full twelve, so a layout
   that legitimately defines fewer leads passes it even when the match itself is wrong: a
   slide screenshot matched `cabrera_6x1_limb` (a 6-lead limb layout) at cost 1.45 and came
   back with 5 of its own 6 leads carrying signal, clean on every earlier gate. This reads
   the matched layout's own template -- from the digitizer's `lead_layouts_all.yml`, plus
   this repo's own layouts in `configs/` -- and flags any lead the template defines that the
   signal does not have. When the template cannot be read, this gate warns rather than
   guesses and does not degrade the record on its own.
5. **Rhythm-strip identity.** On a 3×4 print with a wildcard rhythm lead
   (`standard_3x4_with_rN`), the digitizer identifies which lead the strip is by cosine
   similarity rather than the print's own label, and that guess is sometimes wrong: a strip
   printed as II has come back identified as aVF, or as V3. A full-length lead outside the
   conventional strip set -- II, V1, V5, per AHA/ACCF/HRS 2007 -- is flagged as unverified.
   A layout with no wildcard rhythm lead (a 12×1, a 6×2) never triggers this.

All five run on `--digitize-only` too, so `--fail-on-degraded` is meaningful without
paying for the interpretation stage.

Every result also carries `gates`: the stable, machine-readable ids of whichever gates
fired, alongside the prose in `warnings`. `contract.py` reads this list to pick a
`failure_reason` instead of re-parsing warning text, and `_report_record` prints the ids
next to `[DEGRADED]` on the CLI. Gate 3 only ever warns, never degrades, so it has no id.

| Gate | id(s) |
|---|---|
| 1. Layout | `layout-unknown` |
| 2. Coverage | `no-signal` (nothing recovered at all), `no-full-length-lead` (no lead reached `--coverage-min`) |
| 4. Template completeness | `leads-missing-from-template` |
| 5. Rhythm-strip identity | `rhythm-strip-unverified` |

Two more ids are not gates on the signal itself but cover the batch-level failures the
contract also has to classify: `digitizer-no-output` (the image the digitizer skipped, see
[Batch runs](#batch-runs)) and `interpretation-error` (an exception raised inside the
interpretation stage).

Observed on the sample ECGs:

| Input | Layout | Leads | Verdict |
|---|---|---|---|
| clean scan | `standard_3x4_with_r3` | 12/12, rhythm II/V1/V5 | clean, `SINUS RHYTHM 0.99` |
| low-res scan | `standard_3x4_with_r1` | none ≥60% | `DEGRADED`, fell back to V2 alone |
| unmatched layout | `Unknown layout` | 3/12 | `DEGRADED`, lead identity unreliable |

The second case passes the layout gate and the third passes the coverage gate, which is
why both exist.

Those rows were recorded with the upscale off, and the second one is what it is for: at
native resolution that scan reaches no full-length lead and falls back to a fragment, while
the same image at ×3 comes back 12/12 and clean. Upscaling is the first defence; the gates
are what catch the images it cannot save.

**Upscaling can move a failure from gate 1 to gate 2 or 3, so all three matter.** The
right-sided ECG at native resolution makes the digitizer give up, `Unknown layout`, cost
1.0, caught by gate 1. Given more pixels it stops giving up and matches a *wrong* layout
instead, with a cost that looks perfectly ordinary (`standard_3x4` at 0.271, against 0.167
for the correct forced layout). Nothing in the cost separates them; what does is that no
lead reaches full length. Never gate on `matching_cost` alone.

### Batch runs

A failed image does not stop the batch and does not vanish from it either. The digitizer
catches its own per-image errors and exits 0, so a skipped image simply produces no CSV;
the pipeline compares what it sent against what came back and returns a `degraded` record
carrying the error for each one. The run summary counts digitized images against submitted
ones, `--fail-on-degraded` catches the failures, and a batch where *every* image failed
exits 1 rather than reporting an empty success.

The ECGFounder checkpoint is read **once per run**, not once per ECG, 370 MB and over a
second each time. On a four-ECG batch that alone is a 5.5× difference.

Subdirectories under `--images` are processed, and their structure is mirrored into the
output. Records are keyed by their path relative to the output root, so `batch-1/ecg` and
`batch-2/ecg` stay distinct, the digitizer writes one `digitization_metadata.csv` per
subdirectory and all of them are read.

The output directory is **not** wiped. The digitizer offers to do it
(`clear_output_dir_if_exists`) but empties everything in there, ours or not; the config
turns it off and the pipeline deletes exactly the three artifacts of each record it is about
to rewrite. That also stops a record which succeeded last run and failed this one from
having its stale CSV read as fresh.

In a backend, use `--fail-on-degraded` (exit code 2):

```bash
python -m ecg_pipeline --images in/ --out out/ --fail-on-degraded
```

`matching_cost` is reported but **not thresholded by default**. It is an unbounded
residual (mean grid distance × a scaling factor), not a normalized score, and `1.0` is a
hardcoded sentinel for "no layout matched" rather than a measurement. Calibrate on your
own data before enabling `--max-matching-cost`.

## Feeding app-EKG

`ecg_pipeline/contract.py` renders a result in the shape app-EKG's `EcgAnalysisService`
consumes, so whatever ends up serving HTTP is mapping fields rather than deciding what the
pipeline meant.

```python
from ecg_pipeline import contract, pipeline

results = pipeline.run(image_dir="in/", output_dir="out/")
analysis = contract.to_analysis(results[0], study_id="study-abc", completed_at=timestamp)
```

Three conversions it performs, each one a way to put a plausible lie on a screen if skipped:

- **Units.** The digitizer writes microvolts; `EcgSignal` is millivolts.
- **Gaps stay gaps -- dropouts don't.** A lead becomes a list of continuous segments
  stamped with the second each begins, never a padded array. On a 3×4 print a grid lead
  exists for 2.5 of the 10 seconds, and the app's signal model is built so that drawing a
  line across the other 7.5 takes deliberate effort. Within a printed stretch, though, the
  digitizer sometimes drops the trace for a handful of milliseconds under a label or a bold
  grid line; `contract.lead_segments` bridges an internal NaN run of at most
  `MAX_BRIDGED_GAP_SECONDS` (0.04 s -- one small grid square at 25 mm/s, shorter than any
  QRS complex) by linear interpolation and leaves every longer hole split, so a real
  never-recorded stretch is never quietly filled. What was bridged is reported separately,
  per lead, by `contract.signal_bridging_report`, so a caller such as api-EKG's diagnostics
  can keep the counts without the app ever seeing them (`EcgSignal`'s shape does not grow
  a field for this).
- **Right-sided leads get their names back.** The digitizer identifies leads by position, so
  a right-sided print's V4R/V5R/V6R land in the V4/V5/V6 slots; on that layout they are
  relabelled rather than shown as left-sided leads.

`contract.signal_from_csv(csv_path, lead_layout="", fs=CANONICAL_FS)` is the entry point for
the trace alone, applying the same three conversions without running interpretation. Call it
right after digitization -- to show the trace to a user while the interpretation stage is
still running, say -- instead of reaching into `to_signal` and the digitizer's CSV format
directly; `to_analysis` calls it internally for the `signal` field above.

Each observation also carries `aboveThreshold`. When a threshold applied to the result --
explicit or the default one described above under "Scores are rankings, not calibrated
probabilities" -- classes that cleared it come first as `true`. Among the rest, only the
classes that actually had a threshold to clear come back `false`; every other class is
`null`. A flat threshold covers every class, so with one of those the rest of the ranking
is all `false`, same as before. A per-class threshold set can be partial -- the provisional
fold-10 set in `configs/thresholds/provisional/` supplies 23 of 150 classes, see
`configs/thresholds/README.md` -- and for a class with no entry there, "no threshold was
applied" is a different claim from "checked and absent"; reporting it as `false` would say
127 classes were ruled out when they were never evaluated. With no threshold applied at all
it is `null` on every row: "no verdict computed", not "checked and absent". This is an
additional field on the existing observation shape, not a new one; app-EKG's
`observationFrom` parser picks named fields off the response rather than rejecting unknown
ones, so it reads the rest unaffected.

Two things it deliberately does not do:

- **`measurements` is always `null`.** Rate, PR, QRS, QT, QTc and axis need wave
  delineation, which this pipeline does not do, and `EcgMeasurements` has no partial form.
  Filling PR and QT with anything would be inventing measurements of a patient.
- **Labels pass through as the model produced them** (`SINUS RHYTHM`, not a phrase for a
  patient to read). Rewriting them is a clinical and product decision. `docs/etiquetas_es_borrador.csv`
  and `docs/etiquetas_es_borrador.md` are a first Spanish-language draft of that decision, for
  a cardiologist to correct line by line; the pipeline does not read the Spanish text, only
  the `categoria` column (see below).

Each observation also carries `category`, one of the 10 fixed slugs the CSV's `categoria`
column assigns every label to (`ritmo`, `conduccion`, `repolarizacion`, `isquemia_infarto`,
`marcapasos`, `eje`, `hipertrofia`, `tecnico`, `resumen`, `otro`), so the app can group
findings without embedding clinical judgement of its own. `contract.observation_category`
looks the label up in `ecg_pipeline/label_categories.py`, generated from the CSV by
`scripts/generate_label_categories.py` -- rerun that script after editing the CSV rather
than hand-patching the generated module. A label the model emits that the CSV does not
cover falls back to `otro` rather than raising: the model's label set can move ahead of the
draft, and a response should never fail over one unclassified finding.

A `degraded` result is emitted as `status: "failed"`, not as observations. The contract has
`ready` and `failed` and nothing in between, so a reading that must not be trusted goes
over as `failed` with a cause. "Digitized, but too poor to read" is `trace-incomplete`:
the layout matched and a CSV exists, but no lead carries signal or none was printed at
full length. It used to collapse into `unexpected`, which told the user the server had
failed when the photograph had.

## Tests

```bash
python -m unittest discover -s tests
```

Fast, and they need neither the model weights nor a digitizer checkout, the safety logic
runs on synthetic arrays and `digitize()` is exercised against a mocked subprocess. They do
need `numpy` and `scipy` (`pip install -r requirements.txt`); torch is never imported, so
the suite runs long before the ~740 MB of checkpoints are in place.

Run them with the interpreter that has those installed. A bare `python` without numpy does
**not** fail loudly, `unittest discover` reports the modules it could import and counts the
rest as a single error, so a green-looking run can be missing whole files. Check the test
count.

The suite covers the quality gates, the metadata parsing, checkout resolution, and the
argument construction that the digitizer's CLI is picky about. Two cases worth knowing:
`test_zscore_is_unit_invariant` is the regression guard for the µV/mV mismatch (see
above), and `test_input_and_output_are_passed_as_positional_overrides` pins the fact
that the digitizer takes overrides positionally, a `--overrides` flag makes it exit 2.

### Integration check

```bash
python scripts/integration_check.py                    # run + compare
python scripts/integration_check.py --update-reference  # regenerate the stored reference
```

Deliberately **not** part of `unittest discover -s tests`: it shells out to the real
digitizer over `tests/integration/images/normal.png` and needs the checkout and its model
weights, about two minutes of CPU. Run it by hand after touching `patches/`, `configs/`, or
updating the digitizer checkout -- the synthetic-array suite above cannot see a regression
that only shows up when the real segmentation network reads a real scan differently than
before. It compares the fresh canonical CSV against a stored reference
(`tests/integration/expected/`) on the layout name, the set of leads carrying signal, each
lead's coverage (within 0.02), and each lead's Pearson correlation over the overlapping
samples (at least 0.99) -- tolerances rather than exact equality, because a rerun of the
identical image is a floating-point pipeline and can differ in the last bit. Prints a table
and exits non-zero on any mismatch. See `tests/integration/README.md` for where the fixture
image comes from and when to regenerate the reference.

## Layout

```
ecg_pipeline/
  digitizer.py                    external-process wrapper (licensing boundary lives here)
  preprocess.py                   image preparation before digitization (upscaling)
  pipeline.py                     orchestration
  cli.py                          command line
  interpret/
    waveform.py                   pure signal primitives, shared
    representative_beat.py        phase-aligned median beat (the morphology pathway)
    interpret_ecg.py              ECGFounder pathways, quality gating, reporting
    net1d.py                      vendored from ECGFounder (MIT)
configs/                          digitizer configs and lead layouts
patches/                          fixes applied to the external checkout (CC BY-SA 4.0)
scripts/                          setup helpers
tests/                            stdlib unittest suite, no weights required
```

## Status

Not a medical device. No regulatory clearance. Research and development use only 
outputs are not clinical decisions.

Licensed proprietary; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
